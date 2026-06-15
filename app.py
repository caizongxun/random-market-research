"""
app.py  —  Flask K-bar simulation viewer

啟動：
    python app.py
瀏覽器開啟：
    http://localhost:5000
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from calibrated_simulator import CalibratedTheta, build_params_from_theta
from market_estimator import MarketParameterEstimator
from us_equity_simulator import USStockFutureSimulator

app = Flask(__name__)

# ─────────────────────────────────────────
# 簡易記憶體快取（避免重複下載）
# ─────────────────────────────────────────
_ohlcv_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 1800  # 30 分鐘


def get_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    key = f"{symbol}_{start}_{end}"
    now = time.time()
    if key in _ohlcv_cache:
        ts, df = _ohlcv_cache[key]
        if now - ts < CACHE_TTL:
            return df
    raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    _ohlcv_cache[key] = (now, df)
    return df


def calibrate_theta(df: pd.DataFrame) -> CalibratedTheta:
    """
    使用 MarketParameterEstimator.fit() 校準 theta。
    MarketParams 欄位對應：
      realized_vol          -> vol
      mean_reversion_strength -> mr_coeff
      smart_money_ratio     -> node_coeff
      trend_strength        -> momentum_strength
      hurst_proxy           -> momentum_decay (proxy)
      vol_trend             -> breakout_boost (proxy)
    """
    estimator = MarketParameterEstimator()
    params = estimator.fit(df)
    return CalibratedTheta(
        vol=float(params.realized_vol),
        mr_coeff=float(params.mean_reversion_strength) * 0.1 + 0.01,
        node_coeff=float(params.smart_money_ratio),
        momentum_strength=float(params.trend_strength) * 0.5 + 0.1,
        momentum_decay=float(np.clip(params.hurst_proxy, 0.5, 0.95)),
        breakout_boost=float(np.clip(params.vol_trend, 1.0, 3.0)),
    )


def auto_calibrate(df_window: pd.DataFrame) -> dict:
    """
    簡易 auto_calibrate：用最近 500 根估算 drift_scale / drift_decay / vol_multiplier
    """
    window = df_window.tail(500)
    log_rets = np.diff(np.log(window["Close"].values))
    vol = float(np.std(log_rets)) if len(log_rets) >= 20 else 0.015
    drift = float(np.mean(log_rets)) if len(log_rets) >= 20 else 0.0

    drift_scale = float(np.clip(abs(drift) / max(vol, 1e-6) * 0.8 + 0.5, 0.3, 3.2))
    drift_decay = 0.05
    vol_multiplier = float(np.clip(vol / 0.015, 0.5, 3.0))
    intra_bar = float(np.clip(vol * 1.5, 0.005, 0.06))

    return {
        "drift_scale": round(drift_scale, 3),
        "drift_decay": round(drift_decay, 3),
        "vol_multiplier": round(vol_multiplier, 3),
        "intra_bar": round(intra_bar, 4),
    }


def run_simulation(
    df_hist: pd.DataFrame,
    theta: CalibratedTheta,
    auto_params: dict,
    forecast_bars: int,
    n_paths: int,
    seed: int | None = None,
) -> list[list[dict]]:
    """
    回傳 n_paths 條模擬路徑，每條是一個 list of OHLC dict。
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_paths).tolist()

    params = build_params_from_theta(theta)
    params["drift_scale"] = auto_params["drift_scale"]
    params["drift_decay"] = auto_params["drift_decay"]
    params["vol_multiplier"] = auto_params["vol_multiplier"]
    params["intra_bar"] = auto_params["intra_bar"]

    close_arr = df_hist["Close"].values
    high_arr = df_hist["High"].values
    low_arr = df_hist["Low"].values
    start_price = float(close_arr[-1])

    paths = []
    for s in seeds:
        sim = USStockFutureSimulator(
            params=params,
            n_bars=forecast_bars,
            start_price=start_price,
            history_close=close_arr,
            history_high=high_arr,
            history_low=low_arr,
            seed=int(s),
        )
        bars = sim.simulate()
        # 相容 list/dict 與 dataclass/namedtuple 兩種回傳格式
        def _get(b, key):
            if isinstance(b, dict):
                return b[key]
            return getattr(b, key)
        path = [
            {
                "open": round(float(_get(b, "open")), 4),
                "high": round(float(_get(b, "high")), 4),
                "low": round(float(_get(b, "low")), 4),
                "close": round(float(_get(b, "close")), 4),
            }
            for b in bars
        ]
        paths.append(path)
    return paths


def df_to_ohlc_list(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        d = row["Date"]
        ts = int(d.timestamp()) if hasattr(d, "timestamp") else int(pd.Timestamp(d).timestamp())
        records.append({
            "time": ts,
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
        })
    return records


# ─────────────────────────────────────────
# API
# ─────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body = request.get_json(force=True)
        symbol = body.get("symbol", "AAPL").upper().strip()
        end_date_str = body.get("end_date", "2025-01-01")
        forecast_bars = int(body.get("forecast_bars", 30))
        n_paths = min(int(body.get("n_paths", 5)), 20)
        hist_bars = int(body.get("hist_bars", 120))
        seed = body.get("seed", None)
        if seed is not None:
            seed = int(seed)

        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        fetch_start = (end_dt - timedelta(days=900)).strftime("%Y-%m-%d")
        fetch_end = (end_dt + timedelta(days=forecast_bars * 2 + 30)).strftime("%Y-%m-%d")

        df_all = get_ohlcv(symbol, fetch_start, fetch_end)
        if df_all.empty or len(df_all) < 60:
            return jsonify({"error": f"資料不足：{symbol} {fetch_start}~{fetch_end}"}), 400

        df_hist = df_all[df_all["Date"] <= end_dt].copy()
        df_future = df_all[df_all["Date"] > end_dt].head(forecast_bars).copy()

        if len(df_hist) < 60:
            return jsonify({"error": "end_date 之前的歷史資料不足 60 根"}), 400

        theta = calibrate_theta(df_hist)
        auto_params = auto_calibrate(df_hist)

        df_display_hist = df_hist.tail(hist_bars).copy()

        paths = run_simulation(
            df_hist=df_hist,
            theta=theta,
            auto_params=auto_params,
            forecast_bars=forecast_bars,
            n_paths=n_paths,
            seed=seed,
        )

        last_date = df_hist["Date"].iloc[-1]
        if not df_future.empty:
            future_dates = df_future["Date"].tolist()
        else:
            future_dates = []
            d = last_date
            while len(future_dates) < forecast_bars:
                d = d + timedelta(days=1)
                if d.weekday() < 5:
                    future_dates.append(d)

        while len(future_dates) < forecast_bars:
            d = future_dates[-1] + timedelta(days=1)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            future_dates.append(d)

        future_dates = future_dates[:forecast_bars]
        future_ts = [int(pd.Timestamp(d).timestamp()) for d in future_dates]

        sim_paths_ts = []
        for path in paths:
            path_ts = []
            for i, bar in enumerate(path):
                if i < len(future_ts):
                    path_ts.append({"time": future_ts[i], **bar})
            sim_paths_ts.append(path_ts)

        return jsonify({
            "symbol": symbol,
            "end_date": end_date_str,
            "hist_candles": df_to_ohlc_list(df_display_hist),
            "actual_candles": df_to_ohlc_list(df_future),
            "sim_paths": sim_paths_ts,
            "auto_params": auto_params,
            "theta": {
                "vol": round(theta.vol, 5),
                "mr_coeff": round(theta.mr_coeff, 5),
                "momentum_strength": round(theta.momentum_strength, 4),
            },
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────
# 前端 HTML
# ─────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Simulation Viewer</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f0f11;
    --surface: #18181c;
    --border: #2a2a30;
    --text: #d4d4d8;
    --muted: #71717a;
    --accent: #4f98a3;
    --green: #26a69a;
    --red: #ef5350;
    --actual: #f0c040;
    --radius: 8px;
    --font: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; }
  body { display: flex; flex-direction: column; height: 100vh; }

  /* ── Header ── */
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 12px 20px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  header h1 { font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: 0.03em; margin-right: 8px; }
  .ctrl-group { display: flex; align-items: center; gap: 8px; }
  .ctrl-group label { font-size: 12px; color: var(--muted); white-space: nowrap; }
  input, select {
    background: #0f0f11; border: 1px solid var(--border);
    color: var(--text); border-radius: 6px;
    padding: 5px 10px; font-size: 13px;
    outline: none; transition: border-color 0.15s;
  }
  input:focus, select:focus { border-color: var(--accent); }
  input[type="text"] { width: 80px; }
  input[type="date"] { width: 145px; }
  input[type="number"] { width: 60px; }
  button {
    padding: 6px 16px; border-radius: 6px; border: none;
    background: var(--accent); color: #fff; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: background 0.15s;
    white-space: nowrap;
  }
  button:hover { background: #3d7d87; }
  button:disabled { background: #2a4a4f; color: #71717a; cursor: not-allowed; }
  #resim-btn {
    background: #2a3a2a; color: #6daa45; border: 1px solid #3a5a3a;
  }
  #resim-btn:hover { background: #3a5a3a; }

  /* ── Info bar ── */
  #info-bar {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 6px 20px;
    background: #12121a;
    border-bottom: 1px solid var(--border);
    font-size: 12px; color: var(--muted);
    flex-shrink: 0;
    min-height: 28px;
  }
  #info-bar span { white-space: nowrap; }
  .tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  .tag-green { background: #1a3a30; color: var(--green); }
  .tag-yellow { background: #3a3010; color: var(--actual); }
  .tag-muted { background: #1e1e24; color: var(--muted); }

  /* ── Chart ── */
  #chart-wrap {
    flex: 1; position: relative; overflow: hidden;
  }
  #chart { width: 100%; height: 100%; }

  /* ── Legend ── */
  #legend {
    position: absolute; top: 12px; left: 16px;
    background: rgba(15,15,17,0.85);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 12px;
    font-size: 12px; line-height: 1.8;
    pointer-events: none;
    z-index: 10;
  }
  .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }

  /* ── Loading overlay ── */
  #loading {
    display: none;
    position: absolute; inset: 0;
    background: rgba(15,15,17,0.7);
    align-items: center; justify-content: center;
    font-size: 14px; color: var(--accent);
    z-index: 100;
    flex-direction: column; gap: 12px;
  }
  #loading.show { display: flex; }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Error toast ── */
  #toast {
    display: none;
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #3a1020; border: 1px solid #ef5350; color: #ef9090;
    padding: 10px 20px; border-radius: 8px; font-size: 13px;
    z-index: 200; max-width: 90vw; text-align: center;
  }
  #toast.show { display: block; }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    header { gap: 8px; padding: 8px 12px; }
    header h1 { display: none; }
    .ctrl-group { gap: 6px; }
    input[type="text"] { width: 70px; }
    input[type="date"] { width: 130px; }
  }
</style>
</head>
<body>

<header>
  <h1>Market Sim</h1>
  <div class="ctrl-group">
    <label>股票</label>
    <input type="text" id="symbol" value="AAPL" placeholder="AAPL" />
  </div>
  <div class="ctrl-group">
    <label>截止日</label>
    <input type="date" id="end-date" value="2024-06-01" />
  </div>
  <div class="ctrl-group">
    <label>預測根數</label>
    <input type="number" id="forecast-bars" value="30" min="5" max="120" />
  </div>
  <div class="ctrl-group">
    <label>模擬路徑數</label>
    <input type="number" id="n-paths" value="5" min="1" max="20" />
  </div>
  <div class="ctrl-group">
    <label>歷史顯示根數</label>
    <input type="number" id="hist-bars" value="60" min="20" max="250" />
  </div>
  <button id="run-btn" onclick="runSim()">執行</button>
  <button id="resim-btn" onclick="resim()" disabled>重新抽樣</button>
</header>

<div id="info-bar">
  <span id="info-text">輸入參數後點擊「執行」</span>
</div>

<div id="chart-wrap">
  <div id="chart"></div>
  <div id="legend">
    <div><span class="legend-dot" style="background:#6b7280"></span>歷史 K 棒</div>
    <div><span class="legend-dot" style="background:rgba(79,152,163,0.6)"></span>模擬路徑（隨機抽樣）</div>
    <div><span class="legend-dot" style="background:var(--actual)"></span>實際走勢</div>
  </div>
  <div id="loading"><div class="spinner"></div><span>模擬中...</span></div>
</div>

<div id="toast"></div>

<script>
let chart = null;
let histSeries = null;
let simSeriesList = [];
let actualSeries = null;
let lastData = null;
let lastSeed = null;

// ── 調色盤 ──
const SIM_COLORS = [
  'rgba(79,152,163,0.75)',
  'rgba(109,170,69,0.70)',
  'rgba(240,192,64,0.65)',
  'rgba(210,99,167,0.65)',
  'rgba(85,145,199,0.70)',
  'rgba(187,101,59,0.70)',
  'rgba(132,110,220,0.65)',
  'rgba(79,200,140,0.65)',
  'rgba(220,150,79,0.65)',
  'rgba(160,79,163,0.65)',
];

function initChart() {
  if (chart) { chart.remove(); chart = null; }
  const el = document.getElementById('chart');
  chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: {
      background: { color: '#0f0f11' },
      textColor: '#71717a',
    },
    grid: {
      vertLines: { color: '#1a1a20' },
      horzLines: { color: '#1a1a20' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2a30' },
    timeScale: { borderColor: '#2a2a30', timeVisible: true, secondsVisible: false },
  });
  window.addEventListener('resize', () => {
    if (chart) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
  });
}

function clearSeries() {
  if (chart) {
    simSeriesList.forEach(s => chart.removeSeries(s));
    if (histSeries) chart.removeSeries(histSeries);
    if (actualSeries) chart.removeSeries(actualSeries);
  }
  simSeriesList = [];
  histSeries = null;
  actualSeries = null;
}

function renderData(data, chosenPaths) {
  clearSeries();
  const allPaths = data.sim_paths;
  let paths = chosenPaths;
  if (!paths) {
    const n = Math.min(parseInt(document.getElementById('n-paths').value) || 5, allPaths.length);
    const idx = shuffleIndices(allPaths.length).slice(0, n);
    paths = idx.map(i => allPaths[i]);
  }

  // ── 歷史 K 棒（灰色）──
  histSeries = chart.addCandlestickSeries({
    upColor: '#4b5563',
    downColor: '#374151',
    borderUpColor: '#6b7280',
    borderDownColor: '#4b5563',
    wickUpColor: '#6b7280',
    wickDownColor: '#4b5563',
  });
  histSeries.setData(data.hist_candles);

  // ── 模擬路徑 ──
  paths.forEach((path, i) => {
    const color = SIM_COLORS[i % SIM_COLORS.length];
    const series = chart.addCandlestickSeries({
      upColor: color,
      downColor: color,
      borderUpColor: color,
      borderDownColor: color,
      wickUpColor: color,
      wickDownColor: color,
    });
    series.setData(path);
    simSeriesList.push(series);
  });

  // ── 實際走勢（黃色）──
  if (data.actual_candles && data.actual_candles.length > 0) {
    actualSeries = chart.addCandlestickSeries({
      upColor: '#f0c040',
      downColor: '#c07820',
      borderUpColor: '#f0c040',
      borderDownColor: '#c07820',
      wickUpColor: '#f0c040',
      wickDownColor: '#c07820',
    });
    actualSeries.setData(data.actual_candles);
  }

  chart.timeScale().fitContent();
  return paths;
}

function shuffleIndices(n) {
  const arr = Array.from({length: n}, (_, i) => i);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function runSim() {
  const symbol = document.getElementById('symbol').value.trim().toUpperCase();
  const endDate = document.getElementById('end-date').value;
  const forecastBars = parseInt(document.getElementById('forecast-bars').value);
  const nPaths = parseInt(document.getElementById('n-paths').value);
  const histBars = parseInt(document.getElementById('hist-bars').value);

  if (!symbol || !endDate) { showToast('請填寫股票代碼和截止日期'); return; }

  setLoading(true);
  document.getElementById('run-btn').disabled = true;
  document.getElementById('resim-btn').disabled = true;

  try {
    const fetchPaths = Math.max(nPaths * 3, 15);
    lastSeed = Math.floor(Math.random() * 99999);
    const resp = await fetch('/api/simulate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        symbol, end_date: endDate,
        forecast_bars: forecastBars,
        n_paths: fetchPaths,
        hist_bars: histBars,
        seed: lastSeed,
      }),
    });
    const data = await resp.json();
    if (data.error) {
      showToast(data.error);
      if (data.trace) console.error(data.trace);
      return;
    }

    lastData = data;
    if (!chart) initChart();
    renderData(data, null);

    const ap = data.auto_params;
    const th = data.theta;
    document.getElementById('info-text').innerHTML = [
      `<span class="tag tag-muted">${data.symbol}  截至 ${data.end_date}</span>`,
      `<span>drift_scale <b>${ap.drift_scale}</b></span>`,
      `<span>drift_decay <b>${ap.drift_decay}</b></span>`,
      `<span>vol_mult <b>${ap.vol_multiplier}</b></span>`,
      `<span>theta.vol <b>${th.vol}</b></span>`,
      `<span>theta.mr <b>${th.mr_coeff}</b></span>`,
      `<span>theta.mom <b>${th.momentum_strength}</b></span>`,
      `<span class="tag tag-yellow">黃色 = 實際走勢</span>`,
      `<span class="tag tag-green">彩色 = 模擬路徑（${data.sim_paths.length} 條中隨機取 ${nPaths}）</span>`,
    ].join('');

    document.getElementById('resim-btn').disabled = false;
  } catch(e) {
    showToast('請求失敗：' + e.message);
  } finally {
    setLoading(false);
    document.getElementById('run-btn').disabled = false;
  }
}

function resim() {
  if (!lastData) return;
  renderData(lastData, null);
}

function setLoading(on) {
  document.getElementById('loading').classList.toggle('show', on);
}

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 4000);
}

// ── 初始化 ──
initChart();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("\n  Market Simulation Viewer")
    print("  http://localhost:5000\n")
    app.run(debug=True, port=5000)
