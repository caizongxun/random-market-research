"""
app.py  —  Flask K-bar simulation viewer

啟動：
    python app.py
瀏覽器開啟：
    http://localhost:5000
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

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

_ohlcv_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 1800


# ─────────────────────────────────────────────────────────────────────────────
# Medoid 路徑選取（與 rolling_forward.py 邏輯完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def select_medoid_path(all_paths: np.ndarray) -> tuple[int, np.ndarray]:
    """從 all_paths (n, T) 中選出距離其他所有路徑 Euclidean 距離總和最小的那條。"""
    n = all_paths.shape[0]
    if n == 1:
        return 0, all_paths[0]
    if n > 200:
        idx_sample = np.random.choice(n, 200, replace=False)
        subset = all_paths[idx_sample]
        diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
        dist_sums = np.sqrt(np.sum(diff ** 2, axis=-1)).sum(axis=1)
        medoid_local = int(np.argmin(dist_sums))
        medoid_idx = int(idx_sample[medoid_local])
    else:
        diff = all_paths[:, np.newaxis, :] - all_paths[np.newaxis, :, :]
        dist_sums = np.sqrt(np.sum(diff ** 2, axis=-1)).sum(axis=1)
        medoid_idx = int(np.argmin(dist_sums))
    return medoid_idx, all_paths[medoid_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 資料取得 / 參數估計
# ─────────────────────────────────────────────────────────────────────────────
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


def get_market_params(df: pd.DataFrame):
    return MarketParameterEstimator().fit(df)


def get_theta(market_params) -> CalibratedTheta:
    p = market_params
    return CalibratedTheta(
        vol=float(p.realized_vol),
        drift=float(p.ewma_drift),
        mr_coeff=float(np.clip(p.mean_reversion_strength * 0.1 + 0.01, 0.01, 0.15)),
        node_coeff=float(np.clip(p.smart_money_ratio * 0.04, 0.005, 0.05)),
        hurst_proxy=float(p.hurst_proxy),
        momentum_strength=float(np.clip(p.trend_strength * 0.5 + 0.1, 0.05, 0.8)),
        momentum_decay=float(np.clip(p.hurst_proxy, 0.5, 0.95)),
        breakout_boost=float(np.clip(p.vol_trend, 0.2, 1.5)),
    )


def auto_calibrate_drift(df_window: pd.DataFrame) -> dict:
    log_rets = np.diff(np.log(df_window.tail(500)["Close"].values))
    vol   = float(np.std(log_rets))  if len(log_rets) >= 20 else 0.015
    drift = float(np.mean(log_rets)) if len(log_rets) >= 20 else 0.0
    return {
        "drift_scale":   round(float(np.clip(abs(drift) / max(vol, 1e-6) * 0.8 + 0.5, 0.3, 3.2)), 3),
        "drift_decay":   0.05,
        "vol_multiplier":round(float(np.clip(vol / 0.015, 0.5, 3.0)), 3),
        "intra_bar":     round(float(np.clip(vol * 1.5, 0.005, 0.06)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 模擬核心：一次跑 n_sim_paths 條，選 medoid，與 pipeline 一致
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(
    df_hist: pd.DataFrame,
    forecast_bars: int,
    auto_params: dict,
    n_sim_paths: int = 500,
    seed: int | None = None,
) -> list[dict]:
    """
    跑 n_sim_paths 條路徑，用 medoid 選取最具代表性的單條路徑。
    OHLCV 建構邏輯與 rolling_forward.py 一致：
      - open / high / low  來自 SimulationResult 的 representative_path OHLCV
      - close              替換為 medoid 收盤序列
    回傳單條路徑的 list[dict{open,high,low,close}]。
    """
    market_params     = get_market_params(df_hist)
    theta             = get_theta(market_params)
    calibrated_params = build_params_from_theta(theta, market_params)

    sim = USStockFutureSimulator(
        params=calibrated_params,
        forecast_steps=forecast_bars,
        n_paths=n_sim_paths,
        seed=seed,
        vol_scale=float(auto_params["vol_multiplier"]),
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_scale=float(auto_params["drift_scale"]),
        drift_decay_rate=float(auto_params["drift_decay"]),
    )
    result = sim.simulate()

    # Medoid：在所有 close 路徑中選最中心那條
    all_closes = result.future_paths[:, :forecast_bars]   # (n_sim_paths, T)
    _, medoid_close = select_medoid_path(all_closes)       # (T,)

    # OHLCV 骨幹來自 representative_path（RMSE-to-median 最小）
    T = forecast_bars
    o = result.ohlcv_open[:T]
    h = result.ohlcv_high[:T]
    l = result.ohlcv_low[:T]
    c = medoid_close[:T]  # close 替換為 medoid

    # 保持 high >= max(open, close)、low <= min(open, close)
    h = np.maximum(h, np.maximum(o, c))
    l = np.minimum(l, np.minimum(o, c))

    return [
        {
            "open":  round(float(o[t]), 4),
            "high":  round(float(h[t]), 4),
            "low":   round(float(l[t]), 4),
            "close": round(float(c[t]), 4),
        }
        for t in range(T)
    ]


def df_to_ohlc_list(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        d  = row["Date"]
        ts = int(d.timestamp()) if hasattr(d, "timestamp") else int(pd.Timestamp(d).timestamp())
        records.append({
            "time":  ts,
            "open":  round(float(row["Open"]),  4),
            "high":  round(float(row["High"]),  4),
            "low":   round(float(row["Low"]),   4),
            "close": round(float(row["Close"]), 4),
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body         = request.get_json(force=True)
        symbol       = body.get("symbol",       "AAPL").upper().strip()
        end_date_str = body.get("end_date",      "2025-01-01")
        forecast_bars = int(body.get("forecast_bars", 30))
        hist_bars    = int(body.get("hist_bars",  120))
        n_sim_paths  = int(body.get("n_sim_paths", 500))  # 內部模擬路徑數，預設 500
        seed         = body.get("seed", None)
        if seed is not None:
            seed = int(seed)

        end_dt      = datetime.strptime(end_date_str, "%Y-%m-%d")
        fetch_start = (end_dt - timedelta(days=900)).strftime("%Y-%m-%d")
        fetch_end   = (end_dt + timedelta(days=forecast_bars * 2 + 30)).strftime("%Y-%m-%d")

        df_all = get_ohlcv(symbol, fetch_start, fetch_end)
        if df_all.empty or len(df_all) < 60:
            return jsonify({"error": f"資料不足：{symbol} {fetch_start}~{fetch_end}"}), 400

        df_hist   = df_all[df_all["Date"] <= end_dt].copy()
        df_future = df_all[df_all["Date"] >  end_dt].head(forecast_bars).copy()

        if len(df_hist) < 100:
            return jsonify({"error": "end_date 之前的歷史資料不足 100 根"}), 400

        auto_params = auto_calibrate_drift(df_hist)

        # 單次模擬，回傳唯一一條 medoid 路徑
        medoid_path = run_simulation(
            df_hist=df_hist,
            forecast_bars=forecast_bars,
            auto_params=auto_params,
            n_sim_paths=n_sim_paths,
            seed=seed,
        )

        # 時間軸
        last_date = df_hist["Date"].iloc[-1]
        if not df_future.empty:
            future_dates = df_future["Date"].tolist()
        else:
            future_dates = []
            d = last_date
            while len(future_dates) < forecast_bars:
                d += timedelta(days=1)
                if d.weekday() < 5:
                    future_dates.append(d)

        while len(future_dates) < forecast_bars:
            d = future_dates[-1] + timedelta(days=1)
            while d.weekday() >= 5:
                d += timedelta(days=1)
            future_dates.append(d)

        future_ts = [int(pd.Timestamp(d).timestamp()) for d in future_dates[:forecast_bars]]
        sim_candles = [
            {"time": future_ts[i], **bar}
            for i, bar in enumerate(medoid_path)
            if i < len(future_ts)
        ]

        # 展示用參數
        mp = get_market_params(df_hist)
        th = get_theta(mp)

        return jsonify({
            "symbol":         symbol,
            "end_date":       end_date_str,
            "hist_candles":   df_to_ohlc_list(df_hist.tail(hist_bars)),
            "actual_candles": df_to_ohlc_list(df_future),
            "sim_candles":    sim_candles,   # 唯一一條 medoid 路徑
            "n_sim_paths":    n_sim_paths,
            "auto_params":    auto_params,
            "theta": {
                "vol":               round(th.vol,               5),
                "mr_coeff":          round(th.mr_coeff,          5),
                "momentum_strength": round(th.momentum_strength, 4),
            },
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# 前端 HTML
# ─────────────────────────────────────────────────────────────────────────────
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
    --bg: #0f0f11; --surface: #18181c; --border: #2a2a30;
    --text: #d4d4d8; --muted: #71717a; --accent: #4f98a3;
    --green: #26a69a; --red: #ef5350; --actual: #f0c040;
    --radius: 8px; --font: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }
  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; }
  body { display: flex; flex-direction: column; height: 100vh; }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 12px 20px; background: var(--surface);
    border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  header h1 { font-size: 15px; font-weight: 600; color: var(--accent); letter-spacing: 0.03em; margin-right: 8px; }
  .ctrl-group { display: flex; align-items: center; gap: 8px; }
  .ctrl-group label { font-size: 12px; color: var(--muted); white-space: nowrap; }
  input {
    background: #0f0f11; border: 1px solid var(--border);
    color: var(--text); border-radius: 6px;
    padding: 5px 10px; font-size: 13px; outline: none;
    transition: border-color 0.15s;
  }
  input:focus { border-color: var(--accent); }
  input[type="text"]   { width: 80px; }
  input[type="date"]   { width: 145px; }
  input[type="number"] { width: 60px; }
  button {
    padding: 6px 16px; border-radius: 6px; border: none;
    background: var(--accent); color: #fff; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: background 0.15s; white-space: nowrap;
  }
  button:hover  { background: #3d7d87; }
  button:disabled { background: #2a4a4f; color: #71717a; cursor: not-allowed; }
  #resim-btn { background: #2a3a2a; color: #6daa45; border: 1px solid #3a5a3a; }
  #resim-btn:hover { background: #3a5a3a; }
  #info-bar {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 6px 20px; background: #12121a;
    border-bottom: 1px solid var(--border);
    font-size: 12px; color: var(--muted); flex-shrink: 0; min-height: 28px;
  }
  #info-bar span { white-space: nowrap; }
  .tag { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
  .tag-teal   { background: #1a3035; color: var(--accent); }
  .tag-yellow { background: #3a3010; color: var(--actual); }
  .tag-muted  { background: #1e1e24; color: var(--muted); }
  #chart-wrap { flex: 1; position: relative; overflow: hidden; }
  #chart { width: 100%; height: 100%; }
  #legend {
    position: absolute; top: 12px; left: 16px;
    background: rgba(15,15,17,0.88); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 8px 12px;
    font-size: 12px; line-height: 1.9; pointer-events: none; z-index: 10;
  }
  .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
  #loading {
    display: none; position: absolute; inset: 0;
    background: rgba(15,15,17,0.72);
    align-items: center; justify-content: center;
    font-size: 14px; color: var(--accent);
    z-index: 100; flex-direction: column; gap: 12px;
  }
  #loading.show { display: flex; }
  .spinner {
    width: 32px; height: 32px; border: 3px solid var(--border);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #toast {
    display: none; position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: #3a1020; border: 1px solid #ef5350; color: #ef9090;
    padding: 10px 20px; border-radius: 8px; font-size: 13px;
    z-index: 200; max-width: 90vw; text-align: center;
  }
  #toast.show { display: block; }
  @media (max-width: 768px) {
    header { gap: 8px; padding: 8px 12px; }
    header h1 { display: none; }
    input[type="text"] { width: 70px; }
    input[type="date"] { width: 130px; }
  }
</style>
</head>
<body>
<header>
  <h1>Market Sim</h1>
  <div class="ctrl-group"><label>股票</label><input type="text" id="symbol" value="AAPL" /></div>
  <div class="ctrl-group"><label>截止日</label><input type="date" id="end-date" value="2024-06-01" /></div>
  <div class="ctrl-group"><label>預測根數</label><input type="number" id="forecast-bars" value="30" min="5" max="120" /></div>
  <div class="ctrl-group"><label>歷史顯示根數</label><input type="number" id="hist-bars" value="60" min="20" max="250" /></div>
  <div class="ctrl-group"><label>模擬路徑數</label><input type="number" id="n-sim-paths" value="500" min="50" max="2000" /></div>
  <button id="run-btn" onclick="runSim()">執行</button>
  <button id="resim-btn" onclick="runSim()" disabled>重新模擬</button>
</header>
<div id="info-bar"><span id="info-text">輸入參數後點擊「執行」</span></div>
<div id="chart-wrap">
  <div id="chart"></div>
  <div id="legend">
    <div><span class="legend-dot" style="background:#6b7280"></span>歷史 K 棒</div>
    <div><span class="legend-dot" style="background:#4f98a3"></span>模擬（Medoid 路徑）</div>
    <div><span class="legend-dot" style="background:#f0c040"></span>實際走勢</div>
  </div>
  <div id="loading"><div class="spinner"></div><span id="loading-text">模擬中...</span></div>
</div>
<div id="toast"></div>
<script>
let chart = null, histSeries = null, simSeries = null, actualSeries = null;

function initChart() {
  if (chart) { chart.remove(); chart = null; }
  const el = document.getElementById('chart');
  chart = LightweightCharts.createChart(el, {
    width: el.clientWidth, height: el.clientHeight,
    layout: { background: { color: '#0f0f11' }, textColor: '#71717a' },
    grid: { vertLines: { color: '#1a1a20' }, horzLines: { color: '#1a1a20' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#2a2a30' },
    timeScale: { borderColor: '#2a2a30', timeVisible: true, secondsVisible: false },
  });
  window.addEventListener('resize', () => {
    if (chart) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
  });
}

function clearSeries() {
  if (!chart) return;
  if (histSeries)   { chart.removeSeries(histSeries);   histSeries   = null; }
  if (simSeries)    { chart.removeSeries(simSeries);    simSeries    = null; }
  if (actualSeries) { chart.removeSeries(actualSeries); actualSeries = null; }
}

function renderData(data) {
  clearSeries();

  // 歷史 K 棒（暗灰）
  histSeries = chart.addCandlestickSeries({
    upColor: '#4b5563', downColor: '#374151',
    borderUpColor: '#6b7280', borderDownColor: '#4b5563',
    wickUpColor:   '#6b7280', wickDownColor:   '#4b5563',
  });
  histSeries.setData(data.hist_candles);

  // Medoid 模擬路徑（主色）
  simSeries = chart.addCandlestickSeries({
    upColor: '#4f98a3', downColor: '#2a5f68',
    borderUpColor: '#4f98a3', borderDownColor: '#2a5f68',
    wickUpColor:   '#4f98a3', wickDownColor:   '#2a5f68',
  });
  simSeries.setData(data.sim_candles);

  // 實際走勢（黃）
  if (data.actual_candles && data.actual_candles.length > 0) {
    actualSeries = chart.addCandlestickSeries({
      upColor: '#f0c040', downColor: '#c07820',
      borderUpColor: '#f0c040', borderDownColor: '#c07820',
      wickUpColor:   '#f0c040', wickDownColor:   '#c07820',
    });
    actualSeries.setData(data.actual_candles);
  }

  chart.timeScale().fitContent();
}

async function runSim() {
  const symbol       = document.getElementById('symbol').value.trim().toUpperCase();
  const endDate      = document.getElementById('end-date').value;
  const forecastBars = parseInt(document.getElementById('forecast-bars').value);
  const histBars     = parseInt(document.getElementById('hist-bars').value);
  const nSimPaths    = parseInt(document.getElementById('n-sim-paths').value) || 500;

  if (!symbol || !endDate) { showToast('請填寫股票代碼和截止日'); return; }

  setLoading(true, `模擬中（${nSimPaths} 條路徑）...`);
  document.getElementById('run-btn').disabled   = true;
  document.getElementById('resim-btn').disabled = true;

  try {
    const resp = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol,
        end_date:      endDate,
        forecast_bars: forecastBars,
        hist_bars:     histBars,
        n_sim_paths:   nSimPaths,
        seed:          Math.floor(Math.random() * 99999),
      }),
    });
    const data = await resp.json();
    if (data.error) { showToast(data.error); if (data.trace) console.error(data.trace); return; }

    if (!chart) initChart();
    renderData(data);

    const ap = data.auto_params, th = data.theta;
    document.getElementById('info-text').innerHTML = [
      `<span class="tag tag-muted">${data.symbol}  截至 ${data.end_date}</span>`,
      `<span>Medoid / <b>${data.n_sim_paths}</b> 條</span>`,
      `<span>drift_scale <b>${ap.drift_scale}</b></span>`,
      `<span>vol_mult <b>${ap.vol_multiplier}</b></span>`,
      `<span>theta.vol <b>${th.vol}</b></span>`,
      `<span>theta.mr <b>${th.mr_coeff}</b></span>`,
      `<span>theta.mom <b>${th.momentum_strength}</b></span>`,
      `<span class="tag tag-yellow">黃色=實際</span>`,
      `<span class="tag tag-teal">藍綠=Medoid 模擬</span>`,
    ].join('');

    document.getElementById('resim-btn').disabled = false;
  } catch (e) {
    showToast('請求失敗：' + e.message);
  } finally {
    setLoading(false);
    document.getElementById('run-btn').disabled = false;
  }
}

function setLoading(on, msg) {
  document.getElementById('loading').classList.toggle('show', on);
  if (msg) document.getElementById('loading-text').textContent = msg;
}

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 4500);
}

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
