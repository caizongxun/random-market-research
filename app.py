"""
app.py  —  Flask K-bar simulation viewer

啟動：
    python app.py
瀏覽器開啟：
    http://localhost:5000

模擬優先順序：
  1. 若 cache/{SYM}_theta.json 存在 → 直接用 pipeline 校準好的 theta
     若同時有 results/fear/{SYM}_fear_profile.json → 套用 fear profile 調整
     若同時有 models/param_model_{SYM}.joblib → surrogate 取代 grid search
  2. 否則 fallback：MarketParameterEstimator 即時估算（原有邏輯不動）

  /api/simulate 現在直接呼叫 scripts/rolling_forward.rolling_forward()，
  與 run_pipeline.py 完全共用同一套「分段推進＋真實資料校準」機制。
"""

from __future__ import annotations

import json
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

ROOT      = Path(__file__).parent
CACHE_DIR = ROOT / "cache"

# ── 注入 src/ 和 scripts/ ────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backbone_fitter import BackboneFitter
from calibrated_simulator import CalibratedTheta, build_params_from_theta
from market_estimator import MarketParameterEstimator
from us_equity_simulator import USStockFutureSimulator
from rolling_forward import rolling_forward, _load_vix_series

app = Flask(__name__)

_ohlcv_cache: dict[str, tuple[float, pd.DataFrame]] = {}
CACHE_TTL = 1800


# ─────────────────────────────────────────────────────────────────────────────
# Medoid 路徑選取（保留，run_simulation fallback 仍用）
# ─────────────────────────────────────────────────────────────────────────────
def select_medoid_path(all_paths: np.ndarray) -> tuple[int, np.ndarray]:
    n = all_paths.shape[0]
    if n == 1:
        return 0, all_paths[0]
    if n > 200:
        idx_s = np.random.choice(n, 200, replace=False)
        sub   = all_paths[idx_s]
        d     = np.sqrt(np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)).sum(axis=1)
        return int(idx_s[np.argmin(d)]), all_paths[int(idx_s[np.argmin(d)])]
    d   = np.sqrt(np.sum((all_paths[:, None, :] - all_paths[None, :, :]) ** 2, axis=-1)).sum(axis=1)
    idx = int(np.argmin(d))
    return idx, all_paths[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 資料取得
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


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline cache 讀取
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline_theta(symbol: str) -> CalibratedTheta | None:
    path = CACHE_DIR / f"{symbol}_theta.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        theta = CalibratedTheta.from_dict(data)
        print(f"[app] pipeline theta 載入 ← {path}")
        return theta
    except Exception as e:
        print(f"[app] theta 讀取失敗（{e}），fallback 到即時估算")
        return None


def load_pipeline_agent_profile(symbol: str) -> dict | None:
    """
    讀取 Step 3 產出的 fear_profile（results/fear/{SYM}_fear_profile.json），
    並將其欄位對應至 rolling_forward.apply_agent_profile 所需的 schema：
      retail_momentum_strength  ← 1.0 - fear_drift_decay_rate（恐慌越強動能越弱）
      retail_panic_sensitivity  ← fear_vol_mult 正規化到 [0.1, 0.9]
      inst_mr_strength          ← 固定 0.5（fear_profile 無此資訊）
      vix_level                 ← 20.0（無 VIX 時使用基準值）
    """
    # Step 3 寫到 results/fear/；cache/ 下若有舊版也嘗試讀取
    candidates = [
        ROOT / "results" / "fear" / f"{symbol}_fear_profile.json",
        CACHE_DIR / f"{symbol}_fear_profile.json",   # 相容舊路徑
    ]
    path = None
    for c in candidates:
        if c.exists():
            path = c
            break
    if path is None:
        return None

    try:
        with open(path) as f:
            fp = json.load(f)

        # 欄位映射
        fear_drift_decay = float(fp.get("fear_drift_decay_rate", 0.3))
        fear_vol_mult    = float(fp.get("fear_vol_mult", 1.3))

        profile = {
            # 動能強度：恐慌衰減越高 → 動能越低
            "retail_momentum_strength": round(float(np.clip(1.0 - fear_drift_decay, 0.1, 0.9)), 3),
            # 恐慌靈敏度：vol_mult 正規化（1.0 = 無放大 → 0.0）
            "retail_panic_sensitivity": round(float(np.clip((fear_vol_mult - 1.0) / 2.0, 0.1, 0.9)), 3),
            # 機構 MR 強度：fear_profile 無此資訊，保持中性
            "inst_mr_strength": 0.5,
            "vix_level": 20.0,
            # 保留原始 fear 欄位備查
            "_fear_threshold":        fp.get("fear_threshold"),
            "_fear_vol_mult":         fear_vol_mult,
            "_fear_drift_decay_rate": fear_drift_decay,
            "_source":                str(path),
        }
        print(f"[app] fear_profile 載入 ← {path}")
        print(f"  retail_momentum={profile['retail_momentum_strength']}  "
              f"panic_sens={profile['retail_panic_sensitivity']}  "
              f"inst_mr={profile['inst_mr_strength']}")
        return profile
    except Exception as e:
        print(f"[app] fear_profile 讀取失敗（{e}），跳過")
        return None


def load_pipeline_param_model(symbol: str):
    """讀取 Step 5 訓練好的 GBM 代理模型。"""
    path = ROOT / "models" / f"param_model_{symbol}.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        payload = joblib.load(path)
        print(f"[app] param_model 載入 ← {path}")
        return payload
    except Exception as e:
        print(f"[app] param_model 讀取失敗（{e}），跳過")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# apply_agent_profile（與 rolling_forward.py 邏輯一致，保留供 fallback 路徑用）
# ─────────────────────────────────────────────────────────────────────────────
def apply_agent_profile(
    profile: dict,
    base_momentum_boost: float,
    base_mr_coeff: float,
    vix_now: float | None = None,
) -> tuple[float, float]:
    retail_mom  = float(profile.get("retail_momentum_strength",  0.5))
    panic_sens  = float(profile.get("retail_panic_sensitivity",  0.3))
    inst_mr_raw = float(profile.get("inst_mr_strength", 0.5))
    inst_mr     = float(np.clip(inst_mr_raw, 0.1, 1.0))
    vix_cal     = float(profile.get("vix_level") or 20.0)
    vix_ref     = vix_now if vix_now is not None else vix_cal

    mom_adj   = (retail_mom - 0.5) * 0.6
    panic_adj = -panic_sens * float(np.clip((vix_ref - 20) / 30, 0.0, 1.0)) * 0.3
    new_momentum_boost = float(np.clip(base_momentum_boost + mom_adj + panic_adj, 0.3, 2.5))

    mr_adj = (inst_mr - 0.5) * 0.04
    new_mr_coeff = float(np.clip(base_mr_coeff + mr_adj, 0.01, 0.20))
    return new_momentum_boost, new_mr_coeff


# ─────────────────────────────────────────────────────────────────────────────
# 原有即時估算工具（保留供 fallback）
# ─────────────────────────────────────────────────────────────────────────────
def get_market_params(df: pd.DataFrame):
    return MarketParameterEstimator().fit(df)


def get_theta_live(market_params) -> CalibratedTheta:
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
        "drift_scale":    round(float(np.clip(abs(drift) / max(vol, 1e-6) * 0.8 + 0.5, 0.3, 3.2)), 3),
        "drift_decay":    0.005,
        "vol_multiplier": round(float(np.clip(vol / 0.015, 0.5, 3.0)), 3),
        "intra_bar":      round(float(np.clip(vol * 1.5, 0.005, 0.06)), 4),
    }


def build_backbone_fwd(
    close_hist: np.ndarray,
    forecast_bars: int,
    short_drift_weight: float = 0.4,
) -> tuple[np.ndarray, float]:
    fitter    = BackboneFitter(n_seg=6, smooth_reg=0.5)
    window    = close_hist[-500:] if len(close_hist) > 500 else close_hist
    bb_result = fitter.fit(window)
    last_drift = float(bb_result.segment_drifts[-1])

    log_rets    = np.diff(np.log(window.astype(float)))
    short_drift = float(np.mean(log_rets[-5:])) if len(log_rets) >= 5 else last_drift
    blend_drift = short_drift_weight * short_drift + (1.0 - short_drift_weight) * last_drift

    decay_factor = 0.98
    t            = np.arange(1, forecast_bars + 1, dtype=float)
    decayed      = blend_drift * (decay_factor ** t)

    last_price = float(close_hist[-1])
    bb_fwd     = last_price * np.cumprod(1.0 + decayed)
    return bb_fwd, blend_drift


# ─────────────────────────────────────────────────────────────────────────────
# df_to_ohlc_list
# ─────────────────────────────────────────────────────────────────────────────
def df_to_ohlc_list(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in df.iterrows():
        d  = row["Date"]
        ts = int(d.timestamp()) if hasattr(d, "timestamp") else int(pd.Timestamp(d).timestamp())
        out.append({
            "time":  ts,
            "open":  round(float(row["Open"]),  4),
            "high":  round(float(row["High"]),  4),
            "low":   round(float(row["Low"]),   4),
            "close": round(float(row["Close"]), 4),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# API  —  /api/simulate
#   使用 rolling_forward() 取代單次連續模擬，與 run_pipeline.py 完全共用引擎。
#   需要先跑過 run_pipeline.py 產生 cache/{SYM}_theta.json，否則回傳 400。
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        body          = request.get_json(force=True)
        symbol        = body.get("symbol",        "AAPL").upper().strip()
        end_date_str  = body.get("end_date",       "2025-01-01")
        forecast_bars = int(body.get("forecast_bars", 30))
        hist_bars     = int(body.get("hist_bars",  120))
        n_sim_paths   = int(body.get("n_sim_paths", 500))
        step          = int(body.get("step", 5))

        end_dt      = datetime.strptime(end_date_str, "%Y-%m-%d")
        fetch_start = (end_dt - timedelta(days=900)).strftime("%Y-%m-%d")
        fetch_end   = (end_dt + timedelta(days=forecast_bars * 2 + 30)).strftime("%Y-%m-%d")

        df_all = get_ohlcv(symbol, fetch_start, fetch_end)
        if df_all.empty or len(df_all) < 60:
            return jsonify({"error": f"資料不足：{symbol} {fetch_start}~{fetch_end}"}), 400

        # 找歷史截止 index
        mask = df_all["Date"] <= end_dt
        if not mask.any():
            return jsonify({"error": "找不到指定日期之前的資料"}), 400
        train_end_idx = int(mask.values.nonzero()[0][-1]) + 1

        df_hist          = df_all.iloc[:train_end_idx]
        actual_future_df = df_all.iloc[train_end_idx: train_end_idx + forecast_bars].copy()

        if len(df_hist) < 100:
            return jsonify({"error": "end_date 之前的歷史資料不足 100 根"}), 400

        # 讀取 Pipeline cache
        theta = load_pipeline_theta(symbol)
        if theta is None:
            return jsonify({
                "error": (
                    f"尚未找到 cache/{symbol}_theta.json，"
                    "請先執行 python scripts/run_pipeline.py "
                    f"--symbol {symbol} 產生 pipeline cache"
                )
            }), 400

        agent_profile     = load_pipeline_agent_profile(symbol)
        param_model       = load_pipeline_param_model(symbol)
        vix_series        = _load_vix_series(str(CACHE_DIR))

        # SimArgs：對應 rolling_forward._run_one_step 所有 args 屬性
        class SimArgs:
            def __init__(self):
                self.step              = step
                self.total_bars        = forecast_bars
                self.lookback          = hist_bars
                self.n_paths           = n_sim_paths
                self.calib_window      = 500
                self.no_garch          = False
                self.garch_model       = "gjr-garch"
                self.auto_calibrate    = True
                self.short_drift_weight= 0.4
                self.sr_window         = 90
                self.sr_bins           = 40
                self.sr_top_k          = 5
                self.sr_pivot_order    = 5
                self.sr_zone_pct       = 0.035
                self.no_sr             = False
                self.dynamic_drift     = True
                self.mr_rate           = 0.08
                self.trend_strength    = 0.5
                self.vol_refit         = 5
                self.early_decay_bars  = 5
                self.reversal_window   = 3
                self.verbose           = False

        args = SimArgs()

        # 呼叫核心引擎
        result = rolling_forward(
            df=df_all,
            train_end_idx=train_end_idx,
            theta=theta,
            args=args,
            actual_future_df=actual_future_df if not actual_future_df.empty else None,
            param_model_payload=param_model,
            agent_profile=agent_profile,
            vix_series=vix_series,
        )

        # 展平分段 OHLC（pred_ohlcs 是 list of dict{open/high/low/close: np.ndarray}）
        sim_candles_raw: list[dict] = []
        for rnd_ohlc in result["pred_ohlcs"]:
            if rnd_ohlc is None:
                continue
            opens  = rnd_ohlc["open"]
            highs  = rnd_ohlc["high"]
            lows   = rnd_ohlc["low"]
            closes = rnd_ohlc["close"]
            for i in range(len(opens)):
                sim_candles_raw.append({
                    "open":  round(float(opens[i]),  4),
                    "high":  round(float(highs[i]),  4),
                    "low":   round(float(lows[i]),   4),
                    "close": round(float(closes[i]), 4),
                })

        # 時間軸對齊
        future_dates = actual_future_df["Date"].tolist() if not actual_future_df.empty else []
        d = future_dates[-1] if future_dates else end_dt
        while len(future_dates) < forecast_bars:
            d += timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d)

        future_ts = [int(pd.Timestamp(dt).timestamp()) for dt in future_dates[:forecast_bars]]
        sim_candles = [
            {"time": future_ts[i], **bar}
            for i, bar in enumerate(sim_candles_raw)
            if i < len(future_ts)
        ]

        # 組合 tag
        cache_tag = "pipeline cache"
        agent_tag = "agent ON"  if agent_profile else "agent OFF"
        model_tag = "model ON"  if param_model   else "model OFF"

        return jsonify({
            "symbol":         symbol,
            "end_date":       end_date_str,
            "hist_candles":   df_to_ohlc_list(df_hist.tail(hist_bars)),
            "actual_candles": df_to_ohlc_list(actual_future_df),
            "sim_candles":    sim_candles,
            "backbone_line":  [],   # rolling 模式下骨幹每輪重建，此處留空
            "n_sim_paths":    n_sim_paths,
            "auto_params":    {
                "drift_scale":    "Dynamic",
                "drift_decay":    "Dynamic",
                "vol_multiplier": "Dynamic",
            },
            "theta":       {"vol": round(theta.vol, 5), "mr_coeff": round(theta.mr_coeff, 5)},
            "blend_drift": "Dynamic",
            "cache_tag":   cache_tag,
            "agent_tag":   agent_tag,
            "model_tag":   model_tag,
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
    --backbone: #ff9900;
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
  input[type="checkbox"] { width: auto; cursor: pointer; }
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
  .tag-green  { background: #1a2e1a; color: #6daa45; }
  .tag-orange { background: #2e1e0a; color: #ff9900; }
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
  <div class="ctrl-group"><label>滾動步長</label><input type="number" id="rolling-step" value="5" min="1" max="30" /></div>
  <div class="ctrl-group"><label>歷史顯示根數</label><input type="number" id="hist-bars" value="60" min="20" max="250" /></div>
  <div class="ctrl-group"><label>模擬路徑數</label><input type="number" id="n-sim-paths" value="500" min="50" max="2000" /></div>
  <div class="ctrl-group"><label>顯示骨幹</label><input type="checkbox" id="show-backbone" checked /></div>
  <button id="run-btn" onclick="runSim()">執行</button>
  <button id="resim-btn" onclick="runSim()" disabled>重新模擬</button>
</header>
<div id="info-bar"><span id="info-text">輸入參數後點擊「執行」</span></div>
<div id="chart-wrap">
  <div id="chart"></div>
  <div id="legend">
    <div><span class="legend-dot" style="background:#6b7280"></span>歷史 K 棒</div>
    <div><span class="legend-dot" style="background:#4f98a3"></span>模擬（Rolling Medoid）</div>
    <div><span class="legend-dot" style="background:#f0c040"></span>實際走勢</div>
  </div>
  <div id="loading"><div class="spinner"></div><span id="loading-text">模擬中...</span></div>
</div>
<div id="toast"></div>
<script>
let chart = null, histSeries = null, simSeries = null, actualSeries = null, backboneSeries = null;

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
  [histSeries, simSeries, actualSeries, backboneSeries].forEach(s => { if (s) chart.removeSeries(s); });
  histSeries = simSeries = actualSeries = backboneSeries = null;
}

function renderData(data) {
  clearSeries();
  const showBb = document.getElementById('show-backbone').checked;

  histSeries = chart.addCandlestickSeries({
    upColor: '#4b5563', downColor: '#374151',
    borderUpColor: '#6b7280', borderDownColor: '#4b5563',
    wickUpColor: '#6b7280', wickDownColor: '#4b5563',
  });
  histSeries.setData(data.hist_candles);

  simSeries = chart.addCandlestickSeries({
    upColor: '#4f98a3', downColor: '#2a5f68',
    borderUpColor: '#4f98a3', borderDownColor: '#2a5f68',
    wickUpColor: '#4f98a3', wickDownColor: '#2a5f68',
  });
  simSeries.setData(data.sim_candles);

  if (showBb && data.backbone_line && data.backbone_line.length > 0) {
    backboneSeries = chart.addLineSeries({
      color: '#ff9900', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: false,
    });
    backboneSeries.setData(data.backbone_line);
  }

  if (data.actual_candles && data.actual_candles.length > 0) {
    actualSeries = chart.addCandlestickSeries({
      upColor: '#f0c040', downColor: '#c07820',
      borderUpColor: '#f0c040', borderDownColor: '#c07820',
      wickUpColor: '#f0c040', wickDownColor: '#c07820',
    });
    actualSeries.setData(data.actual_candles);
  }

  chart.timeScale().fitContent();
}

async function runSim() {
  const symbol       = document.getElementById('symbol').value.trim().toUpperCase();
  const endDate      = document.getElementById('end-date').value;
  const forecastBars = parseInt(document.getElementById('forecast-bars').value);
  const step         = parseInt(document.getElementById('rolling-step').value) || 5;
  const histBars     = parseInt(document.getElementById('hist-bars').value);
  const nSimPaths    = parseInt(document.getElementById('n-sim-paths').value) || 500;

  if (!symbol || !endDate) { showToast('請填寫股票代碼和截止日'); return; }

  setLoading(true, `Rolling 模擬中（步長 ${step}，${nSimPaths} 條路徑）...`);
  document.getElementById('run-btn').disabled   = true;
  document.getElementById('resim-btn').disabled = true;

  try {
    const resp = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol, end_date: endDate,
        forecast_bars: forecastBars, hist_bars: histBars,
        n_sim_paths: nSimPaths,
        step: step,
        seed: Math.floor(Math.random() * 99999),
      }),
    });
    const data = await resp.json();
    if (data.error) { showToast(data.error); if (data.trace) console.error(data.trace); return; }

    if (!chart) initChart();
    renderData(data);

    const th = data.theta || {};
    const cacheTag = '<span class="tag tag-green">pipeline cache</span>';
    const agentTag = data.agent_tag === 'agent ON'
      ? '<span class="tag tag-orange">agent ON</span>'
      : '<span class="tag tag-muted">agent OFF</span>';
    const modelTag = data.model_tag === 'model ON'
      ? '<span class="tag tag-teal">model ON</span>'
      : '<span class="tag tag-muted">model OFF</span>';

    document.getElementById('info-text').innerHTML = [
      `<span class="tag tag-muted">${data.symbol}  截至 ${data.end_date}</span>`,
      cacheTag,
      agentTag,
      modelTag,
      `<span>Rolling step=<b>${step}</b></span>`,
      `<span>Medoid / <b>${data.n_sim_paths}</b> 條</span>`,
      `<span>theta.vol <b>${th.vol ?? '-'}</b></span>`,
      `<span>theta.mr <b>${th.mr_coeff ?? '-'}</b></span>`,
      `<span class="tag tag-yellow">黃=實際</span>`,
      `<span class="tag tag-teal">藍綠=Rolling Medoid</span>`,
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
  el.textContent = msg; el.classList.add('show');
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
