"""
collect_training_data.py

對指定時間段做 Walk-Forward Grid Search：
  - 每隔 --step 根取一個預測起點
  - 對每個起點用 grid search 找讓 mae_pct 最小的 (drift_scale, drift_decay) 組合
  - momentum_boost 固定 0.8（grid search 結果全選 0.8，不納入搜尋）
  - 同時記錄前500根的特徵向量
  - 把 (特徵, 最佳參數) 寫進 CSV

特徵版本 v2（30 個 + GARCH 5 個 optional）
新增特徵：
  Group A（趨勢強度）: trend_strength_adx, rsi_14, price_pos_52w, obv_slope
  Group B（開盤/流動性）: open_gap_mean, open_gap_vol, open_range_ratio, amihud_illiq
  Group C（市場狀態）: hurst_exp, trend_consistency_20, vol_of_vol, skew_20

用法：
  python scripts/collect_training_data.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --end-date 2024-12-31 \\
    --step 5 \\
    --forecast 30 \\
    --output results/training_data_AAPL.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from backbone_fitter import BackboneFitter
from calibrated_simulator import CalibratedTheta, build_params_from_theta
from us_equity_simulator import USStockFutureSimulator


# ─────────────────────────────────────────────────────────────
# Grid 定義
# ─────────────────────────────────────────────────────────────
DRIFT_SCALE_GRID    = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2]
MOMENTUM_BOOST_FIXED = 0.8            # grid search 結果全選 0.8，固定不搜尋
DRIFT_DECAY_GRID    = [0.03, 0.05, 0.07, 0.10, 0.13]


# ─────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────
def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def garch_features(close_arr):
    """回傳 GARCH 特徵，失敗時回傳空 dict。"""
    try:
        from arch import arch_model
        rets = pd.Series(np.diff(np.log(close_arr)) * 100).dropna()
        if len(rets) < 60:
            return {}
        am  = arch_model(rets, vol="Garch", p=1, o=1, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
        p   = res.params
        alpha = float(p.get("alpha[1]", 0))
        gamma = float(p.get("gamma[1]", 0))
        beta  = float(p.get("beta[1]",  0))
        fc    = res.forecast(horizon=1, reindex=False)
        fvol  = float(np.sqrt(fc.variance.values[-1, 0])) / 100
        return {
            "garch_alpha":       round(alpha, 4),
            "garch_gamma":       round(gamma, 4),
            "garch_beta":        round(beta, 4),
            "garch_persistence": round(alpha + beta + 0.5 * gamma, 4),
            "garch_forecast_vol": round(fvol * 100, 4),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# Group A：趨勢強度特徵
# ─────────────────────────────────────────────────────────────
def _adx(h, l, c, period=14):
    """計算 ADX(period)，回傳最後一個值。資料不足時回傳 20.0（中性）。"""
    n = len(c)
    if n < period + 2:
        return 20.0
    tr  = np.maximum(h[1:] - l[1:],
          np.maximum(np.abs(h[1:] - c[:-1]),
                     np.abs(l[1:] - c[:-1])))
    dmp = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                   np.maximum(h[1:] - h[:-1], 0.0), 0.0)
    dmm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                   np.maximum(l[:-1] - l[1:], 0.0), 0.0)

    def _ema_wilder(arr, p):
        out = np.empty(len(arr))
        out[0] = arr[0]
        k = 1.0 / p
        for i in range(1, len(arr)):
            out[i] = out[i-1] * (1 - k) + arr[i] * k
        return out

    atr   = _ema_wilder(tr,  period)
    dipos = _ema_wilder(dmp, period) / (atr + 1e-8) * 100
    dinos = _ema_wilder(dmm, period) / (atr + 1e-8) * 100
    dx    = np.abs(dipos - dinos) / (dipos + dinos + 1e-8) * 100
    adx   = _ema_wilder(dx, period)
    return float(adx[-1])


def _rsi(close_arr, period=14):
    """標準 RSI(period)，值域 0~100。資料不足回傳 50.0（中性）。"""
    if len(close_arr) < period + 1:
        return 50.0
    delta = np.diff(close_arr.astype(float))
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = float(np.mean(gain[-period:]))
    avg_l = float(np.mean(loss[-period:]))
    if avg_l < 1e-10:
        return 100.0
    rs = avg_g / avg_l
    return round(100.0 - 100.0 / (1.0 + rs), 4)


def _obv_slope(close_arr, volume_arr, window=20):
    """OBV 最近 window 日線性回歸斜率（正規化到均值）。"""
    if len(close_arr) < window + 1:
        return 0.0
    c = close_arr[-window-1:]
    v = volume_arr[-window-1:]
    direction = np.sign(np.diff(c))
    obv = np.cumsum(direction * v[1:])
    if len(obv) < 2:
        return 0.0
    x = np.arange(len(obv), dtype=float)
    slope = float(np.polyfit(x, obv, 1)[0])
    mean_vol = float(np.mean(v) + 1e-8)
    return round(slope / mean_vol, 6)


# ─────────────────────────────────────────────────────────────
# Group B：開盤 / 流動性特徵
# ─────────────────────────────────────────────────────────────
def _open_gap_stats(open_arr, close_arr, window=20):
    """
    計算開盤跳空統計。
    open_gap = open_t / close_{t-1} - 1
    回傳 (mean, std) 最近 window 根。
    """
    if len(open_arr) < window + 1:
        window = len(open_arr) - 1
    if window < 2:
        return 0.0, 0.0
    gaps = open_arr[1:] / (close_arr[:-1] + 1e-8) - 1.0
    gaps = gaps[-window:]
    return float(np.mean(gaps)), float(np.std(gaps))


def _open_range_ratio(open_arr, high_arr, low_arr, close_arr, window=20):
    """
    開盤到首次高/低的 range 佔全日 range 的比例代理（日線）。
    用 |open - (high 或 low 較近者)| / (high - low) 估計。
    """
    if len(open_arr) < window:
        window = len(open_arr)
    o = open_arr[-window:]
    h = high_arr[-window:]
    l = low_arr[-window:]
    c = close_arr[-window:]
    full_range = h - l + 1e-8
    open_to_extreme = np.minimum(np.abs(o - h), np.abs(o - l))
    ratio = open_to_extreme / full_range
    return round(float(np.mean(ratio)), 4)


def _amihud_illiq(close_arr, volume_arr, window=20):
    """
    Amihud 非流動性：mean(|ret| / dollar_volume) * 1e6。
    值越大代表流動性越差（每單位成交量衝擊越大）。
    """
    if len(close_arr) < window + 1:
        window = len(close_arr) - 1
    if window < 2:
        return 0.0
    c = close_arr[-window-1:]
    v = volume_arr[-window-1:]
    rets   = np.abs(np.diff(np.log(c + 1e-8)))
    dollar = (c[1:] * v[1:] + 1e-8)
    illiq  = rets / dollar * 1e6
    return round(float(np.mean(illiq)), 6)


# ─────────────────────────────────────────────────────────────
# Group C：市場狀態分類特徵
# ─────────────────────────────────────────────────────────────
def _hurst_rs(close_arr, window=100):
    """
    R/S 分析估計 Hurst exponent。
    < 0.5 → 均值回歸，> 0.5 → 趨勢，≈ 0.5 → 隨機遊走。
    使用多個 lag 做 OLS 估計，資料不足時回傳 0.5（中性）。
    """
    c = close_arr[-window:] if len(close_arr) >= window else close_arr
    n = len(c)
    if n < 20:
        return 0.5
    log_rets = np.diff(np.log(c + 1e-8))
    lags = [max(2, n // 8), max(4, n // 4), max(8, n // 2), n - 1]
    lags = sorted(set([l for l in lags if 2 <= l < len(log_rets)]))
    if len(lags) < 2:
        return 0.5
    rs_vals = []
    for lag in lags:
        sub = log_rets[:lag]
        mean_sub = np.mean(sub)
        dev = np.cumsum(sub - mean_sub)
        r = float(np.max(dev) - np.min(dev))
        s = float(np.std(sub, ddof=1) + 1e-10)
        rs_vals.append(r / s)
    log_lags = np.log(lags)
    log_rs   = np.log(np.array(rs_vals) + 1e-10)
    if np.ptp(log_lags) < 1e-8:
        return 0.5
    h = float(np.polyfit(log_lags, log_rs, 1)[0])
    return round(float(np.clip(h, 0.1, 0.9)), 4)


def _trend_consistency(log_rets, window=20):
    """
    最近 window 日中方向一致的比例：0.5 = 隨機，1.0 = 全部同方向。
    定義為 max(漲日比例, 跌日比例)。
    """
    if len(log_rets) < window:
        window = len(log_rets)
    if window < 2:
        return 0.5
    r = log_rets[-window:]
    up_ratio = float(np.sum(r > 0)) / window
    return round(max(up_ratio, 1.0 - up_ratio), 4)


def _vol_of_vol(log_rets, outer_window=60, inner_window=5):
    """
    二階波動率：計算 rolling(inner_window) std，再取其在 outer_window 內的 std。
    """
    if len(log_rets) < outer_window:
        outer_window = len(log_rets)
    if outer_window < inner_window + 2:
        return 0.0
    rets = log_rets[-outer_window:]
    roll_vols = [
        float(np.std(rets[max(0, i-inner_window):i]))
        for i in range(inner_window, len(rets) + 1)
    ]
    if len(roll_vols) < 2:
        return 0.0
    return round(float(np.std(roll_vols)) * 100, 4)


# ─────────────────────────────────────────────────────────────
# 主特徵提取函式（v2）
# ─────────────────────────────────────────────────────────────
def extract_features(df_window, theta_vol):
    """從一段 OHLCV 視窗計算特徵向量（v2，30個 + GARCH 5個 optional）。"""
    c = df_window["Close"].values.astype(float)
    o = df_window["Open"].values.astype(float)
    h = df_window["High"].values.astype(float)
    l = df_window["Low"].values.astype(float)
    v = df_window["Volume"].values.astype(float)

    log_rets = np.diff(np.log(c))

    # ── 原有特徵（v1） ──────────────────────────────────────
    vol_20  = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else float(np.std(log_rets))
    vol_60  = float(np.std(log_rets[-60:])) if len(log_rets) >= 60 else float(np.std(log_rets))
    vol_all = float(np.std(log_rets))

    drift_20  = float(np.mean(log_rets[-20:])) if len(log_rets) >= 20 else float(np.mean(log_rets))
    drift_60  = float(np.mean(log_rets[-60:])) if len(log_rets) >= 60 else float(np.mean(log_rets))
    drift_all = float(np.mean(log_rets))

    s = pd.Series(log_rets)
    ret_autocorr = float(s.autocorr(lag=1)) if len(s) > 10 else 0.0
    ret_autocorr = 0.0 if np.isnan(ret_autocorr) else ret_autocorr

    vret = pd.Series(v).pct_change().dropna()
    vol_autocorr = float(vret.autocorr(lag=1)) if len(vret) > 10 else 0.0
    vol_autocorr = max(0.0, 0.0 if np.isnan(vol_autocorr) else vol_autocorr)

    body_pct   = np.abs(c - o) / (o + 1e-8)
    avg_body   = float(np.mean(body_pct))
    body_size  = np.abs(c - o) + 1e-8
    sr         = (h - l) / body_size
    median_sr  = float(np.median(sr))
    p90_sr     = float(np.percentile(sr, 90))

    # 骨幹特徵
    try:
        fitter = BackboneFitter(n_seg=6, smooth_reg=0.5)
        bb = fitter.fit(c[-120:] if len(c) >= 120 else c)
        bb_last_drift = float(bb.segment_drifts[-1])
        bb_last_vol   = float(bb.segment_vols[-1])
        bb_drift_std  = float(np.std(bb.segment_drifts))
        bb_vol_std    = float(np.std(bb.segment_vols))
    except Exception:
        bb_last_drift = drift_all
        bb_last_vol   = vol_all
        bb_drift_std  = 0.0
        bb_vol_std    = 0.0

    # ── Group A：趨勢強度 ───────────────────────────────────
    adx_val      = _adx(h, l, c, period=14)
    rsi_val      = _rsi(c, period=14)
    # 52 週（252 日）相對位置，不足 252 日用全部
    c_52w = c[-252:] if len(c) >= 252 else c
    c_min, c_max = float(np.min(c_52w)), float(np.max(c_52w))
    price_pos_52w = round((c[-1] - c_min) / (c_max - c_min + 1e-8), 4)
    obv_slope_val = _obv_slope(c, v, window=20)

    # ── Group B：開盤 / 流動性 ──────────────────────────────
    gap_mean, gap_vol   = _open_gap_stats(o, c, window=20)
    open_rng_ratio      = _open_range_ratio(o, h, l, c, window=20)
    amihud_val          = _amihud_illiq(c, v, window=20)

    # ── Group C：市場狀態 ───────────────────────────────────
    hurst_val           = _hurst_rs(c, window=100)
    trend_cons_val      = _trend_consistency(log_rets, window=20)
    vov_val             = _vol_of_vol(log_rets, outer_window=60, inner_window=5)
    skew_20_val         = float(pd.Series(log_rets[-20:]).skew()) if len(log_rets) >= 20 else 0.0
    skew_20_val         = 0.0 if np.isnan(skew_20_val) else round(skew_20_val, 4)

    feat = {
        # v1 原有
        "vol_20":               round(vol_20 * 100, 4),
        "vol_60":               round(vol_60 * 100, 4),
        "vol_all":              round(vol_all * 100, 4),
        "vol_ratio_20_60":      round(vol_20 / (vol_60 + 1e-8), 4),
        "vol_ratio_rv_theta":   round(vol_20 / (theta_vol + 1e-8), 4),
        "drift_20":             round(drift_20 * 100, 4),
        "drift_60":             round(drift_60 * 100, 4),
        "drift_all":            round(drift_all * 100, 4),
        "drift_ratio_20_all":   round(drift_20 / (abs(drift_all) + 1e-8) * np.sign(drift_all + 1e-12), 4),
        "ret_autocorr":         round(ret_autocorr, 4),
        "vol_autocorr":         round(vol_autocorr, 4),
        "avg_body_pct":         round(avg_body * 100, 4),
        "median_sr":            round(median_sr, 4),
        "p90_sr":               round(p90_sr, 4),
        "bb_last_drift":        round(bb_last_drift * 100, 4),
        "bb_last_vol":          round(bb_last_vol * 100, 4),
        "bb_drift_std":         round(bb_drift_std * 100, 4),
        "bb_vol_std":           round(bb_vol_std * 100, 4),
        # Group A
        "trend_strength_adx":   round(adx_val, 4),
        "rsi_14":               round(rsi_val, 4),
        "price_pos_52w":        price_pos_52w,
        "obv_slope":            obv_slope_val,
        # Group B
        "open_gap_mean":        round(gap_mean * 100, 4),
        "open_gap_vol":         round(gap_vol * 100, 4),
        "open_range_ratio":     open_rng_ratio,
        "amihud_illiq":         amihud_val,
        # Group C
        "hurst_exp":            hurst_val,
        "trend_consistency_20": trend_cons_val,
        "vol_of_vol":           vov_val,
        "skew_20":              skew_20_val,
    }
    feat.update(garch_features(c))
    return feat


def run_sim_and_mae(
    df, train_end_idx, theta, args,
    drift_scale, momentum_boost, drift_decay,
    vol_multiplier, n_paths=200,
):
    """跑一次模擬，回傳 mae_pct（越小越好）。"""
    ESTIMATOR_LB = 500
    train_df    = df.iloc[train_end_idx - args.lookback: train_end_idx]
    estimate_df = df.iloc[train_end_idx - ESTIMATOR_LB: train_end_idx]
    future_df   = df.iloc[train_end_idx: train_end_idx + args.forecast]

    if len(future_df) < args.forecast:
        return None

    close_hist  = train_df["Close"].values.astype(float)
    start_price = float(close_hist[-1])
    actual_close = future_df["Close"].values.astype(float)

    fitter    = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb_result = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol) * vol_multiplier
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

    rv = float(np.std(np.diff(np.log(close_hist[-21:]))))
    vol_scale = float(np.clip(rv / max(theta.vol, 1e-8), 0.6, 4.0))

    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = dataclasses.replace(
        base_params, last_close=start_price,
        momentum_bias=0.0, node_breakout_state=0
    )
    params_fwd = build_params_from_theta(theta, params_fwd)

    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=args.forecast,
        n_paths=n_paths,
        seed=42,
        vol_scale=vol_scale,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength * momentum_boost,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_fwd,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=0.06,
        intra_bar_steps=3,
        drift_decay_rate=drift_decay,
        drift_scale=drift_scale,
        momentum_anchor_weight=0.45,
    )
    result = sim.simulate()
    n = min(len(actual_close), len(result.median_path))
    mae = float(np.mean(np.abs(actual_close[:n] - result.median_path[:n]) / start_price * 100))
    return mae


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    required=True)
    p.add_argument("--theta",     required=True)
    p.add_argument("--end-date",  default=None,
                   help="資料截止日（預測最後起點），預設今天")
    p.add_argument("--lookback",  type=int, default=120)
    p.add_argument("--forecast",  type=int, default=30)
    p.add_argument("--step",      type=int, default=5,
                   help="每隔幾根取一個起點")
    p.add_argument("--n-paths",   type=int, default=200,
                   help="grid search 用的路徑數（小一點快很多）")
    p.add_argument("--calib-window", type=int, default=500)
    p.add_argument("--output",    default="results/training_data.csv")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    ESTIMATOR_LB = 500
    MIN_BEFORE = ESTIMATOR_LB + args.lookback

    print(f"Downloading {args.symbol}...")
    end_dt   = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today()
    start_dt = end_dt - pd.DateOffset(years=5)
    dl_end   = end_dt + pd.DateOffset(days=args.forecast * 2 + 10)

    df_raw = yf.download(
        args.symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=dl_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    print(f"Total bars: {len(df)}")

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        dates = pd.to_datetime(df["Datetime"])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])

    mask = dates <= end_dt
    if not mask.any():
        raise ValueError(f"{end_dt.date()} 之前找不到資料")
    max_end_idx = int(mask.values.nonzero()[0][-1]) + 1

    # 收集所有候選起點（不重複）
    seen = set()
    candidates = []
    idx = max_end_idx
    while idx >= MIN_BEFORE:
        if idx + args.forecast <= len(df) and idx not in seen:
            candidates.append(idx)
            seen.add(idx)
        idx -= args.step
    candidates = list(reversed(candidates))
    print(f"候選起點數: {len(candidates)}")

    grid = list(product(DRIFT_SCALE_GRID, DRIFT_DECAY_GRID))
    total_combos = len(grid)
    print(f"Grid 大小: {total_combos} 組  x  {len(candidates)} 起點 = {total_combos * len(candidates)} 次模擬")
    print(f"  (momentum_boost 固定 {MOMENTUM_BOOST_FIXED}，不納入 grid search)")
    print("開始收集... (每個起點約需數秒)\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    FEATURE_KEYS = None

    for i, train_end_idx in enumerate(candidates):
        date_str = str(dates.iloc[train_end_idx - 1].date())
        _row_prefix = f"[{i+1}/{len(candidates)}] 起點 {date_str} (idx={train_end_idx})  "

        # 1. 計算特徵
        calib_df = df.iloc[train_end_idx - args.calib_window: train_end_idx]
        feats = extract_features(calib_df, theta.vol)
        if FEATURE_KEYS is None:
            FEATURE_KEYS = list(feats.keys())

        # 2. auto-calibrate 的 vol_multiplier（固定用，不 grid search）
        log_rets = np.diff(np.log(
            df.iloc[train_end_idx - args.calib_window: train_end_idx]["Close"].values.astype(float)
        ))
        rv = float(np.std(log_rets))
        vol_multiplier = float(np.clip(rv / max(theta.vol, 1e-8), 0.5, 3.0))

        # 3. grid search（只搜 drift_scale x drift_decay）
        best_mae    = float("inf")
        best_params = {"drift_scale": 1.0, "momentum_boost": MOMENTUM_BOOST_FIXED, "drift_decay": 0.07}

        for ds, dd in grid:
            mae = run_sim_and_mae(
                df, train_end_idx, theta, args,
                drift_scale=ds, momentum_boost=MOMENTUM_BOOST_FIXED, drift_decay=dd,
                vol_multiplier=vol_multiplier, n_paths=args.n_paths,
            )
            if mae is not None and mae < best_mae:
                best_mae    = mae
                best_params = {"drift_scale": ds, "momentum_boost": MOMENTUM_BOOST_FIXED, "drift_decay": dd}

        print(_row_prefix + f"best_mae={best_mae:.3f}%  ds={best_params['drift_scale']}  "
              f"dd={best_params['drift_decay']}")

        row = {
            "date":            date_str,
            "train_end_idx":   train_end_idx,
            "best_mae":        round(best_mae, 4),
            **{f"feat_{k}": v for k, v in feats.items()},
            "target_drift_scale":    best_params["drift_scale"],
            "target_momentum_boost": best_params["momentum_boost"],
            "target_drift_decay":    best_params["drift_decay"],
        }
        rows.append(row)

    # 寫 CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✔ 已儲存 {len(rows)} 筆訓練資料 → {out_path}")
        print(f"  特徵數: {len(FEATURE_KEYS)} 個（不含 GARCH optional）")
    else:
        print("\n⚠ 沒有產生任何資料，請確認時間範圍與資料筆數")


if __name__ == "__main__":
    main()
