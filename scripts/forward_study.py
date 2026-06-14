"""
forward_study.py  v8  (auto-calibrate)

新增 --auto-calibrate:
  掃描前 n 根 K 棒的統計特徵，自動決定以下參數：
    intra_bar       ← avg_body_pct
    shadow_noise    ← median shadow_ratio
    shadow_clamp    ← p90 shadow_ratio
    momentum_boost  ← volume autocorrelation
    drift_decay     ← price return autocorrelation
    vol_multiplier  ← realized_vol / theta.vol
    drift_scale     ← last_seg_drift / hist_drift_all (相對斜率比)

  手動傳入的同名參數會覆蓋自動校準值。
  --calib-window 控制校準用的 K 棒數（預設 500）。
"""

from __future__ import annotations

import argparse
import json
import sys
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from backbone_fitter import BackboneFitter
from calibrated_simulator import CalibratedTheta, build_params_from_theta
from us_equity_simulator import USStockFutureSimulator
from candle_renderer import render_forecast_candles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",            required=True)
    p.add_argument("--theta",             required=True)
    p.add_argument("--lookback",          type=int,   default=120)
    p.add_argument("--forecast",          type=int,   default=30)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--n-paths",           type=int,   default=500)
    p.add_argument("--n-seg",             type=int,   default=6)
    p.add_argument("--smooth-reg",        type=float, default=0.5)
    p.add_argument("--backbone-mr",       type=float, default=0.06)
    p.add_argument("--period",            default="3y")
    p.add_argument("--interval",          default="1d")
    p.add_argument("--hist-window",       type=int,   default=60)
    p.add_argument("--intra-bar",         type=int,   default=None)
    p.add_argument("--drift-decay",       type=float, default=None)
    p.add_argument("--drift-scale",       type=float, default=None)
    p.add_argument("--anchor-weight",     type=float, default=0.45)
    p.add_argument("--vol-multiplier",    type=float, default=None)
    p.add_argument("--recent-vol-window", type=int,   default=20)
    p.add_argument("--vol-scale-min",     type=float, default=0.6)
    p.add_argument("--vol-scale-max",     type=float, default=4.0)
    p.add_argument("--shadow-noise",      type=float, default=None)
    p.add_argument("--shadow-clamp",      type=float, default=None)
    p.add_argument("--momentum-boost",    type=float, default=None)
    p.add_argument("--path-spread",       type=float, default=1.0)
    p.add_argument("--output",            default="forward_study")
    # --- auto-calibrate ---
    p.add_argument("--auto-calibrate",    action="store_true",
                   help="從前 n 根 K 棒自動決定模擬參數")
    p.add_argument("--calib-window",      type=int, default=500,
                   help="auto-calibrate 使用的 K 棒數（預設 500）")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def recent_realized_vol(close_arr: np.ndarray, window: int) -> float:
    w = min(window, len(close_arr) - 1)
    log_rets = np.diff(np.log(close_arr[-(w + 1):]))
    return float(np.std(log_rets))


def auto_calibrate(df: pd.DataFrame, window: int, theta_vol: float,
                   last_seg_drift: float) -> dict:
    """
    從前 window 根 K 棒的統計特徵自動計算模擬參數。
    欄位名稱使用 ensure_ohlcv 正規化後的大寫名。
    """
    d = df.tail(window).copy()
    o = d["Open"].values.astype(float)
    h = d["High"].values.astype(float)
    l = d["Low"].values.astype(float)
    c = d["Close"].values.astype(float)
    v = d["Volume"].values.astype(float)

    # 1. 實體大小 → intra_bar
    body_pct = np.abs(c - o) / (o + 1e-8)
    avg_body = float(np.mean(body_pct))
    if avg_body > 0.015:
        intra_bar = 2
    elif avg_body > 0.008:
        intra_bar = 3
    else:
        intra_bar = 4

    # 2. 影線比例 → shadow_noise + shadow_clamp
    body_size = np.abs(c - o) + 1e-8
    full_range = h - l
    sr = full_range / body_size
    median_sr = float(np.median(sr))
    p90_sr    = float(np.percentile(sr, 90))
    shadow_noise = float(np.clip(median_sr * 0.08, 0.06, 0.30))
    shadow_clamp = float(np.clip(p90_sr * 0.8, 1.5, 5.0))

    # 3. 成交量自相關 → momentum_boost
    v_ret = pd.Series(v).pct_change().dropna()
    vol_autocorr = float(v_ret.autocorr(lag=1)) if len(v_ret) > 10 else 0.0
    vol_autocorr = max(vol_autocorr, 0.0)  # 只用正相關
    momentum_boost = float(np.clip(1.0 + vol_autocorr * 2.0, 1.0, 3.0))

    # 4. 價格報酬自相關 → drift_decay
    #    正自相關（趨勢市）→ 慢衰減；負自相關（震盪市）→ 快衰減
    p_ret = pd.Series(c).pct_change().dropna()
    ret_autocorr = float(p_ret.autocorr(lag=1)) if len(p_ret) > 10 else 0.0
    drift_decay = float(np.clip(0.07 - ret_autocorr * 0.06, 0.01, 0.15))

    # 5. 實現波動率 → vol_multiplier
    log_rets = np.diff(np.log(c))
    rv = float(np.std(log_rets))
    vol_multiplier = float(np.clip(rv / max(theta_vol, 1e-8), 0.5, 3.0))

    # 6. 相對斜率比 → drift_scale
    #    最近段斜率 vs 整體歷史斜率 → 判斷現在是「強勢」還是「弱勢」
    hist_drift_all = float(np.mean(log_rets))
    if abs(hist_drift_all) > 1e-6:
        relative_momentum = last_seg_drift / abs(hist_drift_all)
        drift_scale = float(np.clip(relative_momentum * 0.6, 0.5, 2.5))
    else:
        drift_scale = 1.0

    return {
        "intra_bar":       intra_bar,
        "shadow_noise":    round(shadow_noise, 3),
        "shadow_clamp":    round(shadow_clamp, 2),
        "momentum_boost":  round(momentum_boost, 2),
        "drift_decay":     round(drift_decay, 4),
        "vol_multiplier":  round(vol_multiplier, 3),
        "drift_scale":     round(drift_scale, 3),
        # 診斷用
        "_avg_body_pct":   round(avg_body * 100, 4),
        "_median_sr":      round(median_sr, 3),
        "_p90_sr":         round(p90_sr, 3),
        "_vol_autocorr":   round(vol_autocorr, 4),
        "_ret_autocorr":   round(ret_autocorr, 4),
        "_rv_pct":         round(rv * 100, 4),
        "_hist_drift_all": round(hist_drift_all * 100, 4),
        "_last_seg_drift": round(last_seg_drift * 100, 4),
    }


def clamp_shadows(ohlcv_open, ohlcv_high, ohlcv_low, ohlcv_close, shadow_clamp: float):
    if shadow_clamp <= 0:
        return ohlcv_high.copy(), ohlcv_low.copy()
    body_top   = np.maximum(ohlcv_open, ohlcv_close)
    body_bot   = np.minimum(ohlcv_open, ohlcv_close)
    body_size  = body_top - body_bot
    max_shadow = body_size * shadow_clamp
    new_high   = np.minimum(ohlcv_high, body_top + max_shadow)
    new_low    = np.maximum(ohlcv_low,  body_bot - max_shadow)
    new_high   = np.maximum(new_high, body_top)
    new_low    = np.minimum(new_low,  body_bot)
    return new_high, new_low


def compute_metrics(actual, median, p25, p75, p10, p90, start_price):
    n = min(len(actual), len(median))
    if n == 0:
        return {}
    act, med  = actual[:n], median[:n]
    hit_25_75 = float(np.mean((act >= p25[:n]) & (act <= p75[:n])))
    hit_10_90 = float(np.mean((act >= p10[:n]) & (act <= p90[:n])))
    actual_dir    = np.sign(np.diff(np.concatenate([[start_price], act])))
    median_dir    = np.sign(np.diff(np.concatenate([[start_price], med])))
    direction_acc = float(np.mean(actual_dir == median_dir))
    end_error = float(abs(act[-1] - med[-1]) / start_price * 100)
    mae_pct   = float(np.mean(np.abs(act - med) / start_price * 100))
    max_dev   = float(np.max(np.abs(act - med) / start_price * 100))
    return {
        "n_compared":        n,
        "hit_rate_25_75":    round(hit_25_75,    4),
        "hit_rate_10_90":    round(hit_10_90,    4),
        "direction_acc":     round(direction_acc, 4),
        "end_error_pct":     round(end_error,    4),
        "mae_pct":           round(mae_pct,      4),
        "max_deviation_pct": round(max_dev,      4),
        "bars_above_p90":    int(np.sum(act > p90[:n])),
        "bars_below_p10":    int(np.sum(act < p10[:n])),
    }


def print_diagnostics(
    args, theta, close_hist, actual_close,
    rv, vol_scale, last_drift, last_vol, bb_result,
    result, clamped_high, clamped_low, start_price, metrics,
    calib_info=None
):
    T = args.forecast
    actual_end   = float(actual_close[-1]) if len(actual_close) else None
    median_end   = float(result.median_path[-1])
    actual_chg   = (actual_end / start_price - 1) * 100 if actual_end else None
    median_chg   = (median_end / start_price - 1) * 100
    p10_end      = float(result.p10[-1])
    p90_end      = float(result.p90[-1])
    band_width   = (p90_end - p10_end) / start_price * 100
    center_bias  = (median_end - actual_end) / start_price * 100 if actual_end else None

    if len(actual_close) >= 2:
        actual_daily_drift = float(np.mean(np.diff(np.log(actual_close))))
    else:
        actual_daily_drift = None
    median_daily_drift = float(np.mean(np.diff(np.log(result.median_path))))

    hist_drift_20  = float(np.mean(np.diff(np.log(close_hist[-21:]))))
    hist_drift_60  = float(np.mean(np.diff(np.log(close_hist[-61:])))) if len(close_hist) >= 61 else None
    hist_drift_all = float(np.mean(np.diff(np.log(close_hist))))

    seg_drifts = [f"{d*100:+.3f}%" for d in bb_result.segment_drifts]
    seg_vols   = [f"{v*100:.3f}%"  for v in bb_result.segment_vols]

    body_sizes    = np.abs(result.ohlcv_close - result.ohlcv_open)
    upper_shadows = clamped_high - np.maximum(result.ohlcv_open, result.ohlcv_close)
    lower_shadows = np.minimum(result.ohlcv_open, result.ohlcv_close) - clamped_low
    shadow_total  = upper_shadows + lower_shadows
    shadow_ratio  = shadow_total / (body_sizes + 1e-8)
    avg_shadow_ratio = float(np.mean(shadow_ratio))
    p50_shadow_ratio = float(np.median(shadow_ratio))
    p90_shadow_ratio = float(np.percentile(shadow_ratio, 90))
    avg_body_pct  = float(np.mean(body_sizes / start_price * 100))
    direction_consistency = float(np.mean(
        np.sign(result.ohlcv_close[1:] - result.ohlcv_close[:-1]) ==
        np.sign(result.median_path[1:] - result.median_path[:-1])
    ))
    hist_ret_std = float(np.std(np.diff(np.log(close_hist)))) * 100

    print()
    print("=" * 60)
    print("  VERBOSE DIAGNOSTICS (v8 auto-calibrate)")
    print("=" * 60)

    # --- 自動校準區塊 ---
    if calib_info is not None:
        print(f"\n[自動校準結果 (window={args.calib_window})]")
        print(f"  avg_body_pct (hist)  : {calib_info['_avg_body_pct']:.3f}%  → intra_bar={args.intra_bar}")
        print(f"  median_shadow_ratio  : {calib_info['_median_sr']:.3f}  → shadow_noise={args.shadow_noise}")
        print(f"  p90_shadow_ratio     : {calib_info['_p90_sr']:.3f}  → shadow_clamp={args.shadow_clamp}")
        print(f"  vol_autocorr         : {calib_info['_vol_autocorr']:+.4f}  → momentum_boost={args.momentum_boost}")
        print(f"  ret_autocorr         : {calib_info['_ret_autocorr']:+.4f}  → drift_decay={args.drift_decay}")
        print(f"  realized_vol/theta   : {calib_info['_rv_pct']:.4f}%/{theta.vol*100:.4f}%  → vol_multiplier={args.vol_multiplier}")
        print(f"  last_seg / hist_all  : {calib_info['_last_seg_drift']:+.4f}% / {calib_info['_hist_drift_all']:+.4f}%  → drift_scale={args.drift_scale}")

    print(f"\n[資料概況]")
    print(f"  symbol        : {args.symbol}")
    print(f"  lookback      : {args.lookback} bars")
    print(f"  forecast      : {T} bars")
    print(f"  start_price   : {start_price:.4f}")
    if actual_end:
        print(f"  actual_end    : {actual_end:.4f}  ({actual_chg:+.2f}%)")
    print(f"  median_end    : {median_end:.4f}  ({median_chg:+.2f}%)")
    print(f"  p10_end       : {p10_end:.4f}")
    print(f"  p90_end       : {p90_end:.4f}")

    print(f"\n[帶子中心偏移分析]")
    if center_bias is not None:
        direction = (
            "\u2191 median 高低估 \u2192 降低 drift_scale" if center_bias > 0
            else "\u2193 median 低估 \u2192 提高 drift_scale"
        )
        flag = " \u2705" if abs(center_bias) <= 3 else " \u274c"
        print(f"  center_bias        : {center_bias:+.2f}%  {direction}{flag}")
    print(f"  band_width_p10_90  : {band_width:.2f}%  (判斷: >{T*0.5:.0f}%=寬, <{T*0.2:.0f}%=窄)")

    print(f"\n[K 棒形態診斷]")
    status_body = " \u2705" if 0.3 <= avg_body_pct <= 1.5 else (" \u26a0 太大" if avg_body_pct > 1.5 else " \u274c 太小")
    status_shad = " \u2705" if avg_shadow_ratio <= 2.0 else f" \u274c 過大"
    print(f"  avg_body_pct       : {avg_body_pct:.3f}%  (目標: 0.3~1.5%){status_body}")
    print(f"  avg_shadow_ratio   : {avg_shadow_ratio:.2f}   (目標: 0.5~2.0){status_shad}")
    print(f"  p50_shadow_ratio   : {p50_shadow_ratio:.2f}")
    print(f"  p90_shadow_ratio   : {p90_shadow_ratio:.2f}")
    print(f"  shadow_clamp       : {args.shadow_clamp}x")
    print(f"  hist_ret_std       : {hist_ret_std:.3f}%/day")
    print(f"  direction_consist  : {direction_consistency:.2f}")

    print(f"\n[漂移對照]")
    print(f"  theta.drift             : {theta.drift*100:+.4f}%/day")
    print(f"  last_seg_drift (bb)     : {last_drift*100:+.4f}%/day")
    print(f"  hist_drift_20d          : {hist_drift_20*100:+.4f}%/day")
    if hist_drift_60:
        print(f"  hist_drift_60d          : {hist_drift_60*100:+.4f}%/day")
    print(f"  hist_drift_all({args.lookback:3d}d)  : {hist_drift_all*100:+.4f}%/day")
    print(f"  median_daily_drift      : {median_daily_drift*100:+.4f}%/day")
    if actual_daily_drift:
        print(f"  actual_daily_drift      : {actual_daily_drift*100:+.4f}%/day")
        drift_gap = (median_daily_drift - actual_daily_drift) * 100
        print(f"  drift_gap(med-act)      : {drift_gap:+.4f}%/day")

    print(f"\n[波動率分析]")
    print(f"  theta.vol (long-term)   : {theta.vol*100:.4f}%/day")
    print(f"  last_seg_vol (bb)       : {last_vol*100:.4f}%/day")
    print(f"  recent_vol ({args.recent_vol_window}d)       : {rv*100:.4f}%/day")
    print(f"  vol_scale (rv/theta)    : {vol_scale:.3f}")
    print(f"  vol_multiplier          : {args.vol_multiplier}")
    print(f"  effective_vol           : {rv * vol_scale * args.vol_multiplier * 100:.4f}%/day")

    print(f"\n[骨幹分段]")
    print(f"  n_seg         : {args.n_seg}")
    print(f"  segment_drifts: {seg_drifts}")
    print(f"  segment_vols  : {seg_vols}")
    print(f"  backbone_MSE  : {bb_result.fit_mse:.6f}")

    print(f"\n[預測參數]")
    print(f"  drift_scale    : {args.drift_scale}")
    print(f"  drift_decay    : {args.drift_decay}  (t30 殘餘: {np.exp(-args.drift_decay*30)*100:.1f}%)")
    print(f"  anchor_weight  : {args.anchor_weight}")
    print(f"  backbone_mr    : {args.backbone_mr}")
    print(f"  intra_bar      : {args.intra_bar}")
    print(f"  shadow_noise   : {args.shadow_noise}")
    print(f"  shadow_clamp   : {args.shadow_clamp}")
    print(f"  momentum_boost : {args.momentum_boost}")
    print(f"  path_spread    : {args.path_spread}")
    print(f"  n_paths        : {args.n_paths}")

    print(f"\n[表現指標]")
    for k, v in metrics.items():
        flag = ""
        if k == "hit_rate_10_90":
            flag = " \u2705 好" if v >= 0.7 else (" \u26a0 可接受" if v >= 0.5 else " \u274c 帶子太窄")
        if k == "hit_rate_25_75":
            flag = " \u2705 好" if v >= 0.4 else (" \u26a0 可接受" if v >= 0.25 else " \u274c 帶子心線偏差")
        if k == "direction_acc":
            flag = " \u2705 好" if v >= 0.6 else (" \u26a0 遠於隨機" if v >= 0.5 else " \u274c 差於擲母")
        if k == "bars_above_p90" and isinstance(v, int):
            pct = v / max(metrics.get("n_compared", 30), 1)
            if pct > 0.3: flag = f" \u274c 帶子偶爾偏低({pct:.0%})"
        if k == "bars_below_p10" and isinstance(v, int):
            pct = v / max(metrics.get("n_compared", 30), 1)
            if pct > 0.3: flag = f" \u274c 帶子偶爾偏高({pct:.0%})"
        print(f"  {k:25s}: {v}{flag}")

    print(f"\n[自動建議]")
    suggestions = []
    if metrics.get("hit_rate_10_90", 0) < 0.5:
        suggestions.append("  \u2022 帶子太窄: --vol-multiplier +0.3")
    if metrics.get("hit_rate_10_90", 0) > 0.95 and metrics.get("hit_rate_25_75", 0) < 0.25:
        suggestions.append("  \u2022 帶子寬但中心線偏: 調整 drift_scale")
    if center_bias is not None:
        if center_bias > 3.0:
            new_ds = round(args.drift_scale * 0.7, 2)
            suggestions.append(f"  \u2022 median 明顯偏高 {center_bias:+.1f}%: --drift-scale {new_ds}")
        elif center_bias < -3.0:
            new_ds = round(min(args.drift_scale * 1.35, 2.0), 2)
            suggestions.append(f"  \u2022 median 明顯偏低 {center_bias:+.1f}%: --drift-scale {new_ds}")
        else:
            suggestions.append(f"  \u2022 帶子中心線偏移 {center_bias:+.1f}% \u2705 可接受")
    if metrics.get("direction_acc", 1) < 0.5:
        suggestions.append(
            f"  \u2022 方向準確度差: --momentum-boost {round(min(args.momentum_boost + 0.3, 2.5), 1)}"
        )
    if not suggestions:
        suggestions.append("  \u2022 目前參數已處於較好狀態 \u2705")
    for s in suggestions:
        print(s)
    print("=" * 60)


def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    print(f"Downloading {args.symbol}...")
    df_raw = yf.download(args.symbol, period=args.period, interval=args.interval,
                         auto_adjust=False, progress=False)
    df = ensure_ohlcv(df_raw)
    print(f"Total bars: {len(df)}")

    ESTIMATOR_LB = 500
    needed = ESTIMATOR_LB + args.lookback + args.forecast
    if len(df) < needed:
        raise ValueError(f"Need {needed} bars, got {len(df)}")

    train_end_idx = len(df) - args.forecast
    train_df      = df.iloc[train_end_idx - args.lookback: train_end_idx]
    estimate_df   = df.iloc[train_end_idx - ESTIMATOR_LB: train_end_idx]
    future_df     = df.iloc[train_end_idx: train_end_idx + args.forecast]

    close_hist       = train_df["Close"].values
    start_price      = float(close_hist[-1])
    hist_open        = train_df["Open"].values.astype(float)
    hist_high        = train_df["High"].values.astype(float)
    hist_low         = train_df["Low"].values.astype(float)
    hist_close       = close_hist
    hist_volume      = train_df["Volume"].values.astype(float)
    hist_volume_norm = hist_volume / (hist_volume.mean() + 1e-8)

    actual_close       = future_df["Close"].values.astype(float)
    actual_open        = future_df["Open"].values.astype(float)
    actual_high        = future_df["High"].values.astype(float)
    actual_low         = future_df["Low"].values.astype(float)
    actual_volume      = future_df["Volume"].values.astype(float)
    actual_volume_norm = actual_volume / (hist_volume.mean() + 1e-8)

    print(f"Train end: {start_price:.2f}  forecast={args.forecast} bars")

    fitter    = BackboneFitter(n_seg=args.n_seg, smooth_reg=args.smooth_reg)
    bb_result = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol)
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

    rv = recent_realized_vol(close_hist, args.recent_vol_window)
    vol_scale = float(np.clip(
        rv / max(theta.vol, 1e-8),
        args.vol_scale_min, args.vol_scale_max,
    ))

    # === AUTO CALIBRATE ===
    calib_info = None
    calib_window = min(args.calib_window, len(df) - args.forecast)
    calib_df = df.iloc[train_end_idx - calib_window: train_end_idx]

    if args.auto_calibrate:
        print(f"\n[auto-calibrate] 掃描前 {calib_window} 根 K 棒...")
        calib = auto_calibrate(
            calib_df, window=calib_window,
            theta_vol=theta.vol, last_seg_drift=last_drift
        )
        calib_info = calib
        # 套用校準值，手動傳入的參數覆蓋
        if args.intra_bar       is None: args.intra_bar       = calib["intra_bar"]
        if args.shadow_noise    is None: args.shadow_noise    = calib["shadow_noise"]
        if args.shadow_clamp    is None: args.shadow_clamp    = calib["shadow_clamp"]
        if args.momentum_boost  is None: args.momentum_boost  = calib["momentum_boost"]
        if args.drift_decay     is None: args.drift_decay     = calib["drift_decay"]
        if args.vol_multiplier  is None: args.vol_multiplier  = calib["vol_multiplier"]
        if args.drift_scale     is None: args.drift_scale     = calib["drift_scale"]
        print(f"  intra_bar={args.intra_bar}  shadow_noise={args.shadow_noise}"
              f"  shadow_clamp={args.shadow_clamp}")
        print(f"  momentum_boost={args.momentum_boost}  drift_decay={args.drift_decay}"
              f"  vol_multiplier={args.vol_multiplier}  drift_scale={args.drift_scale}")
    else:
        # 沒有 auto-calibrate 時套用預設值（向後相容 v7）
        if args.intra_bar      is None: args.intra_bar      = 2
        if args.shadow_noise   is None: args.shadow_noise   = 0.15
        if args.shadow_clamp   is None: args.shadow_clamp   = 2.0
        if args.momentum_boost is None: args.momentum_boost = 1.6
        if args.drift_decay    is None: args.drift_decay    = 0.04
        if args.vol_multiplier is None: args.vol_multiplier = 1.2
        if args.drift_scale    is None: args.drift_scale    = 1.18

    vol_fwd_scaled = vol_fwd * args.vol_multiplier

    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = dataclasses.replace(
        base_params, last_close=start_price,
        momentum_bias=0.0, node_breakout_state=0
    )
    params_fwd = build_params_from_theta(theta, params_fwd)
    effective_momentum = theta.momentum_strength * args.momentum_boost

    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=args.forecast,
        n_paths=args.n_paths,
        seed=args.seed,
        vol_scale=vol_scale * args.path_spread,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=effective_momentum,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_fwd,
        vol_schedule=vol_fwd_scaled,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=args.backbone_mr,
        intra_bar_steps=args.intra_bar,
        drift_decay_rate=args.drift_decay,
        drift_scale=args.drift_scale,
        momentum_anchor_weight=args.anchor_weight,
    )
    result = sim.simulate()

    clamped_high, clamped_low = clamp_shadows(
        result.ohlcv_open, result.ohlcv_high,
        result.ohlcv_low,  result.ohlcv_close,
        args.shadow_clamp
    )

    metrics = compute_metrics(
        actual=actual_close, median=result.median_path,
        p25=result.p25, p75=result.p75,
        p10=result.p10, p90=result.p90,
        start_price=start_price,
    )

    print_diagnostics(
        args=args, theta=theta,
        close_hist=close_hist, actual_close=actual_close,
        rv=rv, vol_scale=vol_scale,
        last_drift=last_drift, last_vol=last_vol,
        bb_result=bb_result,
        result=result,
        clamped_high=clamped_high, clamped_low=clamped_low,
        start_price=start_price, metrics=metrics,
        calib_info=calib_info,
    )

    out_prefix   = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(str(out_prefix) + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "symbol":           args.symbol,
            "forecast_steps":   args.forecast,
            "lookback":         args.lookback,
            "drift_scale":      args.drift_scale,
            "drift_decay":      args.drift_decay,
            "vol_multiplier":   args.vol_multiplier,
            "recent_vol":       round(rv, 6),
            "vol_scale":        round(vol_scale, 4),
            "start_price":      start_price,
            "actual_end":       float(actual_close[-1]) if len(actual_close) else None,
            "median_end":       float(result.median_path[-1]),
            "rep_end":          float(result.representative_path[-1]),
            "auto_calibrate":   args.auto_calibrate,
            "calib_window":     args.calib_window,
        }, f, indent=2)

    actual_deviation = None
    if len(actual_close) > 0:
        m = min(len(actual_close), len(result.median_path))
        actual_deviation = (actual_close[:m] - result.median_path[:m]) / start_price * 100

    mode_tag = f"auto(w={args.calib_window})" if args.auto_calibrate else "manual"
    hit_str = ""
    if metrics:
        hit_str = (
            f"  |  hit25-75={metrics['hit_rate_25_75']:.0%}"
            f"  hit10-90={metrics['hit_rate_10_90']:.0%}"
            f"  dir_acc={metrics['direction_acc']:.0%}"
            f"  MAE={metrics['mae_pct']:.2f}%"
            f"  end_err={metrics['end_error_pct']:.2f}%"
        )
    title = (
        f"{args.symbol} | v8 {mode_tag} | ds={args.drift_scale}"
        f"  mb={args.momentum_boost}  dd={args.drift_decay}\n"
        f"rv={rv:.4f}  vol_x={args.vol_multiplier}"
        f"  intra={args.intra_bar}  clamp={args.shadow_clamp}" + hit_str
    )

    fwd_volume_norm = result.ohlcv_volume / (result.ohlcv_volume.mean() + 1e-8)
    chart_path = Path(str(out_prefix) + "_candles.png")
    fig = render_forecast_candles(
        hist_open=hist_open, hist_high=hist_high,
        hist_low=hist_low,   hist_close=hist_close,
        hist_volume=hist_volume_norm,
        fwd_open=result.ohlcv_open,   fwd_high=clamped_high,
        fwd_low=clamped_low,          fwd_close=result.ohlcv_close,
        fwd_volume=fwd_volume_norm,
        p25=result.p25, p75=result.p75,
        p10=result.p10, p90=result.p90,
        actual_open=actual_open   if len(actual_close) > 0 else None,
        actual_high=actual_high   if len(actual_close) > 0 else None,
        actual_low=actual_low     if len(actual_close) > 0 else None,
        actual_close=actual_close if len(actual_close) > 0 else None,
        actual_volume=actual_volume_norm if len(actual_close) > 0 else None,
        title=title,
        output_path=chart_path,
        volume_nodes=base_params.volume_nodes,
        hist_window=args.hist_window,
        actual_deviation=actual_deviation,
        metrics=metrics,
    )
    plt.close(fig)

    print(f"\n\u2714 Forward study v8 完成")
    print(f"  K 棒圖 : {chart_path}")
    print(f"  指標   : {metrics_path}")


if __name__ == "__main__":
    main()
