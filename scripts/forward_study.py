"""
forward_study.py  v7

修復（v7）—套用 v6 自動建議：
  1. intra_bar 預設 2（減少影線）
  2. shadow_noise 預設 0.15
  3. drift_scale 預設 1.05
  4. momentum_boost 預設 1.5
  5. anchor_weight 預設 0.4
  6. 新增 --shadow-clamp：限制影線最大倍率（限制 high/low 跟實體的比例）
  7. 新增 shadow ratio 直方圖診斷（分布輸出）

Example:
    python scripts/forward_study.py \\
        --symbol AAPL \\
        --theta results/theta_aapl.json \\
        --lookback 120 --forecast 30 \\
        --seed 42 --n-paths 500 \\
        --backbone-mr 0.06 --n-seg 6 \\
        --hist-window 60 --intra-bar 2 \\
        --drift-decay 0.05 --drift-scale 1.05 --anchor-weight 0.4 \\
        --vol-multiplier 1.5 --recent-vol-window 20 \\
        --shadow-noise 0.15 --shadow-clamp 2.0 \\
        --momentum-boost 1.5 --path-spread 1.0 \\
        --output results/forward_aapl
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
    p.add_argument("--intra-bar",         type=int,   default=2,    help="預設 2: 最少影線")
    p.add_argument("--drift-decay",       type=float, default=0.05)
    p.add_argument("--drift-scale",       type=float, default=1.05, help="預設 1.05: v6 建議")
    p.add_argument("--anchor-weight",     type=float, default=0.4,  help="預設 0.4: v6 建議")
    p.add_argument("--vol-multiplier",    type=float, default=1.5)
    p.add_argument("--recent-vol-window", type=int,   default=20)
    p.add_argument("--vol-scale-min",     type=float, default=0.6)
    p.add_argument("--vol-scale-max",     type=float, default=4.0)
    p.add_argument("--shadow-noise",      type=float, default=0.15, help="預設 0.15: v6 建議")
    p.add_argument("--shadow-clamp",      type=float, default=2.0,
                   help="影線最大倍率，限制 (high-low)/body <= clamp (0=不限制)")
    p.add_argument("--momentum-boost",   type=float, default=1.5,  help="預設 1.5: v6 建議")
    p.add_argument("--path-spread",      type=float, default=1.0)
    p.add_argument("--output",            default="forward_study")
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


def clamp_shadows(ohlcv_open, ohlcv_high, ohlcv_low, ohlcv_close, shadow_clamp: float):
    """強制影線不超過實體的 shadow_clamp 倍"""
    if shadow_clamp <= 0:
        return ohlcv_high.copy(), ohlcv_low.copy()
    body_top    = np.maximum(ohlcv_open, ohlcv_close)
    body_bot    = np.minimum(ohlcv_open, ohlcv_close)
    body_size   = body_top - body_bot
    max_shadow  = body_size * shadow_clamp
    new_high    = np.minimum(ohlcv_high, body_top + max_shadow)
    new_low     = np.maximum(ohlcv_low,  body_bot - max_shadow)
    # 確保 high >= max(open,close) 且 low <= min(open,close)
    new_high    = np.maximum(new_high, body_top)
    new_low     = np.minimum(new_low,  body_bot)
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
    result, clamped_high, clamped_low, start_price, metrics
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

    # K 棒形態（使用 clamped high/low）
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

    # 比對歷史 K 棒
    hist_close_arr = close_hist
    hist_open_arr  = np.array([])
    # 只用收益率過似估算體積大小
    hist_ret_std   = float(np.std(np.diff(np.log(hist_close_arr)))) * 100

    print()
    print("=" * 60)
    print("  VERBOSE DIAGNOSTICS (v7)")
    print("=" * 60)

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
    status_shad = " \u2705" if avg_shadow_ratio <= 2.0 else f" \u274c 過大(clamp={args.shadow_clamp})"
    print(f"  avg_body_pct       : {avg_body_pct:.3f}%  (目標: 0.3~1.5%){status_body}")
    print(f"  avg_shadow_ratio   : {avg_shadow_ratio:.2f}   (目標: 0.5~2.0){status_shad}")
    print(f"  p50_shadow_ratio   : {p50_shadow_ratio:.2f}   (中位數, 較少受極端影線影響)")
    print(f"  p90_shadow_ratio   : {p90_shadow_ratio:.2f}   (極端 K 棒影線狀況)")
    print(f"  shadow_clamp       : {args.shadow_clamp}x  (影線/實體上限)")
    print(f"  hist_ret_std       : {hist_ret_std:.3f}%/day  (歷史收益率標準差參考)")
    print(f"  direction_consist  : {direction_consistency:.2f}   (高=K 棒方向跟 median 一致)")
    if avg_shadow_ratio > 2.5:
        suggest_intra = max(1, args.intra_bar - 1)
        suggest_sn    = round(max(0.05, args.shadow_noise - 0.1), 2)
        suggest_clamp = round(max(0.5,  args.shadow_clamp - 0.5), 1)
        print(f"  \u2757 影線過多：建議 --intra-bar {suggest_intra} / --shadow-noise {suggest_sn} / --shadow-clamp {suggest_clamp}")
    if avg_body_pct > 2.5:
        suggest_ds = round(args.drift_scale * 0.8, 2)
        print(f"  \u2757 實體過大(路徑跳動太劇烈)：建議 --drift-scale {suggest_ds} / --intra-bar 增加")

    print(f"\n[漬動對照]")
    print(f"  theta.drift             : {theta.drift*100:+.4f}%/day")
    print(f"  last_seg_drift (bb)     : {last_drift*100:+.4f}%/day")
    print(f"  hist_drift_20d          : {hist_drift_20*100:+.4f}%/day")
    if hist_drift_60:
        print(f"  hist_drift_60d          : {hist_drift_60*100:+.4f}%/day")
    print(f"  hist_drift_all({args.lookback:3d}d)  : {hist_drift_all*100:+.4f}%/day")
    print(f"  median_daily_drift      : {median_daily_drift*100:+.4f}%/day  (實際模擬)")
    if actual_daily_drift:
        print(f"  actual_daily_drift      : {actual_daily_drift*100:+.4f}%/day  (實際後續)")
        drift_gap = (median_daily_drift - actual_daily_drift) * 100
        print(f"  drift_gap(med-act)      : {drift_gap:+.4f}%/day  (正=median偏快, 負=median偏慢)")

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
            if pct > 0.3: flag = f" \u274c 帶子偶尔偏低({pct:.0%})"
        if k == "bars_below_p10" and isinstance(v, int):
            pct = v / max(metrics.get("n_compared", 30), 1)
            if pct > 0.3: flag = f" \u274c 帶子偶尔偏高({pct:.0%})"
        print(f"  {k:25s}: {v}{flag}")

    print(f"\n[自動建議]")
    suggestions = []
    if metrics.get("hit_rate_10_90", 0) < 0.5:
        suggestions.append("  \u2022 帶子太窄: --vol-multiplier +0.3 或減小 --recent-vol-window")
    if metrics.get("hit_rate_10_90", 0) > 0.95 and metrics.get("hit_rate_25_75", 0) < 0.25:
        suggestions.append("  \u2022 帶子寬但中心線偏: 調整 drift_scale 是首要任務")
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
            f"  \u2022 方向准確度差: --momentum-boost {round(min(args.momentum_boost + 0.3, 2.5), 1)}"
            f" 或 --anchor-weight {round(min(args.anchor_weight + 0.1, 0.7), 1)}"
        )
    if avg_shadow_ratio > 2.5:
        suggestions.append(
            f"  \u2022 影線過多({avg_shadow_ratio:.1f}x): --intra-bar {max(1, args.intra_bar-1)}"
            f" / --shadow-noise {round(max(0.05, args.shadow_noise-0.1), 2)}"
            f" / --shadow-clamp {round(max(0.5, args.shadow_clamp-0.5), 1)}"
        )
    if avg_body_pct > 2.5:
        suggestions.append(f"  \u2022 實體過大({avg_body_pct:.2f}%): --drift-scale {round(args.drift_scale*0.8,2)} / --intra-bar 增加")
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

    # v7: 影線 clamp 後用於畫圖和診斷
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
        }, f, indent=2)

    actual_deviation = None
    if len(actual_close) > 0:
        m = min(len(actual_close), len(result.median_path))
        actual_deviation = (actual_close[:m] - result.median_path[:m]) / start_price * 100

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
        f"{args.symbol} | v7 | intra={args.intra_bar}  ds={args.drift_scale}"
        f"  mb={args.momentum_boost}  clamp={args.shadow_clamp}\n"
        f"rv={rv:.4f}  vol_scale={vol_scale:.2f}  vol_x={args.vol_multiplier}"
        f"  aw={args.anchor_weight}" + hit_str
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

    print(f"\n\u2714 Forward study v7 完成")
    print(f"  K 棒圖 : {chart_path}")
    print(f"  指標   : {metrics_path}")


if __name__ == "__main__":
    main()
