"""
forward_study.py  v3

修復（v3）：
  1. 新增 --drift-decay   (預設 0.05)  → drift 隨時間指數衰減
  2. 新增 --drift-scale   (預設 0.5)   → 校準 drift 整體縮放（對抗 lookback 偏差）
  3. 新增 --anchor-weight (預設 0.3)   → momentum anchor 強度
  drift-scale 直覺：
    - 1.0 = 完全信任校準期的 drift（舊行為）
    - 0.5 = drift 減半（建議起點，lookback 包含大漲時用）
    - 0.0 = 純隨機漫步（不加 drift）

Example:
    python scripts/forward_study.py \\
        --symbol AAPL \\
        --theta results/theta_aapl.json \\
        --lookback 120 --forecast 30 \\
        --seed 42 --n-paths 500 \\
        --backbone-mr 0.06 --n-seg 6 \\
        --hist-window 60 \\
        --drift-decay 0.05 \\
        --drift-scale 0.5 \\
        --anchor-weight 0.3 \\
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

DARK = "#0e0e0e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",        required=True)
    p.add_argument("--theta",         required=True)
    p.add_argument("--lookback",      type=int,   default=120)
    p.add_argument("--forecast",      type=int,   default=30)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n-paths",       type=int,   default=500)
    p.add_argument("--n-seg",         type=int,   default=6)
    p.add_argument("--smooth-reg",    type=float, default=0.5)
    p.add_argument("--backbone-mr",   type=float, default=0.06)
    p.add_argument("--period",        default="3y")
    p.add_argument("--interval",      default="1d")
    p.add_argument("--hist-window",   type=int,   default=60)
    p.add_argument("--intra-bar",     type=int,   default=8)
    # v3 新參數
    p.add_argument("--drift-decay",   type=float, default=0.05,
                   help="Drift 衰減速率（0=不衰減，0.1=快速衰減）")
    p.add_argument("--drift-scale",   type=float, default=0.5,
                   help="Drift 整體縮放（1=原始，0.5=減半，0=純隨機漫步）")
    p.add_argument("--anchor-weight", type=float, default=0.3,
                   help="Momentum anchor 強度（0~1）")
    p.add_argument("--output",        default="forward_study")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def build_schedules(segment_drifts, segment_vols, n_steps):
    n_seg   = len(segment_drifts)
    seg_len = n_steps // n_seg
    d_arr   = np.empty(n_steps)
    v_arr   = np.empty(n_steps)
    for s in range(n_seg):
        lo = s * seg_len
        hi = lo + seg_len if s < n_seg - 1 else n_steps
        d_arr[lo:hi] = segment_drifts[s]
        v_arr[lo:hi] = segment_vols[s]
    return d_arr, v_arr


def compute_metrics(actual, median, p25, p75, p10, p90, start_price):
    n = min(len(actual), len(median))
    if n == 0:
        return {}
    act, med  = actual[:n], median[:n]
    hit_25_75 = float(np.mean((act >= p25[:n]) & (act <= p75[:n])))
    hit_10_90 = float(np.mean((act >= p10[:n]) & (act <= p90[:n])))
    actual_dir   = np.sign(np.diff(np.concatenate([[start_price], act])))
    median_dir   = np.sign(np.diff(np.concatenate([[start_price], med])))
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


def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))
    print(f"Loaded theta: vol={theta.vol:.5f}  drift={theta.drift:+.6f}  hurst={theta.hurst_proxy:.3f}")
    print(f"v3 params:  drift_decay={args.drift_decay}  drift_scale={args.drift_scale}  anchor_weight={args.anchor_weight}")

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

    close_hist   = train_df["Close"].values
    start_price  = float(close_hist[-1])

    hist_open    = train_df["Open"].values.astype(float)
    hist_high    = train_df["High"].values.astype(float)
    hist_low     = train_df["Low"].values.astype(float)
    hist_close   = close_hist
    hist_volume  = train_df["Volume"].values.astype(float)
    hist_volume_norm = hist_volume / (hist_volume.mean() + 1e-8)

    actual_close  = future_df["Close"].values.astype(float)
    actual_open   = future_df["Open"].values.astype(float)
    actual_high   = future_df["High"].values.astype(float)
    actual_low    = future_df["Low"].values.astype(float)
    actual_volume = future_df["Volume"].values.astype(float)
    actual_volume_norm = actual_volume / (hist_volume.mean() + 1e-8)

    print(f"Train end: {start_price:.2f}  forecast={args.forecast} bars")

    fitter    = BackboneFitter(n_seg=args.n_seg, smooth_reg=args.smooth_reg)
    bb_result = fitter.fit(close_hist)
    print(f"Backbone MSE={bb_result.fit_mse:.6f}")

    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol)
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

    vol_scale = float(np.clip(theta.vol / max(last_vol, 1e-8), 0.5, 3.0))
    print(f"vol_scale={vol_scale:.3f}")

    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = dataclasses.replace(base_params, last_close=start_price,
                                      momentum_bias=0.0, node_breakout_state=0)
    params_fwd  = build_params_from_theta(theta, params_fwd)

    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=args.forecast,
        n_paths=args.n_paths,
        seed=args.seed,
        vol_scale=vol_scale,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_fwd,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=args.backbone_mr,
        intra_bar_steps=args.intra_bar,
        # v3
        drift_decay_rate=args.drift_decay,
        drift_scale=args.drift_scale,
        momentum_anchor_weight=args.anchor_weight,
    )
    result = sim.simulate()

    metrics = compute_metrics(
        actual=actual_close, median=result.median_path,
        p25=result.p25, p75=result.p75,
        p10=result.p10, p90=result.p90,
        start_price=start_price,
    )
    print("\n=== Forward Study Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    out_prefix   = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(str(out_prefix) + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "symbol":        args.symbol,
            "forecast_steps": args.forecast,
            "lookback":      args.lookback,
            "drift_scale":   args.drift_scale,
            "drift_decay":   args.drift_decay,
            "start_price":   start_price,
            "actual_end":    float(actual_close[-1]) if len(actual_close) else None,
            "median_end":    float(result.median_path[-1]),
            "rep_end":       float(result.representative_path[-1]),
        }, f, indent=2)

    actual_deviation = None
    if len(actual_close) > 0:
        m = min(len(actual_close), len(result.median_path))
        actual_deviation = (actual_close[:m] - result.median_path[:m]) / start_price * 100

    hit_str = ""
    if metrics:
        hit_str = (
            f"  |  hit25-75={metrics['hit_rate_25_75']:.0%}  "
            f"hit10-90={metrics['hit_rate_10_90']:.0%}  "
            f"dir_acc={metrics['direction_acc']:.0%}  "
            f"MAE={metrics['mae_pct']:.2f}%  "
            f"end_err={metrics['end_error_pct']:.2f}%"
        )
    title = (
        f"{args.symbol} | Forecast Candles (intra-bar={args.intra_bar}) | "
        f"lookback={args.lookback}  forecast={args.forecast}  start={start_price:.2f}\n"
        f"vol={theta.vol:.4f}  hurst={theta.hurst_proxy:.3f}  bb_mr={args.backbone_mr:.3f}  "
        f"vol_scale={vol_scale:.2f}  drift_scale={args.drift_scale}  decay={args.drift_decay}" + hit_str
    )

    fwd_volume_norm = result.ohlcv_volume / (result.ohlcv_volume.mean() + 1e-8)

    chart_path = Path(str(out_prefix) + "_candles.png")
    fig = render_forecast_candles(
        hist_open=hist_open,
        hist_high=hist_high,
        hist_low=hist_low,
        hist_close=hist_close,
        hist_volume=hist_volume_norm,
        fwd_open=result.ohlcv_open,
        fwd_high=result.ohlcv_high,
        fwd_low=result.ohlcv_low,
        fwd_close=result.ohlcv_close,
        fwd_volume=fwd_volume_norm,
        p25=result.p25,
        p75=result.p75,
        p10=result.p10,
        p90=result.p90,
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

    print(f"\n✔ Forward study v3 完成")
    print(f"  K 棒圖 : {chart_path}")
    print(f"  指標   : {metrics_path}")


if __name__ == "__main__":
    main()
