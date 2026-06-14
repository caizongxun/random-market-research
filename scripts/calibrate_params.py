"""
calibrate_params.py  v3

兩階段校準流程：
  Phase 1 - BackboneFitter：確定性分段漂移擬合，得到骨幹路徑
  Phase 2 - ParamCalibrator：以骨幹為 target，校準隨機演化參數

輸出：
  - <output>.json  最佳 theta
  - <output>_backbone.png  Phase 1 骨幹擬合圖
  - <output>_calibration.png  Phase 2 帶子校準圖（骨幹 + 帶子 + 白線）

Example:
    python scripts/calibrate_params.py \\
        --symbol AAPL --period 3y --interval 1d \\
        --lookback 120 --seed 42 \\
        --n-seg 6 --smooth-reg 0.5 \\
        --lw-vol 1.0 --lw-acf 0.8 --lw-dd 0.6 \\
        --lw-node 0.5 --lw-skew 0.4 \\
        --lw-trend 2.0 --lw-end 1.5 \\
        --maxiter 600 \\
        --output calibrated_theta_aapl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from param_calibrator import ParamCalibrator, LossWeights
from backbone_fitter import BackboneFitter, plot_backbone
from calibrated_simulator import CalibratedTheta, CalibratedForwardSimulator

DARK = "#0e0e0e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",     required=True)
    p.add_argument("--period",     default="3y")
    p.add_argument("--interval",   default="1d")
    p.add_argument("--lookback",   type=int,   default=120)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-paths",    type=int,   default=200)
    p.add_argument("--maxiter",    type=int,   default=600)
    # Phase 1 骨幹參數
    p.add_argument("--n-seg",      type=int,   default=6,
                   help="骨幹分段數（3-8，越多越貼合真實但可能過擬合）")
    p.add_argument("--smooth-reg", type=float, default=0.5,
                   help="相鄰段漂移平滑懲罰")
    # Loss 權重
    p.add_argument("--lw-vol",     type=float, default=1.0)
    p.add_argument("--lw-acf",     type=float, default=0.8)
    p.add_argument("--lw-dd",      type=float, default=0.6)
    p.add_argument("--lw-node",    type=float, default=0.5)
    p.add_argument("--lw-skew",    type=float, default=0.4)
    p.add_argument("--lw-trend",   type=float, default=2.0)
    p.add_argument("--lw-end",     type=float, default=1.5)
    p.add_argument("--output",     default="calibrated_theta.json")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def main():
    args = parse_args()

    print(f"[1/4] Downloading {args.symbol}...")
    df_raw = yf.download(
        args.symbol, period=args.period, interval=args.interval,
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    print(f"      Total bars: {len(df)}")

    ESTIMATOR_LB = 500
    if len(df) < ESTIMATOR_LB + args.lookback:
        raise ValueError(f"Need {ESTIMATOR_LB + args.lookback} bars, got {len(df)}")

    calib_df    = df.iloc[-args.lookback:]
    estimate_df = df.iloc[-ESTIMATOR_LB:]
    close_vals  = calib_df["Close"].values
    start_price = float(close_vals[0])

    # ── Phase 1：骨幹擬合 ──────────────────────────────────────────────
    print(f"[2/4] Phase 1 - Backbone fitting (n_seg={args.n_seg})...")
    fitter      = BackboneFitter(n_seg=args.n_seg, smooth_reg=args.smooth_reg)
    bb_result   = fitter.fit(close_vals)
    print(f"      Backbone MSE={bb_result.fit_mse:.6f}")
    print(f"      Segment drifts: {[f'{d*100:+.3f}%' for d in bb_result.segment_drifts]}")

    # 儲存骨幹圖
    out_path      = Path(args.output)
    bb_img_path   = out_path.with_name(out_path.stem + "_backbone.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig_bb, ax_bb = plt.subplots(figsize=(14, 5))
    fig_bb.patch.set_facecolor(DARK)
    plot_backbone(
        close_vals, bb_result,
        title=f"{args.symbol} | Phase 1 Backbone | lookback={args.lookback}  n_seg={args.n_seg}",
        ax=ax_bb, show=False,
    )
    fig_bb.savefig(bb_img_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig_bb)
    print(f"      Backbone chart: {bb_img_path}")

    # ── Phase 2：校準隨機演化 ──────────────────────────────────────────
    print(f"[3/4] Phase 2 - Calibrating stochastic params (maxiter={args.maxiter})...")
    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)

    weights = LossWeights(
        vol=args.lw_vol, acf=args.lw_acf, drawdown=args.lw_dd,
        node=args.lw_node, skew=args.lw_skew,
        trend=args.lw_trend, end=args.lw_end,
    )
    calibrator = ParamCalibrator(
        base_params=base_params,
        history_close=close_vals,
        weights=weights,
        n_sim_paths=args.n_paths,
        seed=args.seed,
        maxiter=args.maxiter,
        verbose=True,
        backbone_path=bb_result.backbone,  # ★ 骨幹當 target
    )
    theta = calibrator.calibrate()

    with open(out_path, "w") as f:
        json.dump(theta.to_dict(), f, indent=2)
    print(f"      Theta saved: {out_path}")
    print(json.dumps(theta.to_dict(), indent=2))

    # ── Phase 2 驗證圖 ────────────────────────────────────────────────
    print("[4/4] Rendering calibration check chart...")
    import dataclasses

    params_vis = dataclasses.replace(
        base_params,
        last_close=start_price,
        momentum_bias=0.0,
        node_breakout_state=0,
    )
    fwd_sim    = CalibratedForwardSimulator(
        theta=theta, base_params=params_vis,
        forecast_steps=len(close_vals),
        n_paths=args.n_paths, seed=args.seed,
    )
    sim_result = fwd_sim.simulate()
    paths      = sim_result.future_paths
    x_hist     = np.arange(len(close_vals))

    fig_cal, ax = plt.subplots(figsize=(16, 6))
    fig_cal.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    for pi in range(min(80, paths.shape[0])):
        ax.plot(x_hist, paths[pi], color="cyan", alpha=0.03, lw=0.7)
    ax.fill_between(x_hist, sim_result.p25, sim_result.p75,
                    color="#00e5ff", alpha=0.15, label="25-75%")
    ax.fill_between(x_hist, sim_result.p10, sim_result.p90,
                    color="#00e5ff", alpha=0.07, label="10-90%")
    ax.plot(x_hist, sim_result.median_path,
            color="yellow", lw=2, label="Median sim")
    # ★ 骨幹路徑（橘色）
    ax.plot(x_hist, bb_result.backbone,
            color="#ff9900", lw=2, linestyle="--", label="Backbone", alpha=0.85)
    ax.plot(x_hist, close_vals,
            color="white", lw=1.6, label="Real close", alpha=0.9)

    # 骨幹分段邊界
    seg_len = len(close_vals) // args.n_seg
    for s in range(1, args.n_seg):
        ax.axvline(s * seg_len, color="#555", lw=1, linestyle=":", alpha=0.7)

    for node in base_params.volume_nodes:
        ax.axhline(node, color="orange", lw=0.8, alpha=0.3, linestyle=":")

    ax.set_title(
        f"{args.symbol} | Calibration v3 (2-phase) | lookback={args.lookback}  "
        f"start={start_price:.2f}\n"
        f"vol={theta.vol:.4f}  drift={theta.drift:+.5f}  hurst={theta.hurst_proxy:.3f}  "
        f"mr={theta.mr_coeff:.3f}  node={theta.node_coeff:.3f}  "
        f"mom_str={theta.momentum_strength:.2f}  backbone_seg={args.n_seg}",
        color="white", fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
              edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="#888")

    cal_img_path = out_path.with_name(out_path.stem + "_calibration.png")
    fig_cal.savefig(cal_img_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig_cal)
    print(f"      Calibration chart: {cal_img_path}")
    print("\n✔ 兩階段校準完成。")
    print(f"  骨幹圖：{bb_img_path}")
    print(f"  帶子圖：{cal_img_path}")
    print(f"  Theta：{out_path}")


if __name__ == "__main__":
    main()
