"""
calibrate_params.py  v2

對指定股票的指定時間段，校準模擬器參數 theta。
輸出：最佳 theta（JSON） + 校準效果圖（帶子懲罰渲白線）

Example:
    python scripts/calibrate_params.py \\
        --symbol AAPL --period 3y --interval 1d \\
        --lookback 120 --seed 42 \\
        --lw-vol 1.0 --lw-acf 0.8 --lw-dd 0.6 \\
        --lw-node 0.5 --lw-skew 0.4 \\
        --lw-trend 1.5 --lw-end 1.2 \\
        --output calibrated_theta_aapl.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from param_calibrator import ParamCalibrator, LossWeights
from calibrated_simulator import CalibratedTheta, CalibratedForwardSimulator

DARK = "#0e0e0e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",   required=True)
    p.add_argument("--period",   default="3y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--lookback", type=int, default=120)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--n-paths",  type=int, default=200)
    p.add_argument("--maxiter",  type=int, default=500)
    p.add_argument("--lw-vol",   type=float, default=1.0)
    p.add_argument("--lw-acf",   type=float, default=0.8)
    p.add_argument("--lw-dd",    type=float, default=0.6)
    p.add_argument("--lw-node",  type=float, default=0.5)
    p.add_argument("--lw-skew",  type=float, default=0.4)
    p.add_argument("--lw-trend", type=float, default=1.5,
                   help="區間方向懲罰權重（新增）")
    p.add_argument("--lw-end",   type=float, default=1.2,
                   help="終點價格懲罰權重（新增）")
    p.add_argument("--output",   default="calibrated_theta.json")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def main():
    args = parse_args()

    print(f"Downloading {args.symbol}...")
    df_raw = yf.download(
        args.symbol, period=args.period, interval=args.interval,
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    total = len(df)
    print(f"Total bars: {total}")

    ESTIMATOR_LB = 500
    if total < ESTIMATOR_LB + args.lookback:
        raise ValueError(f"Need {ESTIMATOR_LB + args.lookback} bars, got {total}")

    # 最後 lookback 根 = 校準段
    calib_df    = df.iloc[-args.lookback:]
    estimate_df = df.iloc[-ESTIMATOR_LB:]

    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)

    weights = LossWeights(
        vol=args.lw_vol, acf=args.lw_acf, drawdown=args.lw_dd,
        node=args.lw_node, skew=args.lw_skew,
        trend=args.lw_trend, end=args.lw_end,
    )

    print(f"\nCalibrating on last {args.lookback} bars (start_price={calib_df['Close'].iloc[0]:.2f})...")
    calibrator = ParamCalibrator(
        base_params=base_params,
        history_close=calib_df["Close"].values,
        weights=weights,
        n_sim_paths=args.n_paths,
        seed=args.seed,
        maxiter=args.maxiter,
        verbose=True,
    )
    theta = calibrator.calibrate()

    # 儲存 JSON
    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(theta.to_dict(), f, indent=2)
    print(f"\nTheta saved: {out_json}")
    print(json.dumps(theta.to_dict(), indent=2))

    # ---------- 畫校準結果圖 ----------
    import dataclasses
    print("\nRendering calibration check chart...")
    close_vals  = calib_df["Close"].values
    n           = len(close_vals)
    x_hist      = np.arange(n)
    start_price = float(close_vals[0])

    # 模擬從同一起點出發
    params_vis = dataclasses.replace(
        base_params,
        last_close=start_price,
        momentum_bias=0.0,
        node_breakout_state=0,
    )
    fwd_sim = CalibratedForwardSimulator(
        theta=theta,
        base_params=params_vis,
        forecast_steps=n,
        n_paths=args.n_paths,
        seed=args.seed,
    )
    sim_result = fwd_sim.simulate()
    paths      = sim_result.future_paths

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    for pi in range(min(100, paths.shape[0])):
        ax.plot(x_hist, paths[pi], color="cyan", alpha=0.04, lw=0.7)
    ax.fill_between(x_hist, sim_result.p25, sim_result.p75,
                    color="#00e5ff", alpha=0.15, label="25-75%")
    ax.fill_between(x_hist, sim_result.p10, sim_result.p90,
                    color="#00e5ff", alpha=0.07, label="10-90%")
    ax.plot(x_hist, sim_result.median_path,
            color="yellow", lw=2, label="Median sim")
    ax.plot(x_hist, close_vals,
            color="white", lw=1.6, label="Real close", alpha=0.9)

    # 分段走勢標註
    segs = calibrator._segment_trend(close_vals)
    seg_n = n // 3
    for si, (d, x0) in enumerate(zip(segs, [0, seg_n, seg_n * 2])):
        color = "#66ff66" if d > 0 else "#ff6666"
        ax.annotate(
            f"{'+' if d>0 else ''}{d:.1f}",
            xy=(x0 + seg_n // 2, close_vals[x0 + seg_n // 2]),
            color=color, fontsize=8, ha="center",
            xytext=(0, 14), textcoords="offset points",
        )

    for node in base_params.volume_nodes:
        ax.axhline(node, color="orange", lw=0.8, alpha=0.35, linestyle=":")

    ax.set_title(
        f"{args.symbol} | Calibration Check v2 | lookback={args.lookback}  "
        f"start={start_price:.2f}\n"
        f"vol={theta.vol:.4f}  drift={theta.drift:+.5f}  hurst={theta.hurst_proxy:.3f}  "
        f"mr={theta.mr_coeff:.3f}  node={theta.node_coeff:.3f}  "
        f"mom_str={theta.momentum_strength:.2f}",
        color="white", fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
              edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="#888")

    img_path = out_json.with_suffix(".png")
    fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Chart saved: {img_path}")
    print("\n✔ 校準完成。確認帶子有沒有包住白線，再跟进開始對比圖。")


if __name__ == "__main__":
    main()
