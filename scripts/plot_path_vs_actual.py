"""
plot_path_vs_actual.py  v4 — Calibrated Conditional Simulation

新增 --calibrate 旗標：
  啟用後會先校準 theta（使模擬帶能框住歷史 K 棒），
  再用相同 theta 往後延伸模擬。

不啟用（預設）：沿用 v3 的條件式模擬（估算參數 + 動量偏向）。

Example（校準版）：
    python scripts/plot_path_vs_actual.py \\
        --symbol AAPL --period 3y --interval 1d \\
        --lookback 120 --forecast 30 --paths 300 \\
        --n-samples 4 --seed 42 --calibrate \\
        --lw-vol 1.0 --lw-acf 0.8 --lw-dd 0.6 \\
        --lw-node 0.5 --lw-skew 0.4 \\
        --output results_calibrated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from us_equity_simulator import USStockFutureSimulator
from calibrated_simulator import CalibratedTheta, CalibratedForwardSimulator
from param_calibrator import ParamCalibrator, LossWeights

DARK = "#0e0e0e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    required=True)
    p.add_argument("--period",    default="3y")
    p.add_argument("--interval",  default="1d")
    p.add_argument("--lookback",  type=int, default=120)
    p.add_argument("--forecast",  type=int, default=30)
    p.add_argument("--paths",     type=int, default=300)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--vol-scale", type=float, default=1.5)
    p.add_argument("--output",    default="results_compare")
    # 校準模式
    p.add_argument("--calibrate", action="store_true",
                   help="啟用參數校準（較慢，但前半帶子能框住真實 K 棒）")
    p.add_argument("--calib-paths",  type=int, default=150)
    p.add_argument("--calib-maxiter",type=int, default=300)
    p.add_argument("--lw-vol",   type=float, default=1.0)
    p.add_argument("--lw-acf",   type=float, default=0.8)
    p.add_argument("--lw-dd",    type=float, default=0.6)
    p.add_argument("--lw-node",  type=float, default=0.5)
    p.add_argument("--lw-skew",  type=float, default=0.4)
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def draw_candles(ax, df_slice, x_offset=0, alpha=1.0,
                up_col="#26a69a", down_col="#ef5350"):
    for i, row in df_slice.reset_index(drop=True).iterrows():
        x       = i + x_offset
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color   = up_col if c >= o else down_col
        ax.plot([x, x], [l, h], color=color, lw=0.8, alpha=alpha)
        rect_y  = min(o, c)
        rect_h  = max(abs(c - o), (h - l) * 0.01)
        bar = mpatches.FancyBboxPatch(
            (x - 0.35, rect_y), 0.7, rect_h,
            boxstyle="square,pad=0",
            linewidth=0, facecolor=color, alpha=alpha,
        )
        ax.add_patch(bar)


def plot_one_window(
    ax_main, ax_vol,
    hist_df, actual_df,
    sim_hist_paths, sim_fwd_paths,
    sim_hist_result, sim_fwd_result,
    params, title: str,
    show_paths: int = 80,
    calibrated: bool = False,
):
    n_hist     = len(hist_df)
    n_forecast = sim_fwd_paths.shape[1]
    n_calib    = sim_hist_paths.shape[1] if sim_hist_paths is not None else 0

    x_hist   = np.arange(n_hist)
    x_future = np.arange(n_hist, n_hist + n_forecast)

    ax_main.axvline(n_hist - 0.5, color="#555", lw=1.2, linestyle="--", alpha=0.7)

    # === 左半：歷史帶子（校準版才顯示） ===
    if calibrated and sim_hist_paths is not None:
        for pi in range(min(show_paths, sim_hist_paths.shape[0])):
            ax_main.plot(x_hist, sim_hist_paths[pi],
                         color="#ff9800", alpha=0.03, lw=0.7)
        ax_main.fill_between(
            x_hist, sim_hist_result.p25, sim_hist_result.p75,
            color="#ff9800", alpha=0.12, label="25-75% (calib)"
        )
        ax_main.fill_between(
            x_hist, sim_hist_result.p10, sim_hist_result.p90,
            color="#ff9800", alpha=0.05, label="10-90% (calib)"
        )
        ax_main.plot(x_hist, sim_hist_result.median_path,
                     color="#ffd54f", lw=1.2, alpha=0.7, label="Median calib")

    # 歷史 K 棒（在帶子上面）
    draw_candles(ax_main, hist_df, alpha=0.9)

    # Volume Profile 節點
    for node, strength in zip(params.volume_nodes, params.volume_node_strength):
        ax_main.axhline(
            node, color="orange",
            lw=0.6 + 1.0 * strength,
            alpha=0.25 + 0.3 * strength, linestyle=":",
        )

    # === 右半：未來模擬帶子 ===
    for pi in range(min(show_paths, sim_fwd_paths.shape[0])):
        ax_main.plot(x_future, sim_fwd_paths[pi],
                     color="cyan", alpha=0.04, lw=0.7)
    ax_main.fill_between(x_future, sim_fwd_result.p25, sim_fwd_result.p75,
                         color="#00e5ff", alpha=0.15, label="25-75% (fwd)")
    ax_main.fill_between(x_future, sim_fwd_result.p10, sim_fwd_result.p90,
                         color="#00e5ff", alpha=0.07, label="10-90% (fwd)")
    ax_main.plot(x_future, sim_fwd_result.median_path,
                 color="yellow", lw=1.8, label="Median fwd", zorder=5)

    # 真實未來 K 棒
    draw_candles(ax_main, actual_df, x_offset=n_hist, alpha=1.0)

    # 動量標注
    mo     = params.momentum_bias
    bo     = params.node_breakout_state
    mo_str = f"mom={'+'  if mo>=0 else ''}{mo:.5f}"
    bo_str = {1: " | ↑node突破", -1: " | ↓node被拒", 0: ""}.get(bo, "")
    ax_main.text(
        n_hist + 1,
        ax_main.get_ylim()[1] * 0.99,
        mo_str + bo_str,
        color="#aaffaa", fontsize=7, va="top",
    )

    if calibrated:
        ax_main.text(
            2, ax_main.get_ylim()[1] * 0.99,
            "[校準帶子：橘色=以校準 theta 重現歷史；青色=延伸預測]",
            color="#ffcc80", fontsize=7, va="top",
        )

    # 成交量
    if "Volume" in hist_df.columns:
        vols   = hist_df["Volume"].values
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for o, c in zip(hist_df["Open"].values, hist_df["Close"].values)]
        ax_vol.bar(np.arange(n_hist), vols, color=colors, alpha=0.5, width=0.8)
    ax_vol.set_xlim(ax_main.get_xlim())
    ax_vol.set_facecolor(DARK)
    ax_vol.tick_params(colors="#888", labelsize=7)
    ax_vol.set_ylabel("Volume", color="#888", fontsize=7)
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x:.0f}")
    )
    ax_main.set_facecolor(DARK)
    ax_main.tick_params(colors="#888", labelsize=8)
    ax_main.set_title(title, color="white", fontsize=8.5, pad=5)
    ax_main.legend(loc="upper left", fontsize=6.5,
                   facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")
    ax_main.set_xlim(-1, n_hist + n_forecast + 1)
    ax_main.text(
        n_hist - 0.5, ax_main.get_ylim()[0],
        " Forecast →", color="#888", fontsize=7, va="bottom",
    )


def main():
    args = parse_args()
    out  = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.symbol}...")
    df_raw = yf.download(
        args.symbol, period=args.period, interval=args.interval,
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    total = len(df)
    print(f"Total bars: {total}  |  calibrate={args.calibrate}")

    ESTIMATOR_LB = 500
    min_needed   = ESTIMATOR_LB + args.forecast
    if total < min_needed:
        raise ValueError(f"Need {min_needed} bars, got {total}")

    estimator = MarketParameterEstimator(
        lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10
    )
    weights = LossWeights(
        vol=args.lw_vol, acf=args.lw_acf, drawdown=args.lw_dd,
        node=args.lw_node, skew=args.lw_skew,
    )
    rng = np.random.default_rng(args.seed)

    max_start    = total - min_needed
    sample_starts = [
        int(max_start * i / max(args.n_samples - 1, 1))
        for i in range(args.n_samples)
    ] if args.n_samples > 1 else [max_start]
    sample_starts = [min(s, max_start) for s in sample_starts]

    plt.style.use("dark_background")

    for fig_idx, train_start in enumerate(sample_starts):
        train_end = train_start + ESTIMATOR_LB
        test_end  = train_end + args.forecast

        train_df_full = df.iloc[train_start:train_end]
        actual_df     = df.iloc[train_end:test_end]
        display_start = max(train_start, train_end - args.lookback)
        hist_df       = df.iloc[display_start:train_end]

        try:
            params = estimator.fit(train_df_full, symbol=args.symbol)
        except Exception as e:
            print(f"  window {fig_idx}: estimator failed: {e}")
            continue

        sim_seed = int(rng.integers(0, 2**31))

        # ---- 校準 ----
        if args.calibrate:
            print(f"  window {fig_idx+1}: calibrating...")
            calibrator = ParamCalibrator(
                base_params=params,
                history_close=hist_df["Close"].values,
                weights=weights,
                n_sim_paths=args.calib_paths,
                n_sim_steps=args.lookback,
                seed=sim_seed,
                maxiter=args.calib_maxiter,
                verbose=True,
            )
            theta = calibrator.calibrate()

            # 左半：用 theta 重現歷史帶子
            hist_sim = CalibratedForwardSimulator(
                theta=theta, base_params=params,
                forecast_steps=args.lookback,
                n_paths=args.paths, seed=sim_seed,
            )
            hist_result = hist_sim.simulate()

            # 右半：用 theta + 動量偏向往後預測
            import dataclasses
            params_fwd = dataclasses.replace(
                params,
                realized_vol=theta.vol, ewma_vol=theta.vol,
                drift=theta.drift, ewma_drift=theta.drift,
                hurst_proxy=theta.hurst_proxy,
            )
            fwd_sim = CalibratedForwardSimulator(
                theta=theta, base_params=params_fwd,
                forecast_steps=args.forecast,
                n_paths=args.paths, seed=sim_seed,
            )
            fwd_result = fwd_sim.simulate()
            calibrated = True

        else:
            hist_result = None
            sim = USStockFutureSimulator(
                params=params,
                forecast_steps=args.forecast,
                n_paths=args.paths,
                seed=sim_seed,
                vol_scale=args.vol_scale,
            )
            fwd_result  = sim.simulate()
            calibrated  = False

        date_str = (
            str(df["index"].iloc[train_end - 1])[:10]
            if "index" in df.columns else f"bar{train_end}"
        )
        bo_label = {1: "↑node突破", -1: "↓node被拒", 0: "node中立"}.get(
            params.node_breakout_state, ""
        )
        mode_tag = "[CALIBRATED]" if calibrated else "[v3-conditional]"

        fig, axes = plt.subplots(
            2, 1, figsize=(20, 9),
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04},
        )
        fig.patch.set_facecolor(DARK)

        plot_one_window(
            ax_main=axes[0], ax_vol=axes[1],
            hist_df=hist_df, actual_df=actual_df,
            sim_hist_paths=hist_result.future_paths if calibrated else None,
            sim_fwd_paths=fwd_result.future_paths,
            sim_hist_result=hist_result,
            sim_fwd_result=fwd_result,
            params=params,
            title=(
                f"{args.symbol}  {mode_tag}  |  w{fig_idx+1}/{args.n_samples}  |  {date_str}  |  "
                f"Hurst={params.hurst_proxy:.3f}  SMR={params.smart_money_ratio:.2f}  "
                f"vol={params.ewma_vol:.4f}(x{params.vol_trend:.2f})  "
                f"drift={params.ewma_drift:+.5f}  mom={params.momentum_bias:+.5f}  {bo_label}"
            ),
            calibrated=calibrated,
        )

        fname = out / f"compare_{args.symbol}_w{fig_idx+1:02d}_{date_str}.png"
        fig.savefig(fname, dpi=160, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        print(f"  Saved: {fname.name}")

    print(f"\nDone. {args.n_samples} charts saved to {out}/")


if __name__ == "__main__":
    main()
