"""
walk_forward_validation.py

Walk-forward 驗證：
  1. 從歷史資料中滾動取窗口（每次用 lookback 根估參數）
  2. 模擬接下來 forecast_steps 根的機率分佈
  3. 和真實未來 N 根對比

輸出：
  - walk_forward_summary.csv        每個窗口的量化指標
  - coverage_plot.png               信賴帶覆蓋率隨時間變化
  - path_comparison_sample.png      3 個樣本窗口的路徑對比
  - calibration_plot.png            機率校準圖（預測 P10/P25/P75/P90 的真實命中率）
  - metrics_summary.json            整體統計

Example:
    python scripts/walk_forward_validation.py --symbol AAPL --period 5y --interval 1d --lookback 500 --forecast 30 --paths 500 --stride 20 --seed 42 --output-dir results_wf_aapl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from us_equity_simulator import USStockFutureSimulator

DARK = "#0d0d0d"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--period", default="5y")
    p.add_argument("--interval", default="1d")
    p.add_argument("--lookback", type=int, default=500)
    p.add_argument("--forecast", type=int, default=30)
    p.add_argument("--paths", type=int, default=500)
    p.add_argument("--stride", type=int, default=20,
                   help="每隔 stride 根滑動一次窗口")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results_wf")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def compute_wf_metrics(true_path, sim_result):
    actual_end = true_path[-1]
    start = true_path[0]

    p10_end = sim_result.p10[-1]
    p25_end = sim_result.p25[-1]
    p75_end = sim_result.p75[-1]
    p90_end = sim_result.p90[-1]
    median_end = sim_result.median_path[-1]

    in_50 = np.all((true_path >= sim_result.p25) & (true_path <= sim_result.p75))
    in_80 = np.all((true_path >= sim_result.p10) & (true_path <= sim_result.p90))

    end_in_50 = float(p25_end <= actual_end <= p75_end)
    end_in_80 = float(p10_end <= actual_end <= p90_end)

    pred_dir = 1 if median_end >= start else -1
    actual_dir = 1 if actual_end >= start else -1
    dir_correct = int(pred_dir == actual_dir)

    mape = abs(median_end - actual_end) / abs(actual_end)

    return {
        "actual_ret": actual_end / start - 1.0,
        "median_ret": median_end / start - 1.0,
        "dir_correct": dir_correct,
        "end_in_50": end_in_50,
        "end_in_80": end_in_80,
        "path_in_50": int(in_50),
        "path_in_80": int(in_80),
        "mape": mape,
        "below_p10": int(actual_end < p10_end),
        "below_p25": int(actual_end < p25_end),
        "below_p75": int(actual_end < p75_end),
        "below_p90": int(actual_end < p90_end),
        "p10_end": p10_end,
        "p25_end": p25_end,
        "p75_end": p75_end,
        "p90_end": p90_end,
        "median_end": median_end,
        "actual_end": actual_end,
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.symbol}...")
    df = yf.download(args.symbol, period=args.period, interval=args.interval,
                     auto_adjust=False, progress=False)
    df = ensure_ohlcv(df)
    total_bars = len(df)
    print(f"Total bars: {total_bars}")

    min_needed = args.lookback + args.forecast
    if total_bars < min_needed:
        raise ValueError(f"Need at least {min_needed} bars, got {total_bars}")

    estimator = MarketParameterEstimator(lookback=args.lookback, vp_bins=40)
    rows = []
    sample_windows = []

    starts = list(range(0, total_bars - min_needed, args.stride))
    total_windows = len(starts)
    print(f"Running {total_windows} windows (stride={args.stride})...")

    rng_seed = np.random.default_rng(args.seed)

    for idx, start_i in enumerate(starts):
        end_train = start_i + args.lookback
        end_test  = end_train + args.forecast

        train_df = df.iloc[start_i:end_train]
        test_close = df["Close"].iloc[end_train:end_test].values.astype(float)

        if len(test_close) < args.forecast:
            continue

        try:
            params = estimator.fit(train_df, symbol=args.symbol)
        except Exception as e:
            print(f"  window {idx}: estimator failed: {e}")
            continue

        sim = USStockFutureSimulator(
            params=params,
            forecast_steps=args.forecast,
            n_paths=args.paths,
            seed=int(rng_seed.integers(0, 2**31)),
        )
        result = sim.simulate()

        metrics = compute_wf_metrics(test_close, result)
        metrics["window_idx"] = idx
        metrics["train_end_bar"] = end_train
        date_val = df.index[end_train - 1]
        metrics["date"] = str(date_val.date()) if hasattr(date_val, "date") else str(date_val)
        metrics["hurst_proxy"] = params.hurst_proxy
        metrics["smart_money_ratio"] = params.smart_money_ratio
        metrics["realized_vol"] = params.realized_vol
        rows.append(metrics)

        if len(sample_windows) < 3:
            sample_windows.append({
                "params": params,
                "result": result,
                "test_close": test_close,
                "date": metrics["date"],
                "idx": idx,
            })

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{total_windows} done")

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(out / "walk_forward_summary.csv", index=False)
    print(f"Saved {len(df_metrics)} windows to walk_forward_summary.csv")

    summary = {
        "symbol": args.symbol,
        "total_windows": len(df_metrics),
        "direction_accuracy": float(df_metrics["dir_correct"].mean()),
        "end_in_50pct_band": float(df_metrics["end_in_50"].mean()),
        "end_in_80pct_band": float(df_metrics["end_in_80"].mean()),
        "path_fully_in_50pct": float(df_metrics["path_in_50"].mean()),
        "path_fully_in_80pct": float(df_metrics["path_in_80"].mean()),
        "mean_mape": float(df_metrics["mape"].mean()),
        "median_mape": float(df_metrics["mape"].median()),
        "actual_below_p10": float(df_metrics["below_p10"].mean()),
        "actual_below_p25": float(df_metrics["below_p25"].mean()),
        "actual_below_p75": float(df_metrics["below_p75"].mean()),
        "actual_below_p90": float(df_metrics["below_p90"].mean()),
    }
    with (out / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # --- Plot 1: coverage + direction + mape ---
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        ax.set_facecolor(DARK)
        ax.grid(alpha=0.1)

    axes[0].plot(df_metrics["window_idx"],
                 df_metrics["end_in_50"].rolling(10, min_periods=1).mean(),
                 color="lime", label="50% band coverage (roll10)")
    axes[0].plot(df_metrics["window_idx"],
                 df_metrics["end_in_80"].rolling(10, min_periods=1).mean(),
                 color="deepskyblue", label="80% band coverage (roll10)")
    axes[0].axhline(0.5, color="lime", lw=0.6, alpha=0.4, linestyle=":")
    axes[0].axhline(0.8, color="deepskyblue", lw=0.6, alpha=0.4, linestyle=":")
    axes[0].set_title(f"{args.symbol} Walk-Forward: Band Coverage Rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(df_metrics["window_idx"],
                 df_metrics["dir_correct"].rolling(20, min_periods=1).mean(),
                 color="orange", label="Direction accuracy (roll20)")
    axes[1].axhline(0.5, color="white", lw=0.6, alpha=0.4, linestyle=":")
    axes[1].set_title("Direction Accuracy (Rolling 20 windows)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    axes[2].plot(df_metrics["window_idx"],
                 df_metrics["mape"].rolling(10, min_periods=1).mean(),
                 color="red", label="MAPE of median (roll10)")
    axes[2].set_title("Median Forecast MAPE")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out / "coverage_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: calibration ---
    nominal = [0.10, 0.25, 0.75, 0.90]
    actual_fracs = [
        summary["actual_below_p10"],
        summary["actual_below_p25"],
        summary["actual_below_p75"],
        summary["actual_below_p90"],
    ]
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.grid(alpha=0.12)
    ax.plot([0, 1], [0, 1], color="white", lw=1, linestyle="--", label="Perfect calibration")
    ax.scatter(nominal, actual_fracs, color="cyan", s=100, zorder=5)
    for nom, act in zip(nominal, actual_fracs):
        ax.annotate(f"  P{int(nom*100)}: {act:.2f}", (nom, act), color="cyan", fontsize=11)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical fraction below")
    ax.set_title(f"{args.symbol} Probability Calibration Plot")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(out / "calibration_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: 3 sample path comparisons ---
    if sample_windows:
        fig = plt.figure(figsize=(18, 5 * len(sample_windows)))
        fig.patch.set_facecolor(DARK)
        gs = gridspec.GridSpec(len(sample_windows), 1, hspace=0.5)

        for si, sw in enumerate(sample_windows):
            ax = fig.add_subplot(gs[si])
            ax.set_facecolor(DARK)
            ax.grid(alpha=0.1)

            r = sw["result"]
            tc = sw["test_close"]
            x = np.arange(args.forecast)

            for pi in range(min(args.paths, 100)):
                ax.plot(x, r.future_paths[pi], color="cyan", alpha=0.04, lw=0.7)
            ax.plot(x, r.median_path, color="yellow", lw=2, label="Median sim")
            ax.fill_between(x, r.p25, r.p75, color="lime", alpha=0.18, label="25-75%")
            ax.fill_between(x, r.p10, r.p90, color="deepskyblue", alpha=0.12, label="10-90%")
            ax.plot(x, tc, color="white", lw=1.8, label="Actual")
            ax.set_title(
                f"Window {sw['idx']} | date={sw['date']} | "
                f"Hurst={sw['params'].hurst_proxy:.3f} | "
                f"SMR={sw['params'].smart_money_ratio:.2f}"
            )
            ax.legend(loc="upper left")

        fig.savefig(out / "path_comparison_sample.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nAll outputs saved to: {out}/")
    print("  walk_forward_summary.csv")
    print("  metrics_summary.json")
    print("  coverage_plot.png")
    print("  calibration_plot.png")
    print("  path_comparison_sample.png")


if __name__ == "__main__":
    main()
