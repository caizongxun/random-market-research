"""
run_us_stock_experiment.py

使用免費美股資料（Yahoo Finance via yfinance）進行：
- 下載歷史日 K
- 估計最近 500 根 K 棒的市場隱含參數
- 模擬未來 30 根 K 的機率路徑
- 輸出圖與參數摘要

Example:
    python scripts/run_us_stock_experiment.py --symbol AAPL --period 3y --interval 1d --lookback 500 --forecast 30 --paths 1000 --seed 42 --output-dir results_aapl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from us_equity_simulator import USStockFutureSimulator


DARK_BG = "#0d0d0d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US stock future path experiment")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--period", default="3y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--lookback", type=int, default=500)
    parser.add_argument("--forecast", type=int, default=30)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="results_us")
    return parser.parse_args()


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.symbol} from Yahoo Finance...")
    df = yf.download(args.symbol, period=args.period, interval=args.interval, auto_adjust=False, progress=False)
    df = ensure_ohlcv(df)
    if len(df) < args.lookback:
        raise ValueError(f"Downloaded rows={len(df)} < lookback={args.lookback}")

    estimator = MarketParameterEstimator(lookback=args.lookback, vp_bins=40)
    params = estimator.fit(df, symbol=args.symbol)

    sim = USStockFutureSimulator(
        params=params,
        forecast_steps=args.forecast,
        n_paths=args.paths,
        seed=args.seed,
    )
    result = sim.simulate()

    # Save parameter summary
    summary = {
        "symbol": params.symbol,
        "last_close": params.last_close,
        "realized_vol": params.realized_vol,
        "drift": params.drift,
        "hurst_proxy": params.hurst_proxy,
        "avg_range_ratio": params.avg_range_ratio,
        "gap_std": params.gap_std,
        "trend_strength": params.trend_strength,
        "mean_reversion_strength": params.mean_reversion_strength,
        "smart_money_ratio": params.smart_money_ratio,
        "volume_nodes": params.volume_nodes,
        "volume_node_strength": params.volume_node_strength,
        "nearest_upper_node": result.nearest_upper_node,
        "nearest_lower_node": result.nearest_lower_node,
        "breakout_prob_up": result.breakout_prob_up,
        "breakout_prob_down": result.breakout_prob_down,
    }
    save_json(summary, out / "estimated_params.json")

    # Plot
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.grid(alpha=0.12)

    hist_close = df["Close"].tail(args.lookback).reset_index(drop=True)
    axes[0].plot(hist_close.index, hist_close.values, color="white", lw=1.0, label="Historical Close")
    for node, strength in zip(params.volume_nodes, params.volume_node_strength):
        axes[0].axhline(node, color="orange", lw=0.8 + 1.2 * strength, alpha=0.35, linestyle="--")
    axes[0].set_title(f"{args.symbol} historical {args.lookback} bars + volume nodes")
    axes[0].legend(loc="upper left")

    x_future = np.arange(len(hist_close), len(hist_close) + args.forecast)
    for i in range(min(args.paths, 150)):
        axes[1].plot(x_future, result.future_paths[i], color="cyan", alpha=0.03, lw=0.8)
    axes[1].plot(x_future, result.median_path, color="yellow", lw=1.8, label="Median path")
    axes[1].fill_between(x_future, result.p25, result.p75, color="lime", alpha=0.18, label="25-75% band")
    axes[1].fill_between(x_future, result.p10, result.p90, color="deepskyblue", alpha=0.12, label="10-90% band")
    axes[1].axhline(params.last_close, color="white", lw=0.8, alpha=0.4, linestyle=":")
    if result.nearest_upper_node is not None:
        axes[1].axhline(result.nearest_upper_node, color="red", lw=0.8, alpha=0.4, linestyle="--")
    if result.nearest_lower_node is not None:
        axes[1].axhline(result.nearest_lower_node, color="orange", lw=0.8, alpha=0.4, linestyle="--")
    axes[1].set_title(
        f"Future {args.forecast} bars simulation | up_break={result.breakout_prob_up:.2%} | down_break={result.breakout_prob_down:.2%}"
    )
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out / "simulation_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to: {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
