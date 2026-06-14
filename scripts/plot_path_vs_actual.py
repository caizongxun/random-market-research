"""
plot_path_vs_actual.py

對一段指定的歷史區間：
1. 畫出歷史 K 棒（lookback 根）
2. 畫出 Volume Profile 節點
3. 畫出模擬的未來多條路徑 + 分位帶
4. 畫出真實未來 K 棒（紅綠角色區分漲跌）

Example:
    python scripts/plot_path_vs_actual.py \\
        --symbol AAPL --period 5y --interval 1d \\
        --lookback 120 --forecast 30 --paths 300 \\
        --seed 42 --output results_compare

Tip: 可用 --n-samples 控制輸出幾張不同窗口的對比圖
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

DARK = "#0e0e0e"
GRID = "#1e1e1e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    required=True)
    p.add_argument("--period",    default="3y")
    p.add_argument("--interval",  default="1d")
    p.add_argument("--lookback",  type=int, default=120,
                   help="展示的歷史 K 棒數（估參數用全部 500 根）")
    p.add_argument("--forecast",  type=int, default=30)
    p.add_argument("--paths",     type=int, default=300)
    p.add_argument("--n-samples", type=int, default=4,
                   help="輸出幾個不同窗口的對比圖")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--vol-scale", type=float, default=1.5)
    p.add_argument("--output",    default="results_compare")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def draw_candles(ax, df_slice, x_offset=0, alpha=1.0, up_col="#26a69a", down_col="#ef5350"):
    """df_slice: DataFrame with Open/High/Low/Close columns, indexed 0..N"""
    for i, row in df_slice.iterrows():
        x    = i + x_offset
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        color = up_col if c >= o else down_col
        # 影線
        ax.plot([x, x], [l, h], color=color, lw=0.8, alpha=alpha)
        # 實體
        rect_y = min(o, c)
        rect_h = abs(c - o) if abs(c - o) > 0 else (h - l) * 0.01
        bar = mpatches.FancyBboxPatch(
            (x - 0.35, rect_y), 0.7, rect_h,
            boxstyle="square,pad=0",
            linewidth=0, facecolor=color, alpha=alpha,
        )
        ax.add_patch(bar)


def sim_ohlcv_from_paths(paths: np.ndarray, start_price: float) -> pd.DataFrame:
    """
    將模擬路徑轉成似乎 OHLCV 的 DataFrame（用中位數和分位數代替展示）。
    Close = 中位數, High = p90, Low = p10, Open = 前一根 Close
    """
    median = np.median(paths, axis=0)
    p10    = np.percentile(paths, 10, axis=0)
    p90    = np.percentile(paths, 90, axis=0)

    opens  = np.concatenate([[start_price], median[:-1]])
    rows = []
    for t in range(len(median)):
        rows.append({
            "Open":  opens[t],
            "High":  p90[t],
            "Low":   p10[t],
            "Close": median[t],
        })
    return pd.DataFrame(rows)


def plot_one_window(
    ax_main, ax_vol,
    hist_df,        # 展示用歷史 K 棒 (lookback 根)
    actual_df,      # 真實未來 K 棒 (forecast 根)
    sim_paths,      # shape: (n_paths, forecast)
    sim_result,
    params,
    title: str,
    show_paths: int = 60,
):
    n_hist     = len(hist_df)
    n_forecast = sim_paths.shape[1]
    x_hist     = np.arange(n_hist)
    x_future   = np.arange(n_hist, n_hist + n_forecast)

    # 分隔線
    ax_main.axvline(n_hist - 0.5, color="#555", lw=1.2, linestyle="--", alpha=0.7)

    # 歷史 K 棒
    hist_indexed = hist_df.reset_index(drop=True)
    draw_candles(ax_main, hist_indexed, x_offset=0, alpha=0.85)

    # Volume Profile 節點
    for node, strength in zip(params.volume_nodes, params.volume_node_strength):
        ax_main.axhline(
            node, color="orange", lw=0.6 + 1.0 * strength,
            alpha=0.25 + 0.3 * strength, linestyle=":",
        )

    # 模擬路徑（不超過 show_paths 條）
    for pi in range(min(show_paths, sim_paths.shape[0])):
        ax_main.plot(x_future, sim_paths[pi], color="cyan", alpha=0.04, lw=0.7)

    # 分位帶
    ax_main.fill_between(
        x_future, sim_result.p25, sim_result.p75,
        color="#00e5ff", alpha=0.15, label="25-75%",
    )
    ax_main.fill_between(
        x_future, sim_result.p10, sim_result.p90,
        color="#00e5ff", alpha=0.07, label="10-90%",
    )
    ax_main.plot(x_future, sim_result.median_path, color="yellow",
                 lw=1.8, label="Median sim", zorder=5)

    # 真實未來 K 棒
    actual_indexed = actual_df.reset_index(drop=True)
    draw_candles(ax_main, actual_indexed, x_offset=n_hist, alpha=1.0)

    # 歷史成交量柱狀圖
    if "Volume" in hist_df.columns:
        vols   = hist_df["Volume"].values
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for o, c in zip(hist_df["Open"].values, hist_df["Close"].values)]
        ax_vol.bar(x_hist, vols, color=colors, alpha=0.5, width=0.8)
    ax_vol.set_xlim(ax_main.get_xlim())
    ax_vol.set_facecolor(DARK)
    ax_vol.tick_params(colors="#888", labelsize=7)
    ax_vol.set_ylabel("Volume", color="#888", fontsize=7)
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x:.0f}")
    )

    # 裝飾
    ax_main.set_facecolor(DARK)
    ax_main.tick_params(colors="#888", labelsize=8)
    ax_main.set_title(title, color="white", fontsize=10, pad=6)
    ax_main.legend(
        loc="upper left", fontsize=7,
        facecolor="#1a1a1a", edgecolor="#333", labelcolor="white",
    )
    ax_main.set_xlim(-1, n_hist + n_forecast + 1)

    # 標註分隔線
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
    print(f"Total bars: {total}")

    ESTIMATOR_LOOKBACK = 500
    min_needed = ESTIMATOR_LOOKBACK + args.forecast
    if total < min_needed:
        raise ValueError(f"Need {min_needed} bars, got {total}")

    estimator = MarketParameterEstimator(lookback=ESTIMATOR_LOOKBACK, vp_bins=40)
    rng       = np.random.default_rng(args.seed)

    # 均勻地選取 n_samples 個窗口
    max_start  = total - min_needed
    if args.n_samples <= 1:
        sample_starts = [max_start]
    else:
        sample_starts = [
            int(max_start * i / (args.n_samples - 1))
            for i in range(args.n_samples)
        ]
    sample_starts = [min(s, max_start) for s in sample_starts]

    plt.style.use("dark_background")

    for fig_idx, train_start in enumerate(sample_starts):
        train_end = train_start + ESTIMATOR_LOOKBACK
        test_end  = train_end + args.forecast

        train_df_full = df.iloc[train_start:train_end]
        actual_df     = df.iloc[train_end:test_end]

        # 展示用：只取最後 lookback 根
        display_start = max(train_start, train_end - args.lookback)
        hist_df       = df.iloc[display_start:train_end]

        try:
            params = estimator.fit(train_df_full, symbol=args.symbol)
        except Exception as e:
            print(f"  window {fig_idx}: estimator failed: {e}")
            continue

        sim = USStockFutureSimulator(
            params=params,
            forecast_steps=args.forecast,
            n_paths=args.paths,
            seed=int(rng.integers(0, 2**31)),
            vol_scale=args.vol_scale,
        )
        result = sim.simulate()

        date_str = str(df["index"].iloc[train_end - 1])[:10] if "index" in df.columns else f"bar{train_end}"

        # 建圖：主圖 + 小成交量圖
        fig, axes = plt.subplots(
            2, 1,
            figsize=(20, 9),
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04},
            sharex=False,
        )
        fig.patch.set_facecolor(DARK)

        plot_one_window(
            ax_main=axes[0],
            ax_vol=axes[1],
            hist_df=hist_df,
            actual_df=actual_df,
            sim_paths=result.future_paths,
            sim_result=result,
            params=params,
            title=(
                f"{args.symbol}  |  window {fig_idx+1}/{args.n_samples}  |  "
                f"estimate end: {date_str}  |  "
                f"Hurst={params.hurst_proxy:.3f}  "
                f"SMR={params.smart_money_ratio:.2f}  "
                f"vol={params.ewma_vol:.4f}  "
                f"drift={params.ewma_drift:.5f}"
            ),
            show_paths=80,
        )

        fname = out / f"compare_{args.symbol}_w{fig_idx+1:02d}_{date_str}.png"
        fig.savefig(fname, dpi=160, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        print(f"  Saved: {fname.name}")

    print(f"\nDone. {args.n_samples} charts saved to {out}/")


if __name__ == "__main__":
    main()
