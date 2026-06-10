"""
run_simulation.py

執行完整模擬實驗，產出 5 張圖：
  phase1 : 價格走勢 + Volume Profile + 支撐阻力
  phase2 : 自動型態偵測
  phase3 : Occupation Time
  phase4 : GBM vs OrderBook vs MultiAgent 三方對比
  phase5 : smart_money_ratio 敏感度分析（0.0 / 0.3 / 0.7 / 0.95）

用法：
    python scripts/run_simulation.py
    python scripts/run_simulation.py --mode multiagent --steps 100000 --seed 42
    python scripts/run_simulation.py --smart-ratio 0.5 --no-show
    python scripts/run_simulation.py --no-show --output-dir results/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_simulator import MarketSimulator, SimConfig
from pattern_visualizer import PatternVisualizerAuto
from utils import (
    plot_volume_profile,
    compute_occupation_time,
    random_walk_baseline,
    detect_sr_levels,
)

DARK_BG = "#0d0d0d"


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Market Simulator")
    parser.add_argument("--mode",  default="multiagent",
                        choices=["gbm", "orderbook", "multiagent"])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed",  type=int, default=None)
    parser.add_argument("--smart-ratio", type=float, default=0.3,
                        help="Smart money ratio (0~1). Only used in multiagent mode.")
    parser.add_argument("--bar-size", type=int, default=100,
                        help="Ticks per OHLCV bar.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures instead of displaying.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save figures when --no-show is set.")
    return parser.parse_args()


def save_or_show(fig: plt.Figure, name: str, output_dir: str, save: bool) -> None:
    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def dark_fig(*args, **kwargs) -> tuple:
    plt.style.use("dark_background")
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(DARK_BG)
    return fig


def dark_ax(ax: plt.Axes) -> plt.Axes:
    ax.set_facecolor(DARK_BG)
    return ax


# ──────────────────────────────────────────────────────────────────────
# Phase 1: 價格走勢 + Volume Profile
# ──────────────────────────────────────────────────────────────────────

def phase1(price_data: np.ndarray, save: bool, output_dir: str) -> None:
    fig = dark_fig(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    ax_p = dark_ax(fig.add_subplot(gs[0, :3]))
    ax_v = dark_ax(fig.add_subplot(gs[0, 3]))
    plot_volume_profile(price_data, ax_p, ax_v, bins=50, highlight_sr=True)
    fig.suptitle("Phase 1: Price Action & Liquidity Walls",
                 fontsize=14, color="white", y=1.01)
    plt.tight_layout()
    save_or_show(fig, "phase1", output_dir, save)


# ──────────────────────────────────────────────────────────────────────
# Phase 2: 自動型態偵測
# ──────────────────────────────────────────────────────────────────────

def phase2(price_data: np.ndarray, save: bool, output_dir: str) -> None:
    fig = dark_fig(figsize=(18, 7))
    ax = dark_ax(fig.add_subplot(111))
    viz = PatternVisualizerAuto(price_data, order=40)
    viz.plot_base_chart(ax, color="white", label="Random Price", subsample=5)
    detected = viz.find_and_draw_patterns(ax, max_patterns=6)
    ax.set_title("Phase 2: Auto-Detected Patterns on Random Data",
                 fontsize=14, color="white")
    ax.grid(True, alpha=0.1)
    ax.legend(loc="upper right", frameon=True, facecolor="#1a1a1a", fontsize=8)
    proof = (
        f"RESULT:\n"
        f"  {len(detected)} patterns detected on pure random data.\n"
        f"  Geometry emerges from local extrema — not market intelligence."
    )
    ax.text(0.02, 0.05, proof, transform=ax.transAxes, fontsize=9,
            va="bottom", color="white",
            bbox=dict(boxstyle="round", facecolor="#3d0000", alpha=0.6, edgecolor="red"))
    plt.tight_layout()
    save_or_show(fig, "phase2", output_dir, save)


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Occupation Time
# ──────────────────────────────────────────────────────────────────────

def phase3(price_data: np.ndarray, save: bool, output_dir: str) -> None:
    df = compute_occupation_time(price_data, bins=100)
    fig = dark_fig(figsize=(16, 6))
    axes = [dark_ax(fig.add_subplot(1, 2, i + 1)) for i in range(2)]

    price_range = df["price_level"].max() - df["price_level"].min()
    axes[0].barh(df["price_level"], df["occupation_time"],
                 height=price_range / 100, color="orange", alpha=0.6)
    axes[0].set_xlabel("Occupation Time (ticks)", color="white")
    axes[0].set_ylabel("Price Level", color="white")
    axes[0].set_title("Occupation Time Distribution", color="white")
    axes[0].grid(alpha=0.1)

    sr = detect_sr_levels(price_data)
    axes[1].plot(price_data[::10], color="cyan", lw=0.5, alpha=0.8)
    for lvl in sr:
        axes[1].axhline(lvl, color="red", lw=0.6, alpha=0.5, linestyle="--")
    axes[1].set_title(f"S/R from Occupation Time ({len(sr)} levels)", color="white")
    axes[1].grid(alpha=0.1)

    fig.suptitle("Phase 3: Statistical Origin of Support/Resistance",
                 fontsize=13, color="white")
    plt.tight_layout()
    save_or_show(fig, "phase3", output_dir, save)


# ──────────────────────────────────────────────────────────────────────
# Phase 4: 三種模型對比
# ──────────────────────────────────────────────────────────────────────

def phase4(steps: int, seed, save: bool, output_dir: str) -> None:
    gbm_data = MarketSimulator(
        mode="gbm", config=SimConfig(steps=steps, sigma=0.0004, seed=seed)
    ).run()
    ob_data = MarketSimulator(
        mode="orderbook", config=SimConfig(steps=steps, sigma=0.0004, seed=seed)
    ).run()
    ma_data = MarketSimulator(
        mode="multiagent", config=SimConfig(steps=steps, seed=seed)
    ).run()
    rw_data = random_walk_baseline(steps)

    fig = dark_fig(figsize=(18, 14))
    datasets = [
        (rw_data,  "Pure Random Walk (White Noise Cumsum)",  "grey"),
        (gbm_data, "GBM (Geometric Brownian Motion)",        "cyan"),
        (ob_data,  "OrderBook Simulation",                   "dodgerblue"),
        (ma_data,  "MultiAgent (Noise Trader + Smart Money + MM)", "orange"),
    ]
    for idx, (data, title, color) in enumerate(datasets):
        ax = dark_ax(fig.add_subplot(4, 1, idx + 1))
        ax.plot(data, color=color, lw=0.6, alpha=0.9)
        for lvl in detect_sr_levels(data):
            ax.axhline(lvl, color="red", lw=0.5, alpha=0.4, linestyle="--")
        ax.set_title(title, color="white", fontsize=11)
        ax.grid(alpha=0.1)

    fig.suptitle("Phase 4: Four Models — All Produce Visually Similar Patterns",
                 fontsize=13, color="white")
    plt.tight_layout()
    save_or_show(fig, "phase4", output_dir, save)


# ──────────────────────────────────────────────────────────────────────
# Phase 5: smart_money_ratio 敏感度分析（新增）
# ──────────────────────────────────────────────────────────────────────

def phase5(steps: int, seed, save: bool, output_dir: str) -> None:
    """
    固定其他參數，只改變 smart_money_ratio，
    觀察市場結構如何隨大戶比例改變。
    """
    ratios = [0.0, 0.3, 0.7, 0.95]
    labels = [
        "0.0  — 全散戶，純噪音",
        "0.3  — 接近現實（大戶少但影響大）",
        "0.7  — 大戶主導，明顯 mean-reversion",
        "0.95 — 近乎機構市場，極度平滑",
    ]
    colors = ["red", "orange", "cyan", "lime"]

    fig = dark_fig(figsize=(18, 16))

    for idx, (ratio, label, color) in enumerate(zip(ratios, labels, colors)):
        cfg = SimConfig(
            steps=steps,
            seed=seed,
            smart_money_ratio=ratio,
        )
        data = MarketSimulator(mode="multiagent", config=cfg).run()

        ax_price   = dark_ax(fig.add_subplot(4, 4, idx * 4 + 1))
        ax_profile = dark_ax(fig.add_subplot(4, 4, idx * 4 + 2))
        ax_hist    = dark_ax(fig.add_subplot(4, 4, idx * 4 + 3))
        ax_occ     = dark_ax(fig.add_subplot(4, 4, idx * 4 + 4))

        # 價格走勢
        ax_price.plot(data, color=color, lw=0.5, alpha=0.9)
        for lvl in detect_sr_levels(data):
            ax_price.axhline(lvl, color="red", lw=0.5, alpha=0.3, linestyle="--")
        ax_price.set_title(f"smart_ratio={ratio}\n{label}",
                           color="white", fontsize=8)
        ax_price.grid(alpha=0.08)

        # Volume Profile
        import seaborn as sns
        sns.histplot(y=data, bins=40, color=color, alpha=0.4,
                     ax=ax_profile, kde=True)
        ax_profile.set_title("Volume Profile", color="white", fontsize=8)
        ax_profile.grid(alpha=0.08)
        ax_profile.set_ylabel("")

        # 回報分佈（fat tail 觀察）
        returns = np.diff(data) / data[:-1]
        ax_hist.hist(returns, bins=80, color=color, alpha=0.6, density=True)
        ax_hist.set_title("Return Dist.", color="white", fontsize=8)
        ax_hist.set_xlim(-0.02, 0.02)
        ax_hist.grid(alpha=0.08)

        # Occupation Time
        df_occ = compute_occupation_time(data, bins=60)
        price_range = df_occ["price_level"].max() - df_occ["price_level"].min()
        ax_occ.barh(df_occ["price_level"], df_occ["occupation_time"],
                    height=price_range / 60, color=color, alpha=0.5)
        ax_occ.set_title("Occupation Time", color="white", fontsize=8)
        ax_occ.grid(alpha=0.08)

    fig.suptitle(
        "Phase 5: smart_money_ratio Sensitivity\n"
        "(Price Action | Volume Profile | Return Dist. | Occupation Time)",
        fontsize=13, color="white",
    )
    plt.tight_layout()
    save_or_show(fig, "phase5", output_dir, save)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"Mode: {args.mode} | Steps: {args.steps:,} | Seed: {args.seed} "
          f"| smart_ratio: {args.smart_ratio}")

    cfg = SimConfig(
        steps=args.steps,
        seed=args.seed,
        smart_money_ratio=args.smart_ratio,
    )
    sim = MarketSimulator(mode=args.mode, config=cfg)
    data = sim.run()

    print(f"  {len(data):,} ticks generated. "
          f"Range: {data.min():.2f} — {data.max():.2f}")

    save = args.no_show
    out  = args.output_dir

    print("[Phase 1] Price Action + Volume Profile")
    phase1(data, save, out)

    print("[Phase 2] Pattern Detection")
    phase2(data, save, out)

    print("[Phase 3] Occupation Time")
    phase3(data, save, out)

    print("[Phase 4] Four-Model Comparison")
    phase4(args.steps, args.seed, save, out)

    print("[Phase 5] smart_money_ratio Sensitivity")
    phase5(args.steps, args.seed, save, out)

    print("Done.")


if __name__ == "__main__":
    main()
