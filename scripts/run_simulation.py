"""
run_simulation.py

直接執行腳本（不需要 Jupyter）。
產出：
  1. Phase 1: 價格走勢 + Volume Profile + 支撐阻力
  2. Phase 2: 自動型態偵測標記
  3. Phase 3: Occupation Time 分佈
  4. 對比：GBM vs. OrderBook vs. 純白噪音

用法：
    python scripts/run_simulation.py
    python scripts/run_simulation.py --mode gbm --steps 50000 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 讓 src/ 可被 import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from market_simulator import MarketSimulator, SimConfig
from pattern_visualizer import PatternVisualizerAuto
from utils import (
    plot_volume_profile,
    compute_occupation_time,
    random_walk_baseline,
    detect_sr_levels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Random Market Simulator')
    parser.add_argument('--mode',  default='orderbook', choices=['gbm', 'orderbook'])
    parser.add_argument('--steps', type=int, default=100_000)
    parser.add_argument('--seed',  type=int, default=None)
    parser.add_argument('--bar-size', type=int, default=100,
                        help='Ticks per OHLCV bar')
    parser.add_argument('--no-show', action='store_true',
                        help='Save figures instead of displaying')
    return parser.parse_args()


def phase1_price_and_profile(price_data: np.ndarray, save: bool = False) -> None:
    """Phase 1：價格走勢 + Volume Profile + 支撐阻力"""
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    ax_price   = fig.add_subplot(gs[0, :3])
    ax_profile = fig.add_subplot(gs[0, 3])

    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0d0d0d')
    for ax in [ax_price, ax_profile]:
        ax.set_facecolor('#0d0d0d')

    plot_volume_profile(price_data, ax_price, ax_profile, bins=50, highlight_sr=True)
    fig.suptitle(
        'Phase 1: Random Walk Price Action & Liquidity Walls',
        fontsize=14, color='white', y=1.01
    )
    plt.tight_layout()
    if save:
        plt.savefig('phase1.png', dpi=150, bbox_inches='tight')
        print('Saved: phase1.png')
    else:
        plt.show()
    plt.close()


def phase2_pattern_detection(price_data: np.ndarray, save: bool = False) -> None:
    """Phase 2：自動偵測型態（頭肩頂、雙底、趨勢線）"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    viz = PatternVisualizerAuto(price_data, order=40)
    viz.plot_base_chart(ax, color='white', label='Random Price', subsample=5)
    detected = viz.find_and_draw_patterns(ax, max_patterns=6)

    ax.set_title('Phase 2: Auto-Detected Patterns on Random Data', fontsize=14, color='white')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.1)
    ax.legend(loc='upper right', frameon=True, facecolor='#1a1a1a', fontsize=8)

    proof_text = (
        f'RESULT:\n'
        f'  Detected {len(detected)} patterns on pure random data.\n'
        f'  Patterns emerge from local extrema geometry,\n'
        f'  not from market intelligence.'
    )
    props = dict(boxstyle='round', facecolor='#3d0000', alpha=0.6, edgecolor='red')
    ax.text(0.02, 0.05, proof_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', color='white', bbox=props)

    plt.tight_layout()
    if save:
        plt.savefig('phase2.png', dpi=150, bbox_inches='tight')
        print('Saved: phase2.png')
    else:
        plt.show()
    plt.close()


def phase3_occupation_time(price_data: np.ndarray, save: bool = False) -> None:
    """Phase 3：Occupation Time — 支撐阻力天然形成的統計根源"""
    plt.style.use('dark_background')
    df = compute_occupation_time(price_data, bins=100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d0d0d')
    for ax in axes:
        ax.set_facecolor('#0d0d0d')

    # 左：水平條形圖（Occupation Time vs. Price Level）
    axes[0].barh(df['price_level'], df['occupation_time'],
                 height=(df['price_level'].max() - df['price_level'].min()) / 100,
                 color='orange', alpha=0.6)
    axes[0].set_xlabel('Occupation Time (ticks)', color='white')
    axes[0].set_ylabel('Price Level', color='white')
    axes[0].set_title('Occupation Time Distribution\n(why S/R levels emerge)', color='white')
    axes[0].grid(alpha=0.1)

    # 右：標記 S/R 在價格走勢上的位置
    sr_levels = detect_sr_levels(price_data)
    axes[1].plot(price_data[::10], color='cyan', lw=0.5, alpha=0.8)
    for lvl in sr_levels:
        axes[1].axhline(lvl, color='red', lw=0.6, alpha=0.5, linestyle='--')
    axes[1].set_title(f'S/R Levels from Occupation Time\n({len(sr_levels)} levels detected)', color='white')
    axes[1].grid(alpha=0.1)

    fig.suptitle('Phase 3: Statistical Origin of Support/Resistance', fontsize=13, color='white')
    plt.tight_layout()
    if save:
        plt.savefig('phase3.png', dpi=150, bbox_inches='tight')
        print('Saved: phase3.png')
    else:
        plt.show()
    plt.close()


def phase4_comparison(price_data: np.ndarray, steps: int, save: bool = False) -> None:
    """Phase 4：三種隨機過程對比（GBM / OrderBook / Pure Random Walk）"""
    plt.style.use('dark_background')

    gbm_sim   = MarketSimulator(mode='gbm',       config=SimConfig(steps=steps, sigma=0.002))
    ob_sim    = MarketSimulator(mode='orderbook', config=SimConfig(steps=steps))
    gbm_data  = gbm_sim.run()
    ob_data   = ob_sim.run()
    rw_data   = random_walk_baseline(steps)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    fig.patch.set_facecolor('#0d0d0d')
    datasets = [
        (rw_data,  'Pure Random Walk (White Noise Cumsum)', 'grey'),
        (gbm_data, 'GBM (Geometric Brownian Motion)',       'cyan'),
        (ob_data,  'OrderBook Simulation',                  'orange'),
    ]
    for ax, (data, title, color) in zip(axes, datasets):
        ax.set_facecolor('#0d0d0d')
        ax.plot(data, color=color, lw=0.6, alpha=0.9)
        sr = detect_sr_levels(data)
        for lvl in sr:
            ax.axhline(lvl, color='red', lw=0.5, alpha=0.4, linestyle='--')
        ax.set_title(title, color='white', fontsize=11)
        ax.grid(alpha=0.1)

    fig.suptitle('Phase 4: Three Random Processes — All Produce Similar Patterns',
                 fontsize=13, color='white')
    plt.tight_layout()
    if save:
        plt.savefig('phase4.png', dpi=150, bbox_inches='tight')
        print('Saved: phase4.png')
    else:
        plt.show()
    plt.close()


def main() -> None:
    args = parse_args()

    print(f'Running simulation: mode={args.mode}, steps={args.steps:,}, seed={args.seed}')

    cfg = SimConfig(
        steps=args.steps,
        seed=args.seed,
    )
    sim = MarketSimulator(mode=args.mode, config=cfg)
    GLOBAL_PRICE_DATA = sim.run()

    print(f'Generated {len(GLOBAL_PRICE_DATA):,} ticks. '
          f'Price range: {GLOBAL_PRICE_DATA.min():.2f} - {GLOBAL_PRICE_DATA.max():.2f}')

    save = args.no_show

    print('\n[Phase 1] Price Action + Volume Profile...')
    phase1_price_and_profile(GLOBAL_PRICE_DATA, save=save)

    print('[Phase 2] Pattern Detection...')
    phase2_pattern_detection(GLOBAL_PRICE_DATA, save=save)

    print('[Phase 3] Occupation Time Analysis...')
    phase3_occupation_time(GLOBAL_PRICE_DATA, save=save)

    print('[Phase 4] Three-Way Comparison...')
    phase4_comparison(GLOBAL_PRICE_DATA, steps=args.steps, save=save)

    print('\nDone.')


if __name__ == '__main__':
    main()
