"""
utils.py

通用工具函式：Volume Profile、支撐阻力偵測、統計工具。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Optional


def compute_volume_profile(
    price_data: np.ndarray,
    bins: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    計算 Volume Profile（價格密度直方圖）。

    Returns
    -------
    counts     : 各 bin 的計數
    bin_edges  : bin 邊界
    bin_centers: bin 中心價格
    """
    counts, bin_edges = np.histogram(price_data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_edges, bin_centers


def detect_sr_levels(
    price_data: np.ndarray,
    bins: int = 50,
    peak_distance: int = 5,
    min_prominence: Optional[float] = None,
) -> np.ndarray:
    """
    從 Volume Profile 中偵測支撐阻力價位。

    原理：Volume Profile 的高密度 bin（直方圖高峰）對應
    價格停留時間最長的區域，這些區域在隨機路徑中天然形成。

    Returns
    -------
    sr_prices : 偵測到的支撐阻力價位陣列
    """
    counts, bin_edges, bin_centers = compute_volume_profile(price_data, bins)

    kwargs = {'distance': peak_distance}
    if min_prominence is not None:
        kwargs['prominence'] = min_prominence

    peaks, _ = find_peaks(counts, **kwargs)
    return bin_centers[peaks]


def compute_occupation_time(
    price_data: np.ndarray,
    bins: int = 100,
) -> pd.DataFrame:
    """
    計算隨機路徑在各價格區間的停留時間（Occupation Time）。
    這是支撐阻力「天然形成」的核心統計機制。

    Returns
    -------
    pd.DataFrame，欄位：price_level, occupation_time, fraction
    """
    counts, bin_edges, bin_centers = compute_volume_profile(price_data, bins)
    total = counts.sum()
    df = pd.DataFrame({
        'price_level': bin_centers,
        'occupation_time': counts,
        'fraction': counts / total,
    })
    return df


def random_walk_baseline(
    n: int = 100_000,
    seed: int | None = 42,
) -> np.ndarray:
    """
    純隨機遊走基準（白噪音累積），用於與模擬器結果對比。
    P(t) = P(t-1) + epsilon，epsilon ~ N(0, 1)
    """
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n)
    return 100 + np.cumsum(steps)


def plot_volume_profile(
    price_data: np.ndarray,
    ax_price: plt.Axes,
    ax_profile: plt.Axes,
    bins: int = 50,
    highlight_sr: bool = True,
) -> None:
    """
    在 ax_price 繪製價格走勢，在 ax_profile 繪製 Volume Profile。
    若 highlight_sr=True，在 ax_price 上標記偵測到的支撐阻力線。
    """
    import seaborn as sns

    # 價格走勢
    ax_price.plot(price_data, color='cyan', lw=0.8, alpha=0.9)
    ax_price.set_title('Simulated Price Action', color='white')
    ax_price.grid(alpha=0.15)

    # Volume Profile
    sns.histplot(y=price_data, bins=bins, color='orange', alpha=0.5,
                 ax=ax_profile, kde=True)
    ax_profile.set_title('Volume Profile', color='white')
    ax_profile.grid(alpha=0.15)
    ax_profile.set_ylabel('')

    # 支撐阻力標記
    if highlight_sr:
        sr_levels = detect_sr_levels(price_data, bins=bins)
        for level in sr_levels:
            ax_price.axhline(level, color='red', lw=0.7, linestyle='--', alpha=0.5)
            ax_price.text(
                len(price_data) * 0.01, level,
                f'{level:.1f}', color='red', fontsize=7, va='bottom'
            )
