"""
candle_renderer.py

將模擬出來的 OHLCV 陣列用 mplfinance 畫成 K 棒圖。
支援：
  - 左半段：歷史 K 棒（真實數據）
  - 右半段：預測 K 棒（intra-bar 模擬）+ 中位數帶子
  - 下方：逐步偏差柱狀圖
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import mplfinance as mpf

DARK = "#0e0e0e"


def _make_ohlcv_df(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    start_idx: int = 0,
) -> pd.DataFrame:
    """建立 mplfinance 需要的 DatetimeIndex OHLCV DataFrame"""
    n  = len(closes)
    idx = pd.date_range(start="2020-01-01", periods=start_idx + n, freq="B")  # 工作日
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": closes, "Volume": volumes},
        index=idx[start_idx:],
    )
    return df


def render_forecast_candles(
    # 歷史歟段（真實）
    hist_open:   np.ndarray,
    hist_high:   np.ndarray,
    hist_low:    np.ndarray,
    hist_close:  np.ndarray,
    hist_volume: np.ndarray,
    # 預測歟段（intra-bar 模擬）
    fwd_open:    np.ndarray,
    fwd_high:    np.ndarray,
    fwd_low:     np.ndarray,
    fwd_close:   np.ndarray,
    fwd_volume:  np.ndarray,
    # 模擬帶子（基於 median_path）
    p25: np.ndarray,
    p75: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    # 實際後續 K 棒（可第 None）
    actual_open:   Optional[np.ndarray] = None,
    actual_high:   Optional[np.ndarray] = None,
    actual_low:    Optional[np.ndarray] = None,
    actual_close:  Optional[np.ndarray] = None,
    actual_volume: Optional[np.ndarray] = None,
    # 其他
    title: str = "Forecast Candles",
    output_path: Optional[Path] = None,
    volume_nodes: Optional[list] = None,
    hist_window: int = 60,   # 顯示歷史幾根（節省畫面）
    actual_deviation: Optional[np.ndarray] = None,  # 逐步偏差 %
    metrics: Optional[dict] = None,
) -> plt.Figure:
    """
    繪制包含預測 K 棒的完整圖表。
    回傳 Figure。
    """
    T_hist = len(hist_close)
    T_fwd  = len(fwd_close)
    has_actual = actual_close is not None and len(actual_close) > 0

    # 只顯示最後 hist_window 根歷史
    show_hist = min(hist_window, T_hist)
    sl = slice(T_hist - show_hist, T_hist)

    h_o = hist_open[sl]
    h_h = hist_high[sl]
    h_l = hist_low[sl]
    h_c = hist_close[sl]
    h_v = hist_volume[sl]

    # 建立欷期：歷史 + 預測 + 實際（同一連續工作日序列）
    n_total = show_hist + T_fwd
    idx_full = pd.date_range(start="2020-01-01", periods=n_total, freq="B")
    idx_hist = idx_full[:show_hist]
    idx_fwd  = idx_full[show_hist:]

    # 整合为單一 DataFrame
    all_open   = np.concatenate([h_o, fwd_open])
    all_high   = np.concatenate([h_h, fwd_high])
    all_low    = np.concatenate([h_l, fwd_low])
    all_close  = np.concatenate([h_c, fwd_close])
    all_volume = np.concatenate([h_v, fwd_volume])

    df_all = pd.DataFrame(
        {"Open": all_open, "High": all_high, "Low": all_low,
         "Close": all_close, "Volume": all_volume},
        index=idx_full,
    )

    # mplfinance 自訂樣式
    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        wick={"up": "#26a69a", "down": "#ef5350"},
        volume={"up": "#26a69a88", "down": "#ef535088"},
        edge="inherit",
    )
    fwd_mc = mpf.make_marketcolors(
        up="#ffd54f", down="#ff8a65",   # 預測 K 棒用暖色區分
        wick={"up": "#ffd54f", "down": "#ff8a65"},
        volume={"up": "#ffd54f55", "down": "#ff8a6555"},
        edge="inherit",
    )

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor=DARK,
        edgecolor="#333",
        figcolor=DARK,
        gridcolor="#1e1e1e",
        gridstyle="-",
        y_on_right=True,
    )

    # 建立帶子 addplot
    band_x = np.full(n_total, np.nan)
    band_x[show_hist:] = 1.0  # 占位用

    ap_list = []

    # 25-75% 帶子
    p25_full = np.full(n_total, np.nan)
    p75_full = np.full(n_total, np.nan)
    p10_full = np.full(n_total, np.nan)
    p90_full = np.full(n_total, np.nan)
    p25_full[show_hist:show_hist + len(p25)] = p25
    p75_full[show_hist:show_hist + len(p75)] = p75
    p10_full[show_hist:show_hist + len(p10)] = p10
    p90_full[show_hist:show_hist + len(p90)] = p90

    ap_list.append(mpf.make_addplot(p75_full, color="#00e5ff", alpha=0.5, width=1.2,
                                    fill_between={"y1": p25_full, "y2": p75_full,
                                                  "color": "#00e5ff", "alpha": 0.18}))
    ap_list.append(mpf.make_addplot(p90_full, color="#00e5ff", alpha=0.25, width=0.8,
                                    fill_between={"y1": p10_full, "y2": p90_full,
                                                  "color": "#00e5ff", "alpha": 0.08}))
    ap_list.append(mpf.make_addplot(p25_full, color="#00e5ff", alpha=0.5, width=1.2))
    ap_list.append(mpf.make_addplot(p10_full, color="#00e5ff", alpha=0.25, width=0.8))

    # 實際 K 棒吊層（如果有）
    if has_actual:
        n_act = min(len(actual_close), T_fwd)
        act_o = np.full(n_total, np.nan)
        act_h = np.full(n_total, np.nan)
        act_l = np.full(n_total, np.nan)
        act_c = np.full(n_total, np.nan)
        act_o[show_hist:show_hist + n_act] = actual_open[:n_act]
        act_h[show_hist:show_hist + n_act] = actual_high[:n_act]
        act_l[show_hist:show_hist + n_act] = actual_low[:n_act]
        act_c[show_hist:show_hist + n_act] = actual_close[:n_act]
        # 實際 K 棒用白色线条表示（避免覆蓋預測 K 棒）
        ap_list.append(mpf.make_addplot(act_c, color="white", width=1.8,
                                         type="line", alpha=0.9))

    # --- 繪制 ---
    n_panels = 2 if actual_deviation is not None else 1
    ratios   = (3, 1) if n_panels == 2 else None

    fig, axes = mpf.plot(
        df_all,
        type="candle",
        style=style,
        addplot=ap_list,
        volume=True,
        title=title,
        returnfig=True,
        figsize=(18, 10 if n_panels == 2 else 8),
        tight_layout=False,
        warn_too_much_data=9999,
    )

    ax_main = axes[0]

    # 分湋線（歷史 / 預測）
    ax_main.axvline(show_hist - 0.5, color="#888", lw=1.2, ls="-", alpha=0.8)
    ax_main.text(show_hist - 0.5, ax_main.get_ylim()[1],
                 "  Forecast", color="#aaa", fontsize=8, va="top")

    # 預測歗殄段淡色背景
    ax_main.axvspan(show_hist - 0.5, n_total - 0.5,
                    color="#ffffff", alpha=0.025)

    # Volume nodes
    if volume_nodes:
        for node in volume_nodes:
            ax_main.axhline(node, color="orange", lw=0.7, alpha=0.25, ls=":")

    # 值願題色調補丁
    ax_main.title.set_color("white")
    ax_main.title.set_fontsize(9)
    fig.patch.set_facecolor(DARK)

    # 对齊円魔
    for ax in axes:
        ax.tick_params(colors="#888")
        ax.yaxis.label.set_color("#888")
        ax.xaxis.label.set_color("#888")

    # 如果有偏差面板
    if actual_deviation is not None and len(axes) >= 3:
        ax_dev = axes[2]
        ax_dev.set_facecolor(DARK)
        x_dev  = np.arange(show_hist, show_hist + len(actual_deviation))
        colors = ["#66ff66" if v >= 0 else "#ff6666" for v in actual_deviation]
        ax_dev.bar(x_dev, actual_deviation, color=colors, alpha=0.75, width=0.8)
        ax_dev.axhline(0, color="#555", lw=0.8)
        ax_dev.set_ylabel("Actual\u2212Median (%)", color="#888", fontsize=8)
        ax_dev.tick_params(colors="#888")
        ax_dev.set_facecolor(DARK)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK)

    return fig
