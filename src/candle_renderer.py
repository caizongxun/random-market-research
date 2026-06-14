"""
candle_renderer.py  v9

改動：
  - 預測 K 棒改為黃金色（漲/跌用不同深淺區分），與历史 K 棒明顯區隔
  - 中位數路徑改為麮黃線（不再用白色，避免與實際走勢混淡）
  - 實際走勢線保留白色
  - 預測區背景加深
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

COLOR_HIST_UP    = "#26a69a"
COLOR_HIST_DN    = "#ef5350"
COLOR_FWD_UP     = "#ffd54f"
COLOR_FWD_DN     = "#ffab40"
COLOR_MEDIAN     = "#ffe600"
COLOR_ACTUAL     = "#ffffff"
COLOR_BAND_25_75 = "#00e5ff"
BAND_ALPHA_INNER = 0.18
BAND_ALPHA_OUTER = 0.08


def _make_ohlcv_df(opens, highs, lows, closes, volumes, start_idx=0):
    n   = len(closes)
    idx = pd.date_range(start="2020-01-01", periods=start_idx + n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": closes, "Volume": volumes},
        index=idx[start_idx:],
    )


def _draw_forecast_candles(
    ax, x_start, fwd_open, fwd_high, fwd_low, fwd_close,
    color_up=COLOR_FWD_UP, color_dn=COLOR_FWD_DN,
    alpha_body=0.85, alpha_wick=0.7, lw_wick=0.9,
):
    """mplfinance 不支援多區段不同顏色，用 patches 手畫黃金 K 棒。"""
    for i, (o, h, l, c) in enumerate(zip(fwd_open, fwd_high, fwd_low, fwd_close)):
        x   = x_start + i
        clr = color_up if c >= o else color_dn
        ax.plot([x, x], [l, h], color=clr, lw=lw_wick, alpha=alpha_wick,
                solid_capstyle="round", zorder=3)
        body_bot = min(o, c)
        body_h   = max(abs(c - o), (h - l) * 0.01)
        rect = mpatches.FancyBboxPatch(
            (x - 0.35, body_bot), 0.70, body_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=clr, alpha=alpha_body, zorder=4,
        )
        ax.add_patch(rect)


def render_forecast_candles(
    hist_open, hist_high, hist_low, hist_close, hist_volume,
    fwd_open, fwd_high, fwd_low, fwd_close, fwd_volume,
    p25, p75, p10, p90,
    median_path: Optional[np.ndarray] = None,
    actual_open=None, actual_high=None,
    actual_low=None,  actual_close=None,
    actual_volume=None,
    title: str = "Forecast Candles",
    output_path: Optional[Path] = None,
    volume_nodes: Optional[list] = None,
    hist_window: int = 60,
    actual_deviation: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
) -> plt.Figure:

    T_hist     = len(hist_close)
    T_fwd      = len(fwd_close)
    has_actual = actual_close is not None and len(actual_close) > 0
    show_hist  = min(hist_window, T_hist)
    sl         = slice(T_hist - show_hist, T_hist)

    h_o = hist_open[sl];  h_h = hist_high[sl]
    h_l = hist_low[sl];   h_c = hist_close[sl]
    h_v = hist_volume[sl]

    n_total  = show_hist + T_fwd
    idx_full = pd.date_range(start="2020-01-01", periods=n_total, freq="B")

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

    mc = mpf.make_marketcolors(
        up=COLOR_HIST_UP, down=COLOR_HIST_DN,
        wick={"up": COLOR_HIST_UP, "down": COLOR_HIST_DN},
        volume={"up": COLOR_HIST_UP + "88", "down": COLOR_HIST_DN + "88"},
        edge="inherit",
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor=DARK, edgecolor="#333", figcolor=DARK,
        gridcolor="#1e1e1e", gridstyle="-", y_on_right=True,
    )

    def _full(arr):
        out = np.full(n_total, np.nan)
        out[show_hist:show_hist + len(arr)] = arr
        return out

    ap_list = []

    p25_f = _full(p25); p75_f = _full(p75)
    p10_f = _full(p10); p90_f = _full(p90)
    ap_list.append(mpf.make_addplot(
        p75_f, color=COLOR_BAND_25_75, alpha=0.5, width=1.2,
        fill_between={"y1": p25_f, "y2": p75_f,
                      "color": COLOR_BAND_25_75, "alpha": BAND_ALPHA_INNER}))
    ap_list.append(mpf.make_addplot(
        p90_f, color=COLOR_BAND_25_75, alpha=0.25, width=0.8,
        fill_between={"y1": p10_f, "y2": p90_f,
                      "color": COLOR_BAND_25_75, "alpha": BAND_ALPHA_OUTER}))
    ap_list.append(mpf.make_addplot(p25_f, color=COLOR_BAND_25_75, alpha=0.5, width=1.2))
    ap_list.append(mpf.make_addplot(p10_f, color=COLOR_BAND_25_75, alpha=0.25, width=0.8))

    # 麮黃中位數線（不加 zorder，mpf 不支援）
    if median_path is not None:
        ap_list.append(mpf.make_addplot(
            _full(median_path), color=COLOR_MEDIAN, width=2.2, alpha=0.92))

    # 白色實際線
    if has_actual:
        n_act = min(len(actual_close), T_fwd)
        ap_list.append(mpf.make_addplot(
            _full(actual_close[:n_act]), color=COLOR_ACTUAL, width=1.8, alpha=0.9))

    fig, axes = mpf.plot(
        df_all, type="candle", style=style,
        addplot=ap_list, volume=True, title=title,
        returnfig=True, figsize=(18, 10),
        tight_layout=False, warn_too_much_data=9999,
    )

    ax_main = axes[0]

    # 預測區深色背景
    ax_main.axvspan(show_hist - 0.5, n_total - 0.5,
                    color="#1a1200", alpha=0.45, zorder=0)
    ax_main.axvline(show_hist - 0.5, color="#888", lw=1.2, ls="-", alpha=0.8)
    ax_main.text(show_hist - 0.5, ax_main.get_ylim()[1],
                 "  Forecast", color="#aaa", fontsize=8, va="top")

    # 手畫黃金 K 棒
    _draw_forecast_candles(
        ax_main, show_hist,
        fwd_open, fwd_high, fwd_low, fwd_close,
    )

    if volume_nodes:
        for node in volume_nodes:
            ax_main.axhline(node, color="orange", lw=0.7, alpha=0.25, ls=":")

    ax_main.title.set_color("white")
    ax_main.title.set_fontsize(9)
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        ax.tick_params(colors="#888")
        ax.yaxis.label.set_color("#888")
        ax.xaxis.label.set_color("#888")

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
