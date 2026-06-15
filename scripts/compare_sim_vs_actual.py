"""
compare_sim_vs_actual.py
讀取模擬輸出的 CSV，和真實 OHLC（從 yfinance 拉或另一個 CSV）並排畫雙軸 K 棒圖。
設計目的：讓 AI / 工程師可以不用上傳截圖，直接貼 CSV 就能目視比較誤差。

Usage:
    # 自動從 yfinance 拉真實數據
    python scripts/compare_sim_vs_actual.py --sim results/AAPL_sim.csv --symbol AAPL

    # 兩個 CSV 都指定（離線 / 不依賴網路）
    python scripts/compare_sim_vs_actual.py \
        --sim results/AAPL_sim.csv \
        --actual cache/AAPL_ohlcv.parquet   # parquet 也可以

    # 只看特定日期區間
    python scripts/compare_sim_vs_actual.py \
        --sim results/AAPL_sim.csv --symbol AAPL \
        --start 2025-09-01 --end 2025-12-31 \
        --output results/compare_AAPL.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DARK = "#0e0e0e"
COL_UP   = "#26a69a"
COL_DOWN = "#ef5350"
COL_SIM_UP   = "#ff9800"
COL_SIM_DOWN = "#e64a19"


# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────

def load_ohlcv(path: str) -> pd.DataFrame:
    """支援 .csv / .parquet，統一回傳 Date + OHLCV DataFrame (DatetimeIndex)。"""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # 統一欄位大小寫
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("date", "datetime", "time", "index"):
            rename[c] = "Date"
        elif cl == "open":
            rename[c] = "Open"
        elif cl == "high":
            rename[c] = "High"
        elif cl == "low":
            rename[c] = "Low"
        elif cl == "close":
            rename[c] = "Close"
        elif cl == "volume":
            rename[c] = "Volume"
    df = df.rename(columns=rename)

    if "Date" not in df.columns:
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance 未安裝，請 pip install yfinance 或改用 --actual 指定 CSV")

    raw = yf.download(symbol, start=start, end=end,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()


# ──────────────────────────────────────────────
# 繪圖
# ──────────────────────────────────────────────

def draw_candles(ax, df: pd.DataFrame, x_offset: int = 0,
                 alpha: float = 1.0,
                 up_col: str = COL_UP, down_col: str = COL_DOWN,
                 width: float = 0.35):
    for i, row in df.reset_index(drop=True).iterrows():
        x = i + x_offset
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        col = up_col if c >= o else down_col
        ax.plot([x, x], [l, h], color=col, lw=0.9, alpha=alpha)
        rect_y = min(o, c)
        rect_h = max(abs(c - o), (h - l) * 0.01)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - width, rect_y), width * 2, rect_h,
            boxstyle="square,pad=0",
            linewidth=0, facecolor=col, alpha=alpha,
        ))


def add_error_bars(ax, sim_df: pd.DataFrame, act_df: pd.DataFrame,
                   x_offset: int = 0):
    """在每根 K 棒之間畫一條線，連接模擬 Close 與真實 Close，標示誤差大小。"""
    n = min(len(sim_df), len(act_df))
    for i in range(n):
        x = i + x_offset
        sc = float(sim_df.iloc[i]["Close"])
        ac = float(act_df.iloc[i]["Close"])
        err = sc - ac
        col = "#ff4444" if abs(err) > 2 else "#ffcc00" if abs(err) > 0.5 else "#88ff88"
        ax.annotate(
            "", xy=(x, sc), xytext=(x, ac),
            arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.6),
        )


def compute_metrics(sim_df: pd.DataFrame, act_df: pd.DataFrame) -> dict:
    n = min(len(sim_df), len(act_df))
    s = sim_df.iloc[:n]["Close"].values.astype(float)
    a = act_df.iloc[:n]["Close"].values.astype(float)
    mae  = np.mean(np.abs(s - a))
    rmse = np.sqrt(np.mean((s - a) ** 2))
    mape = np.mean(np.abs((s - a) / np.where(a == 0, 1e-9, a))) * 100

    # 方向命中（v2 定義：基準是前一根「真實」收盤）
    hits = 0
    valid = 0
    for i in range(1, n):
        p_ret = s[i] - a[i - 1]
        a_ret = a[i] - a[i - 1]
        if p_ret != 0 and a_ret != 0:
            hits  += int(np.sign(p_ret) == np.sign(a_ret))
            valid += 1
    dir_acc = hits / valid * 100 if valid > 0 else float("nan")

    return dict(MAE=mae, RMSE=rmse, MAPE=mape, DirAcc=dir_acc,
                n=n, n_dir=valid)


def plot_comparison(sim_df: pd.DataFrame, act_df: pd.DataFrame,
                    symbol: str, output: str):
    n = min(len(sim_df), len(act_df))
    sim_df = sim_df.iloc[:n].reset_index(drop=True)
    act_df = act_df.iloc[:n].reset_index(drop=True)

    metrics = compute_metrics(sim_df, act_df)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        3, 1, figsize=(max(16, n * 0.22), 12),
        gridspec_kw={"height_ratios": [5, 2, 1.2], "hspace": 0.06},
    )
    fig.patch.set_facecolor(DARK)

    ax_main, ax_err, ax_vol = axes

    # ── 主圖：真實 K 棒（後面，半透明） ──
    draw_candles(ax_main, act_df, alpha=0.85,
                 up_col=COL_UP, down_col=COL_DOWN)

    # ── 主圖：模擬 K 棒（前面，橘色系） ──
    draw_candles(ax_main, sim_df, alpha=0.75,
                 up_col=COL_SIM_UP, down_col=COL_SIM_DOWN)

    # ── 誤差連線 ──
    add_error_bars(ax_main, sim_df, act_df)

    # ── 日期 x 軸標籤（每 5 根一個）──
    step = max(1, n // 20)
    xticks = list(range(0, n, step))
    xlabels = [str(act_df.iloc[i]["Date"])[:10] for i in xticks]
    ax_main.set_xticks(xticks)
    ax_main.set_xticklabels(xlabels, rotation=45, ha="right",
                             fontsize=7, color="#aaa")

    # 圖例
    legend_items = [
        mpatches.Patch(color=COL_UP,      label="Actual ↑"),
        mpatches.Patch(color=COL_DOWN,    label="Actual ↓"),
        mpatches.Patch(color=COL_SIM_UP,  label="Sim ↑"),
        mpatches.Patch(color=COL_SIM_DOWN,label="Sim ↓"),
    ]
    ax_main.legend(handles=legend_items, loc="upper left",
                   fontsize=7.5, facecolor="#1a1a1a",
                   edgecolor="#333", labelcolor="white")

    title_str = (
        f"{symbol}  Sim vs Actual  |  "
        f"MAE={metrics['MAE']:.3f}  RMSE={metrics['RMSE']:.3f}  "
        f"MAPE={metrics['MAPE']:.2f}%  DirAcc={metrics['DirAcc']:.1f}%  "
        f"({metrics['n']} bars)"
    )
    ax_main.set_title(title_str, color="white", fontsize=9, pad=6)
    ax_main.set_facecolor(DARK)
    ax_main.set_xlim(-1, n + 0.5)
    ax_main.tick_params(colors="#888", labelsize=8)

    # ── 誤差子圖 (Close 差值) ──
    close_err = sim_df["Close"].values - act_df["Close"].values
    bar_colors = [("#26a69a" if e >= 0 else "#ef5350") for e in close_err]
    ax_err.bar(range(n), close_err, color=bar_colors, alpha=0.75, width=0.6)
    ax_err.axhline(0, color="#555", lw=0.8)
    ax_err.set_ylabel("Close Δ (Sim−Real)", color="#aaa", fontsize=7.5)
    ax_err.set_facecolor(DARK)
    ax_err.tick_params(colors="#888", labelsize=7)
    ax_err.set_xlim(-1, n + 0.5)
    ax_err.set_xticks(xticks)
    ax_err.set_xticklabels(xlabels, rotation=45, ha="right",
                            fontsize=6.5, color="#aaa")

    # ── 成交量子圖 ──
    if "Volume" in act_df.columns:
        vcols = [COL_UP if c >= o else COL_DOWN
                 for o, c in zip(act_df["Open"], act_df["Close"])]
        ax_vol.bar(range(n), act_df["Volume"].values,
                   color=vcols, alpha=0.5, width=0.7)
    ax_vol.set_facecolor(DARK)
    ax_vol.tick_params(colors="#888", labelsize=6.5)
    ax_vol.set_xlim(-1, n + 0.5)
    ax_vol.set_ylabel("Volume", color="#aaa", fontsize=7)
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v:.0f}")
    )

    # ── 數字指標 ──
    stats_txt = (
        f"MAE: {metrics['MAE']:.4f}    "
        f"RMSE: {metrics['RMSE']:.4f}    "
        f"MAPE: {metrics['MAPE']:.3f}%    "
        f"Direction Accuracy: {metrics['DirAcc']:.1f}%  ({metrics['n_dir']} valid bars)"
    )
    fig.text(0.5, 0.01, stats_txt, ha="center", va="bottom",
             fontsize=8.5, color="#cccccc")

    # ── 儲存 ──
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare simulation CSV vs real OHLC — draw dual candlestick chart."
    )
    p.add_argument("--sim",    required=True,
                   help="模擬輸出 CSV，需含 Date/Open/High/Low/Close 欄位")
    p.add_argument("--actual", default=None,
                   help="真實 OHLC CSV 或 parquet；不給則從 yfinance 拉")
    p.add_argument("--symbol", default="AAPL",
                   help="股票代號（--actual 未指定時用來拉 yfinance）")
    p.add_argument("--start",  default=None,
                   help="過濾起始日 YYYY-MM-DD（可省略）")
    p.add_argument("--end",    default=None,
                   help="過濾結束日 YYYY-MM-DD（可省略）")
    p.add_argument("--output", default=None,
                   help="輸出圖片路徑；預設 results/compare_{symbol}.png")
    return p.parse_args()


def main():
    args = parse_args()

    # 讀模擬
    print(f"Loading sim: {args.sim}")
    sim_df = load_ohlcv(args.sim)

    # 決定日期範圍
    start_dt = args.start or str(sim_df["Date"].min())[:10]
    end_dt   = args.end   or str(sim_df["Date"].max())[:10]

    # 讀真實數據
    if args.actual:
        print(f"Loading actual: {args.actual}")
        act_df = load_ohlcv(args.actual)
    else:
        print(f"Fetching {args.symbol} from yfinance ({start_dt} ~ {end_dt})...")
        # 多拉幾天避免收盤日對不上
        act_start = pd.Timestamp(start_dt) - pd.Timedelta(days=5)
        act_end   = pd.Timestamp(end_dt)   + pd.Timedelta(days=5)
        act_df = fetch_yfinance(args.symbol,
                                str(act_start)[:10],
                                str(act_end)[:10])

    # 日期過濾
    mask_sim = (sim_df["Date"] >= start_dt) & (sim_df["Date"] <= end_dt)
    mask_act = (act_df["Date"] >= start_dt) & (act_df["Date"] <= end_dt)
    sim_df = sim_df[mask_sim].reset_index(drop=True)
    act_df = act_df[mask_act].reset_index(drop=True)

    # 對齊日期（以兩者的交集為準）
    sim_dates = set(sim_df["Date"].dt.date)
    act_dates = set(act_df["Date"].dt.date)
    common = sorted(sim_dates & act_dates)
    if not common:
        raise ValueError(
            f"模擬與真實數據沒有重疊日期！\n"
            f"  sim: {sim_df['Date'].min()} ~ {sim_df['Date'].max()}  ({len(sim_df)} rows)\n"
            f"  act: {act_df['Date'].min()} ~ {act_df['Date'].max()}  ({len(act_df)} rows)"
        )

    common_dt = pd.to_datetime(list(common))
    sim_df = sim_df[sim_df["Date"].isin(common_dt)].reset_index(drop=True)
    act_df = act_df[act_df["Date"].isin(common_dt)].reset_index(drop=True)

    print(f"Aligned bars: {len(common)}  ({common[0]} ~ {common[-1]})")

    out = args.output or f"results/compare_{args.symbol}.png"
    plot_comparison(sim_df, act_df, symbol=args.symbol, output=out)

    # 順便把逐日誤差表印出來
    n = min(len(sim_df), len(act_df))
    rows = []
    for i in range(n):
        s, a = sim_df.iloc[i], act_df.iloc[i]
        rows.append({
            "Date":        str(s["Date"])[:10],
            "Sim_Close":   round(float(s["Close"]), 4),
            "Act_Close":   round(float(a["Close"]), 4),
            "Delta":       round(float(s["Close"]) - float(a["Close"]), 4),
            "Sim_H":       round(float(s["High"]), 4),
            "Act_H":       round(float(a["High"]), 4),
            "Sim_L":       round(float(s["Low"]),  4),
            "Act_L":       round(float(a["Low"]),  4),
        })
    err_df = pd.DataFrame(rows)
    print("\n── Per-bar Error Table ──")
    print(err_df.to_string(index=False))

    out_csv = Path(out).with_suffix(".csv")
    err_df.to_csv(out_csv, index=False)
    print(f"\nError table saved: {out_csv}")


if __name__ == "__main__":
    main()
