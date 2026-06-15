"""
analyze_fear_threshold.py
=========================
分析歷史數據中「高位恐懼心理」的統計特徵，
輸出動態 fear_threshold 的合理初始值與計算方法。

核心邏輯
--------
1. 掃描歷史，找出所有「上漲段」（swing up）
2. 對每段上漲段，記錄：
   - 最大累積漲幅（swing_gain）
   - 漲幅超過各閾值後的後續 N 天平均報酬
   - 漲幅超過各閾值後的波動率放大倍數
   - 漲幅超過閾值後見頂（5日內下跌超過 X%）的機率
3. 用 rolling window 觀察 fear_threshold 如何隨時間漂移
4. 輸出：
   - 靜態建議閾值（全樣本分位數）
   - 動態閾值曲線（rolling 252天窗口）
   - 每個閾值對應的「見頂機率 / drift衰減倍數 / vol放大」

用法
----
  python scripts/analyze_fear_threshold.py --symbol AAPL
  python scripts/analyze_fear_threshold.py --symbol AAPL --swing-method zigzag --pct-drop 0.03
  python scripts/analyze_fear_threshold.py --symbol AAPL MSFT TSLA --output-dir results/fear
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 下載並整理資料
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(symbol: str, years: int = 10) -> pd.DataFrame:
    end   = pd.Timestamp.today()
    start = end - pd.DateOffset(years=years)
    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    if "Date" not in raw.columns and "Datetime" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "Date"})
    date_col = "Date" if "Date" in raw.columns else "Datetime"
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.rename(columns={date_col: "Date"})
    raw = raw.dropna(subset=["Close"])
    return raw.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 找出上漲段（swing identification）
# ─────────────────────────────────────────────────────────────────────────────

def find_swing_ups(
    close: np.ndarray,
    method: str = "zigzag",
    pct_drop: float = 0.03,
    min_gain: float = 0.02,
) -> list[dict]:
    """
    回傳所有上漲段：
      [{"start_idx": i, "peak_idx": j, "gain": float, "bars": int}, ...]

    method='zigzag': 用 zigzag 演算法（pct_drop 為反轉門檻）
    method='rolling': 滾動 N 天高點法（較粗略）
    """
    swings = []

    if method == "zigzag":
        # 簡單 zigzag：找出所有局部高低點序列
        direction = None   # "up" or "down"
        seg_start = 0
        seg_start_price = close[0]
        peak_idx = 0
        peak_price = close[0]

        for i in range(1, len(close)):
            p = close[i]
            if direction is None:
                if p > seg_start_price * (1 + pct_drop / 2):
                    direction = "up"
                    peak_idx = i
                    peak_price = p
                elif p < seg_start_price * (1 - pct_drop / 2):
                    direction = "down"
            elif direction == "up":
                if p > peak_price:
                    peak_idx = i
                    peak_price = p
                elif p < peak_price * (1 - pct_drop):
                    gain = (peak_price - seg_start_price) / seg_start_price
                    if gain >= min_gain:
                        swings.append({
                            "start_idx":   seg_start,
                            "peak_idx":    peak_idx,
                            "start_price": float(seg_start_price),
                            "peak_price":  float(peak_price),
                            "gain":        float(gain),
                            "bars":        peak_idx - seg_start,
                        })
                    seg_start = peak_idx
                    seg_start_price = peak_price
                    direction = "down"
                    peak_idx = i
                    peak_price = p
            elif direction == "down":
                if p < peak_price:
                    peak_idx = i
                    peak_price = p
                elif p > peak_price * (1 + pct_drop):
                    seg_start = peak_idx
                    seg_start_price = peak_price
                    direction = "up"
                    peak_idx = i
                    peak_price = p

        # 最後一段如果是上漲
        if direction == "up":
            gain = (peak_price - seg_start_price) / seg_start_price
            if gain >= min_gain:
                swings.append({
                    "start_idx":   seg_start,
                    "peak_idx":    peak_idx,
                    "start_price": float(seg_start_price),
                    "peak_price":  float(peak_price),
                    "gain":        float(gain),
                    "bars":        peak_idx - seg_start,
                })

    return swings


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 對每個閾值計算統計
# ─────────────────────────────────────────────────────────────────────────────

def compute_threshold_stats(
    close: np.ndarray,
    swings: list[dict],
    thresholds: list[float],
    lookahead: int = 10,
    peak_drop_pct: float = 0.03,
) -> pd.DataFrame:
    """
    對每個閾值，統計：
    - n_events: 累漲超過閾值的事件數
    - peak_prob: 事件後 lookahead 根內見頂（下跌 peak_drop_pct）的機率
    - avg_next_ret: 事件後 lookahead 根平均報酬（%/day）
    - vol_ratio: 事件後 lookahead 根的 σ / 全樣本 σ
    - drift_decay: 事件後斜率衰減（每日漲幅縮減比例）
    """
    global_vol = float(np.std(np.diff(np.log(close + 1e-8))))

    rows = []
    for thr in thresholds:
        events = []
        for sw in swings:
            s_i  = sw["start_idx"]
            pk_i = sw["peak_idx"]
            s_p  = sw["start_price"]
            # 找出在上漲段中累漲首次超過 thr 的那根
            for idx in range(s_i, pk_i + 1):
                cum_gain = (close[idx] - s_p) / s_p
                if cum_gain >= thr:
                    events.append({"trigger_idx": idx, "swing": sw})
                    break

        if not events:
            rows.append({
                "threshold":    thr,
                "n_events":     0,
                "peak_prob":    np.nan,
                "avg_next_ret": np.nan,
                "vol_ratio":    np.nan,
                "drift_decay":  np.nan,
            })
            continue

        peak_hits   = 0
        next_rets   = []
        vol_ratios  = []
        drift_decays= []

        for ev in events:
            t = ev["trigger_idx"]
            end_i = min(t + lookahead, len(close) - 1)
            if end_i <= t:
                continue

            window = close[t: end_i + 1]
            log_r  = np.diff(np.log(window))

            # 見頂：窗口內最大回撤超過 peak_drop_pct
            peak_in_window = np.max(window)
            min_after_peak = np.min(window[np.argmax(window):])
            drawdown = (min_after_peak - peak_in_window) / peak_in_window
            if drawdown <= -peak_drop_pct:
                peak_hits += 1

            # 平均日報酬
            avg_r = float(np.mean(log_r)) * 100 if len(log_r) > 0 else 0.0
            next_rets.append(avg_r)

            # 波動率比
            local_vol = float(np.std(log_r)) if len(log_r) > 1 else global_vol
            vol_ratios.append(local_vol / (global_vol + 1e-8))

            # drift 衰減：比較窗口前半 vs 後半平均報酬
            half = max(1, len(log_r) // 2)
            first_half = float(np.mean(log_r[:half])) if half > 0 else 0.0
            second_half= float(np.mean(log_r[half:])) if half > 0 else 0.0
            decay = second_half / (abs(first_half) + 1e-6)  # < 1 代表在衰減
            drift_decays.append(decay)

        rows.append({
            "threshold":    thr,
            "n_events":     len(events),
            "peak_prob":    round(peak_hits / len(events), 4) if events else np.nan,
            "avg_next_ret": round(float(np.mean(next_rets)), 4) if next_rets else np.nan,
            "vol_ratio":    round(float(np.mean(vol_ratios)), 4) if vol_ratios else np.nan,
            "drift_decay":  round(float(np.mean(drift_decays)), 4) if drift_decays else np.nan,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: 動態 fear_threshold（rolling window）
# ─────────────────────────────────────────────────────────────────────────────

def compute_dynamic_threshold(
    close: np.ndarray,
    dates: pd.Series,
    window: int = 252,
    pct_drop: float = 0.03,
    min_gain: float = 0.02,
    target_peak_prob: float = 0.55,
    thresholds: list[float] | None = None,
    lookahead: int = 10,
) -> pd.DataFrame:
    """
    rolling window 計算：每個時間點往前看 window 根，
    找出使「見頂機率 >= target_peak_prob」的最小累漲閾值。
    這就是當前的「動態 fear_threshold」。
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.02, 0.25, 0.01)]

    records = []
    step_size = max(1, window // 20)  # 每隔 step_size 根計算一次（加速）

    for end_i in range(window, len(close), step_size):
        win_close = close[max(0, end_i - window): end_i]
        win_swings = find_swing_ups(win_close, pct_drop=pct_drop, min_gain=min_gain)

        if not win_swings:
            records.append({
                "date":            dates.iloc[end_i],
                "fear_threshold":  np.nan,
                "n_swings":        0,
            })
            continue

        # 找最小閾值使 peak_prob >= target_peak_prob
        stats = compute_threshold_stats(
            win_close, win_swings, thresholds,
            lookahead=lookahead, peak_drop_pct=0.03,
        )
        valid = stats[stats["peak_prob"] >= target_peak_prob]
        if valid.empty:
            best_thr = float(stats.loc[stats["peak_prob"].idxmax(), "threshold"]) if not stats["peak_prob"].isna().all() else np.nan
        else:
            best_thr = float(valid["threshold"].min())

        records.append({
            "date":            dates.iloc[end_i],
            "fear_threshold":  round(best_thr, 4) if not np.isnan(best_thr) else np.nan,
            "n_swings":        len(win_swings),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 從動態曲線算出「當前建議閾值」
# ─────────────────────────────────────────────────────────────────────────────

def current_fear_threshold(
    dynamic_df: pd.DataFrame,
    recent_window: int = 63,   # 約 3 個月
    method: str = "median",    # "median" / "ewm" / "p25"
) -> float:
    """
    從動態曲線的最近 recent_window 個點估計「現在」的 fear_threshold。
    method:
      median  → 穩健，不受極端值影響
      ewm     → 指數加權，更偏重最近
      p25     → 用 25 分位（保守，提早觸發）
    """
    recent = dynamic_df["fear_threshold"].dropna().tail(recent_window)
    if recent.empty:
        return 0.07  # fallback
    if method == "median":
        return float(recent.median())
    elif method == "ewm":
        return float(recent.ewm(span=recent_window // 3).mean().iloc[-1])
    elif method == "p25":
        return float(recent.quantile(0.25))
    return float(recent.median())


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: 畫圖
# ─────────────────────────────────────────────────────────────────────────────

def render_analysis(
    symbol: str,
    close: np.ndarray,
    dates: pd.Series,
    swings: list[dict],
    stats_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    cur_threshold: float,
    output_dir: str,
):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax_price   = fig.add_subplot(gs[0, :])
    ax_peak    = fig.add_subplot(gs[1, 0])
    ax_vol     = fig.add_subplot(gs[1, 1])
    ax_dynamic = fig.add_subplot(gs[2, :])

    # ── 價格 + 上漲段標注 ──────────────────────────────────────
    ax_price.plot(dates.values, close, color="#555", lw=0.9, label="Close")
    for sw in swings:
        si = sw["start_idx"]
        pi = sw["peak_idx"]
        ax_price.axvspan(
            dates.iloc[si], dates.iloc[pi],
            alpha=0.12, color="#26A69A",
        )
        ax_price.annotate(
            f"+{sw['gain']*100:.1f}%",
            xy=(dates.iloc[pi], close[pi]),
            fontsize=6, color="#1B5E20", va="bottom",
            xytext=(0, 4), textcoords="offset points",
        )
    ax_price.set_title(f"{symbol} — 歷史收盤 + 上漲段（綠色區域）", fontsize=11)
    ax_price.set_ylabel("價格")
    ax_price.grid(True, alpha=0.25)

    # ── 各閾值 → 見頂機率 ────────────────────────────────────
    valid = stats_df.dropna(subset=["peak_prob"])
    ax_peak.bar(
        valid["threshold"] * 100,
        valid["peak_prob"] * 100,
        width=0.8, color="#EF5350", alpha=0.8,
    )
    ax_peak.axhline(55, color="#888", lw=1, ls="--", label="55%基準線")
    ax_peak.axvline(cur_threshold * 100, color="#1A237E", lw=1.5, ls="-.",
                    label=f"建議閾值={cur_threshold*100:.1f}%")
    ax_peak.set_xlabel("累積漲幅閾值 (%)")
    ax_peak.set_ylabel("見頂機率 (%)")
    ax_peak.set_title("各閾值對應見頂機率（10日內）")
    ax_peak.legend(fontsize=7)
    ax_peak.grid(True, alpha=0.25)

    # ── 各閾值 → 波動率放大倍數 ──────────────────────────────
    valid2 = stats_df.dropna(subset=["vol_ratio"])
    ax_vol.plot(
        valid2["threshold"] * 100,
        valid2["vol_ratio"],
        marker="o", markersize=4,
        color="#7B1FA2", lw=1.5,
    )
    ax_vol.axhline(1.0, color="#888", lw=1, ls="--", label="基準（無放大）")
    ax_vol.axvline(cur_threshold * 100, color="#1A237E", lw=1.5, ls="-.",
                   label=f"建議閾值={cur_threshold*100:.1f}%")
    ax_vol.set_xlabel("累積漲幅閾值 (%)")
    ax_vol.set_ylabel("局部 σ / 全樣本 σ")
    ax_vol.set_title("各閾值對應波動率放大倍數")
    ax_vol.legend(fontsize=7)
    ax_vol.grid(True, alpha=0.25)

    # ── 動態閾值時間序列 ─────────────────────────────────────
    dyn_valid = dynamic_df.dropna(subset=["fear_threshold"])
    ax_dynamic.plot(
        dyn_valid["date"], dyn_valid["fear_threshold"] * 100,
        color="#FF7043", lw=1.4, label="rolling fear_threshold",
    )
    # EWM 平滑線
    ewm_vals = dyn_valid["fear_threshold"].ewm(span=10).mean() * 100
    ax_dynamic.plot(
        dyn_valid["date"], ewm_vals,
        color="#1565C0", lw=1.8, ls="--", label="EWM 平滑",
    )
    ax_dynamic.axhline(cur_threshold * 100, color="#1A237E", lw=1.2, ls="-.",
                       label=f"當前建議={cur_threshold*100:.1f}%")
    ax_dynamic.set_title("動態 fear_threshold 曲線（rolling 252天）", fontsize=11)
    ax_dynamic.set_ylabel("fear_threshold (%)")
    ax_dynamic.legend(fontsize=8)
    ax_dynamic.grid(True, alpha=0.25)

    fig.suptitle(
        f"{symbol} — High-Fear Threshold Analysis\n"
        f"建議初始值: {cur_threshold*100:.1f}%  "
        f"（3個月 median rolling，target peak_prob≥55%）",
        fontsize=12, y=1.01,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"{symbol}_fear_threshold.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ 圖表 → {fpath}")
    return str(fpath)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: 輸出 JSON（給 rolling_forward 使用）
# ─────────────────────────────────────────────────────────────────────────────

def build_fear_profile(
    symbol: str,
    stats_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    cur_threshold: float,
    output_dir: str,
) -> dict:
    """
    輸出結構化 JSON，後續可直接被 rolling_forward / forward_study 載入。

    {
      "symbol": "AAPL",
      "fear_threshold": 0.072,         ← 建議初始值
      "fear_vol_mult": 1.35,            ← 高位波動率放大倍數
      "fear_drift_decay": 0.6,          ← drift 衰減速率（高位）
      "reversal_prob_per_bar": 0.08,    ← 每根 K 棒見頂機率（超過閾值後）
      "threshold_by_regime": {          ← 不同市場狀態下的建議值
        "low_vol": 0.09,
        "high_vol": 0.05,
      },
      "dynamic_history": [...]          ← 動態閾值歷史（可用於再估計）
    }
    """
    # 在建議閾值附近取 vol_ratio 和 peak_prob
    near = stats_df[
        (stats_df["threshold"] >= cur_threshold - 0.01) &
        (stats_df["threshold"] <= cur_threshold + 0.02)
    ]
    vol_mult = float(near["vol_ratio"].mean()) if not near.empty else 1.3
    peak_prob= float(near["peak_prob"].mean()) if not near.empty else 0.55
    drift_decay_raw = float(near["drift_decay"].mean()) if not near.empty else 0.5

    # drift_decay < 1 代表高位斜率衰減，轉成比例
    fear_drift_decay = float(np.clip(1.0 - drift_decay_raw, 0.1, 0.9))

    # 每根見頂機率 ≈ peak_prob / lookahead
    reversal_prob = float(np.clip(peak_prob / 10, 0.02, 0.25))

    # 低波動 / 高波動下的閾值（分位數）
    dyn_vals = dynamic_df["fear_threshold"].dropna()
    low_vol_thr  = float(dyn_vals.quantile(0.75)) if len(dyn_vals) >= 4 else cur_threshold * 1.2
    high_vol_thr = float(dyn_vals.quantile(0.25)) if len(dyn_vals) >= 4 else cur_threshold * 0.8

    profile = {
        "symbol":                symbol,
        "fear_threshold":        round(cur_threshold, 4),
        "fear_vol_mult":         round(vol_mult, 3),
        "fear_drift_decay_rate": round(fear_drift_decay, 3),
        "reversal_prob_per_bar": round(reversal_prob, 4),
        "threshold_by_regime": {
            "low_vol":  round(low_vol_thr, 4),
            "high_vol": round(high_vol_thr, 4),
        },
        "stats_table": stats_df.to_dict(orient="records"),
        "dynamic_history": [
            {
                "date": str(r["date"])[:10],
                "fear_threshold": round(r["fear_threshold"], 4)
                if not pd.isna(r["fear_threshold"]) else None,
            }
            for _, r in dynamic_df.iterrows()
        ],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / f"{symbol}_fear_profile.json"
    with open(jpath, "w") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"✔ JSON → {jpath}")
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="分析高位恐懼閾值（fear_threshold），輸出動態建議值與統計圖"
    )
    p.add_argument("--symbol",          nargs="+", required=True,
                   help="股票代碼（可多個，如 AAPL MSFT TSLA）")
    p.add_argument("--years",           type=int,   default=10,
                   help="下載幾年歷史（預設 10）")
    p.add_argument("--swing-method",    default="zigzag",
                   choices=["zigzag"],
                   help="上漲段識別方法")
    p.add_argument("--pct-drop",        type=float, default=0.03,
                   help="Zigzag 反轉門檻（預設 3%%）")
    p.add_argument("--min-gain",        type=float, default=0.02,
                   help="最小有效上漲幅度（預設 2%%）")
    p.add_argument("--lookahead",       type=int,   default=10,
                   help="見頂判斷的前瞻天數（預設 10）")
    p.add_argument("--rolling-window",  type=int,   default=252,
                   help="動態閾值的 rolling 窗口（預設 252 根 = 1 年）")
    p.add_argument("--target-peak-prob",type=float, default=0.55,
                   help="動態閾值目標見頂機率（預設 0.55）")
    p.add_argument("--recent-window",   type=int,   default=63,
                   help="「當前」閾值取最近幾個點（預設 63 = 3 個月）")
    p.add_argument("--threshold-method",default="median",
                   choices=["median", "ewm", "p25"],
                   help="當前建議值的估計方法")
    p.add_argument("--output-dir",      default="results/fear",
                   help="輸出目錄")
    return p.parse_args()


def main():
    args = parse_args()

    thresholds = [round(x, 3) for x in np.arange(0.02, 0.26, 0.01)]

    for symbol in args.symbol:
        print(f"\n{'='*60}")
        print(f"  {symbol}  — 高位恐懼閾值分析")
        print(f"{'='*60}")

        # 1. 下載
        print(f"  下載 {symbol} {args.years} 年資料...")
        df = fetch_data(symbol, years=args.years)
        close = df["Close"].values.astype(float)
        dates = df["Date"]
        print(f"  共 {len(df)} 根 K 棒")

        # 2. 找上漲段
        swings = find_swing_ups(
            close,
            method=args.swing_method,
            pct_drop=args.pct_drop,
            min_gain=args.min_gain,
        )
        print(f"  找到 {len(swings)} 個上漲段（最小漲幅 {args.min_gain*100:.0f}%，"
              f"回撤門檻 {args.pct_drop*100:.0f}%）")
        if swings:
            gains = [sw["gain"] for sw in swings]
            print(f"  漲幅分布：min={min(gains)*100:.1f}%  "
                  f"median={np.median(gains)*100:.1f}%  "
                  f"max={max(gains)*100:.1f}%  "
                  f"p75={np.percentile(gains,75)*100:.1f}%")

        # 3. 全樣本閾值統計
        print(f"  計算各閾值統計...")
        stats_df = compute_threshold_stats(
            close, swings, thresholds,
            lookahead=args.lookahead,
        )

        # 4. 動態閾值
        print(f"  計算動態 fear_threshold（rolling {args.rolling_window}天）...")
        dynamic_df = compute_dynamic_threshold(
            close, dates,
            window=args.rolling_window,
            pct_drop=args.pct_drop,
            min_gain=args.min_gain,
            target_peak_prob=args.target_peak_prob,
            thresholds=thresholds,
            lookahead=args.lookahead,
        )

        # 5. 當前建議值
        cur_thr = current_fear_threshold(
            dynamic_df,
            recent_window=args.recent_window,
            method=args.threshold_method,
        )

        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  建議 fear_threshold = {cur_thr*100:.1f}%                 │")
        print(f"  │  （方法：{args.threshold_method}，近 {args.recent_window} 個窗口點）    │")

        # 顯示建議閾值附近的統計
        near = stats_df[
            (stats_df["threshold"] >= cur_thr - 0.01) &
            (stats_df["threshold"] <= cur_thr + 0.02)
        ]
        if not near.empty:
            print(f"  │  見頂機率：{near['peak_prob'].mean()*100:.1f}%                     │")
            print(f"  │  波動率放大：{near['vol_ratio'].mean():.2f}x               │")
            print(f"  │  drift 衰減係數：{near['drift_decay'].mean():.2f}           │")
        print(f"  └─────────────────────────────────────────────┘")

        # 顯示全部統計表
        print(f"\n  閾值統計表（全樣本）：")
        print(f"  {'閾值':>6}  {'事件數':>6}  {'見頂機率':>8}  "
              f"{'均日報酬':>9}  {'vol倍率':>7}  {'drift衰減':>8}")
        print(f"  {'─'*58}")
        for _, row in stats_df.iterrows():
            marker = " ← 建議" if abs(row["threshold"] - cur_thr) < 0.005 else ""
            print(f"  {row['threshold']*100:>5.0f}%  "
                  f"{row['n_events']:>6.0f}  "
                  f"{row['peak_prob']*100 if not np.isnan(row['peak_prob']) else 0:>7.1f}%  "
                  f"{row['avg_next_ret'] if not np.isnan(row['avg_next_ret']) else 0:>+9.3f}%  "
                  f"{row['vol_ratio'] if not np.isnan(row['vol_ratio']) else 0:>7.2f}x  "
                  f"{row['drift_decay'] if not np.isnan(row['drift_decay']) else 0:>8.3f}"
                  f"{marker}")

        # 6. 畫圖
        render_analysis(
            symbol, close, dates, swings, stats_df, dynamic_df, cur_thr,
            output_dir=args.output_dir,
        )

        # 7. 輸出 JSON
        profile = build_fear_profile(
            symbol, stats_df, dynamic_df, cur_thr,
            output_dir=args.output_dir,
        )

        print(f"\n  ✔ fear_profile 已儲存 → results/fear/{symbol}_fear_profile.json")
        print(f"  → 在 rolling_forward.py 加 --fear-profile results/fear/{symbol}_fear_profile.json")


if __name__ == "__main__":
    main()
