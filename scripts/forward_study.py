"""
forward_study.py  v1

後續走勢對比研究：
  1. 載入已校準的 theta JSON + 歷史數據（lookback 段）
  2. 從最後一根 K 棒往後模擬 forecast_steps 步
  3. 下載真實後續 K 棒，疊在帶子上
  4. 計算量化指標：
       - hit_rate_25_75  : 實際走勢在 25-75% 帶內的比例
       - hit_rate_10_90  : 實際走勢在 10-90% 帶內的比例
       - direction_acc   : 方向準確率（漲跌一致）
       - end_error_pct   : 終點價格誤差 %
       - mae_pct         : 逐步 MAE（中線 vs 實際）%
  5. 輸出對比圖 + 指標 JSON

Example:
    # 用已有的 theta 做預測，然後拉實際數據對比
    python scripts/forward_study.py \\
        --symbol AAPL \\
        --theta calibrated_theta_aapl.json \\
        --lookback 120 \\
        --forecast 30 \\
        --seed 42 \\
        --n-paths 500 \\
        --backbone-mr 0.06 \\
        --n-seg 6 \\
        --output forward_study_aapl

    # 如果預測日期已經過去，會自動下載真實後續 K 棒
    # 如果還未到期，只顯示模擬帶子（無白線實際）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from backbone_fitter import BackboneFitter
from calibrated_simulator import CalibratedTheta, build_params_from_theta
from us_equity_simulator import USStockFutureSimulator

DARK = "#0e0e0e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",      required=True)
    p.add_argument("--theta",       required=True,  help="校準後的 theta JSON 路徑")
    p.add_argument("--lookback",    type=int,  default=120, help="訓練窗口長度")
    p.add_argument("--forecast",    type=int,  default=30,  help="往後模擬幾步")
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--n-paths",     type=int,  default=500)
    p.add_argument("--n-seg",       type=int,  default=6)
    p.add_argument("--smooth-reg",  type=float, default=0.5)
    p.add_argument("--backbone-mr", type=float, default=0.06)
    p.add_argument("--period",      default="3y")
    p.add_argument("--interval",    default="1d")
    p.add_argument("--output",      default="forward_study", help="輸出前綴（無副檔名）")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def compute_metrics(
    actual: np.ndarray,
    median: np.ndarray,
    p25: np.ndarray,
    p75: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    start_price: float,
) -> dict:
    n = len(actual)
    m = min(n, len(median))
    if m == 0:
        return {}

    act = actual[:m]
    med = median[:m]
    q25 = p25[:m]
    q75 = p75[:m]
    q10 = p10[:m]
    q90 = p90[:m]

    hit_25_75 = float(np.mean((act >= q25) & (act <= q75)))
    hit_10_90 = float(np.mean((act >= q10) & (act <= q90)))

    # 方向準確率：每步實際漲跌 vs 中線漲跌
    actual_dir = np.sign(np.diff(np.concatenate([[start_price], act])))
    median_dir = np.sign(np.diff(np.concatenate([[start_price], med])))
    direction_acc = float(np.mean(actual_dir == median_dir))

    end_error = float(abs(act[-1] - med[-1]) / start_price * 100)
    mae_pct   = float(np.mean(np.abs(act - med) / start_price * 100))

    # 最大偏離：實際距離帶子中線最大距離
    max_dev = float(np.max(np.abs(act - med) / start_price * 100))

    # 上穿/下穿帶子次數（10-90%）
    above_90 = int(np.sum(act > q90))
    below_10 = int(np.sum(act < q10))

    return {
        "n_compared": m,
        "hit_rate_25_75":  round(hit_25_75,  4),
        "hit_rate_10_90":  round(hit_10_90,  4),
        "direction_acc":   round(direction_acc, 4),
        "end_error_pct":   round(end_error,  4),
        "mae_pct":         round(mae_pct,    4),
        "max_deviation_pct": round(max_dev, 4),
        "bars_above_p90":  above_90,
        "bars_below_p10":  below_10,
    }


def build_schedules(segment_drifts, segment_vols, n_steps):
    n_seg   = len(segment_drifts)
    seg_len = n_steps // n_seg
    d_arr   = np.empty(n_steps)
    v_arr   = np.empty(n_steps)
    for s in range(n_seg):
        lo = s * seg_len
        hi = lo + seg_len if s < n_seg - 1 else n_steps
        d_arr[lo:hi] = segment_drifts[s]
        v_arr[lo:hi] = segment_vols[s]
    return d_arr, v_arr


def main():
    args = parse_args()

    # 載入 theta
    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))
    print(f"Loaded theta: vol={theta.vol:.5f}  drift={theta.drift:+.6f}  hurst={theta.hurst_proxy:.3f}")

    # 下載歷史資料（lookback + 一些餘裕）
    print(f"Downloading {args.symbol} ({args.period}, {args.interval})...")
    df_raw = yf.download(
        args.symbol, period=args.period, interval=args.interval,
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    print(f"Total bars: {len(df)}")

    ESTIMATOR_LB = 500
    needed = ESTIMATOR_LB + args.lookback + args.forecast
    if len(df) < needed:
        raise ValueError(f"Need {needed} bars (500 estimator + {args.lookback} lookback + {args.forecast} forecast), got {len(df)}")

    # 訓練窗口 = lookback 段的最後一根之前
    # 預測起點 = train_end
    train_end_idx = len(df) - args.forecast  # 假設最後 forecast 根是「未來」
    train_df      = df.iloc[train_end_idx - args.lookback: train_end_idx]
    estimate_df   = df.iloc[train_end_idx - ESTIMATOR_LB: train_end_idx]
    future_df     = df.iloc[train_end_idx: train_end_idx + args.forecast]

    close_hist    = train_df["Close"].values
    start_price   = float(close_hist[-1])
    actual_future = future_df["Close"].values  # 真實後續走勢

    train_dates   = train_df["Date"].values if "Date" in train_df else np.arange(len(close_hist))
    future_dates  = future_df["Date"].values if "Date" in future_df else np.arange(len(actual_future))

    print(f"Train  : {len(close_hist)} bars  start={close_hist[0]:.2f}  end={start_price:.2f}")
    print(f"Forecast: {args.forecast} bars   actual_end={actual_future[-1] if len(actual_future) else 'N/A'}")

    # 骨幹擬合（在 train 段）
    fitter    = BackboneFitter(n_seg=args.n_seg, smooth_reg=args.smooth_reg)
    bb_result = fitter.fit(close_hist)
    print(f"Backbone MSE={bb_result.fit_mse:.6f}")
    print(f"Segment drifts: {[f'{d*100:+.3f}%' for d in bb_result.segment_drifts]}")

    # 用骨幹最後一段的漂移延伸作為 forecast drift schedule
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol)
    # 骨幹延伸：從 start_price 用 last_drift 直線延伸
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

    vol_scale = float(np.clip(theta.vol / max(last_vol, 1e-8), 0.5, 3.0))
    print(f"Forward vol_scale={vol_scale:.3f}")

    # 估算器（用 train 前的 500 根）
    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    import dataclasses
    params_fwd  = dataclasses.replace(base_params, last_close=start_price,
                                      momentum_bias=0.0, node_breakout_state=0)
    params_fwd  = build_params_from_theta(theta, params_fwd)

    # 模擬
    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=args.forecast,
        n_paths=args.n_paths,
        seed=args.seed,
        vol_scale=vol_scale,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_fwd,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=args.backbone_mr,
    )
    result = sim.simulate()

    # 量化指標
    metrics = compute_metrics(
        actual=actual_future,
        median=result.median_path,
        p25=result.p25, p75=result.p75,
        p10=result.p10, p90=result.p90,
        start_price=start_price,
    )
    print("\n=== Forward Study Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    # 儲存指標
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(str(out_prefix) + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({**metrics,
                   "symbol": args.symbol,
                   "forecast_steps": args.forecast,
                   "lookback": args.lookback,
                   "start_price": start_price,
                   "actual_end": float(actual_future[-1]) if len(actual_future) else None,
                   "median_end": float(result.median_path[-1]),
                   }, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # ── 繪圖 ──────────────────────────────────────────────────────────────
    x_hist  = np.arange(len(close_hist))
    x_fwd   = np.arange(len(close_hist), len(close_hist) + args.forecast)
    x_fwd_a = np.arange(len(close_hist), len(close_hist) + len(actual_future))

    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(DARK)

    ax = axes[0]
    ax.set_facecolor(DARK)

    # 歷史走勢（灰色）
    ax.plot(x_hist, close_hist, color="#aaaaaa", lw=1.2, alpha=0.7, label="History")
    # 骨幹（訓練段）
    ax.plot(x_hist, bb_result.backbone, color="#ff9900", lw=1.5, ls="--",
            alpha=0.7, label="Backbone (train)")
    # 骨幹延伸（預測段）
    ax.plot(x_fwd, bb_fwd, color="#ff9900", lw=1.2, ls=":",
            alpha=0.5, label="Backbone ext.")

    # 模擬路徑（細線）
    for pi in range(min(60, result.future_paths.shape[0])):
        ax.plot(x_fwd, result.future_paths[pi], color="cyan", alpha=0.02, lw=0.5)

    # 帶子
    ax.fill_between(x_fwd, result.p25, result.p75,
                    color="#00e5ff", alpha=0.22, label="25-75%")
    ax.fill_between(x_fwd, result.p10, result.p90,
                    color="#00e5ff", alpha=0.09, label="10-90%")

    # 中線
    ax.plot(x_fwd, result.median_path, color="yellow", lw=2.2, label="Median sim")

    # 實際後續（白線）
    if len(actual_future) > 0:
        ax.plot(x_fwd_a, actual_future, color="white", lw=2.0, label="Actual", zorder=10)
        ax.scatter([x_fwd_a[-1]], [actual_future[-1]],
                   color="white", s=40, zorder=11)

    # 起點垂直線
    ax.axvline(len(close_hist) - 0.5, color="#888", lw=1.2, ls="-", alpha=0.8)
    ax.text(len(close_hist) - 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else start_price * 1.02,
            " Forecast start", color="#aaa", fontsize=8, va="top")

    # Volume nodes
    for node in base_params.volume_nodes:
        ax.axhline(node, color="orange", lw=0.7, alpha=0.25, ls=":")

    hit_str = ""
    if metrics:
        hit_str = (f"  |  hit25-75={metrics['hit_rate_25_75']:.0%}  "
                   f"hit10-90={metrics['hit_rate_10_90']:.0%}  "
                   f"dir_acc={metrics['direction_acc']:.0%}  "
                   f"MAE={metrics['mae_pct']:.2f}%  "
                   f"end_err={metrics['end_error_pct']:.2f}%")

    ax.set_title(
        f"{args.symbol} | Forward Study | lookback={args.lookback}  forecast={args.forecast}"
        f"  start={start_price:.2f}\n"
        f"vol={theta.vol:.4f}  hurst={theta.hurst_proxy:.3f}  bb_mr={args.backbone_mr:.3f}"
        + hit_str,
        color="white", fontsize=9,
    )
    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
              edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="#888")

    # ── 下方面板：逐步偏差 ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(DARK)
    if len(actual_future) > 0:
        m_steps = min(len(actual_future), len(result.median_path))
        dev_pct = (actual_future[:m_steps] - result.median_path[:m_steps]) / start_price * 100
        colors  = ["#66ff66" if v >= 0 else "#ff6666" for v in dev_pct]
        ax2.bar(x_fwd[:m_steps], dev_pct, color=colors, alpha=0.75, width=0.8)
        ax2.axhline(0, color="#555", lw=0.8)
        ax2.set_ylabel("Actual - Median (%)", color="#888", fontsize=8)
        ax2.tick_params(colors="#888")
        ax2.set_title("Per-bar deviation: Actual − Median", color="#888", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Actual data not yet available",
                 ha="center", va="center", color="#666", transform=ax2.transAxes)
        ax2.tick_params(colors="#888")

    plt.tight_layout()
    chart_path = Path(str(out_prefix) + "_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  Chart saved : {chart_path}")

    print("\n✔ Forward study 完成。")
    print(f"  圖表：{chart_path}")
    print(f"  指標：{metrics_path}")


if __name__ == "__main__":
    main()
