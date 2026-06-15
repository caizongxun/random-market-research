"""
rolling_forward.py

核心邏輯
--------
每「輪」只預測 N 根（預設 5），然後把「真實」的 N 根 K 棒
接在歷史尾端，再重新校準 + 預測下一輪。

流程示意（N=5，共跑 6 輪 = 30 根）：

  輪次  訓練窗口末端          預測範圍
  ──────────────────────────────────────────
  0     t=0                  t=1~5   (預測5根)
  1     t=5  (真實K棒填入)   t=6~10
  2     t=10                 t=11~15
  ...

特點
----
- 每輪都重新呼叫 auto_calibrate + GJR-GARCH + backbone → drift 永遠基於最新資料
- 預測窗口短 (5根) → 命中率高
- 結果彙整成一張完整圖 + CSV + JSON

用法
----
  python scripts/rolling_forward.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --end-date 2025-01-01 \\
    --total-bars 30 \\
    --step 5 \\
    --auto-calibrate \\
    --verbose

  # 只預測未來（沒有實際收盤可對照）
  python scripts/rolling_forward.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --auto-calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator       import MarketParameterEstimator
from backbone_fitter        import BackboneFitter
from calibrated_simulator   import CalibratedTheta, build_params_from_theta
from us_equity_simulator    import USStockFutureSimulator

# ── forward_study の共用函式を inline import ──────────────────────────────────
from forward_study import (
    garch_vol_forecast,
    compute_sr_drift_adjustment,
    auto_calibrate,
    dynamic_drift_schedule,
    pick_representative_path,
    ensure_ohlcv,
    _draw_candles,
    _ARCH_AVAILABLE,
)

try:
    from arch import arch_model as _arch_model
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# 單輪預測（predict step=N bars from close_hist）
# ──────────────────────────────────────────────────────────────────────────────
def _run_one_step(
    close_hist: np.ndarray,
    theta: CalibratedTheta,
    args,
    full_df_up_to_now: pd.DataFrame,
) -> tuple[np.ndarray, dict | None]:
    """
    用 close_hist（最近 lookback 根）做一輪 step 根預測。
    回傳 (rep_close[step], ohlc_dict or None)
    """
    step = args.step

    # auto-calibrate
    calib = auto_calibrate(
        full_df_up_to_now["Close"].values.astype(float),
        theta,
        calib_window=args.calib_window,
        use_garch=(not args.no_garch) and _ARCH_AVAILABLE,
        garch_model=args.garch_model,
    )

    intra_bar      = calib["intra_bar"]
    shadow_noise   = calib["shadow_noise"]
    shadow_clamp   = calib["shadow_clamp"]
    momentum_boost = calib["momentum_boost"]
    drift_decay    = calib["drift_decay"]
    vol_multiplier = calib["vol_multiplier"]
    drift_scale    = calib["drift_scale"]
    drift_clamp_max= calib["drift_clamp_max"]

    # backbone + blend_drift
    fitter    = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb_result = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])

    log_rets   = np.diff(np.log(close_hist))
    short_drift= float(np.mean(log_rets[-5:])) if len(log_rets) >= 5 else last_drift
    w          = float(np.clip(args.short_drift_weight, 0.0, 1.0))
    blend_drift= float(np.clip(
        w * short_drift + (1.0 - w) * last_drift,
        -drift_clamp_max, drift_clamp_max
    ))

    # 支撐壓力修正
    start_price = float(close_hist[-1])
    if not args.no_sr:
        drift_adj, _ = compute_sr_drift_adjustment(
            full_df_up_to_now["Close"].values.astype(float),
            start_price=start_price,
            sr_window=args.sr_window,
            n_bins=args.sr_bins,
            top_k=args.sr_top_k,
            pivot_order=args.sr_pivot_order,
            support_zone_pct=args.sr_zone_pct,
            resist_zone_pct=args.sr_zone_pct,
        )
        blend_drift = float(np.clip(
            blend_drift + drift_adj, -drift_clamp_max, drift_clamp_max
        ))

    # backbone forward（供 mean-revert 參考）
    drift_fwd = np.full(step, blend_drift)
    bb_fwd    = start_price * np.cumprod(1 + drift_fwd)
    vol_fwd   = np.full(step, last_vol) * vol_multiplier

    # drift schedule（動態或靜態）
    if args.dynamic_drift:
        drift_schedule, _ = dynamic_drift_schedule(
            close_hist=close_hist,
            blend_drift=blend_drift,
            backbone_schedule=bb_fwd,
            forecast=step,
            drift_decay=drift_decay,
            early_decay_bars=args.early_decay_bars,
            mr_rate=args.mr_rate,
            trend_strength=args.trend_strength,
            vol_regime_refit=args.vol_refit,
            garch_model=args.garch_model,
            drift_clamp_max=drift_clamp_max,
            momentum_signal=short_drift,
        )
    else:
        fast_decay    = min(drift_decay * 2.0, 0.99)
        cur           = blend_drift
        drift_schedule= np.empty(step)
        for t in range(step):
            drift_schedule[t] = cur
            cur *= (1.0 - (fast_decay if t < args.early_decay_bars else drift_decay))

    # estimator + simulator
    ESTIMATOR_LB = 500
    rv_window  = min(21, len(close_hist))
    rv         = float(np.std(np.diff(np.log(close_hist[-rv_window:]))))
    vol_scale  = float(np.clip(rv / max(theta.vol, 1e-8), 0.6, 4.0))

    est_df     = full_df_up_to_now.iloc[-ESTIMATOR_LB:]
    estimator  = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params= estimator.fit(est_df, symbol="")
    params_fwd = build_params_from_theta(theta, base_params)

    import dataclasses
    params_fwd = dataclasses.replace(params_fwd, last_close=start_price)

    momentum_bias = float(getattr(params_fwd, "momentum_bias", 0.0))
    if blend_drift < 0 and len(log_rets) >= args.reversal_window:
        if all(r > 0 for r in log_rets[-args.reversal_window:]):
            momentum_bias = 0.0

    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=step,
        n_paths=args.n_paths,
        seed=42,
        vol_scale=vol_scale,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength * momentum_boost,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_schedule,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=0.06,
        intra_bar_steps=intra_bar,
        drift_decay_rate=drift_decay,
        drift_scale=drift_scale,
        momentum_anchor_weight=0.45,
    )
    result = sim.simulate()

    rep_close, ohlc = pick_representative_path(result)
    if len(rep_close) == 0:
        rep_close = np.full(step, start_price)
        ohlc = None

    return rep_close[:step], (ohlc if ohlc is None else {k: v[:step] for k, v in ohlc.items()})


# ──────────────────────────────────────────────────────────────────────────────
# 主滾動迴圈
# ──────────────────────────────────────────────────────────────────────────────
def rolling_forward(
    df: pd.DataFrame,
    train_end_idx: int,
    theta: CalibratedTheta,
    args,
    actual_future_df: pd.DataFrame | None = None,
) -> dict:
    """
    從 train_end_idx 開始，每輪預測 step 根、然後用真實 K 棒推進。
    若 actual_future_df 不足，就用預測值填充（純未來模式）。

    回傳 result dict 含：
        pred_closes  : np.ndarray (total_bars,)
        pred_ohlcs   : list of dict
        actual_closes: np.ndarray or None
        round_details: list of per-round dicts
    """
    step       = args.step
    total_bars = args.total_bars
    lookback   = args.lookback

    pred_closes  : list[float] = []
    pred_ohlcs   : list[dict]  = []   # 每輪一個 dict
    round_details: list[dict]  = []

    # 滑動「訓練尾端」指標（在 df 中的絕對位置）
    cur_end = train_end_idx

    rounds = int(np.ceil(total_bars / step))
    bars_done = 0

    for rnd in range(rounds):
        need = min(step, total_bars - bars_done)
        if need <= 0:
            break

        # 訓練窗口
        win_start  = max(0, cur_end - lookback)
        close_hist = df.iloc[win_start:cur_end]["Close"].values.astype(float)
        full_window= df.iloc[max(0, cur_end - 500): cur_end].copy()

        if len(close_hist) < 10:
            break

        start_price = float(close_hist[-1])
        print(f"  [輪 {rnd+1}/{rounds}]  訓練截至 idx={cur_end-1}  起點={start_price:.2f}  預測 {need} 根")

        rep, ohlc = _run_one_step(close_hist, theta, args, full_window)
        rep  = rep[:need]
        if ohlc is not None:
            ohlc = {k: v[:need] for k, v in ohlc.items()}

        # 本輪 actual（有的話）
        actual_slice: np.ndarray | None = None
        if actual_future_df is not None:
            sl = actual_future_df.iloc[bars_done: bars_done + need]
            if len(sl) > 0:
                actual_slice = sl["Close"].values.astype(float)

        # 計算本輪誤差
        if actual_slice is not None and len(actual_slice) == need:
            mae_pct = float(np.mean(
                np.abs(rep - actual_slice) / start_price * 100
            ))
            dir_hits = int(np.sum(
                np.sign(rep - start_price) == np.sign(actual_slice - start_price)
            ))
        else:
            mae_pct  = None
            dir_hits = None

        round_details.append({
            "round":       rnd + 1,
            "start_price": round(start_price, 4),
            "pred_end":    round(float(rep[-1]), 4),
            "actual_end":  round(float(actual_slice[-1]), 4) if actual_slice is not None and len(actual_slice) > 0 else None,
            "mae_pct":     round(mae_pct, 4) if mae_pct is not None else None,
            "dir_hits":    dir_hits,
            "bars":        need,
        })

        pred_closes.extend(rep.tolist())
        pred_ohlcs.append(ohlc)

        # 推進 cur_end：優先用真實 K 棒，不夠就用預測值
        if actual_future_df is not None and len(actual_future_df) > bars_done + need - 1:
            # 用真實資料推進 df（只更新指標）
            cur_end += need
        else:
            # 純未來：把預測值臨時插入 df
            cur_end += need

        bars_done += need

    pred_arr = np.array(pred_closes)
    actual_arr: np.ndarray | None = None
    if actual_future_df is not None and len(actual_future_df) >= total_bars:
        actual_arr = actual_future_df.iloc[:total_bars]["Close"].values.astype(float)
    elif actual_future_df is not None and len(actual_future_df) > 0:
        actual_arr = actual_future_df["Close"].values.astype(float)

    return {
        "pred_closes":   pred_arr,
        "pred_ohlcs":    pred_ohlcs,
        "actual_closes": actual_arr,
        "round_details": round_details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 圖表
# ──────────────────────────────────────────────────────────────────────────────
def render_rolling(
    symbol: str,
    hist_close: np.ndarray,
    result: dict,
    step: int,
    end_date_str: str,
    output_dir: str,
):
    pred   = result["pred_closes"]
    actual = result["actual_closes"]
    ohlcs  = result["pred_ohlcs"]
    total  = len(pred)

    fig, ax = plt.subplots(figsize=(16, 6))

    # 歷史
    hist_x = np.arange(-len(hist_close), 0)
    ax.plot(hist_x, hist_close, color="#555", lw=1.2, label="歷史", zorder=4)

    # 預測 K 棒（每輪一個顏色循環）
    palette = ["#26A69A", "#42A5F5", "#AB47BC", "#FF7043", "#66BB6A", "#FFA726"]
    for rnd_i, ohlc in enumerate(ohlcs):
        start_bar = rnd_i * step
        n = min(step, total - start_bar)
        x_arr = np.arange(start_bar, start_bar + n)
        bull = palette[rnd_i % len(palette)]
        bear = _darken(bull)

        if ohlc is not None:
            o = ohlc["open"][:n]
            h = ohlc["high"][:n]
            l = ohlc["low"][:n]
            c = ohlc["close"][:n]
        else:
            c = pred[start_bar: start_bar + n]
            prev = hist_close[-1] if start_bar == 0 else pred[start_bar - 1]
            o = np.concatenate([[prev], c[:-1]])
            sp = c * 0.005
            h = np.maximum(o, c) + sp
            l = np.minimum(o, c) - sp

        _draw_candles(ax, x_arr, o, h, l, c,
                      bull_color=bull, bear_color=bear, alpha=0.85)

        # 輪次分隔線（除第一輪）
        if rnd_i > 0:
            ax.axvline(start_bar, color="#888", lw=0.6, ls=":", alpha=0.6)

    # 實際收盤
    if actual is not None and len(actual) > 0:
        ax.plot(np.arange(len(actual)), actual,
                color="#43A047", lw=1.8, ls="--", label="實際收盤", zorder=6)

    # 輪次標籤
    for rnd_i in range(len(ohlcs)):
        x_mid = rnd_i * step + step / 2
        ax.text(x_mid, ax.get_ylim()[1] * 0.99 if ax.get_ylim()[1] > 0 else 1,
                f"R{rnd_i+1}", ha="center", va="top",
                fontsize=7, color="#555", alpha=0.8)

    ax.axvline(0, color="#999", lw=0.8, ls=":")
    ax.set_title(f"{symbol}  Rolling {step}-bar Forward  [{end_date_str}]")
    ax.set_xlabel("交易日（相對預測起點）")
    ax.set_ylabel("價格")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 下方子圖：每輪 MAE ──
    round_details = result["round_details"]
    maes   = [r["mae_pct"] for r in round_details if r["mae_pct"] is not None]
    rounds_with_mae = [r["round"] for r in round_details if r["mae_pct"] is not None]

    if maes:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        colors2   = ["#EF5350" if m > np.mean(maes) else "#26A69A" for m in maes]
        ax2.bar(rounds_with_mae, maes, color=colors2, alpha=0.8, width=0.6)
        ax2.axhline(np.mean(maes), color="#888", lw=1, ls="--",
                    label=f"平均 MAE={np.mean(maes):.2f}%")
        ax2.set_title(f"{symbol}  每輪 MAE (close)")
        ax2.set_xlabel("輪次")
        ax2.set_ylabel("MAE %")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        mae_path = out / f"{symbol}_{end_date_str}_rolling{step}_mae.png"
        fig2.savefig(mae_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"✔ MAE 圖 → {mae_path}")

    out   = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{symbol}_{end_date_str}_rolling{step}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ 預測圖 → {fname}")
    return str(fname)


def _darken(hex_color: str, factor: float = 0.65) -> str:
    """把 #RRGGBB 加深（用於熊線）"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor)
    )


# ──────────────────────────────────────────────────────────────────────────────
# OHLC 對照表（滾動版）
# ──────────────────────────────────────────────────────────────────────────────
def print_rolling_comparison(
    result: dict,
    start_price: float,
    symbol: str,
    step: int,
):
    pred   = result["pred_closes"]
    actual = result["actual_closes"]
    ohlcs  = result["pred_ohlcs"]
    n      = len(pred)

    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  Rolling {step}-bar Forward  —  {symbol}  (共 {n} 根 K 棒)")
    print(sep)
    hdr = (f"  {'Rnd':>3} {'Bar':>3}  "
           f"{'P-Close':>8} {'P-Ret%':>7}  "
           f"{'A-Close':>8} {'A-Ret%':>7}  "
           f"{'Err%':>7} {'Dir':>3}")
    print(hdr)
    print("─" * 80)

    prev_p = start_price
    prev_a = start_price
    global_i = 0
    close_errs = []
    dir_correct = 0

    for rnd_i, rnd in enumerate(result["round_details"]):
        bars_this = rnd["bars"]
        for b in range(bars_this):
            p_c = float(pred[global_i])
            p_ret = (p_c - prev_p) / prev_p * 100

            if actual is not None and global_i < len(actual):
                a_c   = float(actual[global_i])
                a_ret = (a_c - prev_a) / prev_a * 100
                err   = (p_c - a_c) / a_c * 100
                close_errs.append(abs(err))
                p_dir = "+" if p_c >= prev_p else "-"
                a_dir = "+" if a_c >= prev_a else "-"
                match = "✓" if p_dir == a_dir else "✗"
                if p_dir == a_dir:
                    dir_correct += 1
                print(f"  {rnd_i+1:>3} {b+1:>3}  "
                      f"{p_c:>8.2f} {p_ret:>+7.2f}%  "
                      f"{a_c:>8.2f} {a_ret:>+7.2f}%  "
                      f"{err:>+7.2f}% {match:>3}")
                prev_a = a_c
            else:
                print(f"  {rnd_i+1:>3} {b+1:>3}  "
                      f"{p_c:>8.2f} {p_ret:>+7.2f}%  "
                      f"{'N/A':>8} {'N/A':>7}  "
                      f"{'N/A':>7} {'N/A':>3}")
            prev_p = p_c
            global_i += 1

        # 輪次小計線
        if actual is not None and global_i <= len(actual):
            rnd_mae = rnd.get("mae_pct")
            rnd_dir = rnd.get("dir_hits")
            if rnd_mae is not None:
                print(f"  {'':>7}  → 本輪 MAE={rnd_mae:.2f}%  方向命中={rnd_dir}/{bars_this}")
        print("  " + "·" * 40)

    print(sep)
    if close_errs:
        total_n = len(close_errs)
        print(f"  合計 {total_n} 根  MAE={np.mean(close_errs):.2f}%  "
              f"MAX={np.max(close_errs):.2f}%  "
              f"方向命中={dir_correct}/{total_n} ({dir_correct/total_n*100:.0f}%)")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# parse_args
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",         required=True)
    p.add_argument("--theta",          required=True)
    p.add_argument("--end-date",       default=None,
                   help="預測起點日期 (YYYY-MM-DD)，不填則用今日")
    p.add_argument("--total-bars",     type=int, default=30,
                   help="總共要預測幾根 K 棒 (預設 30)")
    p.add_argument("--step",           type=int, default=5,
                   help="每輪預測幾根，然後用真實 K 棒推進 (預設 5)")
    p.add_argument("--lookback",       type=int, default=120)
    p.add_argument("--n-paths",        type=int, default=500)
    p.add_argument("--calib-window",   type=int, default=500)
    p.add_argument("--no-garch",        action="store_true")
    p.add_argument("--garch-model",    default="gjr-garch",
                   choices=["gjr-garch", "garch"])
    p.add_argument("--auto-calibrate",  action="store_true", default=True)
    p.add_argument("--output-dir",     default="results")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument("--short-drift-weight", type=float, default=0.4)
    # 支撐壓力
    p.add_argument("--sr-window",      type=int,   default=90)
    p.add_argument("--sr-bins",        type=int,   default=40)
    p.add_argument("--sr-top-k",       type=int,   default=5)
    p.add_argument("--sr-pivot-order", type=int,   default=5)
    p.add_argument("--sr-zone-pct",    type=float, default=0.035)
    p.add_argument("--no-sr",           action="store_true")
    # 動態 drift
    p.add_argument("--dynamic-drift",   action="store_true", default=True)
    p.add_argument("--no-dynamic-drift", dest="dynamic_drift", action="store_false")
    p.add_argument("--mr-rate",        type=float, default=0.08)
    p.add_argument("--trend-strength", type=float, default=0.5)
    p.add_argument("--vol-refit",      type=int,   default=5)
    p.add_argument("--early-decay-bars", type=int, default=5,
                   help="rolling 模式建議 5 (等於 step)")
    p.add_argument("--reversal-window", type=int,  default=3)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    end_dt   = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today()
    start_dt = end_dt - pd.DateOffset(years=3)
    dl_end   = end_dt + pd.DateOffset(days=args.total_bars * 2 + 20)

    print(f"\n下載 {args.symbol}  {start_dt.date()} ~ {dl_end.date()} ...")
    df_raw = yf.download(
        args.symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=dl_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    print(f"共 {len(df)} 根 K 棒")

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        dates = pd.to_datetime(df["Datetime"])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])

    mask = dates <= end_dt
    if not mask.any():
        raise ValueError(f"找不到 {end_dt.date()} 之前的資料")
    train_end_idx = int(mask.values.nonzero()[0][-1]) + 1

    future_df = df.iloc[train_end_idx: train_end_idx + args.total_bars].copy()
    train_df  = df.iloc[train_end_idx - args.lookback: train_end_idx]
    hist_close= train_df["Close"].values.astype(float)
    start_price = float(hist_close[-1])

    actual_future_df = future_df if len(future_df) > 0 else None

    end_date_str = end_dt.strftime("%Y-%m-%d")
    print(f"\n[rolling_forward]  {args.symbol}  起點={end_date_str}  "
          f"step={args.step}  total={args.total_bars}  "
          f"dynamic_drift={args.dynamic_drift}")

    result = rolling_forward(
        df=df,
        train_end_idx=train_end_idx,
        theta=theta,
        args=args,
        actual_future_df=actual_future_df,
    )

    print_rolling_comparison(
        result=result,
        start_price=start_price,
        symbol=args.symbol,
        step=args.step,
    )

    render_rolling(
        symbol=args.symbol,
        hist_close=hist_close,
        result=result,
        step=args.step,
        end_date_str=end_date_str,
        output_dir=args.output_dir,
    )

    # 儲存 JSON
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol":       args.symbol,
        "end_date":     end_date_str,
        "step":         args.step,
        "total_bars":   args.total_bars,
        "start_price":  round(start_price, 4),
        "round_details": result["round_details"],
        "dynamic_drift": args.dynamic_drift,
    }
    if result["actual_closes"] is not None:
        pred   = result["pred_closes"]
        actual = result["actual_closes"]
        na     = min(len(pred), len(actual))
        mae    = float(np.mean(np.abs(pred[:na] - actual[:na]) / start_price * 100))
        dirs   = int(np.sum(np.sign(pred[:na] - start_price) ==
                            np.sign(actual[:na] - start_price)))
        summary["overall_mae_pct"]   = round(mae, 4)
        summary["dir_accuracy_pct"]  = round(dirs / na * 100, 2)
        print(f"\n  整體 MAE={mae:.2f}%   方向命中={dirs}/{na} ({dirs/na*100:.0f}%)")

    json_path = out_path / f"{args.symbol}_{end_date_str}_rolling{args.step}.json"
    with open(json_path, "w") as fj:
        json.dump(summary, fj, indent=2, ensure_ascii=False)
    print(f"✔ 結果 JSON → {json_path}")


if __name__ == "__main__":
    main()
