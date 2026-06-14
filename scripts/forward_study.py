"""
forward_study.py  v5

新增（v5）：詳細診斷 log
  - 每個階段的計算細節都印出
  - 帶子中心線 vs 實際走勢的偏移分析
  - 自動建議 drift_scale 調整范圍
  - 路徑散度統計（帶子寬窄程度評估）

Example:
    python scripts/forward_study.py \\
        --symbol AAPL \\
        --theta results/theta_aapl.json \\
        --lookback 120 --forecast 30 \\
        --seed 42 --n-paths 500 \\
        --backbone-mr 0.06 --n-seg 6 \\
        --hist-window 60 --intra-bar 8 \\
        --drift-decay 0.05 --drift-scale 0.5 --anchor-weight 0.3 \\
        --vol-multiplier 1.5 --recent-vol-window 20 \\
        --output results/forward_aapl
"""

from __future__ import annotations

import argparse
import json
import sys
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from backbone_fitter import BackboneFitter
from calibrated_simulator import CalibratedTheta, build_params_from_theta
from us_equity_simulator import USStockFutureSimulator
from candle_renderer import render_forecast_candles

SEP = "-" * 60


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",            required=True)
    p.add_argument("--theta",             required=True)
    p.add_argument("--lookback",          type=int,   default=120)
    p.add_argument("--forecast",          type=int,   default=30)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--n-paths",           type=int,   default=500)
    p.add_argument("--n-seg",             type=int,   default=6)
    p.add_argument("--smooth-reg",        type=float, default=0.5)
    p.add_argument("--backbone-mr",       type=float, default=0.06)
    p.add_argument("--period",            default="3y")
    p.add_argument("--interval",          default="1d")
    p.add_argument("--hist-window",       type=int,   default=60)
    p.add_argument("--intra-bar",         type=int,   default=8)
    p.add_argument("--drift-decay",       type=float, default=0.05)
    p.add_argument("--drift-scale",       type=float, default=0.5)
    p.add_argument("--anchor-weight",     type=float, default=0.3)
    p.add_argument("--vol-multiplier",    type=float, default=1.5)
    p.add_argument("--recent-vol-window", type=int,   default=20)
    p.add_argument("--vol-scale-min",     type=float, default=0.6)
    p.add_argument("--vol-scale-max",     type=float, default=4.0)
    p.add_argument("--output",            default="forward_study")
    return p.parse_args()


def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def recent_realized_vol(close_arr: np.ndarray, window: int) -> float:
    w = min(window, len(close_arr) - 1)
    log_rets = np.diff(np.log(close_arr[-(w + 1):]))
    return float(np.std(log_rets))


def compute_metrics(actual, median, p25, p75, p10, p90, start_price):
    n = min(len(actual), len(median))
    if n == 0:
        return {}
    act, med  = actual[:n], median[:n]
    hit_25_75 = float(np.mean((act >= p25[:n]) & (act <= p75[:n])))
    hit_10_90 = float(np.mean((act >= p10[:n]) & (act <= p90[:n])))
    actual_dir    = np.sign(np.diff(np.concatenate([[start_price], act])))
    median_dir    = np.sign(np.diff(np.concatenate([[start_price], med])))
    direction_acc = float(np.mean(actual_dir == median_dir))
    end_error = float(abs(act[-1] - med[-1]) / start_price * 100)
    mae_pct   = float(np.mean(np.abs(act - med) / start_price * 100))
    max_dev   = float(np.max(np.abs(act - med) / start_price * 100))
    return {
        "n_compared":        n,
        "hit_rate_25_75":    round(hit_25_75,    4),
        "hit_rate_10_90":    round(hit_10_90,    4),
        "direction_acc":     round(direction_acc, 4),
        "end_error_pct":     round(end_error,    4),
        "mae_pct":           round(mae_pct,      4),
        "max_deviation_pct": round(max_dev,      4),
        "bars_above_p90":    int(np.sum(act > p90[:n])),
        "bars_below_p10":    int(np.sum(act < p10[:n])),
    }


def print_diagnostics(
    args, theta, close_hist, actual_close,
    rv, vol_scale, last_drift, last_vol, bb_result,
    result, start_price, metrics
):
    """v5 詳細診斷 log"""
    T = args.forecast

    # 實際終點 vs 預測終點
    actual_end   = float(actual_close[-1]) if len(actual_close) else None
    median_end   = float(result.median_path[-1])
    actual_chg   = (actual_end / start_price - 1) * 100 if actual_end else None
    median_chg   = (median_end / start_price - 1) * 100
    p10_end      = float(result.p10[-1])
    p90_end      = float(result.p90[-1])
    band_width   = (p90_end - p10_end) / start_price * 100

    # 帶子中心偏移：(median_end - actual_end) / start_price * 100
    center_bias  = (median_end - actual_end) / start_price * 100 if actual_end else None
    # 正 = median 偏高, 負 = median 偏低

    # 實際路徑的日平均漬動
    if len(actual_close) >= 2:
        actual_daily_drift = float(np.mean(np.diff(np.log(actual_close))))
    else:
        actual_daily_drift = None
    median_daily_drift = float(np.mean(np.diff(np.log(result.median_path))))

    # 近期歷史走勢（用於對照）
    hist_drift_20  = float(np.mean(np.diff(np.log(close_hist[-21:]))))
    hist_drift_60  = float(np.mean(np.diff(np.log(close_hist[-61:])))) if len(close_hist) >= 61 else None
    hist_drift_all = float(np.mean(np.diff(np.log(close_hist))))

    # 分段 vol 資訊
    seg_drifts = [f"{d*100:+.3f}%" for d in bb_result.segment_drifts]
    seg_vols   = [f"{v*100:.3f}%"  for v in bb_result.segment_vols]

    print()
    print("=" * 60)
    print("  VERBOSE DIAGNOSTICS (v5)")
    print("=" * 60)

    print(f"\n[資料概況]")
    print(f"  symbol        : {args.symbol}")
    print(f"  lookback      : {args.lookback} bars")
    print(f"  forecast      : {T} bars")
    print(f"  start_price   : {start_price:.4f}")
    print(f"  actual_end    : {actual_end:.4f}  ({actual_chg:+.2f}%)" if actual_end else "  actual_end    : N/A")
    print(f"  median_end    : {median_end:.4f}  ({median_chg:+.2f}%)")
    print(f"  p10_end       : {p10_end:.4f}")
    print(f"  p90_end       : {p90_end:.4f}")

    print(f"\n[帶子中心偏移分析]")
    if center_bias is not None:
        direction = "↑ median 高低估(帶子心線偏高)→ 考慮降低 drift_scale" if center_bias > 0 \
                    else "↓ median 低估(帶子心線偏低) → 考慮提高 drift_scale"
        print(f"  center_bias   : {center_bias:+.2f}%  {direction}")
    print(f"  band_width_p10_90 : {band_width:.2f}%  (判斷: >{T*0.5:.0f}% 為寬帶, <{T*0.2:.0f}% 為窄帶)")

    print(f"\n[漬動對照]")
    print(f"  theta.drift             : {theta.drift*100:+.4f}%/day")
    print(f"  last_seg_drift (bb)     : {last_drift*100:+.4f}%/day")
    print(f"  hist_drift_20d          : {hist_drift_20*100:+.4f}%/day")
    if hist_drift_60:
        print(f"  hist_drift_60d          : {hist_drift_60*100:+.4f}%/day")
    print(f"  hist_drift_all({args.lookback}d)  : {hist_drift_all*100:+.4f}%/day")
    print(f"  median_daily_drift      : {median_daily_drift*100:+.4f}%/day  (實際模擬出來的)")
    if actual_daily_drift:
        print(f"  actual_daily_drift      : {actual_daily_drift*100:+.4f}%/day  (實際後續)")
        drift_gap = (median_daily_drift - actual_daily_drift) * 100
        print(f"  drift_gap(med-act)      : {drift_gap:+.4f}%/day  (正=median偏快, 負=median偏慢)")

    print(f"\n[波動率分析]")
    print(f"  theta.vol (long-term)   : {theta.vol*100:.4f}%/day")
    print(f"  last_seg_vol (bb)       : {last_vol*100:.4f}%/day")
    print(f"  recent_vol ({args.recent_vol_window}d)       : {rv*100:.4f}%/day")
    print(f"  vol_scale (rv/theta)    : {vol_scale:.3f}")
    print(f"  vol_multiplier          : {args.vol_multiplier}")
    print(f"  effective_vol           : {rv * vol_scale * args.vol_multiplier * 100:.4f}%/day")

    print(f"\n[骨幹分段]")
    print(f"  n_seg         : {args.n_seg}")
    print(f"  segment_drifts: {seg_drifts}")
    print(f"  segment_vols  : {seg_vols}")
    print(f"  backbone_MSE  : {bb_result.fit_mse:.6f}")

    print(f"\n[預測參數]")
    print(f"  drift_scale    : {args.drift_scale}")
    print(f"  drift_decay    : {args.drift_decay}  (t30 殘餘: {np.exp(-args.drift_decay*30)*100:.1f}%)")
    print(f"  anchor_weight  : {args.anchor_weight}")
    print(f"  backbone_mr    : {args.backbone_mr}")
    print(f"  n_paths        : {args.n_paths}")
    print(f"  intra_bar      : {args.intra_bar}")

    print(f"\n[表現指標]")
    for k, v in metrics.items():
        flag = ""
        if k == "hit_rate_10_90":
            if v >= 0.7:   flag = " ✅ 好"
            elif v >= 0.5: flag = " ⚠ 可接受"
            else:          flag = " ❌ 帶子太窄"
        if k == "hit_rate_25_75":
            if v >= 0.4:   flag = " ✅ 好"
            elif v >= 0.25: flag = " ⚠ 可接受"
            else:           flag = " ❌ 帶子心線偏差"
        if k == "direction_acc":
            if v >= 0.6:   flag = " ✅ 好"
            elif v >= 0.5: flag = " ⚠ 遠於隨機"
            else:          flag = " ❌ 差於擲母(中心線方向错)"
        if k == "bars_above_p90" and isinstance(v, int):
            pct = v / max(metrics.get("n_compared", 30), 1)
            if pct > 0.3:  flag = f" ❌ 帶子偶尔偏低({pct:.0%})"
        if k == "bars_below_p10" and isinstance(v, int):
            pct = v / max(metrics.get("n_compared", 30), 1)
            if pct > 0.3:  flag = f" ❌ 帶子偶尔偏高({pct:.0%})"
        print(f"  {k:25s}: {v}{flag}")

    # 自動建議
    print(f"\n[自動建議]")
    suggestions = []

    if metrics.get("hit_rate_10_90", 0) < 0.5:
        suggestions.append("  • 帶子太窄: 嘗試 --vol-multiplier +0.3 或 --recent-vol-window 縮小")
    if metrics.get("hit_rate_10_90", 0) > 0.95 and metrics.get("hit_rate_25_75", 0) < 0.25:
        suggestions.append("  • 帶子寬但中心線偏差: 調整 drift_scale 是首要任務")
    if center_bias is not None:
        if center_bias > 3.0:
            new_ds = round(args.drift_scale * 0.6, 2)
            suggestions.append(f"  • median 明顯偏高 {center_bias:+.1f}%: --drift-scale {new_ds} (當前 {args.drift_scale} 降至 {new_ds})")
        elif center_bias < -3.0:
            new_ds = round(min(args.drift_scale * 1.5, 1.5), 2)
            suggestions.append(f"  • median 明顯偏低 {center_bias:+.1f}%: --drift-scale {new_ds} (當前 {args.drift_scale} 提至 {new_ds})")
        elif abs(center_bias) <= 3.0:
            suggestions.append(f"  • 帶子中心線偏移 {center_bias:+.1f}% (小於 3%，可接受)")
    if metrics.get("direction_acc", 1) < 0.5:
        if actual_daily_drift and actual_daily_drift > 0 and median_daily_drift < 0:
            suggestions.append("  • 方向完全相反: 試試 --drift-scale 0.8 或檢查 backbone 最後一段是否是下降段")
        else:
            suggestions.append("  • 方向准確度差: 考慮調整 --drift-decay 或 --anchor-weight")
    if not suggestions:
        suggestions.append("  • 目前參數已處於較好狀態，可繼續微調")

    for s in suggestions:
        print(s)
    print("=" * 60)


def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    print(f"Downloading {args.symbol}...")
    df_raw = yf.download(args.symbol, period=args.period, interval=args.interval,
                         auto_adjust=False, progress=False)
    df = ensure_ohlcv(df_raw)
    print(f"Total bars: {len(df)}")

    ESTIMATOR_LB = 500
    needed = ESTIMATOR_LB + args.lookback + args.forecast
    if len(df) < needed:
        raise ValueError(f"Need {needed} bars, got {len(df)}")

    train_end_idx = len(df) - args.forecast
    train_df      = df.iloc[train_end_idx - args.lookback: train_end_idx]
    estimate_df   = df.iloc[train_end_idx - ESTIMATOR_LB: train_end_idx]
    future_df     = df.iloc[train_end_idx: train_end_idx + args.forecast]

    close_hist  = train_df["Close"].values
    start_price = float(close_hist[-1])

    hist_open   = train_df["Open"].values.astype(float)
    hist_high   = train_df["High"].values.astype(float)
    hist_low    = train_df["Low"].values.astype(float)
    hist_close  = close_hist
    hist_volume = train_df["Volume"].values.astype(float)
    hist_volume_norm = hist_volume / (hist_volume.mean() + 1e-8)

    actual_close  = future_df["Close"].values.astype(float)
    actual_open   = future_df["Open"].values.astype(float)
    actual_high   = future_df["High"].values.astype(float)
    actual_low    = future_df["Low"].values.astype(float)
    actual_volume = future_df["Volume"].values.astype(float)
    actual_volume_norm = actual_volume / (hist_volume.mean() + 1e-8)

    print(f"Train end: {start_price:.2f}  forecast={args.forecast} bars")

    fitter    = BackboneFitter(n_seg=args.n_seg, smooth_reg=args.smooth_reg)
    bb_result = fitter.fit(close_hist)

    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol)
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

    rv = recent_realized_vol(close_hist, args.recent_vol_window)
    vol_scale = float(np.clip(
        rv / max(theta.vol, 1e-8),
        args.vol_scale_min,
        args.vol_scale_max,
    ))

    vol_fwd_scaled = vol_fwd * args.vol_multiplier

    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = dataclasses.replace(base_params, last_close=start_price,
                                      momentum_bias=0.0, node_breakout_state=0)
    params_fwd  = build_params_from_theta(theta, params_fwd)

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
        vol_schedule=vol_fwd_scaled,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=args.backbone_mr,
        intra_bar_steps=args.intra_bar,
        drift_decay_rate=args.drift_decay,
        drift_scale=args.drift_scale,
        momentum_anchor_weight=args.anchor_weight,
    )
    result = sim.simulate()

    metrics = compute_metrics(
        actual=actual_close, median=result.median_path,
        p25=result.p25, p75=result.p75,
        p10=result.p10, p90=result.p90,
        start_price=start_price,
    )

    # v5 詳細 log
    print_diagnostics(
        args=args, theta=theta,
        close_hist=close_hist, actual_close=actual_close,
        rv=rv, vol_scale=vol_scale,
        last_drift=last_drift, last_vol=last_vol,
        bb_result=bb_result,
        result=result, start_price=start_price,
        metrics=metrics,
    )

    out_prefix   = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(str(out_prefix) + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "symbol":           args.symbol,
            "forecast_steps":   args.forecast,
            "lookback":         args.lookback,
            "drift_scale":      args.drift_scale,
            "drift_decay":      args.drift_decay,
            "vol_multiplier":   args.vol_multiplier,
            "recent_vol":       round(rv, 6),
            "vol_scale":        round(vol_scale, 4),
            "start_price":      start_price,
            "actual_end":       float(actual_close[-1]) if len(actual_close) else None,
            "median_end":       float(result.median_path[-1]),
            "rep_end":          float(result.representative_path[-1]),
        }, f, indent=2)

    actual_deviation = None
    if len(actual_close) > 0:
        m = min(len(actual_close), len(result.median_path))
        actual_deviation = (actual_close[:m] - result.median_path[:m]) / start_price * 100

    # 標題：简潔版
    hit_str = ""
    if metrics:
        hit_str = (
            f"  |  hit25-75={metrics['hit_rate_25_75']:.0%}"
            f"  hit10-90={metrics['hit_rate_10_90']:.0%}"
            f"  dir_acc={metrics['direction_acc']:.0%}"
            f"  MAE={metrics['mae_pct']:.2f}%"
            f"  end_err={metrics['end_error_pct']:.2f}%"
        )
    title = (
        f"{args.symbol} | v5 | lookback={args.lookback}  forecast={args.forecast}  start={start_price:.2f}\n"
        f"rv={rv:.4f}  vol_scale={vol_scale:.2f}  vol_x={args.vol_multiplier}"
        f"  ds={args.drift_scale}  decay={args.drift_decay}" + hit_str
    )

    fwd_volume_norm = result.ohlcv_volume / (result.ohlcv_volume.mean() + 1e-8)
    chart_path = Path(str(out_prefix) + "_candles.png")
    fig = render_forecast_candles(
        hist_open=hist_open, hist_high=hist_high,
        hist_low=hist_low,   hist_close=hist_close,
        hist_volume=hist_volume_norm,
        fwd_open=result.ohlcv_open,   fwd_high=result.ohlcv_high,
        fwd_low=result.ohlcv_low,     fwd_close=result.ohlcv_close,
        fwd_volume=fwd_volume_norm,
        p25=result.p25, p75=result.p75,
        p10=result.p10, p90=result.p90,
        actual_open=actual_open   if len(actual_close) > 0 else None,
        actual_high=actual_high   if len(actual_close) > 0 else None,
        actual_low=actual_low     if len(actual_close) > 0 else None,
        actual_close=actual_close if len(actual_close) > 0 else None,
        actual_volume=actual_volume_norm if len(actual_close) > 0 else None,
        title=title,
        output_path=chart_path,
        volume_nodes=base_params.volume_nodes,
        hist_window=args.hist_window,
        actual_deviation=actual_deviation,
        metrics=metrics,
    )
    plt.close(fig)

    print(f"\n✔ Forward study v5 完成")
    print(f"  K 棒圖 : {chart_path}")
    print(f"  指標   : {metrics_path}")


if __name__ == "__main__":
    main()
