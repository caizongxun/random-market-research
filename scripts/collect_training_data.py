"""
collect_training_data.py

對指定時間段做 Walk-Forward Grid Search：
  - 每隔 --step 根取一個預測起點
  - 對每個起點用 grid search 找讓 mae_pct 最小的 (drift_scale, drift_decay) 組合
  - momentum_boost 固定 0.8（grid search 結果全選 0.8，不納入搜尋）
  - 同時記錄前500根的特徵向量
  - 把 (特徵, 最佳參數) 寫進 CSV

用法：
  python scripts/collect_training_data.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --end-date 2024-12-31 \\
    --step 5 \\
    --forecast 30 \\
    --output results/training_data_AAPL.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_estimator import MarketParameterEstimator
from backbone_fitter import BackboneFitter
from calibrated_simulator import CalibratedTheta, build_params_from_theta
from us_equity_simulator import USStockFutureSimulator


# ─────────────────────────────────────────────────────────────
# Grid 定義
# ─────────────────────────────────────────────────────────────
DRIFT_SCALE_GRID    = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.2]
MOMENTUM_BOOST_FIXED = 0.8            # grid search 結果全選 0.8，固定不搜尋
DRIFT_DECAY_GRID    = [0.03, 0.05, 0.07, 0.10, 0.13]


# ─────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────
def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def garch_features(close_arr):
    """回傳 GARCH 特徵，失敗時回傳空 dict。"""
    try:
        from arch import arch_model
        rets = pd.Series(np.diff(np.log(close_arr)) * 100).dropna()
        if len(rets) < 60:
            return {}
        am  = arch_model(rets, vol="Garch", p=1, o=1, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
        p   = res.params
        alpha = float(p.get("alpha[1]", 0))
        gamma = float(p.get("gamma[1]", 0))
        beta  = float(p.get("beta[1]",  0))
        fc    = res.forecast(horizon=1, reindex=False)
        fvol  = float(np.sqrt(fc.variance.values[-1, 0])) / 100
        return {
            "garch_alpha":       round(alpha, 4),
            "garch_gamma":       round(gamma, 4),
            "garch_beta":        round(beta, 4),
            "garch_persistence": round(alpha + beta + 0.5 * gamma, 4),
            "garch_forecast_vol": round(fvol * 100, 4),
        }
    except Exception:
        return {}


def extract_features(df_window, theta_vol):
    """從一段 OHLCV 視窗計算特徵向量。"""
    c = df_window["Close"].values.astype(float)
    o = df_window["Open"].values.astype(float)
    h = df_window["High"].values.astype(float)
    l = df_window["Low"].values.astype(float)
    v = df_window["Volume"].values.astype(float)

    log_rets = np.diff(np.log(c))

    vol_20  = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else float(np.std(log_rets))
    vol_60  = float(np.std(log_rets[-60:])) if len(log_rets) >= 60 else float(np.std(log_rets))
    vol_all = float(np.std(log_rets))

    drift_20  = float(np.mean(log_rets[-20:])) if len(log_rets) >= 20 else float(np.mean(log_rets))
    drift_60  = float(np.mean(log_rets[-60:])) if len(log_rets) >= 60 else float(np.mean(log_rets))
    drift_all = float(np.mean(log_rets))

    s = pd.Series(log_rets)
    ret_autocorr = float(s.autocorr(lag=1)) if len(s) > 10 else 0.0
    ret_autocorr = 0.0 if np.isnan(ret_autocorr) else ret_autocorr

    vret = pd.Series(v).pct_change().dropna()
    vol_autocorr = float(vret.autocorr(lag=1)) if len(vret) > 10 else 0.0
    vol_autocorr = max(0.0, 0.0 if np.isnan(vol_autocorr) else vol_autocorr)

    body_pct   = np.abs(c - o) / (o + 1e-8)
    avg_body   = float(np.mean(body_pct))
    body_size  = np.abs(c - o) + 1e-8
    sr         = (h - l) / body_size
    median_sr  = float(np.median(sr))
    p90_sr     = float(np.percentile(sr, 90))

    # 骨幹特徵
    try:
        fitter = BackboneFitter(n_seg=6, smooth_reg=0.5)
        bb = fitter.fit(c[-120:] if len(c) >= 120 else c)
        bb_last_drift = float(bb.segment_drifts[-1])
        bb_last_vol   = float(bb.segment_vols[-1])
        bb_drift_std  = float(np.std(bb.segment_drifts))
        bb_vol_std    = float(np.std(bb.segment_vols))
    except Exception:
        bb_last_drift = drift_all
        bb_last_vol   = vol_all
        bb_drift_std  = 0.0
        bb_vol_std    = 0.0

    feat = {
        "vol_20":          round(vol_20 * 100, 4),
        "vol_60":          round(vol_60 * 100, 4),
        "vol_all":         round(vol_all * 100, 4),
        "vol_ratio_20_60": round(vol_20 / (vol_60 + 1e-8), 4),
        "vol_ratio_rv_theta": round(vol_20 / (theta_vol + 1e-8), 4),
        "drift_20":        round(drift_20 * 100, 4),
        "drift_60":        round(drift_60 * 100, 4),
        "drift_all":       round(drift_all * 100, 4),
        "drift_ratio_20_all": round(drift_20 / (abs(drift_all) + 1e-8) * np.sign(drift_all + 1e-12), 4),
        "ret_autocorr":    round(ret_autocorr, 4),
        "vol_autocorr":    round(vol_autocorr, 4),
        "avg_body_pct":    round(avg_body * 100, 4),
        "median_sr":       round(median_sr, 4),
        "p90_sr":          round(p90_sr, 4),
        "bb_last_drift":   round(bb_last_drift * 100, 4),
        "bb_last_vol":     round(bb_last_vol * 100, 4),
        "bb_drift_std":    round(bb_drift_std * 100, 4),
        "bb_vol_std":      round(bb_vol_std * 100, 4),
    }
    feat.update(garch_features(c))
    return feat


def run_sim_and_mae(
    df, train_end_idx, theta, args,
    drift_scale, momentum_boost, drift_decay,
    vol_multiplier, n_paths=200,
):
    """跑一次模擬，回傳 mae_pct（越小越好）。"""
    ESTIMATOR_LB = 500
    train_df    = df.iloc[train_end_idx - args.lookback: train_end_idx]
    estimate_df = df.iloc[train_end_idx - ESTIMATOR_LB: train_end_idx]
    future_df   = df.iloc[train_end_idx: train_end_idx + args.forecast]

    if len(future_df) < args.forecast:
        return None

    close_hist  = train_df["Close"].values.astype(float)
    start_price = float(close_hist[-1])
    actual_close = future_df["Close"].values.astype(float)

    fitter    = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb_result = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol) * vol_multiplier
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

    rv = float(np.std(np.diff(np.log(close_hist[-21:]))))
    vol_scale = float(np.clip(rv / max(theta.vol, 1e-8), 0.6, 4.0))

    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = dataclasses.replace(
        base_params, last_close=start_price,
        momentum_bias=0.0, node_breakout_state=0
    )
    params_fwd = build_params_from_theta(theta, params_fwd)

    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=args.forecast,
        n_paths=n_paths,
        seed=42,
        vol_scale=vol_scale,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength * momentum_boost,
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_fwd,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=0.06,
        intra_bar_steps=3,
        drift_decay_rate=drift_decay,
        drift_scale=drift_scale,
        momentum_anchor_weight=0.45,
    )
    result = sim.simulate()
    n = min(len(actual_close), len(result.median_path))
    mae = float(np.mean(np.abs(actual_close[:n] - result.median_path[:n]) / start_price * 100))
    return mae


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",    required=True)
    p.add_argument("--theta",     required=True)
    p.add_argument("--end-date",  default=None,
                   help="資料截止日（預測最後起點），預設今天")
    p.add_argument("--lookback",  type=int, default=120)
    p.add_argument("--forecast",  type=int, default=30)
    p.add_argument("--step",      type=int, default=5,
                   help="每隔幾根取一個起點")
    p.add_argument("--n-paths",   type=int, default=200,
                   help="grid search 用的路徑數（小一點快很多）")
    p.add_argument("--calib-window", type=int, default=500)
    p.add_argument("--output",    default="results/training_data.csv")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    ESTIMATOR_LB = 500
    MIN_BEFORE = ESTIMATOR_LB + args.lookback

    print(f"Downloading {args.symbol}...")
    end_dt   = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today()
    start_dt = end_dt - pd.DateOffset(years=5)
    dl_end   = end_dt + pd.DateOffset(days=args.forecast * 2 + 10)

    df_raw = yf.download(
        args.symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=dl_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    print(f"Total bars: {len(df)}")

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        dates = pd.to_datetime(df["Datetime"])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])

    mask = dates <= end_dt
    if not mask.any():
        raise ValueError(f"{end_dt.date()} 之前找不到資料")
    max_end_idx = int(mask.values.nonzero()[0][-1]) + 1

    # 收集所有候選起點（不重複）
    seen = set()
    candidates = []
    idx = max_end_idx
    while idx >= MIN_BEFORE:
        if idx + args.forecast <= len(df) and idx not in seen:
            candidates.append(idx)
            seen.add(idx)
        idx -= args.step
    candidates = list(reversed(candidates))
    print(f"候選起點數: {len(candidates)}")

    grid = list(product(DRIFT_SCALE_GRID, DRIFT_DECAY_GRID))
    total_combos = len(grid)
    print(f"Grid 大小: {total_combos} 組  x  {len(candidates)} 起點 = {total_combos * len(candidates)} 次模擬")
    print(f"  (momentum_boost 固定 {MOMENTUM_BOOST_FIXED}，不納入 grid search)")
    print("開始收集... (每個起點約需數秒)\n")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    FEATURE_KEYS = None

    for i, train_end_idx in enumerate(candidates):
        date_str = str(dates.iloc[train_end_idx - 1].date())
        _row_prefix = f"[{i+1}/{len(candidates)}] 起點 {date_str} (idx={train_end_idx})  "

        # 1. 計算特徵
        calib_df = df.iloc[train_end_idx - args.calib_window: train_end_idx]
        feats = extract_features(calib_df, theta.vol)
        if FEATURE_KEYS is None:
            FEATURE_KEYS = list(feats.keys())

        # 2. auto-calibrate 的 vol_multiplier（固定用，不 grid search）
        log_rets = np.diff(np.log(
            df.iloc[train_end_idx - args.calib_window: train_end_idx]["Close"].values.astype(float)
        ))
        rv = float(np.std(log_rets))
        vol_multiplier = float(np.clip(rv / max(theta.vol, 1e-8), 0.5, 3.0))

        # 3. grid search（只搜 drift_scale x drift_decay）
        best_mae    = float("inf")
        best_params = {"drift_scale": 1.0, "momentum_boost": MOMENTUM_BOOST_FIXED, "drift_decay": 0.07}

        for ds, dd in grid:
            mae = run_sim_and_mae(
                df, train_end_idx, theta, args,
                drift_scale=ds, momentum_boost=MOMENTUM_BOOST_FIXED, drift_decay=dd,
                vol_multiplier=vol_multiplier, n_paths=args.n_paths,
            )
            if mae is not None and mae < best_mae:
                best_mae    = mae
                best_params = {"drift_scale": ds, "momentum_boost": MOMENTUM_BOOST_FIXED, "drift_decay": dd}

        print(_row_prefix + f"best_mae={best_mae:.3f}%  ds={best_params['drift_scale']}  "
              f"dd={best_params['drift_decay']}")

        row = {
            "date":            date_str,
            "train_end_idx":   train_end_idx,
            "best_mae":        round(best_mae, 4),
            **{f"feat_{k}": v for k, v in feats.items()},
            "target_drift_scale":    best_params["drift_scale"],
            "target_momentum_boost": best_params["momentum_boost"],
            "target_drift_decay":    best_params["drift_decay"],
        }
        rows.append(row)

    # 寫 CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✔ 已儲存 {len(rows)} 筆訓練資料 → {out_path}")
    else:
        print("\n⚠ 沒有產生任何資料，請確認時間範圍與資料筆數")


if __name__ == "__main__":
    main()
