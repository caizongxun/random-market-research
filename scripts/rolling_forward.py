"""
rolling_forward.py

核心邏輪
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

方向命中定義（v2 修後）
------------------
「預測收盤相對前一根」 vs 「實際收盤相對前一根」
  - P-Ret: (pred_t - actual_{t-1}) / actual_{t-1}
  - A-Ret: (actual_t - actual_{t-1}) / actual_{t-1}
  - Dir 命中: sign(P-Ret) == sign(A-Ret)
兩者基準一致，即可相互比較。

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

  # 加入參數模型覆蓋 drift_scale / drift_decay：
  python scripts/rolling_forward.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --end-date 2025-01-01 \\
    --total-bars 30 \\
    --step 5 \\
    --auto-calibrate \\
    --param-model models/param_model_AAPL.joblib \\
    --verbose
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
# 參數模型載入（optional）
# ──────────────────────────────────────────────────────────────────────────────
def load_param_model(path: str | None):
    """載入 joblib 封裝的參數模型。若 path 為 None 回傳 None。"""
    if path is None:
        return None
    try:
        import joblib
    except ImportError:
        raise ImportError("需要 joblib：pip install joblib")
    payload = joblib.load(path)
    print(f"[param-model] 載入 {path}  "
          f"({payload['model_name']}, n_feat={len(payload['feature_cols'])})")
    return payload


# 複製自 collect_training_data.py 的特徵提取
def _garch_features(close_arr):
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
            "garch_alpha":        round(alpha, 4),
            "garch_gamma":        round(gamma, 4),
            "garch_beta":         round(beta, 4),
            "garch_persistence":  round(alpha + beta + 0.5 * gamma, 4),
            "garch_forecast_vol": round(fvol * 100, 4),
        }
    except Exception:
        return {}


def _extract_features(df_window: pd.DataFrame, theta_vol: float) -> dict:
    """與 collect_training_data.py 完全一致的特徵提取。"""
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

    body_pct  = np.abs(c - o) / (o + 1e-8)
    avg_body  = float(np.mean(body_pct))
    body_size = np.abs(c - o) + 1e-8
    sr        = (h - l) / body_size
    median_sr = float(np.median(sr))
    p90_sr    = float(np.percentile(sr, 90))

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
        "vol_20":             round(vol_20 * 100, 4),
        "vol_60":             round(vol_60 * 100, 4),
        "vol_all":            round(vol_all * 100, 4),
        "vol_ratio_20_60":    round(vol_20 / (vol_60 + 1e-8), 4),
        "vol_ratio_rv_theta": round(vol_20 / (theta_vol + 1e-8), 4),
        "drift_20":           round(drift_20 * 100, 4),
        "drift_60":           round(drift_60 * 100, 4),
        "drift_all":          round(drift_all * 100, 4),
        "drift_ratio_20_all": round(
            drift_20 / (abs(drift_all) + 1e-8) * np.sign(drift_all + 1e-12), 4
        ),
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
    feat.update(_garch_features(c))
    return feat


def apply_param_model(
    payload,
    df_window: pd.DataFrame,
    theta_vol: float,
    calib: dict,
    verbose: bool = False,
) -> dict:
    """
    用模型預測 drift_scale / drift_decay，覆蓋 auto_calibrate 輸出。
    momentum_boost 保留 auto_calibrate 的值（模型 CV MAE≈0，無信號）。
    """
    if payload is None:
        return calib

    feats = _extract_features(df_window, theta_vol)
    feature_cols = payload["feature_cols"]
    models       = payload["models"]

    # 對齊特徵順序，缺少的填 0
    x = np.array([[feats.get(fc.replace("feat_", ""), 0.0) for fc in feature_cols]])

    pred_ds = float(models["target_drift_scale"].predict(x)[0])
    pred_dd = float(models["target_drift_decay"].predict(x)[0])

    # 合理邊界防爆
    pred_ds = float(np.clip(pred_ds, 0.2, 4.0))
    pred_dd = float(np.clip(pred_dd, 0.01, 0.30))

    if verbose:
        print(f"    [param-model]  auto: ds={calib['drift_scale']:.3f}  dd={calib['drift_decay']:.3f}"
              f"  →  model: ds={pred_ds:.3f}  dd={pred_dd:.3f}")

    updated = dict(calib)
    updated["drift_scale"]  = pred_ds
    updated["drift_decay"]  = pred_dd
    return updated


# ──────────────────────────────────────────────────────────────────────────────
# 單輪預測
# ──────────────────────────────────────────────────────────────────────────────
def _run_one_step(
    close_hist: np.ndarray,
    theta: CalibratedTheta,
    args,
    full_df_up_to_now: pd.DataFrame,
    param_model_payload=None,
) -> tuple[np.ndarray, dict | None]:
    step = args.step

    calib = auto_calibrate(
        full_df_up_to_now["Close"].values.astype(float),
        theta,
        calib_window=args.calib_window,
        use_garch=(not args.no_garch) and _ARCH_AVAILABLE,
        garch_model=args.garch_model,
    )

    # ── 參數模型覆蓋 ──────────────────────────────────────────
    if param_model_payload is not None:
        calib_window_df = full_df_up_to_now.iloc[-args.calib_window:]
        calib = apply_param_model(
            param_model_payload,
            calib_window_df,
            theta.vol,
            calib,
            verbose=getattr(args, "verbose", False),
        )
    # ─────────────────────────────────────────────────────────

    intra_bar       = calib["intra_bar"]
    momentum_boost  = calib["momentum_boost"]
    drift_decay     = calib["drift_decay"]
    vol_multiplier  = calib["vol_multiplier"]
    drift_scale     = calib["drift_scale"]
    drift_clamp_max = calib["drift_clamp_max"]

    fitter     = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb_result  = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])

    log_rets    = np.diff(np.log(close_hist))
    short_drift = float(np.mean(log_rets[-5:])) if len(log_rets) >= 5 else last_drift
    w           = float(np.clip(args.short_drift_weight, 0.0, 1.0))
    blend_drift = float(np.clip(
        w * short_drift + (1.0 - w) * last_drift,
        -drift_clamp_max, drift_clamp_max
    ))

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

    drift_fwd = np.full(step, blend_drift)
    bb_fwd    = start_price * np.cumprod(1 + drift_fwd)
    vol_fwd   = np.full(step, last_vol) * vol_multiplier

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
        fast_decay     = min(drift_decay * 2.0, 0.99)
        cur            = blend_drift
        drift_schedule = np.empty(step)
        for t in range(step):
            drift_schedule[t] = cur
            cur *= (1.0 - (fast_decay if t < args.early_decay_bars else drift_decay))

    ESTIMATOR_LB = 500
    rv_window  = min(21, len(close_hist))
    rv         = float(np.std(np.diff(np.log(close_hist[-rv_window:]))))
    vol_scale  = float(np.clip(rv / max(theta.vol, 1e-8), 0.6, 4.0))

    est_df      = full_df_up_to_now.iloc[-ESTIMATOR_LB:]
    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(est_df, symbol="")
    params_fwd  = build_params_from_theta(theta, base_params)

    import dataclasses
    params_fwd = dataclasses.replace(params_fwd, last_close=start_price)

    if blend_drift < 0 and len(log_rets) >= args.reversal_window:
        if all(r > 0 for r in log_rets[-args.reversal_window:]):
            params_fwd = dataclasses.replace(params_fwd, momentum_bias=0.0)

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
    param_model_payload=None,
) -> dict:
    step       = args.step
    total_bars = args.total_bars
    lookback   = args.lookback

    pred_closes  : list[float] = []
    pred_ohlcs   : list[dict]  = []
    round_details: list[dict]  = []

    cur_end   = train_end_idx
    rounds    = int(np.ceil(total_bars / step))
    bars_done = 0

    actual_prev_closes: list[float] = []

    for rnd in range(rounds):
        need = min(step, total_bars - bars_done)
        if need <= 0:
            break

        win_start   = max(0, cur_end - lookback)
        close_hist  = df.iloc[win_start:cur_end]["Close"].values.astype(float)
        full_window = df.iloc[max(0, cur_end - 500): cur_end].copy()

        if len(close_hist) < 10:
            break

        start_price = float(close_hist[-1])
        print(f"  [輪 {rnd+1}/{rounds}]  訓練截至 idx={cur_end-1}  起點={start_price:.2f}  預測 {need} 根")

        rep, ohlc = _run_one_step(
            close_hist, theta, args, full_window,
            param_model_payload=param_model_payload,
        )
        rep = rep[:need]
        if ohlc is not None:
            ohlc = {k: v[:need] for k, v in ohlc.items()}

        actual_slice: np.ndarray | None = None
        if actual_future_df is not None:
            sl = actual_future_df.iloc[bars_done: bars_done + need]
            if len(sl) > 0:
                actual_slice = sl["Close"].values.astype(float)

        if actual_slice is not None and len(actual_slice) == need:
            prev_actual_for_dir = np.concatenate([[start_price], actual_slice[:-1]])
            mae_pct  = float(np.mean(
                np.abs(rep - actual_slice) / start_price * 100
            ))
            p_rets   = rep - prev_actual_for_dir
            a_rets   = actual_slice - prev_actual_for_dir
            dir_hits = int(np.sum(np.sign(p_rets) == np.sign(a_rets)))
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
        if actual_slice is not None and len(actual_slice) == need:
            actual_prev_closes.extend(
                np.concatenate([[start_price], actual_slice[:-1]]).tolist()
            )
        else:
            actual_prev_closes.extend([None] * need)

        cur_end   += need
        bars_done += need

    pred_arr  = np.array(pred_closes)
    actual_arr: np.ndarray | None = None
    if actual_future_df is not None and len(actual_future_df) >= total_bars:
        actual_arr = actual_future_df.iloc[:total_bars]["Close"].values.astype(float)
    elif actual_future_df is not None and len(actual_future_df) > 0:
        actual_arr = actual_future_df["Close"].values.astype(float)

    return {
        "pred_closes":        pred_arr,
        "pred_ohlcs":         pred_ohlcs,
        "actual_closes":      actual_arr,
        "actual_prev_closes": actual_prev_closes,
        "round_details":      round_details,
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
    label_suffix: str = "",
):
    pred   = result["pred_closes"]
    actual = result["actual_closes"]
    ohlcs  = result["pred_ohlcs"]
    total  = len(pred)

    fig, ax = plt.subplots(figsize=(16, 6))

    hist_x = np.arange(-len(hist_close), 0)
    ax.plot(hist_x, hist_close, color="#555", lw=1.2, label="歷史", zorder=4)

    palette = ["#26A69A", "#42A5F5", "#AB47BC", "#FF7043", "#66BB6A", "#FFA726"]
    for rnd_i, ohlc in enumerate(ohlcs):
        start_bar = rnd_i * step
        n         = min(step, total - start_bar)
        x_arr     = np.arange(start_bar, start_bar + n)
        bull      = palette[rnd_i % len(palette)]
        bear      = _darken(bull)

        if ohlc is not None:
            o = ohlc["open"][:n]
            h = ohlc["high"][:n]
            l = ohlc["low"][:n]
            c = ohlc["close"][:n]
        else:
            c    = pred[start_bar: start_bar + n]
            prev = hist_close[-1] if start_bar == 0 else pred[start_bar - 1]
            o    = np.concatenate([[prev], c[:-1]])
            sp   = c * 0.005
            h    = np.maximum(o, c) + sp
            l    = np.minimum(o, c) - sp

        _draw_candles(ax, x_arr, o, h, l, c,
                      bull_color=bull, bear_color=bear, alpha=0.85)

        if rnd_i > 0:
            ax.axvline(start_bar, color="#888", lw=0.6, ls=":", alpha=0.6)

    if actual is not None and len(actual) > 0:
        ax.plot(np.arange(len(actual)), actual,
                color="#43A047", lw=1.8, ls="--", label="實際收盤", zorder=6)

    for rnd_i in range(len(ohlcs)):
        x_mid = rnd_i * step + step / 2
        ylim  = ax.get_ylim()
        ax.text(x_mid, ylim[1] - (ylim[1] - ylim[0]) * 0.02,
                f"R{rnd_i+1}", ha="center", va="top",
                fontsize=7, color="#555", alpha=0.8)

    ax.axvline(0, color="#999", lw=0.8, ls=":")
    suffix_display = f" [{label_suffix}]" if label_suffix else ""
    ax.set_title(f"{symbol}  Rolling {step}-bar Forward  [{end_date_str}]{suffix_display}")
    ax.set_xlabel("交易日（相對預測起點）")
    ax.set_ylabel("價格")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    round_details   = result["round_details"]
    maes            = [r["mae_pct"] for r in round_details if r["mae_pct"] is not None]
    rounds_with_mae = [r["round"]   for r in round_details if r["mae_pct"] is not None]

    if maes:
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        avg       = np.mean(maes)
        colors2   = ["#EF5350" if m > avg else "#26A69A" for m in maes]
        ax2.bar(rounds_with_mae, maes, color=colors2, alpha=0.8, width=0.6)
        ax2.axhline(avg, color="#888", lw=1, ls="--",
                    label=f"平均 MAE={avg:.2f}%")
        ax2.set_title(f"{symbol}  每輪 MAE (close){suffix_display}")
        ax2.set_xlabel("輪次")
        ax2.set_ylabel("MAE %")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        out2 = Path(output_dir)
        out2.mkdir(parents=True, exist_ok=True)
        suffix_file = f"_{label_suffix}" if label_suffix else ""
        mae_path = out2 / f"{symbol}_{end_date_str}_rolling{step}{suffix_file}_mae.png"
        fig2.savefig(mae_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"✔ MAE 圖 → {mae_path}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix_file = f"_{label_suffix}" if label_suffix else ""
    fname = out / f"{symbol}_{end_date_str}_rolling{step}{suffix_file}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ 預測圖 → {fname}")
    return str(fname)


def _darken(hex_color: str, factor: float = 0.65) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Terminal 對照表
# ──────────────────────────────────────────────────────────────────────────────
def print_rolling_comparison(
    result: dict,
    start_price: float,
    symbol: str,
    step: int,
    label_suffix: str = "",
):
    pred            = result["pred_closes"]
    actual          = result["actual_closes"]
    actual_prev_arr = result.get("actual_prev_closes", [])
    n               = len(pred)

    suffix_display = f" [{label_suffix}]" if label_suffix else ""
    sep = "─" * 84
    print(f"\n{sep}")
    print(f"  Rolling {step}-bar Forward  —  {symbol}  (共 {n} 根 K 棒){suffix_display}")
    print(f"  Dir 定義: sign(pred - prev_actual) == sign(actual - prev_actual)")
    print(sep)
    hdr = (f"  {'Rnd':>3} {'Bar':>3}  "
           f"{'P-Close':>8} {'P-Ret%':>7}  "
           f"{'A-Close':>8} {'A-Ret%':>7}  "
           f"{'Err%':>7} {'Dir':>3}")
    print(hdr)
    print("─" * 84)

    global_i    = 0
    close_errs  = []
    dir_correct = 0
    total_bars_with_actual = 0

    for rnd_i, rnd in enumerate(result["round_details"]):
        bars_this = rnd["bars"]
        for b in range(bars_this):
            p_c      = float(pred[global_i])
            prev_act = float(actual_prev_arr[global_i]) if global_i < len(actual_prev_arr) and actual_prev_arr[global_i] is not None else None

            p_ret = (p_c - prev_act) / prev_act * 100 if prev_act else float("nan")

            if actual is not None and global_i < len(actual):
                a_c   = float(actual[global_i])
                a_ret = (a_c - prev_act) / prev_act * 100 if prev_act else float("nan")
                err   = (p_c - a_c) / a_c * 100
                close_errs.append(abs(err))
                total_bars_with_actual += 1

                p_up  = p_ret >= 0
                a_up  = a_ret >= 0
                match = "✓" if p_up == a_up else "✗"
                if p_up == a_up:
                    dir_correct += 1

                print(f"  {rnd_i+1:>3} {b+1:>3}  "
                      f"{p_c:>8.2f} {p_ret:>+7.2f}%  "
                      f"{a_c:>8.2f} {a_ret:>+7.2f}%  "
                      f"{err:>+7.2f}% {match:>3}")
            else:
                print(f"  {rnd_i+1:>3} {b+1:>3}  "
                      f"{p_c:>8.2f} {p_ret:>+7.2f}%  "
                      f"{'N/A':>8} {'N/A':>7}  "
                      f"{'N/A':>7} {'N/A':>3}")
            global_i += 1

        rnd_mae = rnd.get("mae_pct")
        rnd_dir = rnd.get("dir_hits")
        if rnd_mae is not None:
            print(f"  {'':>7}  → 本輪 MAE={rnd_mae:.2f}%  方向命中={rnd_dir}/{bars_this}")
        print("  " + "·" * 42)

    print(sep)
    if close_errs:
        tn = total_bars_with_actual
        print(f"  合計 {tn} 根  MAE={np.mean(close_errs):.2f}%  "
              f"MAX={np.max(close_errs):.2f}%  "
              f"方向命中={dir_correct}/{tn} ({dir_correct/tn*100:.0f}%)")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# parse_args
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",              required=True)
    p.add_argument("--theta",               required=True)
    p.add_argument("--end-date",            default=None)
    p.add_argument("--total-bars",          type=int,   default=30)
    p.add_argument("--step",                type=int,   default=5)
    p.add_argument("--lookback",            type=int,   default=120)
    p.add_argument("--n-paths",             type=int,   default=500)
    p.add_argument("--calib-window",        type=int,   default=500)
    p.add_argument("--no-garch",            action="store_true")
    p.add_argument("--garch-model",         default="gjr-garch",
                   choices=["gjr-garch", "garch"])
    p.add_argument("--auto-calibrate",      action="store_true", default=True)
    p.add_argument("--param-model",         default=None,
                   help="joblib 模型路徑，覆蓋 drift_scale / drift_decay")
    p.add_argument("--output-dir",          default="results")
    p.add_argument("--verbose",             action="store_true")
    p.add_argument("--short-drift-weight",  type=float, default=0.4)
    p.add_argument("--sr-window",           type=int,   default=90)
    p.add_argument("--sr-bins",             type=int,   default=40)
    p.add_argument("--sr-top-k",            type=int,   default=5)
    p.add_argument("--sr-pivot-order",      type=int,   default=5)
    p.add_argument("--sr-zone-pct",         type=float, default=0.035)
    p.add_argument("--no-sr",               action="store_true")
    p.add_argument("--dynamic-drift",       action="store_true", default=True)
    p.add_argument("--no-dynamic-drift",    dest="dynamic_drift", action="store_false")
    p.add_argument("--mr-rate",             type=float, default=0.08)
    p.add_argument("--trend-strength",      type=float, default=0.5)
    p.add_argument("--vol-refit",           type=int,   default=5)
    p.add_argument("--early-decay-bars",    type=int,   default=5)
    p.add_argument("--reversal-window",     type=int,   default=3)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    param_model_payload = load_param_model(getattr(args, "param_model", None))
    label_suffix = "param-model" if param_model_payload is not None else ""

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

    future_df        = df.iloc[train_end_idx: train_end_idx + args.total_bars].copy()
    train_df         = df.iloc[train_end_idx - args.lookback: train_end_idx]
    hist_close       = train_df["Close"].values.astype(float)
    start_price      = float(hist_close[-1])
    actual_future_df = future_df if len(future_df) > 0 else None

    end_date_str = end_dt.strftime("%Y-%m-%d")
    print(f"\n[rolling_forward]  {args.symbol}  起點={end_date_str}  "
          f"step={args.step}  total={args.total_bars}  "
          f"dynamic_drift={args.dynamic_drift}  "
          f"param_model={'ON' if param_model_payload else 'OFF'}")

    result = rolling_forward(
        df=df,
        train_end_idx=train_end_idx,
        theta=theta,
        args=args,
        actual_future_df=actual_future_df,
        param_model_payload=param_model_payload,
    )

    print_rolling_comparison(
        result=result,
        start_price=start_price,
        symbol=args.symbol,
        step=args.step,
        label_suffix=label_suffix,
    )

    render_rolling(
        symbol=args.symbol,
        hist_close=hist_close,
        result=result,
        step=args.step,
        end_date_str=end_date_str,
        output_dir=args.output_dir,
        label_suffix=label_suffix,
    )

    # 儲存 JSON
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol":        args.symbol,
        "end_date":      end_date_str,
        "step":          args.step,
        "total_bars":    args.total_bars,
        "start_price":   round(start_price, 4),
        "param_model":   str(args.param_model) if args.param_model else None,
        "round_details": result["round_details"],
        "dynamic_drift": args.dynamic_drift,
        "dir_definition": "sign(pred-prev_actual)==sign(actual-prev_actual)",
    }

    if result["actual_closes"] is not None:
        pred   = result["pred_closes"]
        actual = result["actual_closes"]
        prev_a = result["actual_prev_closes"]
        na     = min(len(pred), len(actual), len(prev_a))

        mae = float(np.mean(
            np.abs(pred[:na] - actual[:na]) / start_price * 100
        ))

        p_rets = np.array([
            pred[i] - prev_a[i]
            for i in range(na)
            if prev_a[i] is not None
        ])
        a_rets = np.array([
            actual[i] - prev_a[i]
            for i in range(na)
            if prev_a[i] is not None
        ])
        valid_n = len(p_rets)
        dirs    = int(np.sum(np.sign(p_rets) == np.sign(a_rets)))

        summary["overall_mae_pct"]  = round(mae, 4)
        summary["dir_accuracy_pct"] = round(dirs / valid_n * 100, 2) if valid_n > 0 else None
        print(f"\n  整體 MAE={mae:.2f}%   方向命中={dirs}/{valid_n} ({dirs/valid_n*100:.0f}%)")

    suffix_file = f"_{label_suffix}" if label_suffix else ""
    json_path = out_path / f"{args.symbol}_{end_date_str}_rolling{args.step}{suffix_file}.json"
    with open(json_path, "w") as fj:
        json.dump(summary, fj, indent=2, ensure_ascii=False)
    print(f"✔ 結果 JSON → {json_path}")


if __name__ == "__main__":
    main()
