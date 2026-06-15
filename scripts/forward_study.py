"""
forward_study.py  v11.2  (GJR-GARCH + auto-calibrate + OHLC K線預測)

v11.2 新增：
  1. blend_drift 動態截斷 — MAX = 0.5 × σ_garch（無 GARCH 時 fallback 到 0.5 × rv）
     防止 short_drift(5d) 在短暫回調時把 drift 拉爆

v11.1 已有：
  1. print_ohlc_comparison() — 每根 K 棒對照表
  2. 修正實際收盤線消失問題

v11 已有：
  1. momentum_bias + node_breakout_state
  2. backbone drift = 0.4 × short_drift(5d) + 0.6 × last_drift

v10 已有：
  1. render_forecast 手繪 K 線
  2. pick_representative_path 從 SimulationResult 直接讀 OHLC

用法：
  python scripts/forward_study.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --auto-calibrate \\
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

try:
    from arch import arch_model as _arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# GJR-GARCH 波動率預測
# ──────────────────────────────────────────────────────────────────────────────
def garch_vol_forecast(close_arr, model_type: str = "gjr-garch"):
    if not _ARCH_AVAILABLE:
        return None, {"error": "arch not installed"}
    try:
        rets = pd.Series(np.diff(np.log(close_arr)) * 100).dropna()
        if len(rets) < 60:
            return None, {"error": f"too few returns: {len(rets)}"}
        o_val = 1 if model_type == "gjr-garch" else 0
        am  = _arch_model(rets, vol="Garch", p=1, o=o_val, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
        p   = res.params
        alpha = float(p.get("alpha[1]", 0))
        gamma = float(p.get("gamma[1]", 0))
        beta  = float(p.get("beta[1]",  0))
        persistence = alpha + beta + 0.5 * gamma
        fc   = res.forecast(horizon=1, reindex=False)
        fvol = float(np.sqrt(fc.variance.values[-1, 0])) / 100
        info = {
            "alpha": round(alpha, 4), "gamma": round(gamma, 4),
            "beta":  round(beta, 4),  "persistence": round(persistence, 4),
            "forecast_vol_pct": round(fvol * 100, 4),
        }
        return fvol, info
    except Exception as e:
        return None, {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 輔助
# ──────────────────────────────────────────────────────────────────────────────
def ensure_ohlcv(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().reset_index()


def pick_representative_path(result, paths_fallback: np.ndarray | None = None):
    rep = getattr(result, "representative_path", None)
    if rep is not None and len(rep) > 0:
        o = getattr(result, "ohlcv_open",  None)
        h = getattr(result, "ohlcv_high",  None)
        l = getattr(result, "ohlcv_low",   None)
        c = getattr(result, "ohlcv_close", None)
        if (o is not None and len(o) > 0 and
                h is not None and len(h) > 0 and
                l is not None and len(l) > 0 and
                c is not None and len(c) > 0):
            return np.asarray(rep), {
                "open":  np.asarray(o),
                "high":  np.asarray(h),
                "low":   np.asarray(l),
                "close": np.asarray(c),
            }
        return np.asarray(rep), None

    paths = paths_fallback
    if paths is None:
        paths = getattr(result, "future_paths", None)
    if paths is None or len(paths) == 0:
        return np.array([]), None

    paths = np.asarray(paths)
    pointwise_median = np.median(paths, axis=0)
    maes = np.mean(np.abs(paths - pointwise_median), axis=1)
    best_idx = int(np.argmin(maes))
    return paths[best_idx], None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",         required=True)
    p.add_argument("--theta",          required=True)
    p.add_argument("--end-date",       default=None)
    p.add_argument("--lookback",       type=int,   default=120)
    p.add_argument("--forecast",       type=int,   default=30)
    p.add_argument("--n-paths",        type=int,   default=500)
    p.add_argument("--intra-bar",      type=int,   default=None)
    p.add_argument("--shadow-noise",   type=float, default=None)
    p.add_argument("--shadow-clamp",   type=float, default=None)
    p.add_argument("--momentum-boost", type=float, default=None)
    p.add_argument("--drift-decay",    type=float, default=None)
    p.add_argument("--vol-multiplier", type=float, default=None)
    p.add_argument("--drift-scale",    type=float, default=None)
    p.add_argument("--body-scale-max", type=float, default=None)
    p.add_argument("--auto-calibrate",  action="store_true")
    p.add_argument("--calib-window",   type=int,   default=500)
    p.add_argument("--no-garch",        action="store_true")
    p.add_argument("--garch-model",    default="gjr-garch",
                   choices=["gjr-garch", "garch"])
    p.add_argument("--param-model",    default=None)
    p.add_argument("--output-dir",     default="results")
    p.add_argument("--verbose",         action="store_true")
    p.add_argument("--short-drift-weight", type=float, default=0.4,
                   help="近期 5 日 drift 在 backbone 混合中的權重 (0~1)")
    p.add_argument("--drift-clamp",    type=float, default=None,
                   help="手動指定 blend_drift 截斷上限 (絕對值, 每日)。不指定則動態用 0.5×σ_garch")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# auto_calibrate
# ──────────────────────────────────────────────────────────────────────────────
def auto_calibrate(
    close_arr,
    theta: CalibratedTheta,
    calib_window: int = 500,
    use_garch: bool = True,
    garch_model: str = "gjr-garch",
):
    c = close_arr[-calib_window:] if len(close_arr) > calib_window else close_arr
    log_rets = np.diff(np.log(c))
    rv = float(np.std(log_rets))

    garch_vol, garch_info = None, {}
    vol_source = "rv"
    if use_garch and _ARCH_AVAILABLE:
        garch_vol, garch_info = garch_vol_forecast(c, model_type=garch_model)
        if garch_vol is not None and "error" not in garch_info:
            vol_source = "garch"

    base_vol = garch_vol if vol_source == "garch" else rv
    vol_multiplier = float(np.clip(base_vol / max(theta.vol, 1e-8), 0.5, 3.0))

    body_pct = np.abs(np.diff(c)) / (c[:-1] + 1e-8)
    avg_body = float(np.mean(body_pct))
    intra_bar = 3 if avg_body > 0.012 else 2

    shadow_noise   = float(np.clip(rv * 4.0, 0.08, 0.30))
    shadow_clamp   = 2.5 if rv > 0.018 else 2.0
    momentum_boost = 0.8
    drift_decay    = float(np.clip(0.02 + rv * 1.5, 0.03, 0.15))
    drift_scale    = float(np.clip(rv / max(theta.vol, 1e-8), 0.4, 3.0))

    # 動態 drift 截斷上限：0.5 × σ (每日)
    drift_clamp_max = float(base_vol * 0.5)

    return {
        "intra_bar":        intra_bar,
        "shadow_noise":     round(shadow_noise, 3),
        "shadow_clamp":     round(shadow_clamp, 2),
        "momentum_boost":   momentum_boost,
        "drift_decay":      round(drift_decay, 4),
        "vol_multiplier":   round(vol_multiplier, 4),
        "drift_scale":      round(drift_scale, 4),
        "drift_clamp_max":  round(drift_clamp_max, 5),
        "_vol_source":      vol_source,
        "_avg_body_pct":    round(avg_body * 100, 3),
        "_garch_info":      garch_info,
        "_base_vol":        round(float(base_vol), 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# print_diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def print_diagnostics(args, calib_info: dict | None, model_predicted_params: dict | None,
                      momentum_bias: float = 0.0, breakout_state: int = 0,
                      blend_drift: float = 0.0, short_drift: float = 0.0,
                      last_drift: float = 0.0, drift_clamp_max: float | None = None):
    print("\n" + "─" * 60)
    print("  VERBOSE DIAGNOSTICS (v11.2)")
    print("─" * 60)
    if calib_info:
        gi  = calib_info.get("_garch_info", {})
        vs  = calib_info.get("_vol_source", "?")
        abp = calib_info.get("_avg_body_pct", "?")
        print(f"  vol_source    : {vs}")
        print(f"  avg_body_pct  : {abp}%")
        if vs == "garch" and gi and "error" not in gi:
            print(f"  GJR-GARCH     : σ={gi.get('forecast_vol_pct','?')}%/day  "
                  f"α={gi.get('alpha','?')}  γ={gi.get('gamma','?')}  "
                  f"β={gi.get('beta','?')}  persist={gi.get('persistence','?')}")
        elif "error" in gi:
            print(f"  GARCH error   : {gi['error']}")
    print(f"\n  [動能狀態]")
    print(f"  momentum_bias   : {momentum_bias:+.4f}  (estimator 原值，不歸零)")
    print(f"  breakout_state  : {breakout_state}")
    print(f"\n  [backbone drift 混合]")
    print(f"  short_drift(5d) : {short_drift*100:+.4f}%/day")
    print(f"  last_drift(bb)  : {last_drift*100:+.4f}%/day")
    raw_blend = args.short_drift_weight * short_drift + (1 - args.short_drift_weight) * last_drift
    clamp_str = f"  → 截斷至 {blend_drift*100:+.4f}%/day  (MAX=±{drift_clamp_max*100:.4f}%)" \
        if drift_clamp_max and abs(raw_blend) > drift_clamp_max else ""
    print(f"  blend_drift     : {blend_drift*100:+.4f}%/day  "
          f"(w={args.short_drift_weight:.1f}×short + {1-args.short_drift_weight:.1f}×long)"
          + clamp_str)
    if drift_clamp_max:
        print(f"  drift_clamp_max : ±{drift_clamp_max*100:.4f}%/day  (0.5×σ_garch)")
    print(f"\n  [最終模擬參數]")
    print(f"  intra_bar       : {args.intra_bar}")
    print(f"  momentum_boost  : {args.momentum_boost}")
    print(f"  drift_decay     : {args.drift_decay}")
    print(f"  vol_multiplier  : {args.vol_multiplier}")
    print(f"  drift_scale     : {args.drift_scale}")
    if model_predicted_params:
        print(f"\n  [param-model 覆蓋]")
        print(f"  drift_scale   : {model_predicted_params.get('drift_scale', 'N/A')}")
        print(f"  drift_decay   : {model_predicted_params.get('drift_decay', 'N/A')}")
    print("─" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# print_ohlc_comparison
# ──────────────────────────────────────────────────────────────────────────────
def print_ohlc_comparison(
    pred_ohlc: dict | None,
    pred_close: np.ndarray,
    actual_df: pd.DataFrame | None,
    start_price: float,
    symbol: str,
):
    n = len(pred_close)
    has_actual = actual_df is not None and len(actual_df) >= 1
    has_pred_ohlc = pred_ohlc is not None

    sep = "─" * 100
    print(f"\n{sep}")
    print(f"  OHLC 對照表 — {symbol}  (共 {n} 根 K 棒)")
    print(sep)

    hdr = (f"  {'Bar':>3}  "
           f"{'P-Open':>8} {'P-High':>8} {'P-Low':>8} {'P-Close':>8} {'P-Ret%':>7}  "
           f"{'A-Open':>8} {'A-High':>8} {'A-Low':>8} {'A-Close':>8} {'A-Ret%':>7}  "
           f"{'ClsErr%':>8} {'Dir':>4}")
    print(hdr)
    print("─" * 100)

    prev_pred_c  = start_price
    prev_act_c   = start_price
    cum_pred_ret = 0.0
    cum_act_ret  = 0.0

    close_errs = []

    for i in range(n):
        p_c = float(pred_close[i])
        p_o = float(pred_ohlc["open"][i])  if has_pred_ohlc else float(pred_close[i - 1] if i > 0 else start_price)
        p_h = float(pred_ohlc["high"][i])  if has_pred_ohlc else p_c * 1.005
        p_l = float(pred_ohlc["low"][i])   if has_pred_ohlc else p_c * 0.995

        p_ret = (p_c - prev_pred_c) / prev_pred_c * 100

        if has_actual and i < len(actual_df):
            row   = actual_df.iloc[i]
            a_o   = float(row.get("Open",  row.get("open",  p_o)))
            a_h   = float(row.get("High",  row.get("high",  p_h)))
            a_l   = float(row.get("Low",   row.get("low",   p_l)))
            a_c   = float(row.get("Close", row.get("close", p_c)))
            a_ret = (a_c - prev_act_c) / prev_act_c * 100

            cls_err = (p_c - a_c) / a_c * 100
            close_errs.append(abs(cls_err))

            p_dir = "+" if p_c >= prev_pred_c else "-"
            a_dir = "+" if a_c >= prev_act_c  else "-"
            dir_match = "✓" if p_dir == a_dir else "✗"

            print(f"  {i+1:>3}  "
                  f"{p_o:>8.2f} {p_h:>8.2f} {p_l:>8.2f} {p_c:>8.2f} {p_ret:>+7.2f}%  "
                  f"{a_o:>8.2f} {a_h:>8.2f} {a_l:>8.2f} {a_c:>8.2f} {a_ret:>+7.2f}%  "
                  f"{cls_err:>+8.2f}% {dir_match:>4}")

            prev_act_c  = a_c
            cum_act_ret = (a_c - start_price) / start_price * 100
        else:
            print(f"  {i+1:>3}  "
                  f"{p_o:>8.2f} {p_h:>8.2f} {p_l:>8.2f} {p_c:>8.2f} {p_ret:>+7.2f}%  "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>7}  "
                  f"{'N/A':>8} {'N/A':>4}")

        prev_pred_c = p_c
        cum_pred_ret = (p_c - start_price) / start_price * 100

    print("─" * 100)
    print(f"  累計預測: {cum_pred_ret:+.2f}%   "
          + (f"累計實際: {cum_act_ret:+.2f}%   "
             f"MAE(close): {float(np.mean(close_errs)):.2f}%   "
             f"MAX(close): {float(np.max(close_errs)):.2f}%"
             if close_errs else "實際數據不足，無法計算"))
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 手繪 K 線
# ──────────────────────────────────────────────────────────────────────────────
def _draw_candles(ax, x_arr, o_arr, h_arr, l_arr, c_arr,
                  width=0.6,
                  bull_color="#26A69A", bear_color="#EF5350",
                  alpha=0.9):
    for xi, o, h, l, c in zip(x_arr, o_arr, h_arr, l_arr, c_arr):
        color  = bull_color if c >= o else bear_color
        ax.plot([xi, xi], [l, h], color=color, lw=0.9, alpha=alpha, zorder=2)
        body_y = min(o, c)
        body_h = abs(c - o)
        if body_h < 1e-8:
            ax.plot([xi - width / 2, xi + width / 2],
                    [c, c], color=color, lw=1.2, alpha=alpha, zorder=3)
        else:
            rect = mpatches.FancyBboxPatch(
                (xi - width / 2, body_y), width, body_h,
                boxstyle="square,pad=0",
                linewidth=0.4,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
                zorder=3,
            )
            ax.add_patch(rect)


# ──────────────────────────────────────────────────────────────────────────────
# render_forecast
# ──────────────────────────────────────────────────────────────────────────────
def render_forecast(
    symbol, hist_close, result, forecast, end_date_str,
    output_dir, mode_tag,
    rep_close, ohlc,
    actual_close=None,
    actual_ohlc: dict | None = None,
):
    fig, ax = plt.subplots(figsize=(15, 6))

    hist_x = np.arange(-len(hist_close), 0)
    ax.plot(hist_x, hist_close, color="#555555", lw=1.2, label="歷史收盤", zorder=4)

    fwd_x = np.arange(0, forecast)
    if hasattr(result, "p10") and result.p10 is not None:
        ax.fill_between(fwd_x, result.p10[:forecast], result.p90[:forecast],
                        alpha=0.12, color="#2196F3", label="P10-P90")
        ax.fill_between(fwd_x, result.p25[:forecast], result.p75[:forecast],
                        alpha=0.22, color="#2196F3", label="P25-P75")

    n = min(len(rep_close), forecast)
    x_candle = np.arange(n)

    if ohlc is not None:
        o_arr = ohlc["open"][:n]
        h_arr = ohlc["high"][:n]
        l_arr = ohlc["low"][:n]
        c_arr = ohlc["close"][:n]
    else:
        c_arr = rep_close[:n]
        o_arr = np.concatenate([[hist_close[-1]], c_arr[:-1]])
        spread = c_arr * 0.005
        h_arr  = np.maximum(o_arr, c_arr) + spread
        l_arr  = np.minimum(o_arr, c_arr) - spread

    _draw_candles(ax, x_candle, o_arr, h_arr, l_arr, c_arr)

    bull_patch = mpatches.Patch(color="#26A69A", label="預測漲K")
    bear_patch = mpatches.Patch(color="#EF5350", label="預測跌K")

    _actual_c = None
    if actual_ohlc is not None and "close" in actual_ohlc:
        _actual_c = np.asarray(actual_ohlc["close"])
    elif actual_close is not None:
        _actual_c = np.asarray(actual_close)

    if _actual_c is not None and len(_actual_c) > 0:
        na = min(len(_actual_c), forecast)
        ax.plot(fwd_x[:na], _actual_c[:na],
                color="#43A047", lw=1.5, ls="--", label="實際收盤", zorder=5)

    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set_title(f"{symbol}  {end_date_str}  [{mode_tag}]")
    ax.set_xlabel("交易日（相對預測起點）")
    ax.set_ylabel("價格")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [bull_patch, bear_patch],
              labels  + ["預測漲K", "預測跌K"],
              loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    out   = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{symbol}_{end_date_str}_{mode_tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ 圖表已儲存 → {fname}")
    return str(fname)


# ──────────────────────────────────────────────────────────────────────────────
# 特徵提取
# ──────────────────────────────────────────────────────────────────────────────
def extract_features_for_model(df_window, theta_vol):
    c     = df_window["Close"].values.astype(float)
    o_arr = df_window["Open"].values.astype(float)
    h     = df_window["High"].values.astype(float)
    l     = df_window["Low"].values.astype(float)
    v     = df_window["Volume"].values.astype(float)

    log_rets  = np.diff(np.log(c))
    vol_20    = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else float(np.std(log_rets))
    vol_60    = float(np.std(log_rets[-60:])) if len(log_rets) >= 60 else float(np.std(log_rets))
    vol_all   = float(np.std(log_rets))
    drift_20  = float(np.mean(log_rets[-20:])) if len(log_rets) >= 20 else float(np.mean(log_rets))
    drift_60  = float(np.mean(log_rets[-60:])) if len(log_rets) >= 60 else float(np.mean(log_rets))
    drift_all = float(np.mean(log_rets))

    s = pd.Series(log_rets)
    ret_autocorr = float(s.autocorr(lag=1)) if len(s) > 10 else 0.0
    ret_autocorr = 0.0 if np.isnan(ret_autocorr) else ret_autocorr

    vret = pd.Series(v).pct_change().dropna()
    vol_autocorr = float(vret.autocorr(lag=1)) if len(vret) > 10 else 0.0
    vol_autocorr = max(0.0, 0.0 if np.isnan(vol_autocorr) else vol_autocorr)

    body_pct  = np.abs(c - o_arr) / (o_arr + 1e-8)
    avg_body  = float(np.mean(body_pct))
    body_size = np.abs(c - o_arr) + 1e-8
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
        "feat_vol_20":             round(vol_20 * 100, 4),
        "feat_vol_60":             round(vol_60 * 100, 4),
        "feat_vol_all":            round(vol_all * 100, 4),
        "feat_vol_ratio_20_60":    round(vol_20 / (vol_60 + 1e-8), 4),
        "feat_vol_ratio_rv_theta": round(vol_20 / (theta_vol + 1e-8), 4),
        "feat_drift_20":           round(drift_20 * 100, 4),
        "feat_drift_60":           round(drift_60 * 100, 4),
        "feat_drift_all":          round(drift_all * 100, 4),
        "feat_drift_ratio_20_all": round(drift_20 / (abs(drift_all) + 1e-8) * np.sign(drift_all + 1e-12), 4),
        "feat_ret_autocorr":       round(ret_autocorr, 4),
        "feat_vol_autocorr":       round(vol_autocorr, 4),
        "feat_avg_body_pct":       round(avg_body * 100, 4),
        "feat_median_sr":          round(median_sr, 4),
        "feat_p90_sr":             round(p90_sr, 4),
        "feat_bb_last_drift":      round(bb_last_drift * 100, 4),
        "feat_bb_last_vol":        round(bb_last_vol * 100, 4),
        "feat_bb_drift_std":       round(bb_drift_std * 100, 4),
        "feat_bb_vol_std":         round(bb_vol_std * 100, 4),
    }

    if _ARCH_AVAILABLE:
        try:
            rets = pd.Series(np.diff(np.log(c)) * 100).dropna()
            if len(rets) >= 60:
                am  = _arch_model(rets, vol="Garch", p=1, o=1, q=1, dist="t")
                res = am.fit(disp="off", show_warning=False)
                pp  = res.params
                alpha = float(pp.get("alpha[1]", 0))
                gamma = float(pp.get("gamma[1]", 0))
                beta  = float(pp.get("beta[1]",  0))
                fc    = res.forecast(horizon=1, reindex=False)
                fvol  = float(np.sqrt(fc.variance.values[-1, 0])) / 100
                feat.update({
                    "feat_garch_alpha":        round(alpha, 4),
                    "feat_garch_gamma":        round(gamma, 4),
                    "feat_garch_beta":         round(beta, 4),
                    "feat_garch_persistence":  round(alpha + beta + 0.5 * gamma, 4),
                    "feat_garch_forecast_vol": round(fvol * 100, 4),
                })
        except Exception:
            pass

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    with open(args.theta) as f:
        theta = CalibratedTheta.from_dict(json.load(f))

    _pm_pipelines    = None
    _pm_feat_cols    = []
    _pm_target_names = []

    if args.param_model:
        import joblib
        _payload = joblib.load(args.param_model)
        if isinstance(_payload, dict) and "models" in _payload:
            _pm_pipelines    = _payload["models"]
            _pm_feat_cols    = _payload.get("feature_cols", [])
            _pm_target_names = _payload.get("targets", [])
        else:
            raise ValueError(
                f"param_model 格式不符：預期 dict{{'models':...}}，實際為 {type(_payload)}。"
            )
        meta_path = Path(args.param_model).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                param_meta = json.load(f)
            _pm_feat_cols    = param_meta.get("feature_cols", _pm_feat_cols)
            _pm_target_names = param_meta.get("targets",      _pm_target_names)
        print(f"[param-model] 已載入 {args.param_model}")
        print(f"  targets: {_pm_target_names}")
        print(f"  features: {len(_pm_feat_cols)} 個")

    # ── 下載資料 ──
    end_dt   = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.today()
    start_dt = end_dt - pd.DateOffset(years=3)
    dl_end   = end_dt + pd.DateOffset(days=args.forecast * 2 + 10)

    print(f"\n下載 {args.symbol}  {start_dt.date()} ~ {dl_end.date()} ...")
    df_raw = yf.download(
        args.symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=dl_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False, progress=False,
    )
    df = ensure_ohlcv(df_raw)
    print(f"下載完成：{len(df)} 根 K 棒  (訓練截止={end_dt.date()}，預抓至={dl_end.date()})")

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

    train_df  = df.iloc[train_end_idx - args.lookback: train_end_idx]
    future_df = df.iloc[train_end_idx: train_end_idx + args.forecast].copy()

    close_hist   = train_df["Close"].values.astype(float)
    actual_close = future_df["Close"].values.astype(float) if len(future_df) > 0 else None

    actual_ohlc: dict | None = None
    if len(future_df) > 0:
        actual_ohlc = {
            "open":  future_df["Open"].values.astype(float),
            "high":  future_df["High"].values.astype(float),
            "low":   future_df["Low"].values.astype(float),
            "close": future_df["Close"].values.astype(float),
        }

    start_price = float(close_hist[-1])

    # ── auto-calibrate ──
    calib_info             = None
    model_predicted_params = None
    use_garch              = args.auto_calibrate and not args.no_garch

    if args.auto_calibrate:
        calib_window = args.calib_window
        model_label  = args.garch_model if use_garch else "rv"
        print(f"\n[auto-calibrate v11.2] 掃描前 {calib_window} 根 K 棒  (vol_model={model_label})...")
        calib = auto_calibrate(
            df.iloc[:train_end_idx]["Close"].values.astype(float),
            theta,
            calib_window=calib_window,
            use_garch=use_garch,
            garch_model=args.garch_model,
        )
        calib_info = calib
        if args.intra_bar      is None: args.intra_bar      = calib["intra_bar"]
        if args.shadow_noise   is None: args.shadow_noise   = calib["shadow_noise"]
        if args.shadow_clamp   is None: args.shadow_clamp   = calib["shadow_clamp"]
        if args.momentum_boost is None: args.momentum_boost = calib["momentum_boost"]
        if args.drift_decay    is None: args.drift_decay    = calib["drift_decay"]
        if args.vol_multiplier is None: args.vol_multiplier = calib["vol_multiplier"]
        if args.drift_scale    is None: args.drift_scale    = calib["drift_scale"]
        # drift clamp：手動 > 動態計算
        if args.drift_clamp is None:
            args.drift_clamp = calib["drift_clamp_max"]
    else:
        if args.intra_bar      is None: args.intra_bar      = 2
        if args.shadow_noise   is None: args.shadow_noise   = 0.15
        if args.shadow_clamp   is None: args.shadow_clamp   = 2.0
        if args.momentum_boost is None: args.momentum_boost = 1.6
        if args.drift_decay    is None: args.drift_decay    = 0.04
        if args.vol_multiplier is None: args.vol_multiplier = 1.2
        if args.drift_scale    is None: args.drift_scale    = 1.18

    # ── param-model 覆蓋 ──
    if _pm_pipelines is not None:
        feat_df   = df.iloc[train_end_idx - args.calib_window: train_end_idx]
        feats     = extract_features_for_model(feat_df, theta.vol)
        feat_cols = _pm_feat_cols if _pm_feat_cols else list(feats.keys())
        feat_vec  = np.array([[feats.get(k, 0.0) for k in feat_cols]])

        model_predicted_params = {}
        for tname, pipe in _pm_pipelines.items():
            pred  = float(pipe.predict(feat_vec)[0])
            short = tname.replace("target_", "")
            model_predicted_params[short] = round(pred, 4)

        if "drift_scale" in model_predicted_params:
            args.drift_scale = model_predicted_params["drift_scale"]
        if "drift_decay" in model_predicted_params:
            args.drift_decay = model_predicted_params["drift_decay"]
        print(f"[param-model] 預測參數: {model_predicted_params}")

    # ── backbone ──
    fitter     = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb_result  = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])

    log_ret_hist = np.diff(np.log(close_hist))
    short_drift  = float(np.mean(log_ret_hist[-5:])) if len(log_ret_hist) >= 5 else last_drift
    w            = float(np.clip(args.short_drift_weight, 0.0, 1.0))
    blend_drift  = w * short_drift + (1.0 - w) * last_drift

    # ── v11.2：動態截斷 blend_drift ──
    drift_clamp_max = args.drift_clamp  # 已在 auto_calibrate 或手動設定
    if drift_clamp_max is not None and drift_clamp_max > 0:
        blend_drift_raw = blend_drift
        blend_drift = float(np.clip(blend_drift, -drift_clamp_max, drift_clamp_max))
        if abs(blend_drift_raw - blend_drift) > 1e-8:
            print(f"  [drift clamp] {blend_drift_raw*100:+.4f}%/day → {blend_drift*100:+.4f}%/day  "
                  f"(MAX=±{drift_clamp_max*100:.4f}%/day)")

    drift_fwd = np.full(args.forecast, blend_drift)
    vol_fwd   = np.full(args.forecast, last_vol) * (args.vol_multiplier or 1.0)
    bb_fwd    = start_price * np.cumprod(1 + drift_fwd)

    # ── estimator ──
    ESTIMATOR_LB = 500
    rv_window    = min(21, len(close_hist))
    rv           = float(np.std(np.diff(np.log(close_hist[-rv_window:]))))
    vol_scale    = float(np.clip(rv / max(theta.vol, 1e-8), 0.6, 4.0))

    estimate_df = df.iloc[max(0, train_end_idx - ESTIMATOR_LB): train_end_idx]
    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = build_params_from_theta(theta, base_params)

    import dataclasses
    params_fwd = dataclasses.replace(
        params_fwd,
        last_close=start_price,
    )
    momentum_bias   = float(getattr(params_fwd, "momentum_bias",       0.0))
    breakout_state  = int(getattr(params_fwd,   "node_breakout_state", 0))

    if args.verbose:
        print_diagnostics(
            args, calib_info, model_predicted_params,
            momentum_bias=momentum_bias,
            breakout_state=breakout_state,
            blend_drift=blend_drift,
            short_drift=short_drift,
            last_drift=last_drift,
            drift_clamp_max=drift_clamp_max,
        )

    # ── simulate ──
    end_date_str = end_dt.strftime("%Y-%m-%d")
    print(f"\n[simulate] {args.symbol}  起點={end_date_str}  forecast={args.forecast}  n_paths={args.n_paths}")
    print(f"  blend_drift={blend_drift*100:+.4f}%/day  "
          f"(short={short_drift*100:+.4f}%  long={last_drift*100:+.4f}%  w={w})")
    print(f"  momentum_bias={momentum_bias:+.4f}  breakout_state={breakout_state}")
    print(f"  drift_scale={args.drift_scale}  drift_decay={args.drift_decay}  "
          f"vol_multiplier={args.vol_multiplier}  momentum_boost={args.momentum_boost}")

    sim = USStockFutureSimulator(
        params=params_fwd,
        forecast_steps=args.forecast,
        n_paths=args.n_paths,
        seed=42,
        vol_scale=vol_scale,
        mr_coeff=theta.mr_coeff,
        node_coeff=theta.node_coeff,
        momentum_strength=theta.momentum_strength * (args.momentum_boost or 1.0),
        momentum_decay=theta.momentum_decay,
        breakout_boost=theta.breakout_boost,
        drift_schedule=drift_fwd,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=0.06,
        intra_bar_steps=args.intra_bar or 2,
        drift_decay_rate=args.drift_decay or 0.04,
        drift_scale=args.drift_scale or 1.0,
        momentum_anchor_weight=0.45,
    )
    result = sim.simulate()

    rep_close, ohlc = pick_representative_path(result)

    if len(rep_close) == 0:
        rep_close = np.full(args.forecast, start_price)
        ohlc      = None

    rep_close = rep_close[:args.forecast]
    if ohlc is not None:
        ohlc = {k: v[:args.forecast] for k, v in ohlc.items()}

    n          = len(rep_close)
    final_pred = float(rep_close[n - 1])
    total_ret  = (final_pred - start_price) / start_price * 100
    print(f"\n  起點價格 : {start_price:.2f}")
    print(f"  預測終點 : {final_pred:.2f}  ({total_ret:+.1f}%  {n}日)")

    if actual_close is not None and len(actual_close) > 0:
        na  = min(len(actual_close), n)
        mae = float(np.mean(np.abs(actual_close[:na] - rep_close[:na]) / start_price * 100))
        actual_ret = (float(actual_close[na - 1]) - start_price) / start_price * 100
        print(f"  實際終點 : {actual_close[na-1]:.2f}  ({actual_ret:+.1f}%)")
        print(f"  MAE      : {mae:.2f}%")

    print_ohlc_comparison(
        pred_ohlc=ohlc,
        pred_close=rep_close,
        actual_df=future_df.reset_index(drop=True) if len(future_df) > 0 else None,
        start_price=start_price,
        symbol=args.symbol,
    )

    model_tag = "+model" if _pm_pipelines is not None else ""
    mode_tag  = (f"v11-{args.garch_model}(w={args.calib_window}){model_tag}"
                 if args.auto_calibrate else "manual")
    out_path  = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "symbol":         args.symbol,
        "end_date":       end_date_str,
        "start_price":    start_price,
        "forecast_steps": args.forecast,
        "n_paths":        args.n_paths,
        "mode":           mode_tag,
        "params": {
            "intra_bar":          args.intra_bar,
            "momentum_boost":     args.momentum_boost,
            "drift_decay":        args.drift_decay,
            "vol_multiplier":     args.vol_multiplier,
            "drift_scale":        args.drift_scale,
            "blend_drift":        round(blend_drift, 6),
            "short_drift_weight": w,
            "drift_clamp_max":    round(drift_clamp_max, 6) if drift_clamp_max else None,
        },
        "momentum_bias":    momentum_bias,
        "breakout_state":   breakout_state,
        "auto_calibrate":   args.auto_calibrate,
        "model_predicted":  model_predicted_params,
        "rep_path":         [round(float(x), 4) for x in rep_close],
    }

    json_path = out_path / f"{args.symbol}_{end_date_str}_{mode_tag}.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"✔ JSON 已儲存 → {json_path}")

    render_forecast(
        symbol=args.symbol,
        hist_close=close_hist[-60:],
        result=result,
        forecast=args.forecast,
        end_date_str=end_date_str,
        output_dir=args.output_dir,
        mode_tag=mode_tag,
        rep_close=rep_close,
        ohlc=ohlc,
        actual_close=actual_close,
        actual_ohlc=actual_ohlc,
    )
    print(f"✔ 完成")


if __name__ == "__main__":
    main()
