"""
forward_study.py  v9  (GJR-GARCH + auto-calibrate)

v9 修正：
  1. 傳送 median_path 到 render_forecast（修正 KeyError 'median_path'）
  2. render_forecast 接收 median_path 參數
  3. --param-model 旗標：載入 XGBoost 模型預測 drift_scale / drift_decay，
     自動預測 drift_scale / drift_decay，取代 auto-calibrate 的對應值。
  4. 移除 auto-calibrate 區塊的重複 inline print（由 print_diagnostics 統一輸出）

用法：
  python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --auto-calibrate \
    --param-model models/param_model_AAPL.joblib \
    --end-date 2024-01-03
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
    """
    用 GJR-GARCH(1,1,1) 預測下一根 bar 的日波動率（%/day）。
    回傳 (vol_pct, info_dict)；失敗時回傳 (None, {"error": ...})。
    """
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
        fvol = float(np.sqrt(fc.variance.values[-1, 0])) / 100   # daily vol (fraction)

        info = {
            "alpha": round(alpha, 4),
            "gamma": round(gamma, 4),
            "beta":  round(beta, 4),
            "persistence": round(persistence, 4),
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",        required=True)
    p.add_argument("--theta",         required=True)
    p.add_argument("--end-date",      default=None)
    p.add_argument("--lookback",      type=int, default=120)
    p.add_argument("--forecast",      type=int, default=30)
    p.add_argument("--n-paths",       type=int, default=500)
    p.add_argument("--intra-bar",     type=int, default=None)
    p.add_argument("--shadow-noise",  type=float, default=None)
    p.add_argument("--shadow-clamp",  type=float, default=None)
    p.add_argument("--momentum-boost",type=float, default=None)
    p.add_argument("--drift-decay",   type=float, default=None)
    p.add_argument("--vol-multiplier",type=float, default=None)
    p.add_argument("--drift-scale",   type=float, default=None)
    p.add_argument("--body-scale-max",type=float, default=None)
    p.add_argument("--auto-calibrate",    action="store_true")
    p.add_argument("--calib-window",  type=int, default=500)
    p.add_argument("--no-garch",      action="store_true")
    p.add_argument("--garch-model",   default="gjr-garch",
                   choices=["gjr-garch", "garch"])
    p.add_argument("--param-model",   default=None,
                   help="joblib 模型路徑；若提供，用模型預測 drift_scale / "
                        "覆蓋 auto-calibrate 的對應值。")
    p.add_argument("--output-dir",    default="results")
    p.add_argument("--verbose",       action="store_true")
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
    """
    根據近期 K 棒自動設定模擬參數。
    回傳 dict，含 intra_bar / shadow_noise / shadow_clamp /
    momentum_boost / drift_decay / vol_multiplier / drift_scale
    以及隱藏欄位 _vol_source / _avg_body_pct / _garch_info。
    """
    c = close_arr[-calib_window:] if len(close_arr) > calib_window else close_arr

    log_rets = np.diff(np.log(c))
    rv = float(np.std(log_rets))

    # 波動率來源
    garch_vol, garch_info = None, {}
    vol_source = "rv"
    if use_garch and _ARCH_AVAILABLE:
        garch_vol, garch_info = garch_vol_forecast(c, model_type=garch_model)
        if garch_vol is not None and "error" not in garch_info:
            vol_source = "garch"

    base_vol = garch_vol if vol_source == "garch" else rv
    vol_multiplier = float(np.clip(base_vol / max(theta.vol, 1e-8), 0.5, 3.0))

    # 平均 body 大小
    o_arr = None   # 沒有 open 就只用 close
    body_pct = np.abs(np.diff(c)) / (c[:-1] + 1e-8)
    avg_body = float(np.mean(body_pct))
    intra_bar = 3 if avg_body > 0.012 else 2

    # shadow noise
    shadow_noise = float(np.clip(rv * 4.0, 0.08, 0.30))
    shadow_clamp = 2.5 if rv > 0.018 else 2.0

    # momentum & drift
    momentum_boost = 0.8
    drift_decay = float(np.clip(0.02 + rv * 1.5, 0.03, 0.15))
    drift_scale = float(np.clip(rv / max(theta.vol, 1e-8), 0.4, 3.0))

    return {
        "intra_bar":      intra_bar,
        "shadow_noise":   round(shadow_noise, 3),
        "shadow_clamp":   round(shadow_clamp, 2),
        "momentum_boost": momentum_boost,
        "drift_decay":    round(drift_decay, 4),
        "vol_multiplier": round(vol_multiplier, 4),
        "drift_scale":    round(drift_scale, 4),
        # 隱藏欄位
        "_vol_source":    vol_source,
        "_avg_body_pct":  round(avg_body * 100, 3),
        "_garch_info":    garch_info,
    }


# ──────────────────────────────────────────────────────────────────────────────
# print_diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def print_diagnostics(args, calib_info: dict | None, model_predicted_params: dict | None):
    print("\n" + "─" * 60)
    print("  VERBOSE DIAGNOSTICS (v9.2 GJR-GARCH + auto-calibrate + param-model)")
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

    print(f"\n  [最終模擬參數]")
    print(f"  intra_bar     : {args.intra_bar}")
    print(f"  shadow_noise  : {args.shadow_noise}")
    print(f"  shadow_clamp  : {args.shadow_clamp}")
    print(f"  momentum_boost: {args.momentum_boost}")
    print(f"  drift_decay   : {args.drift_decay}")
    print(f"  vol_multiplier: {args.vol_multiplier}")
    print(f"  drift_scale   : {args.drift_scale}")

    if model_predicted_params:
        print(f"\n  [param-model 覆蓋]")
        print(f"  drift_scale   : {model_predicted_params.get('drift_scale', 'N/A')}  ← 模型預測（已覆蓋 auto-calibrate）")
        print(f"  drift_decay   : {model_predicted_params.get('drift_decay', 'N/A')}  ← 模型預測（已覆蓋 auto-calibrate）")

    print("─" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# render_forecast
# ──────────────────────────────────────────────────────────────────────────────
def render_forecast(
    symbol, hist_close, result, forecast, end_date_str,
    output_dir, mode_tag, median_path,
    actual_close=None,
):
    fig, ax = plt.subplots(figsize=(14, 6))

    # 歷史
    hist_x = np.arange(-len(hist_close), 0)
    ax.plot(hist_x, hist_close, color="#555555", lw=1.2, label="歷史收盤")

    # 信賴帶
    fwd_x = np.arange(0, forecast)
    if hasattr(result, "percentile_bands") and result.percentile_bands:
        bands = result.percentile_bands
        p10 = bands.get("p10", bands.get(10))
        p25 = bands.get("p25", bands.get(25))
        p75 = bands.get("p75", bands.get(75))
        p90 = bands.get("p90", bands.get(90))
        if p10 is not None and p90 is not None:
            ax.fill_between(fwd_x, p10[:forecast], p90[:forecast],
                            alpha=0.15, color="#2196F3", label="P10-P90")
        if p25 is not None and p75 is not None:
            ax.fill_between(fwd_x, p25[:forecast], p75[:forecast],
                            alpha=0.25, color="#2196F3", label="P25-P75")

    # 中位數路徑
    mp = median_path[:forecast]
    ax.plot(fwd_x[:len(mp)], mp, color="#E53935", lw=2, label="中位數預測")

    # 實際（若有）
    if actual_close is not None and len(actual_close) > 0:
        n = min(len(actual_close), forecast)
        ax.plot(fwd_x[:n], actual_close[:n],
                color="#43A047", lw=1.5, ls="--", label="實際收盤")

    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set_title(f"{symbol}  {end_date_str}  [{mode_tag}]")
    ax.set_xlabel("交易日（相對預測起點）")
    ax.set_ylabel("收盤價")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{symbol}_{end_date_str}_{mode_tag}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✔ 圖表已儲存 → {fname}")
    return str(fname)


# ──────────────────────────────────────────────────────────────────────────────
# 特徵提取（與 collect_training_data 保持一致）
# ──────────────────────────────────────────────────────────────────────────────
def extract_features_for_model(df_window, theta_vol):
    c = df_window["Close"].values.astype(float)
    o_arr = df_window["Open"].values.astype(float)
    h = df_window["High"].values.astype(float)
    l = df_window["Low"].values.astype(float)
    v = df_window["Volume"].values.astype(float)

    log_rets = np.diff(np.log(c))
    vol_20   = float(np.std(log_rets[-20:])) if len(log_rets) >= 20 else float(np.std(log_rets))
    vol_60   = float(np.std(log_rets[-60:])) if len(log_rets) >= 60 else float(np.std(log_rets))
    vol_all  = float(np.std(log_rets))
    drift_20 = float(np.mean(log_rets[-20:])) if len(log_rets) >= 20 else float(np.mean(log_rets))
    drift_60 = float(np.mean(log_rets[-60:])) if len(log_rets) >= 60 else float(np.mean(log_rets))
    drift_all= float(np.mean(log_rets))

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
        "feat_vol_20":          round(vol_20 * 100, 4),
        "feat_vol_60":          round(vol_60 * 100, 4),
        "feat_vol_all":         round(vol_all * 100, 4),
        "feat_vol_ratio_20_60": round(vol_20 / (vol_60 + 1e-8), 4),
        "feat_vol_ratio_rv_theta": round(vol_20 / (theta_vol + 1e-8), 4),
        "feat_drift_20":        round(drift_20 * 100, 4),
        "feat_drift_60":        round(drift_60 * 100, 4),
        "feat_drift_all":       round(drift_all * 100, 4),
        "feat_drift_ratio_20_all": round(drift_20 / (abs(drift_all) + 1e-8) * np.sign(drift_all + 1e-12), 4),
        "feat_ret_autocorr":    round(ret_autocorr, 4),
        "feat_vol_autocorr":    round(vol_autocorr, 4),
        "feat_avg_body_pct":    round(avg_body * 100, 4),
        "feat_median_sr":       round(median_sr, 4),
        "feat_p90_sr":          round(p90_sr, 4),
        "feat_bb_last_drift":   round(bb_last_drift * 100, 4),
        "feat_bb_last_vol":     round(bb_last_vol * 100, 4),
        "feat_bb_drift_std":    round(bb_drift_std * 100, 4),
        "feat_bb_vol_std":      round(bb_vol_std * 100, 4),
    }

    # GARCH features
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

    # 載入 param-model（若有）
    param_model = None
    param_meta  = None
    if args.param_model:
        import joblib
        param_model = joblib.load(args.param_model)
        meta_path = Path(args.param_model).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                param_meta = json.load(f)
        print(f"[param-model] 已載入 {args.param_model}")

    # 下載資料
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
    print(f"下載完成：{len(df)} 根 K 棒")

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
    future_df = df.iloc[train_end_idx: train_end_idx + args.forecast]

    close_hist   = train_df["Close"].values.astype(float)
    actual_close = future_df["Close"].values.astype(float) if len(future_df) > 0 else None
    start_price  = float(close_hist[-1])

    # ── auto-calibrate ──
    calib_info = None
    model_predicted_params = None
    use_garch = args.auto_calibrate and not args.no_garch

    if args.auto_calibrate:
        calib_window = args.calib_window
        model_label  = args.garch_model if use_garch else "rv"
        print(f"\n[auto-calibrate v9] 掃描前 {calib_window} 根 K 棒  (vol_model={model_label})...")
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
        # 詳細輸出由 print_diagnostics 統一負責，此處不重複列印
    else:
        if args.intra_bar      is None: args.intra_bar      = 2
        if args.shadow_noise   is None: args.shadow_noise   = 0.15
        if args.shadow_clamp   is None: args.shadow_clamp   = 2.0
        if args.momentum_boost is None: args.momentum_boost = 1.6
        if args.drift_decay    is None: args.drift_decay    = 0.04
        if args.vol_multiplier is None: args.vol_multiplier = 1.2
        if args.drift_scale    is None: args.drift_scale    = 1.18

    # ── param-model 覆蓋 drift_scale / drift_decay ──
    if param_model is not None:
        calib_window_fm = args.calib_window
        feat_df = df.iloc[train_end_idx - calib_window_fm: train_end_idx]
        feats = extract_features_for_model(feat_df, theta.vol)

        # 確保特徵順序與訓練時一致
        if param_meta and "feature_names" in param_meta:
            feat_vec = np.array([[feats.get(k, 0.0) for k in param_meta["feature_names"]]])
        else:
            feat_vec = np.array([list(feats.values())])

        preds = param_model.predict(feat_vec)
        # preds shape: (1, n_targets) or (n_targets,)
        preds_flat = np.array(preds).flatten()

        target_names = param_meta.get("target_names", []) if param_meta else []
        model_predicted_params = {}
        for i, tname in enumerate(target_names):
            if i < len(preds_flat):
                model_predicted_params[tname.replace("target_", "")] = round(float(preds_flat[i]), 4)

        if "drift_scale" in model_predicted_params:
            args.drift_scale = model_predicted_params["drift_scale"]
        if "drift_decay" in model_predicted_params:
            args.drift_decay = model_predicted_params["drift_decay"]
        print(f"[param-model] 預測參數: {model_predicted_params}")

    # ── verbose diagnostics ──
    if args.verbose:
        print_diagnostics(args, calib_info, model_predicted_params)

    # ── backbone ──
    fitter    = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb_result = fitter.fit(close_hist)
    last_drift = float(bb_result.segment_drifts[-1])
    last_vol   = float(bb_result.segment_vols[-1])
    drift_fwd  = np.full(args.forecast, last_drift)
    vol_fwd    = np.full(args.forecast, last_vol) * (args.vol_multiplier or 1.0)
    bb_fwd     = start_price * np.cumprod(1 + drift_fwd)

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
        momentum_bias=0.0,
        node_breakout_state=0,
    )

    # ── simulate ──
    end_date_str = end_dt.strftime("%Y-%m-%d")
    print(f"\n[simulate] {args.symbol}  起點={end_date_str}  forecast={args.forecast}  n_paths={args.n_paths}")
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
        shadow_noise_scale=args.shadow_noise or 0.15,
        shadow_clamp_sigma=args.shadow_clamp or 2.0,
        momentum_anchor_weight=0.45,
        body_scale_max=args.body_scale_max or 2.5,
    )
    result = sim.simulate()

    # median_path
    if hasattr(result, "median_path") and result.median_path is not None:
        median_path = result.median_path
    elif hasattr(result, "percentile_bands") and result.percentile_bands:
        bands = result.percentile_bands
        p50 = bands.get("p50", bands.get(50))
        if p50 is not None:
            median_path = p50
        else:
            keys = sorted(bands.keys())
            mid = keys[len(keys) // 2]
            median_path = bands[mid]
    else:
        # fallback: mean of all paths
        if hasattr(result, "paths") and result.paths is not None:
            median_path = np.median(result.paths, axis=0)
        else:
            median_path = np.full(args.forecast, start_price)

    # ── print summary ──
    n = min(len(median_path), args.forecast)
    final_pred  = float(median_path[n - 1])
    total_ret   = (final_pred - start_price) / start_price * 100
    print(f"\n  起點價格 : {start_price:.2f}")
    print(f"  預測終點 : {final_pred:.2f}  ({total_ret:+.1f}%  {n}日)")

    if actual_close is not None and len(actual_close) > 0:
        na = min(len(actual_close), n)
        mae = float(np.mean(np.abs(actual_close[:na] - median_path[:na]) / start_price * 100))
        actual_ret = (float(actual_close[na - 1]) - start_price) / start_price * 100
        print(f"  實際終點 : {actual_close[na-1]:.2f}  ({actual_ret:+.1f}%)")
        print(f"  MAE      : {mae:.2f}%")

    # ── 儲存結果 ──
    model_tag  = f"+model" if param_model is not None else ""
    mode_tag   = f"v9.2-{args.garch_model}(w={args.calib_window}){model_tag}" if args.auto_calibrate else "manual"
    out_path   = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "symbol":         args.symbol,
        "end_date":       end_date_str,
        "start_price":    start_price,
        "forecast_steps": args.forecast,
        "n_paths":        args.n_paths,
        "mode":           mode_tag,
        "params": {
            "intra_bar":      args.intra_bar,
            "shadow_noise":   args.shadow_noise,
            "shadow_clamp":   args.shadow_clamp,
            "momentum_boost": args.momentum_boost,
            "drift_decay":    args.drift_decay,
            "vol_multiplier": args.vol_multiplier,
            "drift_scale":    args.drift_scale,
        },
        "auto_calibrate":   args.auto_calibrate,
        "model_predicted":  model_predicted_params,
        "median_path":      [round(float(x), 4) for x in median_path[:args.forecast]],
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
        median_path=median_path,
        actual_close=actual_close,
    )
    print(f"\n✔ 完成  圖表路徑 → {args.output_dir}/{args.symbol}_{end_date_str}_{mode_tag}.png")


if __name__ == "__main__":
    main()
