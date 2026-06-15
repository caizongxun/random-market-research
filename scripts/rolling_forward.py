"""
rolling_forward.py  (v3 - Medoid + P10/P90 + Coverage + Agent Layer + VIX hookup)

核心流程
--------
每「輪」只預測 N 根（預設 5），然後把真實 N 根接在歷史尾端，
再重新校準 + 預測下一輪。

新增功能（v3）
--------------
1. Medoid 路徑選取：模擬 n_paths 條路徑，選出和其他所有路徑
   Euclidean 距離總和最小的那條（保留波動、轉折，不平滑化）。
2. P10/P90 信心區間：每根 K 棒的第 10、90 百分位數。
3. 覆蓋率評估：實際走勢落在 P10–P90 的比例，目標 ≈ 80%。
4. 代理人行為層：載入 agent_profile.json，調整 momentum_boost
   和 mr_coeff，讓模擬反映散戶/大戶行為特徵。
5. VIX 接入：每輪從 cache/macro.parquet 讀取對應時間點 VIX，
   傳入 apply_agent_profile 做恐慌修正。
6. Hurst 防呆：inst_mr_strength clamp 到 [0.1, 1.0]，避免
   pure-trend 股票把 mr_coeff 壓到下界。

修復（v3）
----------
- Bug fix: SimulationResult 的多路徑屬性是 future_paths，不是
  paths/close_paths/all_paths。之前 fallback 導致 P10=P90=medoid，
  覆蓋率永遠 0%。現在直接用 result.future_paths[:, :step]。

方向命中定義（v2）
------------------
  P-Ret_t = pred_close_t  - actual_close_{t-1}
  A-Ret_t = actual_close_t - actual_close_{t-1}
  命中 = sign(P-Ret) == sign(A-Ret)

用法
----
  python scripts/rolling_forward.py \\
    --symbol AAPL \\
    --theta cache/AAPL_theta.json \\
    --agent-profile cache/AAPL_agent_profile.json \\
    --end-date 2025-01-01 \\
    --total-bars 20 \\
    --step 5
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
# VIX 查詢輔助（v3 新增）
# ──────────────────────────────────────────────────────────────────────────────
def _load_vix_series(cache_dir: str | None = None) -> pd.Series | None:
    """
    從 cache/macro.parquet 讀取 VIX 序列。
    回傳 pd.Series(index=DatetimeIndex, values=float) 或 None。
    """
    if cache_dir is None:
        # 預設相對路徑：scripts/ 的上層 / cache
        cache_dir = str(Path(__file__).parent.parent / "cache")
    macro_path = Path(cache_dir) / "macro.parquet"
    if not macro_path.exists():
        return None
    try:
        df = pd.read_parquet(macro_path)
        # 欄位可能是 'VIX' 或 '^VIX'
        for col in ["VIX", "^VIX", "vix"]:
            if col in df.columns:
                s = df[col].dropna()
                s.index = pd.to_datetime(s.index)
                return s
        return None
    except Exception:
        return None


def _get_vix_at(vix_series: pd.Series | None, date: pd.Timestamp) -> float | None:
    """
    取 date 當天或之前最近一個 VIX 值。
    找不到時回傳 None。
    """
    if vix_series is None or len(vix_series) == 0:
        return None
    candidates = vix_series[vix_series.index <= date]
    if len(candidates) == 0:
        return None
    return float(candidates.iloc[-1])


# ──────────────────────────────────────────────────────────────────────────────
# 代理人行為層
# ──────────────────────────────────────────────────────────────────────────────
def load_agent_profile(path: str | None) -> dict | None:
    """載入 agent_profile.json，回傳 dict 或 None。"""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[agent-profile] 找不到 {path}，跳過代理人行為調整")
        return None
    with open(p) as f:
        profile = json.load(f)
    print(f"[agent-profile] 載入 {path}")
    print(f"  散戶動能={profile.get('retail_momentum_strength', 'N/A'):.3f}  "
          f"恐慌靈敏={profile.get('retail_panic_sensitivity', 'N/A'):.3f}  "
          f"大戶MR={profile.get('inst_mr_strength', 'N/A'):.3f}")
    return profile


def apply_agent_profile(
    profile: dict | None,
    base_momentum_boost: float,
    base_mr_coeff: float,
    vix_now: float | None = None,
) -> tuple[float, float]:
    """
    根據 agent_profile 調整 momentum_boost 和 mr_coeff。

    散戶動能強 → momentum_boost 上調
    大戶均值回歸強 → mr_coeff 上調
    恐慌靈敏度高 + VIX 高 → momentum_boost 下調（拋售壓制動能）

    v3 fix: inst_mr_strength clamp 到 [0.1, 1.0]，避免 H≈1 的股票
    把 mr_coeff 壓到下界導致均值回歸力消失。
    """
    if profile is None:
        return base_momentum_boost, base_mr_coeff

    retail_mom  = float(profile.get("retail_momentum_strength",  0.5))
    panic_sens  = float(profile.get("retail_panic_sensitivity",  0.3))
    # v3: clamp inst_mr_strength，pure-trend (H≈1) 時仍保留最低 0.1
    inst_mr_raw = float(profile.get("inst_mr_strength", 0.5))
    inst_mr     = float(np.clip(inst_mr_raw, 0.1, 1.0))
    vix_cal     = float(profile.get("vix_level") or 20.0)
    vix_ref     = vix_now if vix_now is not None else vix_cal

    # 散戶動能：0.5 為中性，>0.5 上調 momentum_boost
    mom_adj = (retail_mom - 0.5) * 0.6   # 最多 ±0.3

    # 恐慌修正：VIX 高時壓制動能
    panic_adj = -panic_sens * float(np.clip((vix_ref - 20) / 30, 0.0, 1.0)) * 0.3

    new_momentum_boost = float(np.clip(base_momentum_boost + mom_adj + panic_adj, 0.3, 2.5))

    # 大戶 MR：inst_mr > 0.5 增強回歸力（clamp 後不會因 0.0 變成負調整）
    mr_adj = (inst_mr - 0.5) * 0.04   # 最多 ±0.02（clamp 後最少 -0.016）
    new_mr_coeff = float(np.clip(base_mr_coeff + mr_adj, 0.01, 0.20))

    return new_momentum_boost, new_mr_coeff


# ──────────────────────────────────────────────────────────────────────────────
# Medoid 路徑選取
# ──────────────────────────────────────────────────────────────────────────────
def select_medoid_path(all_paths: np.ndarray) -> tuple[int, np.ndarray]:
    """
    從 all_paths (shape: n_paths x n_steps) 選出 medoid。
    Medoid = 和其他所有路徑 Euclidean 距離總和最小的那條。

    Returns:
        medoid_idx : int
        medoid_path: np.ndarray (n_steps,)
    """
    n = all_paths.shape[0]
    if n == 1:
        return 0, all_paths[0]

    # 樣本數大時用隨機子集加速（最多 200 條計算距離矩陣）
    if n > 200:
        idx_sample = np.random.choice(n, 200, replace=False)
        subset = all_paths[idx_sample]
        diff = subset[:, np.newaxis, :] - subset[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        dist_sums = dist_matrix.sum(axis=1)
        medoid_local = int(np.argmin(dist_sums))
        medoid_idx = int(idx_sample[medoid_local])
    else:
        diff = all_paths[:, np.newaxis, :] - all_paths[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        dist_sums = dist_matrix.sum(axis=1)
        medoid_idx = int(np.argmin(dist_sums))

    return medoid_idx, all_paths[medoid_idx]


def compute_pi_bands(
    all_paths: np.ndarray,
    p_low: float = 10.0,
    p_high: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算每個時間步的 P10 / P90 百分位數。

    Args:
        all_paths: shape (n_paths, n_steps)
    Returns:
        p_low_arr, p_high_arr: 各 shape (n_steps,)
    """
    lo = np.percentile(all_paths, p_low,  axis=0)
    hi = np.percentile(all_paths, p_high, axis=0)
    return lo, hi


def compute_coverage(
    actual: np.ndarray,
    p_lo:   np.ndarray,
    p_hi:   np.ndarray,
) -> float:
    """
    計算覆蓋率：實際收盤落在 [p_lo, p_hi] 的比例。
    """
    n = min(len(actual), len(p_lo), len(p_hi))
    if n == 0:
        return float("nan")
    inside = np.sum((actual[:n] >= p_lo[:n]) & (actual[:n] <= p_hi[:n]))
    return float(inside / n)


# ──────────────────────────────────────────────────────────────────────────────
# 參數模型
# ──────────────────────────────────────────────────────────────────────────────
def load_param_model(path: str | None):
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
    if payload is None:
        return calib

    feats = _extract_features(df_window, theta_vol)
    feature_cols = payload["feature_cols"]
    models       = payload["models"]

    x = np.array([[feats.get(fc.replace("feat_", ""), 0.0) for fc in feature_cols]])

    pred_ds = float(models["target_drift_scale"].predict(x)[0])
    pred_dd = float(models["target_drift_decay"].predict(x)[0])

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
# 單輪預測（回傳所有路徑 + medoid + bands）
# ──────────────────────────────────────────────────────────────────────────────
def _run_one_step(
    close_hist: np.ndarray,
    theta: CalibratedTheta,
    args,
    full_df_up_to_now: pd.DataFrame,
    param_model_payload=None,
    agent_profile: dict | None = None,
    vix_series: pd.Series | None = None,
    cutoff_date: pd.Timestamp | None = None,
) -> tuple[np.ndarray, dict | None, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        rep_close   : medoid 路徑收盤 (step,)
        ohlc        : medoid OHLC dict or None
        all_closes  : 所有路徑收盤 (n_paths, step)
        p10         : P10 per bar   (step,)
        p90         : P90 per bar   (step,)

    v3 changes:
        - vix_series / cutoff_date 參數用於查詢當輪 VIX
        - result.future_paths 直接用（修復 P10/P90 = 0% 的 bug）
    """
    step = args.step

    calib = auto_calibrate(
        full_df_up_to_now["Close"].values.astype(float),
        theta,
        calib_window=args.calib_window,
        use_garch=(not args.no_garch) and _ARCH_AVAILABLE,
        garch_model=args.garch_model,
    )

    if param_model_payload is not None:
        calib_window_df = full_df_up_to_now.iloc[-args.calib_window:]
        calib = apply_param_model(
            param_model_payload,
            calib_window_df,
            theta.vol,
            calib,
            verbose=getattr(args, "verbose", False),
        )

    intra_bar       = calib["intra_bar"]
    momentum_boost  = calib["momentum_boost"]
    drift_decay     = calib["drift_decay"]
    vol_multiplier  = calib["vol_multiplier"]
    drift_scale     = calib["drift_scale"]
    drift_clamp_max = calib["drift_clamp_max"]

    # ── 代理人行為調整（v3：傳入實際 VIX）──────────────────
    base_mr = theta.mr_coeff
    # v3: 查詢對應時間點的 VIX
    vix_now = _get_vix_at(vix_series, cutoff_date) if cutoff_date is not None else None
    momentum_boost, base_mr = apply_agent_profile(
        agent_profile,
        momentum_boost,
        base_mr,
        vix_now=vix_now,
    )
    # ────────────────────────────────────────────────────────

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
        mr_coeff=base_mr,
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

    # ── v3 fix: 直接用 result.future_paths（shape: n_paths x n_steps）──
    # 之前程式碼尋找 'paths'/'close_paths'/'all_paths' 屬性，
    # 但 SimulationResult 只有 future_paths，導致 all_closes 永遠是 None，
    # P10/P90 fallback 成 medoid，覆蓋率永遠 0%。
    all_closes = result.future_paths[:, :step]   # (n_paths, step)

    # ── Medoid 選取 ─────────────────────────────────────────
    medoid_idx, medoid_close = select_medoid_path(all_closes)
    p10, p90 = compute_pi_bands(all_closes)

    medoid_close = medoid_close[:step]
    p10 = p10[:step]
    p90 = p90[:step]

    # 從模擬結果中取 representative path 的 OHLC，用 medoid close 覆蓋
    ohlc = None
    rep_close, rep_ohlc = pick_representative_path(result)
    if rep_ohlc is not None:
        ohlc = {k: v[:step] for k, v in rep_ohlc.items()}
        ohlc["close"] = medoid_close

    return medoid_close, ohlc, all_closes, p10, p90


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
    agent_profile: dict | None = None,
    vix_series: pd.Series | None = None,
) -> dict:
    step       = args.step
    total_bars = args.total_bars
    lookback   = args.lookback

    pred_closes  : list[float] = []
    pred_ohlcs   : list[dict]  = []
    pred_p10     : list[float] = []
    pred_p90     : list[float] = []
    round_details: list[dict]  = []

    cur_end   = train_end_idx
    rounds    = int(np.ceil(total_bars / step))
    bars_done = 0

    actual_prev_closes: list[float] = []

    # 取得 df 的日期序列，用於查 VIX
    if "Date" in df.columns:
        df_dates = pd.to_datetime(df["Date"])
    elif "Datetime" in df.columns:
        df_dates = pd.to_datetime(df["Datetime"])
    else:
        df_dates = pd.to_datetime(df.index)

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

        # v3: 取本輪訓練截止日，用於查詢 VIX
        cutoff_date = df_dates.iloc[cur_end - 1] if cur_end - 1 < len(df_dates) else None

        med, ohlc, all_c, p10, p90 = _run_one_step(
            close_hist, theta, args, full_window,
            param_model_payload=param_model_payload,
            agent_profile=agent_profile,
            vix_series=vix_series,
            cutoff_date=cutoff_date,
        )
        med  = med[:need]
        p10  = p10[:need]
        p90  = p90[:need]
        if ohlc is not None:
            ohlc = {k: v[:need] for k, v in ohlc.items()}

        actual_slice: np.ndarray | None = None
        coverage_val: float | None = None
        if actual_future_df is not None:
            sl = actual_future_df.iloc[bars_done: bars_done + need]
            if len(sl) > 0:
                actual_slice = sl["Close"].values.astype(float)

        if actual_slice is not None and len(actual_slice) == need:
            prev_actual_for_dir = np.concatenate([[start_price], actual_slice[:-1]])
            mae_pct  = float(np.mean(
                np.abs(med - actual_slice) / start_price * 100
            ))
            p_rets   = med - prev_actual_for_dir
            a_rets   = actual_slice - prev_actual_for_dir
            dir_hits = int(np.sum(np.sign(p_rets) == np.sign(a_rets)))
            coverage_val = float(compute_coverage(actual_slice, p10, p90))
        else:
            mae_pct  = None
            dir_hits = None

        round_details.append({
            "round":        rnd + 1,
            "start_price":  round(start_price, 4),
            "pred_end":     round(float(med[-1]), 4),
            "actual_end":   round(float(actual_slice[-1]), 4) if actual_slice is not None and len(actual_slice) > 0 else None,
            "mae_pct":      round(mae_pct, 4) if mae_pct is not None else None,
            "dir_hits":     dir_hits,
            "coverage":     round(coverage_val, 4) if coverage_val is not None else None,
            "bars":         need,
        })

        pred_closes.extend(med.tolist())
        pred_ohlcs.append(ohlc)
        pred_p10.extend(p10.tolist())
        pred_p90.extend(p90.tolist())

        if actual_slice is not None and len(actual_slice) == need:
            actual_prev_closes.extend(
                np.concatenate([[start_price], actual_slice[:-1]]).tolist()
            )
        else:
            actual_prev_closes.extend([None] * need)

        cur_end   += need
        bars_done += need

    pred_arr  = np.array(pred_closes)
    p10_arr   = np.array(pred_p10)
    p90_arr   = np.array(pred_p90)
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
        "p10":                p10_arr,
        "p90":                p90_arr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 圖表（加入 P10/P90 區間）
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
    p10    = result.get("p10", None)
    p90    = result.get("p90", None)
    total  = len(pred)

    fig, ax = plt.subplots(figsize=(16, 6))

    hist_x = np.arange(-len(hist_close), 0)
    ax.plot(hist_x, hist_close, color="#555", lw=1.2, label="歷史", zorder=4)

    # P10/P90 信心區間
    if p10 is not None and p90 is not None and len(p10) == total:
        x_band = np.arange(total)
        ax.fill_between(
            x_band, p10, p90,
            alpha=0.15, color="#42A5F5", label="P10–P90 信心區間", zorder=2
        )

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
    ax.set_title(f"{symbol}  Rolling {step}-bar Forward (Medoid+P10/P90)  [{end_date_str}]{suffix_display}")
    ax.set_xlabel("交易日（相對預測起點）")
    ax.set_ylabel("價格")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    round_details   = result["round_details"]
    maes            = [r["mae_pct"]  for r in round_details if r["mae_pct"]  is not None]
    coverages       = [r["coverage"] for r in round_details if r["coverage"] is not None]
    rounds_with_mae = [r["round"]    for r in round_details if r["mae_pct"]  is not None]

    if maes:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 3))
        avg = np.mean(maes)
        colors2 = ["#EF5350" if m > avg else "#26A69A" for m in maes]
        axes2[0].bar(rounds_with_mae, maes, color=colors2, alpha=0.8, width=0.6)
        axes2[0].axhline(avg, color="#888", lw=1, ls="--", label=f"平均 MAE={avg:.2f}%")
        axes2[0].set_title(f"{symbol}  每輪 MAE (Medoid){suffix_display}")
        axes2[0].set_xlabel("輪次")
        axes2[0].set_ylabel("MAE %")
        axes2[0].legend(fontsize=8)
        axes2[0].grid(True, alpha=0.3)

        if coverages:
            cov_rounds = [r["round"] for r in round_details if r["coverage"] is not None]
            avg_cov = np.mean(coverages)
            cov_colors = ["#66BB6A" if c >= 0.7 else "#FF7043" for c in coverages]
            axes2[1].bar(cov_rounds, [c * 100 for c in coverages],
                         color=cov_colors, alpha=0.8, width=0.6)
            axes2[1].axhline(80, color="#888", lw=1, ls="--", label="目標 80%")
            axes2[1].axhline(avg_cov * 100, color="#42A5F5", lw=1, ls=":",
                             label=f"平均={avg_cov*100:.1f}%")
            axes2[1].set_title(f"{symbol}  每輪覆蓋率 P10–P90{suffix_display}")
            axes2[1].set_xlabel("輪次")
            axes2[1].set_ylabel("覆蓋率 %")
            axes2[1].set_ylim(0, 110)
            axes2[1].legend(fontsize=8)
            axes2[1].grid(True, alpha=0.3)

        out2 = Path(output_dir)
        out2.mkdir(parents=True, exist_ok=True)
        suffix_file = f"_{label_suffix}" if label_suffix else ""
        mae_path = out2 / f"{symbol}_{end_date_str}_rolling{step}{suffix_file}_metrics.png"
        fig2.tight_layout()
        fig2.savefig(mae_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"✔ 指標圖 → {mae_path}")

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
    p10_arr         = result.get("p10", np.array([]))
    p90_arr         = result.get("p90", np.array([]))
    n               = len(pred)

    suffix_display = f" [{label_suffix}]" if label_suffix else ""
    sep = "─" * 96
    print(f"\n{sep}")
    print(f"  Rolling {step}-bar Forward (Medoid+P10/P90)  —  {symbol}  (共 {n} 根){suffix_display}")
    print(sep)
    hdr = (f"  {'Rnd':>3} {'Bar':>3}  "
           f"{'P-Close':>8} {'P10':>8} {'P90':>8}  "
           f"{'A-Close':>8} {'A-Ret%':>7}  "
           f"{'Err%':>7} {'Dir':>3} {'Cover':>5}")
    print(hdr)
    print("─" * 96)

    global_i    = 0
    close_errs  = []
    dir_correct = 0
    cover_hits  = 0
    total_bars_with_actual = 0

    for rnd_i, rnd in enumerate(result["round_details"]):
        bars_this = rnd["bars"]
        for b in range(bars_this):
            p_c      = float(pred[global_i])
            p10_v    = float(p10_arr[global_i]) if global_i < len(p10_arr) else p_c
            p90_v    = float(p90_arr[global_i]) if global_i < len(p90_arr) else p_c
            prev_act = float(actual_prev_arr[global_i]) if global_i < len(actual_prev_arr) and actual_prev_arr[global_i] is not None else None

            p_ret = (p_c - prev_act) / prev_act * 100 if prev_act else float("nan")

            if actual is not None and global_i < len(actual):
                a_c   = float(actual[global_i])
                a_ret = (a_c - prev_act) / prev_act * 100 if prev_act else float("nan")
                err   = (p_c - a_c) / a_c * 100
                close_errs.append(abs(err))
                total_bars_with_actual += 1

                p_up   = p_ret >= 0
                a_up   = a_ret >= 0
                match  = "✓" if p_up == a_up else "✗"
                in_pi  = "Y" if p10_v <= a_c <= p90_v else "N"
                if p_up == a_up:
                    dir_correct += 1
                if in_pi == "Y":
                    cover_hits += 1

                print(f"  {rnd_i+1:>3} {b+1:>3}  "
                      f"{p_c:>8.2f} {p10_v:>8.2f} {p90_v:>8.2f}  "
                      f"{a_c:>8.2f} {a_ret:>+7.2f}%  "
                      f"{err:>+7.2f}% {match:>3} {in_pi:>5}")
            else:
                print(f"  {rnd_i+1:>3} {b+1:>3}  "
                      f"{p_c:>8.2f} {p10_v:>8.2f} {p90_v:>8.2f}  "
                      f"{'N/A':>8} {'N/A':>7}  "
                      f"{'N/A':>7} {'N/A':>3} {'N/A':>5}")
            global_i += 1

        rnd_mae = rnd.get("mae_pct")
        rnd_dir = rnd.get("dir_hits")
        rnd_cov = rnd.get("coverage")
        if rnd_mae is not None:
            cov_str = f"  覆蓋率={rnd_cov*100:.1f}%" if rnd_cov is not None else ""
            print(f"  {'':>7}  → 本輪 MAE={rnd_mae:.2f}%  方向命中={rnd_dir}/{bars_this}{cov_str}")
        print("  " + "·" * 48)

    print(sep)
    if close_errs:
        tn = total_bars_with_actual
        cov_rate = cover_hits / tn * 100 if tn > 0 else 0
        print(f"  合計 {tn} 根  MAE={np.mean(close_errs):.2f}%  "
              f"MAX={np.max(close_errs):.2f}%  "
              f"方向命中={dir_correct}/{tn} ({dir_correct/tn*100:.0f}%)  "
              f"覆蓋率(P10-P90)={cov_rate:.1f}%")
    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# parse_args
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",              required=True)
    p.add_argument("--theta",               required=True)
    p.add_argument("--end-date",            default=None)
    p.add_argument("--total-bars",          type=int,   default=20)
    p.add_argument("--step",                type=int,   default=5)
    p.add_argument("--lookback",            type=int,   default=120)
    p.add_argument("--n-paths",             type=int,   default=500)
    p.add_argument("--calib-window",        type=int,   default=500)
    p.add_argument("--no-garch",            action="store_true")
    p.add_argument("--garch-model",         default="gjr-garch",
                   choices=["gjr-garch", "garch"])
    p.add_argument("--auto-calibrate",      action="store_true", default=True)
    p.add_argument("--param-model",         default=None)
    p.add_argument("--agent-profile",       default=None,
                   help="agent_profile.json 路徑，提供散戶/大戶行為參數")
    p.add_argument("--cache-dir",           default=None,
                   help="macro.parquet 所在目錄（預設自動偵測 cache/）")
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
    agent_profile       = load_agent_profile(getattr(args, "agent_profile", None))
    label_suffix = "agent+medoid" if agent_profile is not None else "medoid"

    # v3: 載入 VIX 序列（找不到 macro.parquet 時靜默略過）
    vix_series = _load_vix_series(cache_dir=getattr(args, "cache_dir", None))
    if vix_series is not None:
        print(f"[vix] 載入 {len(vix_series)} 根 VIX  ({vix_series.index[0].date()} ~ {vix_series.index[-1].date()})")
    else:
        print("[vix] macro.parquet 未找到，VIX 恐慌修正停用")

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
          f"agent={'ON' if agent_profile else 'OFF'}  "
          f"param_model={'ON' if param_model_payload else 'OFF'}")

    result = rolling_forward(
        df=df,
        train_end_idx=train_end_idx,
        theta=theta,
        args=args,
        actual_future_df=actual_future_df,
        param_model_payload=param_model_payload,
        agent_profile=agent_profile,
        vix_series=vix_series,
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

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol":         args.symbol,
        "end_date":       end_date_str,
        "step":           args.step,
        "total_bars":     args.total_bars,
        "start_price":    round(start_price, 4),
        "param_model":    str(args.param_model) if args.param_model else None,
        "agent_profile":  str(args.agent_profile) if args.agent_profile else None,
        "round_details":  result["round_details"],
        "dynamic_drift":  args.dynamic_drift,
        "dir_definition": "sign(pred-prev_actual)==sign(actual-prev_actual)",
        "path_selection": "medoid",
    }

    if result["actual_closes"] is not None:
        pred   = result["pred_closes"]
        actual = result["actual_closes"]
        prev_a = result["actual_prev_closes"]
        p10_a  = result["p10"]
        p90_a  = result["p90"]
        na     = min(len(pred), len(actual), len(prev_a))

        mae = float(np.mean(
            np.abs(pred[:na] - actual[:na]) / start_price * 100
        ))

        p_rets = np.array([pred[i] - prev_a[i] for i in range(na) if prev_a[i] is not None])
        a_rets = np.array([actual[i] - prev_a[i] for i in range(na) if prev_a[i] is not None])
        valid_n = len(p_rets)
        dirs    = int(np.sum(np.sign(p_rets) == np.sign(a_rets)))

        coverage = float(compute_coverage(actual[:na], p10_a[:na], p90_a[:na]))

        summary["overall_mae_pct"]   = round(mae, 4)
        summary["dir_accuracy_pct"]  = round(dirs / valid_n * 100, 2) if valid_n > 0 else None
        summary["coverage_p10_p90"]  = round(coverage * 100, 2)
        print(f"\n  整體 MAE={mae:.2f}%   方向命中={dirs}/{valid_n} ({dirs/valid_n*100:.0f}%)  "
              f"覆蓋率(P10-P90)={coverage*100:.1f}%")

    suffix_file = f"_{label_suffix}" if label_suffix else ""
    json_path = out_path / f"{args.symbol}_{end_date_str}_rolling{args.step}{suffix_file}.json"
    with open(json_path, "w") as fj:
        json.dump(summary, fj, indent=2, ensure_ascii=False)
    print(f"✔ 結果 JSON → {json_path}")


if __name__ == "__main__":
    main()
