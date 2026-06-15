"""
forward_study.py  v12.0  (動態 drift — trend-follow + mean-revert + vol-regime)

v12.0 新增：dynamic_drift_schedule()
  每根 K 棒的 drift 不再是固定衰減，而由三個力量疊加決定：

  1. Trend-follow component
     - 以「近 5 日 log-ret 均值」作為 momentum signal
     - blend_drift 作為基準，乘以 (1 + trend_strength × momentum_sign)
     - 每根衰減 drift_decay（同舊邏輯）

  2. Mean-revert component
     - 追蹤「目前路徑相對 BB backbone 的偏離量」
     - 偏離越多 → 往 backbone 拉的修正力越強
     - 係數 mr_rate（預設 0.08）

  3. Vol-regime scaling
     - 用 GJR-GARCH 滾動更新 σ_t（每 refit_every 根更新一次）
     - drift 振幅乘以 vol_regime_scale = σ_0 / σ_t（壓縮高波時的方向性）
     - 截斷在 [0.5, 2.0] 之間

  回傳 np.ndarray shape=(forecast,) 的 drift_schedule。

v11.6 已有：
  支撐/壓力語義修正
  - pivot_low 只能被選為支撐（低於現價）
  - pivot_high 只能被選為壓力（高於現價）

v11.5 已有：
  支撐/壓力偵測 — 雙軌合併法

v11.4 已有：
  支撐/壓力 drift 修正

v11.3 已有：
  1. momentum 翻轉偵測
  2. 前期快速衰減

v11.2 已有：
  blend_drift 動態截斷 MAX = 0.5 × σ_garch

用法：
  python scripts/forward_study.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --auto-calibrate \\
    --verbose

  # 開啟動態 drift（預設啟用）
  python scripts/forward_study.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --auto-calibrate \\
    --dynamic-drift \\
    --mr-rate 0.08 \\
    --trend-strength 0.5 \\
    --verbose

  # 停用（回到 v11 靜態衰減）
  python scripts/forward_study.py \\
    --symbol AAPL \\
    --theta results/theta_aapl.json \\
    --auto-calibrate \\
    --no-dynamic-drift
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
# 支撐/壓力偵測 — 方法一：價格密度直方圖
# ──────────────────────────────────────────────────────────────────────────────
def find_support_resistance(
    close_arr,
    window: int = 90,
    n_bins: int = 40,
    top_k: int = 3,
):
    prices = close_arr[-window:] if len(close_arr) >= window else close_arr
    counts, edges = np.histogram(prices, bins=n_bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    top_idx = np.argsort(counts)[::-1][:top_k]
    dense_levels = sorted(bin_centers[top_idx])
    return [(lv, "histogram") for lv in dense_levels]


# ──────────────────────────────────────────────────────────────────────────────
# 支撐/壓力偵測 — 方法二：局部 Swing Low / Swing High 聚集
# ──────────────────────────────────────────────────────────────────────────────
def find_local_extrema(
    close_arr,
    window: int = 90,
    order: int = 5,
    cluster_pct: float = 0.015,
):
    prices = np.asarray(close_arr[-window:] if len(close_arr) >= window else close_arr)
    n = len(prices)

    raw_lows  = []
    raw_highs = []
    for i in range(order, n - order):
        window_slice = prices[i - order: i + order + 1]
        if prices[i] == window_slice.min():
            raw_lows.append(float(prices[i]))
        if prices[i] == window_slice.max():
            raw_highs.append(float(prices[i]))

    def _cluster(pts):
        if not pts:
            return []
        pts_sorted = sorted(pts)
        clusters = [[pts_sorted[0]]]
        for p in pts_sorted[1:]:
            if (p - clusters[-1][-1]) / (clusters[-1][-1] + 1e-8) <= cluster_pct:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        return [float(np.mean(c)) for c in clusters]

    lows  = [(lv, "pivot_low")  for lv in _cluster(raw_lows)]
    highs = [(lv, "pivot_high") for lv in _cluster(raw_highs)]
    return lows, highs


# ──────────────────────────────────────────────────────────────────────────────
# 支撐/壓力修正主函式（雙軌合併，v11.6 語義修正）
# ──────────────────────────────────────────────────────────────────────────────
def compute_sr_drift_adjustment(
    close_arr,
    start_price: float,
    sr_window: int = 90,
    n_bins: int = 40,
    top_k: int = 5,
    pivot_order: int = 5,
    pivot_cluster_pct: float = 0.015,
    support_zone_pct: float = 0.035,
    resist_zone_pct: float = 0.035,
    max_support_pull: float = 0.0005,
    max_resist_push: float = 0.0003,
):
    hist_levels = find_support_resistance(
        close_arr, window=sr_window, n_bins=n_bins, top_k=top_k
    )
    hist_supports    = [(lv, src) for lv, src in hist_levels if lv < start_price]
    hist_resistances = [(lv, src) for lv, src in hist_levels if lv > start_price]

    pivot_lows, pivot_highs = find_local_extrema(
        close_arr, window=sr_window, order=pivot_order,
        cluster_pct=pivot_cluster_pct,
    )
    pivot_supports    = [(lv, src) for lv, src in pivot_lows  if lv < start_price]
    pivot_resistances = [(lv, src) for lv, src in pivot_highs if lv > start_price]

    supports    = hist_supports    + pivot_supports
    resistances = hist_resistances + pivot_resistances

    nearest_support_tuple  = max(supports,    key=lambda x: x[0]) if supports    else None
    nearest_resist_tuple   = min(resistances, key=lambda x: x[0]) if resistances else None

    nearest_support      = nearest_support_tuple[0]  if nearest_support_tuple  else None
    nearest_support_src  = nearest_support_tuple[1]  if nearest_support_tuple  else None
    nearest_resist       = nearest_resist_tuple[0]   if nearest_resist_tuple   else None
    nearest_resist_src   = nearest_resist_tuple[1]   if nearest_resist_tuple   else None

    drift_adj    = 0.0
    support_pull = 0.0
    resist_push  = 0.0

    if nearest_support is not None:
        dist_sup = (start_price - nearest_support) / start_price
        if 0 < dist_sup < support_zone_pct:
            support_pull = (1.0 - dist_sup / support_zone_pct) * max_support_pull
            drift_adj += support_pull

    if nearest_resist is not None:
        dist_res = (nearest_resist - start_price) / start_price
        if 0 < dist_res < resist_zone_pct:
            resist_push = (1.0 - dist_res / resist_zone_pct) * max_resist_push
            drift_adj -= resist_push

    sr_info = {
        "near_support":      round(nearest_support,  4) if nearest_support  is not None else None,
        "near_support_src":  nearest_support_src,
        "near_resist":       round(nearest_resist,   4) if nearest_resist   is not None else None,
        "near_resist_src":   nearest_resist_src,
        "dist_support":      round((start_price - nearest_support)  / start_price, 4) if nearest_support  is not None else None,
        "dist_resist":       round((nearest_resist  - start_price)  / start_price, 4) if nearest_resist   is not None else None,
        "support_pull":      round(support_pull,  6),
        "resist_push":       round(resist_push,   6),
        "drift_adj":         round(drift_adj,     6),
        "zone_pct":          support_zone_pct,
    }
    return drift_adj, sr_info


# ──────────────────────────────────────────────────────────────────────────────
# v12：動態 drift schedule
# ──────────────────────────────────────────────────────────────────────────────
def dynamic_drift_schedule(
    close_hist: np.ndarray,
    blend_drift: float,
    backbone_schedule: np.ndarray,     # shape (forecast,)  backbone 中位路徑價格
    forecast: int,
    drift_decay: float       = 0.04,
    early_decay_bars: int    = 10,
    mr_rate: float           = 0.08,   # 均值回歸係數
    trend_strength: float    = 0.5,    # trend-follow 放大係數
    vol_regime_refit: int    = 5,      # 每幾根重新估 vol
    garch_model: str         = "gjr-garch",
    drift_clamp_max: float | None = None,
    momentum_signal: float   = 0.0,    # 外部傳入的 short_drift
) -> tuple[np.ndarray, list[dict]]:
    """
    每根 K 棒的 drift 由三個分量疊加：

        d_t = d_base_t                     ← 衰減的基礎 drift
            + trend_t                      ← trend-follow 放大
            + mr_t                         ← mean-revert 向 backbone 拉
            × vol_regime_scale_t           ← vol-regime 壓縮/放大

    回傳：
        drift_schedule  : np.ndarray shape=(forecast,)
        debug_log       : list[dict]  每根的分量細節
    """
    log_rets = np.diff(np.log(close_hist))

    # 初始 vol 估計（作為基準 σ₀）
    sigma_0, _ = garch_vol_forecast(close_hist, model_type=garch_model) \
        if _ARCH_AVAILABLE else (None, {})
    if sigma_0 is None or sigma_0 <= 0:
        sigma_0 = float(np.std(log_rets[-21:])) if len(log_rets) >= 21 else float(np.std(log_rets))

    fast_decay  = min(drift_decay * 2.0, 0.99)
    base_decay  = drift_decay

    current_drift = blend_drift
    # 模擬路徑用來計算偏離 backbone（以 log-price 追蹤）
    sim_log_price = np.log(float(close_hist[-1]))
    sigma_t       = sigma_0

    drift_schedule = np.empty(forecast)
    debug_log      = []

    for t in range(forecast):
        # ── 1. 衰減基礎 ──
        decay = fast_decay if t < early_decay_bars else base_decay
        current_drift *= (1.0 - decay)

        # ── 2. Trend-follow ──
        mom_sign = np.sign(momentum_signal) if abs(momentum_signal) > 1e-8 else 0.0
        trend_component = trend_strength * abs(current_drift) * mom_sign

        # ── 3. Mean-revert 向 backbone ──
        bb_log = np.log(float(backbone_schedule[t]) + 1e-10)
        mr_component = mr_rate * (bb_log - sim_log_price)

        # ── 4. Vol-regime scaling（每 vol_regime_refit 根重估）──
        if _ARCH_AVAILABLE and t % vol_regime_refit == 0 and t > 0:
            # 用假想的歷史 + 模擬 drift 估計 sigma_t（輕量：只更新 EWMA）
            ewma_sigma = float(np.sqrt(
                0.94 * sigma_t ** 2 + 0.06 * current_drift ** 2
            ))
            sigma_t = max(ewma_sigma, 1e-6)

        vol_scale_t = float(np.clip(sigma_0 / (sigma_t + 1e-8), 0.5, 2.0))

        # ── 合併 ──
        d_t = (current_drift + trend_component + mr_component) * vol_scale_t

        # ── clamp ──
        if drift_clamp_max is not None and drift_clamp_max > 0:
            d_t = float(np.clip(d_t, -drift_clamp_max, drift_clamp_max))

        drift_schedule[t] = d_t

        # 更新模擬路徑（用以計算下一步偏離量）
        sim_log_price += d_t

        debug_log.append({
            "t":               t,
            "base_drift":      round(current_drift * 100, 5),
            "trend":           round(trend_component * 100, 5),
            "mr":              round(mr_component * 100, 5),
            "vol_scale":       round(vol_scale_t, 4),
            "final_drift":     round(d_t * 100, 5),
        })

    return drift_schedule, debug_log


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
    p.add_argument("--reversal-window", type=int,  default=3,
                   help="連漲/連跌幾根觸發 momentum 翻轉歸零 (預設 3)")
    p.add_argument("--early-decay-bars", type=int, default=10,
                   help="前 N 根使用 2× drift_decay，之後恢復正常 (預設 10)")
    # 支撐壓力
    p.add_argument("--sr-window",      type=int,   default=90)
    p.add_argument("--sr-bins",        type=int,   default=40)
    p.add_argument("--sr-top-k",       type=int,   default=5)
    p.add_argument("--sr-pivot-order", type=int,   default=5)
    p.add_argument("--sr-zone-pct",    type=float, default=0.035)
    p.add_argument("--no-sr",           action="store_true")
    # v12：動態 drift
    p.add_argument("--dynamic-drift",   action="store_true", default=True,
                   help="啟用動態 drift schedule（預設啟用）")
    p.add_argument("--no-dynamic-drift", dest="dynamic_drift", action="store_false",
                   help="停用動態 drift，回到 v11 靜態衰減")
    p.add_argument("--mr-rate",        type=float, default=0.08,
                   help="mean-revert 向 backbone 的係數（預設 0.08）")
    p.add_argument("--trend-strength", type=float, default=0.5,
                   help="trend-follow 放大係數（預設 0.5）")
    p.add_argument("--vol-refit",      type=int,   default=5,
                   help="vol-regime EWMA 每幾根更新一次（預設 5）")
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
                      last_drift: float = 0.0, drift_clamp_max: float | None = None,
                      momentum_reversed: bool = False,
                      sr_info: dict | None = None,
                      drift_debug: list[dict] | None = None):
    print("\n" + "─" * 60)
    print("  VERBOSE DIAGNOSTICS (v12.0)")
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
    print(f"  momentum_bias   : {momentum_bias:+.4f}")
    print(f"  breakout_state  : {breakout_state}")
    if momentum_reversed:
        print(f"  ⚡ momentum 翻轉偵測：連漲 {args.reversal_window} 根 + blend_drift<0 → momentum_bias 歸零")
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

    # 支撐壓力
    if sr_info is not None:
        zone_pct = sr_info.get("zone_pct", 0.035)
        print(f"\n  [支撐壓力]  門檻={zone_pct*100:.1f}%")
        ns     = sr_info.get("near_support")
        ns_src = sr_info.get("near_support_src", "")
        nr     = sr_info.get("near_resist")
        nr_src = sr_info.get("near_resist_src", "")
        sp     = sr_info.get("support_pull", 0.0)
        rp     = sr_info.get("resist_push",  0.0)
        da     = sr_info.get("drift_adj",    0.0)
        if ns is not None:
            dist_s = sr_info.get("dist_support", 0.0)
            pull_str = f"→ drift +{sp*100:.4f}%/day" if sp > 0 else f"(距離 > {zone_pct*100:.1f}%，無修正)"
            print(f"  near_support  : {ns:.2f}  [{ns_src}]  (距起點 -{dist_s*100:.2f}%)  {pull_str}")
        else:
            print(f"  near_support  : 無")
        if nr is not None:
            dist_r = sr_info.get("dist_resist", 0.0)
            push_str = f"→ drift -{rp*100:.4f}%/day" if rp > 0 else f"(距離 > {zone_pct*100:.1f}%，無修正)"
            print(f"  near_resist   : {nr:.2f}  [{nr_src}]  (距起點 +{dist_r*100:.2f}%)  {push_str}")
        else:
            print(f"  near_resist   : 無")
        if abs(da) > 1e-9:
            print(f"  ✅ SR drift 修正 : {da*100:+.4f}%/day")

    # v12 動態 drift 摘要
    if drift_debug:
        print(f"\n  [動態 drift schedule (v12)]  前5根：")
        hdr = f"  {'t':>3}  {'base%':>8}  {'trend%':>8}  {'mr%':>8}  {'vol_scale':>9}  {'final%':>8}"
        print(hdr)
        for row in drift_debug[:5]:
            print(f"  {row['t']:>3}  {row['base_drift']:>8.4f}  {row['trend']:>8.4f}  "
                  f"{row['mr']:>8.4f}  {row['vol_scale']:>9.4f}  {row['final_drift']:>8.4f}")
        if len(drift_debug) > 5:
            print(f"  ... (共 {len(drift_debug)} 根)")

    print(f"\n  [最終模擬參數]")
    print(f"  intra_bar       : {args.intra_bar}")
    print(f"  momentum_boost  : {args.momentum_boost}")
    print(f"  drift_decay     : {args.drift_decay}")
    print(f"  vol_multiplier  : {args.vol_multiplier}")
    print(f"  drift_scale     : {args.drift_scale}")
    print(f"  dynamic_drift   : {args.dynamic_drift}  (mr_rate={args.mr_rate}  trend_strength={args.trend_strength})")
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
    close_errs   = []

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
    sr_info: dict | None = None,
    drift_schedule: np.ndarray | None = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [3, 1]},
                             sharex=False)
    ax   = axes[0]
    ax_d = axes[1]

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

    if sr_info is not None:
        ns     = sr_info.get("near_support")
        ns_src = sr_info.get("near_support_src", "")
        nr     = sr_info.get("near_resist")
        nr_src = sr_info.get("near_resist_src", "")
        if ns is not None:
            ax.axhline(ns, color="#FF8F00", lw=0.8, ls="--", alpha=0.7,
                       label=f"支撐 {ns:.2f} [{ns_src}]")
        if nr is not None:
            ax.axhline(nr, color="#AB47BC", lw=0.8, ls="--", alpha=0.7,
                       label=f"壓力 {nr:.2f} [{nr_src}]")

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
    ax.set_ylabel("價格")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [bull_patch, bear_patch],
              labels  + ["預測漲K", "預測跌K"],
              loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 下圖：動態 drift ──
    if drift_schedule is not None and len(drift_schedule) > 0:
        colors = ["#26A69A" if d >= 0 else "#EF5350" for d in drift_schedule]
        ax_d.bar(np.arange(len(drift_schedule)), drift_schedule * 100,
                 color=colors, width=0.6, alpha=0.8)
        ax_d.axhline(0, color="#888888", lw=0.7)
        ax_d.set_ylabel("drift (%/day)")
        ax_d.set_xlabel("交易日（相對預測起點）")
        ax_d.set_title("動態 drift schedule (v12)", fontsize=9)
        ax_d.grid(True, alpha=0.2)
    else:
        ax_d.set_visible(False)

    fig.tight_layout()
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
        print(f"\n[auto-calibrate v12.0] 掃描前 {calib_window} 根 K 棒  (vol_model={model_label})...")
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

    # ── 動態截斷 blend_drift ──
    drift_clamp_max = args.drift_clamp
    if drift_clamp_max is not None and drift_clamp_max > 0:
        blend_drift_raw = blend_drift
        blend_drift = float(np.clip(blend_drift, -drift_clamp_max, drift_clamp_max))
        if abs(blend_drift_raw - blend_drift) > 1e-8:
            print(f"  [drift clamp] {blend_drift_raw*100:+.4f}%/day → {blend_drift*100:+.4f}%/day  "
                  f"(MAX=±{drift_clamp_max*100:.4f}%/day)")

    # ── 支撐壓力修正 ──
    sr_info: dict | None = None
    if not args.no_sr:
        sr_close_arr = df.iloc[:train_end_idx]["Close"].values.astype(float)
        drift_adj, sr_info = compute_sr_drift_adjustment(
            sr_close_arr,
            start_price=start_price,
            sr_window=args.sr_window,
            n_bins=args.sr_bins,
            top_k=args.sr_top_k,
            pivot_order=args.sr_pivot_order,
            support_zone_pct=args.sr_zone_pct,
            resist_zone_pct=args.sr_zone_pct,
        )
        if abs(drift_adj) > 1e-9:
            blend_drift += drift_adj
            if drift_clamp_max is not None and drift_clamp_max > 0:
                blend_drift = float(np.clip(blend_drift, -drift_clamp_max, drift_clamp_max))

    # ── v12：backbone forward schedule（供 mean-revert 參考）──
    drift_fwd   = np.full(args.forecast, blend_drift)
    bb_fwd      = start_price * np.cumprod(1 + drift_fwd)
    vol_fwd     = np.full(args.forecast, last_vol) * (args.vol_multiplier or 1.0)

    # ── v11.3：momentum 翻轉偵測 ──
    ESTIMATOR_LB = 500
    rv_window    = min(21, len(close_hist))
    rv           = float(np.std(np.diff(np.log(close_hist[-rv_window:]))))
    vol_scale    = float(np.clip(rv / max(theta.vol, 1e-8), 0.6, 4.0))

    estimate_df = df.iloc[max(0, train_end_idx - ESTIMATOR_LB): train_end_idx]
    estimator   = MarketParameterEstimator(lookback=ESTIMATOR_LB, vp_bins=40, momentum_window=10)
    base_params = estimator.fit(estimate_df, symbol=args.symbol)
    params_fwd  = build_params_from_theta(theta, base_params)

    import dataclasses
    params_fwd = dataclasses.replace(params_fwd, last_close=start_price)
    momentum_bias   = float(getattr(params_fwd, "momentum_bias",       0.0))
    breakout_state  = int(getattr(params_fwd,   "node_breakout_state", 0))

    momentum_reversed = False
    reversal_win = max(1, args.reversal_window)
    if blend_drift < 0 and len(log_ret_hist) >= reversal_win:
        recent_rets = log_ret_hist[-reversal_win:]
        if all(r > 0 for r in recent_rets):
            momentum_bias = 0.0
            momentum_reversed = True

    # ── v12：動態 drift schedule ──
    base_decay  = float(args.drift_decay or 0.04)
    drift_debug: list[dict] = []

    if args.dynamic_drift:
        drift_schedule, drift_debug = dynamic_drift_schedule(
            close_hist=close_hist,
            blend_drift=blend_drift,
            backbone_schedule=bb_fwd,
            forecast=args.forecast,
            drift_decay=base_decay,
            early_decay_bars=args.early_decay_bars,
            mr_rate=args.mr_rate,
            trend_strength=args.trend_strength,
            vol_regime_refit=args.vol_refit,
            garch_model=args.garch_model,
            drift_clamp_max=drift_clamp_max,
            momentum_signal=short_drift,
        )
    else:
        # 回退 v11 靜態衰減
        fast_decay    = min(base_decay * 2.0, 0.99)
        current_drift = blend_drift
        drift_schedule = np.empty(args.forecast)
        for t in range(args.forecast):
            drift_schedule[t] = current_drift
            decay = fast_decay if t < args.early_decay_bars else base_decay
            current_drift *= (1.0 - decay)

    if args.verbose:
        print_diagnostics(
            args, calib_info, model_predicted_params,
            momentum_bias=momentum_bias,
            breakout_state=breakout_state,
            blend_drift=blend_drift,
            short_drift=short_drift,
            last_drift=last_drift,
            drift_clamp_max=drift_clamp_max,
            momentum_reversed=momentum_reversed,
            sr_info=sr_info,
            drift_debug=drift_debug if args.dynamic_drift else None,
        )

    # ── simulate ──
    end_date_str = end_dt.strftime("%Y-%m-%d")
    dyn_tag = "dyn" if args.dynamic_drift else "static"
    print(f"\n[simulate v12] {args.symbol}  起點={end_date_str}  forecast={args.forecast}  n_paths={args.n_paths}  drift={dyn_tag}")
    print(f"  blend_drift={blend_drift*100:+.4f}%/day  "
          f"(short={short_drift*100:+.4f}%  long={last_drift*100:+.4f}%  w={w})")
    if sr_info and abs(sr_info.get("drift_adj", 0.0)) > 1e-9:
        print(f"  SR adj={sr_info['drift_adj']*100:+.4f}%/day")
    print(f"  momentum_bias={momentum_bias:+.4f}  breakout_state={breakout_state}"
          + ("  [翻轉歸零]" if momentum_reversed else ""))
    print(f"  drift_scale={args.drift_scale}  drift_decay={base_decay}  "
          f"vol_multiplier={args.vol_multiplier}  momentum_boost={args.momentum_boost}")
    if args.dynamic_drift:
        print(f"  mr_rate={args.mr_rate}  trend_strength={args.trend_strength}  vol_refit={args.vol_refit}")

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
        drift_schedule=drift_schedule,
        vol_schedule=vol_fwd,
        backbone_schedule=bb_fwd,
        backbone_mr_coeff=0.06,
        intra_bar_steps=args.intra_bar or 2,
        drift_decay_rate=base_decay,
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
    mode_tag  = (f"v12-{dyn_tag}-{args.garch_model}(w={args.calib_window}){model_tag}"
                 if args.auto_calibrate else f"v12-{dyn_tag}-manual")
    out_path  = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "symbol":       args.symbol,
        "end_date":     end_date_str,
        "start_price":  round(start_price, 4),
        "final_pred":   round(final_pred, 4),
        "total_ret_pct": round(total_ret, 4),
        "blend_drift":  round(blend_drift * 100, 5),
        "mode":         mode_tag,
        "dynamic_drift": args.dynamic_drift,
        "mr_rate":      args.mr_rate,
        "trend_strength": args.trend_strength,
        "sr_info":      sr_info,
    }
    if actual_close is not None and len(actual_close) > 0:
        result_dict["actual_ret_pct"] = round(actual_ret, 4)
        result_dict["mae_pct"]        = round(mae, 4)

    json_path = out_path / f"{args.symbol}_{end_date_str}_{mode_tag}.json"
    with open(json_path, "w") as fj:
        json.dump(result_dict, fj, indent=2, ensure_ascii=False)
    print(f"✔ 結果已儲存 → {json_path}")

    render_forecast(
        symbol=args.symbol,
        hist_close=close_hist,
        result=result,
        forecast=args.forecast,
        end_date_str=end_date_str,
        output_dir=args.output_dir,
        mode_tag=mode_tag,
        rep_close=rep_close,
        ohlc=ohlc,
        actual_close=actual_close,
        actual_ohlc=actual_ohlc,
        sr_info=sr_info,
        drift_schedule=drift_schedule,
    )


if __name__ == "__main__":
    main()
