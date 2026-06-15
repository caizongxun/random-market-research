"""
run_pipeline.py
===============
一鍵執行完整流程：

  Step 1  下載歷史資料
  Step 2  自動校準基礎 theta（calibrate_params.py 邏輯）
  Step 3  分析動態 fear_threshold（analyze_fear_threshold.py 邏輯）
  Step 4  收集訓練資料（collect_training_data.py 邏輯，sliding window）
  Step 5  訓練參數代理模型（GBM → joblib，取代反覆擬合的耗時步驟）
  Step 6  執行 Rolling Forward（rolling_forward.py 邏輯）
          ├─ 6a  無 param-model（純 auto_calibrate）
          └─ 6b  有 param-model（加速版）
  Step 7  輸出對比報告（JSON + HTML 摘要）

用法
----
  # 最簡單：只給 symbol
  python scripts/run_pipeline.py --symbol AAPL

  # 多股 + 自訂結束日 + 回測窗口
  python scripts/run_pipeline.py \\
    --symbol AAPL MSFT NVDA \\
    --end-date 2025-06-01 \\
    --total-bars 30 \\
    --step 5 \\
    --output-dir results/pipeline

  # 跳過已完成的步驟（斷點續跑）
  python scripts/run_pipeline.py --symbol AAPL --skip-steps 1 2

  # 只跑到 Step 5（不做 rolling forward，純訓練模型）
  python scripts/run_pipeline.py --symbol AAPL --stop-after 5

設計理念
--------
1. 「慢流程」→ 「快流程」
   calibrate / analyze_fear / collect_data 只要不是最新資料就重用快取。
   GBM 代理模型把「每次都要 zigzag + grid-search 擬合」壓縮到毫秒級推論。

2. 動態 fear_threshold 整合進 rolling_forward
   每輪預測前從 fear_profile.json 讀取 regime 對應的 threshold，
   透過 USStockFutureSimulator 的 fear_state 控制高位行為。

3. 代理模型（param surrogate）
   輸入：100+ 市場特徵（vol、drift、GARCH、backbone、SR...）
   輸出：drift_scale / drift_decay（原本 auto_calibrate 每次要重新擬合）
   使用 LightGBM（自動退回 sklearn GradientBoosting）。
   訓練完後做 3-fold walk-forward CV 輸出 MAE。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# 讓 src/ 可被 import
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title: str):
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


def _cache_valid(path: Path, max_age_hours: float = 23.0) -> bool:
    """若檔案存在且未超過 max_age_hours，視為有效快取。"""
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return age_h < max_age_hours


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — 下載資料
# ══════════════════════════════════════════════════════════════════════════════

def step1_fetch(
    symbol: str,
    years: int,
    cache_dir: Path,
    end_date: str | None,
    force: bool = False,
) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol}_ohlcv.parquet"
    if not force and _cache_valid(cache_path):
        log.info(f"[{symbol}] Step 1  快取命中 → {cache_path}")
        return pd.read_parquet(cache_path)

    end_dt  = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
    # 多抓 1 年供後續滾動評估用
    dl_end  = end_dt + pd.DateOffset(days=252)
    start   = end_dt - pd.DateOffset(years=years)

    log.info(f"[{symbol}] Step 1  下載 {start.date()} ~ {dl_end.date()} ...")
    raw = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=dl_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else "Datetime"
    raw = raw.rename(columns={date_col: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.dropna(subset=["Close"]).reset_index(drop=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache_path, index=False)
    log.info(f"[{symbol}] Step 1  {len(raw)} 根 K 棒 → 快取 {cache_path}")
    return raw


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — 自動校準 theta
# ══════════════════════════════════════════════════════════════════════════════

def step2_calibrate(
    symbol: str,
    df: pd.DataFrame,
    cache_dir: Path,
    calib_bars: int = 500,
    force: bool = False,
) -> dict:
    """
    使用 calibrate_params.py 的校準邏輯。
    輸出 theta dict（含 vol, drift, mr_coeff, ...）儲存為 JSON 快取。
    """
    cache_path = cache_dir / f"{symbol}_theta.json"
    if not force and _cache_valid(cache_path, max_age_hours=23):
        log.info(f"[{symbol}] Step 2  快取命中 → {cache_path}")
        return _load_json(cache_path)

    log.info(f"[{symbol}] Step 2  校準 theta（最近 {calib_bars} 根）...")
    from market_estimator     import MarketParameterEstimator
    from backbone_fitter      import BackboneFitter
    from calibrated_simulator import CalibratedTheta

    sub   = df.tail(calib_bars).copy()
    close = sub["Close"].values.astype(float)

    # 基礎統計
    log_rets = np.diff(np.log(close))
    vol_all  = float(np.std(log_rets))
    drift    = float(np.mean(log_rets))

    # backbone 最後一段
    fitter     = BackboneFitter(n_seg=6, smooth_reg=0.5)
    bb         = fitter.fit(close)
    last_drift = float(bb.segment_drifts[-1])
    last_vol   = float(bb.segment_vols[-1])

    # MarketParameterEstimator
    estimator = MarketParameterEstimator(
        lookback=calib_bars, vp_bins=40, momentum_window=10
    )
    params = estimator.fit(sub, symbol=symbol)

    theta = CalibratedTheta(
        vol              = float(last_vol),
        drift            = float(last_drift),
        mr_coeff         = float(getattr(params, "mr_coeff", 0.05)),
        node_coeff       = float(getattr(params, "node_coeff", 0.05)),
        momentum_strength= float(getattr(params, "momentum_bias", 0.0)),
        momentum_decay   = float(getattr(params, "momentum_decay", 0.9)),
        breakout_boost   = float(getattr(params, "breakout_boost", 1.0)),
    )
    theta_dict = theta.to_dict()
    # 附加診斷欄位
    theta_dict["symbol"]    = symbol
    theta_dict["vol_all"]   = round(vol_all, 6)
    theta_dict["drift_all"] = round(drift, 6)
    theta_dict["calib_bars"]= calib_bars

    _save_json(cache_path, theta_dict)
    log.info(f"[{symbol}] Step 2  θ.vol={theta.vol:.4f}  θ.drift={theta.drift:.4f}  → {cache_path}")
    return theta_dict


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — 動態 fear_threshold
# ══════════════════════════════════════════════════════════════════════════════

def step3_fear_threshold(
    symbol: str,
    df: pd.DataFrame,
    cache_dir: Path,
    pct_drop: float = 0.03,
    min_gain: float = 0.02,
    lookahead: int = 10,
    rolling_window: int = 252,
    target_peak_prob: float = 0.45,    # 放寬為 45% 避免找不到有效閾值
    recent_window: int = 63,
    threshold_method: str = "median",
    force: bool = False,
) -> dict:
    cache_path = cache_dir / f"{symbol}_fear_profile.json"
    if not force and _cache_valid(cache_path, max_age_hours=23):
        log.info(f"[{symbol}] Step 3  快取命中 → {cache_path}")
        return _load_json(cache_path)

    log.info(f"[{symbol}] Step 3  分析動態 fear_threshold ...")
    # 直接複用 analyze_fear_threshold 的函式
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_fear_threshold import (
        find_swing_ups,
        compute_threshold_stats,
        compute_dynamic_threshold,
        current_fear_threshold,
        build_fear_profile,
    )

    close  = df["Close"].values.astype(float)
    dates  = df["Date"]
    thresholds = [round(x, 3) for x in np.arange(0.02, 0.26, 0.01)]

    swings    = find_swing_ups(close, pct_drop=pct_drop, min_gain=min_gain)
    stats_df  = compute_threshold_stats(close, swings, thresholds, lookahead=lookahead)
    dyn_df    = compute_dynamic_threshold(
        close, dates,
        window=rolling_window,
        pct_drop=pct_drop,
        min_gain=min_gain,
        target_peak_prob=target_peak_prob,
        thresholds=thresholds,
        lookahead=lookahead,
    )
    cur_thr = current_fear_threshold(dyn_df, recent_window=recent_window, method=threshold_method)

    profile = build_fear_profile(symbol, stats_df, dyn_df, cur_thr, str(cache_dir))
    log.info(
        f"[{symbol}] Step 3  fear_threshold={cur_thr*100:.1f}%  "
        f"low_vol={profile['threshold_by_regime']['low_vol']*100:.1f}%  "
        f"high_vol={profile['threshold_by_regime']['high_vol']*100:.1f}%"
    )
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — 收集訓練資料（rolling window → feature matrix）
# ══════════════════════════════════════════════════════════════════════════════

def step4_collect_training_data(
    symbol: str,
    df: pd.DataFrame,
    theta_dict: dict,
    cache_dir: Path,
    calib_window: int = 500,
    step_bars: int = 5,
    force: bool = False,
) -> pd.DataFrame:
    """
    每隔 step_bars 根 K 棒往回看 calib_window 根，
    提取特徵 + 跑 auto_calibrate 計算目標值。
    輸出 parquet 快取。
    """
    cache_path = cache_dir / f"{symbol}_training_data.parquet"
    if not force and _cache_valid(cache_path, max_age_hours=23):
        log.info(f"[{symbol}] Step 4  快取命中 ({cache_path})")
        return pd.read_parquet(cache_path)

    log.info(f"[{symbol}] Step 4  收集訓練資料（可能需要 30-120 秒）...")

    from calibrated_simulator import CalibratedTheta
    from forward_study        import auto_calibrate

    theta = CalibratedTheta.from_dict(theta_dict)

    records = []
    close_arr = df["Close"].values.astype(float)
    n = len(close_arr)

    # 從 calib_window 開始，每 step_bars 根取一個樣本
    for end_i in range(calib_window, n, step_bars):
        win   = df.iloc[max(0, end_i - calib_window): end_i].copy()
        close = win["Close"].values.astype(float)
        if len(close) < 100:
            continue

        try:
            calib = auto_calibrate(close, theta, calib_window=min(500, len(close)))
        except Exception:
            continue

        # 特徵提取（與 collect_training_data.py 保持一致）
        log_r = np.diff(np.log(close))
        vol20  = float(np.std(log_r[-20:])) if len(log_r) >= 20 else float(np.std(log_r))
        vol60  = float(np.std(log_r[-60:])) if len(log_r) >= 60 else float(np.std(log_r))
        d20    = float(np.mean(log_r[-20:])) if len(log_r) >= 20 else float(np.mean(log_r))
        d60    = float(np.mean(log_r[-60:])) if len(log_r) >= 60 else float(np.mean(log_r))
        dall   = float(np.mean(log_r))
        vall   = float(np.std(log_r))
        ac1    = float(pd.Series(log_r).autocorr(1))
        ac1    = 0.0 if np.isnan(ac1) else ac1

        try:
            from backbone_fitter import BackboneFitter
            bb = BackboneFitter(n_seg=6, smooth_reg=0.5).fit(close[-120:])
            bbd  = float(bb.segment_drifts[-1]) * 100
            bbv  = float(bb.segment_vols[-1])   * 100
            bbds = float(np.std(bb.segment_drifts)) * 100
        except Exception:
            bbd = bbv = bbds = 0.0

        rec = {
            "date":           str(df["Date"].iloc[end_i - 1])[:10],
            "feat_vol_20":    round(vol20 * 100, 4),
            "feat_vol_60":    round(vol60 * 100, 4),
            "feat_vol_all":   round(vall  * 100, 4),
            "feat_vol_r2060": round(vol20 / (vol60 + 1e-8), 4),
            "feat_vol_r_theta": round(vol20 / (theta.vol + 1e-8), 4),
            "feat_drift_20":  round(d20  * 100, 4),
            "feat_drift_60":  round(d60  * 100, 4),
            "feat_drift_all": round(dall * 100, 4),
            "feat_drift_r":   round(d20 / (abs(dall) + 1e-8) * np.sign(dall + 1e-12), 4),
            "feat_autocorr":  round(ac1, 4),
            "feat_bb_drift":  round(bbd, 4),
            "feat_bb_vol":    round(bbv, 4),
            "feat_bb_ds":     round(bbds, 4),
            "target_drift_scale": round(float(calib["drift_scale"]), 4),
            "target_drift_decay": round(float(calib["drift_decay"]), 4),
        }

        # GARCH 特徵（optional，失敗不中斷）
        try:
            from arch import arch_model
            rets = pd.Series(np.diff(np.log(close)) * 100).dropna()
            if len(rets) >= 60:
                am  = arch_model(rets, vol="Garch", p=1, o=1, q=1, dist="t")
                res = am.fit(disp="off", show_warning=False)
                pr  = res.params
                rec["feat_garch_alpha"]       = round(float(pr.get("alpha[1]", 0)), 4)
                rec["feat_garch_beta"]        = round(float(pr.get("beta[1]",  0)), 4)
                rec["feat_garch_persistence"] = round(
                    float(pr.get("alpha[1]", 0) + pr.get("beta[1]", 0)
                          + 0.5 * pr.get("gamma[1]", 0)), 4
                )
        except Exception:
            pass

        records.append(rec)

    if not records:
        raise RuntimeError(f"[{symbol}] Step 4: 無法收集任何訓練樣本")

    train_df = pd.DataFrame(records)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(cache_path, index=False)
    log.info(f"[{symbol}] Step 4  {len(train_df)} 筆樣本 → {cache_path}")
    return train_df


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — 訓練代理模型（Param Surrogate）
# ══════════════════════════════════════════════════════════════════════════════

def step5_train_surrogate(
    symbol: str,
    train_df: pd.DataFrame,
    cache_dir: Path,
    n_cv_folds: int = 3,
    force: bool = False,
) -> Path:
    """
    訓練 LightGBM（或 sklearn GBM fallback）代理模型，
    輸出 joblib 模型檔。
    """
    model_path = cache_dir / f"param_model_{symbol}.joblib"
    if not force and model_path.exists():
        log.info(f"[{symbol}] Step 5  快取命中 → {model_path}")
        return model_path

    log.info(f"[{symbol}] Step 5  訓練代理模型 ({len(train_df)} 筆) ...")

    try:
        import joblib
    except ImportError:
        raise ImportError("需要 joblib：pip install joblib")

    feat_cols   = [c for c in train_df.columns if c.startswith("feat_")]
    target_cols = ["target_drift_scale", "target_drift_decay"]

    X = train_df[feat_cols].fillna(0).values
    Y = train_df[target_cols].values

    # 選擇 learner
    try:
        import lightgbm as lgb
        def _make_model():
            return lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=5, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10,
                n_jobs=-1, verbose=-1,
            )
        learner_name = "LightGBM"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        def _make_model():
            return GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8,
            )
        learner_name = "sklearn-GBM"

    models    = {}
    cv_scores = {}

    # Walk-forward CV
    fold_size = len(X) // (n_cv_folds + 1)
    for col_i, tgt in enumerate(target_cols):
        y = Y[:, col_i]
        maes = []
        for fold in range(n_cv_folds):
            split = (fold + 1) * fold_size
            X_tr, y_tr = X[:split],  y[:split]
            X_val, y_val = X[split:split + fold_size], y[split:split + fold_size]
            if len(X_val) == 0:
                continue
            m = _make_model()
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)
            maes.append(float(np.mean(np.abs(pred - y_val))))

        # 全量再訓一次
        final_m = _make_model()
        final_m.fit(X, y)
        models[tgt]    = final_m
        cv_scores[tgt] = float(np.mean(maes)) if maes else 0.0
        log.info(
            f"[{symbol}] Step 5  {tgt:>22}  walk-fwd MAE={cv_scores[tgt]:.4f}  ({learner_name})"
        )

    payload = {
        "symbol":       symbol,
        "model_name":   learner_name,
        "feature_cols": feat_cols,
        "models":       models,
        "cv_scores":    cv_scores,
        "n_samples":    len(train_df),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    log.info(f"[{symbol}] Step 5  模型已存 → {model_path}")
    return model_path


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Rolling Forward（整合 fear_profile + param_model）
# ══════════════════════════════════════════════════════════════════════════════

def _get_current_vol_regime(
    close: np.ndarray,
    window: int = 20,
    threshold_mult: float = 1.3,
) -> str:
    """判斷目前是否高波動體制（high_vol / low_vol）。"""
    if len(close) < window + 20:
        return "low_vol"
    recent_vol = float(np.std(np.diff(np.log(close[-window:]))))
    base_vol   = float(np.std(np.diff(np.log(close[-(window * 3):]))))
    return "high_vol" if recent_vol > base_vol * threshold_mult else "low_vol"


def step6_rolling_forward(
    symbol: str,
    df: pd.DataFrame,
    theta_dict: dict,
    fear_profile: dict,
    model_path: Path,
    end_date: str | None,
    args,
    output_dir: Path,
    use_model: bool = True,
) -> dict:
    """
    執行 rolling_forward，自動注入 fear_threshold（依 vol regime 動態選擇）
    以及 param_model（加速擬合）。
    """
    from calibrated_simulator import CalibratedTheta
    from rolling_forward      import (
        rolling_forward as _roll,
        render_rolling,
        print_rolling_comparison,
        load_param_model,
    )
    from forward_study import ensure_ohlcv

    label = "model" if use_model else "base"
    log.info(f"[{symbol}] Step 6{'' if use_model else '-base'}  Rolling Forward ({label}) ...")

    theta = CalibratedTheta.from_dict(theta_dict)
    param_model_payload = None
    if use_model and model_path.exists():
        try:
            import joblib
            param_model_payload = joblib.load(model_path)
        except Exception as e:
            log.warning(f"[{symbol}] 無法載入 param_model: {e}")

    # 取得 fear_threshold（依當前 vol regime 動態選擇）
    close_for_regime = df["Close"].values.astype(float)
    regime = _get_current_vol_regime(close_for_regime)
    fear_thr = fear_profile["threshold_by_regime"].get(regime,
                   fear_profile["fear_threshold"])
    log.info(f"[{symbol}] vol_regime={regime}  fear_threshold={fear_thr*100:.1f}%")

    # 對 rolling_forward args 注入 fear_threshold
    import copy
    step_args = copy.deepcopy(args)
    # USStockFutureSimulator 透過 params 讀 fear_threshold；
    # 這裡把它存進 step_args 讓 _run_one_step 能存取
    step_args.fear_threshold   = fear_thr
    step_args.fear_vol_mult    = fear_profile.get("fear_vol_mult", 1.3)
    step_args.fear_drift_decay = fear_profile.get("fear_drift_decay_rate", 0.6)

    # 決定 train_end_idx
    end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
    else:
        dates = pd.to_datetime(df.iloc[:, 0])
    mask = dates <= end_dt
    if not mask.any():
        raise ValueError(f"[{symbol}] 找不到 {end_dt.date()} 之前的資料")
    train_end_idx = int(mask.values.nonzero()[0][-1]) + 1

    actual_future_df = df.iloc[train_end_idx: train_end_idx + args.total_bars].copy()
    hist_close = df.iloc[
        max(0, train_end_idx - args.lookback): train_end_idx
    ]["Close"].values.astype(float)

    result = _roll(
        df=df,
        train_end_idx=train_end_idx,
        theta=theta,
        args=step_args,
        actual_future_df=actual_future_df if len(actual_future_df) > 0 else None,
        param_model_payload=param_model_payload,
    )

    end_date_str = end_dt.strftime("%Y-%m-%d")
    print_rolling_comparison(
        result=result,
        start_price=float(hist_close[-1]),
        symbol=symbol,
        step=args.step,
        label_suffix=label,
    )
    render_rolling(
        symbol=symbol,
        hist_close=hist_close,
        result=result,
        step=args.step,
        end_date_str=end_date_str,
        output_dir=str(output_dir),
        label_suffix=label,
    )

    # 整合統計
    summary = {
        "symbol":          symbol,
        "end_date":        end_date_str,
        "label":           label,
        "fear_threshold":  round(fear_thr, 4),
        "vol_regime":      regime,
        "param_model_on":  use_model and param_model_payload is not None,
        "round_details":   result["round_details"],
    }
    if result["actual_closes"] is not None:
        pred   = result["pred_closes"]
        actual = result["actual_closes"]
        prev_a = result["actual_prev_closes"]
        na     = min(len(pred), len(actual), len(prev_a))
        mae    = float(np.mean(np.abs(pred[:na] - actual[:na]) / (actual[:na] + 1e-8) * 100))
        p_rets = np.array([pred[i] - prev_a[i] for i in range(na) if prev_a[i] is not None])
        a_rets = np.array([actual[i] - prev_a[i] for i in range(na) if prev_a[i] is not None])
        dirs   = int(np.sum(np.sign(p_rets) == np.sign(a_rets)))
        summary["overall_mae_pct"]  = round(mae, 4)
        summary["dir_accuracy_pct"] = round(dirs / len(p_rets) * 100, 2) if len(p_rets) > 0 else None
        log.info(
            f"[{symbol}] [{label}]  MAE={mae:.2f}%  "
            f"Dir={dirs}/{len(p_rets)} ({dirs/len(p_rets)*100:.0f}%)"
        )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — 生成 HTML 報告
# ══════════════════════════════════════════════════════════════════════════════

def step7_report(all_results: list[dict], output_dir: Path):
    """輸出一份輕量 HTML 對比報告。"""
    report_path = output_dir / "pipeline_report.html"
    rows = ""
    for r in all_results:
        mae_base  = r.get("base",  {}).get("overall_mae_pct",  "N/A")
        mae_model = r.get("model", {}).get("overall_mae_pct",  "N/A")
        dir_base  = r.get("base",  {}).get("dir_accuracy_pct", "N/A")
        dir_model = r.get("model", {}).get("dir_accuracy_pct", "N/A")
        fear_thr  = r.get("model", r.get("base", {})).get("fear_threshold", "N/A")
        regime    = r.get("model", r.get("base", {})).get("vol_regime", "N/A")
        sym       = r.get("symbol", "?")
        row_color = ""
        if isinstance(mae_model, float) and isinstance(mae_base, float):
            row_color = 'style="background:#e8f5e9"' if mae_model < mae_base else 'style="background:#fce4ec"'
        rows += (
            f'<tr {row_color}>'
            f'<td><b>{sym}</b></td>'
            f'<td>{r.get("end_date","?")}</td>'
            f'<td>{_fmt(mae_base)}</td>'
            f'<td>{_fmt(mae_model)}</td>'
            f'<td>{_fmt(dir_base)}%</td>'
            f'<td>{_fmt(dir_model)}%</td>'
            f'<td>{_fmt(fear_thr, pct=True)}</td>'
            f'<td>{regime}</td>'
            f'</tr>\n'
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-TW"><head><meta charset="UTF-8">
<title>Pipeline Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#222}}
h1{{font-size:1.5rem;margin-bottom:.5rem}}p.sub{{color:#666;font-size:.9rem;margin-bottom:2rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px 12px;text-align:right}}
th{{background:#f5f5f5;text-align:center;font-size:.85rem}}
td:first-child,th:first-child{{text-align:left}}
.better{{color:#2e7d32;font-weight:600}}.worse{{color:#c62828;font-weight:600}}
</style></head><body>
<h1>🔬 Market Pipeline Report</h1>
<p class="sub">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
<table>
<thead><tr>
<th>Symbol</th><th>End Date</th>
<th>MAE% (base)</th><th>MAE% (model)</th>
<th>Dir (base)</th><th>Dir (model)</th>
<th>Fear Thr.</th><th>Vol Regime</th>
</tr></thead>
<tbody>
{rows}
</tbody></table>
<p style="margin-top:2rem;font-size:.8rem;color:#999">
Base = auto_calibrate only | Model = + param surrogate | Dir = direction accuracy
</p>
</body></html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    log.info(f"Step 7  報告 → {report_path}")
    return str(report_path)


def _fmt(v, pct: bool = False) -> str:
    if v == "N/A" or v is None:
        return "N/A"
    if pct and isinstance(v, float):
        return f"{v*100:.1f}%"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="一鍵執行完整市場模擬流程")

    # 基礎
    p.add_argument("--symbol",             nargs="+", required=True)
    p.add_argument("--end-date",           default=None, help="分析截止日（YYYY-MM-DD）")
    p.add_argument("--years",              type=int,   default=10, help="下載幾年資料")
    p.add_argument("--output-dir",         default="results/pipeline")
    p.add_argument("--cache-dir",          default="cache")

    # 流程控制
    p.add_argument("--skip-steps",         nargs="*", type=int, default=[],
                   help="跳過指定步驟編號（1-7）")
    p.add_argument("--stop-after",         type=int, default=7,
                   help="執行到第幾步後停止")
    p.add_argument("--force-refresh",      action="store_true",
                   help="忽略快取，全部重新計算")
    p.add_argument("--no-base-run",        action="store_true",
                   help="不跑無模型的 baseline 對比")

    # Fear threshold
    p.add_argument("--fear-pct-drop",      type=float, default=0.03)
    p.add_argument("--fear-min-gain",      type=float, default=0.02)
    p.add_argument("--fear-lookahead",     type=int,   default=10)
    p.add_argument("--fear-target-prob",   type=float, default=0.45)
    p.add_argument("--fear-rolling-window",type=int,   default=252)
    p.add_argument("--fear-recent-window", type=int,   default=63)
    p.add_argument("--fear-method",        default="median",
                   choices=["median", "ewm", "p25"])

    # Training data
    p.add_argument("--calib-window",       type=int,   default=500)
    p.add_argument("--train-step",         type=int,   default=5,
                   help="收集訓練資料時每幾根取一個樣本")
    p.add_argument("--n-cv-folds",         type=int,   default=3)

    # Rolling forward
    p.add_argument("--total-bars",         type=int,   default=30)
    p.add_argument("--step",               type=int,   default=5)
    p.add_argument("--lookback",           type=int,   default=120)
    p.add_argument("--n-paths",            type=int,   default=500)
    p.add_argument("--no-garch",           action="store_true")
    p.add_argument("--garch-model",        default="gjr-garch",
                   choices=["gjr-garch", "garch"])
    p.add_argument("--auto-calibrate",     action="store_true", default=True)
    p.add_argument("--short-drift-weight", type=float, default=0.4)
    p.add_argument("--sr-window",          type=int,   default=90)
    p.add_argument("--sr-bins",            type=int,   default=40)
    p.add_argument("--sr-top-k",           type=int,   default=5)
    p.add_argument("--sr-pivot-order",     type=int,   default=5)
    p.add_argument("--sr-zone-pct",        type=float, default=0.035)
    p.add_argument("--no-sr",              action="store_true")
    p.add_argument("--dynamic-drift",      action="store_true", default=True)
    p.add_argument("--no-dynamic-drift",   dest="dynamic_drift", action="store_false")
    p.add_argument("--mr-rate",            type=float, default=0.08)
    p.add_argument("--trend-strength",     type=float, default=0.5)
    p.add_argument("--vol-refit",          type=int,   default=5)
    p.add_argument("--early-decay-bars",   type=int,   default=5)
    p.add_argument("--reversal-window",    type=int,   default=3)
    p.add_argument("--verbose",            action="store_true")
    return p.parse_args()


def main():
    args       = parse_args()
    out_dir    = Path(args.output_dir)
    cache_dir  = Path(args.cache_dir)
    skip       = set(args.skip_steps)
    stop_after = args.stop_after
    force      = args.force_refresh

    _banner("Market Simulation Pipeline")
    log.info(f"Symbols : {args.symbol}")
    log.info(f"End date: {args.end_date or 'today'}")
    log.info(f"Steps   : 1→{stop_after}  skip={skip or 'none'}")
    log.info(f"Output  : {out_dir}")

    all_reports = []

    for symbol in args.symbol:
        sym_cache = cache_dir / symbol
        sym_out   = out_dir  / symbol
        sym_out.mkdir(parents=True, exist_ok=True)

        report_entry = {"symbol": symbol, "end_date": args.end_date or str(pd.Timestamp.today().date())}
        t0 = time.time()
        _banner(f"{symbol}")

        # ── Step 1 ────────────────────────────────────────────────────────────
        if 1 not in skip and stop_after >= 1:
            df = step1_fetch(
                symbol, args.years, sym_cache, args.end_date, force=force
            )
        else:
            df = pd.read_parquet(sym_cache / f"{symbol}_ohlcv.parquet")
        if stop_after < 2:
            continue

        # ── Step 2 ────────────────────────────────────────────────────────────
        if 2 not in skip:
            theta_dict = step2_calibrate(
                symbol, df, sym_cache, calib_bars=args.calib_window, force=force
            )
        else:
            theta_dict = _load_json(sym_cache / f"{symbol}_theta.json")
        if stop_after < 3:
            continue

        # ── Step 3 ────────────────────────────────────────────────────────────
        if 3 not in skip:
            fear_profile = step3_fear_threshold(
                symbol, df, sym_cache,
                pct_drop=args.fear_pct_drop,
                min_gain=args.fear_min_gain,
                lookahead=args.fear_lookahead,
                rolling_window=args.fear_rolling_window,
                target_peak_prob=args.fear_target_prob,
                recent_window=args.fear_recent_window,
                threshold_method=args.fear_method,
                force=force,
            )
        else:
            fear_profile = _load_json(sym_cache / f"{symbol}_fear_profile.json")
        if stop_after < 4:
            continue

        # ── Step 4 ────────────────────────────────────────────────────────────
        if 4 not in skip:
            train_df = step4_collect_training_data(
                symbol, df, theta_dict, sym_cache,
                calib_window=args.calib_window,
                step_bars=args.train_step,
                force=force,
            )
        else:
            train_df = pd.read_parquet(sym_cache / f"{symbol}_training_data.parquet")
        if stop_after < 5:
            continue

        # ── Step 5 ────────────────────────────────────────────────────────────
        if 5 not in skip:
            model_path = step5_train_surrogate(
                symbol, train_df, sym_cache,
                n_cv_folds=args.n_cv_folds,
                force=force,
            )
        else:
            model_path = sym_cache / f"param_model_{symbol}.joblib"
        if stop_after < 6:
            continue

        # ── Step 6 ────────────────────────────────────────────────────────────
        if 6 not in skip:
            # 6b: with param model
            summary_model = step6_rolling_forward(
                symbol=symbol, df=df, theta_dict=theta_dict,
                fear_profile=fear_profile, model_path=model_path,
                end_date=args.end_date, args=args,
                output_dir=sym_out, use_model=True,
            )
            report_entry["model"] = summary_model

            # 6a: baseline（無 param model）
            if not args.no_base_run:
                summary_base = step6_rolling_forward(
                    symbol=symbol, df=df, theta_dict=theta_dict,
                    fear_profile=fear_profile, model_path=model_path,
                    end_date=args.end_date, args=args,
                    output_dir=sym_out, use_model=False,
                )
                report_entry["base"] = summary_base

            # 儲存單支股票 JSON
            _save_json(sym_out / f"{symbol}_pipeline_result.json", report_entry)
        if stop_after < 7:
            continue

        all_reports.append(report_entry)
        elapsed = time.time() - t0
        log.info(f"[{symbol}] ✔ 完成  耗時 {elapsed:.0f}s")

    # ── Step 7 ────────────────────────────────────────────────────────────────
    if 7 not in skip and stop_after >= 7 and all_reports:
        report_path = step7_report(all_reports, out_dir)
        _banner(f"完成  →  {report_path}")

    # 全部結果彙整 JSON
    if all_reports:
        _save_json(out_dir / "pipeline_summary.json", all_reports)
        log.info(f"彙整 JSON → {out_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
