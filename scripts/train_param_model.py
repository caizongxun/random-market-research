"""
train_param_model.py

用 collect_training_data.py 產生的 CSV 訓練 GBM 代理模型，
預測 (drift_scale, drift_decay)，替代每次的 grid search 擬合。

用法：
  python scripts/train_param_model.py \\
    --csv results/training_data_AAPL.csv \\
    --output models/param_model_AAPL.joblib \\
    --symbol AAPL

輸出 joblib 格式，包含：
  payload = {
    "model_name":   "GradientBoosting",
    "symbol":       "AAPL",
    "feature_cols": [...],       # feat_xxx 欄位名稱
    "models": {
        "target_drift_scale":  <fitted model>,
        "target_drift_decay":  <fitted model>,
    },
    "cv_scores": { ... },        # 5-fold CV MAE
    "train_rows": N,
  }
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    required=True, help="collect_training_data 輸出的 CSV")
    p.add_argument("--output", required=True, help="joblib 輸出路徑，如 models/param_model_AAPL.joblib")
    p.add_argument("--symbol", default="",    help="股票代號（僅記錄用）")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth",    type=int, default=4)
    p.add_argument("--learning-rate",type=float, default=0.05)
    p.add_argument("--cv-folds",     type=int, default=5)
    p.add_argument("--min-rows",     type=int, default=20,
                   help="最少需要幾筆資料才能訓練")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import joblib
    except ImportError:
        raise ImportError("需要 joblib：pip install joblib")

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError:
        raise ImportError("需要 scikit-learn：pip install scikit-learn")

    # ── 讀資料 ──────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"讀入 {len(df)} 筆訓練資料  ({args.csv})")

    if len(df) < args.min_rows:
        raise ValueError(
            f"資料不足 {args.min_rows} 筆（現有 {len(df)} 筆），"
            f"請先執行 collect_training_data.py 收集更多資料。"
        )

    # ── 特徵與目標 ──────────────────────────────────────────
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    targets      = ["target_drift_scale", "target_drift_decay"]

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"CSV 缺少目標欄位：{missing_targets}")

    missing_feats = [c for c in feature_cols if df[c].isnull().all()]
    if missing_feats:
        print(f"⚠ 以下特徵全為 NaN，已移除：{missing_feats}")
        feature_cols = [c for c in feature_cols if c not in missing_feats]

    X = df[feature_cols].fillna(0.0).values
    print(f"特徵數：{len(feature_cols)}   樣本數：{len(X)}")

    # ── 訓練 ────────────────────────────────────────────────
    trained_models = {}
    cv_scores      = {}

    for tgt in targets:
        y = df[tgt].values.astype(float)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm", GradientBoostingRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=0.8,
                random_state=42,
            ))
        ])

        # CV MAE
        cv_mae = -cross_val_score(
            model, X, y,
            cv=min(args.cv_folds, len(X)),
            scoring="neg_mean_absolute_error",
        )
        cv_mean = float(np.mean(cv_mae))
        cv_std  = float(np.std(cv_mae))
        cv_scores[tgt] = {"mae_mean": round(cv_mean, 4), "mae_std": round(cv_std, 4)}
        print(f"  {tgt}  CV MAE = {cv_mean:.4f} ± {cv_std:.4f}")

        model.fit(X, y)
        trained_models[tgt] = model

    # 特徵重要性（GBM 步驟）
    print("\n特徵重要性 (target_drift_scale):")
    gbm_ds = trained_models["target_drift_scale"].named_steps["gbm"]
    imp    = gbm_ds.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    for i in top_idx:
        if imp[i] > 0.005:
            print(f"  {feature_cols[i]:35s}  {imp[i]:.4f}")

    # ── 儲存 ────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name":   "GradientBoosting",
        "symbol":       args.symbol,
        "feature_cols": feature_cols,
        "models":       trained_models,
        "cv_scores":    cv_scores,
        "train_rows":   len(df),
    }
    joblib.dump(payload, out_path)
    print(f"\n✔ 模型已儲存 → {out_path}")
    print(f"  CV MAE: drift_scale={cv_scores['target_drift_scale']['mae_mean']:.4f}  "
          f"drift_decay={cv_scores['target_drift_decay']['mae_mean']:.4f}")


if __name__ == "__main__":
    main()
