"""
train_param_model.py

讀取 collect_training_data.py 產出的 CSV，
訓練 XGBoost 迴歸模型（分開訓練三個目標），
儲存為 joblib 模型檔。

同時輸出特徵重要性報告與 CV 誤差。

用法：
  python scripts/train_param_model.py \\
    --input  results/training_data_AAPL.csv \\
    --output models/param_model_AAPL.joblib

之後在 forward_study.py 加 --param-model models/param_model_AAPL.joblib
即可自動套用模型預測的參數。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="training_data CSV 路徑")
    p.add_argument("--output", required=True, help="輸出 .joblib 模型路徑")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--feature-report", action="store_true",
                   help="印出特徵重要性")
    return p.parse_args()


TARGETS = ["target_drift_scale", "target_momentum_boost", "target_drift_decay"]


def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("feat_")]


def train_models(df, feature_cols, cv_folds):
    try:
        import xgboost as xgb
        MODEL_NAME = "XGBoost"
        def make_model():
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        MODEL_NAME = "RandomForest"
        def make_model():
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )

    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = df[feature_cols].fillna(0).values
    results = {}

    print(f"\n使用模型: {MODEL_NAME}")
    print(f"樣本數: {len(X)}  特徵數: {len(feature_cols)}  CV folds: {cv_folds}\n")

    models = {}
    for target in TARGETS:
        y = df[target].values
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  make_model()),
        ])
        scores = cross_val_score(pipe, X, y,
                                 cv=cv_folds, scoring="neg_mean_absolute_error")
        mae_cv = -scores.mean()
        print(f"  [{target}]  CV MAE = {mae_cv:.4f}  (std={scores.std():.4f})")

        # 全量訓練
        pipe.fit(X, y)
        models[target] = pipe
        results[target] = {"cv_mae": round(float(mae_cv), 4),
                           "cv_std": round(float(scores.std()), 4)}

    return models, results, MODEL_NAME


def feature_importance_report(models, feature_cols):
    print("\n[特徵重要性 Top-10 per target]")
    for target, pipe in models.items():
        model = pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            idxs = np.argsort(imp)[::-1][:10]
            print(f"\n  {target}:")
            for rank, i in enumerate(idxs, 1):
                print(f"    {rank:2d}. {feature_cols[i]:35s} {imp[i]:.4f}")


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    print(f"讀取 {len(df)} 筆資料 from {args.input}")

    feature_cols = get_feature_cols(df)
    print(f"特徵欄位數: {len(feature_cols)}")

    for t in TARGETS:
        if t not in df.columns:
            raise ValueError(f"CSV 缺少目標欄位 {t}")

    models, cv_results, model_name = train_models(df, feature_cols, args.cv_folds)

    if args.feature_report:
        feature_importance_report(models, feature_cols)

    # 儲存模型
    try:
        import joblib
    except ImportError:
        raise ImportError("請先安裝 joblib: pip install joblib")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "models":       models,
        "feature_cols": feature_cols,
        "targets":      TARGETS,
        "model_name":   model_name,
        "cv_results":   cv_results,
    }
    joblib.dump(payload, out_path)
    print(f"\n✔ 模型已儲存 → {out_path}")

    # 同步儲存 metadata JSON（方便查看，不需要載入 joblib）
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_name":   model_name,
            "n_samples":    len(df),
            "n_features":   len(feature_cols),
            "feature_cols": feature_cols,
            "targets":      TARGETS,
            "cv_results":   cv_results,
        }, f, indent=2)
    print(f"   Metadata → {meta_path}")

    print("\n[CV 結果摘要]")
    for t, r in cv_results.items():
        print(f"  {t:35s}: MAE={r['cv_mae']:.4f}  std={r['cv_std']:.4f}")


if __name__ == "__main__":
    main()
