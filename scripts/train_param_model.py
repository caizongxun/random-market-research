"""
train_param_model.py

用 collect_training_data.py 產生的 CSV 訓練代理模型，
預測 (drift_scale, drift_decay)，替代每次的 grid search 擬合。

優先使用 LightGBM（更快、通常更準）；
若未安裝則自動 fallback 到 sklearn GradientBoostingRegressor。

用法：
  python scripts/train_param_model.py \\
    --csv cache/AAPL_training_data.csv \\
    --output models/param_model_AAPL.joblib \\
    --symbol AAPL

輸出 joblib 格式：
  payload = {
    "model_name":   "LightGBM" | "GradientBoosting",
    "symbol":       "AAPL",
    "feature_cols": [...],
    "models": {
        "target_drift_scale":  <fitted model>,
        "target_drift_decay":  <fitted model>,
    },
    "cv_scores": { ... },
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
    p.add_argument("--csv",           required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--symbol",        default="")
    p.add_argument("--n-estimators",  type=int,   default=500)
    p.add_argument("--max-depth",     type=int,   default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--cv-folds",      type=int,   default=5)
    p.add_argument("--min-rows",      type=int,   default=20)
    p.add_argument("--early-stopping",type=int,   default=40,
                   help="LightGBM early stopping rounds（0 = 關閉）")
    p.add_argument("--force-gbm",     action="store_true",
                   help="強制使用 sklearn GBM（跳過 LightGBM）")
    return p.parse_args()


def _build_lgbm(n_estimators, max_depth, learning_rate):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=max(31, 2 ** max_depth - 1),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def _build_gbm(n_estimators, max_depth, learning_rate):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        ))
    ])


def _get_feature_importances(model, model_name: str) -> np.ndarray | None:
    if model_name == "LightGBM":
        return model.feature_importances_
    elif hasattr(model, "named_steps"):
        return model.named_steps["gbm"].feature_importances_
    return None


def main():
    args = parse_args()

    import joblib
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error

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

    bad_feats = [c for c in feature_cols if df[c].isnull().all()]
    if bad_feats:
        print(f"  [WARN] 全 NaN 特徵已移除：{bad_feats}")
        feature_cols = [c for c in feature_cols if c not in bad_feats]

    X = df[feature_cols].fillna(0.0).values
    print(f"特徵數：{len(feature_cols)}   樣本數：{len(X)}")

    # ── 選擇後端 ────────────────────────────────────────────
    model_name = "GradientBoosting"
    use_lgbm   = False
    if not args.force_gbm:
        try:
            import lightgbm  # noqa: F401
            use_lgbm   = True
            model_name = "LightGBM"
            print("  [backend] LightGBM")
        except ImportError:
            print("  [backend] LightGBM 未安裝，fallback 到 sklearn GBM")
            print("            （pip install lightgbm 可啟用更快版本）")
    else:
        print("  [backend] sklearn GradientBoosting（--force-gbm）")

    # ── 訓練 ────────────────────────────────────────────────
    trained_models = {}
    cv_scores      = {}
    kf = KFold(n_splits=min(args.cv_folds, len(X)), shuffle=True, random_state=42)

    for tgt in targets:
        y = df[tgt].values.astype(float)

        # CV
        fold_maes = []
        for train_idx, val_idx in kf.split(X):
            if use_lgbm:
                m = _build_lgbm(args.n_estimators, args.max_depth, args.learning_rate)
                if args.early_stopping > 0:
                    m.fit(
                        X[train_idx], y[train_idx],
                        eval_set=[(X[val_idx], y[val_idx])],
                        callbacks=[
                            __import__("lightgbm").early_stopping(
                                args.early_stopping, verbose=False
                            ),
                            __import__("lightgbm").log_evaluation(period=-1),
                        ],
                    )
                else:
                    m.fit(X[train_idx], y[train_idx])
            else:
                m = _build_gbm(args.n_estimators, args.max_depth, args.learning_rate)
                m.fit(X[train_idx], y[train_idx])
            fold_maes.append(mean_absolute_error(y[val_idx], m.predict(X[val_idx])))

        cv_mean = float(np.mean(fold_maes))
        cv_std  = float(np.std(fold_maes))
        cv_scores[tgt] = {"mae_mean": round(cv_mean, 4), "mae_std": round(cv_std, 4)}
        print(f"  {tgt:25s}  CV MAE = {cv_mean:.4f} ± {cv_std:.4f}")

        # 全量訓練
        if use_lgbm:
            final_model = _build_lgbm(args.n_estimators, args.max_depth, args.learning_rate)
            final_model.fit(X, y)
        else:
            final_model = _build_gbm(args.n_estimators, args.max_depth, args.learning_rate)
            final_model.fit(X, y)
        trained_models[tgt] = final_model

    # ── 特徵重要性 ──────────────────────────────────────────
    print(f"\n特徵重要性 [target_drift_scale] ({model_name}):")
    imp = _get_feature_importances(trained_models["target_drift_scale"], model_name)
    if imp is not None:
        top_idx = np.argsort(imp)[::-1][:10]
        for i in top_idx:
            if imp[i] > 0:
                print(f"  {feature_cols[i]:40s}  {imp[i]:.4f}")

    # ── 儲存 ────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name":   model_name,
        "symbol":       args.symbol,
        "feature_cols": feature_cols,
        "models":       trained_models,
        "cv_scores":    cv_scores,
        "train_rows":   len(df),
    }
    joblib.dump(payload, out_path)
    print(f"\n✔ 模型已儲存 → {out_path}  [{model_name}]")
    print(f"  CV MAE: drift_scale={cv_scores['target_drift_scale']['mae_mean']:.4f}  "
          f"drift_decay={cv_scores['target_drift_decay']['mae_mean']:.4f}")


if __name__ == "__main__":
    main()
