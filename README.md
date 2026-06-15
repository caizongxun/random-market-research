# random-market-research

> **隨機過程 × 股價模擬 × 滾動預測**
>
> 本專案以 Ornstein–Uhlenbeck / GJR-GARCH 混合機制模擬美股日線走勢，
> 並透過滾動 Forward Test 自動評估每輪預測誤差。
> 代理模型（GBM Surrogate）可取代逐次 grid search，大幅提速。

---

## 目錄結構

```
random-market-research/
├── src/                         核心模組
│   ├── backbone_fitter.py       骨幹曲線擬合（分段趨勢）
│   ├── calibrated_simulator.py  CalibratedTheta 資料類別 + build_params_from_theta
│   ├── market_estimator.py      從 OHLCV 估計 OU/VP 參數
│   ├── market_simulator.py      基礎 Monte Carlo 引擎
│   ├── us_equity_simulator.py   美股專用模擬器（drift schedule / SR / GARCH）
│   ├── param_calibrator.py      theta 自動校準邏輯
│   ├── candle_renderer.py       K 棒繪製工具
│   ├── pattern_visualizer.py    型態視覺化
│   └── utils.py                 共用工具
│
├── scripts/                     執行腳本
│   ├── run_pipeline.py          ★ 一鍵端到端流程（主入口）
│   ├── calibrate_params.py      步驟 2：擬合 theta → JSON
│   ├── collect_training_data.py 步驟 4：walk-forward grid search → CSV
│   ├── train_param_model.py     步驟 5：用 CSV 訓練 GBM 代理模型 → joblib
│   ├── rolling_forward.py       步驟 6：滾動 Forward Test（支援 param-model 覆蓋）
│   ├── forward_study.py         單次預測研究 + auto_calibrate 邏輯
│   ├── plot_path_vs_actual.py   路徑 vs 實際 K 棒對比圖
│   └── analyze_fear_threshold.py 動態 fear_threshold 分析
│
├── cache/                       中間結果快取（git-ignored）
│   ├── {SYM}_ohlcv.parquet
│   ├── {SYM}_theta.json
│   ├── {SYM}_fear_profile.json
│   └── {SYM}_training_data.csv
│
├── models/                      訓練好的代理模型（git-ignored）
│   └── param_model_{SYM}.joblib
│
├── results/                     輸出圖表 / JSON / HTML
│   └── pipeline/
│       └── pipeline_report.html
│
└── requirements.txt
```

---

## 快速開始

```bash
pip install -r requirements.txt

# 一鍵跑完所有步驟（AAPL，預測 30 根）
python scripts/run_pipeline.py --symbol AAPL

# 多股 + 指定回測截止日
python scripts/run_pipeline.py \
  --symbol AAPL MSFT NVDA \
  --end-date 2025-06-01 \
  --total-bars 30 --step 5

# 只跑到代理模型訓練（Steps 1-5），不做 rolling forward
python scripts/run_pipeline.py --symbol AAPL --stop-after 5

# 強制重算（忽略快取）
python scripts/run_pipeline.py --symbol AAPL --force-refresh

# 跳過前幾步（已完成）
python scripts/run_pipeline.py --symbol AAPL --skip-steps 1 2 3
```

---

## 七步流程說明

| Step | 腳本 | 輸入 | 輸出 | 說明 |
|------|------|------|------|------|
| 1 | `run_pipeline.py` 內嵌 | — | `cache/{SYM}_ohlcv.parquet` | yfinance 下載 5 年 OHLCV |
| 2 | `calibrate_params.py` | OHLCV | `cache/{SYM}_theta.json` | 擬合 OU + GARCH 參數（theta） |
| 3 | `analyze_fear_threshold.py` | OHLCV + theta | `cache/{SYM}_fear_profile.json` | 動態 fear_threshold 分析（可選） |
| 4 | `collect_training_data.py` | OHLCV + theta | `cache/{SYM}_training_data.csv` | Walk-forward grid search → (特徵, 最佳參數) 對 |
| 5 | `train_param_model.py` | CSV | `models/param_model_{SYM}.joblib` | GBM 代理模型訓練 |
| 6a | `rolling_forward.py` | OHLCV + theta | `results/pipeline/...json/.png` | Baseline rolling forward |
| 6b | `rolling_forward.py` | OHLCV + theta + model | `results/pipeline/..._param-model...` | + 代理模型覆蓋 drift 參數 |
| 7 | `run_pipeline.py` 內嵌 | JSON x2 | `results/pipeline/pipeline_report.html` | Baseline vs Model 對比報告 |

---

## 核心模型架構

### theta（校準參數集）

```python
@dataclass
class CalibratedTheta:
    vol:               float   # 歷史日波動率
    mr_coeff:          float   # 均值回歸係數
    node_coeff:        float   # 節點（支撐/阻力）牽引力
    momentum_strength: float   # 動量強度
    momentum_decay:    float   # 動量衰減率
    breakout_boost:    float   # 突破加速係數
```

### auto_calibrate 動態調整（每輪 rolling forward 前執行）

每輪會根據最近 500 根計算：

- **drift_scale**：骨幹趨勢放大倍數（grid search 最佳或代理模型預測）
- **drift_decay**：趨勢衰減率（同上）
- **vol_multiplier**：RV / theta.vol，動態調整波動率
- **momentum_boost**：固定 0.8
- **intra_bar**：bar 內模擬步數（GARCH 預測高波動時增加）

### 代理模型（Surrogate GBM）

**輸入特徵**（18+ 維）：

| 特徵名 | 說明 |
|--------|------|
| `vol_20 / vol_60 / vol_all` | 短中長期對數收益率標準差（×100） |
| `vol_ratio_20_60` | 短/中期波動率比 |
| `vol_ratio_rv_theta` | 實現波動率 / theta.vol |
| `drift_20 / drift_60 / drift_all` | 短中長期平均日漂移（×100） |
| `drift_ratio_20_all` | 短/全期漂移比（有方向性） |
| `ret_autocorr` | 收益率自相關（lag=1） |
| `vol_autocorr` | 成交量變化自相關（lag=1） |
| `avg_body_pct` | 平均實體 / open 百分比 |
| `median_sr / p90_sr` | 影線比中位數 / 90百分位 |
| `bb_last_drift / bb_last_vol` | 骨幹最後一段趨勢 / 波動率 |
| `bb_drift_std / bb_vol_std` | 骨幹各段趨勢 / 波動率標準差 |
| `garch_alpha/beta/gamma/persistence/forecast_vol` | GJR-GARCH 擬合參數（若可用） |

**輸出目標**：`drift_scale`、`drift_decay`

---

## 方向命中定義（v2）

```
P-Ret_t = pred_close_t  - actual_close_{t-1}
A-Ret_t = actual_close_t - actual_close_{t-1}
Dir 命中 = (sign(P-Ret) == sign(A-Ret))
```

兩者基準一致（前一根的**實際**收盤），避免雙重誤差累積。

---

## 開發狀態與待辦

### 已完成
- [x] OU + GJR-GARCH 混合模擬引擎
- [x] BackboneFitter 分段趨勢擬合
- [x] auto_calibrate（drift_scale / drift_decay / vol_multiplier）
- [x] 支撐/阻力影響的漂移調整（SR drift adjustment）
- [x] 滾動 Forward Test + 方向命中統計（v2 定義）
- [x] Walk-forward grid search → 訓練資料 CSV
- [x] GBM 代理模型訓練腳本（train_param_model.py）
- [x] 一鍵流程腳本（run_pipeline.py）+ 快取機制
- [x] Baseline vs Model HTML 對比報告

### 待辦 / 可探索
- [ ] 代理模型換成 LightGBM / XGBoost（速度提升）
- [ ] 特徵加入 VIX / 市場情緒指標
- [ ] 多時框特徵（週線 / 月線）
- [ ] 模型線上更新（incremental fit）
- [ ] Optuna 超參數搜尋整合
- [ ] Web Dashboard（Streamlit / Dash）

---

## 依賴套件

```
yfinance
numpy
pandas
matplotlib
scikit-learn
joblib
arch          # GJR-GARCH（可選但建議安裝）
pyarrow       # parquet 快取
```

```bash
pip install -r requirements.txt
```

---

## 給下一位 AI 的交接說明

### 最重要的入口

```bash
python scripts/run_pipeline.py --symbol AAPL --end-date 2025-06-01
```

這一行會依序執行全部 7 步，快取有效時自動跳過，最終產生 `results/pipeline/pipeline_report.html`。

### 已知問題 / 注意事項

1. **特徵命名對齊**：`collect_training_data.py` 的 CSV 欄位是 `feat_xxx`（有前綴），`rolling_forward.py` 的 `apply_param_model` 用 `fc.replace("feat_", "")` 轉換後查 `feats` dict，dict key 是不帶前綴的 `xxx`。這是**正確的設計**，不需修改。

2. **theta 依賴**：`calibrate_params.py` 需要能找到 `src/` 下的模組。執行時請確保在 repo 根目錄或已設定 `PYTHONPATH`。

3. **GARCH 可選**：`arch` 套件不是硬性依賴，缺少時會自動降級使用 rolling std 估計波動率。

4. **快取目錄**：`cache/` 和 `models/` 不在 git 追蹤中，部署新機器需重跑 Steps 1-5。

5. **Step 6 輸出 JSON 路徑**：`rolling_forward.py` 產生的 JSON 名稱格式為 `{SYM}_{end_date}_rolling{step}[_param-model].json`，`run_pipeline.py` 的 Step 7 用這個路徑讀取結果，若 end_date 格式不一致會找不到檔案。

### 核心資料流

```
yfinance → OHLCV (parquet)
         → calibrate_params → theta.json
         → collect_training_data (grid search) → training_data.csv
                                              → train_param_model → param_model.joblib
                                                                  ↓
OHLCV + theta → rolling_forward (baseline)    → results JSON/PNG
OHLCV + theta → rolling_forward (+ model)     → results JSON/PNG
                                                  ↓
                                             pipeline_report.html
```
