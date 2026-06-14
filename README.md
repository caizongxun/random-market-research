# random-market-research

隨機市場模擬研究：探討隨機價格路徑中的型態湧現、支撐阻力現象，以及從歷史 K 棒估計參數後去模擬未來路徑。

## 目前功能

### 1. 純隨機與多代理市場模擬

- `src/market_simulator.py`
- `scripts/run_simulation.py`

支援：
- GBM
- OrderBook
- MultiAgent（散戶 / 大戶 / 做市商）

### 2. 從真實美股資料估計參數再模擬未來

- `src/market_estimator.py`
- `src/us_equity_simulator.py`
- `scripts/run_us_stock_experiment.py`

資料來源：免費 Yahoo Finance（透過 `yfinance`）

估計參數包括：
- realized volatility
- drift
- hurst proxy
- avg intraday range ratio
- gap std
- volume profile nodes
- trend strength / mean reversion strength
- smart money ratio（代理估計）

## 安裝

```bash
pip install -r requirements.txt
```

## 執行

### 隨機市場模擬

```bash
python scripts/run_simulation.py --mode multiagent --steps 100000 --seed 42 --no-show --output-dir results/
```

### 美股真實資料 → 估參數 → 模擬未來

```bash
python scripts/run_us_stock_experiment.py --symbol AAPL --period 3y --interval 1d --lookback 500 --forecast 30 --paths 1000 --seed 42 --output-dir results_aapl
```

## 輸出

`run_us_stock_experiment.py` 會輸出：
- `estimated_params.json`
- `simulation_overview.png`

## 注意

這個專案目前是研究用途，不是交易建議。核心用途是：
- 把「像市場」的隨機過程做出來
- 嘗試從歷史 K 棒反推隱含參數
- 觀察用這些參數去隨機演化未來路徑時，會得到什麼樣的機率分佈
