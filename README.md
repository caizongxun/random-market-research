# random-market-research

隨機市場模擬研究：探討隨機價格路徑中的型態湧現與支撐阻力現象。

## 核心問題

> 為什麼純隨機生成的 K 線數據，看起來和真實市場走勢幾乎一樣？
> 為什麼隨機路徑天然會形成「支撐阻力」和「圖形型態」？

## 結構

```
├── notebooks/
│   └── 01_market_simulator.ipynb   # 完整模擬實驗（Jupyter）
├── src/
│   ├── market_simulator.py         # 訂單簿模擬器核心
│   ├── pattern_visualizer.py       # 型態自動偵測視覺化
│   └── utils.py                    # 通用工具
├── scripts/
│   └── run_simulation.py           # 直接執行腳本（非 Jupyter）
└── requirements.txt
```

## 快速開始

```bash
pip install -r requirements.txt
python scripts/run_simulation.py
```

或開啟 Jupyter：

```bash
jupyter notebook notebooks/01_market_simulator.ipynb
```

## 研究方向

- [ ] GBM vs. Order-book 模擬比較
- [ ] Volume Profile 支撐阻力有效性檢驗
- [ ] 型態出現頻率 vs. 隨機基準比較
- [ ] Regime 偵測（趨勢 / 震盪 / 反轉）
- [ ] Bootstrap 顯著性測試
