"""
market_simulator.py

訂單簿模擬器：透過隨機撮合邏輯模擬價格序列。
核心設計參考：
  - 每步隨機產生 bid/ask 壓力，驅動價格變動
  - 支援 GBM 模式（純隨機遊走）與 OrderBook 模式
  - 輸出 tick 級別的 GLOBAL_PRICE_DATA
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SimConfig:
    start_price: float = 100.0
    steps: int = 100_000
    tick_size: float = 0.01
    # GBM 參數
    mu: float = 0.0          # 漂移項（年化，這裡設 0 為純隨機）
    sigma: float = 0.002     # 每步波動率
    # OrderBook 參數
    depth: int = 5           # 單邊掛單層數
    order_volume_mean: float = 100.0
    order_volume_std: float = 30.0
    # 市場衝擊（模擬大單）
    shock_prob: float = 0.001  # 每步發生大衝擊的機率
    shock_magnitude: float = 0.015  # 衝擊幅度（相對於當前價格）
    seed: int | None = None


class MarketSimulator:
    """
    隨機市場模擬器。

    支援兩種模式：
      - 'gbm'        : Geometric Brownian Motion（幾何布朗運動）
      - 'orderbook'  : 簡化訂單簿模型（每步模擬 bid/ask 壓力）

    兩種模式都能產出視覺上與真實 K 線高度相似的價格序列，
    這正是本研究的核心命題。
    """

    def __init__(
        self,
        start_price: float = 100.0,
        mode: Literal['gbm', 'orderbook'] = 'orderbook',
        config: SimConfig | None = None,
    ):
        self.config = config or SimConfig(start_price=start_price)
        self.mode = mode
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def run(self, steps: int | None = None) -> np.ndarray:
        """執行模擬，回傳 tick 價格序列 (1D ndarray)。"""
        n = steps or self.config.steps
        if self.mode == 'gbm':
            return self._run_gbm(n)
        return self._run_orderbook(n)

    def to_ohlcv(
        self,
        price_data: np.ndarray,
        bar_size: int = 100,
    ) -> 'pd.DataFrame':
        """
        將 tick 序列聚合成 OHLCV K 棒。

        Parameters
        ----------
        price_data : 1D ndarray，tick 價格序列
        bar_size   : 每根 K 棒包含的 tick 數

        Returns
        -------
        pd.DataFrame，欄位：open, high, low, close, volume
        """
        import pandas as pd

        n_bars = len(price_data) // bar_size
        trimmed = price_data[:n_bars * bar_size].reshape(n_bars, bar_size)
        volume = self._rng.integers(100, 10_000, size=n_bars)

        df = pd.DataFrame({
            'open':   trimmed[:, 0],
            'high':   trimmed.max(axis=1),
            'low':    trimmed.min(axis=1),
            'close':  trimmed[:, -1],
            'volume': volume,
        })
        df.index = pd.RangeIndex(len(df))
        return df

    # ------------------------------------------------------------------
    # 私有：GBM
    # ------------------------------------------------------------------

    def _run_gbm(self, n: int) -> np.ndarray:
        """幾何布朗運動：P(t) = P(t-1) * exp(mu*dt + sigma*dW)"""
        cfg = self.config
        dt = 1.0
        shocks = self._rng.standard_normal(n)
        log_returns = (cfg.mu - 0.5 * cfg.sigma ** 2) * dt + cfg.sigma * np.sqrt(dt) * shocks
        prices = cfg.start_price * np.exp(np.cumsum(log_returns))
        # 插入大衝擊
        prices = self._apply_shocks(prices)
        return prices

    # ------------------------------------------------------------------
    # 私有：OrderBook
    # ------------------------------------------------------------------

    def _run_orderbook(self, n: int) -> np.ndarray:
        """
        簡化訂單簿模型：
          1. 每步模擬 bid 與 ask 兩側的掛單量
          2. 隨機選擇市價單方向（買 or 賣）
          3. 吃單後計算剩餘不平衡，驅動價格移動
          4. 加入均值回歸微項，防止價格無限漂離
        """
        cfg = self.config
        prices = np.empty(n)
        price = cfg.start_price

        for i in range(n):
            # 雙邊掛單量（截斷正態）
            bid_vol = max(1.0, self._rng.normal(cfg.order_volume_mean, cfg.order_volume_std))
            ask_vol = max(1.0, self._rng.normal(cfg.order_volume_mean, cfg.order_volume_std))

            # 市場方向（1 = 買方主動，-1 = 賣方主動）
            direction = 1 if self._rng.random() < 0.5 else -1

            # 不平衡驅動價格
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
            price_change = direction * cfg.sigma * price * (1 + 0.5 * imbalance)

            # 均值回歸微項（讓路徑更像真實市場，而非無邊際漂移）
            mean_reversion = -0.0001 * (price - cfg.start_price)

            price = max(cfg.tick_size, price + price_change + mean_reversion)
            prices[i] = price

        # 插入大衝擊
        prices = self._apply_shocks(prices)
        return prices

    def _apply_shocks(self, prices: np.ndarray) -> np.ndarray:
        """在隨機時間點插入大幅衝擊，模擬黑天鵝事件。"""
        cfg = self.config
        n = len(prices)
        shock_mask = self._rng.random(n) < cfg.shock_prob
        shock_dirs = self._rng.choice([-1, 1], size=n)
        shock_magnitudes = shock_mask * shock_dirs * cfg.shock_magnitude

        for i in range(1, n):
            if shock_mask[i]:
                prices[i] *= (1 + shock_magnitudes[i])
        return prices
