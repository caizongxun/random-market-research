"""
market_simulator.py

多角色市場模擬器：
  - GBM           : 純幾何布朗運動（基準）
  - OrderBook     : 簡化訂單簿（原版）
  - MultiAgent    : 散戶 / 大戶 / 做市商 三方對立模型

核心參數 smart_money_ratio (0~1):
  0.0 = 市場全為散戶，震盪劇烈
  1.0 = 市場全為大戶，價格高度平滑、mean-reversion 強
  0.3 = 較接近現實（大戶佔少數但影響力大）
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class SimConfig:
    start_price: float = 100.0
    steps: int = 100_000
    tick_size: float = 0.01

    # ── GBM 參數 ──────────────────────────────────────────────────────
    mu: float = 0.0        # 漂移項（設 0 = 純隨機）
    sigma: float = 0.0004  # 每步波動率（降低至 0.0004，比原版 0.002 平滑 5x）

    # ── OrderBook 參數 ────────────────────────────────────────────────
    order_volume_mean: float = 100.0
    order_volume_std: float = 30.0

    # ── MultiAgent 參數 ───────────────────────────────────────────────
    # 散戶（Noise Trader）
    noise_sigma: float = 0.0003      # 散戶每步隨機衝擊大小
    noise_momentum: float = 0.15     # 散戶跟風慣性（0=無記憶, 1=完全跟前一步）

    # 大戶（Smart Money）
    smart_money_ratio: float = 0.3   # 大戶佔成交量比例（0~1）
    smart_sigma: float = 0.0008      # 大戶單筆衝擊大小（比散戶大）
    smart_threshold: float = 0.005   # 偏離均值多少才觸發大戶反向操作
    smart_lookback: int = 500        # 大戶計算「均值」的回望窗口（tick 數）

    # 做市商（Market Maker）
    mm_spread: float = 0.0001        # bid/ask spread（相對價格）
    mm_inventory_limit: float = 200  # 做市商庫存上限，超過後強制平倉

    # ── 衝擊事件（黑天鵝）────────────────────────────────────────────
    shock_prob: float = 0.0005       # 每步發生大衝擊的機率
    shock_magnitude: float = 0.012   # 衝擊幅度

    seed: int | None = None


class MarketSimulator:
    """
    隨機市場模擬器，支援三種模式：

      'gbm'        — Geometric Brownian Motion（基準）
      'orderbook'  — 簡化訂單簿（原版）
      'multiagent' — 散戶/大戶/做市商 三方對立模型

    推薦研究模式：multiagent，可透過 smart_money_ratio 控制市場結構。
    """

    def __init__(
        self,
        start_price: float = 100.0,
        mode: Literal['gbm', 'orderbook', 'multiagent'] = 'multiagent',
        config: SimConfig | None = None,
    ):
        self.config = config or SimConfig(start_price=start_price)
        self.mode = mode
        self._rng = np.random.default_rng(self.config.seed)

    # ──────────────────────────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────────────────────────

    def run(self, steps: int | None = None) -> np.ndarray:
        """執行模擬，回傳 tick 價格序列 (1D ndarray)。"""
        n = steps or self.config.steps
        if self.mode == 'gbm':
            return self._run_gbm(n)
        if self.mode == 'orderbook':
            return self._run_orderbook(n)
        return self._run_multiagent(n)

    def to_ohlcv(self, price_data: np.ndarray, bar_size: int = 100):
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
        trimmed = price_data[: n_bars * bar_size].reshape(n_bars, bar_size)
        volume = self._rng.integers(100, 10_000, size=n_bars)

        df = pd.DataFrame(
            {
                "open": trimmed[:, 0],
                "high": trimmed.max(axis=1),
                "low": trimmed.min(axis=1),
                "close": trimmed[:, -1],
                "volume": volume,
            }
        )
        df.index = pd.RangeIndex(len(df))
        return df

    # ──────────────────────────────────────────────────────────────────
    # 私有：GBM
    # ──────────────────────────────────────────────────────────────────

    def _run_gbm(self, n: int) -> np.ndarray:
        cfg = self.config
        shocks = self._rng.standard_normal(n)
        log_returns = (cfg.mu - 0.5 * cfg.sigma ** 2) + cfg.sigma * shocks
        prices = cfg.start_price * np.exp(np.cumsum(log_returns))
        return self._apply_shocks(prices)

    # ──────────────────────────────────────────────────────────────────
    # 私有：OrderBook（原版，保留相容性）
    # ──────────────────────────────────────────────────────────────────

    def _run_orderbook(self, n: int) -> np.ndarray:
        cfg = self.config
        prices = np.empty(n)
        price = cfg.start_price

        for i in range(n):
            bid_vol = max(1.0, self._rng.normal(cfg.order_volume_mean, cfg.order_volume_std))
            ask_vol = max(1.0, self._rng.normal(cfg.order_volume_mean, cfg.order_volume_std))
            direction = 1 if self._rng.random() < 0.5 else -1
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
            price_change = direction * cfg.sigma * price * (1 + 0.5 * imbalance)
            mean_reversion = -0.0001 * (price - cfg.start_price)
            price = max(cfg.tick_size, price + price_change + mean_reversion)
            prices[i] = price

        return self._apply_shocks(prices)

    # ──────────────────────────────────────────────────────────────────
    # 私有：MultiAgent（核心新模型）
    # ──────────────────────────────────────────────────────────────────

    def _run_multiagent(self, n: int) -> np.ndarray:
        """
        三方對立模型。每一步的價格變動由三個力量疊加：

          price_change = noise_force + smart_force + mm_force

        noise_force  : 散戶噪音，帶動量偏誤（前一步方向的慣性）
        smart_force  : 大戶反向操作，偵測到價格偏離移動均值後介入
        mm_force     : 做市商庫存壓力，庫存過多時輕微反向平倉

        smart_money_ratio 控制 smart_force 的相對強度：
          越高 → 大戶壓制噪音 → 價格更平滑、更有 mean-reversion
          越低 → 散戶主導 → 震盪劇烈、趨勢更明顯
        """
        cfg = self.config
        prices = np.empty(n)
        price = cfg.start_price

        # 動量狀態（散戶跟風）
        prev_direction = 0.0

        # 做市商庫存（正 = 持有多單）
        mm_inventory = 0.0

        # 移動均值（供大戶計算偏離）— 用 exponential moving average 效率更高
        ema_price = cfg.start_price
        ema_alpha = 2.0 / (cfg.smart_lookback + 1)

        for i in range(n):
            # ── 1. 散戶力量（Noise Trader） ──────────────────────────
            raw_direction = 1.0 if self._rng.random() < 0.5 else -1.0
            # 動量：以 noise_momentum 的比例繼承前一步方向
            blended_direction = (
                (1 - cfg.noise_momentum) * raw_direction
                + cfg.noise_momentum * prev_direction
            )
            # 散戶單量：log-normal 分佈（散戶單子大小差異大）
            noise_size = self._rng.lognormal(mean=0.0, sigma=0.6)
            noise_force = blended_direction * cfg.noise_sigma * price * noise_size

            # ── 2. 大戶力量（Smart Money） ────────────────────────────
            deviation = (price - ema_price) / (ema_price + 1e-8)
            smart_force = 0.0
            if abs(deviation) > cfg.smart_threshold:
                # 偏離越大，大戶反向力道越強（非線性）
                smart_intensity = min(3.0, abs(deviation) / cfg.smart_threshold)
                smart_direction = -1.0 if deviation > 0 else 1.0
                smart_size = self._rng.lognormal(mean=0.5, sigma=0.4)  # 大戶單大
                smart_force = (
                    smart_direction
                    * cfg.smart_sigma
                    * price
                    * smart_intensity
                    * smart_size
                    * cfg.smart_money_ratio
                    / (1 - cfg.smart_money_ratio + 1e-8)  # ratio 越高，力道越大
                )

            # ── 3. 做市商力量（Market Maker） ─────────────────────────
            # 做市商收 spread，並累積庫存；庫存超限時施加反向壓力
            mm_trade = self._rng.choice([-1, 1]) * self._rng.uniform(0.5, 2.0)
            mm_inventory += mm_trade
            mm_force = 0.0
            if abs(mm_inventory) > cfg.mm_inventory_limit * 0.5:
                # 庫存壓力：輕微反向推動價格幫助平倉
                mm_force = (
                    -np.sign(mm_inventory)
                    * cfg.mm_spread
                    * price
                    * (abs(mm_inventory) / cfg.mm_inventory_limit)
                )
            # 庫存超限：強制重置（止損平倉）
            if abs(mm_inventory) > cfg.mm_inventory_limit:
                mm_inventory *= 0.5

            # ── 4. 合計，計算新價格 ───────────────────────────────────
            total_change = noise_force + smart_force + mm_force

            # 弱均值回歸（防止長期漂離）
            global_mr = -0.00005 * (price - cfg.start_price)
            price = max(cfg.tick_size, price + total_change + global_mr)

            # 更新 EMA
            ema_price = ema_alpha * price + (1 - ema_alpha) * ema_price

            # 更新動量
            prev_direction = blended_direction

            prices[i] = price

        return self._apply_shocks(prices)

    # ──────────────────────────────────────────────────────────────────
    # 私有：衝擊事件
    # ──────────────────────────────────────────────────────────────────

    def _apply_shocks(self, prices: np.ndarray) -> np.ndarray:
        """在隨機時間點插入大幅衝擊，模擬黑天鵝事件。"""
        cfg = self.config
        n = len(prices)
        shock_mask = self._rng.random(n) < cfg.shock_prob
        shock_dirs = self._rng.choice([-1, 1], size=n)

        for i in range(1, n):
            if shock_mask[i]:
                prices[i] *= 1 + shock_dirs[i] * cfg.shock_magnitude
        return prices
