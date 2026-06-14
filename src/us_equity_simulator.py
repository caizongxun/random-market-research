"""
us_equity_simulator.py

以免費美股資料為輸入：
1. 拉取歷史 OHLCV
2. 從最近 500 根 K 棒估計隱含市場參數
3. 建立帶有 Volume Profile 記憶的未來隨機模擬
4. 產出多條未來路徑與分位帶
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from market_estimator import MarketParams


@dataclass
class SimulationResult:
    future_paths: np.ndarray
    median_path: np.ndarray
    p10: np.ndarray
    p25: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    breakout_prob_up: float
    breakout_prob_down: float
    nearest_upper_node: float | None
    nearest_lower_node: float | None


class USStockFutureSimulator:
    def __init__(self, params: MarketParams, forecast_steps: int = 30, n_paths: int = 1000, seed: int | None = None):
        self.params = params
        self.forecast_steps = forecast_steps
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def simulate(self) -> SimulationResult:
        paths = np.zeros((self.n_paths, self.forecast_steps))
        start = self.params.last_close

        upper_node, lower_node = self._nearest_nodes(start)

        for i in range(self.n_paths):
            price = start
            prev_ret = 0.0
            for t in range(self.forecast_steps):
                noise = self.rng.normal(0.0, self.params.realized_vol)
                drift = self.params.drift

                trend_term = self.params.trend_strength * 0.35 * prev_ret
                mr_term = -self.params.mean_reversion_strength * 0.15 * (price / start - 1.0)
                gap_term = self.rng.normal(0.0, self.params.gap_std * 0.2)
                node_term = self._volume_node_force(price)
                shock = self._shock_component()

                ret = drift + noise + trend_term + mr_term + gap_term + node_term + shock
                price = max(0.01, price * (1.0 + ret))
                prev_ret = ret
                paths[i, t] = price

        median = np.median(paths, axis=0)
        p10 = np.percentile(paths, 10, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        breakout_prob_up = 0.0
        breakout_prob_down = 0.0
        if upper_node is not None:
            breakout_prob_up = float((paths[:, -1] > upper_node).mean())
        if lower_node is not None:
            breakout_prob_down = float((paths[:, -1] < lower_node).mean())

        return SimulationResult(
            future_paths=paths,
            median_path=median,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
            breakout_prob_up=breakout_prob_up,
            breakout_prob_down=breakout_prob_down,
            nearest_upper_node=upper_node,
            nearest_lower_node=lower_node,
        )

    def _nearest_nodes(self, price: float) -> tuple[float | None, float | None]:
        nodes = np.array(self.params.volume_nodes, dtype=float)
        if len(nodes) == 0:
            return None, None
        upper = nodes[nodes > price]
        lower = nodes[nodes < price]
        upper_node = float(upper.min()) if len(upper) else None
        lower_node = float(lower.max()) if len(lower) else None
        return upper_node, lower_node

    def _volume_node_force(self, price: float) -> float:
        if not self.params.volume_nodes:
            return 0.0
        total_force = 0.0
        for node, strength in zip(self.params.volume_nodes, self.params.volume_node_strength):
            dist = (price - node) / node
            width = max(self.params.realized_vol * 8, 0.01)
            pull = -dist * np.exp(-(dist ** 2) / (2 * width ** 2))
            total_force += 0.08 * strength * self.params.smart_money_ratio * pull
        return float(total_force)

    def _shock_component(self) -> float:
        p = 0.01 + 0.03 * (1.0 - self.params.smart_money_ratio)
        if self.rng.random() < p:
            return float(self.rng.normal(0.0, self.params.realized_vol * 4.0))
        return 0.0
