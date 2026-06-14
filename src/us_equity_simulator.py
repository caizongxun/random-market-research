"""
us_equity_simulator.py

v3 新增：條件式模擬（Conditional Simulation）

前幾步加入動量偏向：
1. momentum_bias：最近 N 根平均日回報，作為前幾步的願動力向偏移
2. node_breakout_state：突破上方節點則加挑，被拒則加賣壓
3. vol_trend：波動放大中則放大模擬帶寬
4. 動量衍减：前幾步的偏向會隨時間指數衍减，讓後半段慡層回歸對稱渴布
"""

from __future__ import annotations

import numpy as np
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
    def __init__(
        self,
        params: MarketParams,
        forecast_steps: int = 30,
        n_paths: int = 1000,
        seed: int | None = None,
        vol_scale: float = 1.5,
        use_ewma: bool = True,
        tail_df: int = 5,
        mr_coeff: float = 0.04,
        node_coeff: float = 0.02,
        # v3 動量參數
        momentum_strength: float = 0.6,   # 動量偏向的最大倍數
        momentum_decay: float = 0.75,     # 每步衍减率（指數衍减）
        breakout_boost: float = 0.4,      # 突破/被拒的額外偏向 (x vol)
    ):
        self.params            = params
        self.forecast_steps    = forecast_steps
        self.n_paths           = n_paths
        self.rng               = np.random.default_rng(seed)
        self.vol_scale         = vol_scale
        self.use_ewma          = use_ewma
        self.tail_df           = tail_df
        self.mr_coeff          = mr_coeff
        self.node_coeff        = node_coeff
        self.momentum_strength = momentum_strength
        self.momentum_decay    = momentum_decay
        self.breakout_boost    = breakout_boost

    def simulate(self) -> SimulationResult:
        p = self.params
        start = p.last_close

        # 選擇 vol 和 drift
        if self.use_ewma and hasattr(p, "ewma_vol"):
            base_vol   = p.ewma_vol * self.vol_scale
            base_drift = p.ewma_drift
        else:
            base_vol   = p.realized_vol * self.vol_scale
            base_drift = p.drift

        # vol_trend 放大：如果波動率正在擴大，帶寬再加大
        vt = getattr(p, "vol_trend", 1.0)
        base_vol *= min(vt, 2.0)  # 最多放大 2x，避免爆炸

        # 動量起始偏向
        momentum_bias  = getattr(p, "momentum_bias", 0.0)
        breakout_state = getattr(p, "node_breakout_state", 0)

        # breakout 額外偏向：突破則正，被拒則負
        breakout_extra = breakout_state * self.breakout_boost * base_vol

        upper_node, lower_node = self._nearest_nodes(start)

        # 預先生成學生 t 隨機數
        Z    = self.rng.standard_normal((self.n_paths, self.forecast_steps))
        chi2 = self.rng.chisquare(
            df=self.tail_df, size=(self.n_paths, self.forecast_steps)
        )
        t_samples = Z / np.sqrt(chi2 / self.tail_df)

        paths = np.zeros((self.n_paths, self.forecast_steps))

        for i in range(self.n_paths):
            price    = start
            prev_ret = 0.0
            for t in range(self.forecast_steps):
                # 動量衍减係數：前幾步強，後面歸零
                decay     = self.momentum_strength * (self.momentum_decay ** t)

                noise     = t_samples[i, t] * base_vol
                # 動量偏向：衍减的 momentum_bias，前幾步有方向性
                mom_term  = decay * momentum_bias
                # breakout 突破第一步最強，後面决速衍减
                bo_term   = breakout_extra * (self.momentum_decay ** (t * 2))

                trend_term = p.trend_strength * 0.3 * prev_ret
                mr_term    = -p.mean_reversion_strength * self.mr_coeff * (price / start - 1.0)
                gap_term   = self.rng.normal(0.0, p.gap_std * 0.15)
                node_term  = self._volume_node_force(price, base_vol)
                shock      = self._shock_component(base_vol)

                ret   = base_drift + noise + mom_term + bo_term + trend_term + mr_term + gap_term + node_term + shock
                price = max(0.01, price * (1.0 + ret))
                prev_ret = ret
                paths[i, t] = price

        median = np.median(paths, axis=0)
        p10    = np.percentile(paths, 10, axis=0)
        p25    = np.percentile(paths, 25, axis=0)
        p75    = np.percentile(paths, 75, axis=0)
        p90    = np.percentile(paths, 90, axis=0)

        bup  = float((paths[:, -1] > upper_node).mean()) if upper_node is not None else 0.0
        bdn  = float((paths[:, -1] < lower_node).mean()) if lower_node is not None else 0.0

        return SimulationResult(
            future_paths=paths,
            median_path=median,
            p10=p10, p25=p25, p75=p75, p90=p90,
            breakout_prob_up=bup,
            breakout_prob_down=bdn,
            nearest_upper_node=upper_node,
            nearest_lower_node=lower_node,
        )

    def _nearest_nodes(self, price: float) -> tuple[float | None, float | None]:
        nodes = np.array(self.params.volume_nodes, dtype=float)
        if len(nodes) == 0:
            return None, None
        upper = nodes[nodes > price]
        lower = nodes[nodes < price]
        return (
            float(upper.min()) if len(upper) else None,
            float(lower.max()) if len(lower) else None,
        )

    def _volume_node_force(self, price: float, base_vol: float) -> float:
        if not self.params.volume_nodes:
            return 0.0
        total = 0.0
        for node, strength in zip(self.params.volume_nodes, self.params.volume_node_strength):
            dist  = (price - node) / node
            width = max(base_vol * 8, 0.01)
            pull  = -dist * np.exp(-(dist ** 2) / (2 * width ** 2))
            total += self.node_coeff * strength * self.params.smart_money_ratio * pull
        return float(total)

    def _shock_component(self, base_vol: float) -> float:
        p_shock = 0.008 + 0.025 * (1.0 - self.params.smart_money_ratio)
        if self.rng.random() < p_shock:
            direction = 1.0 if self.rng.random() > 0.5 else -1.0
            return direction * abs(self.rng.normal(0.0, base_vol * 3.5))
        return 0.0
