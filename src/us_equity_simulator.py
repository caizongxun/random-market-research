"""
us_equity_simulator.py  v5

新增：
  1. backbone_schedule  : 每步的骨幹目標價格，mr_term 改往骨幹回歸（非 start_price）
  2. drift_schedule     : 每步漂移率（v4 保留）
  3. vol_schedule       : 每步波動率（v4 保留）

效果：
  - 帶子中線更緊貼骨幹，不再漂離
  - 低波動段帶寬窄、高波動段帶寬寬
  - 帶子不會隨時間無限擴散（soft clamp via backbone mean-reversion）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

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
        momentum_strength: float = 0.6,
        momentum_decay: float = 0.75,
        breakout_boost: float = 0.4,
        # v4 分段排程
        drift_schedule: Optional[np.ndarray] = None,
        vol_schedule: Optional[np.ndarray] = None,
        # v5 骨幹錨點：mr_term 改往骨幹回歸
        backbone_schedule: Optional[np.ndarray] = None,
        backbone_mr_coeff: float = 0.06,   # 骨幹回歸強度（比 mr_coeff 略強）
    ):
        self.params             = params
        self.forecast_steps     = forecast_steps
        self.n_paths            = n_paths
        self.rng                = np.random.default_rng(seed)
        self.vol_scale          = vol_scale
        self.use_ewma           = use_ewma
        self.tail_df            = tail_df
        self.mr_coeff           = mr_coeff
        self.node_coeff         = node_coeff
        self.momentum_strength  = momentum_strength
        self.momentum_decay     = momentum_decay
        self.breakout_boost     = breakout_boost
        self.drift_schedule     = drift_schedule
        self.vol_schedule       = vol_schedule
        self.backbone_schedule  = backbone_schedule
        self.backbone_mr_coeff  = backbone_mr_coeff

    def simulate(self) -> SimulationResult:
        p     = self.params
        start = p.last_close
        T     = self.forecast_steps

        # --- base drift / vol ---
        if self.use_ewma and hasattr(p, "ewma_vol"):
            base_vol   = p.ewma_vol * self.vol_scale
            base_drift = p.ewma_drift
        else:
            base_vol   = p.realized_vol * self.vol_scale
            base_drift = p.drift

        vt = getattr(p, "vol_trend", 1.0)
        base_vol *= min(vt, 2.0)

        # --- per-step arrays ---
        def _interp(arr, T):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == T:
                return arr
            return np.interp(
                np.linspace(0, 1, T),
                np.linspace(0, 1, len(arr)),
                arr,
            )

        d_arr = _interp(self.drift_schedule, T) if self.drift_schedule is not None else np.full(T, base_drift)
        v_arr = np.maximum(
            _interp(self.vol_schedule, T) * self.vol_scale if self.vol_schedule is not None else np.full(T, base_vol),
            base_vol * 0.4,
        )

        # 骨幹目標價格（每步）
        if self.backbone_schedule is not None:
            bb_arr = _interp(self.backbone_schedule, T)
        else:
            # 退化：骨幹 = start + drift 直線
            bb_arr = start * np.cumprod(1 + d_arr)

        # --- 動量 / breakout ---
        momentum_bias  = getattr(p, "momentum_bias", 0.0)
        breakout_state = getattr(p, "node_breakout_state", 0)
        breakout_extra = breakout_state * self.breakout_boost * base_vol

        upper_node, lower_node = self._nearest_nodes(start)

        # --- 隨機數 ---
        Z     = self.rng.standard_normal((self.n_paths, T))
        chi2  = self.rng.chisquare(df=self.tail_df, size=(self.n_paths, T))
        t_smp = Z / np.sqrt(chi2 / self.tail_df)

        paths = np.zeros((self.n_paths, T))

        for i in range(self.n_paths):
            price    = start
            prev_ret = 0.0
            for t in range(T):
                step_drift = d_arr[t]
                step_vol   = v_arr[t]
                bb_target  = bb_arr[t]

                decay    = self.momentum_strength * (self.momentum_decay ** t)
                noise    = t_smp[i, t] * step_vol
                mom_term = decay * momentum_bias
                bo_term  = breakout_extra * (self.momentum_decay ** (t * 2))

                trend_term = p.trend_strength * 0.3 * prev_ret

                # v5 核心改動：mr_term 往骨幹目標回歸
                bb_dev   = (price - bb_target) / bb_target   # 與骨幹的偏差
                mr_term  = -self.backbone_mr_coeff * bb_dev

                gap_term  = self.rng.normal(0.0, p.gap_std * 0.15)
                node_term = self._volume_node_force(price, step_vol)
                shock     = self._shock_component(step_vol)

                ret      = step_drift + noise + mom_term + bo_term + trend_term + mr_term + gap_term + node_term + shock
                price    = max(0.01, price * (1.0 + ret))
                prev_ret = ret
                paths[i, t] = price

        median = np.median(paths, axis=0)
        p10    = np.percentile(paths, 10, axis=0)
        p25    = np.percentile(paths, 25, axis=0)
        p75    = np.percentile(paths, 75, axis=0)
        p90    = np.percentile(paths, 90, axis=0)

        bup = float((paths[:, -1] > upper_node).mean()) if upper_node is not None else 0.0
        bdn = float((paths[:, -1] < lower_node).mean()) if lower_node is not None else 0.0

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

    def _volume_node_force(self, price: float, step_vol: float) -> float:
        if not self.params.volume_nodes:
            return 0.0
        total = 0.0
        for node, strength in zip(self.params.volume_nodes, self.params.volume_node_strength):
            dist  = (price - node) / node
            width = max(step_vol * 8, 0.01)
            pull  = -dist * np.exp(-(dist ** 2) / (2 * width ** 2))
            total += self.node_coeff * strength * self.params.smart_money_ratio * pull
        return float(total)

    def _shock_component(self, step_vol: float) -> float:
        p_shock = 0.008 + 0.025 * (1.0 - self.params.smart_money_ratio)
        if self.rng.random() < p_shock:
            direction = 1.0 if self.rng.random() > 0.5 else -1.0
            return direction * abs(self.rng.normal(0.0, step_vol * 3.5))
        return 0.0
