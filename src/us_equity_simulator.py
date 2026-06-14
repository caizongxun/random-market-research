"""
us_equity_simulator.py  v7

修復：
  1. Drift 衰減 (drift_decay_rate): drift * exp(-decay_rate * t)
     讓預測愈往後，drift 貢獻愈小，路徑自然散開而非單方向飛
  2. Momentum anchor 改為相對預測起點的短期斜率
     均值回歸目標從「骨幹長期值」微調為「短期動量錨點加權」
  3. 所有對外接口向下相容（新參數皆有預設值）
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
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
    # v6+
    representative_path: np.ndarray = field(default_factory=lambda: np.array([]))
    ohlcv_open:   np.ndarray = field(default_factory=lambda: np.array([]))
    ohlcv_high:   np.ndarray = field(default_factory=lambda: np.array([]))
    ohlcv_low:    np.ndarray = field(default_factory=lambda: np.array([]))
    ohlcv_close:  np.ndarray = field(default_factory=lambda: np.array([]))
    ohlcv_volume: np.ndarray = field(default_factory=lambda: np.array([]))


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
        drift_schedule: Optional[np.ndarray] = None,
        vol_schedule: Optional[np.ndarray] = None,
        backbone_schedule: Optional[np.ndarray] = None,
        backbone_mr_coeff: float = 0.06,
        # v6 intra-bar
        intra_bar_steps: int = 8,
        # v7 drift 衰減 + anchor
        drift_decay_rate: float = 0.05,
        drift_scale: float = 1.0,
        momentum_anchor_weight: float = 0.3,
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
        self.intra_bar_steps    = intra_bar_steps
        # v7
        self.drift_decay_rate        = drift_decay_rate
        self.drift_scale             = drift_scale
        self.momentum_anchor_weight  = momentum_anchor_weight

    # ------------------------------------------------------------------ #
    def simulate(self) -> SimulationResult:
        p     = self.params
        start = p.last_close
        T     = self.forecast_steps

        if self.use_ewma and hasattr(p, "ewma_vol"):
            base_vol   = p.ewma_vol * self.vol_scale
            base_drift = p.ewma_drift
        else:
            base_vol   = p.realized_vol * self.vol_scale
            base_drift = p.drift

        vt = getattr(p, "vol_trend", 1.0)
        base_vol *= min(vt, 2.0)

        def _interp(arr, T):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == T:
                return arr
            return np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(arr)), arr)

        d_arr = _interp(self.drift_schedule, T) if self.drift_schedule is not None else np.full(T, base_drift)
        v_arr = np.maximum(
            _interp(self.vol_schedule, T) * self.vol_scale if self.vol_schedule is not None else np.full(T, base_vol),
            base_vol * 0.4,
        )
        bb_arr = _interp(self.backbone_schedule, T) if self.backbone_schedule is not None else start * np.cumprod(1 + d_arr)

        # v7: 預先計算每步的 decay 乘數（exponential decay to 0）
        t_idx   = np.arange(T, dtype=float)
        decay_w = np.exp(-self.drift_decay_rate * t_idx)  # shape (T,)
        # 套用 drift_scale + decay
        d_arr_decayed = d_arr * self.drift_scale * decay_w

        momentum_bias  = getattr(p, "momentum_bias",       0.0)
        breakout_state = getattr(p, "node_breakout_state", 0)
        breakout_extra = breakout_state * self.breakout_boost * base_vol

        # v7: momentum anchor = 起點附近的短期斜率（用 bb_arr 前幾步估算）
        anchor_slope = 0.0
        if len(bb_arr) >= 5:
            anchor_slope = float(np.mean(np.diff(np.log(bb_arr[:5]))))
        # anchor_slope 代表「市場自身認為的短期趨勢」，往後逐漸衰減
        anchor_decay = np.exp(-self.drift_decay_rate * 2.0 * t_idx)
        momentum_anchor_term = anchor_slope * anchor_decay * self.momentum_anchor_weight

        upper_node, lower_node = self._nearest_nodes(start)

        Z     = self.rng.standard_normal((self.n_paths, T))
        chi2  = self.rng.chisquare(df=self.tail_df, size=(self.n_paths, T))
        t_smp = Z / np.sqrt(chi2 / self.tail_df)

        paths = np.zeros((self.n_paths, T))

        for i in range(self.n_paths):
            price    = start
            prev_ret = 0.0
            for t in range(T):
                step_drift = d_arr_decayed[t]          # v7: decayed drift
                step_vol   = v_arr[t]
                bb_target  = bb_arr[t]

                decay_m    = self.momentum_strength * (self.momentum_decay ** t)
                noise      = t_smp[i, t] * step_vol
                mom_term   = decay_m * momentum_bias
                bo_term    = breakout_extra * (self.momentum_decay ** (t * 2))
                trend_term = p.trend_strength * 0.3 * prev_ret

                # v7: backbone MR 目標加入 momentum anchor 偏移
                anchor_offset = float(momentum_anchor_term[t])
                bb_dev    = (price - bb_target) / bb_target
                # 減弱 backbone MR（避免把路徑硬拉向可能偏高的 backbone）
                mr_weight = np.exp(-self.drift_decay_rate * 0.5 * t)  # MR 也隨時間減弱
                mr_term   = -self.backbone_mr_coeff * mr_weight * bb_dev + anchor_offset

                gap_term   = self.rng.normal(0.0, p.gap_std * 0.15)
                node_term  = self._volume_node_force(price, step_vol)
                shock      = self._shock_component(step_vol)

                ret      = step_drift + noise + mom_term + bo_term + trend_term + mr_term + gap_term + node_term + shock
                price    = max(0.01, price * (1.0 + ret))
                prev_ret = ret
                paths[i, t] = price

        median = np.median(paths, axis=0)
        p10    = np.percentile(paths, 10, axis=0)
        p25    = np.percentile(paths, 25, axis=0)
        p75    = np.percentile(paths, 75, axis=0)
        p90    = np.percentile(paths, 90, axis=0)

        # representative path: RMSE vs median 最小
        rmse     = np.sqrt(np.mean((paths - median[None, :]) ** 2, axis=1))
        rep_path = paths[int(np.argmin(rmse))]

        # intra-bar OHLCV
        ohlcv = self._build_ohlcv(rep_path, v_arr, start)

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
            representative_path=rep_path,
            ohlcv_open=ohlcv["open"],
            ohlcv_high=ohlcv["high"],
            ohlcv_low=ohlcv["low"],
            ohlcv_close=ohlcv["close"],
            ohlcv_volume=ohlcv["volume"],
        )

    # ------------------------------------------------------------------ #
    def _build_ohlcv(
        self,
        close_path: np.ndarray,
        v_arr: np.ndarray,
        start_price: float,
    ) -> dict:
        T   = len(close_path)
        K   = self.intra_bar_steps
        rng = self.rng

        o_arr     = np.empty(T)
        h_arr     = np.empty(T)
        l_arr     = np.empty(T)
        c_arr     = close_path.copy()
        v_arr_out = np.empty(T)

        prev_close = start_price
        for t in range(T):
            bar_open  = prev_close
            bar_close = c_arr[t]
            intra_vol = v_arr[t] / np.sqrt(K)

            log_target  = np.log(bar_close / bar_open) if bar_open > 0 else 0.0
            drift_intra = log_target / K

            prices    = np.empty(K + 1)
            prices[0] = bar_open
            for k in range(K):
                noise       = rng.normal(0.0, intra_vol)
                prices[k+1] = max(0.01, prices[k] * np.exp(drift_intra + noise))
            prices[-1] = bar_close

            o_arr[t] = bar_open
            h_arr[t] = float(prices.max())
            l_arr[t] = float(prices.min())
            prev_close = bar_close

            bar_ret    = abs(bar_close / bar_open - 1) if bar_open > 0 else 0.0
            vol_factor = np.exp(rng.normal(0.0, 0.6) + bar_ret * 8)
            v_arr_out[t] = float(vol_factor)

        h_arr = np.maximum(h_arr, np.maximum(o_arr, c_arr))
        l_arr = np.minimum(l_arr, np.minimum(o_arr, c_arr))

        return {"open": o_arr, "high": h_arr, "low": l_arr,
                "close": c_arr, "volume": v_arr_out}

    # ------------------------------------------------------------------ #
    def _nearest_nodes(self, price):
        nodes = np.array(self.params.volume_nodes, dtype=float)
        if len(nodes) == 0:
            return None, None
        upper = nodes[nodes > price]
        lower = nodes[nodes < price]
        return (float(upper.min()) if len(upper) else None,
                float(lower.max()) if len(lower) else None)

    def _volume_node_force(self, price, step_vol):
        if not self.params.volume_nodes:
            return 0.0
        total = 0.0
        for node, strength in zip(self.params.volume_nodes, self.params.volume_node_strength):
            dist  = (price - node) / node
            width = max(step_vol * 8, 0.01)
            pull  = -dist * np.exp(-(dist**2) / (2 * width**2))
            total += self.node_coeff * strength * self.params.smart_money_ratio * pull
        return float(total)

    def _shock_component(self, step_vol):
        p_shock = 0.008 + 0.025 * (1.0 - self.params.smart_money_ratio)
        if self.rng.random() < p_shock:
            d = 1.0 if self.rng.random() > 0.5 else -1.0
            return d * abs(self.rng.normal(0.0, step_vol * 3.5))
        return 0.0
