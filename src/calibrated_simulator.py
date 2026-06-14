"""
calibrated_simulator.py

用校準後的 theta 進行模擬。
theta 由 scripts/calibrate_params.py 產出（JSON），
或直接傳入 dict。
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from market_estimator import MarketParams
from us_equity_simulator import USStockFutureSimulator, SimulationResult


@dataclass
class CalibratedTheta:
    vol:               float = 0.012
    drift:             float = 0.0
    mr_coeff:          float = 0.03
    node_coeff:        float = 0.02
    hurst_proxy:       float = 0.5
    shock_prob:        float = 0.008
    momentum_strength: float = 0.6
    momentum_decay:    float = 0.75
    breakout_boost:    float = 0.4
    vol_scale:         float = 1.0   # 已含在 vol 裡，這裡設 1

    @classmethod
    def from_dict(cls, d: dict) -> "CalibratedTheta":
        return cls(**{k: float(v) for k, v in d.items() if hasattr(cls, k)})

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


def build_params_from_theta(
    theta: CalibratedTheta,
    base_params: MarketParams,
) -> MarketParams:
    """
    用 theta 覆蓋 base_params 中可校準的欄位，
    其他欄位（volume_nodes、gap_std 等）保留原值。
    """
    import dataclasses
    d = dataclasses.asdict(base_params)
    d["realized_vol"]           = theta.vol
    d["ewma_vol"]               = theta.vol
    d["drift"]                  = theta.drift
    d["ewma_drift"]             = theta.drift
    d["hurst_proxy"]            = theta.hurst_proxy
    d["trend_strength"]         = max(0.0, theta.hurst_proxy - 0.5) * 2.0
    d["mean_reversion_strength"]= max(0.0, 0.5 - theta.hurst_proxy) * 2.0
    return MarketParams(**d)


class CalibratedForwardSimulator:
    """
    用校準後的 theta 往後模擬。
    直接包裝 USStockFutureSimulator，覆蓋 mr_coeff / node_coeff 等。
    """
    def __init__(
        self,
        theta: CalibratedTheta,
        base_params: MarketParams,
        forecast_steps: int = 30,
        n_paths: int = 500,
        seed: int | None = None,
    ):
        self.theta  = theta
        self.params = build_params_from_theta(theta, base_params)
        self.forecast_steps = forecast_steps
        self.n_paths = n_paths
        self.seed    = seed

    def simulate(self) -> SimulationResult:
        sim = USStockFutureSimulator(
            params=self.params,
            forecast_steps=self.forecast_steps,
            n_paths=self.n_paths,
            seed=self.seed,
            vol_scale=self.theta.vol_scale,
            mr_coeff=self.theta.mr_coeff,
            node_coeff=self.theta.node_coeff,
            momentum_strength=self.theta.momentum_strength,
            momentum_decay=self.theta.momentum_decay,
            breakout_boost=self.theta.breakout_boost,
        )
        return sim.simulate()
