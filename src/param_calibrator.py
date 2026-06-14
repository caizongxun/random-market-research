"""
param_calibrator.py  v3

修改：
1. trend_match 改成「幅度加權」：大漲/大跌段方向錯，懲罰更重
2. 新增 backbone_path 支援：若傳入骨幹路徑，end_price 和 trend 都對骨幹比較
   (Phase 2 模式：以骨幹為中心做隨機演化校準)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from market_estimator import MarketParams
from calibrated_simulator import CalibratedTheta
from us_equity_simulator import USStockFutureSimulator


@dataclass
class LossWeights:
    vol:      float = 1.0
    acf:      float = 0.8
    drawdown: float = 0.6
    node:     float = 0.5
    skew:     float = 0.4
    trend:    float = 2.0   # v3: 幅度加權後可以調低一點，但方向更精準
    end:      float = 1.5


class ParamCalibrator:
    BOUNDS = {
        "vol":               (0.003, 0.08),
        "drift":             (-0.003, 0.003),
        "mr_coeff":          (0.0,   0.15),
        "node_coeff":        (0.0,   0.08),
        "hurst_proxy":       (0.2,   0.8),
        "shock_prob":        (0.001, 0.03),
        "momentum_strength": (0.1,   1.0),
        "momentum_decay":    (0.5,   0.95),
        "breakout_boost":    (0.0,   1.0),
    }
    PARAM_KEYS = list(BOUNDS.keys())

    def __init__(
        self,
        base_params: MarketParams,
        history_close: np.ndarray,
        weights: Optional[LossWeights] = None,
        n_sim_paths: int = 200,
        seed: int = 0,
        maxiter: int = 600,
        verbose: bool = True,
        backbone_path: Optional[np.ndarray] = None,
    ):
        self.base_params   = base_params
        self.history_close = np.asarray(history_close, dtype=float)
        self.weights       = weights or LossWeights()
        self.n_sim_paths   = n_sim_paths
        self.n_sim_steps   = len(self.history_close)
        self.seed          = seed
        self.maxiter       = maxiter
        self.verbose       = verbose
        self.start_price   = float(self.history_close[0])

        # Phase 2：若有骨幹路徑，trend/end 對骨幹比，其餘統計量對真實比
        self.backbone_path = backbone_path
        target = backbone_path if backbone_path is not None else self.history_close

        self._real_stats  = self._compute_stats(self.history_close)
        self._target_trend = self._segment_trend_weighted(target)
        self._target_end   = float(target[-1])

    # ------------------------------------------------------------------ #
    def calibrate(self, init_theta: Optional[CalibratedTheta] = None) -> CalibratedTheta:
        if init_theta is None:
            init_theta = self._init_from_base_params()

        x0     = self._theta_to_vec(init_theta)
        lo, hi = zip(*[self.BOUNDS[k] for k in self.PARAM_KEYS])
        lo, hi = np.array(lo), np.array(hi)
        iteration = [0]

        def objective(x):
            x_c   = np.clip(x, lo, hi)
            theta = self._vec_to_theta(x_c)
            loss  = self._loss(theta)
            iteration[0] += 1
            if self.verbose and iteration[0] % 50 == 0:
                print(f"  iter {iteration[0]:4d}  loss={loss:.6f}")
            return loss

        result: OptimizeResult = minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": self.maxiter, "xatol": 1e-5,
                     "fatol": 1e-5, "adaptive": True},
        )
        best = self._vec_to_theta(np.clip(result.x, lo, hi))
        if self.verbose:
            print(f"  Done. loss={result.fun:.6f}  iters={result.nit}")
        return best

    # ------------------------------------------------------------------ #
    def _loss(self, theta: CalibratedTheta) -> float:
        paths = self._run_sim(theta)
        w     = self.weights
        total = 0.0
        T     = paths.shape[0]

        stats_list = [self._compute_stats(paths[i]) for i in range(T)]

        def med(key):
            return float(np.median([s[key] for s in stats_list]))

        r = self._real_stats

        total += w.vol      * abs(med("vol")    - r["vol"])    / (r["vol"] + 1e-8)
        for lag in range(1, 6):
            total += w.acf * 0.2 * abs(med(f"acf_{lag}") - r[f"acf_{lag}"])
        total += w.drawdown * abs(med("max_dd") - r["max_dd"]) / (abs(r["max_dd"]) + 1e-8)
        total += w.skew     * abs(med("skew")   - r["skew"])   / (abs(r["skew"]) + 1e-4)

        if self.base_params.volume_nodes:
            real_ns = self._node_behavior_score(
                self.history_close, self.base_params.volume_nodes)
            sim_ns = float(np.median([
                self._node_behavior_score(paths[i], self.base_params.volume_nodes)
                for i in range(T)]))
            total += w.node * abs(sim_ns - real_ns)

        # ★ 幅度加權 trend 懲罰
        tt = self._target_trend  # [(diff, weight), ...]
        penalties = []
        for path in paths:
            pt = self._segment_trend_weighted(path)
            seg_penalty = 0.0
            for (td, tw), (pd, _) in zip(tt, pt):
                if np.sign(td) != np.sign(pd):
                    seg_penalty += tw  # 方向錯就加這段的幅度權重
            penalties.append(seg_penalty)
        total += w.trend * float(np.median(penalties))

        # 終點懲罰
        sim_ends = [paths[i, -1] for i in range(T)]
        total += w.end * abs(float(np.median(sim_ends)) - self._target_end) / (self._target_end + 1e-8)

        return float(total)

    def _run_sim(self, theta: CalibratedTheta) -> np.ndarray:
        import dataclasses
        from calibrated_simulator import build_params_from_theta

        base = dataclasses.replace(
            self.base_params,
            last_close=self.start_price,
            momentum_bias=0.0,
            node_breakout_state=0,
        )
        params = build_params_from_theta(theta, base)
        rng = np.random.default_rng(self.seed)
        sim = USStockFutureSimulator(
            params=params,
            forecast_steps=self.n_sim_steps,
            n_paths=self.n_sim_paths,
            seed=int(rng.integers(0, 2**31)),
            vol_scale=1.0,
            mr_coeff=theta.mr_coeff,
            node_coeff=theta.node_coeff,
            momentum_strength=theta.momentum_strength,
            momentum_decay=theta.momentum_decay,
            breakout_boost=theta.breakout_boost,
        )
        return sim.simulate().future_paths

    @staticmethod
    def _segment_trend_weighted(prices: np.ndarray, n_seg: int = 3):
        """
        回傳 [(diff, weight), ...]
        weight = |diff| / total_movement，幅度大的段權重大
        """
        n   = len(prices)
        seg = n // n_seg
        diffs = []
        for i in range(n_seg):
            s = i * seg
            e = s + seg if i < n_seg - 1 else n
            diffs.append(float(prices[e - 1] - prices[s]))
        total_move = sum(abs(d) for d in diffs) + 1e-8
        return [(d, abs(d) / total_move) for d in diffs]

    @staticmethod
    def _compute_stats(prices: np.ndarray) -> dict:
        rets  = np.diff(np.log(prices + 1e-8))
        stats = {"vol": float(np.std(rets))}
        for lag in range(1, 6):
            acf = float(np.corrcoef(rets[:-lag], rets[lag:])[0, 1]) if len(rets) > lag else 0.0
            stats[f"acf_{lag}"] = acf if not np.isnan(acf) else 0.0
        peak = np.maximum.accumulate(prices)
        dd   = (prices - peak) / (peak + 1e-8)
        stats["max_dd"] = float(dd.min())
        if len(rets) > 3:
            m = rets.mean(); s = rets.std() + 1e-8
            stats["skew"] = float(np.mean(((rets - m) / s) ** 3))
        else:
            stats["skew"] = 0.0
        return stats

    @staticmethod
    def _node_behavior_score(prices: np.ndarray, nodes: list[float]) -> float:
        if len(prices) < 3 or not nodes:
            return 0.0
        nodes_arr = np.array(nodes)
        bounces, total = 0, 0
        for i in range(1, len(prices) - 1):
            p    = prices[i]
            near = np.any(np.abs(nodes_arr - p) / p < 0.02)
            if near:
                total += 1
                pd = np.sign(prices[i] - prices[i - 1])
                nd = np.sign(prices[i + 1] - prices[i])
                if pd != 0 and nd == -pd:
                    bounces += 1
        return bounces / (total + 1e-8)

    def _init_from_base_params(self) -> CalibratedTheta:
        p = self.base_params
        real_drift = float(
            np.log(self.history_close[-1] / self.history_close[0])
            / max(len(self.history_close), 1)
        )
        return CalibratedTheta(
            vol=p.ewma_vol,
            drift=np.clip(real_drift, -0.003, 0.003),
            mr_coeff=0.04, node_coeff=0.02,
            hurst_proxy=p.hurst_proxy,
            shock_prob=0.008,
            momentum_strength=0.6, momentum_decay=0.75, breakout_boost=0.4,
        )

    def _theta_to_vec(self, theta: CalibratedTheta) -> np.ndarray:
        return np.array([getattr(theta, k) for k in self.PARAM_KEYS])

    def _vec_to_theta(self, x: np.ndarray) -> CalibratedTheta:
        return CalibratedTheta(**{k: float(v) for k, v in zip(self.PARAM_KEYS, x)})
