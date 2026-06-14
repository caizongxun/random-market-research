"""
param_calibrator.py

校準模擬器參數 theta，使得用 theta 模擬出來的路徑
在統計上「最像」歷史 K 棒。

Loss Function（可調權重）：
  L(θ) = λ_vol      * |σ_sim - σ_real|
        + λ_acf      * ||acf_sim - acf_real||
        + λ_drawdown * |dd_sim - dd_real|
        + λ_node     * node_behavior_diff
        + λ_skew     * |skew_sim - skew_real|

優化器：scipy Nelder-Mead（無梯度，適合模擬器）
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
from calibrated_simulator import CalibratedTheta, CalibratedForwardSimulator
from us_equity_simulator import USStockFutureSimulator


@dataclass
class LossWeights:
    vol:      float = 1.0   # 波動大小
    acf:      float = 0.8   # 自相關/記憶性（hurst）
    drawdown: float = 0.6   # 最大回撤分佈
    node:     float = 0.5   # volume node 附近的反彈/穿越行為
    skew:     float = 0.4   # 回報偏態


class ParamCalibrator:
    """
    輸入：歷史 K 棒的 close 序列
    輸出：最佳 CalibratedTheta
    """

    # theta 的搜索邊界
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
        n_sim_steps: int | None = None,
        seed: int = 0,
        maxiter: int = 400,
        verbose: bool = True,
    ):
        self.base_params    = base_params
        self.history_close  = np.asarray(history_close, dtype=float)
        self.weights        = weights or LossWeights()
        self.n_sim_paths    = n_sim_paths
        self.n_sim_steps    = n_sim_steps or len(history_close)
        self.seed           = seed
        self.maxiter        = maxiter
        self.verbose        = verbose

        # 預先算好真實歷史的統計量
        self._real_stats = self._compute_stats(self.history_close)

    # ------------------------------------------------------------------ #
    #  公開方法
    # ------------------------------------------------------------------ #

    def calibrate(self, init_theta: Optional[CalibratedTheta] = None) -> CalibratedTheta:
        if init_theta is None:
            init_theta = self._init_from_base_params()

        x0     = self._theta_to_vec(init_theta)
        lo, hi = zip(*[self.BOUNDS[k] for k in self.PARAM_KEYS])
        lo, hi = np.array(lo), np.array(hi)

        iteration = [0]
        best      = [np.inf]

        def objective(x):
            x_clipped = np.clip(x, lo, hi)
            theta     = self._vec_to_theta(x_clipped)
            loss      = self._loss(theta)
            iteration[0] += 1
            if self.verbose and iteration[0] % 50 == 0:
                print(f"  iter {iteration[0]:4d}  loss={loss:.6f}")
            if loss < best[0]:
                best[0] = loss
            return loss

        result: OptimizeResult = minimize(
            objective, x0,
            method="Nelder-Mead",
            options={
                "maxiter": self.maxiter,
                "xatol":   1e-5,
                "fatol":   1e-5,
                "adaptive": True,
            },
        )

        best_theta = self._vec_to_theta(np.clip(result.x, lo, hi))
        if self.verbose:
            print(f"  Calibration done. Final loss={result.fun:.6f}  iters={result.nit}")
        return best_theta

    # ------------------------------------------------------------------ #
    #  內部方法
    # ------------------------------------------------------------------ #

    def _loss(self, theta: CalibratedTheta) -> float:
        paths = self._run_sim(theta)  # (n_paths, n_steps)
        w     = self.weights
        total = 0.0

        sim_stats_list = [self._compute_stats(paths[i]) for i in range(paths.shape[0])]

        # 取中位數統計量比較（比用均值穩定）
        def med(key):
            return float(np.median([s[key] for s in sim_stats_list]))

        r = self._real_stats

        # λ_vol：波動率差距
        total += w.vol * abs(med("vol") - r["vol"]) / (r["vol"] + 1e-8)

        # λ_acf：lag-1 to lag-5 自相關差距
        for lag in range(1, 6):
            k = f"acf_{lag}"
            total += w.acf * 0.2 * abs(med(k) - r[k])

        # λ_drawdown：最大回撤差距
        total += w.drawdown * abs(med("max_dd") - r["max_dd"]) / (abs(r["max_dd"]) + 1e-8)

        # λ_skew：回報偏態差距
        total += w.skew * abs(med("skew") - r["skew"]) / (abs(r["skew"]) + 1e-4)

        # λ_node：在 node 附近的「反彈率」差距
        if self.base_params.volume_nodes:
            real_node_score = self._node_behavior_score(
                self.history_close, self.base_params.volume_nodes
            )
            sim_node_scores = [
                self._node_behavior_score(paths[i], self.base_params.volume_nodes)
                for i in range(paths.shape[0])
            ]
            total += w.node * abs(np.median(sim_node_scores) - real_node_score)

        return float(total)

    def _run_sim(self, theta: CalibratedTheta) -> np.ndarray:
        """
        用 theta 模擬，回傳 price paths（形狀：n_paths × n_steps）
        注意：這裡模擬的是「重現歷史」，所以 forecast_steps = n_sim_steps
        且 momentum_bias 設為 0（校準期不預設方向）
        """
        import dataclasses
        # 暫時把 base_params 的 momentum_bias 清零，讓校準更乾淨
        base = dataclasses.replace(self.base_params, momentum_bias=0.0, node_breakout_state=0)

        from calibrated_simulator import build_params_from_theta
        params = build_params_from_theta(theta, base)

        rng  = np.random.default_rng(self.seed)
        sim  = USStockFutureSimulator(
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
        result = sim.simulate()
        return result.future_paths  # (n_paths, n_steps)

    @staticmethod
    def _compute_stats(prices: np.ndarray) -> dict:
        rets  = np.diff(np.log(prices + 1e-8))
        stats = {"vol": float(np.std(rets))}
        for lag in range(1, 6):
            if len(rets) > lag:
                acf = float(np.corrcoef(rets[:-lag], rets[lag:])[0, 1])
            else:
                acf = 0.0
            stats[f"acf_{lag}"] = acf if not np.isnan(acf) else 0.0

        # 最大回撤
        peak   = np.maximum.accumulate(prices)
        dd     = (prices - peak) / (peak + 1e-8)
        stats["max_dd"] = float(dd.min())

        # 回報偏態
        if len(rets) > 3:
            m = rets.mean()
            s = rets.std() + 1e-8
            stats["skew"] = float(np.mean(((rets - m) / s) ** 3))
        else:
            stats["skew"] = 0.0
        return stats

    @staticmethod
    def _node_behavior_score(prices: np.ndarray, nodes: list[float]) -> float:
        """
        計算價格在 volume node 附近的「反彈率」：
        進入 node ±2% 範圍後，下一步反彈（回頭走）的比例
        越高 = 阻力/支撐越有效
        """
        if len(prices) < 3 or not nodes:
            return 0.0
        nodes_arr = np.array(nodes)
        bounces, total = 0, 0
        for i in range(1, len(prices) - 1):
            p = prices[i]
            near = np.any(np.abs(nodes_arr - p) / p < 0.02)
            if near:
                total += 1
                prev_dir = np.sign(prices[i] - prices[i - 1])
                next_dir = np.sign(prices[i + 1] - prices[i])
                if prev_dir != 0 and next_dir == -prev_dir:
                    bounces += 1
        return bounces / (total + 1e-8)

    def _init_from_base_params(self) -> CalibratedTheta:
        p = self.base_params
        return CalibratedTheta(
            vol=p.ewma_vol,
            drift=p.ewma_drift,
            mr_coeff=0.04,
            node_coeff=0.02,
            hurst_proxy=p.hurst_proxy,
            shock_prob=0.008,
            momentum_strength=0.6,
            momentum_decay=0.75,
            breakout_boost=0.4,
        )

    def _theta_to_vec(self, theta: CalibratedTheta) -> np.ndarray:
        return np.array([getattr(theta, k) for k in self.PARAM_KEYS])

    def _vec_to_theta(self, x: np.ndarray) -> CalibratedTheta:
        d = {k: float(v) for k, v in zip(self.PARAM_KEYS, x)}
        return CalibratedTheta(**d)
