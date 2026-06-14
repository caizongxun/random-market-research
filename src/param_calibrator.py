"""
param_calibrator.py  v2

修正項目：
1. 模擬起始價格改為 history_close[0]（對齊起點）
2. Loss Function 新增 trend_match：懲罰三段區間漲跌方向是否一致
3. 新增 end_price_match：懲罰終點價格差距

Loss Function：
  L(θ) = λ_vol      * |σ_sim - σ_real| / σ_real
        + λ_acf      * Σ|acf_sim - acf_real|
        + λ_drawdown * |dd_sim - dd_real| / |dd_real|
        + λ_node     * |bounce_sim - bounce_real|
        + λ_skew     * |skew_sim - skew_real|
        + λ_trend    * 區間方向不一致率
        + λ_end      * |end_sim - end_real| / end_real
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
    vol:      float = 1.0   # 波動大小
    acf:      float = 0.8   # 自相關/記憶性
    drawdown: float = 0.6   # 最大回撤
    node:     float = 0.5   # volume node 反彈行為
    skew:     float = 0.4   # 回報偏態
    trend:    float = 1.5   # 區間方向懲罰（新增）
    end:      float = 1.2   # 終點價格懲罰（新增）


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
        maxiter: int = 400,
        verbose: bool = True,
    ):
        self.base_params   = base_params
        self.history_close = np.asarray(history_close, dtype=float)
        self.weights       = weights or LossWeights()
        self.n_sim_paths   = n_sim_paths
        self.n_sim_steps   = len(self.history_close)
        self.seed          = seed
        self.maxiter       = maxiter
        self.verbose       = verbose

        # 模擬起始價格：對齊歷史第一根
        self.start_price   = float(self.history_close[0])

        self._real_stats   = self._compute_stats(self.history_close)
        self._real_trend   = self._segment_trend(self.history_close)
        self._real_end     = float(self.history_close[-1])

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
            objective, x0,
            method="Nelder-Mead",
            options={"maxiter": self.maxiter, "xatol": 1e-5,
                     "fatol": 1e-5, "adaptive": True},
        )
        best = self._vec_to_theta(np.clip(result.x, lo, hi))
        if self.verbose:
            print(f"  Done. loss={result.fun:.6f}  iters={result.nit}")
        return best

    # ------------------------------------------------------------------ #
    def _loss(self, theta: CalibratedTheta) -> float:
        paths = self._run_sim(theta)   # (n_paths, n_steps)
        w     = self.weights
        total = 0.0
        T     = paths.shape[0]

        stats_list  = [self._compute_stats(paths[i]) for i in range(T)]
        trends_list = [self._segment_trend(paths[i]) for i in range(T)]
        ends_list   = [paths[i, -1] for i in range(T)]

        def med(key):
            return float(np.median([s[key] for s in stats_list]))

        r = self._real_stats

        # 波動
        total += w.vol * abs(med("vol") - r["vol"]) / (r["vol"] + 1e-8)
        # ACF
        for lag in range(1, 6):
            total += w.acf * 0.2 * abs(med(f"acf_{lag}") - r[f"acf_{lag}"])
        # 最大回撤
        total += w.drawdown * abs(med("max_dd") - r["max_dd"]) / (abs(r["max_dd"]) + 1e-8)
        # 偏態
        total += w.skew * abs(med("skew") - r["skew"]) / (abs(r["skew"]) + 1e-4)

        # Volume node 反彈
        if self.base_params.volume_nodes:
            real_ns = self._node_behavior_score(
                self.history_close, self.base_params.volume_nodes
            )
            sim_ns = float(np.median([
                self._node_behavior_score(paths[i], self.base_params.volume_nodes)
                for i in range(T)
            ]))
            total += w.node * abs(sim_ns - real_ns)

        # 區間方向懲罰：對每條路徑算三段方向不對數，取中位
        # 真實走勢有 3 段，每條路徑與其比較
        rt = self._real_trend  # [d0, d1, d2] 方向
        mismatch_rates = []
        for path in paths:
            pt = self._segment_trend(path)
            mismatches = sum(1 for a, b in zip(rt, pt) if np.sign(a) != np.sign(b))
            mismatch_rates.append(mismatches / 3.0)
        total += w.trend * float(np.median(mismatch_rates))

        # 終點價格懲罰
        sim_end_med = float(np.median(ends_list))
        total += w.end * abs(sim_end_med - self._real_end) / (self._real_end + 1e-8)

        return float(total)

    def _run_sim(self, theta: CalibratedTheta) -> np.ndarray:
        import dataclasses
        from calibrated_simulator import build_params_from_theta

        base = dataclasses.replace(
            self.base_params,
            last_close=self.start_price,  # ★ 對齊起點
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
    def _segment_trend(prices: np.ndarray, n_seg: int = 3) -> list[float]:
        """
        將路徑分成 n_seg 段，回傳每段的結尾價差
        正數 = 上漲段，負數 = 下跌段
        """
        n = len(prices)
        seg = n // n_seg
        diffs = []
        for i in range(n_seg):
            s = i * seg
            e = s + seg if i < n_seg - 1 else n
            diffs.append(float(prices[e - 1] - prices[s]))
        return diffs

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
        # drift 初學失用真實路徑方向估算
        real_drift = float(
            np.log(self.history_close[-1] / self.history_close[0])
            / max(len(self.history_close), 1)
        )
        return CalibratedTheta(
            vol=p.ewma_vol,
            drift=np.clip(real_drift, -0.003, 0.003),
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
        return CalibratedTheta(**{k: float(v) for k, v in zip(self.PARAM_KEYS, x)})
