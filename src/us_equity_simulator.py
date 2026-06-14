"""
us_equity_simulator.py

以免費美股資料為輸入：
1. 拉取歷史 OHLCV
2. 從最近 500 根 K 棒估計隱含市場參數
3. 建立帶有 Volume Profile 記憶的未來隨機模擬
4. 產出多條未來路徑與分位帶

修正 v2（校準修正）：
- vol_scale 參數：直接放大 sigma，讓帶寬符合真實波動
- 使用 ewma_vol 取代 realized_vol 作為基礎波動率
- 使用 ewma_drift 取代簡單均值 drift
- mean-reversion 係數從 0.15 降至 0.04（減少路徑被壓制）
- volume node force 係數從 0.08 降至 0.02（減少路徑黏著節點）
- 加入學生 t 分佈尾部（df=5），模擬肥尾
- shock 機率與幅度微調
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import t as student_t

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
        vol_scale: float = 1.5,          # 校準用：放大 sigma 倍數
        use_ewma: bool = True,           # True=用 ewma_vol/ewma_drift，False=用歷史均值
        tail_df: int = 5,                # 學生 t 的自由度（越小尾部越肥，推薦 4~7）
        mr_coeff: float = 0.04,          # mean-reversion 係數（降低避免路徑被壓制）
        node_coeff: float = 0.02,        # volume node force 係數
    ):
        self.params         = params
        self.forecast_steps = forecast_steps
        self.n_paths        = n_paths
        self.rng            = np.random.default_rng(seed)
        self.vol_scale      = vol_scale
        self.use_ewma       = use_ewma
        self.tail_df        = tail_df
        self.mr_coeff       = mr_coeff
        self.node_coeff     = node_coeff

    def simulate(self) -> SimulationResult:
        paths = np.zeros((self.n_paths, self.forecast_steps))
        start = self.params.last_close

        # 選擇 vol 和 drift
        if self.use_ewma and hasattr(self.params, "ewma_vol"):
            base_vol   = self.params.ewma_vol   * self.vol_scale
            base_drift = self.params.ewma_drift
        else:
            base_vol   = self.params.realized_vol * self.vol_scale
            base_drift = self.params.drift

        upper_node, lower_node = self._nearest_nodes(start)

        # 預先生成學生 t 隨機數（比逐步快）
        # scipy student_t rvs 比較慢，改用 normal / chi2 手動合成
        # t(df) = Z / sqrt(chi2(df)/df)
        Z    = self.rng.standard_normal((self.n_paths, self.forecast_steps))
        chi2 = self.rng.chisquare(df=self.tail_df, size=(self.n_paths, self.forecast_steps))
        t_samples = Z / np.sqrt(chi2 / self.tail_df)  # shape: (n_paths, forecast_steps)

        for i in range(self.n_paths):
            price    = start
            prev_ret = 0.0
            for t in range(self.forecast_steps):
                noise      = t_samples[i, t] * base_vol
                trend_term = self.params.trend_strength * 0.3 * prev_ret
                mr_term    = -self.params.mean_reversion_strength * self.mr_coeff * (price / start - 1.0)
                gap_term   = self.rng.normal(0.0, self.params.gap_std * 0.15)
                node_term  = self._volume_node_force(price, base_vol)
                shock      = self._shock_component(base_vol)

                ret   = base_drift + noise + trend_term + mr_term + gap_term + node_term + shock
                price = max(0.01, price * (1.0 + ret))
                prev_ret = ret
                paths[i, t] = price

        median = np.median(paths, axis=0)
        p10    = np.percentile(paths, 10, axis=0)
        p25    = np.percentile(paths, 25, axis=0)
        p75    = np.percentile(paths, 75, axis=0)
        p90    = np.percentile(paths, 90, axis=0)

        breakout_prob_up   = 0.0
        breakout_prob_down = 0.0
        if upper_node is not None:
            breakout_prob_up   = float((paths[:, -1] > upper_node).mean())
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
        return (float(upper.min()) if len(upper) else None,
                float(lower.max()) if len(lower) else None)

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
        # 衝擊機率：大戶越少市場越容易出現跳動
        p = 0.008 + 0.025 * (1.0 - self.params.smart_money_ratio)
        if self.rng.random() < p:
            direction = 1.0 if self.rng.random() > 0.5 else -1.0
            return direction * abs(self.rng.normal(0.0, base_vol * 3.5))
        return 0.0
