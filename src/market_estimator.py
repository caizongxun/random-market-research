"""
market_estimator.py

從歷史 OHLCV 估計市場隱含參數，供未來路徑模擬使用。

v3 新增：
- MarketParams 新增 momentum_bias：最近 N 根的平均日對數回報（帶方向偏向）
- MarketParams 新增 node_breakout_state：价格相對最近上方 volume node 的狀態
  +1 = 剛突破上方節點（漲势帶鍵）
  -1 = 剛被上方節點拒絕（若候骎线）
   0 = 中立
- 新增 vol_trend：波動率是在擴大還是收縮（EWMA vol 相對全期 vol 的比値）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import find_peaks


@dataclass
class MarketParams:
    symbol: str
    last_close: float
    realized_vol: float
    ewma_vol: float
    drift: float
    ewma_drift: float
    hurst_proxy: float
    avg_range_ratio: float
    gap_std: float
    volume_nodes: list[float]
    volume_node_strength: list[float]
    trend_strength: float
    mean_reversion_strength: float
    smart_money_ratio: float
    # v3 新增
    momentum_bias: float        # 近期 N 根平均日對數回報（帶方向）
    node_breakout_state: int    # +1 突破 / -1 被拒 / 0 中立
    vol_trend: float            # ewma_vol / realized_vol：>1 波動放大中
    recent_returns: list[float] # 最近 momentum_window 根的日對數回報


class MarketParameterEstimator:
    def __init__(
        self,
        lookback: int = 500,
        vp_bins: int = 40,
        ewma_drift_span: int = 60,
        ewma_vol_span: int = 30,
        momentum_window: int = 10,
    ):
        self.lookback        = lookback
        self.vp_bins         = vp_bins
        self.ewma_drift_span = ewma_drift_span
        self.ewma_vol_span   = ewma_vol_span
        self.momentum_window = momentum_window

    def fit(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> MarketParams:
        data = df.copy().tail(self.lookback)
        data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if len(data) < 100:
            raise ValueError("Not enough data to estimate market parameters.")

        close  = data["Close"].astype(float)
        high   = data["High"].astype(float)
        low    = data["Low"].astype(float)
        open_  = data["Open"].astype(float)
        volume = data["Volume"].astype(float)

        log_ret = np.log(close / close.shift(1)).dropna()

        realized_vol = float(log_ret.std())
        drift        = float(log_ret.mean())

        ewma_var = log_ret.ewm(span=self.ewma_vol_span, adjust=False).var()
        ewma_vol = float(np.sqrt(ewma_var.iloc[-1]))
        ewma_vol = max(ewma_vol, realized_vol * 0.5)

        ewma_drift = float(
            log_ret.ewm(span=self.ewma_drift_span, adjust=False).mean().iloc[-1]
        )

        hurst_proxy     = self._hurst_proxy(close.values)
        avg_range_ratio = float(((high - low) / close).mean())
        gap_std         = float(((open_ - close.shift(1)) / close.shift(1)).dropna().std())

        volume_nodes, node_strength = self._volume_profile_nodes(close.values, volume.values)

        trend_strength          = max(0.0, hurst_proxy - 0.5) * 2.0
        mean_reversion_strength = max(0.0, 0.5 - hurst_proxy) * 2.0

        smart_money_ratio = float(np.clip(
            0.35
            + 0.8  * mean_reversion_strength
            - 0.5  * trend_strength
            + 0.2  * (1.0 - min(avg_range_ratio / 0.05, 1.0)),
            0.05, 0.95,
        ))

        # --- v3: momentum_bias ---
        recent_log = log_ret.iloc[-self.momentum_window:]
        momentum_bias  = float(recent_log.mean())       # 近期日平均日回報
        recent_returns = recent_log.tolist()

        # --- v3: node_breakout_state ---
        last_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2]) if len(close) > 1 else last_price
        node_breakout_state = self._node_breakout(prev_price, last_price, volume_nodes)

        # --- v3: vol_trend ---
        vol_trend = float(ewma_vol / (realized_vol + 1e-12))

        return MarketParams(
            symbol=symbol,
            last_close=last_price,
            realized_vol=realized_vol,
            ewma_vol=ewma_vol,
            drift=drift,
            ewma_drift=ewma_drift,
            hurst_proxy=float(hurst_proxy),
            avg_range_ratio=avg_range_ratio,
            gap_std=gap_std,
            volume_nodes=volume_nodes,
            volume_node_strength=node_strength,
            trend_strength=float(np.clip(trend_strength, 0.0, 1.0)),
            mean_reversion_strength=float(np.clip(mean_reversion_strength, 0.0, 1.0)),
            smart_money_ratio=smart_money_ratio,
            momentum_bias=momentum_bias,
            node_breakout_state=node_breakout_state,
            vol_trend=vol_trend,
            recent_returns=recent_returns,
        )

    def _node_breakout(self, prev: float, curr: float, nodes: list[float]) -> int:
        """
        +1: 就在上一根突破了上方 node
        -1: 就在上一根被上方 node 拒絕（融了下來）
         0: 中立
        """
        if not nodes:
            return 0
        nodes_arr = np.array(nodes)
        upper = nodes_arr[nodes_arr > min(prev, curr)]
        if len(upper) == 0:
            return 0
        nearest_upper = float(upper.min())
        # 突破：prev 在 node 以下，curr 在 node 以上
        if prev < nearest_upper <= curr:
            return 1
        # 被拒：prev 在 node 以上，curr 跌回 node 以下
        if prev >= nearest_upper > curr:
            return -1
        return 0

    def _hurst_proxy(self, series: np.ndarray, max_lag: int = 40) -> float:
        lags = range(2, max_lag)
        tau  = []
        for lag in lags:
            diff = series[lag:] - series[:-lag]
            tau.append(np.std(diff) + 1e-12)
        slope = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
        return float(np.clip(slope, 0.0, 1.0))

    def _volume_profile_nodes(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> tuple[list[float], list[float]]:
        counts, bin_edges = np.histogram(prices, bins=self.vp_bins, weights=volumes)
        peaks, _ = find_peaks(counts, distance=2)
        if len(peaks) == 0:
            idx     = int(np.argmax(counts))
            centers = [float((bin_edges[idx] + bin_edges[idx + 1]) / 2)]
            return centers, [1.0]
        centers     = [float((bin_edges[i] + bin_edges[i + 1]) / 2) for i in peaks]
        peak_values = counts[peaks].astype(float)
        peak_values = peak_values / (peak_values.max() + 1e-12)
        return centers, peak_values.tolist()
