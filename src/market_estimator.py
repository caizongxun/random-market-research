"""
market_estimator.py

從歷史 OHLCV 估計市場隱含參數，供未來路徑模擬使用。

修正 v2（校準修正）：
- drift 改用 EWMA（近期權重更高），避免長期歷史稀釋近期趨勢
- 新增 ewma_drift_span 參數（預設 60 根）
- 新增 realized_vol_span：用 EWMA 估計的波動率，對近期波動更敏感
- MarketParams 新增 ewma_drift、ewma_vol 欄位
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.signal import find_peaks


@dataclass
class MarketParams:
    symbol: str
    last_close: float
    realized_vol: float        # 簡單標準差（全 lookback）
    ewma_vol: float            # EWMA 波動率（近期敏感）
    drift: float               # 簡單均值 drift
    ewma_drift: float          # EWMA 加權 drift（近期趨勢）
    hurst_proxy: float
    avg_range_ratio: float
    gap_std: float
    volume_nodes: list[float]
    volume_node_strength: list[float]
    trend_strength: float
    mean_reversion_strength: float
    smart_money_ratio: float


class MarketParameterEstimator:
    def __init__(
        self,
        lookback: int = 500,
        vp_bins: int = 40,
        ewma_drift_span: int = 60,
        ewma_vol_span: int = 30,
    ):
        self.lookback = lookback
        self.vp_bins = vp_bins
        self.ewma_drift_span = ewma_drift_span
        self.ewma_vol_span = ewma_vol_span

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

        # 全期 realized vol & drift
        realized_vol = float(log_ret.std())
        drift        = float(log_ret.mean())

        # EWMA vol（對近期波動更敏感）
        ewma_var = log_ret.ewm(span=self.ewma_vol_span, adjust=False).var()
        ewma_vol = float(np.sqrt(ewma_var.iloc[-1]))
        # 安全下限：至少是全期 vol 的 50%
        ewma_vol = max(ewma_vol, realized_vol * 0.5)

        # EWMA drift（近期趨勢加權）
        ewma_drift = float(
            log_ret.ewm(span=self.ewma_drift_span, adjust=False).mean().iloc[-1]
        )

        hurst_proxy    = self._hurst_proxy(close.values)
        avg_range_ratio = float(((high - low) / close).mean())
        gap_std        = float(((open_ - close.shift(1)) / close.shift(1)).dropna().std())

        volume_nodes, node_strength = self._volume_profile_nodes(close.values, volume.values)

        trend_strength          = max(0.0, hurst_proxy - 0.5) * 2.0
        mean_reversion_strength = max(0.0, 0.5 - hurst_proxy) * 2.0

        smart_money_ratio = float(np.clip(
            0.35
            + 0.8  * mean_reversion_strength
            - 0.5  * trend_strength
            + 0.2  * (1.0 - min(avg_range_ratio / 0.05, 1.0)),
            0.05,
            0.95,
        ))

        return MarketParams(
            symbol=symbol,
            last_close=float(close.iloc[-1]),
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
        )

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
