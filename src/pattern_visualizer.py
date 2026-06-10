"""
pattern_visualizer.py

自動偵測並繪製隨機價格序列中出現的幾何型態。
核心論點：這些型態並非來自市場智慧，而是隨機路徑局部極值
幾何連線後的視覺副產品。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import argrelextrema
from typing import Optional


class PatternVisualizerAuto:
    """
    自動型態視覺化器。

    使用 scipy.signal.argrelextrema 偵測局部高低點，
    再以幾何規則匹配常見技術分析型態。

    Parameters
    ----------
    price_data : 1D ndarray，tick 或 close 價格序列
    order      : argrelextrema 的 order 參數，決定局部極值的「鄰域半徑"
    """

    def __init__(self, price_data: np.ndarray, order: int = 30):
        self.data = np.asarray(price_data)
        self.order = order
        self._local_max_idx: Optional[np.ndarray] = None
        self._local_min_idx: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def find_extrema(self) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (local_max_indices, local_min_indices)。"""
        self._local_max_idx = argrelextrema(self.data, np.greater, order=self.order)[0]
        self._local_min_idx = argrelextrema(self.data, np.less,    order=self.order)[0]
        return self._local_max_idx, self._local_min_idx

    def plot_base_chart(
        self,
        ax: plt.Axes,
        color: str = 'cyan',
        alpha: float = 0.8,
        label: str = 'Price',
        subsample: int = 1,
    ) -> None:
        """繪製基礎價格線。subsample 可降採樣加速大數據繪圖。"""
        idx = np.arange(0, len(self.data), subsample)
        ax.plot(idx, self.data[idx], color=color, lw=0.8, alpha=alpha, label=label)

    def find_and_draw_patterns(
        self,
        ax: plt.Axes,
        max_patterns: int = 8,
    ) -> list[dict]:
        """
        自動偵測並在 ax 上繪製型態標記。
        回傳已偵測到的型態列表。
        """
        if self._local_max_idx is None:
            self.find_extrema()

        detected = []

        # 1. Head & Shoulders
        hs = self._detect_head_and_shoulders(max_n=max_patterns // 2)
        for pat in hs:
            self._draw_head_and_shoulders(ax, pat)
            detected.append({'type': 'head_and_shoulders', **pat})

        # 2. Double Bottom / Double Top
        db = self._detect_double_bottom(max_n=max_patterns // 2)
        for pat in db:
            self._draw_double_bottom(ax, pat)
            detected.append({'type': 'double_bottom', **pat})

        # 3. 趨勢線（最高點連線 / 最低點連線）
        self._draw_trend_lines(ax)

        # 4. 局部極值標記點
        ax.scatter(
            self._local_max_idx, self.data[self._local_max_idx],
            color='red', s=15, zorder=5, alpha=0.6, label='Local High'
        )
        ax.scatter(
            self._local_min_idx, self.data[self._local_min_idx],
            color='lime', s=15, zorder=5, alpha=0.6, label='Local Low'
        )

        return detected

    # ------------------------------------------------------------------
    # 私有：型態偵測
    # ------------------------------------------------------------------

    def _detect_head_and_shoulders(
        self, max_n: int = 4
    ) -> list[dict]:
        """
        偵測頭肩頂：尋找連續 3 個局部高點，中間高點最高。
        條件：|left_shoulder - right_shoulder| / head < 0.05（對稱容忍度）
        """
        peaks = self._local_max_idx
        results = []
        if len(peaks) < 3:
            return results

        for i in range(len(peaks) - 2):
            l, h, r = peaks[i], peaks[i + 1], peaks[i + 2]
            lv, hv, rv = self.data[l], self.data[h], self.data[r]

            if hv > lv and hv > rv:
                symmetry = abs(lv - rv) / hv
                if symmetry < 0.08:
                    results.append({'l': l, 'h': h, 'r': r, 'lv': lv, 'hv': hv, 'rv': rv})
                    if len(results) >= max_n:
                        break
        return results

    def _detect_double_bottom(
        self, max_n: int = 4
    ) -> list[dict]:
        """
        偵測雙底：尋找連續 2 個局部低點，彼此接近。
        條件：|v1 - v2| / mean(v1, v2) < 0.03
        """
        troughs = self._local_min_idx
        results = []
        if len(troughs) < 2:
            return results

        for i in range(len(troughs) - 1):
            t1, t2 = troughs[i], troughs[i + 1]
            v1, v2 = self.data[t1], self.data[t2]
            mean_v = (v1 + v2) / 2
            if abs(v1 - v2) / mean_v < 0.03:
                results.append({'t1': t1, 't2': t2, 'v1': v1, 'v2': v2})
                if len(results) >= max_n:
                    break
        return results

    # ------------------------------------------------------------------
    # 私有：繪圖
    # ------------------------------------------------------------------

    def _draw_head_and_shoulders(
        self, ax: plt.Axes, pat: dict
    ) -> None:
        l, h, r = pat['l'], pat['h'], pat['r']
        lv, hv, rv = pat['lv'], pat['hv'], pat['rv']
        neckline = min(lv, rv) * 0.99

        ax.plot([l, h, r], [lv, hv, rv], 'o--', color='orange', lw=1.2, alpha=0.7, ms=5)
        ax.axhline(neckline, color='orange', lw=0.8, alpha=0.4, linestyle=':')
        ax.annotate(
            'H&S', xy=(h, hv),
            xytext=(h, hv * 1.015),
            color='orange', fontsize=7, ha='center',
            arrowprops=dict(arrowstyle='->', color='orange', lw=0.8)
        )

    def _draw_double_bottom(
        self, ax: plt.Axes, pat: dict
    ) -> None:
        t1, t2 = pat['t1'], pat['t2']
        v1, v2 = pat['v1'], pat['v2']

        ax.plot([t1, t2], [v1, v2], 's--', color='lime', lw=1.2, alpha=0.7, ms=5)
        ax.annotate(
            'Dbl Bot', xy=((t1 + t2) // 2, min(v1, v2)),
            xytext=((t1 + t2) // 2, min(v1, v2) * 0.985),
            color='lime', fontsize=7, ha='center',
        )

    def _draw_trend_lines(
        self, ax: plt.Axes, n_points: int = 3
    ) -> None:
        """連接最後 n_points 個高點與低點，畫出趨勢線。"""
        peaks = self._local_max_idx
        troughs = self._local_min_idx

        if len(peaks) >= n_points:
            px = peaks[-n_points:]
            py = self.data[px]
            coef = np.polyfit(px, py, 1)
            x_range = np.array([px[0], px[-1]])
            ax.plot(x_range, np.polyval(coef, x_range),
                    '--', color='red', lw=0.8, alpha=0.5, label='Resistance Trendline')

        if len(troughs) >= n_points:
            tx = troughs[-n_points:]
            ty = self.data[tx]
            coef = np.polyfit(tx, ty, 1)
            x_range = np.array([tx[0], tx[-1]])
            ax.plot(x_range, np.polyval(coef, x_range),
                    '--', color='lime', lw=0.8, alpha=0.5, label='Support Trendline')
