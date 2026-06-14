"""
backbone_fitter.py

Phase 1：確定性骨幹路徑擬合

概念：
  真實 K 棒可以分解為「骨幹走勢（低頻）+ 隨機雜訊（高頻）」。
  骨幹擬合的目標：找到一組分段漂移率 [d0, d1, ..., dk]，
  使得用這些漂移率生成的確定性路徑（無隨機項）和真實 close 最接近。

方法：
  - 把 lookback 分成 n_seg 段
  - 每段的漂移率用最小二乘法擬合（最小化 MSE）
  - 輸出骨幹路徑（與真實 close 等長，同起點）

用途：
  骨幹路徑給 Phase 2（ParamCalibrator）當 target，
  讓隨機帶子的中線跟著骨幹走。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackboneResult:
    backbone: np.ndarray        # 骨幹路徑，shape (n,)
    segment_drifts: np.ndarray  # 每段漂移率
    segment_vols: np.ndarray    # 每段的殘差波動率（給 Phase 2 參考）
    n_seg: int
    fit_mse: float


class BackboneFitter:
    """
    最小二乘分段漂移擬合器。

    Args:
        n_seg:      分幾段（建議 3-8，太多會過擬合）
        smooth_reg: 相鄰段漂移差距的平滑懲罰，避免相鄰段漂移差太大
    """

    def __init__(self, n_seg: int = 6, smooth_reg: float = 0.5):
        self.n_seg      = n_seg
        self.smooth_reg = smooth_reg

    def fit(self, close: np.ndarray) -> BackboneResult:
        close  = np.asarray(close, dtype=float)
        n      = len(close)
        start  = close[0]
        log_r  = np.diff(np.log(close))   # 對數回報，shape (n-1,)

        # 建立分段指標
        seg_len   = (n - 1) // self.n_seg
        seg_idx   = np.zeros(n - 1, dtype=int)
        for s in range(self.n_seg):
            lo = s * seg_len
            hi = lo + seg_len if s < self.n_seg - 1 else n - 1
            seg_idx[lo:hi] = s

        def build_backbone(drifts):
            """給定每段漂移率，建出對數累積路徑，再還原成價格"""
            per_step = drifts[seg_idx]    # 每步的漂移
            log_path = np.concatenate([[0.0], np.cumsum(per_step)])
            return start * np.exp(log_path)

        def objective(drifts):
            backbone = build_backbone(drifts)
            mse      = float(np.mean((backbone - close) ** 2)) / (start ** 2)
            # 平滑懲罰：相鄰段漂移差距
            smooth   = float(np.sum(np.diff(drifts) ** 2)) * self.smooth_reg
            return mse + smooth

        # 初始值：用整體平均漂移
        mean_drift = float(np.log(close[-1] / close[0])) / (n - 1)
        x0 = np.full(self.n_seg, mean_drift)

        result = minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-7, "adaptive": True},
        )
        drifts   = result.x
        backbone = build_backbone(drifts)

        # 計算每段殘差波動率
        resid     = np.diff(np.log(close)) - drifts[seg_idx]
        seg_vols  = np.array([
            float(np.std(resid[seg_idx == s])) for s in range(self.n_seg)
        ])

        return BackboneResult(
            backbone=backbone,
            segment_drifts=drifts,
            segment_vols=seg_vols,
            n_seg=self.n_seg,
            fit_mse=result.fun,
        )


def plot_backbone(
    close: np.ndarray,
    result: BackboneResult,
    title: str = "Backbone Fit",
    ax=None,
    show: bool = True,
):
    """快速畫骨幹擬合結果（可選：傳入 ax 嵌入到其他圖）"""
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0e0e0e")
        ax.set_facecolor("#0e0e0e")
        plt.style.use("dark_background")

    x = np.arange(len(close))
    ax.plot(x, close,            color="white",  lw=1.5, label="Real close", alpha=0.9)
    ax.plot(x, result.backbone,  color="#ff9900", lw=2.2, label="Backbone",   alpha=0.9)

    # 分段邊界線
    seg_len = len(close) // result.n_seg
    for s in range(1, result.n_seg):
        ax.axvline(s * seg_len, color="#555", lw=1, linestyle="--", alpha=0.6)

    # 標每段漂移
    for s in range(result.n_seg):
        lo = s * seg_len
        hi = lo + seg_len if s < result.n_seg - 1 else len(close)
        mid_x = (lo + hi) // 2
        d     = result.segment_drifts[s]
        color = "#66ff66" if d > 0 else "#ff6666"
        ax.text(
            mid_x, result.backbone[mid_x],
            f"{d*100:+.3f}%/bar",
            color=color, fontsize=7.5, ha="center", va="bottom",
        )

    ax.set_title(title, color="white", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
              edgecolor="#333", labelcolor="white")
    ax.tick_params(colors="#888")

    if standalone and show:
        plt.tight_layout()
        plt.show()
        return None
    return ax
