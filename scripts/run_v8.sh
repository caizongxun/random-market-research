#!/usr/bin/env bash
# v10 參數
# 目標: 壓制前段衝太猛, 讓中段斜率更平穩持續
#
# 改動點 vs v9:
#   drift-scale     1.35 -> 1.15  (center_bias +2.85% 偏高, 整體下修)
#   drift-decay     0.02 -> 0.06  (加快衰減, 前段不衝, 後段靠 backbone 支撐)
#   momentum-boost  2.0  -> 1.6   (降低慣性, 前幾根不要連續爆衝)
#   backbone-mr     0.06 -> 0.12  (加強中段均值回歸力道, 讓路徑平穩貼著骨幹走)
#   anchor-weight   0.4  -> 0.45  (中段錨定力微增, last_seg_drift 持續拉動)
#   其餘不變
python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --lookback 120 --forecast 30 \
    --seed 42 --n-paths 500 \
    --backbone-mr 0.12 --n-seg 6 \
    --hist-window 60 --intra-bar 2 \
    --drift-decay 0.06 --drift-scale 1.15 --anchor-weight 0.45 \
    --vol-multiplier 1.2 --recent-vol-window 20 \
    --shadow-noise 0.15 --shadow-clamp 2.0 \
    --momentum-boost 1.6 --path-spread 1.0 \
    --output results/forward_aapl
