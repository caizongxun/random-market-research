#!/usr/bin/env bash
# v11 參數
# 目標: 純平移上移, 不動形狀
#
# 改動點 vs v10:
#   drift-scale  1.15 -> 1.22  (整體上移 ~1.1%, end_error 前已只剩 1.12%)
#   其餘全部不變
python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --lookback 120 --forecast 30 \
    --seed 42 --n-paths 500 \
    --backbone-mr 0.12 --n-seg 6 \
    --hist-window 60 --intra-bar 2 \
    --drift-decay 0.06 --drift-scale 1.22 --anchor-weight 0.45 \
    --vol-multiplier 1.2 --recent-vol-window 20 \
    --shadow-noise 0.15 --shadow-clamp 2.0 \
    --momentum-boost 1.6 --path-spread 1.0 \
    --output results/forward_aapl
