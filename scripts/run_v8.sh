#!/usr/bin/env bash
# v8 推薦參數
# 改動點 vs v7:
#   drift-scale   1.05 -> 1.30   (中心線上移, v7 log: center_bias -3.67%)
#   intra-bar        2 ->    3   (K 棒實體稍微縮小, avg_body_pct 2.04% 偏大)
#   drift-decay   0.05 -> 0.03   (drift 持續更久, t30 殘餘: 40% vs 22%)
#   其餘不變
python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --lookback 120 --forecast 30 \
    --seed 42 --n-paths 500 \
    --backbone-mr 0.06 --n-seg 6 \
    --hist-window 60 --intra-bar 3 \
    --drift-decay 0.03 --drift-scale 1.30 --anchor-weight 0.4 \
    --vol-multiplier 1.5 --recent-vol-window 20 \
    --shadow-noise 0.15 --shadow-clamp 2.0 \
    --momentum-boost 1.5 --path-spread 1.0 \
    --output results/forward_aapl
