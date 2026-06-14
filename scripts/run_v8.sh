#!/usr/bin/env bash
# v9 參數（檔名保留 run_v8.sh 不變、直接覆蓋）
#
# 改動點 vs v8：
#   drift-scale     1.30 -> 1.35  (再微向上偏移, center_bias -0.54%)
#   intra-bar          3 ->    2  (K 棒實體縮小, 方向更集中)
#   vol-multiplier   1.5 ->  1.2  (降低每步隨機噯，讓路徑更有方向性)
#   momentum-boost   1.5 ->  2.0  (加大連續高低慧性、提升逐日方向一致性)
#   drift-decay     0.03 -> 0.02  (t30 殘餘: 55%, drift 持續更久)
#   其餘不變
python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --lookback 120 --forecast 30 \
    --seed 42 --n-paths 500 \
    --backbone-mr 0.06 --n-seg 6 \
    --hist-window 60 --intra-bar 2 \
    --drift-decay 0.02 --drift-scale 1.35 --anchor-weight 0.4 \
    --vol-multiplier 1.2 --recent-vol-window 20 \
    --shadow-noise 0.15 --shadow-clamp 2.0 \
    --momentum-boost 2.0 --path-spread 1.0 \
    --output results/forward_aapl
