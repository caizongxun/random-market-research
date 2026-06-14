#!/usr/bin/env bash
# v12 參數
# 目標: 修復中段低估問題 (hit_rate_25_75 卡在 0.2)
#
# 問題診斷:
#   end_error=0.98%, center_bias=-1% 終點對, 但中段就已超出 p75
#   原因: drift-decay=0.06 表示 t15 就只剩 41%, 中段 drift 已大幅衰減
#   解法: 降低 drift-decay 讓中段走勢持續, 同時小幅降 drift-scale 补偿終點
#
# 改動點 vs v11:
#   drift-decay  0.06 -> 0.04  (t15 殘餘: 55% vs 41%, 中段更高)
#   drift-scale  1.22 -> 1.18  (微降补偿終點, 不讓最終超出太多)
python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --lookback 120 --forecast 30 \
    --seed 42 --n-paths 500 \
    --backbone-mr 0.12 --n-seg 6 \
    --hist-window 60 --intra-bar 2 \
    --drift-decay 0.04 --drift-scale 1.18 --anchor-weight 0.45 \
    --vol-multiplier 1.2 --recent-vol-window 20 \
    --shadow-noise 0.15 --shadow-clamp 2.0 \
    --momentum-boost 1.6 --path-spread 1.0 \
    --output results/forward_aapl
