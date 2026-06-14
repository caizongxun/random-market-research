#!/usr/bin/env bash
# run_auto.sh — 全自動校準模式
# 不需要手動指定任何模擬參數，全部由前 500 根 K 棒決定
#
# 如果想覆蓋某個參數，直接加上去即可，例如:
#   bash scripts/run_auto.sh --drift-scale 1.3

python scripts/forward_study.py \
    --symbol AAPL \
    --theta results/theta_aapl.json \
    --lookback 120 --forecast 30 \
    --seed 42 --n-paths 500 \
    --backbone-mr 0.12 --n-seg 6 \
    --hist-window 60 \
    --anchor-weight 0.45 \
    --recent-vol-window 20 \
    --auto-calibrate \
    --calib-window 500 \
    --path-spread 1.0 \
    --output results/forward_aapl \
    "$@"
