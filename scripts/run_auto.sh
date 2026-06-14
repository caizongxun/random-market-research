#!/usr/bin/env bash
# run_auto.sh  v9 — GJR-GARCH + auto-calibrate
# 用法:
#   bash scripts/run_auto.sh
#   bash scripts/run_auto.sh --garch-model egarch
#   bash scripts/run_auto.sh --no-garch          # 退回靜態 rv/theta
#   bash scripts/run_auto.sh --momentum-boost 1.5 --drift-scale 2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

SYMBOL="${SYMBOL:-AAPL}"
THETA="${THETA:-$ROOT/theta/aapl_theta.json}"
FORECAST="${FORECAST:-30}"
LOOKBACK="${LOOKBACK:-120}"
N_PATHS="${N_PATHS:-500}"
CALIB_WINDOW="${CALIB_WINDOW:-500}"
OUTPUT_DIR="$ROOT/results"

mkdir -p "$OUTPUT_DIR"

# 安裝 arch（如尚未安裝）
pip install arch --quiet 2>/dev/null || true

echo "======================================"
echo "  Forward Study v9 (GJR-GARCH)"
echo "  Symbol  : $SYMBOL"
echo "  Theta   : $THETA"
echo "  Window  : $CALIB_WINDOW  Forecast: $FORECAST"
echo "======================================"

python "$SCRIPT_DIR/forward_study.py" \
  --symbol       "$SYMBOL" \
  --theta        "$THETA" \
  --forecast     "$FORECAST" \
  --lookback     "$LOOKBACK" \
  --n-paths      "$N_PATHS" \
  --auto-calibrate \
  --calib-window "$CALIB_WINDOW" \
  --output       "$OUTPUT_DIR/forward_${SYMBOL,,}" \
  "$@"
