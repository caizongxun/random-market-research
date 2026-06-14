#!/usr/bin/env bash
# run_auto.sh  v9 — GJR-GARCH + auto-calibrate
# 用法:
#   bash scripts/run_auto.sh
#   bash scripts/run_auto.sh --garch-model egarch
#   bash scripts/run_auto.sh --no-garch
#   bash scripts/run_auto.sh --momentum-boost 1.5 --drift-scale 2.0
#
# 環境變數覆蓋:
#   SYMBOL=TSLA FORECAST=20 bash scripts/run_auto.sh

set -e

# ── Repo root：無論從哪個目錄呼叫都能找到正確位置 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 優先用 git 找 repo root；若不在 git repo 裡則退回 SCRIPT_DIR 的上層
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel &>/dev/null; then
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  ROOT="$(dirname "$SCRIPT_DIR")"
fi

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
echo "  Root    : $ROOT"
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
