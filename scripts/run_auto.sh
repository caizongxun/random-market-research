#!/usr/bin/env bash
# run_auto.sh  v9 — GJR-GARCH + auto-calibrate
# 用法:
#   bash scripts/run_auto.sh
#   bash scripts/run_auto.sh --garch-model egarch
#   bash scripts/run_auto.sh --no-garch
#   SYMBOL=TSLA bash scripts/run_auto.sh
#   THETA=/path/to/theta.json bash scripts/run_auto.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR" rev-parse --show-toplevel &>/dev/null; then
  ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
  ROOT="$(dirname "$SCRIPT_DIR")"
fi

SYMBOL="${SYMBOL:-AAPL}"
SYM_LOWER="${SYMBOL,,}"
FORECAST="${FORECAST:-30}"
LOOKBACK="${LOOKBACK:-120}"
N_PATHS="${N_PATHS:-500}"
CALIB_WINDOW="${CALIB_WINDOW:-500}"
OUTPUT_DIR="$ROOT/results"

# ── 自動偵測 theta 檔位置（可用環境變數 THETA 覆蓋） ──
if [ -z "${THETA:-}" ]; then
  CANDIDATES=(
    "$ROOT/theta_${SYM_LOWER}.json"
    "$ROOT/calibrated_theta_${SYM_LOWER}.json"
    "$ROOT/theta/${SYM_LOWER}_theta.json"
    "$ROOT/theta/aapl_theta.json"
    "$ROOT/results/theta_${SYM_LOWER}.json"
  )
  for candidate in "${CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
      THETA="$candidate"
      break
    fi
  done
fi

if [ -z "${THETA:-}" ]; then
  echo "❌ 找不到 theta 檔，請手動指定："
  echo "   THETA=/path/to/theta.json bash scripts/run_auto.sh"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
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
  --output       "$OUTPUT_DIR/forward_${SYM_LOWER}" \
  "$@"
