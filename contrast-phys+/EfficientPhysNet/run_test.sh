#!/bin/bash
# EfficientPhysNet 推理 + 评估
# 用法: cd contrast-phys+ && ./EfficientPhysNet/run_test.sh [strategy] [time_interval] [save_viz]
# strategy: curriculum(默认) | equal | loss_prop | hybrid | inv_loss
# time_interval: 10(默认) | 20 | 30 | 60 秒
# save_viz: 1(默认) 保存波形图, 0 不保存
set -e
PYTHON="${PYTHON:-/home/vt_ai_test1/mamba-envs/ml/bin/python}"
STRATEGY="${1:-curriculum}"
TIME_INTERVAL="${2:-10}"
SAVE_VIZ="${3:-1}"
CP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CP"
echo "=== Test strategy: $STRATEGY, time_interval: ${TIME_INTERVAL}s ==="
$PYTHON EfficientPhysNet/test/test_epn.py with strategy=$STRATEGY time_interval=$TIME_INTERVAL
RUN_ROOT="$CP/results/EfficientPhysNet/label_ratio_0/inference/$STRATEGY/t${TIME_INTERVAL}"
LATEST_RUN="$(ls -1 "$RUN_ROOT" 2>/dev/null | awk '/^[0-9]+$/ {print $0}' | sort -n | awk 'NF{v=$1} END{print v}')"
PRED_DIR=""
if [ -n "$LATEST_RUN" ]; then
  PRED_DIR="$RUN_ROOT/$LATEST_RUN"
fi

if [ -d "$PRED_DIR" ] && [ -n "$(ls -A "$PRED_DIR"/*.npy 2>/dev/null)" ]; then
  echo "=== Evaluate ==="
  if [ "$SAVE_VIZ" = "1" ]; then
    $PYTHON EfficientPhysNet/evaluation/evaluate.py "$PRED_DIR" --save-viz
  else
    $PYTHON EfficientPhysNet/evaluation/evaluate.py "$PRED_DIR"
  fi
else
  echo "Pred dir not found or empty, run evaluate manually:"
  echo "  python EfficientPhysNet/evaluation/evaluate.py results/EfficientPhysNet/label_ratio_0/inference/$STRATEGY/t${TIME_INTERVAL}/<latest_run>"
fi
