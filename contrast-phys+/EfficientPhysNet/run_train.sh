#!/bin/bash
# EfficientPhysNet 多尺度训练 — 需在 contrast-phys+ 目录运行
# 用法: cd contrast-phys+ && ./EfficientPhysNet/run_train.sh [input_size] [weight_strategy]
# 例: ./EfficientPhysNet/run_train.sh 96 equal
set -e
PYTHON="${PYTHON:-/home/vt_ai_test1/mamba-envs/ml/bin/python}"
INPUT="${1:-96}"
STRATEGY="${2:-equal}"
CP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CP"
echo "=== Train input_size=$INPUT weight_strategy=$STRATEGY ==="
$PYTHON EfficientPhysNet/train/train_multiscale.py with input_size=$INPUT weight_strategy=$STRATEGY
