#!/bin/bash
set -e
PYTHON="${PYTHON:-/home/vt_ai_test1/mamba-envs/ml/bin/python}"
CP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CP"
"$PYTHON" EfficientPhysNet/evaluation/benchmark_recorded_cameras.py "$@"
