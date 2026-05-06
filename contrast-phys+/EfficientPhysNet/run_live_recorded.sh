#!/bin/bash
# 自录视频推理 -> results/EfficientPhysNet/label_ratio_0/live_runs/<时间戳>/
# 用法:
#   ./EfficientPhysNet/run_live_recorded.sh /path/to/video.avi /path/to/landmarks.csv
#   ./EfficientPhysNet/run_live_recorded.sh /path/to/video.avi ""   # 空第二参则用 OpenFace
set -e
PYTHON="${PYTHON:-python}"
CP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CP"
VIDEO="${1:?need video path}"
LM="${2:-}"
SCALES="${3:-10}"
if [ -n "$LM" ]; then
  $PYTHON EfficientPhysNet/live_recorded_infer.py --video "$VIDEO" --landmarks "$LM" --scales "$SCALES"
else
  $PYTHON EfficientPhysNet/live_recorded_infer.py --video "$VIDEO" --scales "$SCALES"
fi
