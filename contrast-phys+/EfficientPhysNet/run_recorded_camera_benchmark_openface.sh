#!/bin/bash
set -e
PYTHON="${PYTHON:-/home/vt_ai_test1/mamba-envs/ml/bin/python}"
CP="$(cd "$(dirname "$0")/.." && pwd)"
OPENFACE_ENV_LIB="/home/vt_ai_test1/miniconda3/envs/openface_env/lib"
OPENFACE_DIR_DEFAULT="/home/vt_ai_test1/KarenHE/contrast-phys/OpenFace/build_openface_env"
if [ -d "$OPENFACE_ENV_LIB" ]; then
  export LD_LIBRARY_PATH="$OPENFACE_ENV_LIB:${LD_LIBRARY_PATH:-}"
fi
ARGS=("$@")
HAS_OPENFACE_DIR=0
for arg in "${ARGS[@]}"; do
  if [ "$arg" = "--openface-dir" ]; then
    HAS_OPENFACE_DIR=1
    break
  fi
done
if [ "$HAS_OPENFACE_DIR" -eq 0 ] && [ -d "$OPENFACE_DIR_DEFAULT" ]; then
  ARGS+=("--openface-dir" "$OPENFACE_DIR_DEFAULT")
fi
cd "$CP"
"$PYTHON" EfficientPhysNet/evaluation/benchmark_recorded_cameras.py "${ARGS[@]}"
