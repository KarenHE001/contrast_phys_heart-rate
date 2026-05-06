#!/bin/bash
# 后台训练 PhysNet 2D（tmux + log）
# 用法: ./run_train_background.sh [0|1] [test]
#   0=unsupervised, 1=supervised
#   第二参数 test=小数据快速验证（4 train, 3 epoch）
# 关 terminal/SSH 后训练继续，tmux attach -t phys2d_0 可查看

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LR=${1:-0}
TEST_MODE=${2:-}
LOG_DIR="$ROOT/PhysNet 2D/results/training_logs"
mkdir -p "$LOG_DIR/label_ratio_$LR"
LOG_FILE="$LOG_DIR/label_ratio_$LR/train_$(date +%Y%m%d_%H%M%S).log"
PYTHON=/home/vt_ai_test1/mamba-envs/ml/bin/python
TMUX_NAME="phys2d_$LR"

if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
    echo "会话 $TMUX_NAME 已存在，请先 tmux kill-session -t $TMUX_NAME"
    exit 1
fi

EXTRA=""
if [ "$TEST_MODE" = "test" ]; then
    EXTRA="test_mode=True total_epoch=3"
    echo "⚠️  测试模式: 4 训练文件, 3 epoch"
fi

echo "启动训练: label_ratio=$LR"
echo "  tmux 会话: $TMUX_NAME"
echo "  log: $LOG_FILE"
echo "  查看: tmux attach -t $TMUX_NAME"
echo "  分离: Ctrl+B D"

tmux new-session -d -s "$TMUX_NAME" "cd '$ROOT' && $PYTHON 'PhysNet 2D/train.py' with label_ratio=$LR $EXTRA 2>&1 | tee '$LOG_FILE'"
