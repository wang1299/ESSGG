#!/bin/bash
set -euo pipefail

# 用法:
#   bash run_eval.sh <model_path> [conf_path] [num_episodes]
# 示例:
#   bash run_eval.sh /path/to/model.pth /home/wgy/RL/config 100
#   bash run_eval.sh /path/to/model_dir /home/wgy/RL/config 50

MODEL_PATH="${1:-}"
CONF_PATH="${2:-/home/wgy/RL/config}"
NUM_EPISODES="${3:-100}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Usage: bash run_eval.sh <model_path_or_dir> [conf_path] [num_episodes]"
  exit 1
fi

# 创建输出目录
mkdir -p /home/wgy/RL/eval_log
mkdir -p /home/wgy/RL/eval_png

# 环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/GroundingDINO"

# 时间戳与输出路径
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="eval_${TIMESTAMP}"
LOG_FILE="/home/wgy/RL/eval_log/${EXP_NAME}.log"
VIZ_DIR="/home/wgy/RL/eval_png/${EXP_NAME}"
mkdir -p "$VIZ_DIR"

echo "Starting evaluation..."
echo "Config path: $CONF_PATH"
echo "Model path: $MODEL_PATH"
echo "Episodes: $NUM_EPISODES"
echo "Log file: $LOG_FILE"
echo "Visualization images: $VIZ_DIR"

# 可选开关（需要关闭就改成 0）
USE_PRECOMPUTED=1
USE_DINO=1

CMD=(
  python RL_training/evaluate_model_weights.py
  --conf_path "$CONF_PATH"
  --model_path "$MODEL_PATH"
  --num_episodes "$NUM_EPISODES"
  --save_frames_to "$VIZ_DIR"
)

if [[ "$USE_PRECOMPUTED" == "1" ]]; then
  CMD+=(--precomputed)
fi

if [[ "$USE_DINO" == "1" ]]; then
  CMD+=(--use_dino)
fi

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

PID=$!
echo "Evaluation process started with PID: $PID"
echo "You can check logs with: tail -f $LOG_FILE"