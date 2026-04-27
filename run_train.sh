#!/bin/bash

# 创建输出目录
mkdir -p /home/wgy/RL/train_log
mkdir -p /home/wgy/RL/train_png

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:$(pwd):/home/wgy/GroundingDINO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optional multi-GPU for heavy RGB encoder forward.
# Usings the last 4 cards: GPU 4 is for GroundingDINO, GPUs 5,6,7 are for policy encoder
export RL_GPU_IDS=${RL_GPU_IDS:-"5,6,7"}
export DINO_DEVICE=${DINO_DEVICE:-"cuda:4"}

# 获取当前时间戳作为 Log 文件名的一部分
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="train_${TIMESTAMP}"
LOG_FILE="/home/wgy/RL/train_log/${EXP_NAME}.log"
VIZ_DIR="/home/wgy/RL/train_png/${EXP_NAME}"

# 创建本次训练的可视化目录
mkdir -p "$VIZ_DIR"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "Visualization images: $VIZ_DIR"

# 运行命令
# 使用 nohup 后台运行
# 注意：--save_frames_to 指向本次训练的专属目录
# [Update] User requested HM3D scenes
HABITAT_SCENES="00016-qk9eeNeR4vw,00017-oEPjPNSPmzL,00023-zepmXAdrpjR,00031-Wo6kuutE9i7,00033-oPj9qMxrDEa,00087-YY8rqV6L6rf,00099-226REUyJh2K,00105-xWvSkKiWQpC,00108-oStKKWkQ1id,00155-iLDo95ZbDJq,00166-RaYrxWt5pR1,00177-VSxVP19Cdyw,00210-j2EJhFEQGCL,00245-741Fdj7NLF9,00250-U3oQjwTuMX8,00251-wsAYBFtQaL7,00254-YMNvYDhK8mB,00255-NGyoyh91xXJ,00269-JNiWU5TZLtt,00299-bdp1XNEdvmW,00304-X6Pct1msZv5,00323-yHLr6bvWsVm,00324-DoSbsoo4EAg,00327-xgLmjqzoAzF,00378-DqJKU7YU7dA,00384-ceJTwFNjqCt,00401-H8rQCnvBgo6,00404-QN2dRqwd84J,00417-nGhNxKrgBPb,00434-L5QEsaVqwrY,00444-sX9xad6ULKc,00466-xAHnY3QzFUN,00506-QVAA6zecMHu,00567-KjZrPggnHm8,00569-YJDUB7hWg9h,00591-JptJPosx1Z6,00598-mt9H8KcxRKD,00612-GsQBY83r3hb,00624-ooq3SnvC79d,00638-iePHCSf119p,00662-aRKASs4e8j1,00669-DNWbUAJYsPy,00680-YmWinf3mhb5,00706-YHmAkqgwe2p,00712-HZ2iMMBsBQ9,00733-GtM3JtRvvvR,00741-w8GiikYuFRk,00745-yX5efd48dLf,00750-E1NrAhMoqvB,00758-HfMobPm86Xn"
nohup python RL_training/main.py \
    --conf_path config \
    --save_model \
    --use_habitat \
    --habitat_scenes "$HABITAT_SCENES" \
    --use_dino \
    --save_frames_to "$VIZ_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training process started with PID: $PID"
echo "You can check logs with: tail -f $LOG_FILE"
