#!/bin/bash

# 创建输出目录
mkdir -p /home/wgy/RL/train_log
mkdir -p /home/wgy/RL/train_png

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/GroundingDINO

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
# [Update] Added --precomputed to avoid Vulkan/X11 rendering issues
nohup python RL_training/main.py \
    --conf_path run_config \
    --save_model \
    --precomputed \
    --use_dino \
    --save_frames_to "$VIZ_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Training process started with PID: $PID"
echo "You can check logs with: tail -f $LOG_FILE"
