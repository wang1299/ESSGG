#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:/home/wgy/RL:/home/wgy/GroundingDINO

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="train_habitat_${TIMESTAMP}"
LOG_FILE="/home/wgy/RL/train_log/${EXP_NAME}.log"
VIZ_DIR="/home/wgy/RL/train_png/${EXP_NAME}"

mkdir -p "$VIZ_DIR"

echo "Starting training with Habitat and HM3D..."
echo "Log file: $LOG_FILE"
echo "Visualization images: $VIZ_DIR"

python RL_training/main.py \
    --conf_path run_config \
    --save_model \
    --use_habitat \
    --save_frames_to "$VIZ_DIR" 
