#!/bin/bash

# Parallel Habitat RL Training Script
# Mirrors the configuration from run_train.sh but uses parallel environment sampling

# Create output directories
mkdir -p /home/wgy/RL/train_log
mkdir -p /home/wgy/RL/train_png

# Set environment variables (same as run_train.sh)
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PYTHONPATH:$(pwd):/home/wgy/GroundingDINO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Keep physical GPU 0 completely unused by masking visible devices to 4,5,6,7.
# Inside the process, these map to logical CUDA devices 0,1,2,3.
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export RL_GPU_IDS="3"
export DINO_DEVICE="cuda:0"
export DINO_DEVICES="cuda:0,cuda:1,cuda:2"
export ENV_GPU_IDS="0,1,2"

# Get timestamp for log file naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="parallel_train_${TIMESTAMP}"
LOG_FILE="/home/wgy/RL/train_log/${EXP_NAME}.log"
VIZ_DIR="/home/wgy/RL/train_png/${EXP_NAME}"

# Create this run's visualization directory
mkdir -p "$VIZ_DIR"

echo "[INFO] Starting parallel Habitat RL training..."
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] Visualization directory: $VIZ_DIR"
echo "[INFO] RL_GPU_IDS: $RL_GPU_IDS"
echo "[INFO] ENV_GPU_IDS: $ENV_GPU_IDS"
echo "[INFO] DINO_DEVICE: $DINO_DEVICE"
echo "[INFO] DINO_DEVICES: $DINO_DEVICES"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Logical GPU mapping: cuda:0->physical 4, cuda:1->physical 5, cuda:2->physical 6, cuda:3->physical 7"
echo "[INFO] Habitat workers and DINO use logical GPUs 0,1,2; policy/update uses logical GPU 3"

# HM3D scenes (same as run_train.sh)
HABITAT_SCENES="00016-qk9eeNeR4vw,00017-oEPjPNSPmzL,00023-zepmXAdrpjR,00031-Wo6kuutE9i7,00033-oPj9qMxrDEa,00087-YY8rqV6L6rf,00099-226REUyJh2K,00105-xWvSkKiWQpC,00108-oStKKWkQ1id,00155-iLDo95ZbDJq,00166-RaYrxWt5pR1,00177-VSxVP19Cdyw,00210-j2EJhFEQGCL,00245-741Fdj7NLF9,00250-U3oQjwTuMX8,00251-wsAYBFtQaL7,00254-YMNvYDhK8mB,00255-NGyoyh91xXJ,00269-JNiWU5TZLtt,00299-bdp1XNEdvmW,00304-X6Pct1msZv5,00323-yHLr6bvWsVm,00324-DoSbsoo4EAg,00327-xgLmjqzoAzF,00378-DqJKU7YU7dA,00384-ceJTwFNjqCt,00401-H8rQCnvBgo6,00404-QN2dRqwd84J,00417-nGhNxKrgBPb,00434-L5QEsaVqwrY,00444-sX9xad6ULKc,00466-xAHnY3QzFUN,00506-QVAA6zecMHu,00567-KjZrPggnHm8,00569-YJDUB7hWg9h,00591-JptJPosx1Z6,00598-mt9H8KcxRKD,00612-GsQBY83r3hb,00624-ooq3SnvC79d,00638-iePHCSf119p,00662-aRKASs4e8j1,00669-DNWbUAJYsPy,00680-YmWinf3mhb5,00706-YHmAkqgwe2p,00712-HZ2iMMBsBQ9,00733-GtM3JtRvvvR,00741-w8GiikYuFRk,00745-yX5efd48dLf,00750-E1NrAhMoqvB,00758-HfMobPm86Xn"

# Count scenes
SCENE_COUNT=$(echo "$HABITAT_SCENES" | tr ',' '\n' | wc -l)
echo "[INFO] Training on $SCENE_COUNT scenes"

# Run training with nohup
nohup /root/miniconda3/envs/habitat/bin/python /home/wgy/RL/train_habitat_parallel.py \
    --conf_path /home/wgy/RL/config \
    --num_workers 4 \
    --episodes 100 \
    --num_steps 4000 \
    --gpu_ids "$RL_GPU_IDS" \
    --env_gpu_ids "$ENV_GPU_IDS" \
    --dino_device "$DINO_DEVICE" \
    --dino_devices "$DINO_DEVICES" \
    --use_dino \
    --dataset_root /home/wgy/hm3d/scene_datasets/hm3d \
    --habitat_scenes "$HABITAT_SCENES" \
    --save_frames_to "$VIZ_DIR" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "[INFO] Training process started with PID: $PID"
echo "[INFO] Check logs with: tail -f $LOG_FILE"
echo "[INFO] Kill with: kill $PID"

# Save PID to a file for easy reference
echo $PID > "$VIZ_DIR/training.pid"

exit 0
