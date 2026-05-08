# 并行 Habitat RL 训练 - 快速开始

仿照 `run_train.sh` 的配置，使用多进程并行采样 + 批量推理 + 多 GPU 分布式。

## 文件说明

| 文件 | 说明 |
|-----|------|
| `run_train_parallel.sh` | nohup 启动脚本（仿照 run_train.sh） |
| `train_habitat_parallel.py` | 并行训练主脚本（改进版本，支持多 GPU + DINO） |
| `components/environments/parallel_habitat_collector.py` | 多进程采样器 |
| `RL_training/runner/parallel_habitat_rl_train_runner.py` | 并行训练循环 |

## 快速启动

### 方式 1: 使用 nohup 脚本（推荐）

```bash
chmod +x /home/wgy/RL/run_train_parallel.sh
nohup /home/wgy/RL/run_train_parallel.sh > /home/wgy/RL/train_log/nohup.log 2>&1 &
```

这将：
- 使用 4 个并行 worker
- 在 GPU 5,6,7 上运行策略推理
- 在 GPU 4 上运行 GroundingDINO 检测
- 训练所有 50 个 HM3D 场景，每个场景 100 个 episode
- 保存可视化和日志到 `/home/wgy/RL/train_log` 和 `/home/wgy/RL/train_png`

### 方式 2: 直接运行 Python 脚本

```bash
cd /home/wgy/RL

python train_habitat_parallel.py \
    --conf_path /home/wgy/RL/config \
    --num_workers 4 \
    --episodes 100 \
    --num_steps 4000 \
    --gpu_ids "5,6,7" \
    --dino_device "cuda:4" \
    --use_dino \
    --dataset_root /home/wgy/hm3d/scene_datasets/hm3d \
    --habitat_scenes "00016-qk9eeNeR4vw,00017-oEPjPNSPmzL,..." \
    --save_frames_to /home/wgy/RL/train_png/my_run
```

### 方式 3: 自定义参数

```bash
# 仅训练 5 个场景，使用 6 个 worker，GPU 0-5
python train_habitat_parallel.py \
    --conf_path /home/wgy/RL/config \
    --num_workers 6 \
    --episodes 200 \
    --gpu_ids "0,1,2,3,4,5" \
    --dino_device "cuda:6" \
    --habitat_scenes "00016-qk9eeNeR4vw,00017-oEPjPNSPmzL,00023-zepmXAdrpjR,00031-Wo6kuutE9i7,00033-oPj9qMxrDEa" \
    --save_frames_to /home/wgy/RL/train_png/test_5scenes
```

## 配置说明

### 与 run_train.sh 的对应关系

```bash
# run_train.sh
export RL_GPU_IDS="5,6,7"
export DINO_DEVICE="cuda:4"
nohup python RL_training/main.py \
    --use_habitat \
    --habitat_scenes "$HABITAT_SCENES" \
    --use_dino \
    ...

# 对应的 run_train_parallel.sh
export RL_GPU_IDS="5,6,7"
export DINO_DEVICE="cuda:4"
nohup python train_habitat_parallel.py \
    --gpu_ids "5,6,7" \
    --dino_device "cuda:4" \
    --use_dino \
    --habitat_scenes "$HABITAT_SCENES" \
    ...
```

### 参数详解

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--num_workers` | 4 | 并行环境数 |
| `--episodes` | 100 | 每个场景的训练 episode 数 |
| `--num_steps` | 4000 | 每个 rollout 收集的步数 |
| `--gpu_ids` | "4,5,6,7" | 策略网络使用的 GPU ID (逗号分隔) |
| `--dino_device` | "cuda:4" | DINO 检测器使用的 GPU |
| `--use_dino` | (flag) | 启用 GroundingDINO 检测 |
| `--dataset_root` | `/home/wgy/hm3d/...` | HM3D 数据集根路径 |
| `--habitat_scenes` | None | 场景 ID 列表 (逗号分隔) |
| `--save_frames_to` | `/home/wgy/RL/train_png` | 可视化输出目录 |
| `--conf_path` | "config" | 配置文件目录 (agent/navigation/env config) |

## 监控训练

### 查看日志

```bash
# 实时监控
tail -f /home/wgy/RL/train_log/parallel_train_*.log

# 查看最新的
tail -f $(ls -t /home/wgy/RL/train_log/*.log | head -1)
```

### TensorBoard 可视化

```bash
tensorboard --logdir /home/wgy/RL/RL_training/runs
```

### 查看 GPU 使用

```bash
watch -n 1 nvidia-smi

# 预期看到：
# GPU 4: GroundingDINO ~50-80% 显存 / 60-90% 使用率
# GPU 5,6,7: 策略推理 ~30-60% 显存 / 40-70% 使用率
```

## 停止训练

```bash
# 获取 PID
cat /home/wgy/RL/train_png/parallel_train_*/training.pid

# 优雅停止（Ctrl+C）
kill -TERM <PID>

# 强制杀死
kill -9 <PID>
```

## 性能预期

- **并行加速**: 4 workers 预期 **2.5-3.2x** 加速
- **GPU 利用率**: 从单环境的 5-10% 提升到 40-80%
- **总训练时间**: 
  - 串行 (main.py): ~300-400 小时 (50 scenes × 100 ep)
  - 并行 (4 workers): ~100-150 小时

## 常见问题

### Q1: "Worker timeout after 60s"
**A**: Habitat 环境步进太慢。尝试：
- 降低 `num_workers` (减少 CPU 竞争)
- 检查 HM3D 数据集是否有损坏

### Q2: GPU 利用率仍然很低 (< 30%)
**A**: 
- 增加 `num_workers` (现在是 4)
- 检查 batch size 是否合适
- 模型参数可能太小，增加 `rgb_dim`/`sg_dim`

### Q3: 显存不足 OOM
**A**:
- 降低 `num_workers`
- 在 `config/navigation_config.yaml` 中减小 `rgb_dim`
- 增加更多 GPU (e.g., 使用 `--gpu_ids "0,1,2,3,4,5,6,7"`)

### Q4: 某个 worker 经常崩溃
**A**:
- Habitat 可能有内存泄漏，定期重启 worker (框架已内置)
- 检查场景数据完整性

## 配置示例

### 轻量配置 (快速测试)
```bash
python train_habitat_parallel.py \
    --num_workers 2 \
    --episodes 10 \
    --habitat_scenes "00016-qk9eeNeR4vw,00017-oEPjPNSPmzL" \
    --save_frames_to /tmp/test
```

### 标准配置 (推荐)
```bash
bash run_train_parallel.sh
```

### 高配配置 (如果 GPU 足够)
```bash
python train_habitat_parallel.py \
    --num_workers 8 \
    --episodes 200 \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --dino_device "cuda:7" \
    --use_dino
```

## 技术细节

### 多 GPU 策略

```
GPU 4: GroundingDINO (DINO)
  ├─ 接收 RGB 帧 (batch B)
  ├─ 返回检测结果
  └─ 显存: ~12GB (SwinT + OGC)

GPUs 5,6,7: 策略网络 (DataParallel)
  ├─ RGB Encoder: DataParallel(gpu_ids=[5,6,7])
  ├─ 图编码器 (LSSG/GSSG): single GPU
  └─ 策略头: single GPU
  
主 GPU (GPU 5):
  ├─ forward_seq() 输出 [B, T, D]
  ├─ policy forward [B, T, num_actions]
  └─ 梯度更新
```

### 环境隔离

```
Main Process (GPU 5,6,7)          Worker Process (CPU Core 0)
    ├─ ParallelHabitatCollector   ├─ HabitatEnv instance
    ├─ task_queue ────────────>   ├─ listen for tasks
    │  (reset, step)              │
    ├─ result_queue <────────────┤ send results
    │  (obs, reward, done)        │
    ├─ Agent.forward_batch()      └─ independent env loop
    └─ update()
```

## 下一步优化

1. **FSDP**: 多机多卡分布式训练
2. **Replay Buffer**: off-policy 学习 (PPO/TRPO)
3. **Scene Rotation**: 动态轮换场景给 workers
4. **AMP**: 自动混合精度降低显存

---

**快速参考**:
```bash
# 启动
bash run_train_parallel.sh

# 监控
tail -f /home/wgy/RL/train_log/parallel_train_*.log

# 查看 GPU
nvidia-smi

# 停止
kill $(cat /home/wgy/RL/train_png/parallel_train_*/training.pid)
```
