# Habitat/HM3D Parallel RL Exploration

This repository currently focuses on reinforcement-learning based indoor exploration in Habitat. The active training pipeline uses HM3D scenes, GroundingDINO perception, and a REINFORCE agent with an LSTM policy. Parallel Habitat workers are used to speed up environment sampling.

The legacy AI2-THOR, A2C, Transformer, imitation-learning, and precomputed-environment code is still present in the repository, but the main runnable path documented here is the Habitat/HM3D parallel training setup.

## Current Setup

| Component | Current implementation |
| --- | --- |
| Simulator | Habitat / habitat-sim |
| Dataset | HM3D, loaded from `/home/wgy/hm3d/scene_datasets/hm3d` |
| Training strategy | REINFORCE + LSTM |
| Parallelism | Multiple Habitat worker processes plus batched policy inference |
| Detector | GroundingDINO |
| Conda environment | `habitat` |
| Main script | `/home/wgy/RL/run_train_parallel.sh` |
| Python entrypoint | `/home/wgy/RL/train_habitat_parallel.py` |
| Logs | `/home/wgy/RL/train_log/parallel_train_*.log` |
| Visualizations | `/home/wgy/RL/train_png/parallel_train_*` |
| TensorBoard | `/home/wgy/RL/RL_training/runs` |

## What Was Modified

The project has been changed from the original AI2-THOR-oriented workflow to a Habitat/HM3D training workflow:

- Habitat is used as the simulator through `components/environments/habitat_env.py`.
- HM3D scene IDs are resolved and filtered in `train_habitat_parallel.py`.
- The parallel Habitat training entrypoint is fixed to `REINFORCE + LSTM` by setting `agent_config["name"] = "reinforce"` and `navigation_config["use_transformer"] = False`.
- GroundingDINO runs as one or more detector service processes and receives RGB frames from the parallel rollout.
- Each worker runs an independent `HabitatEnv` through `ParallelHabitatCollector`.
- The main process performs batched policy inference and then distributes one action per worker.
- Training logs are written to `train_log`.
- Per-run visualizations are written to `train_png`.
- Detection validation images and top-down trajectory images are saved during training.

## Run Training

Use the shell script:

```bash
cd /home/wgy/RL
conda activate habitat
bash /home/wgy/RL/run_train_parallel.sh
```

The script starts training with `nohup`, creates a timestamped experiment name, and writes:

- log file: `/home/wgy/RL/train_log/parallel_train_<timestamp>.log`
- visualization folder: `/home/wgy/RL/train_png/parallel_train_<timestamp>/`
- PID file: `/home/wgy/RL/train_png/parallel_train_<timestamp>/training.pid`

`run_train_parallel.sh` directly invokes `/root/miniconda3/envs/habitat/bin/python`, so it is tied to the `habitat` conda environment even when launched through `nohup`.

To monitor the latest log:

```bash
tail -f $(ls -t /home/wgy/RL/train_log/parallel_train_*.log | head -1)
```

To stop a run:

```bash
kill $(cat /home/wgy/RL/train_png/parallel_train_<timestamp>/training.pid)
```

## Default Parallel Configuration

`run_train_parallel.sh` currently masks physical GPUs `4,5,6,7`:

```bash
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export RL_GPU_IDS="3"
export DINO_DEVICE="cuda:0"
export DINO_DEVICES="cuda:0,cuda:1,cuda:2"
export ENV_GPU_IDS="0,1,2"
```

Inside the process, logical CUDA devices map as:

- `cuda:0` -> physical GPU 4
- `cuda:1` -> physical GPU 5
- `cuda:2` -> physical GPU 6
- `cuda:3` -> physical GPU 7

The default run uses:

- 50 HM3D scenes
- 100 episodes per scene
- 4000 steps per episode
- 4 workers per environment GPU
- 12 total Habitat workers when `ENV_GPU_IDS="0,1,2"`
- GroundingDINO services on logical GPUs `0,1,2`
- RL policy/update on logical GPU `3`

## Parallel Architecture

The current Habitat training stack uses synchronous multi-process sampling:

```text
Main process
  - owns the REINFORCE + LSTM agent
  - batches observations from all workers
  - runs policy inference/update on the RL GPU
  - dispatches one action per worker
  - writes TensorBoard summaries

Worker processes
  - each owns one HabitatEnv
  - receives reset/step/annotation tasks through queues
  - returns Observation objects to the main process
  - saves per-worker visualization files

GroundingDINO service pool
  - runs detector processes on the configured DINO GPUs
  - receives RGB batches from the runner
  - sends detections back for Habitat semantic validation
```

Core files:

- `components/environments/parallel_habitat_collector.py`: starts worker processes and provides `reset_all()`, `step_all()`, `reset_one()`, and `annotate_detections_all()`.
- `RL_training/runner/parallel_habitat_rl_train_runner.py`: batches policy inference, manages per-worker rollout buffers and LSTM hidden states, applies DINO-based discovery reward, and triggers policy updates.
- `components/detectors/grounding_dino_service.py`: runs one or more GroundingDINO detector services and dispatches RGB batches across them.

The runner keeps independent per-environment state:

- `per_env_buffers`: one rollout buffer per Habitat worker.
- `per_env_hidden_states`: one LSTM hidden-state bundle per worker.
- `per_env_last_actions`: one previous-action value per worker.
- `per_env_discovered_objects` and `per_env_discovered_instances`: discovery accounting for DINO reward.

When one worker finishes an episode, only that worker is reset; other workers keep sampling.

## Detection And Visualization Outputs

Each worker writes visualization files under:

```text
/home/wgy/RL/train_png/parallel_train_<timestamp>/worker_<id>/
```

Typical per-episode outputs include:

- `frame_XXXX.png`: periodic RGB/debug frame from Habitat.
- `dino_validation_XXXX.png`: GroundingDINO validation overlay.
- `topdown_XXXX.png`: periodic top-down map snapshot.
- `topdown_trajectory.png`: final top-down trajectory for the episode.
- `trajectory.csv`: per-step agent position, score, coverage, and discovered-instance count.

The fixed visualization interval is controlled by `save_debug_interval` in `HabitatEnv`; the default is every 100 environment steps.

Detection color convention in `dino_validation_XXXX.png`:

- green box: accepted GroundingDINO detection, matched to Habitat semantic GT.
- orange box: rejected GroundingDINO detection.
- red box: visible semantic GT instance that was not matched by an accepted detection.
- blue box: matched Habitat semantic GT box for an accepted detection.

This makes detected, rejected, and missed objects visible in the same validation image.

## Reward Signal

The Habitat reward is based on:

- new validated object discoveries,
- coverage increase on the navigable map,
- per-step penalty `rho`,
- collision penalty when forward movement is blocked.

GroundingDINO detections are validated against Habitat semantic observations before they contribute to score/reward. Background-like labels such as wall, floor, ceiling, window, door, stairs, column, beam, and railing are excluded from reward by default.

## TensorBoard

Training also writes TensorBoard summaries:

```bash
tensorboard --logdir /home/wgy/RL/RL_training/runs
```

Useful scalar groups include per-worker reward/score and aggregate training statistics.

## Troubleshooting

Common issues:

- `CUDA out of memory`: the update step may be too large because rollout data from all workers is flattened into one update batch. Reduce `--num_workers`, reduce `--num_steps`, lower `rgb_dim`/`sg_dim`, or split the update into smaller mini-batches.
- `Worker timeout`: Habitat initialization or stepping is slow. Reduce worker count, check HM3D scene integrity, and inspect the training log for worker errors.
- `ModuleNotFoundError: habitat_sim`: activate the `habitat` conda environment and confirm Habitat/Habitat-Sim are installed there.
- Low GPU utilization: increase worker count only if CPU/memory capacity allows; otherwise the environment side may already be saturated.
- DINO weights missing: verify `/home/wgy/GroundingDINO/weights/groundingdino_swint_ogc.pth`.

Useful monitoring commands:

```bash
tail -f $(ls -t /home/wgy/RL/train_log/parallel_train_*.log | head -1)
watch -n 1 nvidia-smi
tensorboard --logdir /home/wgy/RL/RL_training/runs
```

## Deprecated Docs Merged

The old standalone parallel-training docs were merged into this README and removed:

- `PARALLEL_TRAINING_README.md`: documented the older AI2-THOR/ThorEnv parallel path (`train_parallel.py`, `ParallelEnvCollector`) and no longer matched the active Habitat/HM3D pipeline.
- `PARALLEL_HABITAT_README.md`: contained useful parallel Habitat architecture notes, but its example commands referenced outdated HSSD-style arguments such as `--gpu`, `--config_file`, and `--scene`, while the current script uses HM3D, `--gpu_ids`, `--env_gpu_ids`, `--dino_devices`, and `--habitat_scenes`.

Current documentation should treat this README as the single source of truth for the active training path.

## Modification Checklist

The requested changes are in place:

| Requirement | Status | Relevant files |
| --- | --- | --- |
| Use Habitat simulator | Done | `components/environments/habitat_env.py`, `train_habitat_parallel.py` |
| Use HM3D dataset | Done | `run_train_parallel.sh`, `train_habitat_parallel.py` |
| Use LSTM + REINFORCE | Done | `train_habitat_parallel.py`, `config/agent_config.yaml`, `config/navigation_config.yaml` |
| Use parallel training | Done | `components/environments/parallel_habitat_collector.py`, `RL_training/runner/parallel_habitat_rl_train_runner.py` |
| Use GroundingDINO detector | Done | `components/detectors/grounding_dino_service.py`, `train_habitat_parallel.py` |
| Main script is `/home/wgy/RL/run_train_parallel.sh` | Done | `run_train_parallel.sh` |
| Save training logs | Done | `run_train_parallel.sh` writes `train_log/parallel_train_*.log` |
| Save training visualizations | Done | `run_train_parallel.sh` writes `train_png/parallel_train_*` |
| Save fixed-step detection images | Done | `HabitatEnv._save_dino_validation_overlay()` writes `dino_validation_XXXX.png` |
| Save top-down trajectory image | Done | `HabitatEnv._save_topdown_trajectory()` writes `topdown_trajectory.png` |
| Mark detected and missed objects with different colors | Done | `dino_validation_XXXX.png`: green/orange/red/blue color convention |

## Notes

The active Habitat path currently uses RGB observations and external DINO validation/reward. The older AI2-THOR path still contains local/global scene graph observation code. If scene graph features are needed inside the Habitat policy observation itself, that should be a separate integration step.

## README Maintenance

Whenever this README is read for project orientation or updated after code/config changes, add a short entry below with the UTC time and what changed. Keep entries brief so the log stays useful.

## Change Log

| Time (UTC) | Change |
| --- | --- |
| 2026-05-08 05:46:09 UTC | Merged the two old parallel-training README files into this README, documented their outdated parts, and marked this file as the single source of truth. |
| 2026-05-08 05:46:09 UTC | Added README maintenance rule: future README reads/updates should append a short change note with modification time. |
| 2026-05-08 05:46:09 UTC | Added current `habitat` conda environment, parallel architecture, troubleshooting notes, visualization outputs, and modification checklist. |
