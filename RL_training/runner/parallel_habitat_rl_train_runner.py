"""
Parallel RL training runner for Habitat environment.

Key differences from serial runner:
1. Uses ParallelHabitatCollector to spawn N worker processes, each with independent HabitatEnv
2. Per step: collect obs from all workers -> batch forward -> distribute actions
3. Collects multiple trajectories per update (one per environment)
4. Maintains per-environment hidden states for LSTM

This should significantly speed up wall-clock training by:
- Parallelizing environment stepping (done on separate CPU cores)
- Batching policy inference to utilize GPU better
"""

import json
import sys
import os
from collections import deque
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from components.environments.parallel_habitat_collector import ParallelHabitatCollector
from components.perception.hm3d_labels import HM3D_REWARD_EXCLUDED_LABELS
from components.utils.observation import Observation
from components.utils.rollout_buffer import RolloutBuffer


class ParallelHabitatRLTrainRunner:
    """
    Training runner for Habitat with parallel environment sampling.
    """
    
    def __init__(
        self,
        agent,
        dataset_root: str,
        config_file: str,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        base_scene_ids: Optional[List[str]] = None,
        detection_service: Optional[Any] = None,
        env_config: Optional[Dict[str, Any]] = None,
        scene_count: Optional[int] = None,
        env_gpu_ids: Optional[List[int]] = None,
    ):
        self.agent = agent
        self.num_workers = num_workers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.detection_service = detection_service
        self.env_config = env_config or {}
        self.scene_count = max(int(scene_count or len(base_scene_ids or []) or 1), 1)
        
        # Move agent to device
        self.agent.to(self.device)
        self.agent_config = agent.agent_config
        self.navigation_config = agent.navigation_config
        self.discovery_bonus_scale = float(self.env_config.get("discovery_bonus_scale", 1.0))
        self.score_norm_target = max(float(self.env_config.get("score_norm_target", 120.0)), 1.0)
        self.det_score_thr = float(self.env_config.get("det_score_thr", 0.20))
        self.reward_excluded_labels = {
            str(label) for label in self.env_config.get("reward_excluded_labels", HM3D_REWARD_EXCLUDED_LABELS)
        }
        self.reward_allow_semantic_iou_only = bool(self.env_config.get("reward_allow_semantic_iou_only", False))
        
        # Create parallel environment collector
        env_kwargs = {
            "render": False,
            "width": 300,
            "height": 300,
            "use_detector": False,
            "detector": None,
            "det_score_thr": self.det_score_thr,
            "max_actions": int(self.agent_config.get("num_steps", 4000)),
        }
        if save_dir:
            env_kwargs["save_debug_path"] = save_dir
        for key in [
            "score_norm_target",
            "instance_merge_dist",
            "coverage_cell_size",
            "nav_sample_points",
            "topdown_meters_per_pixel",
            "agent_radius",
            "agent_height",
            "agent_max_climb",
            "navmesh_cell_height",
            "navmesh_cell_size",
            "fill_position_from_gt",
            "rho",
            "coverage_bonus_scale",
            "discovery_bonus_scale",
            "collision_penalty",
            "gt_validation_iou_threshold",
            "gt_validation_mode",
            "success_recall_threshold",
            "success_reward",
            "reward_excluded_labels",
            "max_actions",
            "save_debug_interval",
            "save_debug_path",
        ]:
            if key in self.env_config and self.env_config[key] is not None:
                env_kwargs[key] = self.env_config[key]
        
        # Increase timeout to allow for slower environment initialization
        self.env_collector = ParallelHabitatCollector(
            num_workers=num_workers,
            dataset_root=dataset_root,
            config_file=config_file,
            base_scene_ids=base_scene_ids,
            env_kwargs=env_kwargs,
            timeout=300.0,
            env_gpu_ids=env_gpu_ids,
        )
        
        # Per-environment state tracking
        self.num_envs = num_workers
        self.per_env_buffers: List[RolloutBuffer] = [
            RolloutBuffer(self.agent_config.get("num_steps", 4000))
            for _ in range(num_workers)
        ]
        self.per_env_last_actions = [-1] * num_workers
        self.per_env_scene_indices = [i % self.scene_count for i in range(num_workers)]
        self.per_env_discovered_objects = [set() for _ in range(num_workers)]
        self.per_env_discovered_instances = [set() for _ in range(num_workers)]
        self.per_env_prev_score = [0.0] * num_workers
        self.per_env_hidden_states = [
            {
                "lssg": None,
                "gssg": None,
                "policy": None,
            }
            for _ in range(num_workers)
        ]
        
        # Config
        self.total_episodes = self.agent_config.get("episodes", 500)
        self.num_steps_per_rollout = self.agent_config.get("num_steps", 4000)
        self.log_buffer_size = 40
        self.episode_step_counters = [0] * num_workers
        
        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = agent.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        if self.navigation_config.get("use_transformer"):
            agent_name += "_Transformer"
        else:
            agent_name += "_LSTM"
        log_dir = f"RL_training/runs/{agent_name}_{timestamp}_parallel_{num_workers}w"
        self.writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs: {log_dir}")
        
        full_config = {
            "agent_config": self.agent_config,
            "navigation_config": self.navigation_config,
            "num_workers": num_workers,
        }
        self.writer.add_text("config", json.dumps(full_config, indent=2), 0)
        
        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)
        self.global_step = 0
    
    def _build_batch_dict(self, obs_list: List[Observation]) -> Dict[str, List[List[Any]]]:
        """Convert per-env observations into a [B, T=1] batch dict."""
        return {
            "rgb": [[obs.state[0]] for obs in obs_list],
            "lssg": [[obs.state[1]] for obs in obs_list],
            "gssg": [[obs.state[2]] for obs in obs_list],
            "occupancy": [[obs.state[3]] for obs in obs_list],
            "agent_pos": [[obs.info.get("agent_pos", obs.info.get("agent_position", None))] for obs in obs_list],
        }

    def _stack_lstm_hidden(self, hidden_list, hidden_size: int):
        """Stack per-env LSTM hidden states into [num_layers, B, H]."""
        num_layers = 2
        batch_size = len(hidden_list)
        h_list = []
        c_list = []
        for hidden in hidden_list:
            if hidden is None:
                h_list.append(torch.zeros(num_layers, 1, hidden_size, device=self.device))
                c_list.append(torch.zeros(num_layers, 1, hidden_size, device=self.device))
            else:
                h, c = hidden
                h_list.append(h.to(self.device))
                c_list.append(c.to(self.device))
        h = torch.cat(h_list, dim=1) if batch_size > 0 else torch.zeros(num_layers, 0, hidden_size, device=self.device)
        c = torch.cat(c_list, dim=1) if batch_size > 0 else torch.zeros(num_layers, 0, hidden_size, device=self.device)
        return h, c

    def _split_lstm_hidden(self, hidden):
        """Split batched LSTM hidden states back into per-env tuples."""
        if hidden is None:
            return [None] * self.num_workers
        h, c = hidden
        return [(h[:, i:i+1, :].contiguous(), c[:, i:i+1, :].contiguous()) for i in range(h.size(1))]

    def _build_detection_key(self, detection: Dict[str, Any], label: str):
        box = detection.get("bbox", detection.get("box"))
        if box is not None and len(box) == 4:
            cx = 0.5 * (float(box[0]) + float(box[2]))
            cy = 0.5 * (float(box[1]) + float(box[3]))
            bw = float(box[2]) - float(box[0])
            bh = float(box[3]) - float(box[1])
            return (label, round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1))
        return (label,)

    def _run_detection_batch(self, obs_list: List[Observation]):
        if not obs_list:
            return []

        rgb_batch = [obs.state[0] for obs in obs_list]

        if self.detection_service is not None:
            try:
                return self.detection_service.detect_batch(rgb_batch)
            except Exception as exc:
                print(f"[WARN] DINO service batch failed, falling back to local call: {exc}")

        return [[] for _ in rgb_batch]

    def _apply_detection_reward(self, env_id: int, obs: Observation, detections):
        """Convert a batch of DINO detections into cumulative score and reward."""
        if obs is None:
            return 0.0, 0.0

        target_gt_ids = set()
        if obs.info:
            try:
                target_gt_ids = {int(item) for item in obs.info.get("scene_reward_gt_ids", [])}
            except Exception:
                target_gt_ids = set()

        for det in detections or []:
            if float(det.get("score", 0.0)) < self.det_score_thr:
                continue
            if self.detection_service is not None and det.get("is_gt_valid") is not True:
                continue
            if not self.reward_allow_semantic_iou_only and det.get("gt_match_mode") == "semantic_iou_only":
                continue

            label = str(det.get("canonical_label") or det.get("label") or "unknown")
            if label in self.reward_excluded_labels:
                continue

            if target_gt_ids:
                try:
                    gt_semantic_id = int(det.get("gt_semantic_id"))
                except Exception:
                    continue
                if gt_semantic_id not in target_gt_ids:
                    continue

            self.per_env_discovered_objects[env_id].add(label)
            self.per_env_discovered_instances[env_id].add(self._build_detection_key(det, label))

        discovered_instance_count = len(self.per_env_discovered_instances[env_id])
        discovered_label_count = len(self.per_env_discovered_objects[env_id])
        norm_count = discovered_instance_count if discovered_instance_count > 0 else discovered_label_count
        current_score = min(norm_count / self.score_norm_target, 1.0)
        score_gain = current_score - self.per_env_prev_score[env_id]
        self.per_env_prev_score[env_id] = current_score

        if obs.info is None:
            obs.info = {}
        obs.info["score"] = float(current_score)
        obs.info["num_discovered"] = norm_count
        obs.info["num_discovered_labels"] = discovered_label_count
        obs.info["num_discovered_instances"] = discovered_instance_count

        return current_score, self.discovery_bonus_scale * score_gain
    
    def _get_batch_actions(self, obs_list: List[Observation]):
        """
        Get actions for all environments in parallel via a single batch forward.
        """
        if not obs_list:
            return [], np.array([])

        with torch.no_grad():
            batch_dict = self._build_batch_dict(obs_list)
            last_actions = torch.tensor(
                [[action] for action in self.per_env_last_actions],
                dtype=torch.long,
                device=self.device,
            )

            if self.navigation_config.get("use_transformer"):
                state_seq, _, _ = self.agent.encoder(batch_dict, last_actions)
                policy_hidden = None
                lssg_hidden_out = [None] * len(obs_list)
                gssg_hidden_out = [None] * len(obs_list)
            else:
                hidden_size = self.agent.encoder.lssg_encoder.lstm.hidden_size
                lssg_hidden = self._stack_lstm_hidden(
                    [hidden["lssg"] for hidden in self.per_env_hidden_states],
                    hidden_size,
                )
                gssg_hidden = self._stack_lstm_hidden(
                    [hidden["gssg"] for hidden in self.per_env_hidden_states],
                    hidden_size,
                )
                policy_hidden = self._stack_lstm_hidden(
                    [hidden["policy"] for hidden in self.per_env_hidden_states],
                    self.agent.policy.core.hidden_size,
                )

                state_seq, new_lssg, new_gssg = self.agent.encoder(
                    batch_dict,
                    last_actions,
                    lssg_hidden=lssg_hidden,
                    gssg_hidden=gssg_hidden,
                )
                lssg_hidden_out = self._split_lstm_hidden(new_lssg)
                gssg_hidden_out = self._split_lstm_hidden(new_gssg)

            logits, value, new_policy_hidden = self.agent.policy(
                state_seq,
                hidden=policy_hidden,
            )

            if not self.navigation_config.get("use_transformer"):
                policy_hidden_out = self._split_lstm_hidden(new_policy_hidden)
                for env_id in range(self.num_workers):
                    self.per_env_hidden_states[env_id]["lssg"] = lssg_hidden_out[env_id]
                    self.per_env_hidden_states[env_id]["gssg"] = gssg_hidden_out[env_id]
                    self.per_env_hidden_states[env_id]["policy"] = policy_hidden_out[env_id]

            logits = logits[:, -1, :]
            if value is not None:
                value = value[:, -1]

            probs = torch.softmax(logits, dim=-1)
            from torch.distributions import Categorical
            dist = Categorical(probs=probs)
            actions = dist.sample().tolist()
            values = value.tolist() if value is not None else [0.0] * len(actions)

        return actions, np.array(values)
    
    def run(self):
        """
        Main training loop with parallel environment sampling.
        """
        use_tqdm = sys.stderr.isatty()
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=self.total_episodes, desc="Episodes", ncols=160, leave=False)
        
        episode_count = 0
        total_steps_collected = 0
        
        # Initialize environments
        print("[INFO] Initializing parallel environments...")
        try:
            obs_list = self.env_collector.reset_all(
                scene_ids=[str((i % self.scene_count) + 1) for i in range(self.num_workers)],
                random_start=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to reset environments: {e}")
            self.env_collector.close()
            return
        
        # Reset buffers and hidden states
        for buffer in self.per_env_buffers:
            buffer.clear()
        self.per_env_last_actions = [-1] * self.num_workers
        self.per_env_scene_indices = [i % self.scene_count for i in range(self.num_workers)]
        self.per_env_discovered_objects = [set() for _ in range(self.num_workers)]
        self.per_env_discovered_instances = [set() for _ in range(self.num_workers)]
        self.per_env_prev_score = [0.0] * self.num_workers
        for hidden_dict in self.per_env_hidden_states:
            hidden_dict["lssg"] = None
            hidden_dict["gssg"] = None
            hidden_dict["policy"] = None
        self.episode_step_counters = [0] * self.num_workers
        
        step_in_rollout = 0
        max_score = 0.0
        
        try:
            while episode_count < self.total_episodes:
                # Collect rollout data from all environments
                step_in_rollout += 1
                
                # Get actions for all environments
                actions, values = self._get_batch_actions(obs_list)
                
                # Step all environments
                try:
                    obs_list = self.env_collector.step_all(actions)
                except Exception as e:
                    print(f"[ERROR] Parallel step failed: {e}")
                    break

                detection_batches = self._run_detection_batch(obs_list)
                if self.detection_service is not None:
                    try:
                        detection_batches = self.env_collector.annotate_detections_all(detection_batches)
                    except Exception as exc:
                        print(f"[WARN] Failed to annotate DINO detections in workers: {exc}")
                
                # Collect transitions for each environment
                for env_id, obs in enumerate(obs_list):
                    # Extract info
                    reward = obs.reward
                    terminated = obs.terminated
                    truncated = obs.truncated
                    done = terminated or truncated

                    score, det_bonus = self._apply_detection_reward(env_id, obs, detection_batches[env_id] if env_id < len(detection_batches) else [])
                    reward = float(reward or 0.0) + float(det_bonus)
                    obs.reward = reward
                    if obs.info is None:
                        obs.info = {}
                    obs.info["score"] = float(score)
                    
                    # Add to this env's buffer
                    buffer = self.per_env_buffers[env_id]
                    
                    # Store trajectory step
                    buffer.add(
                        state=obs.state,
                        action=actions[env_id],
                        reward=reward,
                        done=done,
                        hiddens={
                            "lssg": self.per_env_hidden_states[env_id]["lssg"],
                            "gssg": self.per_env_hidden_states[env_id]["gssg"],
                            "policy": self.per_env_hidden_states[env_id]["policy"],
                        },
                        last_action=self.per_env_last_actions[env_id],
                        agent_position=obs.info.get("agent_position", None),
                    )
                    
                    self.per_env_last_actions[env_id] = actions[env_id]
                    self.episode_step_counters[env_id] += 1
                    total_steps_collected += 1
                    
                    # Track episode info
                    if obs.info:
                        score = obs.info.get("score", 0.0)
                        self.writer.add_scalar(f"env_{env_id}/reward", reward, self.global_step)
                        self.writer.add_scalar(f"env_{env_id}/score", score, self.global_step)
                    
                    # Reset if done
                    if done:
                        if obs.info:
                            ep_score = obs.info.get("score", 0.0)
                            ep_steps = self.episode_step_counters[env_id]
                            self.ep_info_buffer.append({
                                "score": ep_score,
                                "steps": ep_steps,
                                "env_id": env_id,
                            })
                            
                            if ep_score > max_score:
                                max_score = ep_score
                            
                            print(f"[Episode {episode_count}] Env{env_id}: Score={ep_score:.2f}, Steps={ep_steps}")
                        
                        # Reset only this environment; do not disturb the other workers.
                        self.per_env_scene_indices[env_id] = (self.per_env_scene_indices[env_id] + 1) % self.scene_count
                        obs_list[env_id] = self.env_collector.reset_one(
                            env_id,
                            scene_id=str(self.per_env_scene_indices[env_id] + 1),
                            random_start=True,
                        )

                        self.per_env_discovered_objects[env_id].clear()
                        self.per_env_discovered_instances[env_id].clear()
                        self.per_env_prev_score[env_id] = 0.0
                        
                        self.episode_step_counters[env_id] = 0
                        self.per_env_last_actions[env_id] = -1
                        
                        # Reset hidden states
                        self.per_env_hidden_states[env_id]["lssg"] = None
                        self.per_env_hidden_states[env_id]["gssg"] = None
                        self.per_env_hidden_states[env_id]["policy"] = None
                        
                        episode_count += 1
                        
                        if pbar:
                            pbar.update(1)
                self.global_step += 1
                
                # Check if we've collected enough steps for an update
                if step_in_rollout >= self.num_steps_per_rollout:
                    print(f"\n[UPDATE] Collected {total_steps_collected} steps across {self.num_workers} envs, updating...")
                    
                    # Perform update using concatenated buffers
                    self._perform_update()
                    
                    # Reset buffers
                    for buffer in self.per_env_buffers:
                        buffer.clear()
                    
                    step_in_rollout = 0
                    
                    # Log stats
                    if len(self.ep_info_buffer) > 0:
                        avg_score = np.mean([e["score"] for e in self.ep_info_buffer])
                        avg_steps = np.mean([e["steps"] for e in self.ep_info_buffer])
                        self.writer.add_scalar("train/avg_score", avg_score, episode_count)
                        self.writer.add_scalar("train/avg_steps", avg_steps, episode_count)
                        self.writer.add_scalar("train/max_score", max_score, episode_count)
                        print(f"[STATS] Avg Score: {avg_score:.2f}, Avg Steps: {avg_steps:.1f}, Max Score: {max_score:.2f}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
        finally:
            if self.detection_service is not None:
                try:
                    self.detection_service.close()
                except Exception:
                    pass
            self.env_collector.close()
            self.writer.close()
            print("[INFO] Training finished")
    
    def _perform_update(self):
        """
        Perform a single policy update using data from all environments.
        """
        total_states = []
        total_actions = []
        total_rewards = []
        total_dones = []
        total_last_actions = []
        total_agent_pos = []

        for buffer in self.per_env_buffers:
            if len(buffer.rewards) == 0:
                continue

            total_states.extend(
                zip(
                    buffer.state_rgb,
                    buffer.state_lssg,
                    buffer.state_gssg,
                    buffer.state_occ,
                )
            )
            total_actions.extend(buffer.actions)
            total_rewards.extend(buffer.rewards)
            total_dones.extend(buffer.dones)
            total_last_actions.extend(buffer.last_actions)
            total_agent_pos.extend(buffer.agent_positions)

        if len(total_actions) == 0:
            return

        # Reuse the agent's normal update path by flattening the parallel rollout
        # back into its shared rollout buffer.
        self.agent.rollout_buffers.clear()
        self.agent.rollout_buffers.add_batch(
            states=total_states,
            actions=total_actions,
            rewards=total_rewards,
            dones=total_dones,
            hiddens=[{"lssg": None, "gssg": None, "policy": None}],
            last_actions=total_last_actions,
            agent_pos=total_agent_pos,
        )

        if hasattr(self.agent, "update"):
            self.agent.update()  # This will use self.agent.rollout_buffers
    
    def _get_batch_from_buffer(self, buffer: RolloutBuffer):
        """
        Convert a RolloutBuffer to a batch dict for forward_update.
        """
        batch_data = buffer.get(self.agent_config.get("gamma", 0.99))
        
        batch = {
            k: batch_data[k] for k in [
                "rgb", "lssg", "gssg", "occ", "actions", "returns", "last_actions", "agent_positions"
            ]
        }
        
        # Convert to tensors and ensure correct shapes
        for k in ["actions", "returns", "last_actions"]:
            if not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.tensor(batch[k], device=self.device)
            if batch[k].dim() == 1:
                batch[k] = batch[k].unsqueeze(0)
        
        for k in ["rgb", "lssg", "gssg", "occ", "agent_positions"]:
            if isinstance(batch[k], list) and not isinstance(batch[k][0], list):
                batch[k] = [batch[k]]
        
        return batch
    
    def close(self):
        """Cleanup resources."""
        self.env_collector.close()
        self.writer.close()
