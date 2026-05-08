"""
Parallel RL training runner using multi-environment collector.
Replaces RLTrainRunner with support for parallel environment interaction.
"""

import json
import sys
from collections import deque
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from components.utils.parallel_collector import ParallelEnvCollector


class ParallelRLTrainRunner:
    """
    RL training runner using multi-environment parallel collection.
    - Spins up N worker processes, each running an independent environment
    - Main process does batch policy inference and updates
    """
    
    def __init__(
        self,
        agent,
        env_class: str,
        env_kwargs: dict,
        num_workers: int = 4,
        device=None,
        save_dir=None,
        scene_numbers=None,
    ):
        self.agent = agent
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.agent_config = agent.agent_config
        self.navigation_config = agent.navigation_config
        self.num_workers = num_workers
        
        if save_dir:
            print(f"[INFO] Training frames will be saved to: {save_dir}")
        
        self.agent.to(self.device)
        
        # Initialize parallel environment collector
        print(f"[INFO] Initializing {num_workers} parallel environments...")
        self.collector = ParallelEnvCollector(
            num_workers=num_workers,
            env_class=env_class,
            env_kwargs=env_kwargs,
            scene_numbers=scene_numbers,
        )
        
        # Training config
        self.total_episodes = self.agent_config.get("episodes", 500)
        self.log_buffer_size = 40
        
        # Episode tracking
        self.episode_steps = [0] * num_workers
        self.episode_rewards = [0.0] * num_workers
        self.episode_scores = [0.0] * num_workers
        self.episode_coverages = [[np.nan] for _ in range(num_workers)]
        
        # Single shared rollout buffer (we'll flatten multi-env data into it)
        self.rollout_buffer = agent.rollout_buffers.__class__(agent.agent_config["num_steps"] * num_workers)
        self.last_obs = [None] * num_workers
        
        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = agent.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        if self.navigation_config["use_transformer"]:
            agent_name += "_Transformer"
        else:
            agent_name += "_LSTM"
        log_dir = f"RL_training/runs/Parallel_{agent_name}_{timestamp}"
        self.writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")
        
        full_config = {"agent_config": self.agent_config, "navigation_config": self.navigation_config}
        self.writer.add_text("full_config", json.dumps(full_config, indent=2), 0)
        
        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)
    
    def run(self):
        """Main training loop."""
        use_tqdm = sys.stderr.isatty()
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=self.total_episodes, desc="Training Episodes", ncols=160, leave=False)
        
        total_transitions = 0
        episode_count = 0
        max_score = 0
        
        # Initial reset: get first observations from all envs
        print("[INFO] Initial environment reset...")
        # Send dummy actions to trigger first step
        dummy_actions = np.zeros(self.num_workers, dtype=int)
        transitions = self.collector.step(dummy_actions)
        
        for trans in transitions:
            worker_id = trans["worker_id"]
            self.last_obs[worker_id] = trans["next_obs"]
        
        print("[INFO] Training started")
        
        steps_in_rollout = 0
        while episode_count < self.total_episodes:
            batch_episode_scores = []
            batch_episode_steps = []
            batch_episode_rewards = []
            
            # Run rollout for num_steps
            while steps_in_rollout < self.agent_config["num_steps"]:
                # Get batch actions from all environments
                if None in self.last_obs:
                    print(f"[WARNING] Some obs is None at step {steps_in_rollout}, using dummy actions")
                    actions = np.zeros(self.num_workers, dtype=int)
                else:
                    actions, values = self.agent.get_batch_actions(self.last_obs)
                
                # Execute step in parallel
                transitions = self.collector.step(actions)
                
                # Collect transitions into shared buffer
                for trans in transitions:
                    worker_id = trans["worker_id"]
                    action = trans["action"]
                    reward = trans["reward"]
                    done = trans["done"]
                    next_obs = trans["next_obs"]
                    info = trans["info"]
                    
                    # Add to shared rollout buffer
                    hiddens = (None, None, None)  # Simplified for now
                    self.rollout_buffer.add(
                        next_obs.state,
                        action,
                        reward,
                        done,
                        hiddens,
                        action,  # last_action
                        info.get("agent_pos", None),
                    )
                    
                    self.episode_rewards[worker_id] += reward
                    self.episode_steps[worker_id] += 1
                    total_transitions += 1
                    
                    if done:
                        # Episode finished
                        final_score = info.get("score", 0.0)
                        self.episode_scores[worker_id] = final_score
                        batch_episode_scores.append(final_score)
                        batch_episode_steps.append(self.episode_steps[worker_id])
                        batch_episode_rewards.append(self.episode_rewards[worker_id])
                        
                        # Reset tracking
                        self.episode_steps[worker_id] = 0
                        self.episode_rewards[worker_id] = 0.0
                    
                    self.last_obs[worker_id] = next_obs
                
                steps_in_rollout += self.num_workers
                if pbar:
                    pbar.update(self.num_workers / self.total_episodes if self.total_episodes > 0 else 1)
            
            # Update agent using collected transitions
            try:
                result = self.agent.update()
                loss = result.get("loss", 0.0)
                entropy = result.get("entropy", 0.0)
            except Exception as e:
                print(f"[ERROR] Failed to update: {e}")
                import traceback
                traceback.print_exc()
                loss = 0.0
                entropy = 0.0
            
            # Log metrics
            if batch_episode_scores:
                mean_score = np.mean(batch_episode_scores)
                mean_steps = np.mean(batch_episode_steps)
                mean_reward = np.mean(batch_episode_rewards)
                
                self.writer.add_scalar("last_episode/Mean_Score", mean_score, episode_count)
                self.writer.add_scalar("last_episode/Mean_Steps", mean_steps, episode_count)
                self.writer.add_scalar("last_episode/Mean_Reward", mean_reward, episode_count)
                self.writer.add_scalar("Loss", loss, episode_count)
                self.writer.add_scalar("Entropy", entropy, episode_count)
                
                self.ep_info_buffer.append({"score": mean_score, "steps": mean_steps, "reward": mean_reward})
                
                if episode_count % 5 == 0:
                    recent_scores = [ep["score"] for ep in self.ep_info_buffer]
                    ma_score = np.mean(recent_scores) if recent_scores else 0
                    if ma_score > max_score:
                        max_score = ma_score
                    print(f"\nEp {episode_count:4d} | MA Score: {ma_score:5.2f} | Max: {max_score:5.2f} | Loss: {loss:.3f}")
            
            episode_count += 1
            steps_in_rollout = 0
        
        if pbar:
            pbar.close()
        
        self.writer.close()
        self.collector.close()
        print("\n[INFO] Training finished.")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, "collector"):
                self.collector.close()
        except:
            pass
