import json
from collections import deque
from datetime import datetime

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class RLTrainRunner:
    def __init__(self, env, agent, device=None):
        self.env = env
        self.agent = agent
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)
        self.agent_config = agent.agent_config
        self.navigation_config = agent.navigation_config

        self.total_episodes = self.agent_config.get("episodes", 500)
        self.scene_numbers = agent.scene_numbers
        self.log_buffer_size = 40

        # --- TensorBoard Writer ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = agent.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        if self.navigation_config["use_transformer"]:
            agent_name += "_Transformer"
        else:
            agent_name += "_LSTM"
        log_dir = f"RL_training/runs/{agent_name}_{timestamp}_{self.env.rho}"
        self.writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

        full_config = {"agent_config": self.agent_config, "navigation_config": self.navigation_config}
        self.writer.add_text("full_config", json.dumps(full_config, indent=2), 0)

        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)

    def robust_reset(self, scene_number, max_retries=3):
        for attempt in range(max_retries):
            try:
                obs = self.env.reset(scene_number=scene_number, random_start=True)
                return obs
            except TimeoutError:
                print(f"[WARNING] TimeoutError at reset, trying restart... ({attempt + 1}/{max_retries})")
                try:
                    self.env.close()
                except Exception:
                    pass
                from components.environments.thor_env import ThorEnv

                self.env = ThorEnv(render=self.env.render, rho=self.env.rho, max_actions=self.agent_config["num_steps"])
        print("[ERROR] Multiple Timeouts at reset - skipping episode.")
        return None

    def robust_step(self, action, max_retries=3):
        for attempt in range(max_retries):
            try:
                obs = self.env.step(action)
                return obs
            except TimeoutError:
                print(f"[WARNING] TimeoutError at step, trying restart... ({attempt + 1}/{max_retries})")
                try:
                    self.env.close()
                except Exception:
                    pass
                from components.environments.thor_env import ThorEnv

                self.env = ThorEnv(render=self.env.render, rho=self.env.rho, max_actions=self.agent_config["num_steps"])
                return None
        print("[ERROR] Multiple Timeouts at step - skipping episode.")
        return None

    def run(self):
        pbar = tqdm(total=self.total_episodes, desc="Training Episodes", ncols=160, leave=False)
        episode_count = 0
        max_score = 0

        while episode_count < self.total_episodes:
            episode_scores = []
            episode_steps = []
            episode_rewards = []
            episode_losses = []
            episode_entropies = []
            episode_policy_losses = []
            episode_value_losses = []
            episode_stds = []

            for scene_number in self.scene_numbers:
                obs = self.robust_reset(scene_number)
                if obs is None:
                    continue
                episode_reward = 0
                episode_steps_scene = 0

                unfinished_episode = False
                while not (obs.terminated or obs.truncated):
                    action, lssg_h, gssg_h, policy_h, last_a, value = self.agent.get_action(obs)
                    next_obs = self.robust_step(action)
                    if next_obs is None:
                        unfinished_episode = True
                        break
                    reward = next_obs.reward
                    done = next_obs.terminated or next_obs.truncated
                    agent_pos = next_obs.info.get("agent_pos", None)

                    hiddens = (lssg_h, gssg_h, policy_h)
                    self.agent.rollout_buffers.add(obs.state, action, reward, done, hiddens, last_a, agent_pos)

                    episode_reward += reward
                    episode_steps_scene += 1

                    obs = next_obs

                if unfinished_episode:
                    self.agent.reset()
                    continue

                result = self.agent.update()
                loss = result.get("loss", None)
                policy_loss = result.get("policy_loss", None)
                value_loss = result.get("value_loss", None)
                entropy = result.get("entropy", None)
                std = result.get("ret_std", None)

                final_score = obs.info.get("score", 0.0)
                episode_scores.append(final_score)
                episode_steps.append(episode_steps_scene)
                episode_rewards.append(episode_reward)
                episode_losses.append(loss)
                episode_entropies.append(entropy)
                if policy_loss is not None:
                    episode_policy_losses.append(policy_loss)
                if value_loss is not None:
                    episode_value_losses.append(value_loss)
                if std is not None:
                    episode_stds.append(std)

            mean_score = np.mean(episode_scores)
            mean_steps = np.mean(episode_steps)
            mean_reward = np.mean(episode_rewards)
            mean_loss = np.mean(episode_losses)
            mean_entropy = np.mean(episode_entropies)
            if episode_policy_losses:
                mean_policy_loss = np.mean(episode_policy_losses)
                self.writer.add_scalar("Loss/policy_loss", mean_policy_loss, episode_count)
            if episode_value_losses:
                mean_value_loss = np.mean(episode_value_losses)
                self.writer.add_scalar("Loss/value_loss", mean_value_loss, episode_count)
            if episode_stds:
                mean_std = np.mean(episode_stds)
                self.writer.add_scalar("policy/std", mean_std, episode_count)

            # --- Logging ---
            self.writer.add_scalar("policy/entropy", mean_entropy, episode_count)
            self.writer.add_scalar("last_episode/Mean_Score", mean_score, episode_count)
            self.writer.add_scalar("last_episode/Mean_Steps", mean_steps, episode_count)
            self.writer.add_scalar("last_episode/Mean_Reward", mean_reward, episode_count)
            self.writer.add_scalar("Loss", mean_loss, episode_count)
            if hasattr(self.env, "rho"):
                self.writer.add_scalar("env/rho", self.env.rho, episode_count)

            pbar.set_postfix(
                {"Loss": f"{mean_loss:.3f}", "Mean Score (last ep)": f"{mean_score:.2f}", "Mean Steps (last ep)": f"{mean_steps:4.1f}"}
            )

            self.ep_info_buffer.append({"reward": mean_reward, "steps": mean_steps, "score": mean_score})

            # Moving averages
            recent_scores = [ep["score"] for ep in list(self.ep_info_buffer)]
            recent_steps = [ep["steps"] for ep in list(self.ep_info_buffer)]
            recent_rewards = [ep["reward"] for ep in list(self.ep_info_buffer)]

            mean_score_total = np.mean(recent_scores) if recent_scores else 0
            mean_steps_total = np.mean(recent_steps) if recent_steps else 0
            mean_reward_total = np.mean(recent_rewards) if recent_rewards else 0

            if mean_score_total > max_score:
                max_score = mean_score_total

            self.writer.add_scalar("Rollout/Mean_Reward", mean_reward_total, episode_count)
            self.writer.add_scalar("Rollout/Mean_Steps", mean_steps_total, episode_count)
            self.writer.add_scalar("Rollout/Mean_Score", mean_score_total, episode_count)
            if mean_score_total > 0:
                self.writer.add_scalar("Rollout/Steps_for_score_1", mean_steps_total / mean_score_total, episode_count)

            # Console output
            if episode_count % 5 == 0:
                print(
                    f"\nEp {episode_count:4d} | "
                    f"MA Score: {mean_score_total:5.2f} | Max Score: {max_score:5.2f} | "
                    f"MA Steps: {mean_steps_total:5.1f} | MA Reward: {mean_reward_total:6.2f}"
                )

            episode_count += 1
            pbar.update(1)

        pbar.close()
        self.writer.close()
        self.env.close()
        print("\n[INFO] Training finished.")
