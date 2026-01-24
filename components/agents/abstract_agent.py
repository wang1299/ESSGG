import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from components.models.feature_encoder import FeatureEncoder
from components.models.navigation_policy import NavigationPolicy
from components.utils.rollout_buffer import RolloutBuffer


class AbstractAgent(nn.Module):
    """
    Gemeinsame Basis für Policy-Gradient-Agenten (REINFORCE, A2C, etc.)
    """

    def __init__(self, env, navigation_config, agent_config, device=None, mapping_path=None):
        super().__init__()
        self.env = env
        self.navigation_config = navigation_config
        self.agent_config = agent_config
        self.alpha = agent_config.get("alpha", 1e-4)
        self.gamma = agent_config.get("gamma", 0.99)
        self.use_transformer = navigation_config["use_transformer"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = len(env.get_actions())
        self.all_scene_numbers = list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))
        self.num_scenes = 10
        self.scene_numbers = self.all_scene_numbers[: self.num_scenes]
        if mapping_path is None:
            mapping_path = os.path.join(os.path.dirname(__file__), "..", "data", "scene_graph_mappings", "default")
        # Feature encoder
        self.encoder = FeatureEncoder(
            self.num_actions,
            rgb_dim=navigation_config["rgb_dim"],
            action_dim=navigation_config["action_dim"],
            sg_dim=navigation_config["sg_dim"],
            mapping_path=mapping_path,
            use_transformer=self.use_transformer,
        ).to(self.device)
        # Policy input dim
        self.input_dim = int(navigation_config["rgb_dim"] + navigation_config["action_dim"] + 2 * navigation_config["sg_dim"])

        self.policy = NavigationPolicy(
            input_dim=self.input_dim,
            hidden_dim=navigation_config["policy_hidden"],
            output_dim=self.num_actions,
            use_transformer=navigation_config["use_transformer"],
            value_head=True if agent_config["name"] in ["a2c"] else False,
            device=self.device,
        ).to(self.device)

        # Adam optimizer for ALL parameters
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        # Internal buffers
        self.rollout_buffers = RolloutBuffer(agent_config["num_steps"])
        self.last_action = -1
        self.lssg_hidden = None
        self.gssg_hidden = None
        self.policy_hidden = None
        self.obs_buffer = []
        self.action_buffer = []

        self.to(self.device)

    def forward(self, obs):
        if self.use_transformer:
            self.obs_buffer.append(obs)
            if len(self.action_buffer) == 0:
                self.action_buffer.append(-1)
            else:
                self.action_buffer.append(self.last_action)
            state_vector, _, _ = self.encoder(self.obs_buffer, self.action_buffer)
            logits, value, _ = self.policy(state_vector)
        else:
            state_vector, self.lssg_hidden, self.gssg_hidden = self.encoder(
                obs, self.last_action, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden
            )
            logits, value, self.policy_hidden = self.policy(state_vector, hidden=self.policy_hidden)

        return logits, value.squeeze(-1) if value is not None else None

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits, value = self.forward(obs)
            if self.use_transformer:
                probs = F.softmax(logits[:, -1], dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = dist.sample().item()
        last_action = self.last_action
        self.last_action = action
        if self.use_transformer:
            return action, None, None, None, last_action, value.reshape(-1)[-1].item() if value is not None else None
        else:
            return action, self.lssg_hidden, self.gssg_hidden, self.policy_hidden, last_action, value.item() if value is not None else None

    def reset(self):
        self.last_action = -1
        self.rollout_buffers.clear()
        if self.use_transformer:
            self.obs_buffer.clear()
            self.action_buffer.clear()
        else:
            self.lssg_hidden = None
            self.gssg_hidden = None
            self.policy_hidden = None

    def _get_update_values(self):
        b = self.rollout_buffers.get(self.gamma)
        batch = {k: b[k] for k in ["rgb", "lssg", "gssg", "occ", "actions", "returns", "last_actions", "agent_positions"]}

        for k in ["actions", "returns", "last_actions"]:
            if not isinstance(batch[k], torch.Tensor):
                batch[k] = torch.tensor(batch[k], device=self.device)
            if batch[k].dim() == 1:
                batch[k] = batch[k].unsqueeze(0)  # [T] → [1,T]

        for k in ["rgb", "lssg", "gssg", "occ", "agent_positions"]:
            if isinstance(batch[k], list):
                batch[k] = [batch[k]]

        return batch

    def forward_update(self, batch):
        state_seq, _, _ = self.encoder.forward_seq(batch, batch["last_actions"])
        logits, value, _ = self.policy(state_seq, hidden=self.policy_hidden)

        if value is None:
            return logits
        else:
            value = value.squeeze(0)
            return logits, value

    def load_weights(self, encoder_path=None, model_path=None, device="cpu"):
        if encoder_path is not None:
            self.encoder.load_weights(encoder_path, device=device)
        elif model_path is not None:
            try:
            # PyTorch ≥ 2.6 默认 weights_only=True，这里显式关闭
                obj = torch.load(model_path, map_location=device, weights_only=False)
            except TypeError:
            # 旧版 PyTorch 没有 weights_only 参数
                obj = torch.load(model_path, map_location=device)

            # 兼容几种老式保存格式
            if isinstance(obj, dict):
                if "state_dict" in obj:
                    state_dict = obj["state_dict"]
                elif "model_state_dict" in obj:
                    state_dict = obj["model_state_dict"]
                else:
                    state_dict = obj
            else:
            # 直接保存了整个 nn.Module
                state_dict = obj.state_dict()

            self.load_state_dict(state_dict)
            self.to(device)
        else:
            raise Exception("No encoder or model specified")

    def save_model(self, path, file_name=None, save_encoder=False):
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        # Gather relevant config parameters for filename
        rgb_dim = self.encoder.rgb_encoder.output_dim
        action_dim = self.encoder.action_emb.embedding.embedding_dim
        sg_dim = self.encoder.lssg_encoder.lstm.hidden_size if not self.use_transformer else self.encoder.lssg_encoder.output_dim
        policy_hidden = self.policy_hidden

        agent_name = self.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        suffix = "_Transformer" if self.use_transformer else "_LSTM"
        model_dir = base_path / f"{agent_name}{suffix}"
        model_dir.mkdir(exist_ok=True)

        # Save encoder separately in the specified directory
        if save_encoder:
            self.encoder.save_model(model_dir)

        # Create filename including config parameters
        filename = f"{agent_name}{suffix}_{rgb_dim}_{action_dim}_{sg_dim}_{policy_hidden}.pth" if file_name is None else file_name
        full_path = model_dir / filename

        # Save model state dict
        torch.save(self.state_dict(), str(full_path))
        print(f"Saved model to {full_path}")

    def get_agent_info(self):
        """
        Return basic information about the agent.
        """
        pass
