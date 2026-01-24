import os

import torch
from torch import nn

from components.models.feature_encoder import FeatureEncoder
from components.models.navigation_policy import NavigationPolicy


class ImitationAgent(nn.Module):
    """
    Imitation learning agent that predicts the next action given multimodal state inputs.
    Consists of a feature encoder and a navigation policy network.
    """

    def __init__(self, navigation_config, num_actions, device=None, mapping_path=None, **kwargs):
        super().__init__()

        # Store configuration and device information
        self.navigation_config = navigation_config
        self.num_actions = num_actions
        self.use_transformer = navigation_config["use_transformer"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if mapping_path is None:
            mapping_path = os.path.join(os.path.dirname(__file__), "..", "data", "scene_graph_mappings", "default")
        # Feature encoder for multimodal input (RGB images, scene graphs, occupancy maps, etc.)
        self.encoder = FeatureEncoder(
            self.num_actions,
            rgb_dim=navigation_config["rgb_dim"],
            action_dim=navigation_config["action_dim"],
            sg_dim=navigation_config["sg_dim"],
            mapping_path=mapping_path,
            use_transformer=self.use_transformer,
        ).to(self.device)

        # Define the input dimension for the policy network based on encoder output
        input_dim = (
            navigation_config["rgb_dim"]
            + navigation_config["action_dim"]
            + 2 * navigation_config["sg_dim"]  # Local and global scene graphs
        )

        # Policy network that predicts action logits from encoded state representation
        self.policy = NavigationPolicy(
            input_dim=input_dim,
            hidden_dim=navigation_config["policy_hidden"],
            output_dim=self.num_actions,
            use_transformer=navigation_config["use_transformer"],
        ).to(self.device)

    def forward(self, x_batch, last_actions):
        """
        Forward pass through encoder and policy network.
        :param x_batch: Multimodal input representing the current state (RGB, maps, graphs)
        :param last_actions: Previously executed actions (used as context)
        :return: Raw action logits for each possible action
        """
        if isinstance(last_actions, torch.Tensor):
            last_actions = last_actions.detach()

        # Forward pass through encoder
        state_seq, _, _ = self.encoder.forward_seq(x_batch, last_actions)

        # Forward pass through policy network
        if self.use_transformer:
            gssg_mask = x_batch["gssg_mask"]
            if not isinstance(gssg_mask, torch.Tensor):
                gssg_mask = torch.tensor(gssg_mask, device=self.device)
            gssg_mask = gssg_mask.bool()
            pad_mask = ~gssg_mask
            logits, _, _ = self.policy(state_seq, pad_mask=pad_mask)
        else:
            logits, _, _ = self.policy(state_seq)
        return logits

    def save_model(self, path):
        """
        Saves both the feature encoder and navigation policy.
        """
        os.makedirs(path, exist_ok=True)
        # Save encoder and policy separately in the specified directory
        self.encoder.save_model(path)
        # self.policy.save_model(path)

    def load_weights(self, encoder_path, policy_path, device="cpu"):
        self.encoder.load_weights(encoder_path, device=device)
        self.policy.load_weights(policy_path, device=device)
        self.to(device)
