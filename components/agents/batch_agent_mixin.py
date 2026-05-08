"""
Batch-enabled agent methods for multi-environment parallel inference.
Extends AbstractAgent to handle multiple observations at once.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class BatchAgentMixin:
    """
    Mixin to add batch inference capabilities to agent classes.
    Call get_batch_actions() to get actions for multiple environments at once.
    
    Note: For simplicity, we handle each observation independently and stack results.
    A fully optimized version would batch-process at the encoder level.
    """

    def get_batch_actions(self, obs_list, deterministic=False):
        """
        Get actions for a batch of observations (one per environment).
        
        Args:
            obs_list: List of Observation objects, length = num_envs
            deterministic: If True, return argmax action; else sample from distribution
        
        Returns:
            actions: np.array of shape [num_envs,]
            values: np.array of shape [num_envs,] or None
        """
        if not obs_list:
            return np.array([]), None
        
        num_envs = len(obs_list)
        actions_list = []
        values_list = []
        
        with torch.no_grad():
            for obs in obs_list:
                # Process each obs individually
                # This is not perfectly optimal, but safe and compatible with current encoder design
                logits, value = self.forward(obs)
                
                probs = F.softmax(logits, dim=-1)
                
                if deterministic:
                    action = torch.argmax(probs, dim=-1).cpu().numpy()
                else:
                    dist = Categorical(probs=probs)
                    action = dist.sample().cpu().numpy()
                
                actions_list.append(action.item() if action.ndim > 0 else action)
                
                if value is not None:
                    values_list.append(value.cpu().numpy().item())
        
        actions = np.array(actions_list, dtype=np.int64)
        values = np.array(values_list) if values_list else None
        
        return actions, values
