import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from components.agents.abstract_agent import AbstractAgent


class ReinforceAgent(AbstractAgent):
    """
    REINFORCE policy-gradient agent using a pretrained feature encoder.
    """

    def __init__(self, env, navigation_config, agent_config, device=None, mapping_path=None):
        super().__init__(env, navigation_config, agent_config, device, mapping_path)
        # self.entropy_coef = agent_config["entropy_coef"]

    def update(self):
        batch = self._get_update_values()

        logits = self.forward_update(batch)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)

        if not isinstance(batch["actions"], torch.Tensor):
            actions = torch.tensor(batch["actions"], device=self.device).view(-1)
        else:
            actions = batch["actions"].to(self.device).view(-1)
        if not isinstance(batch["returns"], torch.Tensor):
            returns = torch.tensor(batch["returns"], device=self.device).view(-1)
        else:
            returns = batch["returns"].to(self.device).view(-1)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        if returns.numel() > 1:
            ret_std = returns.std().item()
        else:
            ret_std = 1.0
        ret_std = ret_std if ret_std > 1e-2 else 1e-2
        returns = (returns - returns.mean()) / ret_std

        policy_loss = -torch.mean(log_probs * returns)
        loss = policy_loss  # - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.reset()

        result = {"loss": loss.item(), "entropy": entropy.item(), "ret_std": ret_std}
        return result

    def get_agent_info(self):
        """
        Return basic information about the agent.
        :return: Dictionary with agent info
        """
        return {"Agent Name": "REINFORCE Agent", "alpha": self.alpha, "gamma": self.gamma}
