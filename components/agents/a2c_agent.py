import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from components.agents.abstract_agent import AbstractAgent


class A2CAgent(AbstractAgent):
    """
    Advantage Actor-Critic agent with shared encoder, policy and value head.
    """

    def __init__(self, env, navigation_config, agent_config, device=None, mapping_path=None):
        super().__init__(env, navigation_config, agent_config, device, mapping_path)
        self.value_coef = agent_config.get("value_coef", 0.5)
        self.entropy_coef = agent_config.get("entropy_coef", 0.1)

    def update(self, obs=None):
        batch = self._get_update_values()

        logits, values = self.forward_update(batch)
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

        # Advantage
        advantages = returns - values.detach()

        policy_loss = -torch.mean(log_probs * advantages)
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.reset()

        result = {"loss": loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "entropy": entropy.item()}
        return result

    def get_agent_info(self):
        """
        Return basic information about the agent.
        :return: Dictionary with agent info
        """
        return {"Agent Name": "A2C Agent", "alpha": self.alpha, "gamma": self.gamma, "entropy_coef": self.entropy_coef}
