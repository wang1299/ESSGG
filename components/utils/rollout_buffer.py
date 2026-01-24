# In utils/rollout_buffer.py

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        # Per-step data
        self.state_rgb = []
        self.state_lssg = []
        self.state_gssg = []
        self.state_occ = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_actions = []
        self.agent_positions = []

        # Hidden states at the beginning of the rollout
        self.initial_lssg_hidden = None
        self.initial_gssg_hidden = None
        self.initial_policy_hidden = None

        self.is_first_add = True

    def clear(self):
        # Per-step data
        self.state_rgb = []
        self.state_lssg = []
        self.state_gssg = []
        self.state_occ = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_actions = []
        self.agent_positions = []

        # Hidden states at the beginning of the rollout
        self.initial_lssg_hidden = None
        self.initial_gssg_hidden = None
        self.initial_policy_hidden = None

        self.is_first_add = True

    def add(self, state, action, reward, done, hiddens, last_action, agent_position):
        # hiddens is a tuple of (lssg_hidden, gssg_hidden, policy_hidden)
        if self.is_first_add:
            # Store initial hidden states on the very first add after a clear
            self.initial_lssg_hidden = hiddens[0]
            self.initial_gssg_hidden = hiddens[1]
            self.initial_policy_hidden = hiddens[2]
            self.is_first_add = False

        s_rgb, s_lssg, s_gssg, s_occ = state
        self.state_rgb.append(s_rgb)
        self.state_lssg.append(s_lssg)
        self.state_gssg.append(s_gssg)
        self.state_occ.append(s_occ)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.last_actions.append(last_action)
        self.agent_positions.append(agent_position)

    def add_batch(self, states, actions, rewards, dones, hiddens, last_actions, agent_pos):
        """
        Adds a whole batch of transitions to the buffer at once.
        """
        if self.is_first_add and hiddens:
            self.initial_lssg_hidden = hiddens[0][0]
            self.initial_gssg_hidden = hiddens[0][1]
            self.initial_policy_hidden = hiddens[0][2]
            self.is_first_add = False

        if states:
            states_rgb, states_lssg, states_gssg, states_occ = zip(*states)
            self.state_rgb.extend(states_rgb)
            self.state_lssg.extend(states_lssg)
            self.state_gssg.extend(states_gssg)
            self.state_occ.extend(states_occ)

        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.dones.extend(dones)
        self.last_actions.extend(last_actions)
        self.agent_positions.extend(agent_pos)

    def is_ready(self):
        return len(self.rewards) >= self.n_steps

    def compute_returns(self, gamma):
        returns = []
        G = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    def get(self, gamma):
        returns = self.compute_returns(gamma)

        batch = {
            "rgb": self.state_rgb,
            "lssg": self.state_lssg,
            "gssg": self.state_gssg,
            "occ": self.state_occ,
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "returns": torch.tensor(returns, dtype=torch.float32),
            "dones": torch.tensor(self.dones, dtype=torch.bool),
            "last_actions": torch.tensor(self.last_actions, dtype=torch.long),
            "agent_positions": self.agent_positions,
            "initial_lssg_hidden": self.initial_lssg_hidden,
            "initial_gssg_hidden": self.initial_gssg_hidden,
            "initial_policy_hidden": self.initial_policy_hidden,
        }
        return batch
