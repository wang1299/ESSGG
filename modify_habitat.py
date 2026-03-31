with open('/home/wgy/RL/components/environments/habitat_env.py', 'r') as f:
    lines = f.read()

# Replace init
lines = lines.replace('''        fill_position_from_gt=False,
        rho=0.1
    ):
        self.rho = rho''',
'''        fill_position_from_gt=False,
        rho=0.1,
        max_actions=40
    ):
        self.rho = rho
        self.max_actions = max_actions
        self.step_count = 0''')

# Replace reset
lines = lines.replace('''    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.sim.reset()''',
'''    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.sim.reset()''')

lines = lines.replace('''        obs = self.sim.get_sensor_observations()
        return self._process_obs(obs)''',
'''        obs = self.sim.get_sensor_observations()
        return self._process_obs(obs, is_reset=True)''')

# Replace step
lines = lines.replace('''    def step(self, action_id):
        # Handle tensor inputs''',
'''    def step(self, action_id):
        self.step_count += 1
        # Handle tensor inputs''')

# Replace _process_obs
lines = lines.replace('''    def _process_obs(self, obs):''',
'''    def _process_obs(self, obs, is_reset=False):''')

lines = lines.replace('''        return Observation([rgb, None, None, None], 0.0, False, False, {})''',
'''        truncated = self.step_count >= self.max_actions
        terminated = False
        reward = 0.0 if (is_reset or terminated) else -self.rho
        return Observation([rgb, None, None, None], reward, terminated, truncated, {})''')

with open('/home/wgy/RL/components/environments/habitat_env.py', 'w') as f:
    f.write(lines)
print("Finished rewriting HabitatEnv")
