with open('/home/wgy/RL/components/environments/habitat_env.py', 'r') as f:
    lines = f.read()

# Add discovered_objects reset
lines = lines.replace('''    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.sim.reset()''',
'''    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.discovered_objects = set()
        self.cumulative_reward = 0.0
        self.sim.reset()''')

# Fix _process_obs to update discovered_objects and cumulative_reward
new_process_obs = '''    def _process_obs(self, obs, is_reset=False):
        rgb = obs["color_sensor"]
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
            
        # Optional: Run detector if enabled
        if self.use_detector and self.detector:
             depth = obs["depth_sensor"]
             agent_state = self.sim.get_agent(0).get_state()
             
             as_dict = {
                 'position': {'x': agent_state.position[0], 'y': agent_state.position[1], 'z': agent_state.position[2]},
                 'rotation': {'x': agent_state.rotation.x, 'y': agent_state.rotation.y, 'z': agent_state.rotation.z, 'w': agent_state.rotation.w}
             }
             
             try:
                 detections = self.detector.detect(rgb, depth_image=depth, agent_state=as_dict)
                 for det in detections:
                     if det.get("score", 0) >= getattr(self, "det_score_thr", 0.2):
                         self.discovered_objects.add(det.get("label", "unknown"))
             except Exception as e:
                 print(f"Warning: Detector failed: {e}")

        truncated = self.step_count >= self.max_actions
        terminated = False
        
        # Give reward for discovery and penalty for step
        reward = 0.0
        if not is_reset:
            reward = -self.rho
        
        # Simple exploration bonus (optional, or just track score)
        # Using the length of discovered objects as a simple score proxy for logs
        current_score = len(self.discovered_objects)
        
        return Observation([rgb, None, None, None], reward, terminated, truncated, {"score": float(current_score)})'''

# Finding where to replace
import re
start_idx = lines.find('    def _process_obs(self, obs, is_reset=False):')
end_idx = lines.find('    def get_actions(self):')

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + new_process_obs + "\n\n" + lines[end_idx:]
else:
    print("Could not find _process_obs block")

with open('/home/wgy/RL/components/environments/habitat_env.py', 'w') as f:
    f.write(lines)
print("Finished adding score to HabitatEnv")
