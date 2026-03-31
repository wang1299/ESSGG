with open('/home/wgy/RL/components/environments/habitat_env.py', 'r') as f:
    lines = f.read()

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
        
        # Simple exploration bonus. 
        # For Habitat we do not have a gt_graph. 
        # So we clip the number of objects by an estimated max or just return a ratio based proxy.
        # But to be consistent with AI2-Thor's "recall_node" which is bounded [0, 1]
        raw_count = len(self.discovered_objects)
        # Using a proxy max obj count of e.g. 50 just so it looks like a ratio, bounded at 1.0.
        MAX_OBJ_ESTIMATE = 50.0 
        current_score = min(raw_count / MAX_OBJ_ESTIMATE, 1.0)
        
        return Observation([rgb, None, None, None], reward, terminated, truncated, {"score": float(current_score), "num_discovered": raw_count})'''

import re
start_idx = lines.find('    def _process_obs(self, obs, is_reset=False):')
end_idx = lines.find('    def get_actions(self):')

if start_idx != -1 and end_idx != -1:
    lines = lines[:start_idx] + new_process_obs + "\n\n" + lines[end_idx:]

with open('/home/wgy/RL/components/environments/habitat_env.py', 'w') as f:
    f.write(lines)
print("done")
