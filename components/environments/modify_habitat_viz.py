with open('/home/wgy/RL/components/environments/habitat_env.py', 'r') as f:
    lines = f.read()

# Replace init
lines = lines.replace('''        fill_position_from_gt=False,
        rho=0.1,
        max_actions=40
    ):
        self.rho = rho
        self.max_actions = max_actions
        self.step_count = 0''',
'''        fill_position_from_gt=False,
        rho=0.1,
        max_actions=40,
        save_debug_path=None
    ):
        self.rho = rho
        self.max_actions = max_actions
        self.step_count = 0
        self.save_debug_path = save_debug_path
        self.episode_id = 0
        
        if self.save_debug_path and not os.path.exists(self.save_debug_path):
            os.makedirs(self.save_debug_path, exist_ok=True)''')


# Fix reset
lines = lines.replace('''    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.discovered_objects = set()
        self.cumulative_reward = 0.0
        self.sim.reset()''',
'''    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.discovered_objects = set()
        self.cumulative_reward = 0.0
        self.scene_number = scene_number if scene_number is not None else 1
        
        if self.save_debug_path:
            self.current_ep_dir = os.path.join(self.save_debug_path, f"ep_{getattr(self, 'episode_id', 0):04d}_scene_{self.scene_number}")
            if not os.path.exists(self.current_ep_dir):
                os.makedirs(self.current_ep_dir, exist_ok=True)
        else:
            self.current_ep_dir = None
            
        self.sim.reset()''')


# Fix _process_obs to save frames
new_process_obs = '''    def _process_obs(self, obs, is_reset=False):
        rgb = obs["color_sensor"]
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
            
        detections = []
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

        # Save Visualizations
        if getattr(self, "current_ep_dir", None) is not None:
            try:
                from PIL import Image, ImageDraw, ImageFont
                VIZ_SCALE = 3
                orig_h, orig_w = rgb.shape[:2]
                vis_img = Image.fromarray(rgb.astype('uint8'), 'RGB')
                vis_img = vis_img.resize((orig_w * VIZ_SCALE, orig_h * VIZ_SCALE), Image.Resampling.NEAREST)
                draw = ImageDraw.Draw(vis_img)
                
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None
                
                for det in detections:
                    box = det.get("bbox", det.get("box"))
                    label = det.get("label", "unknown")
                    score = det.get("score", 0)
                    if box is None: continue
                    if score < getattr(self, "det_score_thr", 0.2): continue
                    
                    sbox = [c * VIZ_SCALE for c in box]
                    color = "green"
                    draw.rectangle(sbox, outline=color, width=max(1, VIZ_SCALE))
                    text = f"{label} {score:.0%}"
                    if font:
                        draw.text((sbox[0]+2, max(0, sbox[1]-10*VIZ_SCALE)), text, fill=color)
                
                save_path = os.path.join(self.current_ep_dir, f"frame_{self.step_count:04d}.png")
                vis_img.save(save_path)
            except Exception as e:
                print(f"Failed to save viz frame: {e}")

        truncated = self.step_count >= self.max_actions
        terminated = False
        
        # Give reward for discovery and penalty for step
        reward = 0.0
        if not is_reset:
            reward = -self.rho
        
        raw_count = len(self.discovered_objects)
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
