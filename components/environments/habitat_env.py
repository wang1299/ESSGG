import habitat_sim
import numpy as np
import os
import random
import csv
from collections import deque
from components.utils.observation import Observation
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn

class HabitatEnv:
    def __init__(
        self,
        dataset_root,
        config_file,
        scene_id,
        render=False,
        width=300,
        height=300,
        use_detector=False,
        detector=None,
        det_score_thr=0.3,
        score_norm_target=120.0,
        instance_merge_dist=0.8,
        coverage_cell_size=0.5,
        nav_sample_points=4000,
        topdown_meters_per_pixel=0.05,
        fill_position_from_gt=False,
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
            os.makedirs(self.save_debug_path, exist_ok=True)
        self.width = width
        self.height = height
        self.dataset_root = dataset_root
        self.use_detector = use_detector
        self.detector = detector
        self.det_score_thr = det_score_thr
        self.score_norm_target = max(float(score_norm_target), 1.0)
        self.instance_merge_dist = max(float(instance_merge_dist), 1e-6)
        self.coverage_cell_size = max(float(coverage_cell_size), 1e-6)
        self.nav_sample_points = max(int(nav_sample_points), 200)
        self.topdown_meters_per_pixel = max(float(topdown_meters_per_pixel), 1e-3)
        self.total_navigable_cells = None
        self.topdown_base_img = None
        self.topdown_bounds = None
        self.topdown_shape = None
        self.traj_pixels = deque()
        
        # Change working directory so habitat can find assets relative to config
        self.initial_cwd = os.getcwd()
        if os.path.exists(dataset_root):
            os.chdir(dataset_root)
        else:
            print(f"Warning: Dataset root {dataset_root} not found. Continuing in {os.getcwd()}")

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = config_file
        sim_cfg.scene_id = scene_id
        sim_cfg.enable_physics = False
        sim_cfg.force_separate_semantic_scene_graph = True
        
        # Create agent config
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        
        # Add sensors
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [height, width]
        
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [height, width]
        
        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
        
        # Action Space
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_left": habitat_sim.agent.ActionSpec(
                "move_left", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_right": habitat_sim.agent.ActionSpec(
                "move_right", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0) 
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
        }
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        try:
            self.sim = habitat_sim.Simulator(cfg)
        except Exception as e:
            print(f"Error initializing Habitat Simulator: {e}")
            # Restore CWD before raising
            os.chdir(self.initial_cwd)
            raise e
        
        # Restore CWD
        os.chdir(self.initial_cwd)

        # Actions mapping matching ThorEnv
        # ["RotateRight", "RotateLeft", "Pass", "MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]
        self.action_mapping = [
            "turn_right",     # 0: RotateRight
            "turn_left",      # 1: RotateLeft
            None,             # 2: Pass
            "move_forward",   # 3: MoveAhead
            "move_right",     # 4: MoveRight
            "move_left",      # 5: MoveLeft
            "move_backward",  # 6: MoveBack
            None              # 7: Pass
        ]

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.discovered_objects = set()
        self.discovered_instances = set()
        self.visited_cells = set()
        self.traj_pixels.clear()
        self.prev_score = 0.0
        self.cumulative_reward = 0.0
        self.scene_number = scene_number if scene_number is not None else 1
        
        if self.save_debug_path:
            self.current_ep_dir = os.path.join(self.save_debug_path, f"ep_{getattr(self, 'episode_id', 0):04d}_scene_{self.scene_number}")
            if not os.path.exists(self.current_ep_dir):
                os.makedirs(self.current_ep_dir, exist_ok=True)
        else:
            self.current_ep_dir = None
            
        self.sim.reset()

        if self.sim.pathfinder.is_loaded and self.total_navigable_cells is None:
            self.total_navigable_cells = self._estimate_navigable_cells()
        
        # Random spawn on navmesh
        if self.sim.pathfinder.is_loaded:
             random_position = self.sim.pathfinder.get_random_navigable_point()
             agent = self.sim.get_agent(0)
             agent_state = agent.get_state()
             agent_state.position = random_position
             
             # Random rotation
             angle = random.uniform(0, 2 * np.pi)
             agent_state.rotation = quat_from_angle_axis(angle, np.array([0.0, 1.0, 0.0]))
             agent.set_state(agent_state)
        
        obs = self.sim.get_sensor_observations()

        # Build reusable topdown base map for this episode.
        if self.current_ep_dir is not None:
            self._prepare_topdown_base_map()

        if self.current_ep_dir is not None:
            self.trajectory_csv = os.path.join(self.current_ep_dir, "trajectory.csv")
            with open(self.trajectory_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "x", "y", "z", "yaw_deg", "score", "coverage", "num_instances"])

        return self._process_obs(obs, is_reset=True)

    def step(self, action_id):
        self.step_count += 1
        # Handle tensor inputs
        if hasattr(action_id, 'item'):
            action_id = action_id.item()
            
        action_name = self.action_mapping[action_id] if 0 <= action_id < len(self.action_mapping) else None
        
        if action_name:
            obs = self.sim.step(action_name)
        else:
            obs = self.sim.get_sensor_observations()
            
        return self._process_obs(obs)

    def _process_obs(self, obs, is_reset=False):
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
                     if det.get("score", 0) >= self.det_score_thr:
                         label = det.get("label", "unknown")
                         self.discovered_objects.add(label)
                         self.discovered_instances.add(self._build_instance_key(det, label))
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
                    if score < self.det_score_thr: continue
                    
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

        # Track trajectory and occupancy-like coverage from visited world cells.
        agent_state = self.sim.get_agent(0).get_state()
        ax = float(agent_state.position[0])
        ay = float(agent_state.position[1])
        az = float(agent_state.position[2])
        cell = self._quantize_world_cell(ax, az)
        self.visited_cells.add(cell)
        denom = max(self.total_navigable_cells or 1, 1)
        coverage = min(len(self.visited_cells) / denom, 1.0)
        
        # Give reward for discovery and penalty for step
        discovered_instance_count = len(self.discovered_instances)
        discovered_label_count = len(self.discovered_objects)
        norm_count = discovered_instance_count if discovered_instance_count > 0 else discovered_label_count
        current_score = min(norm_count / self.score_norm_target, 1.0)

        if is_reset:
            reward = 0.0
        else:
            # Use score improvement as learning signal, with a small step penalty.
            reward = (current_score - self.prev_score) - self.rho
        self.prev_score = current_score

        yaw_deg = float(self._yaw_from_quat(agent_state.rotation))
        if self.current_ep_dir is not None:
            self._save_topdown_step(ax, az)

        if self.current_ep_dir is not None:
            with open(self.trajectory_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    int(self.step_count),
                    round(ax, 4),
                    round(ay, 4),
                    round(az, 4),
                    round(yaw_deg, 2),
                    round(float(current_score), 6),
                    round(float(coverage), 6),
                    int(discovered_instance_count),
                ])

        if (terminated or truncated) and self.current_ep_dir is not None:
            self._save_topdown_trajectory()

        return Observation(
            [rgb, None, None, None],
            reward,
            terminated,
            truncated,
            {
                "score": float(current_score),
                "num_discovered": norm_count,
                "num_discovered_labels": discovered_label_count,
                "num_discovered_instances": discovered_instance_count,
                "coverage": float(coverage),
                "visited_cells": len(self.visited_cells),
                "total_navigable_cells": int(denom),
                "agent_pos": (ax, az),
            },
        )

    def _quantize_world_cell(self, x, z):
        qx = round(float(x) / self.coverage_cell_size) * self.coverage_cell_size
        qz = round(float(z) / self.coverage_cell_size) * self.coverage_cell_size
        return (round(qx, 3), round(qz, 3))

    def _estimate_navigable_cells(self):
        points = set()
        for _ in range(self.nav_sample_points):
            p = self.sim.pathfinder.get_random_navigable_point()
            points.add(self._quantize_world_cell(float(p[0]), float(p[2])))
        return max(len(points), 1)

    def _yaw_from_quat(self, q):
        # q is [x, y, z, w] in Habitat. Extract yaw around Y axis.
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    def _prepare_topdown_base_map(self):
        try:
            if not self.sim.pathfinder.is_loaded:
                return
            td = self.sim.pathfinder.get_topdown_view(
                meters_per_pixel=self.topdown_meters_per_pixel,
                height=0.15,
            )
            td_u8 = np.where(td, 245, 30).astype(np.uint8)
            self.topdown_base_img = np.stack([td_u8, td_u8, td_u8], axis=-1)
            self.topdown_shape = td.shape
            self.topdown_bounds = self.sim.pathfinder.get_bounds()
        except Exception as e:
            print(f"Failed to prepare topdown base map: {e}")
            self.topdown_base_img = None
            self.topdown_shape = None
            self.topdown_bounds = None

    def _world_to_topdown_px(self, x, z):
        if self.topdown_shape is None or self.topdown_bounds is None:
            return None
        min_b, max_b = self.topdown_bounds
        min_x, max_x = float(min_b[0]), float(max_b[0])
        min_z, max_z = float(min_b[2]), float(max_b[2])
        if max_x <= min_x or max_z <= min_z:
            return None

        h, w = int(self.topdown_shape[0]), int(self.topdown_shape[1])
        col = int(np.clip((float(x) - min_x) / (max_x - min_x) * (w - 1), 0, w - 1))
        row_from_bottom = int(np.clip((float(z) - min_z) / (max_z - min_z) * (h - 1), 0, h - 1))
        row = (h - 1) - row_from_bottom
        return (col, row)

    def _save_topdown_step(self, x, z):
        if self.topdown_base_img is None:
            return
        try:
            from PIL import Image, ImageDraw

            px = self._world_to_topdown_px(x, z)
            if px is None:
                return
            self.traj_pixels.append(px)

            img = Image.fromarray(self.topdown_base_img.copy(), mode="RGB")
            draw = ImageDraw.Draw(img)
            # Draw short history for better readability of local motion.
            if len(self.traj_pixels) >= 2:
                recent = list(self.traj_pixels)[-40:]
                draw.line(recent, fill=(255, 180, 40), width=2)

            r = 5
            draw.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r), fill=(255, 40, 40), outline=(255, 255, 255), width=1)
            save_path = os.path.join(self.current_ep_dir, f"topdown_{self.step_count:04d}.png")
            img.save(save_path)
        except Exception as e:
            print(f"Failed to save topdown step map: {e}")

    def _save_topdown_trajectory(self):
        if self.topdown_base_img is None or len(self.traj_pixels) == 0:
            return
        try:
            from PIL import Image, ImageDraw

            img = Image.fromarray(self.topdown_base_img.copy(), mode="RGB")
            draw = ImageDraw.Draw(img)

            pts = list(self.traj_pixels)
            if len(pts) >= 2:
                draw.line(pts, fill=(80, 220, 80), width=2)

            # Start and end markers.
            s = pts[0]
            e = pts[-1]
            rs = 5
            re = 6
            draw.ellipse((s[0] - rs, s[1] - rs, s[0] + rs, s[1] + rs), fill=(80, 140, 255), outline=(255, 255, 255), width=1)
            draw.ellipse((e[0] - re, e[1] - re, e[0] + re, e[1] + re), fill=(255, 60, 60), outline=(255, 255, 255), width=1)

            save_path = os.path.join(self.current_ep_dir, "topdown_trajectory.png")
            img.save(save_path)
        except Exception as e:
            print(f"Failed to save topdown trajectory map: {e}")

    def _build_instance_key(self, det, label):
        pos = det.get("position")
        if isinstance(pos, dict):
            x = pos.get("x", None)
            z = pos.get("z", None)
            if x is not None and z is not None:
                qx = round(float(x) / self.instance_merge_dist) * self.instance_merge_dist
                qz = round(float(z) / self.instance_merge_dist) * self.instance_merge_dist
                return (label, round(qx, 3), round(qz, 3))

        box = det.get("bbox", det.get("box"))
        if box is not None and len(box) == 4:
            cx = 0.5 * (float(box[0]) + float(box[2]))
            cy = 0.5 * (float(box[1]) + float(box[3]))
            bw = float(box[2]) - float(box[0])
            bh = float(box[3]) - float(box[1])
            return (label, round(cx, 1), round(cy, 1), round(bw, 1), round(bh, 1))

        return (label,)

    def get_actions(self):
        # Keep same as ThorEnv for compatibility
        return ["RotateRight", "RotateLeft", "Pass", "MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]

    def close(self):
        self.sim.close()
