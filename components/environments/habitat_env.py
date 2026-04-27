import habitat_sim
import numpy as np
import os
import random
import csv
import sys
import contextlib
import re
from collections import Counter, deque
from components.utils.observation import Observation
from habitat_sim.utils.common import quat_from_angle_axis
import magnum as mn


@contextlib.contextmanager
def _suppress_native_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull)

class HabitatEnv:
    def __init__(
        self,
        dataset_root,
        config_file,
        scene_id,
        scene_ids=None,
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
        agent_radius=0.17,
        agent_height=1.5,
        agent_max_climb=0.1,
        navmesh_cell_height=0.05,
        navmesh_cell_size=0.03,
        fill_position_from_gt=False,
        rho=0.1,
        coverage_bonus_scale=2.0,
        discovery_bonus_scale=1.0,
        collision_penalty=0.05,
        max_actions=40,
        save_debug_path=None
    ):
        self.rho = rho
        self.max_actions = max_actions
        self.step_count = 0
        self.save_debug_path = save_debug_path
        self.episode_id = 0
        self.render = render
        self.config_file = config_file
        self.base_scene_id = scene_id
        self.scene_ids = self._normalize_scene_ids(scene_id, scene_ids)
        self.scene_id = self.scene_ids[0]
        self.current_scene_index = 0
        
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
        self.agent_radius = max(float(agent_radius), 1e-6)
        self.agent_height = max(float(agent_height), 1e-6)
        self.agent_max_climb = max(float(agent_max_climb), 0.0)
        self.navmesh_cell_height = max(float(navmesh_cell_height), 1e-6)
        self.navmesh_cell_size = max(float(navmesh_cell_size), 1e-6)
        self.coverage_bonus_scale = float(coverage_bonus_scale)
        self.discovery_bonus_scale = float(discovery_bonus_scale)
        self.collision_penalty = max(float(collision_penalty), 0.0)
        self.total_navigable_cells = None
        self.topdown_base_img = None
        self.topdown_bounds = None
        self.topdown_shape = None
        self.traj_pixels = deque()
        self.semantic_id_to_label = {}
        self.last_action_name = None
        self.last_start_position_by_scene = {}
        
        # Change working directory so habitat can find assets relative to config
        self.initial_cwd = os.getcwd()
        if os.path.exists(dataset_root):
            os.chdir(dataset_root)
        else:
            print(f"Warning: Dataset root {dataset_root} not found. Continuing in {os.getcwd()}")

        self._load_scene(self.scene_id)
        
        # Restore CWD
        os.chdir(self.initial_cwd)

        # Custom simplified actions
        self.action_mapping = [
            "turn_left",      # 0
            "turn_right",     # 1
            "move_forward",   # 2
        ]

    def _normalize_scene_ids(self, scene_id, scene_ids):
        if scene_ids:
            normalized = [str(scene).strip() for scene in scene_ids if str(scene).strip()]
            return normalized if normalized else [scene_id]
        return [scene_id]

    def _get_largest_island_index(self):
        pf = self.sim.pathfinder
        if not pf.is_loaded or pf.num_islands <= 0:
            return None

        largest_island_idx = -1
        max_area = -1.0
        for i in range(pf.num_islands):
            area = pf.island_area(i)
            if area > max_area:
                max_area = area
                largest_island_idx = i

        return largest_island_idx if largest_island_idx >= 0 else None

    def _snap_to_largest_island(self, position):
        if position is None or not self.sim.pathfinder.is_loaded:
            return position

        pf = self.sim.pathfinder
        largest_island_idx = self._get_largest_island_index()
        candidate = pf.snap_point(np.array(position, dtype=np.float32))

        if largest_island_idx is None:
            return candidate
        if candidate is None:
            return pf.get_random_navigable_point(max_tries=10, island_index=largest_island_idx)

        try:
            candidate_island = pf.get_island(candidate)
        except Exception:
            candidate_island = largest_island_idx

        if candidate_island != largest_island_idx:
            return pf.get_random_navigable_point(max_tries=10, island_index=largest_island_idx)

        return candidate

    def _create_simulator_config(self, scene_id):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = self.config_file
        sim_cfg.scene_id = scene_id
        sim_cfg.enable_physics = False
        sim_cfg.force_separate_semantic_scene_graph = True

        agent_cfg = habitat_sim.agent.AgentConfiguration()

        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [self.height, self.width]

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [self.height, self.width]

        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [self.height, self.width]

        agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

        agent_cfg.action_space = {
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def _load_scene(self, scene_id):
        if hasattr(self, "sim") and self.sim is not None:
            try:
                self.sim.close()
            except Exception:
                pass

        cfg = self._create_simulator_config(scene_id)
        try:
            with _suppress_native_output():
                self.sim = habitat_sim.Simulator(cfg)
                
                # Recompute navmesh based on custom agent dimensions
                navmesh_settings = habitat_sim.NavMeshSettings()
                navmesh_settings.set_defaults()
                navmesh_settings.agent_radius = self.agent_radius
                navmesh_settings.agent_height = self.agent_height
                navmesh_settings.agent_max_climb = self.agent_max_climb
                navmesh_settings.cell_height = self.navmesh_cell_height
                navmesh_settings.cell_size = self.navmesh_cell_size
                
                # Recompute the navmesh
                success = self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
                if not success:
                    print(f"Warning: Failed to recompute navmesh for scene {scene_id}")

                # Start positions are constrained to the largest island during reset.
        except Exception as e:
            print(f"Error initializing Habitat Simulator: {e}")
            os.chdir(self.initial_cwd)
            raise e

        self.scene_id = scene_id
        self.semantic_id_to_label = self._build_semantic_id_label_map()
        self.total_navigable_cells = None
        self.topdown_base_img = None
        self.topdown_bounds = None
        self.topdown_shape = None
        self.traj_pixels = deque()

    def _resolve_scene_index(self, scene_number):
        if not self.scene_ids:
            return 0
        if scene_number is None:
            return self.current_scene_index % len(self.scene_ids)

        try:
            idx = int(scene_number) - 1
        except (TypeError, ValueError):
            idx = self.current_scene_index
        return idx % len(self.scene_ids)

    def _sample_random_start_position(self, min_distance=0.75, max_trials=24):
        if not self.sim.pathfinder.is_loaded:
            return None

        pf = self.sim.pathfinder
        largest_island_idx = self._get_largest_island_index()
        if largest_island_idx is None:
            return pf.get_random_navigable_point(max_tries=10)

        last_pos = self.last_start_position_by_scene.get(self.scene_id)
        best_candidate = None
        best_dist = -1.0

        def _xz(position):
            if position is None:
                return None
            if len(position) >= 3:
                return float(position[0]), float(position[2])
            if len(position) >= 2:
                return float(position[0]), float(position[1])
            return None

        for _ in range(max_trials):
            candidate = pf.get_random_navigable_point(max_tries=10, island_index=largest_island_idx)
            if candidate is None:
                continue
            if last_pos is None:
                return candidate

            candidate_xz = _xz(candidate)
            last_xz = _xz(last_pos)
            if candidate_xz is None or last_xz is None:
                continue

            dist = float(np.hypot(candidate_xz[0] - last_xz[0], candidate_xz[1] - last_xz[1]))
            if dist >= min_distance:
                return candidate
            if dist > best_dist:
                best_dist = dist
                best_candidate = candidate

        if best_candidate is not None:
            return best_candidate
        return pf.get_random_navigable_point(max_tries=10, island_index=largest_island_idx)

    def _sample_poi_start_position(self, scene_id):
        import json
        scene_hash = scene_id.split("-")[0].split("/")[-1] if "/" in scene_id else scene_id.split("-")[0]
        poi_file = f"/home/wgy/RL/pois/{scene_hash}_poi.json"
        
        if os.path.exists(poi_file):
            try:
                with open(poi_file, "r") as f:
                    data = json.load(f)
                    pois = data.get("poi", [])
                    if pois:
                        import random
                        chosen_poi = random.choice(pois)
                        return np.array(chosen_poi["position"], dtype=np.float32)
            except Exception as e:
                print(f"Error loading POI file {poi_file}: {e}")
        
        # Fallback to random navigable point
        return self._sample_random_start_position()

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        self.step_count = 0
        self.discovered_objects = set()
        self.discovered_instances = set()
        self.visited_cells = set()
        self.traj_pixels.clear()
        self.prev_score = 0.0
        self.prev_coverage = 0.0
        self.prev_agent_position = None
        self.cumulative_reward = 0.0
        self.last_action_name = None
        scene_index = self._resolve_scene_index(scene_number)
        target_scene_id = self.scene_ids[scene_index]
        self.current_scene_index = scene_index
        self.scene_number = scene_number if scene_number is not None else scene_index + 1

        if target_scene_id != self.scene_id:
            self._load_scene(target_scene_id)
        
        if self.save_debug_path:
            self.current_ep_dir = os.path.join(self.save_debug_path, f"ep_{getattr(self, 'episode_id', 0):04d}_scene_{self.scene_number}")
            if not os.path.exists(self.current_ep_dir):
                os.makedirs(self.current_ep_dir, exist_ok=True)
        else:
            self.current_ep_dir = None
            
        self.sim.reset()

        if self.sim.pathfinder.is_loaded and self.total_navigable_cells is None:
            self.total_navigable_cells = self._estimate_navigable_cells()
        
        # Spawn policy:
        # 1) explicit start_position/start_rotation if provided
        # 2) random_start=True -> random navmesh start and random yaw
        # 3) otherwise keep simulator default reset pose
        agent = self.sim.get_agent(0)
        agent_state = agent.get_state()
        chosen_start = None

        if start_position is not None:
            chosen_start = self._snap_to_largest_island(start_position)
        elif random_start and self.sim.pathfinder.is_loaded:
            # First try jumping to one of the POIs
            chosen_start = self._sample_poi_start_position(self.scene_id)
            # Ensure the point is perfectly mapped to the largest navmesh
            chosen_start = self._snap_to_largest_island(chosen_start)

        if chosen_start is not None:
            agent_state.position = chosen_start
            self.last_start_position_by_scene[self.scene_id] = tuple(float(v) for v in chosen_start)

        if start_rotation is not None:
            if isinstance(start_rotation, (list, tuple, np.ndarray)) and len(start_rotation) == 4:
                agent_state.rotation = mn.Quaternion(
                    mn.Vector3(float(start_rotation[0]), float(start_rotation[1]), float(start_rotation[2])),
                    float(start_rotation[3]),
                )
            else:
                angle = float(start_rotation)
                agent_state.rotation = quat_from_angle_axis(angle, np.array([0.0, 1.0, 0.0]))
        elif random_start:
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
        self.last_action_name = action_name
        
        if action_name:
            obs = self.sim.step(action_name)
        else:
            obs = self.sim.get_sensor_observations()
            
        return self._process_obs(obs)

    def _process_obs(self, obs, is_reset=False):
        rgb = obs["color_sensor"]
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        semantic_obs = obs.get("semantic_sensor")
            
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
                VIZ_SCALE = 4
                orig_h, orig_w = rgb.shape[:2]
                vis_img = Image.fromarray(rgb.astype('uint8'), 'RGB').convert("RGBA")
                vis_img = vis_img.resize((orig_w * VIZ_SCALE, orig_h * VIZ_SCALE), Image.Resampling.BICUBIC)
                draw = ImageDraw.Draw(vis_img)
                
                try:
                    font = ImageFont.load_default()
                except Exception:
                    font = None

                semantic_boxes = self._extract_visible_semantic_boxes(semantic_obs)
                matched_semantic_indices = set()

                for gt_box in semantic_boxes:
                    sbox = [c * VIZ_SCALE for c in gt_box["bbox"]]
                    draw.rectangle(sbox, outline=(255, 0, 0, 255), width=max(1, VIZ_SCALE))
                    if font:
                        draw.text((sbox[0] + 2, max(0, sbox[1] - 10 * VIZ_SCALE)), f"{gt_box['label']}#{gt_box['semantic_id']}", fill=(255, 0, 0, 255), font=font)
                
                for det in detections:
                    box = det.get("bbox", det.get("box"))
                    label = det.get("label", "unknown")
                    score = det.get("score", 0)
                    if box is None or score < self.det_score_thr:
                        continue

                    matched_index = self._match_detection_to_semantic_box(det, semantic_boxes, matched_semantic_indices)
                    sbox = [c * VIZ_SCALE for c in box]
                    color = (0, 255, 0, 255) if matched_index is not None else (255, 190, 0, 255)
                    draw.rectangle(sbox, outline=color, width=max(1, VIZ_SCALE))
                    if font:
                        draw.text((sbox[0]+2, max(0, sbox[1]-10*VIZ_SCALE)), f"{label} {score:.0%}", fill=color, font=font)

                if semantic_boxes:
                    for gt_box in semantic_boxes:
                        gt_box["display_label"] = f"{gt_box['label']}#{gt_box['semantic_id']}"

                    actual_counts = Counter(gt_box["display_label"] for gt_box in semantic_boxes)
                    matched_counts = Counter(
                        semantic_boxes[idx]["display_label"] for idx in matched_semantic_indices
                    )
                    panel_lines = ["Visible instances:"]
                    for label, count in actual_counts.most_common(8):
                        panel_lines.append(f"{label} x{count}")
                    if len(actual_counts) > 8:
                        panel_lines.append("...")
                    panel_lines.append("Matched instances:")
                    if matched_counts:
                        for label, count in matched_counts.most_common(8):
                            panel_lines.append(f"{label} x{count}")
                    else:
                        panel_lines.append("none")

                    widths = []
                    heights = []
                    for line in panel_lines:
                        if font:
                            bbox = draw.textbbox((0, 0), line, font=font)
                            widths.append(bbox[2] - bbox[0])
                            heights.append(bbox[3] - bbox[1])
                        else:
                            widths.append(len(line) * 6)
                            heights.append(12)

                    line_height = max(max(heights, default=12) + 2, 12 * VIZ_SCALE)
                    panel_w = min(vis_img.size[0] - 10, max(widths, default=0) + 16)
                    panel_h = min(vis_img.size[1] - 10, line_height * len(panel_lines) + 12)
                    draw.rectangle((6, 6, 6 + panel_w, 6 + panel_h), fill=(0, 0, 0, 160))
                    y = 10
                    for line in panel_lines:
                        draw.text((12, y), line, fill=(255, 255, 255, 255), font=font)
                        y += line_height
                
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
        coverage_delta = coverage - self.prev_coverage

        moved_distance = 0.0
        if self.prev_agent_position is not None:
            px, pz = self.prev_agent_position
            moved_distance = float(np.hypot(ax - px, az - pz))
        self.prev_agent_position = (ax, az)
        
        # Give reward for discovery, coverage expansion and penalty for wasted motion.
        discovered_instance_count = len(self.discovered_instances)
        discovered_label_count = len(self.discovered_objects)
        norm_count = discovered_instance_count if discovered_instance_count > 0 else discovered_label_count
        current_score = min(norm_count / self.score_norm_target, 1.0)

        if is_reset:
            reward = 0.0
        else:
            score_gain = current_score - self.prev_score
            reward = (
                self.discovery_bonus_scale * score_gain
                + self.coverage_bonus_scale * coverage_delta
                - self.rho
            )

            # If the agent tried to go forward but barely moved, treat it as a collision/blocked step.
            if getattr(self, "last_action_name", None) == "move_forward" and moved_distance < 0.03:
                reward -= self.collision_penalty

        self.prev_score = current_score
        self.prev_coverage = coverage

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
                "coverage_delta": float(coverage_delta),
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
        # In get_topdown_view(), the image returned maps:
        # Real-world X axis -> Image width (axis 1 / cols)
        # Real-world Z axis -> Image height (axis 0 / rows)
        col = int(np.clip((float(x) - min_x) / (max_x - min_x) * w, 0, w - 1))
        row = int(np.clip((float(z) - min_z) / (max_z - min_z) * h, 0, h - 1))
        
        # NOTE: returning (col, row) because PIL ImageDraw functions expect (X, Y) coordinates
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

    def _build_semantic_id_label_map(self):
        mapping = {}
        try:
            semantic_scene = self.sim.semantic_annotations()
            for obj in getattr(semantic_scene, "objects", []):
                semantic_id = self._parse_semantic_object_id(getattr(obj, "id", None))
                if semantic_id is None:
                    continue
                category = getattr(obj, "category", None)
                label = None
                if category is not None:
                    name_attr = getattr(category, "name", None)
                    if callable(name_attr):
                        label = name_attr()
                    elif name_attr is not None:
                        label = str(name_attr)
                    else:
                        label = str(category)
                if not label:
                    label = f"semantic_{semantic_id}"
                mapping[int(semantic_id)] = str(label)
        except Exception:
            return {}
        return mapping

    def _parse_semantic_object_id(self, semantic_id):
        if semantic_id is None:
            return None
        if isinstance(semantic_id, (int, np.integer)):
            return int(semantic_id)
        semantic_text = str(semantic_id)
        match = re.search(r"(\d+)$", semantic_text)
        if match:
            return int(match.group(1))
        return None

    def _normalize_label_key(self, label):
        return "".join(ch for ch in str(label).lower() if ch.isalnum())

    def _extract_visible_semantic_boxes(self, semantic_obs):
        if semantic_obs is None:
            return []

        semantic_array = np.asarray(semantic_obs)
        if semantic_array.ndim == 3:
            semantic_array = semantic_array[:, :, 0]

        boxes = []
        for semantic_id in np.unique(semantic_array):
            semantic_id = int(semantic_id)
            if semantic_id <= 0:
                continue

            mask = semantic_array == semantic_id
            pixel_count = int(mask.sum())
            if pixel_count < 160:
                continue

            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue

            label = self.semantic_id_to_label.get(semantic_id, f"semantic_{semantic_id}")
            boxes.append(
                {
                    "semantic_id": semantic_id,
                    "label": label,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "pixel_count": pixel_count,
                }
            )

        boxes.sort(key=lambda item: item["pixel_count"], reverse=True)
        return boxes[:12]

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max((ax2 - ax1), 0.0) * max((ay2 - ay1), 0.0)
        area_b = max((bx2 - bx1), 0.0) * max((by2 - by1), 0.0)
        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return float(inter_area / denom)

    def _match_detection_to_semantic_box(self, det, semantic_boxes, matched_semantic_indices):
        det_box = det.get("bbox", det.get("box"))
        if det_box is None:
            return None

        best_index = None
        best_iou = 0.0

        for index, gt_box in enumerate(semantic_boxes):
            if index in matched_semantic_indices:
                continue

            iou = self._bbox_iou(det_box, gt_box["bbox"])
            if self._normalize_label_key(gt_box["label"]) == self._normalize_label_key(det.get("label", "unknown")):
                iou += 0.05
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= 0.10:
            matched_semantic_indices.add(best_index)
            return best_index
        return None

    def get_actions(self):
        # Custom simplified actions
        return ["RotateLeft", "RotateRight", "MoveAhead"]

    def close(self):
        if hasattr(self, "sim") and self.sim is not None:
            self.sim.close()
