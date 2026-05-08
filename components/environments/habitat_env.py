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
from components.perception.hm3d_labels import (
    HM3D_CANONICAL_LABELS,
    HM3D_COMPATIBLE_LABEL_GROUPS,
    HM3D_LABEL_ALIASES,
    HM3D_REWARD_EXCLUDED_LABELS,
)
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


_DINO_CANONICAL_LABELS = {
    "Cabinet",
    "CounterTop",
    "Faucet",
    "Floor",
    "HousePlant",
    "Microwave",
    "Pot",
    "Potato",
    "SinkBasin",
    "SoapBottle",
    "StoveBurner",
    "StoveKnob",
    "Window",
    "Apple",
    "Chair",
    "DiningTable",
    "Plate",
    "Bowl",
    "Knife",
    "Pan",
    "Tomato",
    "Drawer",
    "GarbageCan",
    "Fridge",
    "Bread",
    "Lettuce",
    "Sink",
    "Spatula",
    "Toaster",
    "Cup",
    "PepperShaker",
    "SaltShaker",
    "ButterKnife",
    "Spoon",
    "CoffeeMachine",
    "LightSwitch",
    "Mug",
    "DishSponge",
    "Fork",
    "Ladle",
    "WineBottle",
    "CellPhone",
    "Kettle",
    "Egg",
    "PaperTowelRoll",
    "Book",
    "CreditCard",
    "Stool",
    "Blinds",
    "AluminumFoil",
    "Mirror",
    "Shelf",
    "SideTable",
    "ShelvingUnit",
    "Statue",
    "Vase",
    "Bottle",
    "GarbageBag",
    "Pencil",
    "Curtains",
    "SprayBottle",
    "Pen",
    "Safe",
    "Wall",
}

_DINO_LABEL_ALIASES = {
    "counter": "CounterTop",
    "countertop": "CounterTop",
    "counter top": "CounterTop",
    "plant": "HousePlant",
    "house plant": "HousePlant",
    "sink basin": "SinkBasin",
    "basin": "SinkBasin",
    "sink": "Sink",
    "stove burner": "StoveBurner",
    "burner": "StoveBurner",
    "stove knob": "StoveKnob",
    "knob": "StoveKnob",
    "dining table": "DiningTable",
    "table": "DiningTable",
    "garbage can": "GarbageCan",
    "trash can": "GarbageCan",
    "refrigerator": "Fridge",
    "fridge": "Fridge",
    "pepper shaker": "PepperShaker",
    "salt shaker": "SaltShaker",
    "butter knife": "ButterKnife",
    "coffee machine": "CoffeeMachine",
    "light switch": "LightSwitch",
    "dish sponge": "DishSponge",
    "wine bottle": "WineBottle",
    "cell phone": "CellPhone",
    "phone": "CellPhone",
    "paper towel roll": "PaperTowelRoll",
    "paper towel": "PaperTowelRoll",
    "credit card": "CreditCard",
    "aluminum foil": "AluminumFoil",
    "side table": "SideTable",
    "shelving unit": "ShelvingUnit",
    "shelf": "Shelf",
    "curtain": "Curtains",
    "curtains": "Curtains",
    "spray bottle": "SprayBottle",
}

for _label in _DINO_CANONICAL_LABELS:
    _DINO_LABEL_ALIASES.setdefault(_label, _label)

_DINO_COMPATIBLE_LABEL_GROUPS = [
    {"Sink", "SinkBasin"},
    {"CounterTop", "DiningTable", "SideTable"},
    {"Shelf", "ShelvingUnit", "Cabinet"},
    {"HousePlant", "Plant"},
    {"Bottle", "SoapBottle", "SprayBottle", "WineBottle"},
    {"Cup", "Mug"},
    {"GarbageCan", "GarbageBag"},
]

_DEFAULT_REWARD_EXCLUDED_LABELS = {"Wall", "Floor", "Window"}

# The original project used AI2-THOR object categories. Habitat/HM3D scenes use
# Matterport-style categories, so override the old vocabulary for validation and
# reward accounting while leaving the rest of the environment logic untouched.
_DINO_CANONICAL_LABELS = set(HM3D_CANONICAL_LABELS)
_DINO_LABEL_ALIASES = dict(HM3D_LABEL_ALIASES)
_DINO_COMPATIBLE_LABEL_GROUPS = [set(group) for group in HM3D_COMPATIBLE_LABEL_GROUPS]
_DEFAULT_REWARD_EXCLUDED_LABELS = set(HM3D_REWARD_EXCLUDED_LABELS)


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
        gt_validation_iou_threshold=0.10,
        gt_validation_mode="relaxed",
        success_recall_threshold=0.90,
        success_reward=10.0,
        reward_excluded_labels=None,
        max_actions=40,
        save_debug_interval=100,
        save_debug_path=None,
        worker_id=None,
        gpu_device_id=0,
    ):
        self.rho = rho
        self.max_actions = max_actions
        self.step_count = 0
        self.worker_id = worker_id
        self.gpu_device_id = gpu_device_id
        self.save_debug_path = save_debug_path
        self.episode_id = 0
        self.render = render
        self.config_file = config_file
        self.base_scene_id = scene_id
        self.scene_ids = self._normalize_scene_ids(scene_id, scene_ids)
        self.scene_id = self.scene_ids[0]
        self.current_scene_index = 0
        
        if self.save_debug_path:
            if self.worker_id is not None:
                try:
                    wid = int(self.worker_id)
                except Exception:
                    wid = self.worker_id
                self.save_debug_path = os.path.join(self.save_debug_path, f"worker_{wid}")
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
        self.gt_validation_iou_threshold = max(float(gt_validation_iou_threshold), 0.0)
        self.gt_validation_mode = str(gt_validation_mode or "relaxed").lower()
        self.success_recall_threshold = float(success_recall_threshold)
        self.success_reward = float(success_reward)
        if reward_excluded_labels is None:
            reward_excluded_labels = _DEFAULT_REWARD_EXCLUDED_LABELS
        self.reward_excluded_labels = {str(label) for label in reward_excluded_labels}
        self.save_debug_interval = max(int(save_debug_interval), 1)
        self.total_navigable_cells = None
        self.topdown_base_img = None
        self.topdown_bounds = None
        self.topdown_shape = None
        self.traj_pixels = deque()
        self.traj_pixels_all = deque()
        self.semantic_id_to_label = {}
        self.scene_reward_gt_ids = set()
        self.last_action_name = None
        self.last_start_position_by_scene = {}
        self._last_rgb = None
        self._last_depth = None
        self._last_semantic = None
        self._last_agent_state = None
        self._camera_hfov_deg = 90.0
        self._last_validated_detections = []
        
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
        sim_cfg.gpu_device_id = getattr(self, "gpu_device_id", 0)

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
        self.scene_reward_gt_ids = self._build_scene_reward_gt_ids()
        self.total_navigable_cells = None
        self.topdown_base_img = None
        self.topdown_bounds = None
        self.topdown_shape = None
        self.traj_pixels = deque()
        self.traj_pixels_all = deque()

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

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None, episode_tag=None):
        current_episode_id = self.episode_id
        self.episode_id += 1

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
        self.trajectory_csv = None
        self.traj_pixels_all.clear()
        scene_index = self._resolve_scene_index(scene_number)
        target_scene_id = self.scene_ids[scene_index]
        self.current_scene_index = scene_index
        self.scene_number = scene_number if scene_number is not None else scene_index + 1

        if target_scene_id != self.scene_id:
            self._load_scene(target_scene_id)
        
        if self.save_debug_path:
            if episode_tag:
                dir_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(episode_tag)).strip("_")
            else:
                dir_name = f"worker_ep_{current_episode_id:04d}_scene_{self.scene_number}"
            self.current_ep_dir = os.path.join(self.save_debug_path, dir_name)
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
            os.makedirs(self.current_ep_dir, exist_ok=True)
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

        depth = obs.get("depth_sensor")
        semantic_obs = obs.get("semantic_sensor")

        agent_state = self.sim.get_agent(0).get_state()
        self._last_rgb = rgb
        self._last_depth = depth
        self._last_semantic = semantic_obs
        self._last_agent_state = agent_state
            
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
                 detections = self.validate_detections(detections)
                 for det in detections:
                     if det.get("score", 0) < self.det_score_thr:
                         continue
                     if det.get("is_gt_valid") is not True:
                         continue
                     label = det.get("canonical_label") or det.get("label", "unknown")
                     if label in self.reward_excluded_labels:
                         continue
                     try:
                         gt_semantic_id = int(det.get("gt_semantic_id"))
                     except Exception:
                         continue
                     if self.scene_reward_gt_ids and gt_semantic_id not in self.scene_reward_gt_ids:
                         continue
                     self.discovered_objects.add(label)
                     self.discovered_instances.add(self._build_instance_key(det, label))
             except Exception as e:
                 print(f"Warning: Detector failed: {e}")

        # Save Visualizations
        save_viz_frame = getattr(self, "current_ep_dir", None) is not None and (
            is_reset or self.step_count % self.save_debug_interval == 0
        )

        if save_viz_frame:
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
        if self.topdown_base_img is not None:
            px = self._world_to_topdown_px(ax, az)
            if px is not None:
                self.traj_pixels_all.append(px)
        if self.current_ep_dir is not None and save_viz_frame:
            self._save_topdown_step(ax, az)

        if self.current_ep_dir is not None:
            os.makedirs(self.current_ep_dir, exist_ok=True)
            if not getattr(self, "trajectory_csv", None):
                self.trajectory_csv = os.path.join(self.current_ep_dir, "trajectory.csv")
            write_header = not os.path.exists(self.trajectory_csv)
            with open(self.trajectory_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["step", "x", "y", "z", "yaw_deg", "score", "coverage", "num_instances"])
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
                "scene_reward_gt_ids": sorted(int(item) for item in getattr(self, "scene_reward_gt_ids", set())),
                "scene_reward_gt_count": int(len(getattr(self, "scene_reward_gt_ids", set()))),
            },
        )

    def finalize_episode(self, reason="done"):
        """Persist final per-episode artifacts before the runner resets this worker."""
        if self.current_ep_dir is None:
            return {"saved": False, "reason": reason, "path": None}
        os.makedirs(self.current_ep_dir, exist_ok=True)
        self._save_topdown_trajectory()
        return {
            "saved": os.path.exists(os.path.join(self.current_ep_dir, "topdown_trajectory.png")),
            "reason": reason,
            "path": self.current_ep_dir,
            "steps": int(self.step_count),
        }

    def annotate_detections(self, detections):
        if not detections:
            return []
        if self._last_depth is None or self._last_agent_state is None:
            return self.validate_detections(detections)

        depth = self._last_depth
        if isinstance(depth, np.ndarray) and depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]

        H = int(self.height)
        W = int(self.width)
        hfov = float(self._camera_hfov_deg)
        fx = (W / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        fy = (H / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        cx = W / 2.0
        cy = H / 2.0

        agent_pos = np.array(
            [
                float(self._last_agent_state.position[0]),
                float(self._last_agent_state.position[1]),
                float(self._last_agent_state.position[2]),
            ],
            dtype=np.float32,
        )
        q = self._last_agent_state.rotation
        quat = np.array([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float32)

        annotated = []
        for det in detections:
            box = det.get("bbox", det.get("box"))
            if box is None or len(box) != 4:
                annotated.append(det)
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            u_int = int(np.clip(int(u), 0, W - 1))
            v_int = int(np.clip(int(v), 0, H - 1))

            d = float(depth[v_int, u_int]) if isinstance(depth, np.ndarray) else None
            if d is None or d <= 0.01 or d > 10.0:
                det2 = dict(det)
                det2["position"] = det.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
                annotated.append(det2)
                continue

            z_c = d
            x_c = (u - cx) * d / fx
            y_c = -(v - cy) * d / fy
            cam_point = np.array([x_c, y_c, z_c], dtype=np.float32)
            world_rel = self._quat_rotate_vec(quat, cam_point)
            world_point = world_rel + agent_pos

            det2 = dict(det)
            det2["position"] = {"x": float(world_point[0]), "y": float(world_point[1]), "z": float(world_point[2])}
            annotated.append(det2)

        return self.validate_detections(annotated)

    @staticmethod
    def _label_tokens(label):
        text = str(label or "").replace("_", " ").replace(".", " ")
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
        text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
        return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)]

    @classmethod
    def _canonicalize_label(cls, raw_label):
        tokens = cls._label_tokens(raw_label)
        if not tokens:
            return None, "empty_label"

        alias_tokens = {
            tuple(cls._label_tokens(alias)): canonical
            for alias, canonical in _DINO_LABEL_ALIASES.items()
        }
        full_key = tuple(tokens)
        if full_key in alias_tokens:
            return alias_tokens[full_key], None

        matches = []
        covered = [False] * len(tokens)
        for alias_key, canonical in alias_tokens.items():
            if not alias_key or len(alias_key) > len(tokens):
                continue
            for start in range(0, len(tokens) - len(alias_key) + 1):
                if tuple(tokens[start:start + len(alias_key)]) == alias_key:
                    matches.append((canonical, start, start + len(alias_key)))
                    for idx in range(start, start + len(alias_key)):
                        covered[idx] = True

        canonical_matches = sorted({m[0] for m in matches})
        if len(canonical_matches) != 1:
            return None, "ambiguous_label" if canonical_matches else "unknown_label"
        if not all(covered):
            return None, "compound_label"
        return canonical_matches[0], None

    @staticmethod
    def _labels_compatible(det_label, gt_label):
        if not det_label or not gt_label:
            return False
        if det_label == gt_label:
            return True
        for group in _DINO_COMPATIBLE_LABEL_GROUPS:
            if det_label in group and gt_label in group:
                return True
        return False

    def _extract_all_visible_semantic_boxes(self, semantic_obs, min_pixels=80):
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
            if pixel_count < min_pixels:
                continue

            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue

            raw_label = self.semantic_id_to_label.get(semantic_id, f"semantic_{semantic_id}")
            canonical_label, reason = self._canonicalize_label(raw_label)
            boxes.append(
                {
                    "semantic_id": semantic_id,
                    "label": raw_label,
                    "canonical_label": canonical_label,
                    "canonical_error": reason,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "pixel_count": pixel_count,
                }
            )

        boxes.sort(key=lambda item: item["pixel_count"], reverse=True)
        return boxes

    def validate_detections(self, detections):
        """Validate DINO detections against the current Habitat semantic frame."""
        if not detections:
            self._last_validated_detections = []
            return []

        semantic_boxes = self._extract_all_visible_semantic_boxes(self._last_semantic)
        validated = []
        used_semantic_ids = set()

        for det in detections:
            det2 = dict(det)
            raw_label = det2.get("label", det2.get("class", "unknown"))
            canonical_label, label_error = self._canonicalize_label(raw_label)
            det2["canonical_label"] = canonical_label
            det2["gt_label"] = None
            det2["gt_canonical_label"] = None
            det2["gt_semantic_id"] = None
            det2["gt_iou"] = 0.0
            det2["is_gt_valid"] = False

            if label_error is not None:
                det2["reject_reason"] = label_error
                validated.append(det2)
                continue

            det_box = det2.get("bbox", det2.get("box"))
            if det_box is None or len(det_box) != 4:
                det2["reject_reason"] = "missing_bbox"
                validated.append(det2)
                continue

            best_box = None
            best_iou = 0.0
            best_any_box = None
            best_any_iou = 0.0
            for gt_box in semantic_boxes:
                if gt_box["semantic_id"] in used_semantic_ids:
                    continue
                iou = self._bbox_iou(det_box, gt_box["bbox"])
                if best_any_box is None or iou > best_any_iou:
                    best_any_iou = iou
                    best_any_box = gt_box

                gt_canonical = gt_box.get("canonical_label")
                if self.gt_validation_mode != "off" and not self._labels_compatible(canonical_label, gt_canonical):
                    continue

                if best_box is None or iou > best_iou:
                    best_iou = iou
                    best_box = gt_box

            if best_box is None:
                # HM3D semantic category names are not always aligned with the
                # DINO/AI2-THOR prompt vocabulary. In relaxed mode, keep the
                # whitelist/compound-label filter strict, then allow an IoU-only
                # semantic confirmation when the best GT label cannot itself be
                # canonicalized into our object vocabulary.
                if (
                    self.gt_validation_mode == "relaxed"
                    and best_any_box is not None
                    and best_any_iou >= self.gt_validation_iou_threshold
                    and best_any_box.get("canonical_label") is None
                ):
                    best_box = best_any_box
                    best_iou = best_any_iou
                    det2["gt_match_mode"] = "semantic_iou_only"
                else:
                    det2["reject_reason"] = "no_matching_gt_label"
                    if best_any_box is not None:
                        det2["gt_iou"] = float(best_any_iou)
                        det2["gt_label"] = best_any_box["label"]
                        det2["gt_canonical_label"] = best_any_box.get("canonical_label")
                        det2["gt_semantic_id"] = int(best_any_box["semantic_id"])
                    validated.append(det2)
                    continue
            else:
                det2["gt_match_mode"] = "label_iou"

            det2["gt_iou"] = float(best_iou)
            det2["gt_label"] = best_box["label"]
            det2["gt_canonical_label"] = best_box.get("canonical_label")
            det2["gt_semantic_id"] = int(best_box["semantic_id"])

            if best_iou < self.gt_validation_iou_threshold:
                det2["reject_reason"] = "low_iou"
                validated.append(det2)
                continue

            det2["is_gt_valid"] = True
            det2["reject_reason"] = None
            used_semantic_ids.add(best_box["semantic_id"])
            validated.append(det2)

        self._last_validated_detections = validated
        self._save_dino_validation_overlay(validated, semantic_boxes)
        return validated

    def _save_dino_validation_overlay(self, detections, semantic_boxes):
        if getattr(self, "current_ep_dir", None) is None or self._last_rgb is None:
            return
        if self.step_count % self.save_debug_interval != 0 and self.step_count != 0:
            return

        try:
            from PIL import Image, ImageDraw, ImageFont

            scale = 4
            rgb = self._last_rgb
            orig_h, orig_w = rgb.shape[:2]
            img = Image.fromarray(rgb.astype("uint8"), "RGB").convert("RGBA")
            img = img.resize((orig_w * scale, orig_h * scale), Image.Resampling.BICUBIC)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            semantic_by_id = {box["semantic_id"]: box for box in semantic_boxes}
            accepted_semantic_ids = {
                int(det.get("gt_semantic_id"))
                for det in detections
                if det.get("is_gt_valid", False) is True and det.get("gt_semantic_id") is not None
            }

            # Draw missed visible GT instances first, then overlay DINO boxes.
            # Color convention: green=accepted detection, orange=rejected detection,
            # red=visible semantic instance not matched by any accepted detection.
            for gt_box in semantic_boxes:
                semantic_id = int(gt_box["semantic_id"])
                if semantic_id in accepted_semantic_ids:
                    continue
                sbox = [float(c) * scale for c in gt_box["bbox"]]
                draw.rectangle(sbox, outline=(255, 45, 45, 230), width=max(1, scale))
                label = gt_box.get("canonical_label") or gt_box.get("label", "unknown")
                if font:
                    draw.text(
                        (sbox[0] + 2, max(0, sbox[1] - 10 * scale)),
                        f"MISS {label}",
                        fill=(255, 45, 45, 255),
                        font=font,
                    )

            for det in detections:
                box = det.get("bbox", det.get("box"))
                if box is None or len(box) != 4:
                    continue
                sbox = [float(c) * scale for c in box]
                accepted = bool(det.get("is_gt_valid", False))
                color = (0, 220, 70, 255) if accepted else (255, 170, 0, 255)
                draw.rectangle(sbox, outline=color, width=max(1, scale))

                gt_id = det.get("gt_semantic_id")
                if gt_id in semantic_by_id:
                    gt_sbox = [float(c) * scale for c in semantic_by_id[gt_id]["bbox"]]
                    draw.rectangle(gt_sbox, outline=(80, 160, 255, 220), width=max(1, scale))

                label = det.get("canonical_label") or det.get("label", "unknown")
                if accepted:
                    text = f"OK {label}->{det.get('gt_canonical_label')} IoU {det.get('gt_iou', 0):.2f}"
                else:
                    reason = det.get("reject_reason", "unknown")
                    if det.get("gt_label") is not None:
                        text = f"REJ {label}: {reason} GT {det.get('gt_label')} IoU {det.get('gt_iou', 0):.2f}"
                    else:
                        text = f"REJ {label}: {reason}"
                if font:
                    draw.text((sbox[0] + 2, max(0, sbox[1] - 10 * scale)), text, fill=color, font=font)

            save_path = os.path.join(self.current_ep_dir, f"dino_validation_{self.step_count:04d}.png")
            img.save(save_path)
        except Exception as e:
            print(f"Failed to save DINO validation overlay: {e}")

    @staticmethod
    def _quat_rotate_vec(quat_xyzw: np.ndarray, vec3: np.ndarray):
        x, y, z, w = [float(v) for v in quat_xyzw]
        vx, vy, vz = [float(v) for v in vec3]
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return np.array([rx, ry, rz], dtype=np.float32)

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
        if self.topdown_base_img is None:
            return
        try:
            from PIL import Image, ImageDraw

            img = Image.fromarray(self.topdown_base_img.copy(), mode="RGB")
            draw = ImageDraw.Draw(img)

            pts = self._load_trajectory_pixels_from_csv()
            if not pts:
                pts = list(self.traj_pixels_all)
            if not pts:
                return

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

    def _load_trajectory_pixels_from_csv(self):
        csv_path = getattr(self, "trajectory_csv", None)
        if not csv_path or not os.path.exists(csv_path):
            return []
        pts = []
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        px = self._world_to_topdown_px(float(row["x"]), float(row["z"]))
                    except Exception:
                        continue
                    if px is not None:
                        pts.append(px)
        except Exception as e:
            print(f"Failed to load trajectory CSV for topdown map: {e}")
        return pts

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

    def _build_scene_reward_gt_ids(self):
        reward_ids = set()
        for semantic_id, raw_label in self.semantic_id_to_label.items():
            canonical_label, reason = self._canonicalize_label(raw_label)
            if reason is not None:
                continue
            if canonical_label in self.reward_excluded_labels:
                continue
            reward_ids.add(int(semantic_id))
        return reward_ids

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
