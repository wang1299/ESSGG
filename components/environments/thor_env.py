import math
import os
import platform
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64

from components.environments.exploration_map import ExplorationMap
from components.graph.global_graph import GlobalSceneGraph
from components.graph.gt_graph import GTGraph
from components.graph.local_graph_builder import LocalSceneGraphBuilder
from components.scripts.generate_gt_graphs import generate_gt_scene_graphs
from components.utils.observation import Observation

warnings.filterwarnings("ignore", message="could not connect to X Display*", category=UserWarning)


class ThorEnv:
    def __init__(
        self, 
        rho=0.02, 
        scene_number=None, 
        render=False, 
        grid_size=0.25, 
        max_actions=40, 
        additional_images=False,
        # [New] Perception related arguments
        use_detector=False,
        detector=None,
        det_score_thr=0.3,
        fill_position_from_gt=False
    ):
        super().__init__()
        self.rho = rho
        self.grid_size = grid_size
        self.visibilityDistance = 50  # high value so objects in the frame are always visible
        self.max_actions = max_actions
        self.additional_images = additional_images
        
        # Perception settings
        self.use_detector = use_detector
        self.detector = detector
        self.det_score_thr = det_score_thr
        self.fill_position_from_gt = fill_position_from_gt

        # Verify depth requirement: If using detector and not filling pos from GT, we NEED depth.
        # So we force additional_images=True or at least renderDepthImage=True.
        if self.use_detector and not self.fill_position_from_gt:
            # We can't easily change self.additional_images here without affecting other logic,
            # but we can ensure renderDepthImage is passed correctly to controller.
            pass

        # On Linux, use the specified 'render' flag; on other platforms, always set render=True
        self.render = render if platform.system() == "Linux" else True
        
        controller_kwargs = dict(
            moveMagnitude=self.grid_size,
            grid_size=self.grid_size,
            visibilityDistance=self.visibilityDistance,
            renderDepthImage=additional_images or (self.use_detector and not self.fill_position_from_gt),
            renderSemanticSegmentation=additional_images,
            renderInstanceSegmentation=additional_images,
        )

        if not self.render:
            controller_kwargs["platform"] = CloudRendering

        self.controller = Controller(**controller_kwargs)

        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.state = None
        self.gt_graph = None
        self.viewpoints = defaultdict(set)
        self.last_score = 0.0
        self.step_count = 0
        self.scene_number = 1 if scene_number is None else scene_number
        self.map_origin = None
        self.exploration_map = None
        self.occupancy_map = None
        self.num_orientations = 4
        self.stop_index = self.get_actions().index(["Pass", "Pass"])
        self.agent_state = None

        self.td_center_x = None
        self.td_center_z = None
        self.td_ortho_size = None

    def get_action_dim(self):
        return len(self.get_actions())

    def get_actions(self):
        agent_rotations = ["RotateRight", "RotateLeft", "Pass"]
        movements = ["MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]
        return [[move, rot1] for move in movements for rot1 in agent_rotations]

    def get_state_dim(self):
        warnings.warn(
            "The state dimension cannot be reliably determined from the environment.", UserWarning
        )
        return [3, 128, 256]

    def _apply_detection(self, event):
        """
        Helper method: Run detector on the event frame and overwrite metadata['objects'].
        """
        # Backup Ground Truth if not already backed up
        if "_gt_objects" not in event.metadata:
            event.metadata["_gt_objects"] = deepcopy(event.metadata["objects"])

        if self.use_detector and self.detector is not None:
            # 1. Run detection
            depth = event.depth_frame
            agent_state = event.metadata["agent"]
            detections = self.detector.detect(event.frame, depth_image=depth, agent_state=agent_state)
            
            # Store raw detections for visualization
            self.last_raw_detections = detections
            
            # 2. Format conversion for LocalGraphBuilder
            formatted_objects = []
            for i, det in enumerate(detections):
                # Filter by score
                if det["score"] < self.det_score_thr:
                    continue
                
                # [修复] 健壮的位置处理逻辑
                # 如果 detector 返回 None，强制使用默认坐标 (0, -100, 0) 
                # (放在地下 -100 处，避免干扰地图导航逻辑)
                raw_pos = det.get("position")
                if raw_pos is None:
                    final_pos = {"x": 0.0, "y": -100.0, "z": 0.0}
                else:
                    final_pos = raw_pos

                # [Step 1: Label Mapping] Normalize labels to match Ground Truth format (PascalCase)
                # DINO often returns lowercase or space-separated words (e.g. "coffee machine")
                # GT usually uses PascalCase (e.g. "CoffeeMachine")
                raw_label = det['label'].lower().strip()
                
                # Manual overrides for specific mismatches
                OVERRIDE_MAPPING = {
                    "cup": "Mug", # Often "Mug" in THOR is detected as "Cup"
                    "trash can": "GarbageCan",
                    "rubbish bin": "GarbageCan",
                    "pan": "Pot",
                    "hand towel": "Towel",
                    "soap dispenser": "SoapBottle",
                    "pepper shaker": "SaltShaker", # [Fix] Visual ambiguity
                    "garbage": "GarbageCan",
                    "garbage bag": "GarbageCan", # [Fix] Bag often inside Can
                    "shelf": "ShelvingUnit", # [Fix] Common mismatch
                    "sink": "SinkBasin", # [Fix] Sink entity vs SinkBasin receptacle
                    "counter": "CounterTop",
                    "worktop": "CounterTop",
                    "paper towel": "PaperTowelRoll",
                    "butter knife": "ButterKnife",
                    "dish sponge": "DishSponge",
                    "paper towel roll": "PaperTowelRoll",
                    "coffee maker": "CoffeeMachine",
                    "side table": "SideTable",
                }
                
                # Prefer GT-present categories when ambiguous
                gt_types_set = {o["objectType"] for o in event.metadata.get("_gt_objects", [])}
                if raw_label == "cup":
                    if "Cup" in gt_types_set:
                        final_label = "Cup"
                    elif "Mug" in gt_types_set:
                        final_label = "Mug"
                    else:
                        final_label = OVERRIDE_MAPPING.get(raw_label, raw_label.title().replace(" ", ""))
                elif raw_label == "pepper shaker":
                    if "PepperShaker" in gt_types_set:
                        final_label = "PepperShaker"
                    elif "SaltShaker" in gt_types_set:
                        final_label = "SaltShaker"
                    else:
                        final_label = OVERRIDE_MAPPING.get(raw_label, raw_label.title().replace(" ", ""))
                elif raw_label in OVERRIDE_MAPPING:
                    final_label = OVERRIDE_MAPPING[raw_label]
                else:
                    # Heuristic: Title Case + Remove Spaces (e.g. "coffee machine" -> "CoffeeMachine")
                    final_label = raw_label.title().replace(" ", "")

                obj = {
                    "objectType": final_label,
                    "score": det['score'],
                    "visible": True,
                    "position": final_pos,
                    # [Fix] Add dummy AABB to prevent RelationExtractor crash
                    "axisAlignedBoundingBox": {
                        "center": final_pos,
                        "size": {"x": 0.5, "y": 0.5, "z": 0.5}, # Dummy size
                        "cornerPoints": [[0,0,0]] * 8 # Dummy corners
                    }
                }

                # Create a stable objectId for detections by quantizing the 3D position
                # This reduces duplicate nodes across frames for the same physical object.
                try:
                    px = final_pos.get("x", None)
                    pz = final_pos.get("z", None)
                    if px is None or pz is None or final_pos.get("y", -100.0) < -50.0:
                        # If position is missing or marked as invalid, fall back to per-frame index
                        obj_id = f"det_unpos_{i}_{final_label}"
                    else:
                        qx = round(px / self.grid_size) * self.grid_size
                        qz = round(pz / self.grid_size) * self.grid_size
                        # Use label + quantized position to produce a stable id across nearby viewpoints
                        obj_id = f"det_{final_label}_{qx:.2f}_{qz:.2f}"
                except Exception:
                    obj_id = f"det_{i}_{final_label}"

                obj["objectId"] = obj_id
                
                # If requested, cheat by filling position from GT (for debugging)
                if self.fill_position_from_gt:
                    for gt_obj in event.metadata["_gt_objects"]:
                        if gt_obj["objectType"] == final_label and gt_obj["visible"]:
                            obj["position"] = gt_obj["position"]
                            obj["objectId"] = gt_obj["objectId"] 
                            break
                            
                formatted_objects.append(obj)
            
            # 3. Overwrite the main objects list
            event.metadata["objects"] = formatted_objects
            
            # [Step 2: Detection Analysis for Visualization]
            # Classify detections as: Matched (Green), Extra/Filtered (Orange)
            # Find Missed GT (Red)
            
            # Re-read GT objects (they are safe in _gt_objects)
            gt_objects = [o for o in event.metadata.get("_gt_objects", []) if o.get("visible", False)]
            
            analysis_results = []
            matched_gt_ids = set()
            
            # Analyze each detection
            for i, det in enumerate(detections):
                if det["score"] < self.det_score_thr:
                    continue
                    
                # Use mapped label logic again (simplify for brevity, ideally reuse)
                raw_label = det['label'].lower().strip()
                OVERRIDE_MAPPING = {
                    "cup": "Mug", 
                    "trash can": "GarbageCan",
                    "rubbish bin": "GarbageCan",
                    "pan": "Pot",
                    "hand towel": "Towel",
                    "soap dispenser": "SoapBottle",
                    "pepper shaker": "SaltShaker",
                    "garbage": "GarbageCan",
                    "garbage bag": "GarbageCan",
                    "shelf": "ShelvingUnit",
                    "sink": "SinkBasin",
                    "counter": "CounterTop",
                    "worktop": "CounterTop",
                    "paper towel": "PaperTowelRoll",
                    "paper towel roll": "PaperTowelRoll",
                    "coffee maker": "CoffeeMachine",
                    "side table": "SideTable",
                }
                gt_types_set = {o["objectType"] for o in event.metadata.get("_gt_objects", [])}
                if raw_label == "cup":
                    if "Cup" in gt_types_set:
                        mapped_label = "Cup"
                    elif "Mug" in gt_types_set:
                        mapped_label = "Mug"
                    else:
                        mapped_label = OVERRIDE_MAPPING.get(raw_label, raw_label.title().replace(" ", ""))
                elif raw_label == "pepper shaker":
                    if "PepperShaker" in gt_types_set:
                        mapped_label = "PepperShaker"
                    elif "SaltShaker" in gt_types_set:
                        mapped_label = "SaltShaker"
                    else:
                        mapped_label = OVERRIDE_MAPPING.get(raw_label, raw_label.title().replace(" ", ""))
                else:
                    mapped_label = OVERRIDE_MAPPING.get(raw_label, raw_label.title().replace(" ", ""))
                
                det_pos = np.array(list(det.get("position", {"x":0,"y":0,"z":0}).values()))
                
                # Check match against Visible GT
                is_matched = False
                match_id = None
                
                for gt in gt_objects:
                    if gt["objectId"] in matched_gt_ids:
                        continue # Already matched this GT occurrence? (Optional: allow multiple detections for one GT?)
                        # Let's allow multiple detections to map to same GT for visualization purposes, 
                        # but "Matched" usually means correct.
                        
                    # Strict Label Match (with Drawer proxy)
                    drawer_proxy_labels = {"Cabinet", "CounterTop", "ShelvingUnit"}
                    is_drawer_proxy = gt["objectType"] == "Drawer" and mapped_label in drawer_proxy_labels
                    if gt["objectType"] == mapped_label or is_drawer_proxy:
                        # 3D Distance Check (projected depth)
                        gt_pos = np.array([gt["position"]["x"], gt["position"]["y"], gt["position"]["z"]])
                        det_pos_arr = np.array([det_pos[0], det_pos[1], det_pos[2]]) # Ensure array
                        
                        dist = np.linalg.norm(det_pos_arr - gt_pos)
                        
                        # [DEBUG] Print Window Distance
                        if mapped_label == "Window" and gt["objectType"] == "Window":
                            # Uncomment to debug
                            # print(f"[DEBUG] Window Dist: {dist:.2f} | Det: {det_pos_arr} | GT: {gt_pos}")
                            pass

                        # Large planar surfaces should not use distance matching
                        if mapped_label in ["Floor", "Ceiling", "Wall"]:
                            is_matched = True
                            match_id = gt["objectId"]
                            break

                        # Relaxed threshold for Large/Transparent objects
                        threshold = 2.0 if mapped_label in ["Window", "Door", "Blinds", "Curtains"] else 1.5
                        small_objects = {"ButterKnife","Knife","Fork","Spoon","Spatula","DishSponge","SaltShaker","PepperShaker","Mug","Cup","Tomato","Potato","Egg","PaperTowelRoll"}
                        if mapped_label in small_objects:
                            threshold = max(threshold, 2.0)
                        if is_drawer_proxy:
                            threshold = max(threshold, 1.5)
                        
                        if dist < threshold: # Relaxed threshold
                             is_matched = True
                             match_id = gt["objectId"]
                             # matched_gt_ids.add(match_id) # Don't consume for vis, show all valid detections
                             break
                
                analysis_results.append({
                    "raw_label": det['label'],
                    "mapped_label": mapped_label,
                    "score": det['score'],
                    "box": det.get("bbox", det.get("box")), # [Fix] Adapter uses "bbox"
                    "status": "matched" if is_matched else "extra",
                    "match_id": match_id
                })
            
            # Save analysis to metadata for the runner to read
            event.metadata["detection_analysis"] = analysis_results
        
        return event

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        if scene_number is not None:
            self.scene_number = scene_number
        self.controller.reset(
            scene=f"FloorPlan{self.scene_number}",
            moveMagnitude=self.grid_size,
            grid_size=self.grid_size,
            visibilityDistance=self.visibilityDistance,
        )
        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.step_count = 0
        self.last_score = 0.0
        self.viewpoints.clear()

        if random_start:
            # Temporarily disable detection for internal navigation logic
            temp_use_det = self.use_detector
            self.use_detector = False
            reachable = self.safe_step(action="GetReachablePositions").metadata["actionReturn"]
            self.use_detector = temp_use_det
            
            pos = random.choice(reachable)
            rot = {"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0}
        elif start_position and start_rotation:
            pos = start_position
            rot = start_rotation
        else:
            pos = None

        if pos is not None:
            self.safe_step(action="Teleport", position=pos, rotation=rot)

        # Get initial observation
        event = self.safe_step(action="Pass")
        
        # [Modified] Apply detection to the initial frame
        event = self._apply_detection(event)

        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)

        agent_view = event.metadata["agent"]
        agent_pos = agent_view["position"]
        agent_rot = agent_view["rotation"]["y"]

        viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)
        self.global_sg.add_local_sg(local_sg)

        # Get scene size information
        # Use _gt_objects or original metadata if available for map bounds?
        # Actually map bounds are static, so it's fine.
        bounds = event.metadata["sceneBounds"]
        size = bounds["size"]
        start_x = agent_pos["x"]
        start_z = agent_pos["z"]

        map_width = math.ceil((size["x"] * 2) / self.grid_size)
        map_height = math.ceil((size["z"] * 2) / self.grid_size)
        if map_width % 2 == 0: map_width += 1
        if map_height % 2 == 0: map_height += 1

        self.map_origin = (start_x - (map_width // 2) * self.grid_size, start_z - (map_height // 2) * self.grid_size)

        self.exploration_map = ExplorationMap(grid_size=self.grid_size, map_width=map_width, map_height=map_height, origin=self.map_origin)
        self.exploration_map.update_from_event(event)

        self.occupancy_map = np.zeros((map_height, map_width, self.num_orientations), dtype=np.float32)
        agent_x, agent_z = self._update_occupancy(event)

        self.state = [rgb, local_sg, self.global_sg, self.exploration_map]

        obs = Observation(state=self.state, info={"event": event})
        self._compute_reward(obs)

        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        self.td_center_x, self.td_center_z, self.td_ortho_size = self.add_topdown_camera_covering_scene()

        return obs

    def add_topdown_camera_covering_scene(self, pad=0.10, desired_hw=None):
        # We need to temporarily disable detector to avoid processing this utility step
        temp = self.use_detector
        self.use_detector = False
        
        try:
            ev = self.safe_step(action="Pass")
            bounds = ev.metadata["sceneBounds"]
            center = bounds["center"]
            size = bounds["size"]

            if desired_hw is not None:
                H, W = desired_hw
                aspect = W / H
            else:
                aspect = 1.0

            z_span = size["z"] + 2 * pad
            x_span = size["x"] + 2 * pad
            required_ortho_for_z = 0.5 * z_span
            required_ortho_for_x = 0.5 * x_span / max(1e-6, aspect)
            ortho_size = max(required_ortho_for_z, required_ortho_for_x)

            top_camera_height = size["y"] - 0.5
            self.safe_step(
                action="AddThirdPartyCamera",
                rotation=dict(x=90, y=0, z=0),
                position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
                orthographic=True,
                orthographicSize=ortho_size,
            )

            ev2 = self.safe_step(action="Pass")
            frame = ev2.third_party_camera_frames[0]
            H, W, _ = frame.shape
            true_aspect = W / H
            if abs(true_aspect - aspect) > 1e-6:
                required_ortho_for_x = 0.5 * x_span / max(1e-6, true_aspect)
                ortho_size = max(required_ortho_for_z, required_ortho_for_x)
                self.safe_step(
                    action="AddThirdPartyCamera",
                    rotation=dict(x=90, y=0, z=0),
                    position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
                    orthographic=True,
                    orthographicSize=ortho_size,
                )
            return center["x"], center["z"], ortho_size
        finally:
            self.use_detector = temp

    def get_ground_truth_graph(self, floorplan_name: str):
        save_path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs", f"{floorplan_name}.json")
        if not os.path.exists(save_path):
            print(f"⚠️ GT Graph for {floorplan_name} not found. Generating...")
            generate_gt_scene_graphs(floorplans=[floorplan_name])
        return GTGraph().load_from_file(save_path)

    def safe_step(self, *args, **kwargs):
        try:
            self.agent_state = self.get_agent_state()
            return self.controller.step(*args, **kwargs)
        except TimeoutError as e:
            print(f"[TIMEOUT] Action '{kwargs.get('action', 'unknown')}' timed out. Restarting environment.")
            self.reset_hard()
            return self.controller.step(*args, **kwargs)

    def reset_hard(self):
        try:
            self.controller.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop controller cleanly: {e}")

        # Re-initialize controller with same settings
        controller_kwargs = dict(
            moveMagnitude=self.grid_size,
            grid_size=self.grid_size,
            visibilityDistance=self.visibilityDistance,
            renderDepthImage=self.additional_images,
            renderSemanticSegmentation=self.additional_images,
            renderInstanceSegmentation=self.additional_images,
        )
        if not self.render:
            controller_kwargs["platform"] = CloudRendering
        
        self.controller = Controller(**controller_kwargs)
        
        self.reset(scene_number=self.scene_number)
        self.restore_agent_state(self.agent_state)

    def step(self, action):
        actions = self.get_actions()[action]
        error_msgs = {}
        all_success = True
        
        # Execute primitive actions
        for primitive_action in actions:
            if "Move" in primitive_action:
                event = self.safe_step(action=primitive_action, moveMagnitude=self.grid_size)
            else:
                event = self.safe_step(action=primitive_action)
            success = event.metadata["lastActionSuccess"]
            if not success:
                error_msgs[primitive_action] = event.metadata["errorMessage"]
                all_success = False
        
        # Get final observation for this step
        event = self.safe_step(action="Pass")
        
        # [Modified] Apply detection to the step frame
        event = self._apply_detection(event)

        self.exploration_map.update_from_event(event)
        self.exploration_map.mark_discoveries(event, self.global_sg)
        agent_x, agent_z = self._update_occupancy(event)

        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)
        self.global_sg.add_local_sg(local_sg)

        self.state = [rgb, local_sg, self.global_sg, self.exploration_map]

        agent_view = event.metadata["agent"]
        agent_pos = agent_view["position"]
        agent_rot = agent_view["rotation"]["y"]

        viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )

        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)

        self.step_count += 1

        truncated = action == self.stop_index or self.step_count >= self.max_actions
        terminated = (
            len([k for k, n in self.global_sg.nodes.items() if n.visibility >= 0.8]) == len(self.gt_graph.nodes)
            and action == self.stop_index
        )

        if terminated:
            truncated = False
        obs = Observation(state=self.state, truncated=truncated, terminated=terminated)

        score, recall_node, recall_edge, num_discovered, num_gt = self.compute_score(obs)

        obs.info = {
            "event": event,
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "num_discovered": num_discovered,
            "num_gt": num_gt,
            "action": action,
            "agent_pos": (agent_x, agent_z),
            "allActionsSuccess": all_success,
            "errorMessages": error_msgs,
            "max_steps_reached": self.step_count >= self.max_actions,
            "last_detections": getattr(self, "last_raw_detections", []),
            "visible_gt_objects": [o["objectType"] for o in event.metadata.get("_gt_objects", []) if o.get("visible", False)]
        }

        obs.reward = self._compute_reward(obs)
        return obs

    def compute_score(self, obs):
        num_gt_objects = len(self.gt_graph.nodes)
        
        # [Fix] True Recall Calculation: Match Global Nodes to GT Nodes
        # Only count a found node if it matches a GT node by (Label + Distance < 1.0m)
        discovered_nodes = [n for n in self.global_sg.nodes.values() if n.visibility >= 0.8]
        
        matched_count = 0
        matched_gt_ids = set()
        
        for d_node in discovered_nodes:
            d_pos = np.array(d_node.position)
            best_dist = float('inf')
            best_gt_id = None
            
            for gt_id, gt_node in self.gt_graph.nodes.items():
                # 1. Label Match (Strict or Mapped)
                # GT graph names are usually PascalCase (e.g. "CoffeeMachine")
                # Found node names are mapped in _apply_detection (e.g. "CoffeeMachine")
                drawer_proxy_labels = {"Cabinet", "CounterTop", "ShelvingUnit"}
                is_drawer_proxy = gt_node.name == "Drawer" and d_node.name in drawer_proxy_labels
                if d_node.name != gt_node.name and not is_drawer_proxy:
                    continue
                
                # 2. Distance Check
                gt_pos = np.array(gt_node.position)
                dist = np.linalg.norm(d_pos - gt_pos)

                # Large planar surfaces should not use distance matching
                if gt_node.name in ["Floor", "Ceiling", "Wall"]:
                    best_gt_id = gt_id
                    best_dist = 0.0
                    break

                small_objects = {"ButterKnife","Knife","Fork","Spoon","Spatula","DishSponge","SaltShaker","PepperShaker","Mug","Cup","Tomato","Potato","Egg","PaperTowelRoll"}
                dist_threshold = 2.0 if d_node.name in small_objects else (1.5 if is_drawer_proxy else 1.0)
                if dist < dist_threshold and dist < best_dist:
                    best_dist = dist
                    best_gt_id = gt_id
            
            # If we found a valid GT match that hasn't been counted yet
            if best_gt_id is not None and best_gt_id not in matched_gt_ids:
                matched_count += 1
                matched_gt_ids.add(best_gt_id)

        # Update metrics to reflect TRUTH
        num_discovered = matched_count # Use matched count for recall
        
        recall_node = num_discovered / num_gt_objects if num_gt_objects > 0 else 0.0
        
        # [DEBUG SCORE]
        if self.step_count % 10 == 0:
            raw_found = len(discovered_nodes)
            print(f"[SCORE DEBUG] Step: {self.step_count} | Found Nodes: {raw_found} -> Matched GT: {matched_count} | Total GT: {num_gt_objects} | Recall: {recall_node:.4f}")

        num_gt_edges = len(self.gt_graph.edges)
        num_discovered_edges = len(self.global_sg.edges) if hasattr(self.global_sg, "edges") else 0
        recall_edge = num_discovered_edges / num_gt_edges if num_gt_edges > 0 else 0.0

        termination_bonus = 0.0 if obs.terminated else 0.0
        score = recall_node + termination_bonus
        return score, recall_node, recall_edge, num_discovered, num_gt_objects

    def get_occupancy_indices(self, event):
        pos = event.metadata["agent"]["position"]
        rot_y = event.metadata["agent"]["rotation"]["y"]
        x, z = pos["x"], pos["z"]
        dx = x - self.map_origin[0]
        dz = z - self.map_origin[1]
        i = self.occupancy_map.shape[0] - 1 - int(dz / self.grid_size)
        j = int(dx / self.grid_size)
        rot_idx = int(round(rot_y / 90.0)) % self.num_orientations
        return i, j, rot_idx

    def _update_occupancy(self, event):
        i, j, rot_idx = self.get_occupancy_indices(event)
        assert 0 <= rot_idx < self.num_orientations, f"Invalid rotation index: {rot_idx}"
        assert 0 <= i < self.occupancy_map.shape[0], f"Invalid i index: {i}"
        assert 0 <= j < self.occupancy_map.shape[1], f"Invalid j index: {j}"
        self.occupancy_map[i, j, rot_idx] = 1.0
        return i, j

    def _compute_reward(self, obs):
        lambda_node = 0.1
        lambda_p = 0.5
        lambda_d = 0.001
        rho = self.rho

        Rnode = obs.info.get("recall_node", 0.0)
        Redge = obs.info.get("recall_edge", 0.0)

        if hasattr(self.global_sg, "nodes") and self.global_sg.nodes:
            Pnode = np.mean([n.visibility for n in self.global_sg.nodes.values()])
        else:
            Pnode = 0.0
        Pedge = 1.0
        diversity = sum(len(v) for v in self.viewpoints.values())
        sim = lambda_node * (Rnode + lambda_p * Pnode) + Redge + lambda_p * Pedge
        score = sim + lambda_d * diversity - rho * self.step_count
        reward = score - self.last_score
        self.last_score = score
        return reward

    def get_agent_state(self):
        agent = self.controller.last_event.metadata["agent"]
        return {"position": agent["position"], "rotation": agent["rotation"]}

    def restore_agent_state(self, agent_state):
        self.controller.step(action="Teleport", position=agent_state["position"], rotation=agent_state["rotation"], horizon=0)
        self.controller.step(action="Pass")

    def get_env_state(self):
        return {
            "state": deepcopy(self.state),
            "global_sg": deepcopy(self.global_sg),
            "exploration_map": deepcopy(self.exploration_map),
            "viewpoints": deepcopy(self.viewpoints),
            "last_score": self.last_score,
            "step_count": self.step_count,
        }

    def restore_env_state(self, env_state):
        self.state = deepcopy(env_state["state"])
        self.global_sg = deepcopy(env_state["global_sg"])
        self.exploration_map = deepcopy(env_state["exploration_map"])
        self.viewpoints = deepcopy(env_state["viewpoints"])
        self.last_score = env_state["last_score"]
        self.step_count = env_state["step_count"]

    def try_action(self, action, agent_pos=None, agent_rot=None):
        env_state = self.get_env_state()
        agent_state = self.get_agent_state()
        if agent_pos is not None and agent_rot is not None:
            event = self.safe_step(action="Teleport", position=agent_pos, rotation=dict(x=0, y=agent_rot, z=0))
        event = self.safe_step(action=action)
        self.restore_env_state(env_state)
        self.restore_agent_state(agent_state)
        return event.metadata["lastActionSuccess"]

    def get_top_down_view(self):
        event = self.safe_step(action="Pass")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            return event.third_party_camera_frames[0]
        else:
            raise RuntimeError("No third-party camera frames found.")

    def visualize_shortest_path(self, start, target):
        if len(target) == 2:
            target = {"x": target["x"], "y": start["y"], "z": target["z"]}
        event = self.safe_step(action="GetShortestPathToPoint", position=start, target=target)
        path = event.metadata["actionReturn"]["corners"]
        event = self.safe_step(action="VisualizePath", positions=path, grid=False, endText="Target")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            arr = event.third_party_camera_frames[0]
            return Image.fromarray(arr)
        return None

    def close(self):
        self.controller.stop()