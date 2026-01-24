import math
import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pickle

from components.environments.exploration_map import ExplorationMap
from components.graph.global_graph import GlobalSceneGraph
from components.graph.gt_graph import GTGraph
from components.graph.local_graph_builder import LocalSceneGraphBuilder
from components.scripts.generate_gt_graphs import generate_gt_scene_graphs
from components.utils.observation import Observation


class PrecomputedThorEnv:
    def __init__(
        self,
        rho=0.02,
        scene_number=None,
        render=False,
        grid_size=0.25,
        transition_tables_path="components/data/transition_tables",
        max_actions=40,
        detector=None, # <--- [新增] 接受外部传入的检测器
    ):
        self.rho = rho
        self.scene_number = 1 if scene_number is None else scene_number
        self.max_actions = max_actions
        self.render = render
        self.grid_size = grid_size
        if grid_size != 0.25:
            raise ValueError("PrecomputedThorEnv only supports grid_size of 0.25")

        self.transition_tables_path = transition_tables_path
        self.detector = detector # <--- [新增] 保存检测器
        
        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.state = None
        self.gt_graph = None
        self.viewpoints = defaultdict(set)
        self.last_score = 0.0
        self.step_count = 0
        self.map_origin = None
        self.exploration_map = None
        self.occupancy_map = None
        self.num_orientations = 4
        self.stop_index = self.get_actions().index(["Pass", "Pass"])
        self.last_event = None

        # Load precomputed mapping: dict {(x,z,rotation): event or None}
        with open(f"{self.transition_tables_path}/FloorPlan{self.scene_number}.pkl", "rb") as f:
            data = pickle.load(f)
        self.mapping = data["table"]

        self.current_pos = None 
        self.current_rot = None 

    def get_action_dim(self):
        return len(self.get_actions())

    def get_actions(self):
        agent_rotations = ["RotateRight", "RotateLeft", "Pass"]
        movements = ["MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]
        return [[move, rot1] for move in movements for rot1 in agent_rotations]

    def get_state_dim(self):
        warnings.warn("The state dimension cannot be reliably determined...", UserWarning)
        return [3, 128, 256]

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        if scene_number is not None:
            self.scene_number = scene_number
            with open(os.path.join(self.transition_tables_path, f"FloorPlan{self.scene_number}.pkl"), "rb") as f:
                data = pickle.load(f)
            self.mapping = data["table"]

        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.step_count = 0
        self.last_score = 0.0
        self.viewpoints.clear()
        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        if random_start:
            valid = [k for k, evt in self.mapping.items() if evt is not None]
            x, z, rot = random.choice(valid)
        elif start_position is not None and start_rotation is not None:
            x = round(start_position["x"] / self.grid_size) * self.grid_size
            z = round(start_position["z"] / self.grid_size) * self.grid_size
            rot = start_rotation["y"] % 360
            if (x, z, rot) not in self.mapping or self.mapping[(x, z, rot)] is None:
                raise ValueError("Invalid start state")
        else:
            raise ValueError("Invalid reset arguments")

        self.current_pos = (x, z)
        self.current_rot = rot
        event = self.mapping[(x, z, rot)]
        self.last_event = event

        obs = self._build_observation(event, reset=True)
        self._compute_reward(obs)
        return obs

    def _build_observation(self, event, reset=False):
        rgb = event.frame
        
        # === [修改部分开始] ===
        # 逻辑：如果有检测器，用 DINO；没有检测器，用 metadata (GT)
        if self.detector is not None:
            # 1. 尝试获取深度图 (关键！如果pkl里没有存depth，这里会是None)
            # 有些数据集使用的是 'depth_frame' 属性
            depth = getattr(event, "depth_frame", None)
            
            # 2. 准备 Agent 状态用于坐标变换
            agent_data = event.metadata["agent"]
            agent_state = {
                "position": agent_data["position"],
                "rotation": agent_data["rotation"]
            }
            
            # 3. 运行检测器
            # 注意：如果 depth 是 None，DINO adapter 会返回 (0,0,0) 坐标
            detections = self.detector.detect(rgb, depth, agent_state)
            
            # 4. 构建图
            local_sg = self.builder.build_from_detections(detections)
        else:
            # 原有的 GT 逻辑
            local_sg = self.builder.build_from_metadata(event.metadata)
        # === [修改部分结束] ===

        self.global_sg.add_local_sg(local_sg)

        # Build exploration map and occupancy
        bounds = event.metadata["sceneBounds"]
        size = bounds["size"]
        agent_view = event.metadata["agent"]
        agent_x = agent_view["position"]["x"]
        agent_z = agent_view["position"]["z"]
        agent_rot = agent_view["rotation"]["y"]

        viewpoint = (
            round(agent_x / self.grid_size) * self.grid_size,
            round(agent_z / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)

        if reset:
            map_width = math.ceil((size["x"] * 2) / self.grid_size)
            map_height = math.ceil((size["z"] * 2) / self.grid_size)
            map_width += 1 if map_width % 2 == 0 else 0
            map_height += 1 if map_height % 2 == 0 else 0
            self.map_origin = (agent_x - (map_width // 2) * self.grid_size, agent_z - (map_height // 2) * self.grid_size)
            self.exploration_map = ExplorationMap(grid_size=self.grid_size, map_width=map_width, map_height=map_height, origin=self.map_origin)
            self.occupancy_map = np.zeros((map_height, map_width, self.num_orientations), dtype=np.float32)

        self.exploration_map.update_from_event(event)
        self._update_occupancy(event)

        self.state = [rgb, local_sg, self.global_sg, self.exploration_map]
        return Observation(state=self.state, info={"event": event})

    # ... (后续方法保持不变: get_ground_truth_graph, transition_step, step, compute_score, get_occupancy_indices, _update_occupancy, _compute_reward, get_agent_state, restore_agent_state, get_env_state, restore_env_state, try_action, get_top_down_view, visualize_shortest_path, close)
    # 为了节省篇幅，这里不重复粘贴未修改的辅助函数，请保留你原文件中类下面的其他方法。
    
    def get_ground_truth_graph(self, floorplan_name):
        # 请保留原代码...
        return GTGraph().load_from_file(os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs", f"{floorplan_name}.json"))

    def transition_step(self, action_str):
        # 请保留原代码... (需要完整代码请告诉我，但我建议保留你原有的逻辑以免引入缩进错误)
        if self.current_pos is None: raise ValueError("Call reset() before stepping.")
        x, z = self.current_pos
        rot = self.current_rot
        new_x, new_z, new_rot = x, z, rot
        success = True
        if action_str == "RotateRight": new_rot = (rot + 90) % 360
        elif action_str == "RotateLeft": new_rot = (rot - 90) % 360
        elif action_str.startswith("Move"):
            if action_str == "MoveAhead": dx, dz = 0, self.grid_size
            elif action_str == "MoveBack": dx, dz = 0, -self.grid_size
            elif action_str == "MoveRight": dx, dz = self.grid_size, 0
            elif action_str == "MoveLeft": dx, dz = -self.grid_size, 0
            else: dx, dz = 0, 0
            angle = rot % 360
            if angle == 90: dx, dz = dz, -dx
            elif angle == 180: dx, dz = -dx, -dz
            elif angle == 270: dx, dz = -dz, dx
            new_x, new_z = x + dx, z + dz
        else: pass
        
        key = (round(new_x, 2), round(new_z, 2), new_rot)
        event = self.mapping.get(key, None)
        if event is None:
            event = self.mapping.get((round(x, 2), round(z, 2), new_rot))
            new_x, new_z = x, z
            success = False
        self.current_pos = (new_x, new_z)
        self.current_rot = new_rot
        event.metadata["lastActionSuccess"] = success
        self.last_event = event
        return event

    def step(self, action):
        # 请保留原代码...
        actions = self.get_actions()[action]
        all_success = True
        for primitive_action in actions:
            event = self.transition_step(primitive_action)
            if not event.metadata.get("lastActionSuccess", True): all_success = False
        obs = self._build_observation(event)
        self.step_count += 1
        truncated = action == self.stop_index or self.step_count >= self.max_actions
        terminated = (len([k for k, n in self.global_sg.nodes.items() if n.visibility >= 0.8]) == len(self.gt_graph.nodes) and action == self.stop_index)
        if terminated: truncated = False
        obs.terminated = terminated
        obs.truncated = truncated
        score, recall_node, recall_edge = self.compute_score(obs)
        obs.info = {"event": event, "score": score, "recall_node": recall_node, "recall_edge": recall_edge, "action": action, "agent_pos": self.current_pos, "allActionsSuccess": all_success, "max_steps_reached": self.step_count >= self.max_actions}
        obs.reward = self._compute_reward(obs)
        return obs
    
    # 请确保保留 compute_score, get_occupancy_indices, _update_occupancy, _compute_reward 等所有剩余方法
    def compute_score(self, obs):
        num_gt_objects = len(self.gt_graph.nodes)
        discovered_nodes = [n for n in self.global_sg.nodes.values() if n.visibility >= 0.8]
        num_discovered = len(discovered_nodes)
        recall_node = num_discovered / num_gt_objects if num_gt_objects > 0 else 0.0
        num_gt_edges = len(self.gt_graph.edges)
        num_discovered_edges = len(self.global_sg.edges) if hasattr(self.global_sg, "edges") else 0
        recall_edge = num_discovered_edges / num_gt_edges if num_gt_edges > 0 else 0.0
        termination_bonus = 0.0 if obs.terminated else 0.0
        score = recall_node + termination_bonus
        return score, recall_node, recall_edge

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
        else: Pnode = 0.0
        Pedge = 1.0
        diversity = sum(len(v) for v in self.viewpoints.values())
        sim = lambda_node * (Rnode + lambda_p * Pnode) + Redge + lambda_p * Pedge
        score = sim + lambda_d * diversity - rho * self.step_count
        reward = score - self.last_score
        self.last_score = score
        return reward
    
    def get_agent_state(self):
        ag = self.last_event.metadata["agent"]
        return {"position": (ag["position"]["x"], ag["position"]["z"]), "rotation": ag["rotation"]["y"]}
    def restore_agent_state(self, state): self.current_pos, self.current_rot = state["position"], state["rotation"]
    def get_env_state(self): return {"state": deepcopy(self.state), "global_sg": deepcopy(self.global_sg), "exploration_map": deepcopy(self.exploration_map), "viewpoints": deepcopy(self.viewpoints), "last_score": self.last_score, "step_count": self.step_count}
    def restore_env_state(self, env_state): self.state = deepcopy(env_state["state"]); self.global_sg = deepcopy(env_state["global_sg"]); self.exploration_map = deepcopy(env_state["exploration_map"]); self.viewpoints = deepcopy(env_state["viewpoints"]); self.last_score = env_state["last_score"]; self.step_count = env_state["step_count"]
    def try_action(self, action_str, pos=None, rot=None):
        bx, bz = pos if pos is not None else self.current_pos
        brot = rot if rot is not None else self.current_rot
        nx, nz, nrot = bx, bz, brot
        if action_str == "RotateRight": nrot = (brot + 90) % 360
        elif action_str == "RotateLeft": nrot = (brot - 90) % 360
        elif action_str.startswith("Move"):
            if action_str == "MoveAhead": dx, dz = 0, self.grid_size
            elif action_str == "MoveBack": dx, dz = 0, -self.grid_size
            elif action_str == "MoveRight": dx, dz = self.grid_size, 0
            elif action_str == "MoveLeft": dx, dz = -self.grid_size, 0
            else: dx, dz = 0, 0
            a = brot % 360
            if a == 90: dx, dz = -dz, dx
            elif a == 180: dx, dz = -dx, -dz
            elif a == 270: dx, dz = dz, -dx
            nx, nz = bx + dx, bz + dz
        return self.mapping.get((round(nx, 2), round(nz, 2), nrot)) is not None
    def get_top_down_view(self): return None
    def visualize_shortest_path(self, start, target): return None
    def close(self): pass