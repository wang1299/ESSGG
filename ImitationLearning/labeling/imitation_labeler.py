import ast
import copy
import math
import random
import re

from collections import namedtuple, deque

from components.environments.thor_env import ThorEnv


class ImitationLabeler:
    def __init__(self, env):
        self.env = env

    def compute_score(self, env, visibility_before: dict, event_before_action, viewpoints, alpha=0.8):
        event = env.controller.last_event
        visibility_after = {k: n.visibility for k, n in self.env.global_sg.nodes.items()}

        score = 0.0
        for obj_id, vis_after in visibility_after.items():
            vis_before = visibility_before.get(obj_id, 0.0)

            updated_vis = 1 - (1 - vis_before) * (1 - alpha * vis_after)

            delta_vis = updated_vis - vis_before

            score += delta_vis

            if vis_before < 0.8 <= updated_vis:
                score += 1

        # bonus for exploring a new view
        i, j, rot_idx = self.env.get_occupancy_indices(event)
        occupancy_bonus = 0.9 if self.env.occupancy_map[i][j][rot_idx] == 0 else 0.0

        vp_key = next(iter(viewpoints))
        vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)
        dx = abs(event_before_action.metadata["agent"]["position"]["x"] - vp_pos["x"])
        dz = abs(event_before_action.metadata["agent"]["position"]["z"] - vp_pos["z"])
        steps_to_vp = int(round(dx / self.env.grid_size)) + int(round(dz / self.env.grid_size))

        curr_dx = abs(event.metadata["agent"]["position"]["x"] - vp_pos["x"])
        curr_dz = abs(event.metadata["agent"]["position"]["z"] - vp_pos["z"])
        curr_steps_to_vp = int(round(curr_dx / self.env.grid_size)) + int(round(curr_dz / self.env.grid_size))

        last_rot = int(event_before_action.metadata["agent"]["rotation"]["y"])
        steps_to_rot = abs(((last_rot - vp_rot) // 90)) % 4  # modulo 4 for 0-360°
        curr_rot = int(event.metadata["agent"]["rotation"]["y"])
        curr_steps_to_rot = abs(((curr_rot - vp_rot) // 90)) % 4

        rot_bonus = 0
        if steps_to_vp <= steps_to_rot or curr_steps_to_vp <= curr_steps_to_rot:
            if curr_steps_to_rot < steps_to_rot:
                rot_bonus += len(env.gt_graph.nodes) + 1
            if curr_steps_to_rot > steps_to_rot:
                rot_bonus -= len(env.gt_graph.nodes) + 1

        return score + rot_bonus + occupancy_bonus

    def recover_missing_viewpoints(self, viewpoints, threshold=0.2):
        """
        If some objects are not yet sufficiently visible and all viewpoints have been explored,
        reintroduce viewpoints that help cover the missing objects.
        """
        global_seen = {k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8}
        all_nodes = set(self.env.gt_graph.nodes.keys())

        missing = all_nodes - global_seen
        if not missing:
            return  # All done

        # Search for viewpoints that see these objects
        recovered_viewpoints = {}
        v2o = self.env.gt_graph.viewpoint_to_objects

        for vp_key, obj_list in v2o.items():
            recovered = []
            for obj in obj_list:
                for obj_id, vis in obj.items():
                    if obj_id in missing and vis >= threshold:
                        recovered.append(obj_id)
            if recovered:
                recovered_viewpoints[vp_key] = recovered

        if recovered_viewpoints:
            print(f"Recovered {len(recovered_viewpoints)} viewpoints for missing objects: {missing}")
            viewpoints.update(recovered_viewpoints)

    def get_next_move_action(self, agent_pos, agent_rot, viewpoints, tol: float = 0.2):
        """
        Returns the primitive move action (MoveAhead/Back/Left/Right) that brings the
        agent closer to the next unreached viewpoint.
        """
        if len(agent_rot) == 3:
            agent_rot = int(agent_rot["y"])
        # Filter out already visited viewpoints
        while viewpoints:
            vp_key = next(iter(viewpoints))
            vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)
            if abs(agent_pos["x"] - vp_pos["x"]) < tol and abs(agent_pos["z"] - vp_pos["z"]) < tol:
                if agent_rot == vp_rot:
                    del viewpoints[vp_key]
                else:
                    return "Pass"
            else:
                break

        if not viewpoints:
            self.recover_missing_viewpoints(viewpoints)
            if not viewpoints:
                raise ValueError("No viewpoints left to explore (even after recovery)")

            vp_items = sorted(
                viewpoints.items(),
                key=lambda item: abs(agent_pos["x"] - self.deserialize_viewpoint(item[0])[0]["x"])
                + abs(agent_pos["z"] - self.deserialize_viewpoint(item[0])[0]["z"]),
            )

            for vp_key, _ in vp_items:
                vp_pos_candidate, vp_rot_candidate = self.deserialize_viewpoint(vp_key)
                if abs(agent_pos["x"] - vp_pos_candidate["x"]) < tol and abs(agent_pos["z"] - vp_pos_candidate["z"]) < tol:
                    if agent_rot == vp_rot_candidate:
                        del viewpoints[vp_key]
                        continue
                    else:
                        return "Pass"
                else:
                    vp_pos, vp_rot = vp_pos_candidate, vp_rot_candidate
                    break

        if viewpoints:
            vp_key = next(iter(viewpoints))
            vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)

        path_points = self.get_shortest_path_to_point(agent_pos, vp_pos)

        # Remove already-reached path points
        while path_points and abs(path_points[0]["x"] - agent_pos["x"]) < tol and abs(path_points[0]["z"] - agent_pos["z"]) < tol:
            path_points.pop(0)

        if not path_points:
            raise ValueError("No path points left to explore")

        # Direction to the next target point in the agent's egocentric frame
        target = path_points[0]
        dx = round(target["x"] - agent_pos["x"], 2)
        dz = round(target["z"] - agent_pos["z"], 2)
        theta = math.radians(agent_rot)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        rel_forward = sin_t * dx + cos_t * dz
        rel_right = cos_t * dx - sin_t * dz

        # Prepare possible directions, sorted by absolute value (descending)
        actions = [
            ("MoveAhead" if rel_forward > 0 else "MoveBack", abs(rel_forward)),
            ("MoveRight" if rel_right > 0 else "MoveLeft", abs(rel_right)),
        ]
        actions.sort(key=lambda x: -x[1])  # Sort descending by value

        for action, val in actions:
            if val > 0 and self.env.try_action(action, agent_pos, agent_rot):
                return action

        raise ValueError("No valid move action found")

    def select_best_action(self, viewpoints, planning_steps=3, replan_each_step=True, beam_width=3, use_beam_width=False):
        """
        Plans the best sequence of primitive actions (up to planning_steps) to maximize node discovery.
        Returns either only the first action (for replan_each_step=True) or the whole sequence.
        """
        env_state = self.env.get_env_state()
        agent_state = self.env.get_agent_state()
        visibility_before = {k: n.visibility for k, n in self.env.global_sg.nodes.items()}
        node_before = [k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8]
        total_node_count = len(self.env.gt_graph.nodes)

        if len(node_before) == total_node_count:
            return [self.env.stop_index]

        ActionSeq = namedtuple("ActionSeq", ["seq", "score", "positions", "rotations", "viewpoints"])
        env = ThorEnv()
        env.reset(scene_number=self.env.scene_number)
        queue = deque()

        # Get valid actions for the initial step
        try:
            move_action = self.get_next_move_action(agent_state["position"], agent_state["rotation"], viewpoints)
        except ValueError as e:
            env.close()
            raise ValueError(e)

        valid_action_indices = self.get_valid_action_indices(move_action)

        for i in valid_action_indices:
            env.restore_env_state(env_state)
            env.restore_agent_state(agent_state)
            event_before_action = env.controller.last_event
            obs_new = env.step(i)
            score = self.compute_score(env, visibility_before, event_before_action, viewpoints)
            agent_pos = obs_new.info["event"].metadata["agent"]["position"]
            agent_rot = obs_new.info["event"].metadata["agent"]["rotation"]
            queue.append(ActionSeq([i], score, [agent_pos], [agent_rot], [copy.deepcopy(viewpoints)]))

        if use_beam_width:
            queue = deque(sorted(queue, key=lambda x: x.score, reverse=True)[:beam_width])
        else:
            queue = deque(sorted(queue, key=lambda x: x.score, reverse=True))

        # Planning loop
        for _ in range(1, planning_steps):
            candidates = []
            for action_seq in queue:
                if replan_each_step:
                    viewpoints_copy = {k: v[:] for k, v in viewpoints.items()}
                else:
                    viewpoints_copy = viewpoints

                try:
                    move_action = self.get_next_move_action(action_seq.positions[-1], action_seq.rotations[-1], viewpoints_copy)
                except ValueError as e:
                    env.close()
                    raise ValueError(e)

                valid_action_indices = self.get_valid_action_indices(move_action)
                for action in valid_action_indices:
                    env.restore_env_state(env_state)
                    env.restore_agent_state(agent_state)

                    visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}
                    step_score = 0
                    # Replay the sequence
                    for i, a in enumerate(action_seq.seq):
                        event_before = env.controller.last_event
                        env.step(a)
                        partial_score = self.compute_score(env, visibility_before, event_before, action_seq.viewpoints[i])
                        step_score += partial_score
                        visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}

                    event_before = env.controller.last_event
                    obs = env.step(action)

                    current_nodes = [k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8]
                    if len(current_nodes) == total_node_count:
                        full_seq = action_seq.seq + [action, self.env.stop_index]
                        return full_seq

                    final_score = self.compute_score(env, visibility_before, event_before, viewpoints_copy)
                    total_score = step_score + final_score
                    combined_pos = action_seq.positions + [obs.info["event"].metadata["agent"]["position"]]
                    combined_rot = action_seq.rotations + [obs.info["event"].metadata["agent"]["rotation"]]
                    combined_vp = action_seq.viewpoints + [copy.deepcopy(viewpoints_copy)]
                    candidates.append(ActionSeq(action_seq.seq + [action], total_score, combined_pos, combined_rot, combined_vp))

            if use_beam_width:
                queue = deque(sorted(candidates, key=lambda x: x.score, reverse=True)[:beam_width])
            else:
                queue = deque(sorted(candidates, key=lambda x: x.score, reverse=True))

        env.close()

        if queue:
            max_score = queue[0].score
            best_seq_indices = [i for i, seq in enumerate(queue) if seq.score == max_score]
            best_seq_idx = random.choice(best_seq_indices)
            best_seq = queue[best_seq_idx]
        else:
            best_seq = [random.choice(valid_action_indices)]

        if replan_each_step:
            return [best_seq.seq[0]]
        else:
            return best_seq.seq

    def extract_navmesh_positions_from_error(self, error_msg):
        """
        Extracts the 'closest navmesh positions' for both start and target from an error message.
        Returns two dicts (start, target) or (None, None) if parsing fails.
        """
        pattern = r"closest navmesh position \((-?\d+\.\d+), [\d\.]+, (-?\d+\.\d+)\)"
        matches = re.findall(pattern, error_msg)

        if len(matches) >= 2:
            start_pos = {"x": float(matches[0][0]), "y": 0.900999, "z": float(matches[0][1])}
            target_pos = {"x": float(matches[1][0]), "y": 0.900999, "z": float(matches[1][1])}
            return start_pos, target_pos
        return None, None

    def get_shortest_path_to_point(self, initial_position, target_position, tolerance=0.2):
        if "y" not in initial_position:
            initial_position = {**initial_position, "y": 0.900999}
        if "y" not in target_position:
            target_position = {**target_position, "y": 0.900999}

        try:
            event = self.env.controller.step(
                action="GetShortestPathToPoint", position=initial_position, target=target_position, raise_for_failure=True
            )
            return event.metadata["actionReturn"]["corners"]
        except Exception as e:
            # Try parsing fallback positions
            error_msg = str(e)
            snapped_start, snapped_target = self.extract_navmesh_positions_from_error(error_msg)
            if snapped_start is None or snapped_target is None:
                raise ValueError(f"Path failed and no usable navmesh correction found: {error_msg}")

            dx_start = abs(snapped_start["x"] - initial_position["x"])
            dz_start = abs(snapped_start["z"] - initial_position["z"])
            dx_target = abs(snapped_target["x"] - target_position["x"])
            dz_target = abs(snapped_target["z"] - target_position["z"])

            if dx_start <= tolerance and dz_start <= tolerance and dx_target <= tolerance and dz_target <= tolerance:
                retry_event = self.env.controller.step(
                    action="GetShortestPathToPoint", position=snapped_start, target=snapped_target, raise_for_failure=True
                )
                return retry_event.metadata["actionReturn"]["corners"]
            else:
                raise ValueError(
                    f"Navmesh snap too far from original positions. dx_start={dx_start}, dz_start={dz_start}, "
                    f"dx_target={dx_target}, dz_target={dz_target}. Error was: {error_msg}"
                )

    def get_valid_action_indices(self, move_action):
        actions = self.env.get_actions()
        return [i for i, a in enumerate(actions) if i != self.env.stop_index and move_action == a[0]]

    @classmethod
    def deserialize_viewpoint(cls, s: str):
        try:
            dict_part, rotation = s.split("_")
            pos_dict = ast.literal_eval(dict_part)
            return pos_dict, int(rotation)
        except Exception as e:
            raise ValueError(f"Failed to deserialize viewpoint: {s} ({e})")
