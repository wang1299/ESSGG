import habitat_sim
import numpy as np
import os
import random
from components.utils.observation import Observation
from habitat_sim.utils.common import quat_from_angle_axis, axis_y

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
        fill_position_from_gt=False
    ):
        self.width = width
        self.height = height
        self.dataset_root = dataset_root
        self.use_detector = use_detector
        self.detector = detector
        
        # Change working directory so habitat can find assets relative to config
        self.initial_cwd = os.getcwd()
        if os.path.exists(dataset_root):
            os.chdir(dataset_root)
        else:
            print(f"Warning: Dataset root {dataset_root} not found. Continuing in {os.getcwd()}")

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = config_file
        sim_cfg.scene_id = scene_id
        sim_cfg.enable_physics = True
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
            None,             # 4: MoveRight (Not supported)
            None,             # 5: MoveLeft (Not supported)
            None,             # 6: MoveBack (Not supported)
            None              # 7: Pass
        ]

    def reset(self):
        self.sim.reset()
        
        # Random spawn on navmesh
        if self.sim.pathfinder.is_loaded:
             random_position = self.sim.pathfinder.get_random_navigable_point()
             agent = self.sim.get_agent(0)
             agent_state = agent.get_state()
             agent_state.position = random_position
             
             # Random rotation
             angle = random.uniform(0, 2 * np.pi)
             agent_state.rotation = quat_from_angle_axis(angle, axis_y)
             agent.set_state(agent_state)
        
        obs = self.sim.get_sensor_observations()
        return self._process_obs(obs)

    def step(self, action_id):
        # Handle tensor inputs
        if hasattr(action_id, 'item'):
            action_id = action_id.item()
            
        action_name = self.action_mapping[action_id] if 0 <= action_id < len(self.action_mapping) else None
        
        if action_name:
            obs = self.sim.step(action_name)
        else:
            obs = self.sim.get_sensor_observations()
            
        return self._process_obs(obs)

    def _process_obs(self, obs):
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
                 # Pass to detector
                 # Note: Not storing result as Observation structure doesn't support metadata explicitly
                 # But we run it as requested.
                 detections = self.detector.detect(rgb, depth_image=depth, agent_state=as_dict)
             except Exception as e:
                 print(f"Warning: Detector failed: {e}")

        # Return format expected by agent: [rgb]
        return Observation([rgb], 0.0, False, False, {})

    def get_actions(self):
        # Keep same as ThorEnv for compatibility
        return ["RotateRight", "RotateLeft", "Pass", "MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]

    def close(self):
        self.sim.close()
