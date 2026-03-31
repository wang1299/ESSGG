import habitat_sim

HM3D_ROOT = "/home/wgy/hm3d/scene_datasets/hm3d"
SCENE = f"{HM3D_ROOT}/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
DATASET_CFG = f"{HM3D_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = SCENE
backend_cfg.scene_dataset_config_file = DATASET_CFG

# 添加语义传感器
sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
sem_cfg.resolution = [256, 256]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sem_cfg]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

obs = sim.get_sensor_observations()
print("Loaded HM3D successfully.")
print("Semantic observation keys:", obs.keys())
print("Semantic shape:", obs["semantic"].shape)

sim.close()