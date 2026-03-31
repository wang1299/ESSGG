import sys
import os
sys.path.append('/home/wgy/RL')
from components.environments.habitat_env import HabitatEnv

HM3D_ROOT = "/home/wgy/hm3d/scene_datasets/hm3d"
SCENE = f"{HM3D_ROOT}/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
DATASET_CFG = f"{HM3D_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"

print("Init Env")
env = HabitatEnv(
    dataset_root=HM3D_ROOT,
    config_file=DATASET_CFG,
    scene_id=SCENE,
    render=False,
    use_detector=False
)

print("Env Reset")
obs = env.reset(scene_number=1, random_start=True)
print("Finished reset")
