import datetime
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import torch
# === [关键修改] 先把项目根目录加进去，再 import ===
sys.path.append("/home/wgy/ESSGG") 
sys.path.append("/home/wgy/ESSGG/GroundingDINO")
# === [新增 import] ===
from components.detectors.grounding_dino_adapter import GroundingDINODetector

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Read Configs ---
    agent_config = config["agent_config"]
    navigation_config = config["navigation_config"]
    env_config = config["env_config"]

    encoder_path = Path(agent_config["encoder_path"])
    encoder_path = encoder_path.parent / (encoder_path.stem + "_" + str(navigation_config["use_transformer"]) + encoder_path.suffix)

    # === [新增逻辑] 初始化 Grounding DINO ===
    # 建议将路径写在 config 文件里，这里为了演示直接写死或者用 argparse
    dino_detector = None
    if args.use_dino: # 假设你在 argparse 里加了这个参数，或者直接强制开启
        print("[INFO] Initializing Grounding DINO Detector...")
        # ！！！请修改下面的路径为你的真实路径！！！
        dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        dino_weights = "weights/groundingdino_swint_ogc.pth" # 你的 .pth 文件路径
        
        if not os.path.exists(dino_weights):
            print(f"[WARNING] DINO weights not found at {dino_weights}. Running without detector.")
        else:
            dino_detector = GroundingDINODetector(
                config_path=dino_config,
                checkpoint_path=dino_weights,
                text_prompt="chair . table . bed . sofa . tv . plant",
                box_threshold=0.20,
                text_threshold=0.20
            )

    # Setup environment
    if not args.precomputed:
        print("[INFO] Using live ThorEnv (Real-time Rendering) for TRAINING...")
        env = ThorEnv(
            render=env_config["render"], 
            rho=env_config["rho"], 
            max_actions=agent_config["num_steps"],
            use_detector=(dino_detector is not None),
            detector=dino_detector
        )
    else:
        env = PrecomputedThorEnv(
            rho=env_config["rho"], 
            max_actions=agent_config["num_steps"],
            detector=dino_detector # <--- [关键] 传入检测器
        )

    # Load agent from encoder & policy weights
    if agent_config["name"] == "reinforce":
        agent = ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    elif agent_config["name"] == "a2c":
        agent = A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    else:
        raise Exception("Unknown agent")

    agent.load_weights(encoder_path=encoder_path, device=device)

    # RL training runner
    runner = RLTrainRunner(env=env, agent=agent, device=device)
    runner.run()
    env.close()
    print("[INFO] Training completed.")

    # --- Save final model ---
    if args.save_model:
        save_folder = Path("RL_training") / "runs" / "model_weights"
        save_folder.mkdir(exist_ok=True)
        run_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Python 3.7+ f-string fix
        file_name = f"{run_start}_{agent_config['name']}_agent.pth"
        agent.save_model(str(save_folder), file_name=file_name)


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working director:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()

    from components.agents.a2c_agent import A2CAgent
    from components.agents.reinforce_agent import ReinforceAgent
    from components.environments.thor_env import ThorEnv
    from components.environments.precomputed_thor_env import PrecomputedThorEnv
    from RL_training.runner.rl_train_runner import RLTrainRunner
    from components.utils.utility_functions import read_config, set_seeds

    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str, help="Path to the configuration files.")
    parser.add_argument("--save_model", action="store_true", help="Save model weights.")
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed environment.")
    # [新增] 命令行参数控制是否开启 DINO
    parser.add_argument("--use_dino", action="store_true", help="Use Grounding DINO for perception.")
    
    args = parser.parse_args()

    # Iterate over each configuration file in args.conf_path
    conf_files = Path(args.conf_path).rglob("*.json")
    for conf_file in conf_files:
        print(f"[INFO] Running with configuration: {conf_file}")
        conf = read_config(conf_file)
        set_seeds(conf["seed"])
        main(conf)