import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import torch

sys.path.append("/home/wgy/ESSGG")
sys.path.append("/home/wgy/ESSGG/GroundingDINO")

# === [新增] 确保能找到 GroundingDINO ===
# 如果你已经 pip install -e . 了，这行其实不需要，但加上为了保险
sys.path.append(os.path.join(os.path.dirname(__file__), "../../GroundingDINO"))

def main(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Read Configs ---
    agent_config = config["agent_config"]
    navigation_config = config["navigation_config"]
    env_config = config["env_config"]

    # === [新增逻辑] 初始化 Grounding DINO ===
    dino_detector = None
    if args.use_dino: 
        print("[INFO] Initializing Grounding DINO Detector for Evaluation...")
        # ！！！请确保这里的路径和你 main.py 里的一致！！！
        dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        dino_weights = "weights/groundingdino_swint_ogc.pth" 
        
        # 延迟导入，防止没装库时报错
        from components.detectors.grounding_dino_adapter import GroundingDINODetector
        
        if not os.path.exists(dino_weights):
            print(f"[WARNING] DINO weights not found at {dino_weights}. Running EVAL without detector (GT mode).")
        else:
            dino_detector = GroundingDINODetector(
                config_path=dino_config,
                checkpoint_path=dino_weights,
                text_prompt="chair . table . bed . sofa . tv . plant", # 建议后续根据 config/navigation.json 动态生成
                box_threshold=0.20,  # [Modified] Lower threshold
                text_threshold=0.20  # [Modified] Lower threshold
            )

    # Setup environment
    # [关键修改] 支持动态切换 Precomputed 或 实时 ThorEnv
    if args.precomputed:
        print(f"[INFO] Using PrecomputedThorEnv...")
        env = PrecomputedThorEnv(
            rho=env_config["rho"], 
            max_actions=agent_config["num_steps"],
            detector=dino_detector 
        )
    else:
        print(f"[INFO] Using live ThorEnv (Real-time Rendering)...")
        env = ThorEnv(
            render=env_config.get("render", False), 
            rho=env_config["rho"], 
            max_actions=agent_config["num_steps"],
            # 传递 detector 相关的参数
            use_detector=(dino_detector is not None),
            detector=dino_detector
        )

    # Load agent (初始化结构)
    if agent_config["name"] == "reinforce":
        agent = ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    elif agent_config["name"] == "a2c":
        agent = A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    else:
        raise Exception("Unknown agent")

    # === [修改逻辑] 处理模型路径 ===
    # 不再使用硬编码路径，而是使用命令行传入的 path
    target_path = Path(args.model_path)
    
    # 如果传入的是文件夹，就遍历里面的 .pth；如果是文件，就只加载这一个
    if target_path.is_dir():
        model_files = list(target_path.glob("*.pth"))
    else:
        model_files = [target_path]

    if not model_files:
        print(f"[ERROR] No model files found at {target_path}")
        return

    for file in model_files:
        print(f"\n[INFO] === Evaluating Model: {file.name} ===")
        # 加载权重
        # 注意：这里加上 weights_only=False 以防 PyTorch 版本兼容性问题
        try:
            agent.load_weights(model_path=file, device=device)
        except Exception as e:
            # 如果你的 agent.load_weights 里已经处理了 weights_only，这里可能不需要
            # 但为了保险，如果上面报错，可以手动用 torch.load 加载
            print(f"[WARNING] Standard load failed, trying raw load: {e}")
            state_dict = torch.load(file, map_location=device, weights_only=False)
            agent.policy.load_state_dict(state_dict) 

        # 设置评估场景 (原作者逻辑：使用训练集之后的 3 个场景)
        agent.scene_numbers = agent.all_scene_numbers[agent.num_scenes : agent.num_scenes + 3]
        
        # 运行评估
        runner = RLEvalRunner(env=env, agent=agent, device=device)
        runner.run()
        
    env.close()


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        return

if __name__ == "__main__":
    set_working_directory()

    from components.agents.a2c_agent import A2CAgent
    from components.agents.reinforce_agent import ReinforceAgent
    from components.environments.precomputed_thor_env import PrecomputedThorEnv
    from components.environments.thor_env import ThorEnv
    from RL_training.runner.rl_eval_runner import RLEvalRunner
    from components.utils.utility_functions import read_config, set_seeds

    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str, required=True, help="Path to the configuration files.")
    # [新增] 必须指定要跑哪个模型文件
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth model file or directory.")
    # [新增] DINO 开关
    parser.add_argument("--use_dino", action="store_true", help="Use Grounding DINO for perception.")
    # [新增] 预计算模式开关
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed environment (faster but needs data).")
    
    args = parser.parse_args()

    # Iterate over each configuration file in args.conf_path
    conf_files = Path(args.conf_path).rglob("*.json")
    conf_list = list(conf_files)
    
    if len(conf_list) == 0:
        print(f"No config files found in {args.conf_path}")
    else:
        # 这里默认只读第一个配置，通常够用了
        conf = read_config(conf_list[0])
        set_seeds(conf["seed"])
        main(conf, args)