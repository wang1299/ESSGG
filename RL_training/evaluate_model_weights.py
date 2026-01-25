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
                # Updated text prompt using exactly the 64 categories from object_types.json
                text_prompt="Cabinet . Counter Top . Faucet . Floor . House Plant . Microwave . Pot . Potato . Sink Basin . Soap Bottle . Stove Burner . Stove Knob . Window . Apple . Chair . Dining Table . Plate . Bowl . Knife . Pan . Tomato . Drawer . Garbage Can . Fridge . Bread . Lettuce . Sink . Spatula . Toaster . Cup . Pepper Shaker . Salt Shaker . Butter Knife . Spoon . Coffee Machine . Light Switch . Mug . Dish Sponge . Fork . Ladle . Wine Bottle . Cell Phone . Kettle . Egg . Paper Towel Roll . Book . Credit Card . Stool . Blinds . Aluminum Foil . Mirror . Shelf . Side Table . Shelving Unit . Statue . Vase . Bottle . Garbage Bag . Pencil . Curtains . Spray Bottle . Pen . Safe . Wall .",
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
        runner = RLEvalRunner(
            env=env, 
            agent=agent, 
            device=device,
            save_dir=args.save_frames_to,  # [Modified] Pass save directory
            total_episodes=args.num_episodes # [Modified]
        )
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
    # [新增] 可视化保存
    parser.add_argument("--save_frames_to", type=str, default=None, help="Directory to save visualization frames. If not set, no frames are saved.")
    # [新增] 评估集数
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
    
    args = parser.parse_args()

    # Load all configs
    full_config = {}
    config_dir = Path(args.conf_path)
    
    agent_conf_path = config_dir / "agent.json"
    env_conf_path = config_dir / "env.json"
    nav_conf_path = config_dir / "navigation.json"
    
    if agent_conf_path.exists():
        full_config["agent_config"] = read_config(str(agent_conf_path))
    if env_conf_path.exists():
        full_config["env_config"] = read_config(str(env_conf_path))
    if nav_conf_path.exists():
        full_config["navigation_config"] = read_config(str(nav_conf_path))
        
    # Validation
    if "agent_config" not in full_config or "env_config" not in full_config or "navigation_config" not in full_config:
        print("[ERROR] Missing one of detection/agent/env/navigation config files in: ", args.conf_path)
        # Fallback to current dangerous behavior or just exit
        # Try to find seed from what we have
        pass

    seed = full_config.get("env_config", {}).get("seed", 42)
    set_seeds(seed)

    # [Smart Config Override]
    # If the model path implies a specific architecture, override the config.
    if "LSTM" in str(args.model_path) and full_config["navigation_config"].get("use_transformer", False):
        print("[INFO] Model path contains 'LSTM' but config says Transformer. Switching to LSTM.")
        full_config["navigation_config"]["use_transformer"] = False
    elif "Transformer" in str(args.model_path) and not full_config["navigation_config"].get("use_transformer", False):
        print("[INFO] Model path contains 'Transformer' but config says LSTM. Switching to Transformer.")
        full_config["navigation_config"]["use_transformer"] = True
    
    main(full_config, args)