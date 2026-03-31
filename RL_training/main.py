import datetime
import os
import sys
import signal # Added for signal handling
from argparse import ArgumentParser
from pathlib import Path
import torch
# === [关键修改] 先把项目根目录加进去，再 import ===
sys.path.append("/home/wgy/RL") 
sys.path.append("/home/wgy/GroundingDINO")
# === [新增 import] ===
# from components.detectors.grounding_dino_adapter import GroundingDINODetector

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_ids_env = os.environ.get("RL_GPU_IDS", "").strip()
    parsed_gpu_ids = []
    if torch.cuda.is_available() and gpu_ids_env:
        try:
            parsed_gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip() != ""]
            if len(parsed_gpu_ids) > 0:
                device = torch.device(f"cuda:{parsed_gpu_ids[0]}")
        except Exception:
            print(f"[WARNING] Invalid RL_GPU_IDS='{gpu_ids_env}', fallback to single GPU.")
            parsed_gpu_ids = []

    # --- Read Configs ---
    agent_config = config["agent_config"]
    navigation_config = config["navigation_config"]
    env_config = config["env_config"]

    encoder_path = Path(agent_config["encoder_path"])
    encoder_path = encoder_path.parent / (encoder_path.stem + "_" + str(navigation_config["use_transformer"]) + encoder_path.suffix)

    # === [新增逻辑] 初始化 Grounding DINO ===
    # 建议将路径写在 config 文件里，这里为了演示直接写死或者用 argparse
    dino_detector = None
    if args.save_frames_to is None:
        args.save_frames_to = "/home/wgy/RL/train_png"

    # Always keep a run-level folder like train_YYYYmmdd_HHMMSS under train_png.
    # If caller already passes a train_* folder (e.g., run_train.sh), keep it as-is.
    run_dir_name = datetime.datetime.now().strftime("train_%Y%m%d_%H%M%S")
    base_name = os.path.basename(os.path.normpath(args.save_frames_to))
    if not base_name.startswith("train_"):
        args.save_frames_to = os.path.join(args.save_frames_to, run_dir_name)
    os.makedirs(args.save_frames_to, exist_ok=True)
    print(f"[INFO] Visualization output root: {args.save_frames_to}")

    if args.use_dino:
        from components.detectors.grounding_dino_adapter import GroundingDINODetector
        print("[INFO] Initializing Grounding DINO Detector...")
        dino_device = os.environ.get("DINO_DEVICE", "").strip() or None
        if dino_device is not None:
            print(f"[INFO] Grounding DINO device: {dino_device}")
        # ！！！请修改下面的路径为你的真实路径！！！
        dino_config = "/home/wgy/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        dino_weights = "/home/wgy/GroundingDINO/weights/groundingdino_swint_ogc.pth" # 你的 .pth 文件路径
        
        if not os.path.exists(dino_weights):
            print(f"[WARNING] DINO weights not found at {dino_weights}. Running without detector.")
        else:
            dino_detector = GroundingDINODetector(
                config_path=dino_config,
                checkpoint_path=dino_weights,
                device=dino_device,
                # Updated text prompt using exactly the 64 categories from object_types.json
                # Removed 'House Plant', 'Sink Basin', 'Stove Burner', 'Stove Knob' spaces to match JSON keys better or rely on mapping
                # But actually DINO needs natural language.
                # Let's clean up the prompt to be natural language but covering all 64 categories.
                # Note: "SinkBasin" in json -> "Sink Basin" in prompt is good.
                # JSON: Cabinet, CounterTop, Faucet, Floor, HousePlant, Microwave, Pot, Potato, SinkBasin, SoapBottle, StoveBurner, StoveKnob, Window, Apple, Chair, DiningTable, Plate, Bowl, Knife, Pan, Tomato, Drawer, GarbageCan, Fridge, Bread, Lettuce, Sink, Spatula, Toaster, Cup, PepperShaker, SaltShaker, ButterKnife, Spoon, CoffeeMachine, LightSwitch, Mug, DishSponge, Fork, Ladle, WineBottle, CellPhone, Kettle, Egg, PaperTowelRoll, Book, CreditCard, Stool, Blinds, AluminumFoil, Mirror, Shelf, SideTable, ShelvingUnit, Statue, Vase, Bottle, GarbageBag, Pencil, Curtains, SprayBottle, Pen, Safe, Wall
                
                text_prompt="Cabinet . Counter Top . Faucet . Floor . House Plant . Microwave . Pot . Potato . Sink Basin . Soap Bottle . Stove Burner . Stove Knob . Window . Apple . Chair . Dining Table . Plate . Bowl . Knife . Pan . Tomato . Drawer . Garbage Can . Fridge . Bread . Lettuce . Sink . Spatula . Toaster . Cup . Pepper Shaker . Salt Shaker . Butter Knife . Spoon . Coffee Machine . Light Switch . Mug . Dish Sponge . Fork . Ladle . Wine Bottle . Cell Phone . Kettle . Egg . Paper Towel Roll . Book . Credit Card . Stool . Blinds . Aluminum Foil . Mirror . Shelf . Side Table . Shelving Unit . Statue . Vase . Bottle . Garbage Bag . Pencil . Curtains . Spray Bottle . Pen . Safe . Wall .",
                box_threshold=0.20,
                text_threshold=0.20
            )

    # When DINO is enabled, reserve GPU0 for detector by default.
    if args.use_dino and len(parsed_gpu_ids) > 1 and 0 in parsed_gpu_ids and os.environ.get("ALLOW_GPU0_FOR_POLICY", "0") != "1":
        parsed_gpu_ids = [gid for gid in parsed_gpu_ids if gid != 0]
        if len(parsed_gpu_ids) > 0:
            device = torch.device(f"cuda:{parsed_gpu_ids[0]}")
            print(f"[INFO] Reserved GPU0 for DINO. Policy GPUs: {parsed_gpu_ids}")
        else:
            parsed_gpu_ids = []
            print("[WARNING] RL_GPU_IDS only contained GPU0 while DINO is enabled; fallback to single-device execution.")

    # Setup environment
    if args.use_habitat:
        print("[INFO] Using HabitatEnv for TRAINING...")
        from components.environments.habitat_env import HabitatEnv
        HM3D_ROOT = "/home/wgy/hm3d/scene_datasets/hm3d"
        SCENE = args.habitat_scene if args.habitat_scene else f"{HM3D_ROOT}/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
        DATASET_CFG = f"{HM3D_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"
        
        env = HabitatEnv(
            dataset_root=HM3D_ROOT,
            config_file=DATASET_CFG,
            scene_id=SCENE,
            render=env_config["render"],
            use_detector=(dino_detector is not None),
            detector=dino_detector,
            det_score_thr=env_config.get("det_score_thr", 0.20 if dino_detector is not None else 0.30),
            score_norm_target=env_config.get("score_norm_target", 120.0),
            instance_merge_dist=env_config.get("instance_merge_dist", 0.8),
            coverage_cell_size=env_config.get("coverage_cell_size", 0.5),
            nav_sample_points=env_config.get("nav_sample_points", 4000),
            topdown_meters_per_pixel=env_config.get("topdown_meters_per_pixel", 0.05),
            rho=env_config.get("rho", 0.1),
            max_actions=agent_config["num_steps"],
            save_debug_path=args.save_frames_to
        )
    elif not args.precomputed:
        print("[INFO] Using live ThorEnv (Real-time Rendering) for TRAINING...")
        from components.environments.thor_env import ThorEnv
        env = ThorEnv(
            render=env_config["render"], 
            rho=env_config["rho"], 
            max_actions=agent_config["num_steps"],
            use_detector=(dino_detector is not None),
            detector=dino_detector,
            # Keep this consistent with GroundingDINO box/text thresholds; 0.3 is too strict for tiny objects.
            det_score_thr=0.20 if dino_detector is not None else 0.30,
        )
    else:
        from components.environments.precomputed_thor_env import PrecomputedThorEnv
        env = PrecomputedThorEnv(
            rho=env_config["rho"], 
            max_actions=agent_config["num_steps"],
            detector=dino_detector, # <--- [关键] 传入检测器
            det_score_thr=0.20 if dino_detector is not None else 0.30,
            save_debug_path=args.save_frames_to # [New] Pass viz path
        )

    # Load agent from encoder & policy weights
    if agent_config["name"] == "reinforce":
        agent = ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    elif agent_config["name"] == "a2c":
        agent = A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    else:
        raise Exception("Unknown agent")

    agent.load_weights(encoder_path=encoder_path, device=device)

    # Optional multi-GPU: split heavy RGB encoder forward across GPUs.
    if torch.cuda.is_available() and len(parsed_gpu_ids) > 1:
        agent.device = device
        if not isinstance(agent.encoder.rgb_encoder, torch.nn.DataParallel):
            agent.encoder.rgb_encoder = torch.nn.DataParallel(
                agent.encoder.rgb_encoder,
                device_ids=parsed_gpu_ids,
                output_device=parsed_gpu_ids[0],
            )
        agent.to(device)
        print(f"[INFO] Multi-GPU enabled for RGB encoder on GPUs: {parsed_gpu_ids}")

    # RL training runner
    # [Mod] Pass save_frames_to
    runner = RLTrainRunner(env=env, agent=agent, device=device, save_dir=args.save_frames_to)
    
    # [New] Override scene numbers for Habitat (since it uses a single scene configuration from args right now)
    if args.use_habitat:
        runner.total_episodes = agent_config.get("episodes", 10)  # Use config explicitly, fallback to 10
        agent.scene_numbers = [1]  # Dummy scene number
        runner.scene_numbers = agent.scene_numbers
    
    try:
        runner.run()
        print("[INFO] Training completed.")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Training interrupted by error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

        # --- Save final model ---
        if args.save_model:
            print("[INFO] Saving model checkpoint (finished or interrupted)...")
            save_folder = Path("RL_training") / "runs" / "model_weights"
            save_folder.mkdir(exist_ok=True, parents=True)
            run_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Python 3.7+ f-string fix
            file_name = f"{run_start}_{agent_config['name']}_agent.pth"
            try:
                agent.save_model(str(save_folder), file_name=file_name)
            except Exception as save_err:
                print(f"[ERROR] Failed to save model: {save_err}")


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
    # --- Register signal handler ---
    def handle_sigterm(signum, frame):
        print(f"\n[INFO] Received SIGTERM (signal {signum}). Initiating graceful shutdown...")
        # Raising KeyboardInterrupt is handled by the existing try-except block in main()
        # and will trigger the finally block to save the model.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)
    
    set_working_directory()

    from components.agents.a2c_agent import A2CAgent
    from components.agents.reinforce_agent import ReinforceAgent
    # from components.environments.thor_env import ThorEnv # Moved to main()
    # from components.environments.precomputed_thor_env import PrecomputedThorEnv # Moved to main()
    from RL_training.runner.rl_train_runner import RLTrainRunner
    from components.utils.utility_functions import read_config, set_seeds

    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str, help="Path to the configuration files.")
    parser.add_argument("--save_model", action="store_true", help="Save model weights.")
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed environment.")
    # [新增] 命令行参数控制是否开启 DINO
    parser.add_argument("--use_dino", action="store_true", help="Use Grounding DINO for perception.")
    # [New] Training Visualization
    parser.add_argument("--save_frames_to", type=str, default=None, help="Directory to save training frames (e.g. debug_train_viz).")
    
    # [New] Habitat
    parser.add_argument("--use_habitat", action="store_true", help="Use Habitat instead of AI2Thor.")
    parser.add_argument("--habitat_scene", type=str, default=None, help="Scene ID for Habitat.")
    
    args = parser.parse_args()

    # Support both single-file configs (legacy/run_config) and directory-based configs (config/)
    config_path = Path(args.conf_path)
    if config_path.is_dir():
        # Check if it contains the split config files
        if (config_path / "agent.json").exists():
             # Load structured config
            full_config = {}
            if (config_path / "agent.json").exists():
                full_config["agent_config"] = read_config(str(config_path / "agent.json"))
            if (config_path / "env.json").exists():
                full_config["env_config"] = read_config(str(config_path / "env.json"))
            if (config_path / "navigation.json").exists():
                full_config["navigation_config"] = read_config(str(config_path / "navigation.json"))
            
            seed = full_config.get("env_config", {}).get("seed", 42)
            set_seeds(seed)
            print(f"[INFO] Running training with SPLIT config from: {args.conf_path}")
            main(full_config)
            exit(0)
        else:
            # Fallback: Iterate over all json files in directory (Old Behavior)
            conf_files = sorted(list(config_path.rglob("*.json")))
            if not conf_files:
                print(f"[ERROR] No json config files found in {args.conf_path}")
                exit(1)
            
            for conf_file in conf_files:
                print(f"[INFO] Running with SINGLE-FILE configuration: {conf_file}")
                try:
                    conf = read_config(str(conf_file))
                    # Basic validation to ensure it's a full config
                    if "agent_config" in conf:
                        set_seeds(conf.get("seed", 42))
                        main(conf)
                    else:
                        print(f"[WARNING] Skipping {conf_file} - missing 'agent_config' key")
                except Exception as e:
                    print(f"[ERROR] Failed to run config {conf_file}: {e}")
            exit(0)
    else:
        # Single file provided directly
        print(f"[INFO] Running with SINGLE configuration file: {args.conf_path}")
        conf = read_config(str(config_path))
        set_seeds(conf.get("seed", 42))
        main(conf)
        exit(0)