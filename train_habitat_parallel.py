"""
Parallel Habitat RL training script.
Mirrors the setup from run_train.sh and main.py with multi-environment sampling.

Usage:
    python train_habitat_parallel.py \
        --num_workers 4 \
        --episodes 100 \
        --dataset_root /home/wgy/hm3d/scene_datasets/hm3d \
        --habitat_scenes "00016-qk9eeNeR4vw,00017-oEPjPNSPmzL,..." \
        --use_dino \
        --save_frames_to /path/to/frames
"""

import argparse
import sys
import os
import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/wgy/GroundingDINO")

from components.agents.reinforce_agent import ReinforceAgent
from components.environments.habitat_env import HabitatEnv
from components.detectors.grounding_dino_service import GroundingDINOService, GroundingDINOServicePool
from components.perception.hm3d_labels import HM3D_DINO_PROMPT
from RL_training.runner.parallel_habitat_rl_train_runner import ParallelHabitatRLTrainRunner


def _resolve_habitat_scene_path(hm3d_root: str, token: str) -> str:
    """Resolve a single scene token to its .basis.glb path."""
    token = str(token).strip()
    if not token:
        return None

    token_path = Path(token)
    if token_path.is_file():
        return str(token_path)
    if token_path.is_dir():
        candidates = sorted(token_path.glob("*.basis.glb"))
        if candidates:
            return str(candidates[0])

    root = Path(hm3d_root)
    normalized = token.zfill(5) if token.isdigit() else token
    hash_part = normalized.split('-')[-1]

    search_patterns = [
        f"train/{normalized}/{hash_part}.basis.glb",
        f"train/{normalized}*/**/*.basis.glb",
        f"val/{normalized}/{hash_part}.basis.glb",
        f"val/{normalized}*/**/*.basis.glb",
    ]

    for pattern in search_patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return str(matches[0])

    return None


def _resolve_habitat_scene_list(
    hm3d_root: str,
    single_scene: str = None,
    scene_list: str = None
) -> Tuple[List[str], List[str]]:
    """Resolve a list of scene tokens to their .basis.glb paths."""
    raw_tokens = []
    if scene_list:
        raw_tokens.extend([part.strip() for part in scene_list.split(",") if part.strip()])
    elif single_scene:
        raw_tokens = [str(single_scene).strip()]
    else:
        raw_tokens = ["00800"]

    resolved = []
    missing = []
    seen = set()
    
    for token in raw_tokens:
        scene_path = _resolve_habitat_scene_path(hm3d_root, token)
        if scene_path is None:
            missing.append(token)
            continue
        if scene_path not in seen:
            resolved.append(scene_path)
            seen.add(scene_path)

    return resolved, missing


def _build_habitat_dataset_config(
    base_config_file: str,
    scene_paths: List[str],
    dataset_root: str,
    output_dir: str = None
) -> str:
    """Build a filtered Habitat dataset config for selected scenes."""
    with open(base_config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    selected_names = set()
    for scene_path in scene_paths:
        scene_name = Path(scene_path).parent.name
        if scene_name:
            selected_names.add(scene_name)

    stage_paths = config_data.get("stages", {}).get("paths", {}).get(".glb", [])
    filtered_stage_paths = []
    for item in stage_paths:
        if not any(scene_name in item for scene_name in selected_names):
            continue
        filtered_stage_paths.append(str(Path(dataset_root) / item))

    config_data["stages"]["paths"][".glb"] = filtered_stage_paths
    if "scene_instances" in config_data and "paths" in config_data["scene_instances"]:
        config_data["scene_instances"]["paths"][".json"] = []

    target_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)

    if len(selected_names) <= 3:
        selected_tag = "_".join(sorted(selected_names))
    else:
        import hashlib
        name_str = "_".join(sorted(selected_names)).encode('utf-8')
        name_hash = hashlib.md5(name_str).hexdigest()[:8]
        selected_tag = f"{len(selected_names)}_scenes_{name_hash}"

    target_path = target_dir / f"hm3d_selected_{selected_tag}.scene_dataset_config.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return str(target_path)


def _read_mapping_config(path: Path):
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                config = json.load(f)
            else:
                import yaml
                config = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to load config file {path}: {exc}") from exc

    if config is None:
        print(f"[WARNING] Config file {path} is empty; trying fallback")
        return None
    if not isinstance(config, dict):
        raise TypeError(
            f"Config file {path} must contain a mapping, got {type(config).__name__}"
        )
    return config


def _load_config_mapping(conf_path: str, yaml_name: str, json_name: str) -> dict:
    config_dir = Path(conf_path)
    primary_path = config_dir / yaml_name
    fallback_path = config_dir / json_name

    for path in (primary_path, fallback_path):
        config = _read_mapping_config(path)
        if config is not None:
            if path != primary_path:
                print(f"[INFO] Loaded fallback config from {path}")
            return config

    raise FileNotFoundError(
        f"No usable config found for {yaml_name}; checked {primary_path} and {fallback_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent with parallel Habitat environments"
    )
    
    # Environment
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/wgy/hm3d/scene_datasets/hm3d",
        help="Path to HM3D dataset root"
    )
    parser.add_argument(
        "--habitat_scene",
        type=str,
        default=None,
        help="Single Habitat scene ID"
    )
    parser.add_argument(
        "--habitat_scenes",
        type=str,
        default=None,
        help="Comma-separated list of Habitat scene IDs"
    )
    
    # Training
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers per GPU (if env_gpu_ids is provided) or total workers")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per scene")
    parser.add_argument("--num_steps", type=int, default=4000, help="Steps per rollout")
    
    # GPU
    parser.add_argument("--gpu_ids", type=str, default="4,5,6,7",
                        help="Comma-separated GPU IDs for policy (e.g., '4,5,6,7')")
    parser.add_argument("--env_gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs for environment rendering (e.g., '4,5,6'). If provided, total workers = num_workers * len(env_gpu_ids)")
    parser.add_argument("--dino_device", type=str, default="cuda:4",
                        help="GPU device for GroundingDINO detector")
    parser.add_argument("--dino_devices", type=str, default=None,
                        help="Comma-separated GPU devices for a multi-process GroundingDINO service pool")
    
    # Other
    parser.add_argument("--use_dino", action="store_true", help="Use GroundingDINO detector")
    parser.add_argument("--save_frames_to", type=str, default="/home/wgy/RL/train_png",
                        help="Directory to save visualization frames")
    parser.add_argument("--conf_path", type=str, default="config",
                        help="Path to configuration directory")
    
    args = parser.parse_args()
    
    # Load config
    print(f"[INFO] Loading config from {args.conf_path}")
    agent_config = _load_config_mapping(args.conf_path, "agent_config.yaml", "agent.json")
    navigation_config = _load_config_mapping(args.conf_path, "navigation_config.yaml", "navigation.json")
    env_config = _load_config_mapping(args.conf_path, "env_config.yaml", "env.json")

    # Ensure numeric types are correct (YAML/JSON flavors may parse numbers as strings)
    for k in ["alpha", "gamma", "entropy_coef"]:
        if k in agent_config:
            try:
                agent_config[k] = float(agent_config[k])
            except Exception:
                pass
    for k in ["episodes", "num_steps"]:
        if k in agent_config:
            try:
                agent_config[k] = int(agent_config[k])
            except Exception:
                pass

    # This parallel Habitat entrypoint is intentionally fixed to REINFORCE + LSTM.
    # The wider project still has A2C/Transformer implementations, but this
    # training script should not silently drift into another 2x2 variant.
    agent_config["name"] = "reinforce"
    navigation_config["use_transformer"] = False
    
    # Set up device
    parsed_gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    if torch.cuda.is_available() and parsed_gpu_ids:
        device = torch.device(f"cuda:{parsed_gpu_ids[0]}")
        print(f"[INFO] Using GPU devices: {parsed_gpu_ids}, primary device: {device}")
    else:
        device = torch.device("cpu")
        parsed_gpu_ids = []
        print("[INFO] Using CPU")
    
    # Set environment variables for the runner
    os.environ["RL_GPU_IDS"] = args.gpu_ids
    os.environ["DINO_DEVICE"] = args.dino_device
    
    # Process env_gpu_ids
    env_gpu_ids = []
    if args.env_gpu_ids:
        env_gpu_ids = [int(x.strip()) for x in args.env_gpu_ids.split(",") if x.strip()]
        total_workers = args.num_workers * len(env_gpu_ids)
        print(f"[INFO] Distributing {total_workers} workers across GPUs: {env_gpu_ids} ({args.num_workers} per GPU)")
    else:
        total_workers = args.num_workers
        print(f"[INFO] Using {total_workers} workers on default GPU")
    
    # Initialize GroundingDINO service if requested
    dino_service = None
    if args.use_dino:
        print("[INFO] Initializing GroundingDINO detector...")
        dino_config = "/home/wgy/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        dino_weights = "/home/wgy/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        
        if not os.path.exists(dino_weights):
            print(f"[WARNING] DINO weights not found at {dino_weights}, running without detector")
        else:
            dino_devices = []
            if args.dino_devices:
                dino_devices = [part.strip() for part in args.dino_devices.split(",") if part.strip()]
            if not dino_devices:
                dino_devices = [args.dino_device]
            dino_text_prompt = str(env_config.get("dino_text_prompt", HM3D_DINO_PROMPT))
            dino_box_threshold = float(env_config.get("dino_box_threshold", 0.35))
            dino_text_threshold = float(env_config.get("dino_text_threshold", 0.30))
            print(
                f"[INFO] DINO HM3D prompt with {dino_text_prompt.count('.')} labels; "
                f"box_threshold={dino_box_threshold:.2f}, text_threshold={dino_text_threshold:.2f}"
            )

            if len(dino_devices) == 1:
                dino_service = GroundingDINOService(
                    config_path=dino_config,
                    checkpoint_path=dino_weights,
                    device=dino_devices[0],
                    text_prompt=dino_text_prompt,
                    box_threshold=dino_box_threshold,
                    text_threshold=dino_text_threshold,
                )
                print(f"[INFO] DINO service initialized on {dino_devices[0]}")
            else:
                dino_service = GroundingDINOServicePool(
                    config_path=dino_config,
                    checkpoint_path=dino_weights,
                    devices=dino_devices,
                    text_prompt=dino_text_prompt,
                    box_threshold=dino_box_threshold,
                    text_threshold=dino_text_threshold,
                )
                print(f"[INFO] DINO service pool initialized on {dino_devices}")
    
    # Create output directory
    os.makedirs(args.save_frames_to, exist_ok=True)
    
    # Resolve Habitat scenes
    print("[INFO] Resolving Habitat scenes...")
    hm3d_root = args.dataset_root
    habitat_scene_ids, missing_scenes = _resolve_habitat_scene_list(
        hm3d_root,
        single_scene=args.habitat_scene,
        scene_list=args.habitat_scenes,
    )
    
    if missing_scenes:
        print(f"[WARNING] Missing scenes: {missing_scenes}")
    
    if not habitat_scene_ids:
        print("[ERROR] No valid Habitat scenes found")
        sys.exit(1)
    
    print(f"[INFO] Using {len(habitat_scene_ids)} Habitat scenes:")
    for i, scene_path in enumerate(habitat_scene_ids[:5], 1):
        print(f"  {i}. {Path(scene_path).parent.name}")
    if len(habitat_scene_ids) > 5:
        print(f"  ... and {len(habitat_scene_ids) - 5} more")
    
    # Build filtered dataset config
    print("[INFO] Building filtered Habitat dataset config...")
    base_config_file = os.path.join(hm3d_root, "hm3d_annotated_basis.scene_dataset_config.json")
    dataset_config = _build_habitat_dataset_config(
        base_config_file,
        habitat_scene_ids,
        hm3d_root,
        output_dir=args.save_frames_to,
    )
    print(f"[INFO] Dataset config: {dataset_config}")
    
    # Create dummy HabitatEnv for agent initialization
    print("[INFO] Creating dummy HabitatEnv for agent initialization...")
    # Only forward env_config keys that HabitatEnv actually accepts to avoid
    # unexpected keyword argument errors (e.g., 'seed').
    allowed_env_keys = {
        "width",
        "height",
        "score_norm_target",
        "instance_merge_dist",
        "coverage_cell_size",
        "nav_sample_points",
        "topdown_meters_per_pixel",
        "agent_radius",
        "agent_height",
        "agent_max_climb",
        "navmesh_cell_height",
        "navmesh_cell_size",
        "fill_position_from_gt",
        "rho",
        "coverage_bonus_scale",
        "discovery_bonus_scale",
        "collision_penalty",
        "gt_validation_iou_threshold",
        "gt_validation_mode",
        "success_recall_threshold",
        "success_reward",
        "reward_excluded_labels",
        "max_actions",
        "save_debug_interval",
        "save_debug_path",
    }

    extra_env_kwargs = {
        k: v for k, v in env_config.items()
        if k in allowed_env_keys and v is not None
    }
    extra_env_kwargs.setdefault("max_actions", int(agent_config.get("num_steps", 4000)))
    extra_env_kwargs.setdefault("save_debug_path", args.save_frames_to)

    dummy_env = HabitatEnv(
        dataset_root=hm3d_root,
        config_file=dataset_config,
        scene_id=habitat_scene_ids[0],
        scene_ids=habitat_scene_ids,
        render=False,
        use_detector=False,
        detector=None,
        det_score_thr=float(env_config.get("det_score_thr", 0.30)),
        **extra_env_kwargs,
    )
    
    # Create REINFORCE + LSTM agent
    print("[INFO] Creating REINFORCE + LSTM agent...")
    agent = ReinforceAgent(
        env=dummy_env,
        navigation_config=navigation_config,
        agent_config=agent_config,
        device=device,
    )
    
    # Load pre-trained encoder if available
    encoder_path = Path(agent_config.get("encoder_path", ""))
    if encoder_path.exists():
        encoder_path = encoder_path.parent / (encoder_path.stem + "_" + str(navigation_config["use_transformer"]) + encoder_path.suffix)
        if encoder_path.exists():
            print(f"[INFO] Loading encoder weights from {encoder_path}")
            agent.load_weights(encoder_path=str(encoder_path), device=str(device))
    
    # Multi-GPU setup for RGB encoder
    if torch.cuda.is_available() and len(parsed_gpu_ids) > 1:
        print(f"[INFO] Enabling DataParallel for RGB encoder on GPUs: {parsed_gpu_ids}")
        agent.encoder.rgb_encoder = torch.nn.DataParallel(
            agent.encoder.rgb_encoder,
            device_ids=parsed_gpu_ids,
            output_device=parsed_gpu_ids[0],
        )
    
    # Create parallel training runner
    print(f"[INFO] Creating parallel runner with {total_workers} workers...")
    runner = ParallelHabitatRLTrainRunner(
        agent=agent,
        dataset_root=hm3d_root,
        config_file=dataset_config,
        num_workers=total_workers,
        device=device,
        save_dir=args.save_frames_to,
        base_scene_ids=[Path(p).parent.name for p in habitat_scene_ids],
        detection_service=dino_service,
        env_config=env_config,
        scene_count=len(habitat_scene_ids),
        env_gpu_ids=env_gpu_ids if len(env_gpu_ids) > 0 else None,
    )
    
    # Adjust episode count based on number of scenes
    runner.total_episodes = args.episodes * len(habitat_scene_ids)
    print(f"[INFO] Total episodes: {runner.total_episodes} ({args.episodes} per scene × {len(habitat_scene_ids)} scenes)")
    
    # Start training
    print("[INFO] Starting parallel training...")
    try:
        runner.run()
        print("[INFO] Training completed successfully")
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        runner.close()
        try:
            dummy_env.close()
        except:
            pass


if __name__ == "__main__":
    main()
