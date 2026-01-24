import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch

from components.agents.a2c_agent import A2CAgent
from components.agents.reinforce_agent import ReinforceAgent
from components.environments.precomputed_thor_env import PrecomputedThorEnv
from components.utils.utility_functions import read_config


def _make_agent(agent_config, navigation_config, env, device):
    name = agent_config["name"].lower()
    if name.startswith("reinforce"):
        return ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config, device=device)
    elif name.startswith("a2c"):
        return A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_config['name']}")


def _sample_eval_starts(env, scene_numbers, starts_per_scene, seed=0):
    """Create a fixed evaluation set of (scene, start_pos, start_rot) tuples."""
    rng = np.random.RandomState(seed)
    starts = []
    for scene in scene_numbers:
        for _ in range(starts_per_scene):
            # Force a random start but record it to reuse across all weights
            obs = env.reset(scene_number=scene, random_start=True)
            pos = obs.info["event"].metadata["agent"]["position"]
            rot = obs.info["event"].metadata["agent"]["rotation"]
            starts.append((scene, pos, rot))
    return starts


def _run_episode_and_collect(agent, env, scene, start_pos, start_rot, max_steps=500):
    """Run a single episode from a fixed start and return (steps, score)."""
    # Optional: reset recurrent state if your agent uses RNNs
    if hasattr(agent, "reset"):
        agent.reset()

    obs = env.reset(scene_number=scene, random_start=False, start_position=start_pos, start_rotation=start_rot)

    steps = 0
    while not (obs.terminated or obs.truncated):
        action, *_ = agent.get_action(obs)
        obs = env.step(action)
        steps += 1
        if steps >= max_steps:
            break

    ep_steps = int(obs.info.get("steps", steps))
    ep_score = float(obs.info.get("score", 0.0))
    return ep_steps, ep_score


def _evaluate_weights(weight_paths, agent_config, navigation_config, env, eval_starts, device):
    """Return a list of dicts with mean/std steps & scores for each weight file."""
    results = []
    agent = _make_agent(agent_config, navigation_config, env, device)

    for w in sorted(weight_paths):
        print(f"[INFO] Loading model weights from: {w}")
        agent.load_weights(model_path=str(w), device=device)

        step_list, score_list = [], []
        for scene, pos, rot in eval_starts:
            s, sc = _run_episode_and_collect(agent, env, scene, pos, rot)
            step_list.append(s)
            score_list.append(sc)

        steps = np.array(step_list, dtype=np.float32)
        scores = np.array(score_list, dtype=np.float32)

        res = {
            "weights_path": str(w),
            "mean_steps": float(steps.mean()),
            "std_steps": float(steps.std(ddof=1)) if len(steps) > 1 else 0.0,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "n_episodes": int(len(steps)),
        }
        print(
            f"[EVAL] {w.name}: steps={res['mean_steps']:.2f}±{res['std_steps']:.2f}, "
            f"score={res['mean_score']:.3f}±{res['std_score']:.3f} (n={res['n_episodes']})"
        )
        results.append(res)

    return results


def _pick_best(results, target_min=23, target_max=26, max_mean_steps=39):
    """
    Selection rule:
      - Disqualify any checkpoints with mean_steps >= max_mean_steps.
      1) Prefer weights with mean_steps in [target_min, target_max];
         among them choose highest mean_score, tie-break by smallest std_steps,
         then smallest |mean_steps - mid|.
      2) If none in range: pick minimal distance to the interval (0 if touching),
         tie-break by highest mean_score, then smallest std_steps.
    """
    # Filter out too long runs
    valid = [r for r in results if r["mean_steps"] < max_mean_steps]
    if not valid:
        raise RuntimeError(f"No valid checkpoints: all mean_steps >= {max_mean_steps}")

    mid = 0.5 * (target_min + target_max)

    in_range = [r for r in results if target_min <= r["mean_steps"] <= target_max]
    if in_range:
        in_range.sort(key=lambda r: (-r["mean_score"], r["std_steps"], abs(r["mean_steps"] - mid)))
        return in_range[0], {"reason": "in_range_highest_score_then_stability"}

    # Distance to interval: 0 inside, otherwise amount outside
    def dist_to_interval(m):
        if m < target_min:
            return target_min - m
        if m > target_max:
            return m - target_max
        return 0.0

    ranked = sorted(
        results, key=lambda r: (dist_to_interval(r["mean_steps"]), -r["mean_score"], r["std_steps"], abs(r["mean_steps"] - mid))
    )
    return ranked[0], {"reason": "closest_to_range_then_score_then_stability"}


def get_best_weights(
    agent_config,
    navigation_config,
    scene_numbers,
    starts_per_scene=3,
    weights_dir="RL_training/runs/model_weights/",
    seed=123,
    device=None,
    target_range=(28.0, 32.0),
):
    # Guard: ensure a *specific* directory is passed
    if weights_dir == "RL_training/runs/model_weights/":
        raise ValueError("Please pass a specific weights_dir, not the root folder.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = PrecomputedThorEnv()

    # 1) Build a fixed evaluation set shared by all checkpoints
    eval_starts = _sample_eval_starts(env, scene_numbers, starts_per_scene, seed=seed)

    # 2) Collect weight files
    weight_paths = list(Path(weights_dir).glob("*.pth"))
    if not weight_paths:
        raise FileNotFoundError(f"No *.pth found in {weights_dir}")

    # 3) Evaluate all
    results = _evaluate_weights(weight_paths, agent_config, navigation_config, env, eval_starts, device)

    # 4) Pick the best per rule
    best, info = _pick_best(results, target_min=target_range[0], target_max=target_range[1])
    print(
        f"[SELECT] best={Path(best['weights_path']).name} "
        f"(mean_steps={best['mean_steps']:.2f}±{best['std_steps']:.2f}, "
        f"mean_score={best['mean_score']:.3f}, n={best['n_episodes']}) "
        f"reason={info['reason']}"
    )
    return best, results


def write_results_csv(csv_path, agent_name, results):
    """Write per-weight evaluation results (one row per checkpoint) to CSV."""
    header = ["agent_name", "weights_path", "mean_steps", "std_steps", "mean_score", "std_score", "n_episodes"]
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "agent_name": agent_name,
                    "weights_path": r["weights_path"],
                    "mean_steps": f'{r["mean_steps"]:.6f}',
                    "std_steps": f'{r["std_steps"]:.6f}',
                    "mean_score": f'{r["mean_score"]:.6f}',
                    "std_score": f'{r["std_score"]:.6f}',
                    "n_episodes": r["n_episodes"],
                }
            )


def build_agent_config_and_weights(scene_numbers):
    """
    Build the agent_config_and_weights dict using the BEST checkpoint per agent family.
    Returns the dict and a human-readable summary string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Define your four agent families with config and weight directories
    agent_defs = {
        "REINFORCE_LSTM": {
            "agent_cfg_path": "RL_training/sbatch/configs/REINFORCE_LSTM/agent.json",
            "nav_cfg_path": "RL_training/sbatch/configs/REINFORCE_LSTM/navigation.json",
            "weights_dir": "RL_training/runs/model_weights/REINFORCE_Agent_LSTM",
        },
        "REINFORCE_Transformer": {
            "agent_cfg_path": "RL_training/sbatch/configs/REINFORCE_Transformer/agent.json",
            "nav_cfg_path": "RL_training/sbatch/configs/REINFORCE_Transformer/navigation.json",
            "weights_dir": "RL_training/runs/model_weights/REINFORCE_Agent_Transformer",
        },
        "A2C_LSTM": {
            "agent_cfg_path": "RL_training/sbatch/configs/A2C_LSTM/agent.json",
            "nav_cfg_path": "RL_training/sbatch/configs/A2C_LSTM/navigation.json",
            "weights_dir": "RL_training/runs/model_weights/A2C_Agent_LSTM",
        },
        "A2C_Transformer": {
            "agent_cfg_path": "RL_training/sbatch/configs/A2C_Transformer/agent.json",
            "nav_cfg_path": "RL_training/sbatch/configs/A2C_Transformer/navigation.json",
            "weights_dir": "RL_training/runs/model_weights/A2C_Agent_Transformer",
        },
    }

    # 2) For each family: evaluate all .pth in dir, pick best, store chosen path
    agent_config_and_weights = {}
    summary_lines = []
    csv_dir = Path("RL_training/eval_summaries")

    for agent_name, d in agent_defs.items():
        agent_cfg = read_config(d["agent_cfg_path"], use_print=False)
        nav_cfg = read_config(d["nav_cfg_path"], use_print=False)
        weights_dir = d["weights_dir"]

        best, results = get_best_weights(
            agent_config=agent_cfg,
            navigation_config=nav_cfg,
            scene_numbers=scene_numbers,
            starts_per_scene=4,  # adjust as you like
            weights_dir=weights_dir,
            seed=123,  # fixed seed for reproducibility
            device=device,
            target_range=(23, 26),
        )

        # Persist detailed results per family (optional but recommended)
        write_results_csv(csv_dir / f"{agent_name}_results.csv", agent_name, results)

        chosen = best["weights_path"]
        ms, ss = best["mean_steps"], best["std_steps"]
        sc, scs = best["mean_score"], best["std_score"]

        summary_lines.append(
            f"{agent_name}: {Path(chosen).name}  " f"(mean_steps={ms:.2f}±{ss:.2f}, mean_score={sc:.3f}±{scs:.3f}, n={best['n_episodes']})"
        )

        # Fill downstream dict with chosen checkpoint path
        agent_config_and_weights[agent_name] = (agent_cfg, nav_cfg, chosen)

    summary = "\n".join(summary_lines)
    return agent_config_and_weights, summary


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working director:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()
    # Define your validation scenes once (used for checkpoint selection)
    all_scene_numbers = list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))
    scene_numbers = all_scene_numbers[:10]

    agent_config_and_weights, summary = build_agent_config_and_weights(scene_numbers)

    print("\n=== Selected best checkpoints per agent family ===")
    print(summary)
