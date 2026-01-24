"""Small ablation: compare short RL runs with GT vs simulated detector.

This runs very short training (few episodes) on a PrecomputedThorEnv to estimate
impact of detection noise. Designed as a quick smoke-test, not a full experiment.

Usage: python RL_training/experiments/detector_vs_gt_ablation.py
"""
import json
import os
from pathlib import Path

import torch

from components.environments.precomputed_thor_env import PrecomputedThorEnv
from components.perception.simulated_detector import SimulatedDetector
from components.agents.a2c_agent import A2CAgent
from components.agents.reinforce_agent import ReinforceAgent
from RL_training.runner.rl_train_runner import RLTrainRunner


def run_short(conf_file, use_detector=False, detector_kwargs=None, episodes=3, num_scenes=2):
    detector_kwargs = detector_kwargs or {}
    conf = json.load(open(conf_file, "r"))

    # shorten the run for smoke test
    conf["agent_config"]["episodes"] = episodes
    conf["agent_config"]["num_steps"] = conf["agent_config"].get("num_steps", 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate environment with optional detector
    env = PrecomputedThorEnv(rho=conf["env_config"].get("rho", 0.02), max_actions=conf["agent_config"]["num_steps"], use_detector=use_detector, detector=SimulatedDetector(**detector_kwargs) if use_detector else None)

    # simple pick agent type
    name = conf["agent_config"]["name"].lower()
    if name.startswith("a2c"):
        agent = A2CAgent(env=env, navigation_config=conf["navigation_config"], agent_config=conf["agent_config"], device=device)
    elif name.startswith("reinforce"):
        agent = ReinforceAgent(env=env, navigation_config=conf["navigation_config"], agent_config=conf["agent_config"], device=device)
    else:
        raise RuntimeError("Unknown agent type in config")

    # restrict scenes to a few for speed
    agent.scene_numbers = agent.all_scene_numbers[:num_scenes]

    # run
    runner = RLTrainRunner(env=env, agent=agent, device=device)
    runner.run()

    # collect results from runner.ep_info_buffer
    scores = [d.get("score", 0.0) for d in list(runner.ep_info_buffer)]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {"mean_score": mean_score, "raw_scores": scores}


def main():
    # pick a short config (exists in repo)
    conf_path = Path("RL_training/sbatch/sl_configs/A2C_LSTM/conf_1/config_116739.json")
    if not conf_path.exists():
        print("Config not found, please point to an existing config JSON")
        return

    print("Running GT baseline (PrecomputedThorEnv, GT observations)")
    res_gt = run_short(str(conf_path), use_detector=False, episodes=3, num_scenes=2)
    print("GT mean score:", res_gt)

    print("Running SimulatedDetector baseline (miss_rate=0.2, false_pos_rate=0.1)")
    res_det = run_short(str(conf_path), use_detector=True, detector_kwargs={"miss_rate": 0.2, "false_pos_rate": 0.1, "score_noise": 0.1}, episodes=3, num_scenes=2)
    print("Detector mean score:", res_det)


if __name__ == "__main__":
    main()
