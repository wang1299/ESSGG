import json
import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage


def save_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Persist progress after each *completed* trial."""
    # Make sure output dirs exist
    out_dir = Path("optim")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Save current best params (robust even if best_trial doesn't change)
    best = study.best_trial
    output_model_name = "transformer" if args.use_transformer else "lstm"
    best_path = out_dir / f"best_params_{agent}_{output_model_name}.json"
    with open(best_path, "w") as f:
        json.dump(best.params, f, indent=2)

    # 2) Optional: dump full trials table for auditing/analysis
    #    Note: `trials_dataframe` is cheap enough at this cadence.
    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "system_attrs"))
    df.to_csv(out_dir / f"trials_{agent}_{output_model_name}.csv", index=False)


def get_rho_search_space(use_transformer):
    if use_transformer:
        return [0.12, 0.124, 0.126, 0.128]
    else:
        raise NotImplementedError


def objective(trial, agent, use_transformer):
    set_seeds(42)

    if use_transformer:
        rho_space = get_rho_search_space(use_transformer)
        rho = trial.suggest_categorical("rho", rho_space)
    else:
        rho = trial.suggest_float("rho", 0.03, 0.043, step=0.001)

    agent_config = read_config("config/agent.json", use_print=False)
    agent_config["name"] = agent

    navigation_config = read_config("config/navigation.json", use_print=False)
    navigation_config["use_transformer"] = use_transformer

    env_config = read_config("config/env.json", use_print=False)
    env_config["rho"] = rho

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PrecomputedThorEnv(render=False, rho=rho, max_actions=agent_config["num_steps"])

    runner = RLTrialRunner(
        trial,
        env,
        navigation_config,
        agent_config,
        device,
        params={"rho": rho},
        num_agents=args.num_agents,
        max_episodes=args.max_episodes,
        n_jobs=args.n_jobs,
    )

    try:
        return runner.run()
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception as e:
                print(f"[WARNING] Failed to stop AI2-THOR controller: {e}")


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working directory:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()

    from components.environments.precomputed_thor_env import PrecomputedThorEnv
    from components.utils.utility_functions import read_config, set_seeds
    from optim.runner.trial_runner import RLTrialRunner

    parser = ArgumentParser()

    parser.add_argument(
        "--use_transformer", action="store_true", help="Use transformer model for the agent. If not set, LSTM will be used."
    )
    parser.add_argument("--max_episodes", type=int, default=600, help="Maximum number of episodes to run for each trial.")
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents to run in parallel during optimization.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes to run in parallel.")
    args = parser.parse_args()

    agent = "reinforce"
    navigation_config = read_config("config/navigation.json", use_print=False)
    use_transformer = args.use_transformer

    print(f"Agent: {agent}")
    print(f"Using model: {'transformer' if use_transformer else 'lstm'}")

    # use a TPESampler for intelligent, history-aware sampling
    storage = RDBStorage(
        url=f"sqlite:///optim/optuna_reinforce_{'transformer' if use_transformer else 'lstm'}.db",  # local DB file in working dir
        heartbeat_interval=60,  # seconds; trial sends heartbeat periodically
        grace_period=120,  # fail trial if no heartbeat within this window
    )
    if use_transformer:
        sampler = optuna.samplers.GridSampler(search_space={"rho": get_rho_search_space(use_transformer)})
    else:
        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=2, multivariate=True)  # first 2 trials are random for exploration
    pruner = MedianPruner(
        n_startup_trials=2 if not use_transformer else len(get_rho_search_space(use_transformer)), n_warmup_steps=6, interval_steps=1
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="param_opt_reinforce",
        storage=storage,
        load_if_exists=True,  # critical for resume
    )

    # --- run optimization with callback to persist after each trial ---
    study.optimize(
        partial(objective, agent=agent, use_transformer=use_transformer),
        n_trials=60 if not use_transformer else len(get_rho_search_space(use_transformer)),
        n_jobs=args.n_jobs,
        callbacks=[save_progress_callback],  # persist after every finished trial
    )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best rho: {study.best_trial.params['rho']}")
    print(f"Best entropy coefficient: {study.best_trial.params['entropy_coef']}")

    output_model_name = "lstm" if not use_transformer else "transformer"
    with open(f"optim/best_params_{agent}_{output_model_name}.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
