"""
Multi-environment parallel collector for RL training.
Each worker runs its own environment instance and collects transitions independently.
Transitions are sent via queues to the main training process.
"""

import multiprocessing as mp
import queue
import sys
import time
import traceback
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from components.utils.observation import Observation


def worker_env_loop(
    worker_id: int,
    env_class: str,
    env_kwargs: Dict,
    action_queue: mp.Queue,
    transition_queue: mp.Queue,
    stop_event: mp.Event,
    error_queue: mp.Queue,
):
    """
    Worker process: runs a single environment instance and collects transitions.
    - Pulls actions from action_queue
    - Executes env.step(action)
    - Sends transition (obs, reward, done, info) via transition_queue
    """
    try:
        # Import locally to avoid pickling issues
        if env_class == "ThorEnv":
            from components.environments.thor_env import ThorEnv
            env = ThorEnv(**env_kwargs)
        elif env_class == "PrecomputedThorEnv":
            from components.environments.precomputed_thor_env import PrecomputedThorEnv
            env = PrecomputedThorEnv(**env_kwargs)
        elif env_class == "HabitatEnv":
            from components.environments.habitat_env import HabitatEnv
            env = HabitatEnv(**env_kwargs)
        else:
            raise ValueError(f"Unknown env_class: {env_class}")
        
        print(f"[Worker {worker_id}] Environment initialized")
        
        episode_count = 0
        scene_numbers = env_kwargs.get("scene_numbers", [1, 2, 3])
        scene_idx = 0
        
        obs = env.reset(scene_number=scene_numbers[scene_idx])
        print(f"[Worker {worker_id}] Episode {episode_count} started, scene {scene_numbers[scene_idx]}")
        
        while not stop_event.is_set():
            try:
                # Try to get action with timeout (non-blocking if queue empty)
                action = action_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Execute step
            next_obs = env.step(action)
            reward = next_obs.reward
            done = next_obs.terminated or next_obs.truncated
            info = next_obs.info
            
            # Send transition
            transition = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done,
                "next_obs": next_obs,
                "info": info,
                "worker_id": worker_id,
            }
            transition_queue.put(transition)
            
            if done:
                # Reset for next episode
                episode_count += 1
                scene_idx = (scene_idx + 1) % len(scene_numbers)
                obs = env.reset(scene_number=scene_numbers[scene_idx])
                print(f"[Worker {worker_id}] Episode {episode_count} started, scene {scene_numbers[scene_idx]}")
            else:
                obs = next_obs
    
    except Exception as e:
        error_queue.put((worker_id, traceback.format_exc()))
        print(f"[Worker {worker_id}] ERROR: {e}")
        sys.exit(1)


class ParallelEnvCollector:
    """
    Manages multiple worker processes, each running an environment.
    Collects transitions in parallel and provides them to the main training loop.
    """
    
    def __init__(
        self,
        num_workers: int,
        env_class: str,
        env_kwargs: Dict,
        scene_numbers: Optional[List[int]] = None,
    ):
        """
        Args:
            num_workers: Number of parallel environment workers
            env_class: Name of environment class ("ThorEnv", "PrecomputedThorEnv", "HabitatEnv")
            env_kwargs: Arguments to pass to environment constructor
            scene_numbers: Scene indices to cycle through
        """
        self.num_workers = num_workers
        self.env_class = env_class
        self.env_kwargs = env_kwargs.copy()
        
        # Prepare scene numbers for each worker
        if scene_numbers is None:
            scene_numbers = list(range(1, 31))  # Default: all scenes
        self.scene_numbers = scene_numbers
        
        # Distribute scenes to workers (round-robin)
        self.env_kwargs["scene_numbers"] = self.scene_numbers
        
        # Create queues and events
        self.action_queues = [mp.Queue() for _ in range(num_workers)]
        self.transition_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.error_queue = mp.Queue()
        
        # Start worker processes
        self.processes = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_env_loop,
                args=(
                    worker_id,
                    env_class,
                    self.env_kwargs,
                    self.action_queues[worker_id],
                    self.transition_queue,
                    self.stop_event,
                    self.error_queue,
                ),
                daemon=False,
            )
            p.start()
            self.processes.append(p)
            print(f"[ParallelCollector] Worker {worker_id} started (PID {p.pid})")
        
        # Track which worker is expecting an action next
        self.next_worker_idx = 0
        
        print(f"[ParallelCollector] Initialized {num_workers} workers")
    
    def send_actions(self, actions: np.ndarray) -> None:
        """
        Send a batch of actions to workers (one per worker, round-robin).
        
        Args:
            actions: Array of shape [num_workers,] with action indices
        """
        if len(actions) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} actions, got {len(actions)}")
        
        for worker_id, action in enumerate(actions):
            try:
                self.action_queues[worker_id].put(int(action), block=False)
            except queue.Full:
                print(f"[WARNING] Action queue for worker {worker_id} is full, skipping")
    
    def collect_transitions(self, num_transitions: int = 1) -> List[Dict]:
        """
        Collect a batch of transitions from the queue.
        Blocks until num_transitions are available.
        
        Args:
            num_transitions: Number of transitions to collect
        
        Returns:
            List of transition dicts
        """
        transitions = []
        timeout_per_transition = 5.0  # seconds
        
        for _ in range(num_transitions):
            try:
                trans = self.transition_queue.get(timeout=timeout_per_transition)
                transitions.append(trans)
            except queue.Empty:
                print(f"[WARNING] Timeout collecting transition {len(transitions)}/{num_transitions}")
                break
        
        # Check for errors from workers
        while not self.error_queue.empty():
            worker_id, error_msg = self.error_queue.get_nowait()
            print(f"[ERROR] Worker {worker_id} crashed:\n{error_msg}")
        
        return transitions
    
    def step(self, actions: np.ndarray) -> List[Dict]:
        """
        Convenience: send actions and immediately collect results.
        
        Args:
            actions: Array of shape [num_workers,]
        
        Returns:
            List of transitions
        """
        self.send_actions(actions)
        return self.collect_transitions(num_transitions=self.num_workers)
    
    def close(self) -> None:
        """Stop all worker processes."""
        print("[ParallelCollector] Stopping workers...")
        self.stop_event.set()
        
        for p in self.processes:
            p.join(timeout=5.0)
            if p.is_alive():
                print(f"[WARNING] Process {p.pid} did not terminate, killing...")
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()
        
        print("[ParallelCollector] All workers stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass
