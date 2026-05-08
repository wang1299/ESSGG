"""
Multi-process Habitat environment collector for parallel sampling.

Each worker process runs an independent HabitatEnv and communicates with the main process
via queues. The main process collects observations from all workers, performs batch inference,
and distributes actions back to workers.
"""

import multiprocessing as mp
import queue
import traceback
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from components.utils.observation import Observation
import time


def _worker_loop(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    dataset_root: str,
    config_file: str,
    base_scene_ids: List[str],
    env_kwargs: Dict[str, Any],
):
    """
    Worker process main loop.
    
    Receives tasks from main process:
    - ("reset", scene_id)
    - ("step", action_id)
    - ("close",)
    
    Sends results back via result_queue.
    """
    from components.environments.habitat_env import HabitatEnv
    
    env = None
    
    try:
        # Create environment once per worker
        local_env_kwargs = dict(env_kwargs or {})
        local_env_kwargs.setdefault("worker_id", worker_id)
        env = HabitatEnv(
            dataset_root=dataset_root,
            config_file=config_file,
            scene_id=base_scene_ids[0] if base_scene_ids else "0",
            scene_ids=base_scene_ids,
            **local_env_kwargs
        )
        # Notify main process that worker initialized successfully
        try:
            result_queue.put((worker_id, "init_ok", None))
        except Exception:
            pass
        
        while True:
            try:
                task = task_queue.get(timeout=30)
            except queue.Empty:
                continue
            
            if task is None or (isinstance(task, tuple) and task[0] == "close"):
                break
            
            try:
                if task[0] == "reset":
                    scene_id = task[1] if len(task) > 1 else None
                    random_start = task[2] if len(task) > 2 else True
                    episode_tag = task[3] if len(task) > 3 else None
                    obs = env.reset(
                        scene_number=scene_id,
                        random_start=random_start,
                        episode_tag=episode_tag,
                    )
                    result_queue.put((worker_id, "reset_ok", obs))
                    
                elif task[0] == "step":
                    action_id = task[1]
                    obs = env.step(action_id)
                    result_queue.put((worker_id, "step_ok", obs))

                elif task[0] == "annotate_detections":
                    detections = task[1] if len(task) > 1 else None
                    annotated = env.annotate_detections(detections)
                    result_queue.put((worker_id, "annotate_ok", annotated))

                elif task[0] == "finalize_episode":
                    reason = task[1] if len(task) > 1 else "done"
                    result = env.finalize_episode(reason=reason)
                    result_queue.put((worker_id, "finalize_ok", result))
                    
                else:
                    result_queue.put((worker_id, "error", f"Unknown task: {task[0]}"))
                    
            except Exception as e:
                error_msg = f"Worker {worker_id} error: {str(e)}\n{traceback.format_exc()}"
                result_queue.put((worker_id, "error", error_msg))
    
    except Exception as e:
        error_msg = f"Worker {worker_id} init error: {str(e)}\n{traceback.format_exc()}"
        result_queue.put((worker_id, "init_error", error_msg))
    
    finally:
        if env is not None:
            try:
                env.close()
            except:
                pass


class ParallelHabitatCollector:
    """
    Manages multiple HabitatEnv instances running in separate processes.
    
    Provides synchronous API:
    - reset_all(scene_ids) -> [Obs, ...]
    - step_all(actions) -> [Obs, ...]
    """
    
    def __init__(
        self,
        num_workers: int,
        dataset_root: str,
        config_file: str,
        base_scene_ids: Optional[List[str]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
        env_gpu_ids: Optional[List[int]] = None,
    ):
        """
        Args:
            num_workers: Number of parallel worker processes
            dataset_root: Path to Habitat dataset root
            config_file: Path to Habitat config file
            base_scene_ids: List of scene IDs to use (will be distributed among workers)
            env_kwargs: Additional kwargs for HabitatEnv
            timeout: Timeout for worker responses (seconds)
            env_gpu_ids: List of GPU IDs to distribute workers across
        """
        self.num_workers = num_workers
        self.dataset_root = dataset_root
        self.config_file = config_file
        self.base_scene_ids = base_scene_ids or ["0"]
        self.env_kwargs = env_kwargs or {}
        self.timeout = timeout
        self.env_gpu_ids = env_gpu_ids
        
        self.mp_ctx = mp.get_context("spawn")
        self.task_queues: List[mp.Queue] = []
        self.result_queue = self.mp_ctx.Queue()
        self.processes: List[mp.Process] = []
        
        self._start_workers()
    
    def _start_workers(self):
        """Start all worker processes."""
        for worker_id in range(self.num_workers):
            task_q = self.mp_ctx.Queue()
            self.task_queues.append(task_q)
            
            # Determine which GPU this worker should use if env_gpu_ids is provided
            local_env_kwargs = dict(self.env_kwargs)
            if self.env_gpu_ids and len(self.env_gpu_ids) > 0:
                # We pass the logical gpu_id. In CUDA_VISIBLE_DEVICES="4,5,6,7",
                # logical ID 0 is 4, 1 is 5, 2 is 6.
                gpu_id = self.env_gpu_ids[worker_id % len(self.env_gpu_ids)]
                local_env_kwargs["gpu_device_id"] = gpu_id
            
            p = self.mp_ctx.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    task_q,
                    self.result_queue,
                    self.dataset_root,
                    self.config_file,
                    self.base_scene_ids,
                    local_env_kwargs,
                ),
                daemon=False,
            )
            p.start()
            self.processes.append(p)
            # Brief stagger to avoid thrashing I/O / simulator initialization
            time.sleep(0.5)
        
        print(f"[ParallelHabitatCollector] Started {self.num_workers} worker processes")

        # Wait for every worker to report that its HabitatEnv initialized successfully.
        initialized = set()
        deadline = time.time() + self.timeout
        while len(initialized) < self.num_workers:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Worker init timeout after {self.timeout}s")
            try:
                worker_id, status, result = self.result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                continue

            if status == "init_ok":
                initialized.add(worker_id)
                continue

            if status == "init_error":
                raise RuntimeError(f"Worker {worker_id} init failed: {result}")

            if status in {"reset_ok", "step_ok"}:
                # Should not happen during startup, but keep the queue from blocking.
                continue

            if status == "timeout":
                continue

            raise RuntimeError(f"Worker {worker_id} unexpected startup status {status}: {result}")
    
    def reset_all(
        self,
        scene_ids: Optional[List[str]] = None,
        random_start: bool = True,
        episode_tags: Optional[List[str]] = None,
    ) -> List[Observation]:
        """
        Reset all environments.
        
        Args:
            scene_ids: Optional list of scene IDs (one per worker; if not provided, use default)
            random_start: Whether to randomize start position
        
        Returns:
            List of observations, one per worker
        """
        if scene_ids is None:
            scene_ids = [self.base_scene_ids[i % len(self.base_scene_ids)] for i in range(self.num_workers)]
        
        if episode_tags is None:
            episode_tags = [None] * self.num_workers

        # Send reset tasks to all workers
        for i, (task_q, scene_id) in enumerate(zip(self.task_queues, scene_ids)):
            episode_tag = episode_tags[i] if i < len(episode_tags) else None
            task_q.put(("reset", scene_id, random_start, episode_tag))
        
        # Collect results
        observations = [None] * self.num_workers
        received = 0
        while received < self.num_workers:
            try:
                worker_id, status, result = self.result_queue.get(timeout=self.timeout)
                if status == "reset_ok":
                    observations[worker_id] = result
                    received += 1
                elif status == "timeout":
                    continue
                else:
                    raise RuntimeError(f"Worker {worker_id} reset failed: {result}")
            except queue.Empty:
                raise TimeoutError(f"Worker reset timeout after {self.timeout}s")
        
        return observations

    def reset_one(
        self,
        worker_id: int,
        scene_id: Optional[str] = None,
        random_start: bool = True,
        episode_tag: Optional[str] = None,
    ) -> Observation:
        """Reset a single environment worker and return its observation."""
        if worker_id < 0 or worker_id >= self.num_workers:
            raise IndexError(f"worker_id out of range: {worker_id}")

        self.task_queues[worker_id].put(("reset", scene_id, random_start, episode_tag))

        deadline = time.time() + self.timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Worker {worker_id} reset timeout after {self.timeout}s")

            try:
                got_worker_id, status, result = self.result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                continue

            if got_worker_id != worker_id:
                # Put unrelated results back at the tail so other callers can consume them.
                self.result_queue.put((got_worker_id, status, result))
                continue

            if status == "reset_ok":
                return result
            if status == "timeout":
                continue
            raise RuntimeError(f"Worker {worker_id} reset failed: {result}")
    
    def step_all(self, actions: List[int]) -> List[Observation]:
        """
        Step all environments with given actions.
        
        Args:
            actions: List of action IDs (one per worker)
        
        Returns:
            List of observations, one per worker
        """
        if len(actions) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} actions, got {len(actions)}")
        
        # Send step tasks to all workers
        for task_q, action in zip(self.task_queues, actions):
            task_q.put(("step", action))
        
        # Collect results
        observations = [None] * self.num_workers
        received = 0
        while received < self.num_workers:
            try:
                worker_id, status, result = self.result_queue.get(timeout=self.timeout)
                if status == "step_ok":
                    observations[worker_id] = result
                    received += 1
                elif status == "timeout":
                    continue
                else:
                    raise RuntimeError(f"Worker {worker_id} step failed: {result}")
            except queue.Empty:
                raise TimeoutError(f"Worker step timeout after {self.timeout}s")
        
        return observations

    def annotate_detections_all(self, detections_batch: List[Any]) -> List[Any]:
        if len(detections_batch) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} detection lists, got {len(detections_batch)}")

        for task_q, dets in zip(self.task_queues, detections_batch):
            task_q.put(("annotate_detections", dets))

        results = [None] * self.num_workers
        received = 0
        while received < self.num_workers:
            try:
                worker_id, status, result = self.result_queue.get(timeout=self.timeout)
                if status == "annotate_ok":
                    results[worker_id] = result
                    received += 1
                elif status == "timeout":
                    continue
                else:
                    raise RuntimeError(f"Worker {worker_id} annotate failed: {result}")
            except queue.Empty:
                raise TimeoutError(f"Worker annotate timeout after {self.timeout}s")

        return results

    def finalize_one(self, worker_id: int, reason: str = "done") -> Any:
        """Ask one worker to persist final episode artifacts before reset."""
        if worker_id < 0 or worker_id >= self.num_workers:
            raise IndexError(f"worker_id out of range: {worker_id}")

        self.task_queues[worker_id].put(("finalize_episode", reason))
        deadline = time.time() + self.timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Worker {worker_id} finalize timeout after {self.timeout}s")

            try:
                got_worker_id, status, result = self.result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                continue

            if got_worker_id != worker_id:
                self.result_queue.put((got_worker_id, status, result))
                continue

            if status == "finalize_ok":
                return result
            if status == "timeout":
                continue
            raise RuntimeError(f"Worker {worker_id} finalize failed: {result}")

    def close(self):
        """Terminate all worker processes."""
        # Send close signal to all workers
        for task_q in self.task_queues:
            task_q.put(("close",))
        
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                p.kill()
        
        print("[ParallelHabitatCollector] All workers closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass
