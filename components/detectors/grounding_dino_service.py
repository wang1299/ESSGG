import multiprocessing as mp
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

from components.detectors.grounding_dino_adapter import GroundingDINODetector


def _dino_service_loop(task_queue: mp.Queue, result_queue: mp.Queue, detector_kwargs: Dict[str, Any]):
    detector = None
    try:
        device = detector_kwargs.get("device")
        if isinstance(device, str) and device.startswith("cuda:") and torch.cuda.is_available():
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        detector = GroundingDINODetector(**detector_kwargs)
        result_queue.put(("ready", None, None))

        while True:
            try:
                task = task_queue.get(timeout=30)
            except queue.Empty:
                continue

            if task is None:
                break

            task_type = task[0]
            if task_type == "close":
                break

            if task_type != "detect_batch":
                result_queue.put((task[1], None, f"Unknown task type: {task_type}"))
                continue

            request_id = task[1]
            rgb_batch = task[2]
            detections_batch = []
            try:
                for rgb_image in rgb_batch:
                    detections_batch.append(detector.detect(rgb_image))
                result_queue.put((request_id, detections_batch, None))
            except Exception as exc:
                result_queue.put((request_id, None, str(exc)))

    except Exception as exc:
        try:
            result_queue.put(("init_error", None, str(exc)))
        except Exception:
            pass
    finally:
        if detector is not None:
            try:
                del detector
            except Exception:
                pass


class GroundingDINOService:
    """Single-process GroundingDINO service with a batch request queue."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.mp_ctx = mp.get_context("spawn")
        self.task_queue = self.mp_ctx.Queue()
        self.result_queue = self.mp_ctx.Queue()
        self.request_id = 0
        self.pending_results: Dict[int, Any] = {}

        self.process = self.mp_ctx.Process(
            target=_dino_service_loop,
            args=(
                self.task_queue,
                self.result_queue,
                {
                    "config_path": config_path,
                    "checkpoint_path": checkpoint_path,
                    "text_prompt": text_prompt,
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "device": device,
                },
            ),
            daemon=False,
        )
        self.process.start()

        deadline = time.time() + 300.0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("GroundingDINOService init timeout")
            try:
                status, _, message = self.result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                continue
            if status == "ready":
                break
            if status == "init_error":
                raise RuntimeError(f"GroundingDINOService init failed: {message}")

    def detect_batch(self, rgb_batch: List[Any], timeout: float = 120.0):
        request_id = self.request_id
        self.request_id += 1
        self.task_queue.put(("detect_batch", request_id, rgb_batch))

        if request_id in self.pending_results:
            return self.pending_results.pop(request_id)

        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"GroundingDINOService request {request_id} timeout after {timeout}s")

            try:
                got_request_id, detections_batch, error = self.result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                continue

            if got_request_id == request_id:
                if error is not None:
                    raise RuntimeError(error)
                return detections_batch

            if isinstance(got_request_id, int):
                self.pending_results[got_request_id] = detections_batch

    def close(self):
        try:
            self.task_queue.put(("close",))
        except Exception:
            pass

        if self.process.is_alive():
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()


class GroundingDINOServicePool:
    """Round-robin batch dispatcher across multiple independent DINO services."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompt: str,
        devices: List[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        if not devices:
            raise ValueError("GroundingDINOServicePool requires at least one device")

        self.services = [
            GroundingDINOService(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device,
            )
            for device in devices
        ]
        self.executor = ThreadPoolExecutor(max_workers=len(self.services))

    def _split_indices(self, batch_size: int):
        service_count = len(self.services)
        if batch_size <= 0:
            return []

        base = batch_size // service_count
        remainder = batch_size % service_count
        ranges = []
        start = 0
        for service_idx in range(service_count):
            chunk_size = base + (1 if service_idx < remainder else 0)
            end = start + chunk_size
            if start < end:
                ranges.append((service_idx, start, end))
            start = end
        return ranges

    def detect_batch(self, rgb_batch: List[Any], timeout: float = 120.0):
        if not rgb_batch:
            return []

        ranges = self._split_indices(len(rgb_batch))
        futures = []
        for service_idx, start, end in ranges:
            futures.append(
                self.executor.submit(
                    self.services[service_idx].detect_batch,
                    rgb_batch[start:end],
                    timeout,
                )
            )

        detections_batch = [None] * len(rgb_batch)
        for (service_idx, start, end), future in zip(ranges, futures):
            chunk_results = future.result()
            for offset, dets in enumerate(chunk_results):
                detections_batch[start + offset] = dets

        return detections_batch

    def close(self):
        try:
            self.executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

        for service in self.services:
            try:
                service.close()
            except Exception:
                pass
