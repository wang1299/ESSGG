import numpy as np

from components.environments.habitat_env import HabitatEnv
from RL_training.runner.parallel_habitat_rl_train_runner import ParallelHabitatRLTrainRunner


def _make_env():
    env = object.__new__(HabitatEnv)
    env.semantic_id_to_label = {1: "CounterTop", 2: "Sink", 3: "PaperTowelRoll"}
    env.gt_validation_iou_threshold = 0.10
    env.gt_validation_mode = "relaxed"
    env._last_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    env._last_semantic = np.zeros((20, 20), dtype=np.int32)
    env._last_semantic[1:10, 1:10] = 1
    env._last_semantic[10:19, 1:10] = 2
    env._last_semantic[5:15, 12:19] = 3
    env.current_ep_dir = None
    env.step_count = 0
    env.save_debug_interval = 100
    return env


def test_compound_dino_label_is_rejected():
    env = _make_env()
    dets = [{"label": "CounterTopPlate", "score": 0.9, "bbox": [1, 1, 10, 10]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is False
    assert out[0]["reject_reason"] in {"ambiguous_label", "compound_label"}


def test_relaxed_alias_can_match_semantic_gt():
    env = _make_env()
    dets = [{"label": "Sink Basin", "score": 0.9, "bbox": [1, 10, 10, 19]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is True
    assert out[0]["canonical_label"] == "SinkBasin"
    assert out[0]["gt_semantic_id"] == 2


def test_low_iou_is_rejected():
    env = _make_env()
    dets = [{"label": "Sink Basin", "score": 0.9, "bbox": [0, 0, 2, 2]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is False
    assert out[0]["reject_reason"] == "low_iou"


def test_same_semantic_id_only_validates_once():
    env = _make_env()
    dets = [
        {"label": "Counter Top", "score": 0.9, "bbox": [1, 1, 10, 10]},
        {"label": "Counter Top", "score": 0.8, "bbox": [1, 1, 10, 10]},
    ]

    out = env.validate_detections(dets)

    assert sum(1 for det in out if det["is_gt_valid"]) == 1


def test_relaxed_iou_only_match_for_unknown_gt_vocab():
    env = _make_env()
    env.semantic_id_to_label[1] = "hm3d_custom_surface"
    dets = [{"label": "Counter Top", "score": 0.9, "bbox": [1, 1, 10, 10]}]

    out = env.validate_detections(dets)

    assert out[0]["is_gt_valid"] is True
    assert out[0]["gt_match_mode"] == "semantic_iou_only"
    assert out[0]["gt_semantic_id"] == 1


def test_scene_reward_gt_ids_exclude_structural_labels():
    env = object.__new__(HabitatEnv)
    env.semantic_id_to_label = {1: "Wall", 2: "Floor", 3: "Window", 4: "CounterTop", 5: "Sink"}
    env.reward_excluded_labels = {"Wall", "Floor", "Window"}

    reward_ids = env._build_scene_reward_gt_ids()

    assert reward_ids == {4, 5}


def test_runner_reward_filters_excluded_and_iou_only_detections():
    runner = object.__new__(ParallelHabitatRLTrainRunner)
    runner.det_score_thr = 0.20
    runner.reward_excluded_labels = {"Wall", "Floor", "Window"}
    runner.reward_allow_semantic_iou_only = False
    runner.instance_merge_dist = 0.8

    detections = [
        {"is_gt_valid": True, "score": 0.9, "canonical_label": "Wall", "gt_semantic_id": 1},
        {"is_gt_valid": True, "score": 0.9, "canonical_label": "CounterTop", "gt_semantic_id": 2, "gt_match_mode": "semantic_iou_only"},
        {"is_gt_valid": True, "score": 0.9, "canonical_label": "Sink", "gt_semantic_id": 3},
    ]

    objects = runner._detections_to_metadata_objects(detections, target_gt_ids={2, 3})

    assert len(objects) == 1
    assert objects[0]["gt_semantic_id"] == 3
