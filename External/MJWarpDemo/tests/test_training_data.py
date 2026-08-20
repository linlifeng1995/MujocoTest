import base64
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mjwarp_demo.recorder import EpisodeRecorder, MAX_CONTACTS
from mjwarp_demo.quality_report import build_quality_report
from mjwarp_demo.training.data import (
    assign_group_splits,
    assign_record_splits,
    assign_seed_splits,
    build_manifest,
    load_bc_arrays,
    load_manifest,
)


def _state(frame_id: int, action: list[float]) -> dict:
    return {
        "frame_id": frame_id,
        "sim_time": frame_id * 0.05,
        "qpos": [float(frame_id), 0.0],
        "qvel": [0.1, -0.1],
        "body_position": [[0.0, 0.0, 0.1]],
        "body_quaternion": [[1.0, 0.0, 0.0, 0.0]],
        "body_external_wrench": [[0.0] * 6],
        "goal_position": [0.2, 0.0, 0.0],
        "task_stage": frame_id % 2,
        "distance_to_goal": 0.2 - frame_id * 0.01,
        "action": action,
        "reward": 0.1,
        "terminated": frame_id == 3,
        "success": frame_id == 3,
        "contacts": {
            "count": 0,
            "valid": [False] * MAX_CONTACTS,
            "geom_pair": [[-1, -1]] * MAX_CONTACTS,
            "position": [[0.0, 0.0, 0.0]] * MAX_CONTACTS,
            "normal": [[0.0, 0.0, 0.0]] * MAX_CONTACTS,
            "distance": [0.0] * MAX_CONTACTS,
            "overflow": False,
        },
    }


def _episode(root: Path, seed: int, policy: str) -> Path:
    width, height = 2, 2
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.ones((height, width), dtype="<f4")
    recorder = EpisodeRecorder(
        root,
        f"{policy}_{seed}",
        {
            "task_name": "planar_push",
            "seed": seed,
            "policy": policy,
            "physics_dt": 0.005,
            "control_dt": 0.05,
            "protocol_version": 3,
            "robot": {"id": "planar_arm"},
            "action_spec": {"type": "joint_velocity"},
            "randomization": {"randomization_group": f"legacy-{seed:02d}"},
            "camera_metadata": {"front": {"width": width, "height": height}},
            "data_source": "synthetic_simulation",
            "generation_strategy": policy,
            "license_manifest": "test-license",
        },
        width,
        height,
    )
    for frame_id in range(0, 4):
        capture = {
            "frame_id": frame_id,
            "initial": frame_id == 0,
            "rgb_b64": base64.b64encode(rgba.tobytes()).decode(),
            "depth_b64": base64.b64encode(depth.tobytes()).decode(),
            "instance_b64": base64.b64encode(rgba.tobytes()).decode(),
        }
        state = _state(frame_id, [frame_id / 10.0, -frame_id / 10.0])
        if frame_id == 0:
            recorder.append_initial(state, capture)
        else:
            recorder.append_transition(state, capture)
    return recorder.close(True)


def test_seed_split_is_reproducible_and_shared_between_policies() -> None:
    first = assign_seed_splits(range(20), split_seed=7)
    second = assign_seed_splits(reversed(range(20)), split_seed=7)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}


def test_group_split_is_reproducible_and_prevents_configuration_leakage() -> None:
    groups = [f"panda_pick_place|object-{index % 4}|scene-{index % 5}|group-{index:02d}" for index in range(20)]
    first = assign_group_splits(groups, split_seed=7)
    second = assign_group_splits(reversed(groups), split_seed=7)
    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}


def test_record_split_connects_shared_seed_object_scene_and_randomization() -> None:
    records = [
        {
            "seed": index,
            "object_config_id": f"object-{index}",
            "scene_config_id": f"scene-{index}",
            "randomization_group": f"group-{index}",
            "split_group": f"record-{index}",
        }
        for index in range(8)
    ]
    records[1]["object_config_id"] = records[0]["object_config_id"]
    records[2]["scene_config_id"] = records[1]["scene_config_id"]
    records[3]["randomization_group"] = records[2]["randomization_group"]
    assign_record_splits(records, split_seed=3)
    assert len({records[index]["split"] for index in range(4)}) == 1


def test_manifest_and_behavior_cloning_alignment(tmp_path: Path) -> None:
    for seed in range(10):
        _episode(tmp_path, seed, "expert")
        _episode(tmp_path, seed, "random")
    manifest_path = tmp_path / "manifest.json"
    manifest = build_manifest(tmp_path, manifest_path, scenario="planar_push", split_seed=4)
    loaded = load_manifest(manifest_path)
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    split_by_seed: dict[int, set[str]] = {}
    split_by_group: dict[str, set[str]] = {}
    for record in loaded["records"]:
        split_by_seed.setdefault(record["seed"], set()).add(record["split"])
        split_by_group.setdefault(record["split_group"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in split_by_seed.values())
    assert all(len(splits) == 1 for splits in split_by_group.values())
    x, y, stages, trajectories = load_bc_arrays(loaded, "train", policy="expert")
    assert x.shape[0] == y.shape[0] == stages.shape[0]
    assert sum(count for _, count in trajectories) == len(x)
    # Schema 2.0 row 0 observation is paired directly with transition action 0.
    assert np.allclose(y[0], [0.1, -0.1])
    report = build_quality_report(tmp_path, tmp_path / "quality_report.json", manifest_path)
    assert report["split_leakage_audit"]["passed"] is True


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    _episode(tmp_path, 1, "expert")
    manifest_path = tmp_path / "manifest.json"
    build_manifest(tmp_path, manifest_path, scenario="planar_push")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["split_seed"] = 999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_manifest(manifest_path)


def test_policy_runtime_loads_and_clamps_actions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from mjwarp_demo.learned_policy import LearnedPolicyRuntime
    from mjwarp_demo.training.models import BehaviorCloningPolicy

    artifact = tmp_path / "planar_push" / "bc_test"
    artifact.mkdir(parents=True)
    model = BehaviorCloningPolicy(8, 2)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save(model.state_dict(), artifact / "model.pt")
    np.savez(artifact / "normalization.npz", input_mean=np.zeros(8), input_std=np.ones(8))
    (artifact / "model_spec.json").write_text(
        json.dumps(
            {
                "artifact_id": "planar_push/bc_test",
                "model_type": "behavior_cloning",
                "scenario": "planar_push",
                "input_dim": 8,
                "qpos_dim": 2,
                "qvel_dim": 2,
                "action_dim": 2,
                "hidden_layers": [256, 256, 256],
            }
        ),
        encoding="utf-8",
    )
    runtime = LearnedPolicyRuntime(artifact, max_latency_ms=10000)
    state = {"qpos": [0.0, 0.0], "qvel": [0.0, 0.0], "goal_position": [0.2, 0.0, 0.0], "task_stage": 0}
    runtime.validate_state(state, "planar_push")
    result = runtime.act(state)
    assert result.blocked is False
    assert result.action == pytest.approx([0.0, 0.0])
