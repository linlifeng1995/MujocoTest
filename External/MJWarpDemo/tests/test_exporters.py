import base64
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from mjwarp_demo.exporters import (
    export_lerobot_v3,
    export_lerobot_v3_dataset,
    export_robomimic,
    export_robomimic_dataset,
    write_checksums,
)
from mjwarp_demo.recorder import EpisodeRecorder, MAX_CONTACTS


def _state(frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "sim_time": frame_id * 0.05,
        "qpos": [0.0, 0.0],
        "qvel": [0.0, 0.0],
        "joint_position": [0.0, 0.0],
        "joint_velocity": [0.0, 0.0],
        "joint_effort": [0.0, 0.0],
        "end_effector_position": [0.1 + frame_id * 0.01, 0.0, 0.2],
        "end_effector_quaternion": [1.0, 0.0, 0.0, 0.0],
        "gripper_width": 0.04,
        "body_position": [[0.0, 0.0, 0.0]],
        "body_quaternion": [[1.0, 0.0, 0.0, 0.0]],
        "body_external_wrench": [[0.0] * 6],
        "goal_position": [0.2, 0.0, 0.0],
        "task_stage": frame_id,
        "distance_to_goal": 0.1,
        "action": [0.1, -0.1],
        "action_command": [0.2, -0.2],
        "reward": 1.0,
        "terminated": frame_id == 1,
        "success": frame_id == 1,
        "termination_reason": "success" if frame_id == 1 else "",
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


def _capture(frame_id: int, initial: bool) -> dict:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    depth = np.ones((4, 4), dtype="<f4")
    return {
        "frame_id": frame_id,
        "initial": initial,
        "rgb_b64": base64.b64encode(rgba.tobytes()).decode(),
        "wrist_rgb_b64": base64.b64encode(rgba.tobytes()).decode(),
        "depth_b64": base64.b64encode(depth.tobytes()).decode(),
        "instance_b64": base64.b64encode(rgba.tobytes()).decode(),
    }


def test_vendor_exporters_are_readable(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(
        tmp_path,
        "source",
        {
            "task_name": "panda_pick_place",
            "seed": 1,
            "policy": "expert",
            "physics_dt": 0.002,
            "control_dt": 0.05,
            "protocol_version": 3,
            "robot": {"id": "franka_panda"},
            "action_spec": {"type": "joint_position_target"},
            "randomization": {"randomization_group": "panda-01"},
            "camera_metadata": {"front": {"width": 4, "height": 4}},
            "data_source": "synthetic_simulation",
            "generation_strategy": "expert",
            "license_manifest": "model/third_party/LICENSES.md",
        },
        4,
        4,
    )
    recorder.append_initial(_state(0), _capture(0, True))
    recorder.append_transition(_state(1), _capture(1, False))
    native = recorder.close(True)

    lerobot = export_lerobot_v3(native, tmp_path / "lerobot")
    table = pq.read_table(lerobot / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == 1
    assert (lerobot / "videos/observation.images.front/chunk-000/episode_000000.mp4").exists()

    robomimic = export_robomimic(native, tmp_path / "robomimic.hdf5")
    with h5py.File(robomimic, "r") as dataset:
        assert dataset["data/demo_0/actions"].shape == (1, 2)
        assert dataset["data/demo_0/obs/front_image"].shape == (1, 4, 4, 3)
    checksum = write_checksums(tmp_path)
    assert "robomimic.hdf5" in checksum.read_text(encoding="utf-8")


def test_dataset_exporters_preserve_multiple_episodes_and_splits(tmp_path: Path) -> None:
    paths: list[Path] = []
    for seed in (1, 2):
        recorder = EpisodeRecorder(
            tmp_path,
            f"source_{seed}",
            {
                "task_name": "panda_pick_place",
                "seed": seed,
                "policy": "expert" if seed == 1 else "perturbed",
                "physics_dt": 0.002,
                "control_dt": 0.05,
                "protocol_version": 3,
                "robot": {"id": "franka_panda"},
                "action_spec": {"type": "joint_position_target"},
                "randomization": {"randomization_group": f"panda-{seed:02d}"},
                "camera_metadata": {"front": {"width": 4, "height": 4}},
                "data_source": "synthetic_simulation",
                "generation_strategy": "expert" if seed == 1 else "perturbed",
                "license_manifest": "model/third_party/LICENSES.md",
            },
            4,
            4,
        )
        recorder.append_initial(_state(0), _capture(0, True))
        recorder.append_transition(_state(1), _capture(1, False))
        paths.append(recorder.close(True))

    splits = {paths[0].resolve(): "train", paths[1].resolve(): "validation"}
    lerobot = export_lerobot_v3_dataset(paths, tmp_path / "lerobot_dataset", split_by_path=splits)
    info = json.loads((lerobot / "meta/info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 2
    assert info["robot_type"] == "franka_panda"
    assert info["splits"] == {"train": "0:1", "validation": "1:2"}
    assert (lerobot / "data/chunk-000/episode_000001.parquet").exists()

    robomimic = export_robomimic_dataset(
        paths, tmp_path / "robomimic_dataset.hdf5", split_by_path=splits
    )
    with h5py.File(robomimic, "r") as dataset:
        assert dataset["data/demo_0/actions"].shape == (1, 2)
        assert dataset["data/demo_1/actions"].shape == (1, 2)
        assert dataset["mask/train"].asstr()[...].tolist() == ["demo_0"]
        assert dataset["mask/valid"].asstr()[...].tolist() == ["demo_1"]
