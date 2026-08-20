import base64
from pathlib import Path

import h5py
import numpy as np

from mjwarp_demo.recorder import EpisodeRecorder, MAX_CONTACTS
from mjwarp_demo.validate_dataset import validate_file


def _state(frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "sim_time": frame_id * 0.05,
        "qpos": [0.0] * 9,
        "qvel": [0.0] * 8,
        "body_position": [[0.0, 0.0, 0.0]] * 6,
        "body_quaternion": [[1.0, 0.0, 0.0, 0.0]] * 6,
        "body_external_wrench": [[0.0] * 6] * 6,
        "goal_position": [0.2, 0.0, 0.0],
        "task_stage": 1,
        "distance_to_goal": 0.12,
        "action": [0.1, -0.1],
        "reward": 0.25,
        "terminated": False,
        "success": False,
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


def test_recorder_writes_valid_aligned_episode(tmp_path: Path) -> None:
    width, height = 4, 3
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.ones((height, width), dtype="<f4")
    initial_capture = {
        "frame_id": 0,
        "initial": True,
        "rgb_b64": base64.b64encode(rgba.tobytes()).decode(),
        "depth_b64": base64.b64encode(depth.tobytes()).decode(),
        "instance_b64": base64.b64encode(rgba.tobytes()).decode(),
    }
    capture = {**initial_capture, "frame_id": 1, "initial": False}
    recorder = EpisodeRecorder(
        tmp_path,
        "episode",
        {
            "task_name": "precision_insert",
            "seed": 1,
            "policy": "expert",
            "physics_dt": 0.002,
            "control_dt": 0.05,
            "protocol_version": 3,
            "robot": {"id": "test_robot"},
            "action_spec": {"type": "test"},
            "randomization": {"randomization_group": "test-01"},
            "camera_metadata": {"front": {"width": width, "height": height}},
            "data_source": "synthetic_simulation",
            "generation_strategy": "expert",
            "license_manifest": "test-license",
        },
        width,
        height,
    )
    recorder.append_initial(_state(0), initial_capture)
    recorder.append_transition(_state(1), capture)
    path = recorder.close(False)
    assert validate_file(path) == []
    with h5py.File(path, "r") as episode:
        assert episode["images/front_rgb"].shape == (2, height, width, 3)
        assert episode["images/wrist_rgb"].shape == (2, height, width, 3)
        assert episode["observations/goal_position"].shape == (2, 3)
        assert episode["actions/normalized"].shape == (1, 2)
        assert episode.attrs["schema_version"] == "2.0"
        assert episode.attrs["task_name"] == "precision_insert"
        assert episode.attrs["frame_count"] == 2
        assert episode.attrs["transition_count"] == 1
