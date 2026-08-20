import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from mjwarp_demo.protocol import PROTOCOL_VERSION
from mjwarp_demo.recorder import MAX_CONTACTS
from mjwarp_demo.server import DemoServer
from mjwarp_demo.validate_dataset import validate_file


def _state(frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "sim_time": frame_id * 0.05,
        "qpos": [0.0] * 9,
        "qvel": [0.0] * 8,
        "joint_position": [0.0] * 7,
        "joint_velocity": [0.0] * 7,
        "joint_effort": [0.0] * 7,
        "end_effector_position": [0.4 + frame_id * 0.01, 0.0, 0.3],
        "end_effector_quaternion": [1.0, 0.0, 0.0, 0.0],
        "gripper_width": 0.08,
        "body_position": [[0.0, 0.0, 0.0]] * 6,
        "body_quaternion": [[1.0, 0.0, 0.0, 0.0]] * 6,
        "body_external_wrench": [[0.0] * 6] * 6,
        "goal_position": [0.5, 0.0, 0.2],
        "task_stage": frame_id,
        "distance_to_goal": 0.1,
        "action": [0.1, -0.1],
        "action_command": [0.2, -0.2],
        "reward": 1.0 if frame_id else 0.0,
        "terminated": bool(frame_id),
        "success": bool(frame_id),
        "termination_reason": "success" if frame_id else "",
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


def _capture(frame_id: int, initial: bool, width: int, height: int) -> dict:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.ones((height, width), dtype="<f4")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "capture",
        "request_id": 10 + frame_id,
        "frame_id": frame_id,
        "initial": initial,
        "rgb_b64": base64.b64encode(rgba.tobytes()).decode(),
        "wrist_rgb_b64": base64.b64encode(rgba.tobytes()).decode(),
        "depth_b64": base64.b64encode(depth.tobytes()).decode(),
        "instance_b64": base64.b64encode(rgba.tobytes()).decode(),
    }


class _FakeTask:
    task_name = "panda_pick_place"
    policy = "expert"
    seed = 7
    nworld = 1
    gpu_name = "test-device"
    success = np.asarray([True])
    definition = SimpleNamespace(
        display_name="Panda Pick and Place",
        business_type="robot_manipulation",
        official_reference="test",
    )
    robot = SimpleNamespace(model_source="MuJoCo Menagerie test fixture")
    mj_model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))

    def __init__(self) -> None:
        self.current_state = _state(0)

    def state_dict(self) -> dict:
        return self.current_state

    def step(self, _action=None) -> dict:
        self.current_state = _state(1)
        return self.current_state

    def episode_metadata(self) -> dict:
        return {
            "robot": {"id": "franka_panda", "model_license": "Apache-2.0"},
            "action_spec": {"type": "joint_position_target", "names": ["j1", "gripper"]},
            "randomization": {"object_mass_scale": 1.0},
            "coordinate_frames": {"world": "MuJoCo"},
        }


def test_server_records_schema_v2_transition_end_to_end(tmp_path: Path) -> None:
    width, height = 4, 3
    server = DemoServer(None, tmp_path / "Datasets", "cuda:0")
    task = _FakeTask()
    server.task = task
    server.ensure_task = lambda *_args, **_kwargs: task  # type: ignore[method-assign]

    async def run_episode() -> dict:
        await server.handle(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "record_start",
                "request_id": 1,
                "scenario": task.task_name,
                "episode_id": "server_schema_v2",
                "seed": task.seed,
                "policy": task.policy,
                "image_width": width,
                "image_height": height,
                "camera_metadata": {"front": {"width": width, "height": height}},
            }
        )
        await server.handle(_capture(0, True, width, height))
        await server.handle(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "step",
                "request_id": 2,
                "scenario": task.task_name,
                "action": [0.1, -0.1],
            }
        )
        await server.handle(_capture(1, False, width, height))
        return await server.handle(
            {
                "protocol_version": PROTOCOL_VERSION,
                "type": "record_stop",
                "request_id": 3,
            }
        )

    stopped = asyncio.run(run_episode())
    path = Path(stopped["path"])
    assert stopped["frame_count"] == 1
    assert validate_file(path) == []
    with h5py.File(path, "r") as episode:
        assert episode.attrs["schema_version"] == "2.0"
        assert episode.attrs["frame_count"] == 2
        assert episode.attrs["transition_count"] == 1
        assert episode["transition_observation_index"][...].tolist() == [0]
        assert episode["transition_next_observation_index"][...].tolist() == [1]
        assert episode["actions/normalized"][...].tolist() == [[0.10000000149011612, -0.10000000149011612]]
        assert episode["observations/end_effector_position"].shape == (2, 3)
        assert episode["images/wrist_rgb"].shape == (2, height, width, 3)
