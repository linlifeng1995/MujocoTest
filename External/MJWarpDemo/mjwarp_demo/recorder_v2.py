from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
MAX_CONTACTS = 16
SCHEMA_VERSION = "2.0"


def _attribute_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


class EpisodeRecorder:
    """Schema 2.0 recorder with explicit N transitions and N+1 observations."""

    def __init__(
        self,
        output_dir: Path,
        episode_id: str,
        metadata: dict[str, Any],
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = episode_id
        self.image_width = image_width
        self.image_height = image_height
        self.partial_path = output_dir / f"{episode_id}.partial"
        self.final_path = output_dir / f"{episode_id}.h5"
        self.file = h5py.File(self.partial_path, "w")
        self.observation_count = 0
        self.transition_count = 0
        self.expected_frame_id: int | None = None
        self.closed = False
        self._datasets: dict[str, h5py.Dataset] = {}
        self._last_state: dict[str, Any] | None = None

        attrs = {
            "schema_version": SCHEMA_VERSION,
            "task_name": metadata.get("task_name", "planar_push"),
            "episode_id": episode_id,
            "image_width": image_width,
            "image_height": image_height,
            "coordinate_system_physics": "MuJoCo right-handed Z-up",
            "coordinate_system_render": "Unity left-handed Y-up",
            "transition_semantics": (
                "transition t references observation index t, action t, reward t, "
                "and next observation index t+1"
            ),
            "physics_units": "SI: metre, kilogram, second, radian, newton",
            **metadata,
        }
        for key, value in attrs.items():
            self.file.attrs[key] = _attribute_value(value)

    @property
    def length(self) -> int:
        return self.transition_count

    def _dataset(self, name: str, sample: np.ndarray, *, compression: bool = False) -> h5py.Dataset:
        existing = self._datasets.get(name)
        if existing is not None:
            return existing
        sample = np.asarray(sample)
        kwargs: dict[str, Any] = {
            "shape": (0, *sample.shape),
            "maxshape": (None, *sample.shape),
            "dtype": sample.dtype,
            "chunks": (1, *sample.shape) if sample.shape else (256,),
        }
        if compression and sample.size > 32:
            kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
        dataset = self.file.create_dataset(name, **kwargs)
        self._datasets[name] = dataset
        return dataset

    def _append_at(
        self,
        name: str,
        index: int,
        sample: Any,
        *,
        dtype: Any | None = None,
        compression: bool = False,
    ) -> None:
        array = np.asarray(sample, dtype=dtype)
        dataset = self._dataset(name, array, compression=compression)
        if dataset.shape[0] != index:
            raise RuntimeError(f"dataset {name} expected index {dataset.shape[0]}, got {index}")
        dataset.resize(index + 1, axis=0)
        dataset[index] = array

    def _append_text(self, name: str, index: int, value: str) -> None:
        dataset = self._datasets.get(name)
        if dataset is None:
            dataset = self.file.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                chunks=(256,),
            )
            self._datasets[name] = dataset
        if dataset.shape[0] != index:
            raise RuntimeError(f"dataset {name} expected index {dataset.shape[0]}, got {index}")
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def append_initial(self, state: dict[str, Any], capture: dict[str, Any]) -> None:
        if self.observation_count != 0:
            raise RuntimeError("initial observation has already been recorded")
        self._validate_capture_alignment(state, capture, initial=True)
        self._append_observation(state, capture)
        self._last_state = state
        self.expected_frame_id = int(state["frame_id"]) + 1
        self.file.flush()

    def append_transition(self, state: dict[str, Any], capture: dict[str, Any]) -> None:
        if self._last_state is None:
            raise RuntimeError("initial observation must be recorded before transitions")
        self._validate_capture_alignment(state, capture, initial=False)
        index = self.transition_count
        self._append_at("transition_observation_index", index, self.observation_count - 1, dtype=np.int64)
        self._append_at("transition_next_observation_index", index, self.observation_count, dtype=np.int64)
        self._append_at("transition_timestamps", index, self._last_state["sim_time"], dtype=np.float64)
        self._append_at("actions/normalized", index, state["action"], dtype=np.float32)
        self._append_at("actions/command", index, state.get("action_command", state["action"]), dtype=np.float32)
        self._append_at("rewards", index, state["reward"], dtype=np.float32)
        self._append_at("terminated", index, state["terminated"], dtype=np.bool_)
        self._append_at("success", index, state["success"], dtype=np.bool_)
        self._append_text("termination_reason", index, str(state.get("termination_reason", "")))
        self._append_at(
            "derived_actions/delta_end_effector_pose",
            index,
            self._delta_end_effector_pose(self._last_state, state),
            dtype=np.float32,
        )
        self._append_at(
            "derived_actions/gripper_width", index, state.get("gripper_width", 0.0), dtype=np.float32
        )
        self._append_observation(state, capture)
        self.transition_count += 1
        self._last_state = state
        self.expected_frame_id = int(state["frame_id"]) + 1
        self.file.flush()

    def _validate_capture_alignment(
        self, state: dict[str, Any], capture: dict[str, Any], *, initial: bool
    ) -> None:
        frame_id = int(capture["frame_id"])
        state_frame_id = int(state["frame_id"])
        if frame_id != state_frame_id:
            raise ValueError(f"capture/state frame mismatch: {frame_id} != {state_frame_id}")
        if self.expected_frame_id is not None and frame_id != self.expected_frame_id:
            raise ValueError(f"non-contiguous frame id: expected {self.expected_frame_id}, got {frame_id}")
        if bool(capture.get("initial", False)) != initial:
            raise ValueError(f"capture initial flag mismatch: expected initial={initial}")

    def _append_observation(self, state: dict[str, Any], capture: dict[str, Any]) -> None:
        index = self.observation_count
        rgb = self._decode_rgba(capture["rgb_b64"])[..., :3]
        wrist_rgb = self._decode_rgba(capture.get("wrist_rgb_b64", capture["rgb_b64"]))[..., :3]
        depth = self._decode_depth(capture["depth_b64"])
        instance = self._decode_instance(capture["instance_b64"])
        fields: tuple[tuple[str, Any, Any], ...] = (
            ("timestamps", state["sim_time"], np.float64),
            ("frame_id", state["frame_id"], np.int64),
            ("observations/qpos", state["qpos"], np.float32),
            ("observations/qvel", state["qvel"], np.float32),
            ("observations/joint_position", state.get("joint_position", state["qpos"]), np.float32),
            ("observations/joint_velocity", state.get("joint_velocity", state["qvel"]), np.float32),
            ("observations/joint_effort", state.get("joint_effort", np.zeros(len(state["qvel"]))), np.float32),
            ("observations/end_effector_position", state.get("end_effector_position", [0.0, 0.0, 0.0]), np.float32),
            ("observations/end_effector_quaternion", state.get("end_effector_quaternion", [1.0, 0.0, 0.0, 0.0]), np.float32),
            ("observations/gripper_width", state.get("gripper_width", 0.0), np.float32),
            ("observations/body_position", state["body_position"], np.float32),
            ("observations/body_quaternion", state["body_quaternion"], np.float32),
            ("observations/body_external_wrench", state["body_external_wrench"], np.float32),
            ("observations/goal_position", state["goal_position"], np.float32),
            ("observations/task_stage", state.get("task_stage", 0), np.int16),
            ("observations/distance_to_goal", state.get("distance_to_goal", 0.0), np.float32),
        )
        for name, value, dtype in fields:
            self._append_at(name, index, value, dtype=dtype)

        contacts = state["contacts"]
        for name, value, dtype in (
            ("contacts/count", contacts["count"], np.int16),
            ("contacts/valid", contacts["valid"], np.bool_),
            ("contacts/geom_pair", contacts["geom_pair"], np.int32),
            ("contacts/position", contacts["position"], np.float32),
            ("contacts/normal", contacts["normal"], np.float32),
            ("contacts/distance", contacts["distance"], np.float32),
            ("contacts/overflow", contacts["overflow"], np.bool_),
        ):
            self._append_at(name, index, value, dtype=dtype)

        self._append_at("images/front_rgb", index, rgb, compression=True)
        self._append_at("images/front_depth_m", index, depth.astype(np.float16), compression=True)
        self._append_at("images/front_instance_id", index, instance, compression=True)
        self._append_at("images/wrist_rgb", index, wrist_rgb, compression=True)
        self.observation_count += 1

    @staticmethod
    def _delta_end_effector_pose(previous: dict[str, Any], current: dict[str, Any]) -> np.ndarray:
        p0 = np.asarray(previous.get("end_effector_position", [0.0, 0.0, 0.0]), dtype=np.float64)
        p1 = np.asarray(current.get("end_effector_position", [0.0, 0.0, 0.0]), dtype=np.float64)
        q0 = np.asarray(previous.get("end_effector_quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        q1 = np.asarray(current.get("end_effector_quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        w0, x0, y0, z0 = q0[0], -q0[1], -q0[2], -q0[3]
        w1, x1, y1, z1 = q1
        dq = np.asarray(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=np.float64,
        )
        dq /= max(np.linalg.norm(dq), 1e-9)
        return np.concatenate((p1 - p0, dq)).astype(np.float32)

    def _decode_rgba(self, encoded: str) -> np.ndarray:
        raw = base64.b64decode(encoded, validate=True)
        expected = self.image_width * self.image_height * 4
        if len(raw) != expected:
            raise ValueError(f"RGBA byte count mismatch: expected {expected}, got {len(raw)}")
        return np.flipud(np.frombuffer(raw, dtype=np.uint8).reshape(self.image_height, self.image_width, 4)).copy()

    def _decode_depth(self, encoded: str) -> np.ndarray:
        raw = base64.b64decode(encoded, validate=True)
        expected = self.image_width * self.image_height * 4
        if len(raw) != expected:
            raise ValueError(f"depth byte count mismatch: expected {expected}, got {len(raw)}")
        return np.flipud(np.frombuffer(raw, dtype="<f4").reshape(self.image_height, self.image_width)).copy()

    def _decode_instance(self, encoded: str) -> np.ndarray:
        rgba = self._decode_rgba(encoded)
        return (rgba[..., 0].astype(np.uint16) | (rgba[..., 1].astype(np.uint16) << 8)).astype(np.uint16)

    def close(self, success: bool) -> Path:
        if self.closed:
            return self.final_path
        if self.observation_count != self.transition_count + 1:
            raise RuntimeError(
                f"invalid transition alignment: observations={self.observation_count}, transitions={self.transition_count}"
            )
        self.file.attrs["success_final"] = bool(success)
        self.file.attrs["frame_count"] = self.observation_count
        self.file.attrs["transition_count"] = self.transition_count
        self.file.attrs["termination_reason_final"] = str(
            self._last_state.get("termination_reason", "") if self._last_state else ""
        )
        self.file.flush()
        self.file.close()
        self.closed = True
        os.replace(self.partial_path, self.final_path)
        return self.final_path

    def abort(self, reason: str) -> None:
        if self.closed:
            return
        self.file.attrs["abort_reason"] = reason
        self.file.attrs["frame_count"] = self.observation_count
        self.file.attrs["transition_count"] = self.transition_count
        self.file.flush()
        self.file.close()
        self.closed = True
