from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
MAX_CONTACTS = 16
SCHEMA_VERSION = "1.0"


class EpisodeRecorder:
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
        self.length = 0
        self.expected_frame_id: int | None = None
        self.closed = False
        self._datasets: dict[str, h5py.Dataset] = {}

        attrs = {
            "schema_version": SCHEMA_VERSION,
            "task_name": "planar_push",
            "episode_id": episode_id,
            "image_width": image_width,
            "image_height": image_height,
            "coordinate_system_physics": "MuJoCo right-handed Z-up",
            "coordinate_system_render": "Unity left-handed Y-up",
            **metadata,
        }
        for key, value in attrs.items():
            self.file.attrs[key] = value

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

    def _append(self, name: str, sample: Any, *, dtype: Any | None = None, compression: bool = False) -> None:
        array = np.asarray(sample, dtype=dtype)
        dataset = self._dataset(name, array, compression=compression)
        dataset.resize(self.length + 1, axis=0)
        dataset[self.length] = array

    def append_capture(self, state: dict[str, Any], capture: dict[str, Any]) -> None:
        frame_id = int(capture["frame_id"])
        state_frame_id = int(state["frame_id"])
        if frame_id != state_frame_id:
            raise ValueError(f"capture/state frame mismatch: {frame_id} != {state_frame_id}")
        if self.expected_frame_id is not None and frame_id != self.expected_frame_id:
            raise ValueError(f"non-contiguous frame id: expected {self.expected_frame_id}, got {frame_id}")

        rgb = self._decode_rgba(capture["rgb_b64"])[..., :3]
        depth = self._decode_depth(capture["depth_b64"])
        instance = self._decode_instance(capture["instance_b64"])

        self._append("timestamps", state["sim_time"], dtype=np.float64)
        self._append("observations/qpos", state["qpos"], dtype=np.float32)
        self._append("observations/qvel", state["qvel"], dtype=np.float32)
        self._append("observations/body_position", state["body_position"], dtype=np.float32)
        self._append("observations/body_quaternion", state["body_quaternion"], dtype=np.float32)
        self._append("observations/body_external_wrench", state["body_external_wrench"], dtype=np.float32)
        self._append("actions", state["action"], dtype=np.float32)
        self._append("rewards", state["reward"], dtype=np.float32)
        self._append("terminated", state["terminated"], dtype=np.bool_)
        self._append("success", state["success"], dtype=np.bool_)

        contacts = state["contacts"]
        self._append("contacts/count", contacts["count"], dtype=np.int16)
        self._append("contacts/valid", contacts["valid"], dtype=np.bool_)
        self._append("contacts/geom_pair", contacts["geom_pair"], dtype=np.int32)
        self._append("contacts/position", contacts["position"], dtype=np.float32)
        self._append("contacts/normal", contacts["normal"], dtype=np.float32)
        self._append("contacts/distance", contacts["distance"], dtype=np.float32)
        self._append("contacts/overflow", contacts["overflow"], dtype=np.bool_)

        self._append("images/rgb", rgb, compression=True)
        self._append("images/depth_m", depth.astype(np.float16), compression=True)
        self._append("images/instance_id", instance, compression=True)

        self.length += 1
        self.expected_frame_id = frame_id + 1
        self.file.flush()

    def _decode_rgba(self, encoded: str) -> np.ndarray:
        raw = base64.b64decode(encoded, validate=True)
        expected = self.image_width * self.image_height * 4
        if len(raw) != expected:
            raise ValueError(f"RGBA byte count mismatch: expected {expected}, got {len(raw)}")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(self.image_height, self.image_width, 4)
        return np.flipud(image).copy()

    def _decode_depth(self, encoded: str) -> np.ndarray:
        raw = base64.b64decode(encoded, validate=True)
        expected = self.image_width * self.image_height * 4
        if len(raw) != expected:
            raise ValueError(f"depth byte count mismatch: expected {expected}, got {len(raw)}")
        image = np.frombuffer(raw, dtype="<f4").reshape(self.image_height, self.image_width)
        return np.flipud(image).copy()

    def _decode_instance(self, encoded: str) -> np.ndarray:
        rgba = self._decode_rgba(encoded)
        return (rgba[..., 0].astype(np.uint16) | (rgba[..., 1].astype(np.uint16) << 8)).astype(np.uint16)

    def close(self, success: bool) -> Path:
        if self.closed:
            return self.final_path
        self.file.attrs["success_final"] = bool(success)
        self.file.attrs["frame_count"] = self.length
        self.file.flush()
        self.file.close()
        self.closed = True
        os.replace(self.partial_path, self.final_path)
        return self.final_path

    def abort(self, reason: str) -> None:
        if self.closed:
            return
        self.file.attrs["abort_reason"] = reason
        self.file.attrs["frame_count"] = self.length
        self.file.flush()
        self.file.close()
        self.closed = True
