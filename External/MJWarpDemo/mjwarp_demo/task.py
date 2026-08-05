from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Iterable

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from .recorder import MAX_CONTACTS

PHYSICS_DT = 0.005
ACTION_REPEAT = 10
CONTROL_DT = PHYSICS_DT * ACTION_REPEAT
MAX_FRAMES = 120
MAX_JOINT_SPEED = 2.5
BASE_XY = np.array([-0.35, 0.0], dtype=np.float32)
LINK_LENGTHS = (0.32, 0.28)


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or fallback


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class PlanarPushTask:
    """Batched MJWarp task. Physics is authoritative; Unity is a visual client."""

    def __init__(self, model_path: Path, nworld: int = 1, device: str = "cuda:0") -> None:
        wp.init()
        if device.startswith("cuda") and not wp.is_cuda_available():
            raise RuntimeError("MJWarp demo requires a CUDA-capable GPU; no CUDA device was found")
        self.device = wp.get_device(device)
        self.model_path = model_path
        self.nworld = int(nworld)
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        if not math.isclose(float(self.mj_model.opt.timestep), PHYSICS_DT, abs_tol=1e-9):
            raise ValueError(f"MJCF timestep must be {PHYSICS_DT}")

        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.mj_model)
            self.data = mjw.make_data(self.mj_model, nworld=self.nworld)

        self.shoulder_qpos_adr = int(self.mj_model.jnt_qposadr[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")])
        self.elbow_qpos_adr = int(self.mj_model.jnt_qposadr[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")])
        self.cube_qpos_adr = int(self.mj_model.jnt_qposadr[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")])
        self.cube_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.pusher_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pusher")

        self.rng = np.random.default_rng(0)
        self.policy = "expert"
        self.goal = np.zeros((self.nworld, 3), dtype=np.float32)
        self.stage = np.zeros(self.nworld, dtype=np.int32)
        self.previous_distance = np.zeros(self.nworld, dtype=np.float32)
        self.success_streak = np.zeros(self.nworld, dtype=np.int32)
        self.success = np.zeros(self.nworld, dtype=np.bool_)
        self.terminated = np.zeros(self.nworld, dtype=np.bool_)
        self.random_action = np.zeros((self.nworld, 2), dtype=np.float32)
        self.last_action = np.zeros((self.nworld, 2), dtype=np.float32)
        self.last_reward = np.zeros(self.nworld, dtype=np.float32)
        self.frame_id = 0
        self.seed = 0
        self.reset(seed=0, policy="expert")

    @property
    def gpu_name(self) -> str:
        return self.device.name

    def reset(self, seed: int, policy: str = "expert") -> dict[str, Any]:
        if policy not in {"expert", "random"}:
            raise ValueError(f"unsupported policy: {policy}")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.policy = policy

        qpos = np.tile(np.asarray(self.mj_model.qpos0, dtype=np.float32), (self.nworld, 1))
        qvel = np.zeros((self.nworld, self.mj_model.nv), dtype=np.float32)

        cube_xy = np.column_stack(
            (
                self.rng.uniform(0.00, 0.07, self.nworld),
                self.rng.uniform(-0.11, 0.11, self.nworld),
            )
        ).astype(np.float32)
        target_xy = np.column_stack(
            (
                self.rng.uniform(0.16, 0.22, self.nworld),
                np.clip(cube_xy[:, 1] + self.rng.uniform(-0.07, 0.07, self.nworld), -0.16, 0.16),
            )
        ).astype(np.float32)
        self.goal[:, :2] = target_xy
        self.goal[:, 2] = 0.002

        qpos[:, self.cube_qpos_adr : self.cube_qpos_adr + 3] = np.column_stack(
            (cube_xy, np.full(self.nworld, 0.041, dtype=np.float32))
        )
        qpos[:, self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        push_dir = target_xy - cube_xy
        push_dir /= np.maximum(np.linalg.norm(push_dir, axis=1, keepdims=True), 1e-6)
        behind = cube_xy - push_dir * 0.105
        initial_joint = self._ik(behind)
        initial_joint += self.rng.normal(0.0, 0.015, initial_joint.shape).astype(np.float32)
        qpos[:, self.shoulder_qpos_adr] = initial_joint[:, 0]
        qpos[:, self.elbow_qpos_adr] = initial_joint[:, 1]

        with wp.ScopedDevice(self.device):
            self.data.qpos.assign(qpos)
            self.data.qvel.assign(qvel)
            self.data.ctrl.assign(np.zeros((self.nworld, self.mj_model.nu), dtype=np.float32))
            self.data.time.assign(np.zeros(self.nworld, dtype=np.float32))
            mjw.forward(self.model, self.data)
            wp.synchronize_device(self.device)

        self.stage.fill(0)
        self.success_streak.fill(0)
        self.success.fill(False)
        self.terminated.fill(False)
        self.random_action.fill(0.0)
        self.last_action.fill(0.0)
        self.last_reward.fill(0.0)
        self.frame_id = 0
        cube_after_reset = self._cube_positions()
        self.previous_distance = np.linalg.norm(cube_after_reset[:, :2] - self.goal[:, :2], axis=1).astype(np.float32)
        return self.state_dict()

    def _ik(self, target_xy: np.ndarray) -> np.ndarray:
        l1, l2 = LINK_LENGTHS
        rel = np.asarray(target_xy, dtype=np.float32) - BASE_XY
        radius = np.linalg.norm(rel, axis=1)
        safe_radius = np.clip(radius, abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3)
        rel = rel * (safe_radius / np.maximum(radius, 1e-6))[:, None]
        cos_q2 = np.clip((safe_radius**2 - l1**2 - l2**2) / (2.0 * l1 * l2), -1.0, 1.0)
        q2 = -np.arccos(cos_q2)
        q1 = np.arctan2(rel[:, 1], rel[:, 0]) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        return np.column_stack((_wrap_angle(q1), _wrap_angle(q2))).astype(np.float32)

    def _policy_action(self) -> np.ndarray:
        if self.policy == "random":
            noise = self.rng.uniform(-1.0, 1.0, (self.nworld, 2)).astype(np.float32)
            self.random_action = np.clip(0.86 * self.random_action + 0.24 * noise, -1.0, 1.0)
            return self.random_action.copy()

        cube = self._cube_positions()[:, :2]
        pusher = self._body_positions()[:, self.pusher_body_id, :2]
        push_dir = self.goal[:, :2] - cube
        push_dir /= np.maximum(np.linalg.norm(push_dir, axis=1, keepdims=True), 1e-6)
        behind = cube - push_dir * 0.095
        self.stage[np.linalg.norm(pusher - behind, axis=1) < 0.055] = 1
        push_target = self.goal[:, :2] + push_dir * 0.035
        target = np.where((self.stage == 0)[:, None], behind, push_target)

        desired_joint = self._ik(target)
        qpos = self.data.qpos.numpy()
        current_joint = qpos[:, [self.shoulder_qpos_adr, self.elbow_qpos_adr]]
        error = _wrap_angle(desired_joint - current_joint)
        desired_velocity = np.clip(error * 5.0, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
        return (desired_velocity / MAX_JOINT_SPEED).astype(np.float32)

    def step(self, action: np.ndarray | list[float] | None = None) -> dict[str, Any]:
        if bool(np.all(self.terminated)):
            return self.state_dict()
        start = time.perf_counter()
        if action is None:
            action_array = self._policy_action()
        else:
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.shape == (2,):
                action_array = np.tile(action_array, (self.nworld, 1))
            if action_array.shape != (self.nworld, 2):
                raise ValueError(f"action must have shape (2,) or ({self.nworld}, 2), got {action_array.shape}")
            action_array = np.clip(action_array, -1.0, 1.0)

        qvel = self.data.qvel.numpy()[:, :2]
        desired_velocity = action_array * MAX_JOINT_SPEED
        torque = np.clip(4.0 * (desired_velocity - qvel), -8.0, 8.0).astype(np.float32)
        controls = np.zeros((self.nworld, self.mj_model.nu), dtype=np.float32)
        controls[:, :2] = torque

        with wp.ScopedDevice(self.device):
            self.data.ctrl.assign(controls)
            for _ in range(ACTION_REPEAT):
                mjw.step(self.model, self.data)
            wp.synchronize_device(self.device)

        self.frame_id += 1
        self.last_action = action_array
        cube = self._cube_positions()
        distance = np.linalg.norm(cube[:, :2] - self.goal[:, :2], axis=1).astype(np.float32)
        progress = self.previous_distance - distance
        reached = distance < 0.06
        self.success_streak = np.where(reached, self.success_streak + 1, 0)
        self.success |= self.success_streak >= 3
        out_of_bounds = (np.abs(cube[:, 0]) > 0.8) | (np.abs(cube[:, 1]) > 0.6) | (cube[:, 2] < -0.02)
        timed_out = self.frame_id >= MAX_FRAMES
        self.terminated |= self.success | out_of_bounds | timed_out
        self.last_reward = (5.0 * progress - 0.01 * np.sum(action_array**2, axis=1) + self.success.astype(np.float32)).astype(np.float32)
        self.previous_distance = distance
        elapsed = max(time.perf_counter() - start, 1e-9)
        return self.state_dict(control_steps_per_second=self.nworld / elapsed)

    def _cube_positions(self) -> np.ndarray:
        return self.data.qpos.numpy()[:, self.cube_qpos_adr : self.cube_qpos_adr + 3]

    def _body_positions(self) -> np.ndarray:
        return self.data.xpos.numpy()

    def _contacts_for_world(self, world_id: int) -> dict[str, Any]:
        count_total = int(self.data.nacon.numpy()[0])
        world_ids = self.data.contact.worldid.numpy()[:count_total]
        indices = np.flatnonzero(world_ids == world_id)
        overflow = len(indices) > MAX_CONTACTS
        indices = indices[:MAX_CONTACTS]
        valid = np.zeros(MAX_CONTACTS, dtype=np.bool_)
        geom_pair = np.full((MAX_CONTACTS, 2), -1, dtype=np.int32)
        position = np.zeros((MAX_CONTACTS, 3), dtype=np.float32)
        normal = np.zeros((MAX_CONTACTS, 3), dtype=np.float32)
        distance = np.zeros(MAX_CONTACTS, dtype=np.float32)
        if len(indices):
            valid[: len(indices)] = True
            geom_pair[: len(indices)] = self.data.contact.geom.numpy()[indices]
            position[: len(indices)] = self.data.contact.pos.numpy()[indices]
            frames = self.data.contact.frame.numpy()[indices]
            normal[: len(indices)] = frames[:, 0, :]
            distance[: len(indices)] = self.data.contact.dist.numpy()[indices]
        return {
            "count": int(len(indices)),
            "valid": valid.tolist(),
            "geom_pair": geom_pair.tolist(),
            "position": position.tolist(),
            "normal": normal.tolist(),
            "distance": distance.tolist(),
            "overflow": bool(overflow),
        }

    def state_dict(self, control_steps_per_second: float = 0.0) -> dict[str, Any]:
        qpos = self.data.qpos.numpy()[0]
        qvel = self.data.qvel.numpy()[0]
        body_position = self.data.xpos.numpy()[0]
        body_quaternion = self.data.xquat.numpy()[0]
        body_wrench = self.data.cfrc_ext.numpy()[0]
        sim_time = float(self.data.time.numpy()[0])
        return {
            "frame_id": self.frame_id,
            "sim_time": sim_time,
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
            "body_position": body_position.tolist(),
            "body_quaternion": body_quaternion.tolist(),
            "body_external_wrench": body_wrench.tolist(),
            "action": self.last_action[0].tolist(),
            "reward": float(self.last_reward[0]),
            "terminated": bool(self.terminated[0]),
            "success": bool(self.success[0]),
            "goal_position": self.goal[0].tolist(),
            "contacts": self._contacts_for_world(0),
            "metrics": {
                "nworld": self.nworld,
                "success_count": int(np.count_nonzero(self.success)),
                "mean_reward": float(np.mean(self.last_reward)),
                "control_steps_per_second": float(control_steps_per_second),
                "physics_steps_per_second": float(control_steps_per_second * ACTION_REPEAT),
            },
        }

    def model_spec(self) -> dict[str, Any]:
        geom_types = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        }
        bodies = [
            {
                "id": body_id,
                "name": _name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}"),
            }
            for body_id in range(self.mj_model.nbody)
        ]
        geoms: list[dict[str, Any]] = []
        for geom_id in range(self.mj_model.ngeom):
            geom_type_id = int(self.mj_model.geom_type[geom_id])
            geom_type = geom_types.get(geom_type_id)
            if geom_type is None:
                continue
            geoms.append(
                {
                    "id": geom_id,
                    "name": _name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, f"geom_{geom_id}"),
                    "body_id": int(self.mj_model.geom_bodyid[geom_id]),
                    "type": geom_type,
                    "size": np.asarray(self.mj_model.geom_size[geom_id], dtype=np.float32).tolist(),
                    "position": np.asarray(self.mj_model.geom_pos[geom_id], dtype=np.float32).tolist(),
                    "quaternion": np.asarray(self.mj_model.geom_quat[geom_id], dtype=np.float32).tolist(),
                    "rgba": np.asarray(self.mj_model.geom_rgba[geom_id], dtype=np.float32).tolist(),
                }
            )
        return {
            "name": "planar_push",
            "physics_dt": PHYSICS_DT,
            "control_dt": CONTROL_DT,
            "max_frames": MAX_FRAMES,
            "bodies": bodies,
            "geoms": geoms,
        }


def benchmark_sizes(model_path: Path, sizes: Iterable[int], steps: int = 300, warmup: int = 30, device: str = "cuda:0") -> list[dict[str, Any]]:
    wp.init()
    cuda_device = wp.get_device(device)
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    results: list[dict[str, Any]] = []
    for requested_size in sizes:
        actual_size = int(requested_size)
        fallback = False
        while True:
            try:
                with wp.ScopedDevice(cuda_device):
                    model = mjw.put_model(mj_model)
                    data = mjw.make_data(mj_model, nworld=actual_size)
                    data.ctrl.zero_()
                    for _ in range(warmup):
                        mjw.step(model, data)
                    wp.synchronize_device(cuda_device)
                    with wp.ScopedCapture(device=cuda_device) as capture:
                        mjw.step(model, data)
                    start = time.perf_counter()
                    for _ in range(steps):
                        wp.capture_launch(capture.graph)
                    wp.synchronize_device(cuda_device)
                    elapsed = time.perf_counter() - start
                results.append(
                    {
                        "requested_nworld": int(requested_size),
                        "actual_nworld": actual_size,
                        "fallback": fallback,
                        "steps": steps,
                        "elapsed_seconds": elapsed,
                        "physics_steps_per_second": actual_size * steps / max(elapsed, 1e-9),
                    }
                )
                del data, model
                gc.collect()
                break
            except Exception as exc:
                if requested_size == 1024 and actual_size == 1024:
                    actual_size = 512
                    fallback = True
                    gc.collect()
                    continue
                results.append(
                    {
                        "requested_nworld": int(requested_size),
                        "actual_nworld": 0,
                        "fallback": fallback,
                        "error": str(exc),
                    }
                )
                break
    return results
