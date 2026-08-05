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
from .scenarios import DEFAULT_SCENARIO_ID, ScenarioDefinition, get_scenario

PHYSICS_DT = 0.005
ACTION_REPEAT = 10
CONTROL_DT = PHYSICS_DT * ACTION_REPEAT
MAX_FRAMES = 120
ARM_BASE_XY = np.array([-0.35, 0.0], dtype=np.float32)
ARM_LINK_LENGTHS = (0.32, 0.28)


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or fallback


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class EmbodiedTask:
    """Scenario-aware batched MJWarp task; Unity remains a visual/data client."""

    def __init__(
        self,
        definition: ScenarioDefinition,
        model_path: Path,
        nworld: int = 1,
        device: str = "cuda:0",
    ) -> None:
        wp.init()
        if device.startswith("cuda") and not wp.is_cuda_available():
            raise RuntimeError("MJWarp demo requires a CUDA-capable GPU; no CUDA device was found")
        self.definition = definition
        self.task_name = definition.scenario_id
        self.device = wp.get_device(device)
        self.model_path = Path(model_path)
        self.nworld = int(nworld)
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if not math.isclose(float(self.mj_model.opt.timestep), PHYSICS_DT, abs_tol=1e-9):
            raise ValueError(f"MJCF timestep must be {PHYSICS_DT}")
        if self.mj_model.nu != 2:
            raise ValueError(f"scenario {self.task_name} must expose exactly two actuators")

        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.mj_model)
            self.data = mjw.make_data(self.mj_model, nworld=self.nworld)

        joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in definition.controlled_joints
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError(f"scenario {self.task_name} is missing controlled joints {definition.controlled_joints}")
        self.controlled_qpos_adr = np.asarray([self.mj_model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        self.controlled_dof_adr = np.asarray([self.mj_model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        self.agent_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, definition.agent_body)
        if self.agent_body_id < 0:
            raise ValueError(f"scenario {self.task_name} is missing agent body {definition.agent_body}")

        self.object_qpos_adr: int | None = None
        self.object_body_id: int | None = None
        if definition.object_joint is not None:
            object_joint_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, definition.object_joint
            )
            self.object_qpos_adr = int(self.mj_model.jnt_qposadr[object_joint_id])
        if definition.object_body is not None:
            self.object_body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, definition.object_body
            )

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
        if self.definition.mode == "push":
            self._reset_push(qpos)
        elif self.definition.mode == "insert":
            self._reset_insert(qpos)
        elif self.definition.mode == "reach":
            self._reset_reach(qpos)
        elif self.definition.mode == "navigate":
            self._reset_navigation(qpos)
        else:
            raise ValueError(f"unsupported scenario mode: {self.definition.mode}")

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
        position = self._evaluation_positions()
        self.previous_distance = np.linalg.norm(position[:, :2] - self.goal[:, :2], axis=1).astype(np.float32)
        return self.state_dict()

    def _reset_push(self, qpos: np.ndarray) -> None:
        object_xy = np.column_stack(
            (
                self.rng.uniform(0.00, 0.07, self.nworld),
                self.rng.uniform(-0.11, 0.11, self.nworld),
            )
        ).astype(np.float32)
        goal_xy = np.column_stack(
            (
                self.rng.uniform(0.16, 0.22, self.nworld),
                np.clip(object_xy[:, 1] + self.rng.uniform(-0.07, 0.07, self.nworld), -0.16, 0.16),
            )
        ).astype(np.float32)
        self._place_object_and_arm(qpos, object_xy, goal_xy, height=0.041, behind_distance=0.105)

    def _reset_insert(self, qpos: np.ndarray) -> None:
        goal_xy = np.column_stack(
            (
                self.rng.uniform(0.115, 0.13, self.nworld),
                self.rng.uniform(-0.012, 0.012, self.nworld),
            )
        ).astype(np.float32)
        object_xy = np.column_stack(
            (
                self.rng.uniform(-0.01, 0.025, self.nworld),
                np.clip(goal_xy[:, 1] + self.rng.uniform(-0.012, 0.012, self.nworld), -0.020, 0.020),
            )
        ).astype(np.float32)
        self._place_object_and_arm(qpos, object_xy, goal_xy, height=0.031, behind_distance=0.095)

    def _place_object_and_arm(
        self,
        qpos: np.ndarray,
        object_xy: np.ndarray,
        goal_xy: np.ndarray,
        *,
        height: float,
        behind_distance: float,
    ) -> None:
        if self.object_qpos_adr is None:
            raise ValueError(f"scenario {self.task_name} requires a free object joint")
        self.goal[:, :2] = goal_xy
        self.goal[:, 2] = 0.002
        qpos[:, self.object_qpos_adr : self.object_qpos_adr + 3] = np.column_stack(
            (object_xy, np.full(self.nworld, height, dtype=np.float32))
        )
        qpos[:, self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        push_dir = goal_xy - object_xy
        push_dir /= np.maximum(np.linalg.norm(push_dir, axis=1, keepdims=True), 1e-6)
        behind = object_xy - push_dir * behind_distance
        initial_joint = self._ik(behind)
        initial_joint += self.rng.normal(0.0, 0.015, initial_joint.shape).astype(np.float32)
        qpos[:, self.controlled_qpos_adr] = initial_joint

    def _reset_reach(self, qpos: np.ndarray) -> None:
        stations = np.asarray(
            [[0.02, -0.15], [0.15, 0.0], [0.02, 0.15]], dtype=np.float32
        )
        selected = self.rng.integers(0, len(stations), size=self.nworld)
        self.goal[:, :2] = stations[selected]
        self.goal[:, 2] = 0.06
        home = np.tile(np.asarray([0.02, 0.0], dtype=np.float32), (self.nworld, 1))
        initial_joint = self._ik(home)
        initial_joint += self.rng.normal(0.0, 0.04, initial_joint.shape).astype(np.float32)
        qpos[:, self.controlled_qpos_adr] = initial_joint

    def _reset_navigation(self, qpos: np.ndarray) -> None:
        start_xy = np.column_stack(
            (
                self.rng.uniform(-0.53, -0.46, self.nworld),
                self.rng.uniform(-0.34, 0.34, self.nworld),
            )
        ).astype(np.float32)
        goal_xy = np.column_stack(
            (
                self.rng.uniform(0.46, 0.53, self.nworld),
                self.rng.uniform(-0.34, 0.34, self.nworld),
            )
        ).astype(np.float32)
        qpos[:, self.controlled_qpos_adr] = start_xy
        self.goal[:, :2] = goal_xy
        self.goal[:, 2] = 0.005

    def _ik(self, target_xy: np.ndarray) -> np.ndarray:
        l1, l2 = ARM_LINK_LENGTHS
        rel = np.asarray(target_xy, dtype=np.float32) - ARM_BASE_XY
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
        if self.definition.mode in {"push", "insert"}:
            return self._push_policy_action()
        if self.definition.mode == "reach":
            desired_joint = self._ik(self.goal[:, :2])
            return self._joint_velocity_action(desired_joint)
        if self.definition.mode == "navigate":
            return self._navigation_policy_action()
        raise ValueError(f"unsupported scenario mode: {self.definition.mode}")

    def _push_policy_action(self) -> np.ndarray:
        object_position = self._object_positions()[:, :2]
        pusher = self._body_positions()[:, self.agent_body_id, :2]
        push_dir = self.goal[:, :2] - object_position
        push_dir /= np.maximum(np.linalg.norm(push_dir, axis=1, keepdims=True), 1e-6)
        behind_offset = 0.095 if self.definition.mode == "push" else 0.082
        behind = object_position - push_dir * behind_offset
        distance_to_object = np.linalg.norm(pusher - object_position, axis=1)
        self.stage[(self.stage == 1) & (distance_to_object > 0.12)] = 0
        self.stage[np.linalg.norm(pusher - behind, axis=1) < 0.050] = 1
        contact_depth = 0.055 if self.definition.mode == "push" else 0.042
        push_target = object_position + push_dir * contact_depth
        target = np.where((self.stage == 0)[:, None], behind, push_target)
        return self._joint_velocity_action(self._ik(target))

    def _joint_velocity_action(self, desired_joint: np.ndarray) -> np.ndarray:
        qpos = self.data.qpos.numpy()
        current_joint = qpos[:, self.controlled_qpos_adr]
        error = _wrap_angle(desired_joint - current_joint)
        desired_velocity = np.clip(
            error * 5.0, -self.definition.max_speed, self.definition.max_speed
        )
        return (desired_velocity / self.definition.max_speed).astype(np.float32)

    def _navigation_policy_action(self) -> np.ndarray:
        position = self._body_positions()[:, self.agent_body_id, :2]
        waypoint_left = np.tile(np.asarray([-0.14, 0.0], dtype=np.float32), (self.nworld, 1))
        waypoint_right = np.tile(np.asarray([0.14, 0.0], dtype=np.float32), (self.nworld, 1))
        distance_left = np.linalg.norm(position - waypoint_left, axis=1)
        self.stage[(self.stage == 0) & (distance_left < 0.07)] = 1
        distance_right = np.linalg.norm(position - waypoint_right, axis=1)
        self.stage[(self.stage == 1) & (distance_right < 0.07)] = 2
        target = np.where(
            (self.stage == 0)[:, None],
            waypoint_left,
            np.where((self.stage == 1)[:, None], waypoint_right, self.goal[:, :2]),
        )
        desired_velocity = np.clip(
            (target - position) * 3.0, -self.definition.max_speed, self.definition.max_speed
        )
        return (desired_velocity / self.definition.max_speed).astype(np.float32)

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
        action_array[self.terminated] = 0.0

        qvel = self.data.qvel.numpy()[:, self.controlled_dof_adr]
        desired_velocity = action_array * self.definition.max_speed
        torque = np.clip(
            4.0 * (desired_velocity - qvel),
            -self.definition.torque_limit,
            self.definition.torque_limit,
        ).astype(np.float32)
        controls = np.zeros((self.nworld, self.mj_model.nu), dtype=np.float32)
        controls[:, :2] = torque

        with wp.ScopedDevice(self.device):
            self.data.ctrl.assign(controls)
            for _ in range(ACTION_REPEAT):
                mjw.step(self.model, self.data)
            wp.synchronize_device(self.device)

        self.frame_id += 1
        self.last_action = action_array
        position = self._evaluation_positions()
        distance = np.linalg.norm(position[:, :2] - self.goal[:, :2], axis=1).astype(np.float32)
        progress = self.previous_distance - distance
        reached = distance < self.definition.goal_radius
        self.success_streak = np.where(reached, self.success_streak + 1, 0)
        self.success |= self.success_streak >= self.definition.success_frames
        out_of_bounds = (
            (np.abs(position[:, 0]) > 0.80)
            | (np.abs(position[:, 1]) > 0.62)
            | (position[:, 2] < -0.02)
        )
        timed_out = self.frame_id >= MAX_FRAMES
        self.terminated |= self.success | out_of_bounds | timed_out
        self.last_reward = (
            self.definition.progress_scale * progress
            - 0.01 * np.sum(action_array**2, axis=1)
            + self.success.astype(np.float32)
        ).astype(np.float32)
        self.previous_distance = distance
        elapsed = max(time.perf_counter() - start, 1e-9)
        return self.state_dict(control_steps_per_second=self.nworld / elapsed)

    def _object_positions(self) -> np.ndarray:
        if self.object_qpos_adr is None:
            raise ValueError(f"scenario {self.task_name} does not have a free object")
        return self.data.qpos.numpy()[:, self.object_qpos_adr : self.object_qpos_adr + 3]

    def _body_positions(self) -> np.ndarray:
        return self.data.xpos.numpy()

    def _evaluation_positions(self) -> np.ndarray:
        if self.object_qpos_adr is not None:
            return self._object_positions()
        return self._body_positions()[:, self.agent_body_id, :]

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
            "scenario_id": self.task_name,
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
            "task_stage": int(self.stage[0]),
            "distance_to_goal": float(self.previous_distance[0]),
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
            "name": self.task_name,
            "scenario_id": self.task_name,
            "display_name": self.definition.display_name,
            "business_type": self.definition.business_type,
            "description": self.definition.description,
            "official_reference": self.definition.official_reference,
            "physics_dt": PHYSICS_DT,
            "control_dt": CONTROL_DT,
            "max_frames": MAX_FRAMES,
            "goal_radius": self.definition.goal_radius,
            "camera_position": list(self.definition.camera_position),
            "camera_look_at": list(self.definition.camera_look_at),
            "bodies": bodies,
            "geoms": geoms,
        }


class PlanarPushTask(EmbodiedTask):
    """Backward-compatible entry point retained for existing scripts and tests."""

    def __init__(self, model_path: Path, nworld: int = 1, device: str = "cuda:0") -> None:
        super().__init__(get_scenario(DEFAULT_SCENARIO_ID), model_path, nworld=nworld, device=device)


def create_task(
    scenario_id: str,
    package_root: Path,
    *,
    nworld: int = 1,
    device: str = "cuda:0",
    model_override: Path | None = None,
) -> EmbodiedTask:
    definition = get_scenario(scenario_id)
    model_path = model_override or definition.model_path(package_root)
    return EmbodiedTask(definition, model_path, nworld=nworld, device=device)


def benchmark_sizes(
    model_path: Path,
    sizes: Iterable[int],
    steps: int = 300,
    warmup: int = 30,
    device: str = "cuda:0",
) -> list[dict[str, Any]]:
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
