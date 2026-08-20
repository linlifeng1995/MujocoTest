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
from .scenarios import DEFAULT_SCENARIO_ID, RobotDefinition, ScenarioDefinition, get_robot, get_scenario

PHYSICS_DT = 0.005
ACTION_REPEAT = 10
CONTROL_DT = PHYSICS_DT * ACTION_REPEAT
ARM_BASE_XY = np.array([-0.35, 0.0], dtype=np.float32)
ARM_LINK_LENGTHS = (0.32, 0.28)
PANDA_STAGE_DURATIONS = (50, 70, 35, 60, 90, 65, 35, 45)

CONTACT_CATEGORY_NAMES = {
    0: "unknown",
    1: "robot",
    2: "manipulated_object",
    3: "target_fixture",
    4: "environment",
}
CONTACT_TYPE_NAMES = {
    0: "unknown",
    1: "target_grasp",
    2: "target_goal_contact",
    3: "object_environment",
    4: "non_target_collision",
    5: "robot_self_contact",
}


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
        self.robot: RobotDefinition = get_robot(definition.robot_id)
        self.task_name = definition.scenario_id
        self.device = wp.get_device(device)
        self.model_path = Path(model_path)
        self.nworld = int(nworld)
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if not math.isclose(float(self.mj_model.opt.timestep), PHYSICS_DT, abs_tol=1e-9):
            raise ValueError(f"MJCF timestep must be {PHYSICS_DT}")

        self.actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.robot.actuator_names
            ],
            dtype=np.int32,
        )
        if np.any(self.actuator_ids < 0):
            raise ValueError(f"scenario {self.task_name} is missing actuators {self.robot.actuator_names}")
        if self.mj_model.nu != self.robot.action_dim:
            raise ValueError(
                f"scenario {self.task_name} exposes {self.mj_model.nu} actuators, "
                f"robot contract expects {self.robot.action_dim}"
            )

        joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.robot.controlled_joints
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError(
                f"scenario {self.task_name} is missing controlled joints {self.robot.controlled_joints}"
            )
        self.controlled_joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self.controlled_qpos_adr = np.asarray([self.mj_model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        self.controlled_dof_adr = np.asarray([self.mj_model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
        self.agent_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, definition.agent_body)
        if self.agent_body_id < 0:
            raise ValueError(f"scenario {self.task_name} is missing agent body {definition.agent_body}")
        self.end_effector_site_id = -1
        if self.robot.end_effector_site:
            self.end_effector_site_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.robot.end_effector_site
            )

        self.object_qpos_adr: int | None = None
        self.object_body_id: int | None = None
        if definition.object_joint is not None:
            object_joint_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, definition.object_joint
            )
            if object_joint_id < 0:
                raise ValueError(f"scenario {self.task_name} is missing object joint {definition.object_joint}")
            self.object_qpos_adr = int(self.mj_model.jnt_qposadr[object_joint_id])
        if definition.object_body is not None:
            self.object_body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, definition.object_body
            )
            if self.object_body_id < 0:
                raise ValueError(f"scenario {self.task_name} is missing object body {definition.object_body}")
        self.object_geom_ids = (
            np.flatnonzero(np.asarray(self.mj_model.geom_bodyid) == self.object_body_id).astype(np.int32)
            if self.object_body_id is not None
            else np.asarray([], dtype=np.int32)
        )
        self.body_names = [
            _name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}")
            for body_id in range(self.mj_model.nbody)
        ]
        self.geom_names = [
            _name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, f"geom_{geom_id}")
            for geom_id in range(self.mj_model.ngeom)
        ]
        self.geom_body_ids = np.asarray(self.mj_model.geom_bodyid, dtype=np.int32)
        self.geom_category_ids = np.asarray(
            [self._classify_geom(geom_id) for geom_id in range(self.mj_model.ngeom)],
            dtype=np.int8,
        )
        self.object_bottom_offset = self._object_bottom_offset()
        self.insertion_entry_height = self._insertion_entry_height()
        self.grasp_weld_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, "expert_grasp_weld"
        )

        self.action_dim = self.robot.action_dim
        actuator_ranges = np.asarray(self.mj_model.actuator_ctrlrange[self.actuator_ids], dtype=np.float32)
        self.action_command_low = actuator_ranges[:, 0].copy()
        self.action_command_high = actuator_ranges[:, 1].copy()
        if self.robot.robot_id == "franka_panda":
            self.action_command_high[-1] *= 2.0
        self.base_body_mass = np.asarray(self.mj_model.body_mass, dtype=np.float32).copy()
        self.base_geom_friction = np.asarray(self.mj_model.geom_friction, dtype=np.float32).copy()

        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(self.mj_model)
            self.data = mjw.make_data(self.mj_model, nworld=self.nworld)

        self.rng = np.random.default_rng(0)
        self.policy = "expert"
        self.goal = np.zeros((self.nworld, 3), dtype=np.float32)
        self.stage = np.zeros(self.nworld, dtype=np.int32)
        self.previous_distance = np.zeros(self.nworld, dtype=np.float32)
        self.success_streak = np.zeros(self.nworld, dtype=np.int32)
        self.success = np.zeros(self.nworld, dtype=np.bool_)
        self.terminated = np.zeros(self.nworld, dtype=np.bool_)
        self.termination_reason = np.full(self.nworld, "", dtype=object)
        self.random_action = np.zeros((self.nworld, self.action_dim), dtype=np.float32)
        self.last_action = np.zeros((self.nworld, self.action_dim), dtype=np.float32)
        self.last_action_command = np.zeros((self.nworld, self.action_dim), dtype=np.float32)
        self.last_reward = np.zeros(self.nworld, dtype=np.float32)
        self.frame_id = 0
        self.seed = 0
        self.randomization: dict[str, Any] = {}
        self._panda_waypoints: np.ndarray | None = None
        self._panda_gripper_targets: np.ndarray | None = None
        self._panda_joint_trajectory: np.ndarray | None = None
        self._panda_gripper_trajectory: np.ndarray | None = None
        self._ik_data = mujoco.MjData(self.mj_model) if self.end_effector_site_id >= 0 else None
        self._ik_target_quaternion: np.ndarray | None = None
        self._ik_target_matrix: np.ndarray | None = None
        self._grasp_weld_active = False
        self.reset(seed=0, policy="expert")

    def _classify_geom(self, geom_id: int) -> int:
        body_id = int(self.mj_model.geom_bodyid[geom_id])
        if self.object_body_id is not None and body_id == self.object_body_id:
            return 2
        body_name = self.body_names[body_id]
        if body_name in {"bin", "socket"}:
            return 3
        if body_name.startswith("link") or body_name in {"hand", "left_finger", "right_finger"}:
            return 1
        return 4

    def _object_bottom_offset(self) -> float:
        offsets: list[float] = []
        for geom_id in self.object_geom_ids.tolist():
            geom_type = int(self.mj_model.geom_type[geom_id])
            size = np.asarray(self.mj_model.geom_size[geom_id], dtype=np.float64)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                half_height = float(size[1])
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                half_height = float(size[2])
            else:
                continue
            offsets.append(half_height - float(self.mj_model.geom_pos[geom_id, 2]))
        return max(offsets, default=0.0)

    def _insertion_entry_height(self) -> float:
        socket_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "socket"
        )
        if socket_body_id < 0:
            return 0.0
        wall_tops: list[float] = []
        for geom_id in np.flatnonzero(self.geom_body_ids == socket_body_id).tolist():
            if self.geom_names[geom_id] == "socket_base":
                continue
            wall_tops.append(
                float(self.mj_model.body_pos[socket_body_id, 2])
                + float(self.mj_model.geom_pos[geom_id, 2])
                + float(self.mj_model.geom_size[geom_id, 2])
            )
        return max(wall_tops, default=0.0)

    @property
    def gpu_name(self) -> str:
        return self.device.name

    @property
    def is_panda(self) -> bool:
        return self.robot.robot_id == "franka_panda"

    def reset(self, seed: int, policy: str = "expert") -> dict[str, Any]:
        if policy not in {"expert", "recovery", "perturbed", "random", "learned"}:
            raise ValueError(f"unsupported policy: {policy}")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.policy = policy
        self.randomization = {}

        qpos = np.tile(np.asarray(self.mj_model.qpos0, dtype=np.float32), (self.nworld, 1))
        qvel = np.zeros((self.nworld, self.mj_model.nv), dtype=np.float32)
        if self.is_panda:
            self._reset_panda(qpos)
        elif self.definition.mode == "push":
            self._reset_push(qpos)
        elif self.definition.mode == "insert":
            self._reset_insert(qpos)
        elif self.definition.mode == "reach":
            self._reset_reach(qpos)
        elif self.definition.mode == "navigate":
            self._reset_navigation(qpos)
        else:
            raise ValueError(f"unsupported scenario mode: {self.definition.mode}")

        self._apply_physics_randomization()
        initial_controls = self._initial_controls(qpos)
        with wp.ScopedDevice(self.device):
            self.data.qpos.assign(qpos)
            self.data.qvel.assign(qvel)
            self.data.ctrl.assign(initial_controls)
            self.data.time.assign(np.zeros(self.nworld, dtype=np.float32))
            mjw.forward(self.model, self.data)
            wp.synchronize_device(self.device)
        self._set_expert_grasp_constraint(False)

        self.stage.fill(0)
        self.success_streak.fill(0)
        self.success.fill(False)
        self.terminated.fill(False)
        self.termination_reason[:] = ""
        self.random_action.fill(0.0)
        self.last_reward.fill(0.0)
        self.frame_id = 0
        if self.is_panda:
            self.last_action_command = self._logical_command_from_controls(initial_controls)
            self.last_action = self._normalize_position_command(self.last_action_command)
        else:
            self.last_action.fill(0.0)
            self.last_action_command.fill(0.0)
        position = self._evaluation_positions()
        self.previous_distance = self._task_distance(position)
        return self.state_dict()

    def _reset_panda(self, qpos: np.ndarray) -> None:
        home = np.asarray(self.robot.home_qpos, dtype=np.float32)
        if home.shape != (len(self.controlled_qpos_adr),):
            raise ValueError("Panda home_qpos does not match controlled joint count")
        qpos[:, self.controlled_qpos_adr] = home
        if self.object_qpos_adr is None:
            raise ValueError(f"scenario {self.task_name} requires a free object joint")

        if self.definition.mode == "panda_pick_place":
            object_xy = np.column_stack(
                (self.rng.uniform(0.42, 0.49, self.nworld), self.rng.uniform(-0.18, -0.07, self.nworld))
            ).astype(np.float32)
            fixture_name = "bin"
            object_z = 0.075
            goal_z = 0.082
        else:
            object_xy = np.column_stack(
                (self.rng.uniform(0.40, 0.46, self.nworld), self.rng.uniform(-0.18, -0.10, self.nworld))
            ).astype(np.float32)
            fixture_name = "socket"
            object_z = 0.105
            goal_z = 0.105

        fixture_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, fixture_name
        )
        if fixture_body_id < 0:
            raise ValueError(f"scenario {self.task_name} is missing target fixture {fixture_name}")
        fixture_xy = np.asarray(self.mj_model.body_pos[fixture_body_id, :2], dtype=np.float32)
        goal_xy = np.tile(fixture_xy, (self.nworld, 1))

        qpos[:, self.object_qpos_adr : self.object_qpos_adr + 3] = np.column_stack(
            (object_xy, np.full(self.nworld, object_z, dtype=np.float32))
        )
        qpos[:, self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.goal[:, :2] = goal_xy
        self.goal[:, 2] = goal_z
        self.randomization = {
            "randomization_group": f"panda-{self.seed % 20:02d}",
            "object_position": [float(object_xy[0, 0]), float(object_xy[0, 1]), object_z],
            "goal_position": self.goal[0].astype(float).tolist(),
            "target_fixture": fixture_name,
            "target_fixture_body_id": int(fixture_body_id),
            "target_fixture_randomized": False,
            "object_config_id": f"{self.task_name}-object-config-{self.seed % 20:02d}",
            "scene_config_id": f"{self.task_name}-scene-config-{self.seed % 20:02d}",
            "object_mass_scale": float(self.rng.uniform(0.85, 1.15)),
            "object_friction_scale": float(self.rng.uniform(0.80, 1.20)),
            "motion_noise_std": (
                (0.012 if self.definition.mode == "panda_pick_place" else 0.018)
                if self.policy == "recovery"
                else 0.012
                if self.policy == "perturbed"
                else 0.0
            ),
            "disturbance_profile": (
                (
                    "post_grasp_transport_disturbance_then_expert_recovery"
                    if self.definition.mode == "panda_pick_place"
                    else "approach_and_transport_disturbances_then_expert_recovery"
                )
                if self.policy == "recovery"
                else "continuous_joint_command_noise_with_seeded_failure_injection"
                if self.policy == "perturbed"
                else "none"
            ),
        }
        self._prepare_panda_waypoints(qpos)

    def _prepare_panda_waypoints(self, qpos: np.ndarray) -> None:
        if self._ik_data is None or self.end_effector_site_id < 0:
            raise RuntimeError("Panda scenario requires a valid end-effector site")
        stages = len(PANDA_STAGE_DURATIONS)
        total_frames = sum(PANDA_STAGE_DURATIONS)
        waypoints = np.zeros((self.nworld, stages, 7), dtype=np.float32)
        grippers = np.zeros((self.nworld, stages), dtype=np.float32)
        joint_trajectory = np.zeros((self.nworld, total_frames, 7), dtype=np.float32)
        gripper_trajectory = np.zeros((self.nworld, total_frames), dtype=np.float32)
        for world in range(self.nworld):
            object_position = np.asarray(
                qpos[world, self.object_qpos_adr : self.object_qpos_adr + 3], dtype=np.float64
            )
            goal = np.asarray(self.goal[world], dtype=np.float64)
            if self.definition.mode == "panda_pick_place":
                targets = (
                    object_position + np.asarray([0.0, 0.0, 0.18]),
                    object_position + np.asarray([0.0, 0.0, 0.003]),
                    object_position + np.asarray([0.0, 0.0, 0.003]),
                    object_position + np.asarray([0.0, 0.0, 0.24]),
                    goal + np.asarray([0.0, 0.0, 0.20]),
                    goal + np.asarray([0.0, 0.0, 0.010]),
                    goal + np.asarray([0.0, 0.0, 0.010]),
                    goal + np.asarray([0.0, 0.0, 0.24]),
                )
            else:
                transit_midpoint = 0.5 * (object_position + goal)
                targets = (
                    object_position + np.asarray([0.0, 0.0, 0.20]),
                    object_position + np.asarray([0.0, 0.0, 0.020]),
                    object_position + np.asarray([0.0, 0.0, 0.020]),
                    object_position + np.asarray([0.0, 0.0, 0.25]),
                    transit_midpoint + np.asarray([0.0, 0.0, 0.25]),
                    goal + np.asarray([0.0, 0.0, 0.22]),
                    goal + np.asarray([0.0, 0.0, 0.038]),
                    goal + np.asarray([0.0, 0.0, 0.20]),
                )
            if self.definition.mode == "panda_peg_insert":
                grippers[world] = np.asarray([0.08, 0.08, 0.018, 0.018, 0.018, 0.018, 0.018, 0.08])
            else:
                grippers[world] = np.asarray([0.08, 0.08, 0.0, 0.0, 0.0, 0.0, 0.08, 0.08])
            current = np.asarray(qpos[world], dtype=np.float64).copy()
            self._ik_data.qpos[:] = current
            mujoco.mj_forward(self.mj_model, self._ik_data)
            segment_start_position = np.asarray(
                self._ik_data.site_xpos[self.end_effector_site_id], dtype=np.float64
            ).copy()
            segment_start_gripper = float(np.sum(current[self.controlled_qpos_adr[7:9]]))
            frame_cursor = 0
            for stage_index, (target, duration) in enumerate(zip(targets, PANDA_STAGE_DURATIONS)):
                # The Panda position actuators settle slightly below the kinematic target under
                # gravity in MJWarp. Calibrate the expert waypoint, while keeping recorded state
                # and commanded state separate in Schema 2.0.
                compensated_target = target + np.asarray([0.018, -0.004, 0.024])
                subsegments = max(1, math.ceil(duration / 5))
                boundaries = np.linspace(0, duration, subsegments + 1, dtype=np.int32)
                previous_q = current[self.controlled_qpos_adr[:7]].copy()
                for subsegment in range(1, subsegments + 1):
                    alpha = subsegment / subsegments
                    intermediate_target = (
                        (1.0 - alpha) * segment_start_position + alpha * compensated_target
                    )
                    current = self._solve_site_ik(current, intermediate_target)
                    next_q = current[self.controlled_qpos_adr[:7]].copy()
                    local_start = int(boundaries[subsegment - 1])
                    local_end = int(boundaries[subsegment])
                    count = local_end - local_start
                    for local_index in range(count):
                        blend = (local_index + 1) / count
                        joint_trajectory[world, frame_cursor + local_start + local_index] = (
                            (1.0 - blend) * previous_q + blend * next_q
                        )
                    previous_q = next_q
                waypoints[world, stage_index] = current[self.controlled_qpos_adr[:7]]
                target_gripper = float(grippers[world, stage_index])
                gripper_trajectory[world, frame_cursor : frame_cursor + duration] = np.linspace(
                    segment_start_gripper, target_gripper, duration, dtype=np.float32
                )
                frame_cursor += duration
                segment_start_position = compensated_target
                segment_start_gripper = target_gripper
        self._panda_waypoints = waypoints
        self._panda_gripper_targets = grippers
        self._panda_joint_trajectory = joint_trajectory
        self._panda_gripper_trajectory = gripper_trajectory

    def _solve_site_ik(self, initial_qpos: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        assert self._ik_data is not None
        data = self._ik_data
        qpos = np.asarray(initial_qpos, dtype=np.float64).copy()
        if self._ik_target_matrix is None:
            data.qpos[:] = qpos
            mujoco.mj_forward(self.mj_model, data)
            self._ik_target_matrix = np.asarray(
                data.site_xmat[self.end_effector_site_id], dtype=np.float64
            ).reshape(3, 3).copy()
        target_matrix = self._ik_target_matrix
        arm_dofs = self.controlled_dof_adr[:7]
        for _ in range(400):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.mj_model, data)
            position_error = np.asarray(target_position) - data.site_xpos[self.end_effector_site_id]
            current_matrix = np.asarray(
                data.site_xmat[self.end_effector_site_id], dtype=np.float64
            ).reshape(3, 3)
            rotation_error = 0.5 * sum(
                np.cross(current_matrix[:, axis], target_matrix[:, axis]) for axis in range(3)
            )
            if np.linalg.norm(position_error) < 0.0015 and np.linalg.norm(rotation_error) < 0.025:
                break
            jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.mj_model, data, jacp, jacr, self.end_effector_site_id)
            orientation_weight = 0.25
            jacobian = np.vstack((jacp[:, arm_dofs], orientation_weight * jacr[:, arm_dofs]))
            error = np.concatenate((position_error, orientation_weight * rotation_error))
            delta_arm = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 7.5e-4 * np.eye(6), error
            )
            delta_arm = np.clip(delta_arm, -0.12, 0.12)
            velocity = np.zeros(self.mj_model.nv, dtype=np.float64)
            velocity[arm_dofs] = delta_arm
            mujoco.mj_integratePos(self.mj_model, qpos, velocity, 0.7)
            for joint_id, qpos_adr in zip(self.controlled_joint_ids[:7], self.controlled_qpos_adr[:7]):
                if self.mj_model.jnt_limited[joint_id]:
                    qpos[qpos_adr] = np.clip(qpos[qpos_adr], *self.mj_model.jnt_range[joint_id])
        return qpos

    def _apply_physics_randomization(self) -> None:
        body_mass = self.base_body_mass.copy()
        geom_friction = self.base_geom_friction.copy()
        if self.object_body_id is not None and self.randomization:
            body_mass[self.object_body_id] *= float(self.randomization.get("object_mass_scale", 1.0))
            geom_friction[self.object_geom_ids] *= float(
                self.randomization.get("object_friction_scale", 1.0)
            )
        if hasattr(self.model, "body_mass"):
            self.model.body_mass.assign(body_mass)
        if hasattr(self.model, "geom_friction"):
            self.model.geom_friction.assign(geom_friction)

    def _set_expert_grasp_constraint(self, active: bool) -> None:
        if self.grasp_weld_id < 0 or self._grasp_weld_active == active:
            return
        with wp.ScopedDevice(self.device):
            if active:
                eq_data = self.model.eq_data.numpy().copy()
                body_position = self.data.xpos.numpy()[0]
                body_quaternion = self.data.xquat.numpy()[0]
                body_matrix = self.data.xmat.numpy()[0]
                body1 = self.agent_body_id
                assert self.object_body_id is not None
                body2 = self.object_body_id
                anchor1 = np.asarray(body_matrix[body1], dtype=np.float64).reshape(3, 3).T @ (
                    body_position[body2] - body_position[body1]
                )
                q1_inverse = np.asarray(body_quaternion[body1], dtype=np.float64).copy()
                q1_inverse[1:] *= -1.0
                relative_quaternion = np.zeros(4, dtype=np.float64)
                mujoco.mju_mulQuat(
                    relative_quaternion,
                    q1_inverse,
                    np.asarray(body_quaternion[body2], dtype=np.float64),
                )
                eq_data[:, self.grasp_weld_id, 0:3] = 0.0
                eq_data[:, self.grasp_weld_id, 3:6] = anchor1
                eq_data[:, self.grasp_weld_id, 6:10] = relative_quaternion
                self.model.eq_data.assign(eq_data)
            eq_active = self.data.eq_active.numpy().copy()
            eq_active[:, self.grasp_weld_id] = active
            self.data.eq_active.assign(eq_active)
            wp.synchronize_device(self.device)
        self._grasp_weld_active = active

    def _initial_controls(self, qpos: np.ndarray) -> np.ndarray:
        controls = np.zeros((self.nworld, self.mj_model.nu), dtype=np.float32)
        if self.is_panda:
            command = np.column_stack(
                (qpos[:, self.controlled_qpos_adr[:7]], np.full(self.nworld, 0.08, dtype=np.float32))
            )
            controls[:, self.actuator_ids] = self._controls_from_logical_command(command)
        return controls

    def _logical_command_from_controls(self, controls: np.ndarray) -> np.ndarray:
        command = np.asarray(controls[:, self.actuator_ids], dtype=np.float32).copy()
        if self.is_panda:
            command[:, -1] *= 2.0
        return command

    def _controls_from_logical_command(self, command: np.ndarray) -> np.ndarray:
        controls = np.asarray(command, dtype=np.float32).copy()
        if self.is_panda:
            controls[:, -1] *= 0.5
        return controls

    def _normalize_position_command(self, command: np.ndarray) -> np.ndarray:
        span = np.maximum(self.action_command_high - self.action_command_low, 1e-6)
        return np.clip(
            2.0 * (command - self.action_command_low[None, :]) / span[None, :] - 1.0,
            -1.0,
            1.0,
        ).astype(np.float32)

    def _denormalize_position_action(self, action: np.ndarray) -> np.ndarray:
        span = self.action_command_high - self.action_command_low
        return (
            self.action_command_low[None, :] + 0.5 * (action + 1.0) * span[None, :]
        ).astype(np.float32)

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
        self.randomization = {
            "randomization_group": f"legacy-{self.seed % 20:02d}",
            "object_position": [float(value) for value in qpos[0, self.object_qpos_adr : self.object_qpos_adr + 3]],
            "goal_position": self.goal[0].astype(float).tolist(),
        }

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
        self.randomization = {
            "randomization_group": f"legacy-{self.seed % 20:02d}",
            "goal_position": self.goal[0].astype(float).tolist(),
        }

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
        self.randomization = {
            "randomization_group": f"legacy-{self.seed % 20:02d}",
            "start_position": start_xy[0].astype(float).tolist(),
            "goal_position": self.goal[0].astype(float).tolist(),
        }

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
        if self.policy == "learned":
            raise RuntimeError("learned policy steps require an explicit action from the model runtime")
        if self.is_panda:
            return self._panda_policy_action()
        if self.policy == "random":
            noise = self.rng.uniform(-1.0, 1.0, (self.nworld, self.action_dim)).astype(np.float32)
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

    def _panda_stage_index(self) -> int:
        cumulative = 0
        for index, duration in enumerate(PANDA_STAGE_DURATIONS):
            cumulative += duration
            if self.frame_id < cumulative:
                return index
        return len(PANDA_STAGE_DURATIONS) - 1

    def _panda_policy_action(self) -> np.ndarray:
        current_joint = self.data.qpos.numpy()[:, self.controlled_qpos_adr]
        if self.policy == "random":
            noise = self.rng.normal(0.0, 0.7, (self.nworld, self.action_dim)).astype(np.float32)
            self.random_action = np.clip(0.88 * self.random_action + 0.12 * noise, -1.0, 1.0)
            centered = np.column_stack(
                (current_joint[:, :7], np.sum(current_joint[:, 7:9], axis=1))
            )
            command = centered + self.random_action * np.asarray(
                [0.10, 0.10, 0.10, 0.08, 0.10, 0.10, 0.10, 0.015], dtype=np.float32
            )
            command = np.clip(command, self.action_command_low, self.action_command_high)
            return self._normalize_position_command(command)

        assert self._panda_joint_trajectory is not None and self._panda_gripper_trajectory is not None
        stage = self._panda_stage_index()
        self.stage[:] = stage
        if self.definition.mode == "panda_peg_insert":
            # The disclosed expert weld only stabilizes transport. It is released before
            # the physical insertion phase so it cannot force the peg through the socket.
            self._set_expert_grasp_constraint(3 <= stage <= 5)
        trajectory_index = min(self.frame_id, self._panda_joint_trajectory.shape[1] - 1)
        target = np.column_stack(
            (
                self._panda_joint_trajectory[:, trajectory_index, :],
                self._panda_gripper_trajectory[:, trajectory_index],
            )
        )
        current = np.column_stack(
            (current_joint[:, :7], np.sum(current_joint[:, 7:9], axis=1))
        )
        max_delta = np.asarray([0.050] * 7 + [0.006], dtype=np.float32)
        command = current + np.clip(target - current, -max_delta, max_delta)
        if self.policy == "recovery":
            # Short, disclosed disturbances move the robot away from the nominal
            # trajectory. Expert control resumes with enough time to demonstrate a
            # correction instead of turning the entire episode into a timeout.
            in_recovery_window = (
                230 <= self.frame_id < 240
                if self.definition.mode == "panda_pick_place"
                else (80 <= self.frame_id < 90) or (230 <= self.frame_id < 240)
            )
            if in_recovery_window:
                noise_std = 0.012 if self.definition.mode == "panda_pick_place" else 0.018
                command[:, :7] += self.rng.normal(0.0, noise_std, (self.nworld, 7)).astype(np.float32)
        elif self.policy == "perturbed":
            command[:, :7] += self.rng.normal(0.0, 0.012, (self.nworld, 7)).astype(np.float32)
            if self.seed % 4 == 0 and stage == 5:
                command[:, -1] = 0.08
        command = np.clip(command, self.action_command_low, self.action_command_high)
        return self._normalize_position_command(command)

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
            if action_array.shape == (self.action_dim,):
                action_array = np.tile(action_array, (self.nworld, 1))
            if action_array.shape != (self.nworld, self.action_dim):
                raise ValueError(
                    f"action must have shape ({self.action_dim},) or "
                    f"({self.nworld}, {self.action_dim}), got {action_array.shape}"
                )
            action_array = np.clip(action_array, -1.0, 1.0)
        action_array[self.terminated] = 0.0

        controls = np.zeros((self.nworld, self.mj_model.nu), dtype=np.float32)
        if self.robot.control_mode == "joint_position_target":
            command = self._denormalize_position_action(action_array)
            controls[:, self.actuator_ids] = self._controls_from_logical_command(command)
        else:
            qvel = self.data.qvel.numpy()[:, self.controlled_dof_adr]
            desired_velocity = action_array * self.definition.max_speed
            command = np.clip(
                4.0 * (desired_velocity - qvel),
                -self.definition.torque_limit,
                self.definition.torque_limit,
            ).astype(np.float32)
            controls[:, self.actuator_ids] = command

        with wp.ScopedDevice(self.device):
            self.data.ctrl.assign(controls)
            for _ in range(ACTION_REPEAT):
                mjw.step(self.model, self.data)
            wp.synchronize_device(self.device)

        self.frame_id += 1
        self.last_action = action_array
        self.last_action_command = command
        position = self._evaluation_positions()
        distance = self._task_distance(position)
        progress = self.previous_distance - distance
        reached = self._success_condition(position, distance)
        self.success_streak = np.where(reached, self.success_streak + 1, 0)
        self.success |= self.success_streak >= self.definition.success_frames
        out_of_bounds = (
            (np.abs(position[:, 0]) > 1.10)
            | (np.abs(position[:, 1]) > 0.85)
            | (position[:, 2] < -0.02)
        )
        timed_out = self.frame_id >= self.definition.max_frames
        self.terminated |= self.success | out_of_bounds | timed_out
        for index in range(self.nworld):
            if self.success[index]:
                self.termination_reason[index] = "success"
            elif out_of_bounds[index]:
                self.termination_reason[index] = "object_out_of_bounds"
            elif timed_out:
                if self.is_panda and self.stage[index] <= 2:
                    self.termination_reason[index] = "grasp_failed_or_timeout"
                elif self.definition.mode == "panda_peg_insert":
                    self.termination_reason[index] = "insertion_jam_or_timeout"
                else:
                    self.termination_reason[index] = "timeout"
        self.last_reward = (
            self.definition.progress_scale * progress
            - 0.002 * np.sum(action_array**2, axis=1)
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

    def _task_distance(self, position: np.ndarray) -> np.ndarray:
        if self.is_panda:
            return np.linalg.norm(position - self.goal, axis=1).astype(np.float32)
        return np.linalg.norm(position[:, :2] - self.goal[:, :2], axis=1).astype(np.float32)

    def _success_condition(self, position: np.ndarray, distance: np.ndarray) -> np.ndarray:
        if self.definition.mode == "panda_pick_place":
            return (
                (self.stage >= 6)
                & (distance < self.definition.goal_radius)
                & (position[:, 2] < 0.15)
            )
        if self.definition.mode == "panda_peg_insert":
            reached = np.zeros(self.nworld, dtype=np.bool_)
            for world_id in range(self.nworld):
                contacts = self._contacts_for_world(world_id)
                metrics = self._task_metrics_for_world(world_id, contacts)
                reached[world_id] = (
                    self.stage[world_id] >= 6
                    and metrics["axial_error_m"] < min(self.definition.goal_radius, 0.012)
                    and metrics["insertion_depth_m"] >= 0.035
                    and metrics["maximum_target_penetration_m"] <= 0.003
                    and metrics["object_contact_load_n"] <= 250.0
                )
            return reached
        return distance < self.definition.goal_radius

    def _contact_type(self, geom1: int, geom2: int) -> int:
        category1 = int(self.geom_category_ids[geom1])
        category2 = int(self.geom_category_ids[geom2])
        categories = {category1, category2}
        if categories == {1, 2}:
            return 1
        if categories == {2, 3}:
            return 2
        if categories == {2, 4}:
            return 3
        if category1 == 1 and category2 == 1:
            return 5
        if 1 in categories and (3 in categories or 4 in categories):
            return 4
        return 0

    def _contacts_for_world(self, world_id: int) -> dict[str, Any]:
        count_total = int(self.data.nacon.numpy()[0])
        world_ids = self.data.contact.worldid.numpy()[:count_total]
        indices = np.flatnonzero(world_ids == world_id)
        overflow = len(indices) > MAX_CONTACTS
        indices = indices[:MAX_CONTACTS]
        valid = np.zeros(MAX_CONTACTS, dtype=np.bool_)
        geom_pair = np.full((MAX_CONTACTS, 2), -1, dtype=np.int32)
        body_pair = np.full((MAX_CONTACTS, 2), -1, dtype=np.int32)
        category_pair = np.zeros((MAX_CONTACTS, 2), dtype=np.int8)
        type_id = np.zeros(MAX_CONTACTS, dtype=np.int8)
        is_target = np.zeros(MAX_CONTACTS, dtype=np.bool_)
        position = np.zeros((MAX_CONTACTS, 3), dtype=np.float32)
        normal = np.zeros((MAX_CONTACTS, 3), dtype=np.float32)
        distance = np.zeros(MAX_CONTACTS, dtype=np.float32)
        if len(indices):
            valid[: len(indices)] = True
            geom_pair[: len(indices)] = self.data.contact.geom.numpy()[indices]
            valid_pairs = geom_pair[: len(indices)]
            body_pair[: len(indices)] = self.geom_body_ids[valid_pairs]
            category_pair[: len(indices)] = self.geom_category_ids[valid_pairs]
            for contact_index, (geom1, geom2) in enumerate(valid_pairs.tolist()):
                contact_type = self._contact_type(int(geom1), int(geom2))
                type_id[contact_index] = contact_type
                is_target[contact_index] = contact_type in {1, 2}
            position[: len(indices)] = self.data.contact.pos.numpy()[indices]
            frames = self.data.contact.frame.numpy()[indices]
            normal[: len(indices)] = frames[:, 0, :]
            distance[: len(indices)] = self.data.contact.dist.numpy()[indices]
        return {
            "count": int(len(indices)),
            "valid": valid.tolist(),
            "geom_pair": geom_pair.tolist(),
            "body_pair": body_pair.tolist(),
            "category_pair": category_pair.tolist(),
            "type_id": type_id.tolist(),
            "is_target": is_target.tolist(),
            "position": position.tolist(),
            "normal": normal.tolist(),
            "distance": distance.tolist(),
            "overflow": bool(overflow),
        }

    def _task_metrics_for_world(
        self, world_id: int, contacts: dict[str, Any] | None = None
    ) -> dict[str, float]:
        object_position = self._object_positions()[world_id]
        axial_error = float(np.linalg.norm(object_position[:2] - self.goal[world_id, :2]))
        insertion_depth = 0.0
        if self.definition.mode == "panda_peg_insert" and self.insertion_entry_height > 0.0:
            object_bottom = float(object_position[2]) - self.object_bottom_offset
            insertion_depth = max(0.0, self.insertion_entry_height - object_bottom)

        contact_data = contacts if contacts is not None else self._contacts_for_world(world_id)
        valid = np.asarray(contact_data["valid"], dtype=np.bool_)
        distances = np.asarray(contact_data["distance"], dtype=np.float32)
        target = np.asarray(contact_data["type_id"], dtype=np.int8) == 2
        target_penetrations = -distances[valid & target]
        maximum_target_penetration = float(max(0.0, target_penetrations.max(initial=0.0)))

        object_contact_load = 0.0
        if self.object_body_id is not None:
            wrench = np.asarray(
                self.data.cfrc_ext.numpy()[world_id, self.object_body_id], dtype=np.float32
            )
            object_contact_load = float(np.linalg.norm(wrench[3:6]))
        return {
            "insertion_depth_m": float(insertion_depth),
            "axial_error_m": axial_error,
            "object_contact_load_n": object_contact_load,
            "maximum_target_penetration_m": maximum_target_penetration,
        }

    def state_dict(self, control_steps_per_second: float = 0.0) -> dict[str, Any]:
        qpos = self.data.qpos.numpy()[0]
        qvel = self.data.qvel.numpy()[0]
        body_position = self.data.xpos.numpy()[0]
        body_quaternion = self.data.xquat.numpy()[0]
        body_wrench = self.data.cfrc_ext.numpy()[0]
        sim_time = float(self.data.time.numpy()[0])
        joint_position = qpos[self.controlled_qpos_adr]
        joint_velocity = qvel[self.controlled_dof_adr]
        try:
            joint_effort = self.data.qfrc_actuator.numpy()[0, self.controlled_dof_adr]
        except (AttributeError, IndexError):
            joint_effort = np.zeros(len(self.controlled_dof_adr), dtype=np.float32)
        if self.end_effector_site_id >= 0:
            end_effector_position = self.data.site_xpos.numpy()[0, self.end_effector_site_id]
            site_matrix = np.asarray(
                self.data.site_xmat.numpy()[0, self.end_effector_site_id], dtype=np.float64
            ).reshape(9)
            end_effector_quaternion = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(end_effector_quaternion, site_matrix)
        else:
            end_effector_position = body_position[self.agent_body_id]
            end_effector_quaternion = body_quaternion[self.agent_body_id]
        gripper_width = float(np.sum(joint_position[-2:])) if self.is_panda else 0.0
        contacts = self._contacts_for_world(0)
        task_metrics = self._task_metrics_for_world(0, contacts)
        return {
            "frame_id": self.frame_id,
            "sim_time": sim_time,
            "scenario_id": self.task_name,
            "robot_id": self.robot.robot_id,
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
            "joint_position": joint_position.tolist(),
            "joint_velocity": joint_velocity.tolist(),
            "joint_effort": np.asarray(joint_effort, dtype=np.float32).tolist(),
            "end_effector_position": np.asarray(end_effector_position, dtype=np.float32).tolist(),
            "end_effector_quaternion": np.asarray(end_effector_quaternion, dtype=np.float32).tolist(),
            "gripper_width": gripper_width,
            "body_position": body_position.tolist(),
            "body_quaternion": body_quaternion.tolist(),
            "body_external_wrench": body_wrench.tolist(),
            "action": self.last_action[0].tolist(),
            "action_command": self.last_action_command[0].tolist(),
            "reward": float(self.last_reward[0]),
            "terminated": bool(self.terminated[0]),
            "success": bool(self.success[0]),
            "termination_reason": str(self.termination_reason[0]),
            "goal_position": self.goal[0].tolist(),
            "task_stage": int(self.stage[0]),
            "distance_to_goal": float(self.previous_distance[0]),
            "contacts": contacts,
            "task_metrics": task_metrics,
            "metrics": {
                "nworld": self.nworld,
                "success_count": int(np.count_nonzero(self.success)),
                "mean_reward": float(np.mean(self.last_reward)),
                "control_steps_per_second": float(control_steps_per_second),
                "physics_steps_per_second": float(control_steps_per_second * ACTION_REPEAT),
            },
        }

    def episode_metadata(self) -> dict[str, Any]:
        contact_semantics = {
            "body_id_to_name": {str(index): name for index, name in enumerate(self.body_names)},
            "geom_id_to_name": {str(index): name for index, name in enumerate(self.geom_names)},
            "geom_id_to_body_id": {
                str(index): int(body_id) for index, body_id in enumerate(self.geom_body_ids.tolist())
            },
            "geom_id_to_category": {
                str(index): CONTACT_CATEGORY_NAMES[int(category)]
                for index, category in enumerate(self.geom_category_ids.tolist())
            },
            "category_ids": {str(index): name for index, name in CONTACT_CATEGORY_NAMES.items()},
            "contact_type_ids": {str(index): name for index, name in CONTACT_TYPE_NAMES.items()},
            "object_geom_ids": self.object_geom_ids.astype(int).tolist(),
            "object_instance_ids": (self.object_geom_ids.astype(int) + 1).tolist(),
            "target_contact_type_ids": [1, 2],
        }
        return {
            "robot": {
                "id": self.robot.robot_id,
                "display_name": self.robot.display_name,
                "controlled_joint_names": list(self.robot.controlled_joints),
                "gripper_joint_names": list(self.robot.gripper_joint_names),
                "end_effector_body": self.robot.end_effector_body,
                "end_effector_site": self.robot.end_effector_site or "",
                "model_source": self.robot.model_source,
                "model_license": self.robot.model_license,
            },
            "action_spec": {
                "type": self.robot.control_mode,
                "names": list(self.robot.action_names),
                "units": list(self.robot.action_units),
                "normalized_low": [-1.0] * self.action_dim,
                "normalized_high": [1.0] * self.action_dim,
                "command_low": self.action_command_low.astype(float).tolist(),
                "command_high": self.action_command_high.astype(float).tolist(),
                "control_hz": 1.0 / CONTROL_DT,
            },
            "randomization": self.randomization,
            "contact_semantics": contact_semantics,
            "depth_spec": {
                "unit": "metre",
                "invalid_depth_value": 0.0,
                "valid_mask_dataset": "images/front_depth_valid",
            },
        }

    def model_spec(self) -> dict[str, Any]:
        geom_types = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
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
            geom_group = int(self.mj_model.geom_group[geom_id])
            # Menagerie uses group 2 for visual meshes and group 3 for collision
            # proxies. Unity must display the licensed visual assets, not the low-poly
            # collision geometry. Task primitives remain in group 0.
            if geom_group == 3:
                continue
            vertices: list[list[float]] | None = None
            triangles: list[int] | None = None
            if geom_type == "mesh":
                if geom_group not in {0, 2}:
                    continue
                mesh_id = int(self.mj_model.geom_dataid[geom_id])
                if mesh_id < 0:
                    continue
                vertex_start = int(self.mj_model.mesh_vertadr[mesh_id])
                vertex_count = int(self.mj_model.mesh_vertnum[mesh_id])
                face_start = int(self.mj_model.mesh_faceadr[mesh_id])
                face_count = int(self.mj_model.mesh_facenum[mesh_id])
                vertices = np.asarray(
                    self.mj_model.mesh_vert[vertex_start : vertex_start + vertex_count],
                    dtype=np.float32,
                ).tolist()
                triangles = np.asarray(
                    self.mj_model.mesh_face[face_start : face_start + face_count],
                    dtype=np.int32,
                ).reshape(-1).tolist()
            rgba = np.asarray(self.mj_model.geom_rgba[geom_id], dtype=np.float32).copy()
            if geom_type == "mesh" and rgba[3] < 0.1:
                rgba = np.asarray([0.55, 0.58, 0.62, 1.0], dtype=np.float32)
            geoms.append(
                {
                    "id": geom_id,
                    "name": _name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, f"geom_{geom_id}"),
                    "body_id": int(self.mj_model.geom_bodyid[geom_id]),
                    "type": geom_type,
                    "size": np.asarray(self.mj_model.geom_size[geom_id], dtype=np.float32).tolist(),
                    "position": np.asarray(self.mj_model.geom_pos[geom_id], dtype=np.float32).tolist(),
                    "quaternion": np.asarray(self.mj_model.geom_quat[geom_id], dtype=np.float32).tolist(),
                    "rgba": rgba.tolist(),
                    "group": geom_group,
                    "visual_role": "licensed_visual" if geom_group == 2 else "task_geometry",
                    "vertices": vertices,
                    "triangles": triangles,
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
            "max_frames": self.definition.max_frames,
            "goal_radius": self.definition.goal_radius,
            "camera_position": list(self.definition.camera_position),
            "camera_look_at": list(self.definition.camera_look_at),
            "camera_fov_degrees": self.definition.camera_fov_degrees,
            "camera_near_clip_m": self.definition.camera_near_clip_m,
            "robot": self.episode_metadata()["robot"],
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
