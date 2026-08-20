from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotDefinition:
    robot_id: str
    display_name: str
    controlled_joints: tuple[str, ...]
    actuator_names: tuple[str, ...]
    control_mode: str
    action_names: tuple[str, ...]
    action_units: tuple[str, ...]
    end_effector_body: str
    end_effector_site: str | None = None
    gripper_joint_names: tuple[str, ...] = ()
    home_qpos: tuple[float, ...] = ()
    model_source: str = ""
    model_license: str = ""

    @property
    def action_dim(self) -> int:
        return len(self.actuator_names)


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    display_name: str
    business_type: str
    description: str
    model_file: str
    mode: str
    robot_id: str
    agent_body: str
    object_joint: str | None
    object_body: str | None
    goal_radius: float
    success_frames: int
    max_speed: float
    torque_limit: float
    progress_scale: float
    max_frames: int
    camera_position: tuple[float, float, float]
    camera_look_at: tuple[float, float, float]
    official_reference: str
    camera_fov_degrees: float = 48.0
    camera_near_clip_m: float = 0.03

    def model_path(self, package_root: Path) -> Path:
        return package_root / "model" / self.model_file


ROBOTS: dict[str, RobotDefinition] = {
    "planar_arm_2d": RobotDefinition(
        robot_id="planar_arm_2d",
        display_name="二维双关节机械臂",
        controlled_joints=("shoulder", "elbow"),
        actuator_names=("shoulder_motor", "elbow_motor"),
        control_mode="normalized_joint_velocity",
        action_names=("shoulder_velocity", "elbow_velocity"),
        action_units=("rad/s", "rad/s"),
        end_effector_body="pusher",
    ),
    "planar_mobile_base": RobotDefinition(
        robot_id="planar_mobile_base",
        display_name="二维移动底盘",
        controlled_joints=("drive_x", "drive_y"),
        actuator_names=("drive_x_motor", "drive_y_motor"),
        control_mode="normalized_joint_velocity",
        action_names=("base_velocity_x", "base_velocity_y"),
        action_units=("m/s", "m/s"),
        end_effector_body="mobile_base",
    ),
    "franka_panda": RobotDefinition(
        robot_id="franka_panda",
        display_name="Franka Emika Panda + parallel gripper",
        controlled_joints=(
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ),
        actuator_names=(
            "actuator1",
            "actuator2",
            "actuator3",
            "actuator4",
            "actuator5",
            "actuator6",
            "actuator7",
            "actuator8",
        ),
        control_mode="joint_position_target",
        action_names=(
            "joint1_target",
            "joint2_target",
            "joint3_target",
            "joint4_target",
            "joint5_target",
            "joint6_target",
            "joint7_target",
            "gripper_width_target",
        ),
        action_units=("rad", "rad", "rad", "rad", "rad", "rad", "rad", "m"),
        end_effector_body="hand",
        end_effector_site="gripper",
        gripper_joint_names=("finger_joint1", "finger_joint2"),
        home_qpos=(0.0, -0.55, 0.0, -2.15, 0.0, 1.65, 0.78, 0.04, 0.04),
        model_source="https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda",
        model_license="Apache-2.0",
    ),
}


SCENARIOS: dict[str, ScenarioDefinition] = {
    "planar_push": ScenarioDefinition(
        scenario_id="planar_push",
        display_name="物流推运",
        business_type="仓储与非抓取式操作",
        description="二维机械臂把散件推送到指定收货区域，生成成功与失败操作轨迹。",
        model_file="planar_push.xml",
        mode="push",
        robot_id="planar_arm_2d",
        agent_body="pusher",
        object_joint="cube_free",
        object_body="cube",
        goal_radius=0.06,
        success_frames=3,
        max_speed=2.5,
        torque_limit=8.0,
        progress_scale=5.0,
        max_frames=120,
        camera_position=(-0.02, -0.88, 0.86),
        camera_look_at=(-0.04, 0.0, 0.035),
        official_reference="aloha_clutter / contact-rich manipulation",
    ),
    "precision_insert": ScenarioDefinition(
        scenario_id="precision_insert",
        display_name="精密装配",
        business_type="制造业装配与插入",
        description="机械臂把方形定位块推入带导向槽的装配工位，强调接触和毫米级到位。",
        model_file="precision_insert.xml",
        mode="insert",
        robot_id="planar_arm_2d",
        agent_body="pusher",
        object_joint="workpiece_free",
        object_body="workpiece",
        goal_radius=0.04,
        success_frames=5,
        max_speed=1.4,
        torque_limit=8.0,
        progress_scale=7.0,
        max_frames=120,
        camera_position=(-0.02, -0.88, 0.86),
        camera_look_at=(-0.02, 0.0, 0.035),
        official_reference="aloha_sdf / AlohaSinglePeg contact insertion",
    ),
    "quality_inspection": ScenarioDefinition(
        scenario_id="quality_inspection",
        display_name="质量检测到位",
        business_type="工业质检与多工位巡检",
        description="机械臂末端依次面向随机检测工位精确到位，用于视觉定位与动作控制数据。",
        model_file="quality_inspection.xml",
        mode="reach",
        robot_id="planar_arm_2d",
        agent_body="pusher",
        object_joint=None,
        object_body=None,
        goal_radius=0.035,
        success_frames=5,
        max_speed=2.5,
        torque_limit=8.0,
        progress_scale=4.0,
        max_frames=120,
        camera_position=(-0.02, -0.88, 0.86),
        camera_look_at=(-0.04, 0.0, 0.035),
        official_reference="myoarm / articulated reach control",
    ),
    "warehouse_navigation": ScenarioDefinition(
        scenario_id="warehouse_navigation",
        display_name="仓储移动机器人",
        business_type="AMR 路径规划与避障",
        description="二维移动底盘穿过货架通道到达随机库位，生成导航、避障和失败轨迹。",
        model_file="warehouse_navigation.xml",
        mode="navigate",
        robot_id="planar_mobile_base",
        agent_body="mobile_base",
        object_joint=None,
        object_body=None,
        goal_radius=0.07,
        success_frames=3,
        max_speed=0.9,
        torque_limit=10.0,
        progress_scale=3.0,
        max_frames=120,
        camera_position=(0.0, -1.12, 1.45),
        camera_look_at=(0.0, 0.0, 0.0),
        official_reference="unitree_g1_flat / unitree_g1_hfield locomotion",
    ),
    "panda_pick_place": ScenarioDefinition(
        scenario_id="panda_pick_place",
        display_name="Panda 抓取放置",
        business_type="标准单臂抓放与数据交付",
        description="Franka Panda 抓取随机方块并放入目标料盒，生成成功和结构化失败轨迹。",
        model_file="third_party/franka_emika_panda/pilot_pick_place.xml",
        mode="panda_pick_place",
        robot_id="franka_panda",
        agent_body="hand",
        object_joint="object_free",
        object_body="object",
        goal_radius=0.075,
        success_frames=4,
        max_speed=1.0,
        torque_limit=100.0,
        progress_scale=4.0,
        max_frames=470,
        camera_position=(0.95, -0.58, 0.68),
        camera_look_at=(0.50, 0.0, 0.15),
        official_reference="DROID / robomimic Lift and Can / MimicGen Stack",
        camera_fov_degrees=34.0,
    ),
    "panda_peg_insert": ScenarioDefinition(
        scenario_id="panda_peg_insert",
        display_name="Panda 精密插入",
        business_type="接触型装配与插入",
        description="Franka Panda 抓取定位销并插入随机孔位，记录接触、卡滞和成功轨迹。",
        model_file="third_party/franka_emika_panda/pilot_peg_insert.xml",
        mode="panda_peg_insert",
        robot_id="franka_panda",
        agent_body="hand",
        object_joint="peg_free",
        object_body="peg",
        goal_radius=0.025,
        success_frames=2,
        max_speed=1.0,
        torque_limit=100.0,
        progress_scale=6.0,
        max_frames=520,
        camera_position=(0.93, -0.56, 0.66),
        camera_look_at=(0.51, 0.02, 0.15),
        official_reference="MimicGen Square / Isaac Factory peg insertion",
        camera_fov_degrees=34.0,
    ),
}

DEFAULT_SCENARIO_ID = "planar_push"


def get_scenario(scenario_id: str | None) -> ScenarioDefinition:
    key = scenario_id or DEFAULT_SCENARIO_ID
    try:
        return SCENARIOS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported scenario: {key}; available={sorted(SCENARIOS)}") from exc


def get_robot(robot_id: str) -> RobotDefinition:
    try:
        return ROBOTS[robot_id]
    except KeyError as exc:
        raise ValueError(f"unsupported robot: {robot_id}; available={sorted(ROBOTS)}") from exc


def scenario_summaries() -> list[dict[str, str]]:
    return [
        {
            "scenario_id": item.scenario_id,
            "display_name": item.display_name,
            "business_type": item.business_type,
            "description": item.description,
        }
        for item in SCENARIOS.values()
    ]
