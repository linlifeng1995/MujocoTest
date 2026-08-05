from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    display_name: str
    business_type: str
    description: str
    model_file: str
    mode: str
    controlled_joints: tuple[str, str]
    agent_body: str
    object_joint: str | None
    object_body: str | None
    goal_radius: float
    success_frames: int
    max_speed: float
    torque_limit: float
    progress_scale: float
    camera_position: tuple[float, float, float]
    camera_look_at: tuple[float, float, float]
    official_reference: str

    def model_path(self, package_root: Path) -> Path:
        return package_root / "model" / self.model_file


SCENARIOS: dict[str, ScenarioDefinition] = {
    "planar_push": ScenarioDefinition(
        scenario_id="planar_push",
        display_name="物流推运",
        business_type="仓储与非抓取式操作",
        description="二维机械臂把散件推送到指定收货区域，生成成功与失败操作轨迹。",
        model_file="planar_push.xml",
        mode="push",
        controlled_joints=("shoulder", "elbow"),
        agent_body="pusher",
        object_joint="cube_free",
        object_body="cube",
        goal_radius=0.06,
        success_frames=3,
        max_speed=2.5,
        torque_limit=8.0,
        progress_scale=5.0,
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
        controlled_joints=("shoulder", "elbow"),
        agent_body="pusher",
        object_joint="workpiece_free",
        object_body="workpiece",
        goal_radius=0.04,
        success_frames=5,
        max_speed=1.4,
        torque_limit=8.0,
        progress_scale=7.0,
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
        controlled_joints=("shoulder", "elbow"),
        agent_body="pusher",
        object_joint=None,
        object_body=None,
        goal_radius=0.035,
        success_frames=5,
        max_speed=2.5,
        torque_limit=8.0,
        progress_scale=4.0,
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
        controlled_joints=("drive_x", "drive_y"),
        agent_body="mobile_base",
        object_joint=None,
        object_body=None,
        goal_radius=0.07,
        success_frames=3,
        max_speed=0.9,
        torque_limit=10.0,
        progress_scale=3.0,
        camera_position=(0.0, -1.12, 1.45),
        camera_look_at=(0.0, 0.0, 0.0),
        official_reference="unitree_g1_flat / unitree_g1_hfield locomotion",
    ),
}

DEFAULT_SCENARIO_ID = "planar_push"


def get_scenario(scenario_id: str | None) -> ScenarioDefinition:
    key = scenario_id or DEFAULT_SCENARIO_ID
    try:
        return SCENARIOS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported scenario: {key}; available={sorted(SCENARIOS)}") from exc


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
