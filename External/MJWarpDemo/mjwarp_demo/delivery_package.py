from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .preview_dataset import write_episode_preview


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == source.stat().st_size:
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_docs(
    delivery_dir: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    split_counts = report.get("split_leakage_audit", {}).get("split_counts", {})
    readme = f"""# Franka Panda Data Mini-Pilot v0.2

这是供机器人厂商算法工程师评审的仿真数据样例，不代表已经完成真实机器人 Sim-to-Real 验证。

- Episode：{report['episode_count']}（抓放与定位销插入）
- 成功率：{report['success_rate']:.1%}
- 控制频率：20 Hz
- 图像：320×240，前视 RGB + 腕部 RGB
- 划分：train={split_counts.get('train', 0)}，validation={split_counts.get('validation', 0)}，test={split_counts.get('test', 0)}

目录：

- `native_hdf5/`：信息最完整的 Schema 2.0 真源，包含深度、实例 ID、接触和随机化参数。
- `lerobot_v3/`：Parquet 状态/动作与 MP4 双相机视频。
- `robomimic/dataset.hdf5`：模仿学习兼容格式。
- `dataset_manifest.json`：Episode 划分和防泄漏分组。
- `quality_report.json`：逐 Episode 质量指标和门槛。
- `vendor_loader_example.py`：三种格式最小读取示例。
- `preview_dataset.py`：原生 HDF5 双相机预览生成器。
- `validate_delivery.py`：一键反向读取与 SHA256 校验。

验证命令：

```powershell
python validate_delivery.py .
python vendor_loader_example.py .
```
"""
    data_card = f"""# 数据卡

## 用途与边界

本数据用于验证厂商能否读取字段、训练模仿学习基线并评审数据质量。数据全部来自 MJWarp 仿真，Unity 负责视觉渲染与采集。当前不包含真实传感器噪声、真实标定漂移、机器人磨损或真实控制延迟，因此不得据此宣称真实机器人成功率。

## 数据构成

- 机器人：Franka Panda 7 自由度机械臂 + 平行夹爪
- 任务：`panda_pick_place`、`panda_peg_insert`
- 生成策略：脚本状态机 + 逆运动学 + 关节位置目标
- Episode：{report['episode_count']}
- Transition：{sum(int(item['transitions']) for item in report['episodes'])}
- 成功率：{report['success_rate']:.1%}
- Seed：0～4，每个任务各 5 条专家轨迹
- 许可：Panda 模型来自 MuJoCo Menagerie，Apache-2.0；详见 `THIRD_PARTY_LICENSES.md`

## 已验证质量

```json
{json.dumps(report['quality_gates'], ensure_ascii=False, indent=2)}
```

## 已知限制

- 当前只有专家成功轨迹，没有纳入结构化失败和受控扰动数据。
- `object_contact_load_n` 当前由 MJWarp `cfrc_ext` 提供但实测恒为 0，不能作为有效力控标签；后续需改为 contact constraint force。
- 仅 5 个 seed，规模适合接口和训练冒烟，不适合评估泛化上限。
- 未经真实 Panda 和厂商控制栈闭环验证。
"""
    fields = """# 字段说明

## Schema 2.0 时序语义

第 t 个 transition 明确表示：`observation_t -> action_t -> next_observation_t`。原生 HDF5 中 observation 数量为 N+1，action/reward/done 数量为 N。

## 关键字段

| 类别 | 字段 | 含义 |
|---|---|---|
| 状态 | `observations/joint_position` | Panda 关节位置，rad；夹爪关节为 m |
| 状态 | `observations/joint_velocity` | 关节速度 |
| 状态 | `observations/end_effector_position` | 末端世界坐标位置，m |
| 状态 | `observations/end_effector_quaternion` | 末端四元数，MuJoCo wxyz |
| 动作 | `actions/command` | 7 维关节位置目标 + 1 维夹爪宽度 |
| 动作 | `actions/normalized` | 归一化到 [-1, 1] 的动作 |
| 派生动作 | `derived_actions/delta_end_effector_pose` | 跨机器人接口参考，不是物理执行命令 |
| 图像 | `images/front_rgb` / `images/wrist_rgb` | uint8 RGB，320×240 |
| 深度 | `images/front_depth_m` | 米制线性深度，0 表示无效 |
| 分割 | `images/front_instance_id` / `images/wrist_instance_id` | uint16 实例 ID |
| 接触 | `contacts/*` | geom/body/category pair、位置、法向、距离和目标接触语义 |
| 任务 | `task_metrics/*` | 插入深度、轴向偏差、目标穿透等 |

相机内外参、坐标系、单位、随机化参数、机器人定义和接触 ID 映射保存在每个 HDF5 Episode 属性中。
"""
    (delivery_dir / "README.md").write_text(readme, encoding="utf-8")
    (delivery_dir / "DATA_CARD.md").write_text(data_card, encoding="utf-8")
    (delivery_dir / "FIELD_REFERENCE.md").write_text(fields, encoding="utf-8")


def package_vendor_delivery(
    paths: list[Path],
    delivery_dir: Path,
    manifest_path: Path,
    report: dict[str, Any],
    package_root: Path,
) -> None:
    delivery_dir.mkdir(parents=True, exist_ok=True)
    native_dir = delivery_dir / "native_hdf5"
    for source in paths:
        _link_or_copy(source, native_dir / source.name)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shutil.copy2(manifest_path, delivery_dir / "dataset_manifest.json")
    shutil.copy2(package_root / "model" / "third_party" / "LICENSES.md", delivery_dir / "THIRD_PARTY_LICENSES.md")
    shutil.copy2(package_root / "vendor_loader_example.py", delivery_dir / "vendor_loader_example.py")
    shutil.copy2(package_root / "validate_delivery.py", delivery_dir / "validate_delivery.py")
    shutil.copy2(package_root / "mjwarp_demo" / "preview_dataset.py", delivery_dir / "preview_dataset.py")
    _write_docs(delivery_dir, report, manifest)

    preview_dir = delivery_dir / "previews"
    seen_tasks: set[str] = set()
    for path in paths:
        record = next(item for item in manifest["records"] if item["path"] == path.name)
        task = str(record["scenario"])
        if task not in seen_tasks:
            write_episode_preview(path, preview_dir)
            seen_tasks.add(task)
