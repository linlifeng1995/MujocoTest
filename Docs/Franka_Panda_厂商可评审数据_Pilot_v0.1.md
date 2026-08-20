# Franka Panda 厂商可评审数据 Pilot v0.1

## 1. 交付目的

本 Pilot 的目标不是宣称已经完成 Sim-to-Real，而是向机器人厂商算法工程师提供一套能够：

1. 使用标准工具读取；
2. 明确理解每个字段的时序、单位和坐标系；
3. 用于行为克隆等基础训练；
4. 复现指定随机种子的仿真过程；
5. 独立校验文件完整性和数据质量；

的数据样例。

## 2. 当前已经实现

### 2.1 机器人与任务

- Franka Emika Panda 七轴机械臂；
- 对称平行夹爪，一个逻辑控制量 `gripper_width_target`；
- `panda_pick_place`：抓取方块并放入料盒；
- `panda_peg_insert`：抓取带法兰定位销并插入孔位；
- MJWarp 是唯一物理真源，Unity 只负责显示和采集图像；
- 机器人定义和任务定义已经分离，控制维度不再固定为两个执行器。

Panda 模型来自 MuJoCo Menagerie，固定到 commit
`da76818e269b82289eba39808e2fb91d679d6994`，许可证为 Apache-2.0。

### 2.2 专家轨迹

专家策略使用脚本状态机、带竖直姿态约束的逆运动学和稠密笛卡尔路径生成：

```text
接近物体 → 下降 → 闭合夹爪 → 抬升 → 高位搬运 → 对准目标 → 放置/插入
```

插入任务中的细长定位销容易在仿真接触离散化下从夹爪中滑落。Pilot 使用一个只在
“已完成物理抓取、开始搬运”阶段启用的 MJWarp weld equality 约束稳定专家演示，并在
到达插入阶段后终止。这个约束属于生成策略的一部分，必须在数据卡中披露；它不能被
解释为真实夹爪抓取性能证明。下一阶段应使用更高保真指尖材料、夹持力标定和真实 Panda
数据替换或校准此约束。

### 2.3 动作合同

主要动作：

```text
joint_position_target[7]  单位 rad
gripper_width_target[1]   单位 m
```

每一步同时保存：

- `actions/normalized`：归一化到 `[-1, 1]` 的训练动作；
- `actions/command`：发送给机器人控制器的物理量命令；
- `derived_actions/delta_end_effector_pose`：由相邻状态计算的末端位姿增量；
- `derived_actions/gripper_width`：派生的夹爪开度。

## 3. Schema 2.0

一个包含 `N` 个动作的 Episode 保存 `N+1` 个 observation：

```text
observation[0] --action[0]--> observation[1]
observation[1] --action[1]--> observation[2]
...
observation[N-1] --action[N-1]--> observation[N]
```

通过下面两个索引显式表达对应关系：

- `transition_observation_index = [0, 1, ..., N-1]`
- `transition_next_observation_index = [1, 2, ..., N]`

这消除了旧 Schema 1.1 中“同一行是动作后的状态，但又保存产生该状态的动作”造成的
隐式错一帧问题。

### 3.1 Observation 字段

- `timestamps`、`frame_id`
- `observations/qpos`、`observations/qvel`
- `observations/joint_position`、`joint_velocity`、`joint_effort`
- `observations/end_effector_position`、`end_effector_quaternion`
- `observations/gripper_width`
- `observations/body_position`、`body_quaternion`、`body_external_wrench`
- `observations/goal_position`、`task_stage`、`distance_to_goal`
- 固定容量接触数组：位置、法向、几何体对、穿透距离和 overflow 标记
- `images/front_rgb`
- `images/front_depth_m`
- `images/front_instance_id`
- `images/wrist_rgb`

### 3.2 Episode 元数据

- 机器人、任务、模型来源和许可证；
- controlled joint、action name、单位、上下限；
- seed 和实际随机化参数；
- MuJoCo、MJWarp、Warp、Unity 和协议版本；
- 物理与控制频率；
- 前视/腕部相机内参、安装关系和畸变模型；
- 成功状态和结构化终止原因。

复杂元数据使用 JSON 字符串保存在 HDF5 attributes 中。

## 4. 对外交付格式

### 4.1 原生 HDF5

信息最完整的数据真源，保存深度、实例分割、接触和随机化参数。

### 4.2 LeRobot v3

导出器生成：

```text
meta/info.json
meta/stats.json
meta/tasks.jsonl
meta/episodes/chunk-000/*.parquet
data/chunk-000/*.parquet
videos/observation.images.front/chunk-000/*.mp4
videos/observation.images.wrist/chunk-000/*.mp4
```

### 4.3 robomimic

导出器生成标准 `data/demo_0`、`obs`、`next_obs`、`actions`、`rewards`、`dones`
以及 train/valid mask 兼容结构。

## 5. 使用命令

```powershell
cd External\MJWarpDemo

# 校验原生 Episode
.\.venv\Scripts\python.exe -m mjwarp_demo.validate_dataset ..\..\Datasets

# 导出整个任务目录的两种厂商格式，并按 manifest 写入 split
.\.venv\Scripts\python.exe -m mjwarp_demo.exporters `
  ..\..\Datasets `
  ..\..\Delivery\<task> `
  --format all `
  --manifest ..\..\Artifacts\<task>\dataset_manifest.json

# 生成质量报告
.\.venv\Scripts\python.exe -m mjwarp_demo.quality_report `
  ..\..\Datasets `
  --output ..\..\Delivery\quality_report.json `
  --manifest ..\..\Artifacts\<task>\dataset_manifest.json
```

## 6. Pilot 生产规则

- 默认分辨率 `320×240`，控制频率 `20 Hz`，物理频率 `200 Hz`；
- 每任务目标 500 个 Episode，共 1,000 个；
- 当前任务的标准生产按钮生成 375 条专家轨迹和 125 条受控扰动轨迹；最终以实际
  `success` 和结构化 `termination_reason` 统计约 75% 成功、25% 失败，不以策略名称代替结果；
- 按 seed、对象配置和随机化组做 Episode 级 train/validation/test 划分；
- 禁止按帧随机拆分，禁止同一 seed 跨集合；
- 每个任务先输出 20 条轻量样例供厂商验收读取程序；
- 未经许可审计的公开轨迹不进入商业交付包。

## 7. 厂商评审清单

厂商收到样例后，应至少确认：

- 能否读取原生 HDF5、LeRobot Parquet/MP4 和 robomimic HDF5；
- action 含义是否符合其控制栈；
- Panda joint name、顺序、单位和坐标系是否可接受；
- 相机内外参是否满足视觉策略训练；
- 是否需要 ROS 2 MCAP、真实控制器输出或力矩/电流字段；
- 是否接受派生的末端增量动作；
- 其训练代码是否能在样例包上完成一次可复现的过拟合测试。

## 8. 已知边界

- 当前没有真实 Panda 数据闭环验证；
- 插入专家使用已披露的阶段性 weld 约束；
- 视觉端当前使用 Panda 低模碰撞网格代理，正式外观模型仍需按许可证和性能要求验收；
- 相机畸变默认为零，尚未通过真实标定板验证；
- 当前成功率只代表已运行的单 seed 冒烟验证，尚未完成 100 个未见 seed 的正式统计；
- LeRobot 导出遵循 v3 的 Parquet/MP4/metadata 组织方式，正式交付前仍应使用厂商目标版本的
  `lerobot` DataLoader 做一次端到端兼容性验收。
