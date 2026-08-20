# MJWarp × Unity 具身训练数据 Demo

这是一个以 MJWarp 为唯一物理真源、Unity URP 为可视化和多模态采集端的具身数据演示。项目采用 `Unity 6000.3.11f1 + URP 17.3.0`，面向 Unity Editor 技术验证。

设计依据：

- [MJWarp 官方文档](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)：MJWarp 面向 NVIDIA GPU 上的大批量高吞吐仿真。
- [MJWarp 官方示例](https://github.com/google-deepmind/mujoco_warp)：覆盖 ALOHA 接触操作、Unitree 平地/地形移动、机械臂和柔性体等任务。

本项目第一轮选择 primitive 刚体任务，保证 MJCF 可以被 Unity 运行时代理完整复现；官方大型 mesh、柔性体和批量 GPU 渲染留作后续扩展。

## 业务场景

每类任务拥有独立 Unity Scene、MJCF、随机化、专家策略、奖励和成功条件：

| Unity Scene | 场景 ID | 业务用途 | 官方能力参考 |
|---|---|---|---|
| `PlanarPushScene` | `planar_push` | 物流推运、非抓取式操作 | `aloha_clutter` |
| `PrecisionAssemblyScene` | `precision_insert` | 方形定位块精密装配、导向插入 | `aloha_sdf` / `AlohaSinglePeg` |
| `QualityInspectionScene` | `quality_inspection` | 多工位质检到位 | `myoarm` / articulated reach |
| `WarehouseNavigationScene` | `warehouse_navigation` | AMR 导航与货架避障 | `unitree_g1_flat` / `unitree_g1_hfield` |
| `PandaPickPlaceScene` | `panda_pick_place` | Franka Panda 抓取放置 | MuJoCo Menagerie Panda / robomimic |
| `PandaPegInsertScene` | `panda_peg_insert` | Franka Panda 定位销插入 | MimicGen Square / Isaac Factory |

运行面板可以在 6 个 Scene 之间切换。Scene 被加入 `ProjectSettings/EditorBuildSettings.asset`，也可以直接从 `Assets/Scenes/` 打开。

## 快速开始

1. 使用 Unity `6000.3.11f1` 打开工程。
2. 打开 `Assets/Scenes/PlanarPushScene.unity` 或其他业务 Scene。
3. 如果需要重建 Python 环境，在 `External/MJWarpDemo/` 执行：

   ```powershell
   C:\Users\datamesh-u3d\.local\bin\uv.exe sync --python 3.12
   ```

4. 点击 Play。Unity 会自动启动 Python 服务并连接 `127.0.0.1:8765`。
5. 选择专家、随机或已加载的学习策略，设置 seed 和录制选项，然后运行单回合、批量采集或 GPU 性能测试。

首次运行可能需要编译 CUDA 内核；后续启动会使用缓存。

独立启动后端：

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m mjwarp_demo.server --scenario precision_insert
```

## 数据输出

每个 episode 写入 `Datasets/*.h5`，写入期间使用 `.partial`。当前 HDF5 Schema `2.0` 使用显式的 `N` 个 transition 与 `N+1` 个 observation：

- `transition_observation_index[t] = t`
- `transition_next_observation_index[t] = t + 1`
- `actions/normalized` 与 `actions/command`
- 前视 RGB/深度/实例分割和腕部 RGB
- Panda 关节、末端位姿、夹爪开度、接触和完整随机化元数据

视觉数据为 `320×240` RGB、米制线性深度和 `uint16` 实例 ID。每个物理状态必须收到同 frame ID 的 Unity 图像才能继续写入。

验证数据：

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m mjwarp_demo.validate_dataset ..\..\Datasets
```

`External/MJWarpDemo/example_pytorch.py` 展示了如何读取为 PyTorch tensor。

## 数据应用闭环

HDF5 不是直接放进 Unity 控制机器人，而是先训练成模型，再由 Python 后端在每个控制帧推理动作。第一版支持状态模仿学习，输入为 `qpos + qvel + goal_position + task_stage`，输出两个归一化动作；MJWarp 始终是唯一物理真源。

安装独立训练依赖，不影响仅运行后端的默认环境：

```powershell
cd External\MJWarpDemo
uv sync --python 3.12 --group training
```

建议先在 Unity 面板生成 `planar_push` 的 500 条专家轨迹和 500 条随机轨迹。现有少量数据只适合冒烟验证。然后执行：

```powershell
# 1. 校验数据并按 seed 划分 70%/15%/15%
.\.venv\Scripts\python.exe -m mjwarp_demo.training.index_data `
  --dataset-dir ..\..\Datasets `
  --output ..\..\Artifacts\planar_push\dataset_manifest.json `
  --scenario planar_push

# 2. 训练状态行为克隆策略
.\.venv\Scripts\python.exe -m mjwarp_demo.training.train_bc `
  --manifest ..\..\Artifacts\planar_push\dataset_manifest.json

# 3. 在 test 划分离线评估
.\.venv\Scripts\python.exe -m mjwarp_demo.training.evaluate_offline `
  --artifact ..\..\Artifacts\planar_push\<bc_run_id> `
  --split test

# 4. 在 100 个未见 seed 上运行专家/随机/学习策略闭环对比
.\.venv\Scripts\python.exe -m mjwarp_demo.training.evaluate_closed_loop `
  --artifact ..\..\Artifacts\planar_push\<bc_run_id> `
  --episodes 100
```

训练完成后重新进入 Play，点击“刷新模型列表”“加载模型”，再选择“学习策略”。面板会显示模型动作、推理延迟、累计奖励以及安全归零原因。模型加载失败、场景维度不匹配、输出含 NaN/Inf 或推理超过 100ms 时，后端不会执行异常动作。

每次训练在 `Artifacts/<scenario>/<run_id>/` 生成：

- `model.pt`：PyTorch 权重；
- `model_spec.json`：场景、字段顺序、输入输出维度和动作范围；
- `normalization.npz`：仅由训练集计算的归一化参数；
- `metrics.json`：训练和验证指标；
- `dataset_manifest.json`：数据文件、seed 划分和 SHA256 真源。

另外提供三个可运行的数据应用案例：

```powershell
# 专家+随机轨迹最近 10 帧 -> 成功/严重碰撞/越界/超时概率
.\.venv\Scripts\python.exe -m mjwarp_demo.training.train_risk --manifest <manifest.json>

# RGB -> 实例 ID 语义类别，报告验证 mIoU
.\.venv\Scripts\python.exe -m mjwarp_demo.training.train_segmentation --manifest <manifest.json>

# 当前状态+动作 -> 下一状态+奖励，报告单步和滚动 RMSE
.\.venv\Scripts\python.exe -m mjwarp_demo.training.train_dynamics --manifest <manifest.json>
```

`example_pytorch.py` 同时展示行为克隆、实例分割和动力学三种正确配对方式。Schema 2.0 直接使用 `observation[t] -> action[t] -> observation[t+1]`；读取器仍兼容旧 Schema 1.1。

## 验证命令

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m pytest -m "not integration"
.\.venv\Scripts\python.exe -m mjwarp_demo.evaluate --episodes 10
.\.venv\Scripts\python.exe -m mjwarp_demo.smoke --output <临时目录>
```

原始平面推运任务在 RTX 3070 Ti 上的基线为：专家策略 10 个 seed 成功 9 个，随机策略成功 1 个。CUDA Graph 实测约为 1 world `2,594`、64 worlds `137,963`、256 worlds `477,691`、1024 worlds `1,574,462` physics steps/s；不包含 Unity 渲染、GPU Readback 和 HDF5 压缩。

## 设计边界

- Unity 不运行第二套 MuJoCo 物理；运行时物体无 Collider。
- 完整多模态数据只采集 selected world；批量 world 用于纯物理吞吐。
- 状态策略已经支持 PyTorch 训练和 Python 后端闭环部署；风险、视觉分割和动力学案例当前用于离线训练与指标演示。
- 不包含在线强化学习、真实机器人控制、遥操作、Unity Sentis 或独立 Player 发布。
- MJWarp 官方说明其优势是高吞吐而非单环境低延迟，因此 Unity 预览固定使用 1 world，训练/性能路径再使用批量 world。
