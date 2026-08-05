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

运行面板可以在 4 个 Scene 之间切换。Scene 被加入 `ProjectSettings/EditorBuildSettings.asset`，也可以直接从 `Assets/Scenes/` 打开。

## 快速开始

1. 使用 Unity `6000.3.11f1` 打开工程。
2. 打开 `Assets/Scenes/PlanarPushScene.unity` 或其他业务 Scene。
3. 如果需要重建 Python 环境，在 `External/MJWarpDemo/` 执行：

   ```powershell
   C:\Users\datamesh-u3d\.local\bin\uv.exe sync --python 3.12
   ```

4. 点击 Play。Unity 会自动启动 Python 服务并连接 `127.0.0.1:8765`。
5. 选择专家/随机策略、seed 和录制选项，然后运行单回合、生成 20 回合或运行 GPU 性能测试。

首次运行可能需要编译 CUDA 内核；后续启动会使用缓存。

独立启动后端：

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m mjwarp_demo.server --scenario precision_insert
```

## 数据输出

每个 episode 写入 `Datasets/*.h5`，写入期间使用 `.partial`。HDF5 schema `1.1` 在原有状态、动作、奖励、接触和三路图像基础上新增：

- `observations/goal_position`
- `observations/task_stage`
- `observations/distance_to_goal`
- 场景名称、业务类型和官方能力参考属性

视觉数据为 `320×240` RGB、米制线性深度和 `uint16` 实例 ID。每个物理状态必须收到同 frame ID 的 Unity 图像才能继续写入。

验证数据：

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m mjwarp_demo.validate_dataset ..\..\Datasets
```

`External/MJWarpDemo/example_pytorch.py` 展示了如何读取为 PyTorch tensor。

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
- 各 Scene 当前都是可验证的数据生成原型，不包含模型训练、遥操作或独立 Player 发布。
- MJWarp 官方说明其优势是高吞吐而非单环境低延迟，因此 Unity 预览固定使用 1 world，训练/性能路径再使用批量 world。
