# MJWarp × Unity Embodied Data Demo

这是一个以 MJWarp 为唯一物理真源、Unity URP 为可视化和多模态采集端的具身数据演示。默认任务是二维双关节机械臂将方块推入随机目标区域。

## 快速开始

1. 使用 Unity `6000.3.11f1` 打开工程和 `Assets/Scenes/SampleScene.unity`。
2. 后端环境已经由项目内的 `External/MJWarpDemo/.venv` 管理；重建环境时在该目录执行：

   ```powershell
   C:\Users\datamesh-u3d\.local\bin\uv.exe sync --python 3.12
   ```

3. 点击 Play。运行时控制器会自动启动 Python 服务并连接 `127.0.0.1:8765`。MJWarp 第一次运行需要编译 CUDA 内核，可能等待约 1–2 分钟；后续启动会使用缓存。
4. 在左侧面板选择 Expert/Random、seed 和是否录制，然后运行单回合、生成 20 回合验收集或执行 GPU Benchmark。

如果希望单独启动后端：

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m mjwarp_demo.server
```

## 数据输出

完成的 episode 写入 `Datasets/*.h5`，写入过程中使用 `.partial`。每帧包含：

- `qpos`、`qvel`、body 位姿和外力；
- action、reward、terminated、success；
- 最多 16 个接触的 geom pair、位置、法线和距离；
- `320×240` RGB、米制线性深度和 `uint16` instance ID。

验证数据：

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m mjwarp_demo.validate_dataset ..\..\Datasets
```

`External/MJWarpDemo/example_pytorch.py` 展示了如何转换成 PyTorch tensor。

## 验证命令

```powershell
cd External\MJWarpDemo
.\.venv\Scripts\python.exe -m pytest -m "not integration"
.\.venv\Scripts\python.exe -m mjwarp_demo.evaluate --episodes 10
.\.venv\Scripts\python.exe -m mjwarp_demo.smoke --output <临时目录>
```

当前实机基线为 RTX 3070 Ti：专家策略 10 个 seed 中成功 9 个，随机策略成功 1 个。CUDA Graph 实测约为 1 world `2,594`、64 worlds `137,963`、256 worlds `477,691`、1024 worlds `1,574,462` physics steps/s，原始结果保存在 `External/MJWarpDemo/benchmark_rtx3070ti.json`。该基准不包含 Unity 渲染、GPU readback 或 HDF5 压缩时间。

## 设计边界

- Unity 不运行第二套 MuJoCo 物理；可视代理由后端根据 MJCF primitive geom 描述动态生成。
- 完整视觉数据只采集 selected world；`1/64/256/1024` worlds 用于无渲染批量吞吐演示。
- 当前面向 Unity Editor 技术验证，不包含训练算法、遥操作、复杂 mesh 机器人或独立 Player 打包。
