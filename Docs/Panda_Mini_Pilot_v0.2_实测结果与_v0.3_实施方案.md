# Panda Mini-Pilot v0.2 实测结果与 v0.3 实施方案

## 1. 本轮结论

Panda Mini-Pilot v0.2 已经完成“厂商能读取、字段无歧义、质量可验证、数据可训练”的接口型样例，但尚未达到“学习策略能够闭环完成任务”的训练型数据集标准。

这两句话需要同时成立：

1. 数据产品链路已经跑通，可以拿给机器人厂商做格式和质量评审。
2. 当前只有 5 个 seed 的专家轨迹，不能支撑可靠的闭环模仿学习，更不能声称完成 Sim-to-Real。

## 2. 已完成的正式产出

### 2.1 最终数据批次

- 数据目录：`Datasets/PandaMiniPilotV02Final/`
- 任务：`panda_pick_place`、`panda_peg_insert`
- 每任务 5 个 seed，共 10 个 Episode
- 总 transition：3,883
- 控制频率：20 Hz
- 图像：320×240，前视 RGB、深度、实例 ID，腕部 RGB、实例 ID
- 成功率：10/10

### 2.2 厂商交付目录

- 交付目录：`Delivery/PandaMiniPilotV02Final/`
- 原生 Schema 2.0 HDF5：10 个 Episode
- LeRobot v3：10 个 Parquet、20 个 MP4
- robomimic：10 个 demo
- 训练/验证/测试：6/2/2，按任务、物体配置、场景配置和随机化组划分
- 包含 README、数据卡、字段说明、许可证、DataLoader、预览脚本和独立 validator
- SHA256 校验文件：65 个，全部通过

运行验证：

```powershell
cd Delivery\PandaMiniPilotV02Final
python validate_delivery.py .
python vendor_loader_example.py .
```

## 3. 数据质量实测

| 指标 | 结果 |
|---|---:|
| Native HDF5 validator | 10/10 通过 |
| Episode 成功率 | 100% |
| contact overflow | 0 |
| 机械臂动作饱和 | 0 |
| 插入目标最大穿透 | 2.43 mm |
| 插入前视可评审率最低值 | 93.6% |
| 抓放前视可评审率最低值 | 95.7% |
| 腕部 stage 2～6 目标可见率最低值 | 100% |
| split 泄漏检查 | 通过 |

相机调试给出的经验：

- 腕部相机不能放在 hand 中心线后方，会被 link7 完全遮挡。
- 腕部侧装后，抓取后的关键阶段目标可见率从 0 提升到 100%。
- Panda 前视相机 42° FOV 下目标 bbox 多数只有 0.7%～0.9%；收窄到 34° 后，可评审帧比例稳定超过 93%。

## 4. 训练基线实测

使用状态行为克隆，输入为机器人/物体状态、目标位置和任务阶段，输出为 7 维关节位置目标与 1 维夹爪命令。

### 4.1 离线结果

| 任务 | 最佳验证 MSE | test MAE | test MSE |
|---|---:|---:|---:|
| panda_pick_place | 0.000084 | 0.007500 | 0.000402 |
| panda_peg_insert | 0.000505 | 0.005442 | 0.000151 |

离线误差较低只表示模型能够拟合已收集轨迹附近的动作，不能证明模型能闭环控制。

### 4.2 未见 seed 闭环结果

每个任务在 seed 10000～10009 上比较专家、随机和学习策略：

| 任务 | 专家 | 随机 | 学习策略 |
|---|---:|---:|---:|
| panda_pick_place | 100% | 0% | 0% |
| panda_peg_insert | 100% | 0% | 0% |

两个学习策略都在所有回合跑到超时上限。安全层只各阻断 1 次，因此根因不是动作被安全层大量拒绝，而是：

1. 每个任务实际只有 3 条训练 Episode，状态覆盖严重不足。
2. 数据几乎全是单一路径的成功专家轨迹。
3. 模型产生小误差后进入训练集没有覆盖的状态，后续误差继续累积。
4. 纯行为克隆没有看到“偏离后如何回到轨迹”的动作。

## 5. 扰动策略实测

### 5.1 `perturbed`：失败数据来源

连续 0.012 rad 关节命令噪声，并包含按 seed 注入的失败：

- 抓放：1/5 成功
- 插入：0/5 成功
- 插入最大穿透：8.84 mm
- 终止类型包括 timeout、insertion jam、object out of bounds

结论：该策略适合生成结构化失败样本，不适合当作成功恢复轨迹。

### 5.2 `recovery`：成功恢复数据来源

新增短窗口扰动，随后恢复专家控制：

- 抓放：只在抓取后的运输阶段注入 0.012 rad 扰动，5/5 成功
- 插入：接近与运输阶段各注入一个 0.018 rad 短窗口扰动，5/5 成功
- 插入最大穿透低于 3 mm
- 动作饱和与 contact overflow 均为 0

结论：`recovery` 可以作为下一批训练数据的核心，不再只复制名义专家轨迹。

## 6. v0.3 下一阶段目标

下一目标不是立刻扩到 1,000 条，而是先证明“增加恢复数据后，闭环 BC 能从 0% 提升到可用水平”。

### 6.1 数据规模与配比

每个任务先生产 50 条，共 100 条：

| 数据类型 | 每任务数量 | 预期结果 | 用途 |
|---|---:|---|---|
| nominal expert | 20 | 成功 | 学习标准任务路径 |
| recovery | 20 | 成功 | 学习偏离后的修正动作 |
| perturbed failure | 10 | 失败 | 风险识别、失败分类和数据边界 |

建议先跑无图 physics preflight，再对通过门槛的 seed 进行 Unity 视觉采集，避免把明显损坏的轨迹编码成大体积视频。

### 6.2 必须新增的标签

每条恢复/失败 Episode 增加：

- `disturbance_profile`
- `disturbance_window_start/end`
- `disturbance_std`
- `failure_type`
- `recovery_started_frame`
- `recovered_to_nominal_frame`
- `recovery_duration_frames`
- `pre_disturbance_state_error`
- `maximum_post_disturbance_state_error`

### 6.3 训练策略

1. BC 训练只使用 nominal expert + recovery 成功轨迹。
2. perturbed failure 不直接作为动作监督，先用于训练风险/成功分类器。
3. 训练采样对 recovery 窗口加权，避免被大量平稳帧淹没。
4. 分别训练抓放与插入模型，不混合任务。
5. 保留 seed 10000 起的闭环评估集，不进入数据生产 seed。

### 6.4 v0.3 验收门槛

在 20 个未见 seed 上：

- 专家成功率：抓放 ≥90%，插入 ≥75%
- recovery 数据成功率：两个任务均 ≥80%
- 学习策略闭环成功率：抓放 ≥50%，插入 ≥30%
- 学习策略至少高于随机策略 30 个百分点
- 若达到上述门槛，再扩到每任务 200 条并挑战抓放 70%、插入 50%
- 所有 v0.2 数据质量门槛继续保持

## 7. 实施顺序

1. 将 recovery 扰动窗口和幅度完整写入 Schema 元数据与逐帧字段。
2. 增加结构化失败类型，不再只写 timeout。
3. 对 seed 0～99 跑 physics preflight，筛选出 20 expert、20 recovery、10 failure。
4. 用 Unity 采集 100 条视觉 Episode。
5. 重新生成 manifest、三种格式、质量报告和 checksum。
6. 训练 BC 与风险分类器，跑 20 个未见 seed 闭环。
7. 根据闭环结果决定是否扩大到 400 条或调整为 delta end-effector action。

## 8. 尚未解决的技术风险

- `object_contact_load_n` 当前实测恒为 0，MJWarp `cfrc_ext` 没有提供预期接触载荷；必须研究 `efc_force` 或 contact constraint force，修复前不能把该字段作为有效力控标签。
- 当前插入任务运输阶段使用已披露的 expert weld 稳定 peg；它在物理插入前释放，但仍会降低运输难度，数据卡必须持续说明。
- 当前动作是绝对关节位置目标。若增加 recovery 数据后闭环仍失败，应优先对比 `delta joint position` 与 `delta end-effector pose`，而不是只扩大网络。
- 还没有真实 Panda、真实相机标定或厂商控制栈验收。
