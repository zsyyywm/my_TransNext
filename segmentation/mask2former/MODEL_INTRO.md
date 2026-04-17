# Mask2Former + TransNeXt-Tiny 分割模型说明

本文档介绍本目录下用于 **二分类语义分割**（背景 / 前景）的模型结构、配置文件、自定义模块与训练产物。

---

## 1. 总体概览


| 项目   | 说明                                                                     |
| ---- | ---------------------------------------------------------------------- |
| 任务   | 语义分割，**2 类**（`background`、`foreground`）                                |
| 框架   | MMSegmentation `EncoderDecoder`                                        |
| 分割头  | **Mask2Former**（query + mask，Transformer 解码）                           |
| 骨干   | **TransNeXt-Tiny**（`transnext_tiny`）                                   |
| 输入尺度 | 训练裁剪 **512×512**（与配置一致）                                                |
| 预训练  | 通常从 **ADE20K 上训练好的** Mask2Former+TransNeXt-Tiny 权重微调（见父配置 `load_from`） |


数据与训练超参以 `**mask2former_transnext_tiny_dataa_512x512_iou.py`** 为基准；注释中的「DataA」为历史命名，实际数据根路径指向 `**DataA-B/DataB**`。

---

## 2. 创新点说明（总览）

本节把「**相对基线多做了什么**、**解决什么问题**、**对应哪份配置/代码**」写清楚，便于写大创/专利/论文的「技术方案」段落。

### 2.1 场景与痛点（创新动机）


| 现象                     | 对分割的影响                                    |
| ---------------------- | ----------------------------------------- |
| 航拍前景（线、细小目标）**像素占比极低** | 类极不平衡，易整图预测为背景                            |
| 线状结构**细、易断**           | 普通卷积/下采样易丢高频细节                            |
| 背景复杂、纹理干扰              | 需要**通道/空间上的抑制与强调**，突出前景响应                 |
| 多尺度（远近、分辨率）            | 需在送入 Mask2Former 前**统一强化多尺度特征**，而非只依赖单一尺度 |


**基线模型**（`mask2former_transnext_tiny_dataa_512x512_iou.py`）已用强主干 + Mask2Former 处理上述问题的一部分，但 **不在 backbone 与解码头之间增加面向「细线 / 小目标」的归纳偏置**。

### 2.2 创新主张

在 **不改变各尺度通道数** 的前提下，在 **TransNeXt 与 Mask2Former 之间**插入可配置的 **多尺度创新 Neck（`MultiScaleInnovationNeck`）**，通过 **通道门控 + 空间门控 +（可选）条带/空洞/跨尺度注入/线几何增强** 提升细小、线状前景的分割质量；**损失函数仍沿用 Mask2Former 官方 CE + Dice**，创新集中在 **特征融合结构**，便于与基线做公平对比（同一数据、同一训练协议、仅 neck 开关不同）。

### 2.3 三条实验线（对照关系）


| 层级               | 配置文件                                                       | 相对基线的增量                                        | 适用说明                          |
| ---------------- | ---------------------------------------------------------- | ---------------------------------------------- | ----------------------------- |
| **基线**           | `mask2former_transnext_tiny_dataa_512x512_iou.py`              | 无自定义 neck                                      | 对照组；证明「仅微调」上限                 |
| **Innov**        | `mask2former_transnext_tiny_dataa_512x512_innov.py`        | **CFEMGate + SpatialGate7 + RCIF-Lite（高分辨率层）** | 默认创新组合：强调通道/空间 + 线状敏感卷积       |
| **Innov Strong** | `mask2former_transnext_tiny_dataa_512x512_innov_strong.py` | 在 Innov 上再开 **跨尺度注入、联合注意、线几何增强**               | 结构更重，需单独训练；权重与纯 Innov **不通用** |


### 2.4 各子模块「解决什么」（与文献思路对应）


| 模块                      | 解决什么        | 技术要点（便于写权利要求）                                          | 文献/方向参照（非照搬）                                        |
| ----------------------- | ----------- | ------------------------------------------------------ | --------------------------------------------------- |
| **CFEMGate**            | 通道冗余、背景通道干扰 | 全局 avg/max 池化 → 融合 → Sigmoid **通道权重**，逐通道乘到特征上         | CIFNet 等多尺度/上下文网络中的 **通道聚焦** 思路                     |
| **SpatialGate7**        | 空间上前景区域不突出  | 通道统计 → **7×7 卷积** 得空间 mask，与 CBAM 类机制同族                | 注意力/门控类方法                                           |
| **RCIFLite**            | 线太细、方向性强    | **水平/垂直条带 depthwise** + **空洞 depthwise** → 1×1 融合 + 残差 | 遥感/电力线类任务中常见的 **条带 + 空洞** 归纳                        |
| **CoarseToFineInject**  | 深层语义强但分辨率低  | **粗尺度特征上采样** 注入相邻 **更高分辨率** 层                          | 巡检/多尺度融合类论文中的 **MFAM 式跨层注入** 的简化                    |
| **JointAttentionLite**  | 通道与空间需联合抑制  | **通道注意 + 空间注意** 串联（实现接近 CBAM，求稳）                       | 联合注意（JA）类思路                                         |
| **LineGeometryEnhance** | 几何上线条方向多样   | **多方向条带 + 空洞** 增强线响应                                   | PLGAN 等强调 **线几何** 的方向；**本实现不含 GAN、不含 Hough 参数空间损失** |


### 2.5 代码与注册位置


| 内容            | 路径                                                                        |
| ------------- | ------------------------------------------------------------------------- |
| 子模块实现         | `custom_innovations/modules.py`                                           |
| Neck 组装与按尺度开关 | `custom_innovations/neck.py`                                              |
| 在模型里接 neck    | 对应 `*_innov*.py` 中 `model.neck` 与 `decode_head.in_channels` 与基线一致         |
| 模块注册          | 配置 `custom_imports`；`train.py` 内会 `import custom_innovations` 以便构建 Runner |


### 2.6 写材料时可用的「创新点」条目（可直接拆成条）

1. **即插即用多尺度 Neck**：插入位置固定为 backbone 输出与 Mask2Former 输入之间，**通道维 `[72,144,288,576]` 不变**，易迁移到其它分割头实验。
2. **面向类不平衡场景的显式门控**：通道门控（CFEMGate）+ 空间门控（SpatialGate7），抑制背景、增强前景响应。
3. **面向线状/细长结构的轻量卷积设计**：RCIFLite（条带 + 空洞 + 残差）；Strong 版再叠加 LineGeometryEnhance。
4. **可选跨尺度语义注入**：CoarseToFineInject，把粗尺度语义压到高分辨率支路（Strong）。
5. **实验设计清晰**：基线 / Innov / Innov Strong 三档，便于消融与专利「优选实施例」表述。

---

## 3. 网络结构

```
输入图像
    ↓
TransNeXt-Tiny（4 个尺度特征）
    ↓
[可选] MultiScaleInnovationNeck（见第 5 节）
    ↓
Mask2FormerHead
    ├─ MSDeformAttnPixelDecoder（多尺度可变形注意力像素解码）
    ├─ Mask2Former Transformer Decoder（多层 query）
    └─ 损失：分类 CE + Mask CE + Dice（与官方 Mask2Former 设定一致）
```

**多尺度通道**（与解码头 `in_channels` 一致）：

`[72, 144, 288, 576]`（从浅到深 / 高分辨率到低分辨率，与 MMSeg 惯例一致）。

---

## 4. 配置文件对照

均在 `configs/` 下，**在 `mask2former` 目录内** 执行 `train.py` / `test.py`。


| 文件                                                         | 说明                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| `mask2former_transnext_tiny_dataa_512x512_iou.py`              | **基线**：数据、按 epoch 训练、二分类头、无自定义 neck                          |
| `mask2former_transnext_tiny_dataa_512x512_innov.py`        | 在基线上增加 **MultiScaleInnovationNeck**（CFEM + 空间门 + RCIF-Lite）  |
| `mask2former_transnext_tiny_dataa_512x512_innov_strong.py` | 在 innov 上再开 **跨尺度注入、联合注意、细线几何增强**（结构更深，需单独训练，权重与纯 innov 不通用） |


**常用命令示例**：

```bash
# 训练（基线）
python train.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py

# 测试（加载该次 run 下 best mIoU）
python test.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py --best
```

innov 系列将配置名换成对应文件名即可。

---

## 5. 自定义模块（`custom_innovations/`）

与骨干解耦，通过 **neck** 插入 backbone 与 Mask2Former 解码器之间；**不改变各尺度通道数**。

### 5.1 `MultiScaleInnovationNeck`（`neck.py`）

对 4 个尺度的特征逐层处理，可选组合如下。


| 模块                      | 文件           | 作用简述                                                     |
| ----------------------- | ------------ | -------------------------------------------------------- |
| **CFEMGate**            | `modules.py` | 通道门控（CIFNet CFEM 思路）：全局 avg/max 池化后融合再 sigmoid           |
| **SpatialGate7**        | `modules.py` | 空间门控（CBAM 式）：通道维统计后经 7×7 卷积                              |
| **RCIFLite**            | `modules.py` | 轻量 RCIF：水平/垂直条带 depthwise + 空洞 depthwise，1×1 融合残差，利于线状结构 |
| **CoarseToFineInject**  | `modules.py` | 粗尺度特征上采样注入相邻高分辨率层（巡检类论文中 MFAM 的简化）                       |
| **JointAttentionLite**  | `modules.py` | 通道注意 + 空间注意（JA 思路，实现接近 CBAM，便于稳定训练）                      |
| **LineGeometryEnhance** | `modules.py` | 多方向条带 + 空洞卷积（PLGAN 几何归纳的轻量替代，**不含 GAN / Hough 损失**）      |


**基线 innov** 默认：`use_cfem` + `use_spatial_gate` + 在 **索引 0（最高分辨率）** 使用 RCIF-Lite。

**innov_strong** 额外开启：`use_coarse_fuse`、`use_joint_attention`、`use_line_geometry`（默认仍在索引 0 做细线增强）。

注册：在配置中 `custom_imports` 加入 `custom_innovations`；`train.py` 也会尝试 `import custom_innovations` 以注册模块。

---

## 6. 数据与增强

- **根路径**（相对 `mask2former`）：`../../../../DataA-B/DataB`
- **训练**：`image/train` + `mask/train`（jpg / png，mask 为 0/1）
- **验证 / 测试**：`image/val|test` + `mask/val|test`
- **训练流水线**：随机多尺度 → RandomCrop 512 → 翻转 → 光度畸变等
- **评估指标**：`IoUMetric`，含 **mIoU、mDice、mFscore** 等

---

## 7. 训练与 Checkpoint

- **调度**：按 **epoch** 的 `EpochBasedTrainLoop`（如默认 10 epoch，可在配置中修改）。
- **学习率**：`PolyLR`（按 epoch，`power=0.9` 等，见配置）。
- **CheckpointHook**（基线配置意图）：
  - `**save_best='mIoU'`**：验证集 mIoU **创新高** 时写入 `best_mIoU_*.pth`
  - `**save_last=False`**：训练结束**不会**自动保存 last
  - **间隔极大**：不按固定 epoch 额外存「中间 epoch 权重」
- `**CheckpointToLogDirHook`**（`train.py`）：把 IoU 等 best 的 `out_dir` 指到本次 `log_dir`（通常为 `data/checkpoints1/train_<时间>/`；`val/loss` best 在 `data/checkpoints2/val_loss_best/<时间>/`），与 `.log`、`vis_data` 同目录，便于归档。

**注意**：若训练在 **第一个完整 epoch 结束前的验证** 之前就中断，则 **可能从未触发验证**，从而 **不会产生任何 `.pth`**。需要中途权重时可考虑开启 `save_last=True` 或减小 `checkpoint` 的 `interval`（需与所用 MMEngine 版本 API 一致）。

---

## 8. 测试与日志产物

- `**python test.py ...**`：在测试集上跑 **TestLoop**，结束后可在可视化后端写出 **单行 JSON**（含 `mIoU`、`aAcc`、`mDice` 等），常出现在该次 run 目录下的 `*.json`。
- **训练过程**：`vis_data/<时间戳>.json` 多为 **JSONL**（一行一步的 `loss`、`lr`、`epoch`、`step`），与「测试汇总单行 JSON」用途不同，勿混用解读。

---

## 9. 环境提示（Windows）

配置中 `env_cfg` 使用 `**gloo`** 与 DataLoader `**spawn**`，便于单机 Windows 训练；在 `mask2former` 目录下运行脚本，保证相对路径（数据、`load_from`）正确。

---

## 10. 相关文件速查


| 路径                                                          | 内容                                       |
| ----------------------------------------------------------- | ---------------------------------------- |
| `train.py`                                                  | 训练入口、`CheckpointToLogDirHook`、其它自定义 Hook |
| `test.py`                                                   | 测试入口                                     |
| `custom_innovations/modules.py`                             | 各创新子模块实现                                 |
| `custom_innovations/neck.py`                                | `MultiScaleInnovationNeck`               |
| `configs/mask2former_transnext_tiny_160k_ade20k-512x512.py` | TransNeXt-Tiny 骨干与解码头通道、优化器分组等           |


---

*文档随仓库配置维护；若你修改了 epoch 数、数据路径或 checkpoint 策略，请同步更新本节或配置内注释。*