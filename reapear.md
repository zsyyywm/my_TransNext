# 语义分割对比实验复现模板（TransNeXt → MambaVision 范式）

本文档是**可迁移的复现清单**：以本仓库中 **Mask2Former + TransNeXt（二分类 DataA/B）** 为「先复现的基线」，**MambaVision-Tiny + UPerNet（MMSeg）** 为「按同一课题要求对齐的第二条线」。以后换主干、换数据集时，可按章节逐项替换名称与路径，保留**流程与工程约定**。

---

## 1. 文档用途与适用边界

| 用途 | 说明 |
|------|------|
| 给未来的自己 / 协作者 | 快速对齐：**数据布局、指标定义、存盘规则、终端长什么样、Hook 从哪来**。 |
| 适用任务 | **二类语义分割**（背景 + 前景，如电线）；验证/测试关心 **前景 IoU** 及 P/R/F1、aAcc。 |
| 不适用 | 多类全景、检测、纯分类；或与 MMEngine 训练循环完全不同的框架（需自行改写「等价物」一节）。 |

---

## 2. 课题级硬需求（两条线必须一致）

以下在 TransNeXt 与 MambaVision 侧应**语义一致**（实现文件不同，但行为对齐）。

| 需求 | 约定 |
|------|------|
| **主监控指标** | 验证集 **前景 IoU**；在 MMSeg/MMDet 体系里常写作 **`val/IoU`**（由自定义 `BinaryForegroundIoUMetric` 把逐类 IoU 汇总成「前景类」一条）。 |
| **存 best 规则** | `CheckpointHook`：`save_best='val/IoU'`、`rule='greater'`；文件名通常形如 `best_val_IoU_*.pth`（以实际框架为准）。 |
| **周期 checkpoint** | 可关或极大 `interval`，避免磁盘占满；课题侧常用「**只关心 best**」。Mamba 配置里用极大 `interval` 配合 `save_last=False` 等，等价「不按 epoch 频繁落盘」。 |
| **早停** | 连续 **`patience` 次验证**（例如 50）监控指标**未刷新最优**则 `stop_training`；监控项与 `save_best` 一致（如 `val/IoU`，`rule='greater'`）。 |
| **日志与权重同目录** | 训练产生的 **`.log` / `json` / `vis_data`** 与 **best 权重**在同一「实验目录」下，便于打包、对比、写论文附录路径。TransNeXt：`CheckpointToLogDirHook`；Mamba：`wire_seg_hooks.CheckpointToLogDirHook`（将各 `CheckpointHook.out_dir` 指到 `runner.log_dir`）。 |
| **终端可读性** | 除框架默认一行日志外，有 **彩色进度 + 每 epoch 验证后指标表**（与 `mask2former/train.py` 中 `ConsoleSummaryHook` 对齐）。 |
| **曲线与表格** | 训练目录内 **`val_metrics.csv`**、**`train_curves.png`**、**前景趋势图**（如 `val_foreground_trends.png`），便于画论文图与对照 epoch。 |

---

## 3. 数据与目录（两条线共用约定）

### 3.1 布局（相对训练工程根目录）

```
DataA-B/
  DataA/
    image/train|val|test/*.jpg
    mask/train|val|test/*.png
  DataB/
    （同上）
```

- **二值 mask**：前景/背景两类；`reduce_zero_label` 与数据集 `metainfo.classes` 需与 head `num_classes=2` 一致。  
- **不要用 0 类当「忽略」** 除非你明确在 `LoadAnnotations` 与 loss 里处理 ignore index。

### 3.2 环境变量（便于换机器）

| 变量 | 作用 |
|------|------|
| `WIRE_SEG_DATAA_ROOT` / `WIRE_SEG_DATAB_ROOT` | 覆盖配置里默认的相对 `data_root`。 |
| `MAMBAVISION_TINY_PRETRAINED`（仅 Mamba 线） | 覆盖主干预训练权重路径；不配则用 `checkpoint/pretrained/*.pth.tar` 或官方 URL。 |

### 3.3 不外传

大体积 **`DataA-B`**、**权重**、**`checkpoint/` 整树** 不进 Git；`.gitignore` 必须写死。

---

## 4. 指标：`BinaryForegroundIoUMetric` 在做什么

**文件**：`binary_fg_metrics.py`（TransNeXt 在 `segmentation/mask2former/`；Mamba 在 `semantic_segmentation/`）。

**要点**：

- 继承 MMSeg 的 **`IoUMetric`**，在 `compute_metrics` 里从逐类 **IoU / Fscore / Precision / Recall** 中取出 **`foreground_index`**（默认 **1**）对应前景。  
- 写入 `metrics` 的键：`IoU`、`F1`、`Precision`、`Recall`、以及 **`aAcc`**（若上游表里有）；并带 **`prefix`**（如 `val`）→ 日志里为 **`val/IoU`** 等。  
- **`CheckpointHook.save_best='val/IoU'`** 依赖的就是这个 **prefix + 键名**。

**配置侧**：`iou_metrics` 需包含能算出 P/R/F 的项（如 `mIoU` 与 `mFscore`），否则 F1/P/R 不全。

---

## 5. 训练循环与「权重怎么保存」

### 5.1 MMEngine 里谁在写 `val/IoU`

1. **ValLoop** 跑完 → **Evaluator**（你的 `BinaryForegroundIoUMetric`）`compute_metrics` → 结果进入 **`runner.message_hub` / `metrics` 字典**。  
2. **`CheckpointHook.after_val_epoch`** 读取 `save_best` 对应标量，与历史最优比较，满足 `rule` 则写入 **`best_*.pth`**（路径由 `out_dir` + `filename_tmpl` 决定）。  
3. 因此 **`save_best` 字符串必须与 metric 产出完全一致**（含 `val/` 前缀）。

### 5.2 早停（`ValLossPatienceEarlyStopHook`）

- **配置**：`monitor` 与 `save_best` 一致（如 `val/IoU`），`rule` 与 checkpoint 一致（`greater`）。  
- **注入**：Mamba 在 `wire_seg_hooks.apply_wire_seg_training_options(cfg)` 里根据 `wire_seg_iou_early_stop_patience` 追加 Hook；TransNeXt 在 `train.py` 里 `_inject_iou_early_stop` 等逻辑等价。  
- **触发**：连续 `patience` 次验证未提升 → 设置 `train_loop.stop_training = True`，本轮结束后训练结束（不会无限跑满 `max_epochs`）。

### 5.3 日志目录与时间戳

- **`cfg.timestamp`**：用于 `work_dir` 下子目录名（如 `train_YYYYMMDD_HHMMSS`），与 MMEngine 默认纯时间戳冲突时，可用 **monkey-patch `Runner.setup_env`**（TransNeXt `train.py` 已做）把 `runner._timestamp` 写回配置里的字符串。  
- **Mamba**：配置里直接设 `timestamp = f'train_{_wire_ts}'` 等，与 `CheckpointToLogDirHook` 配合，使 **best 与 log 同目录**。

### 5.4 预训练（Mamba 线）

- **主干 `pretrained`**：优先本地文件（避免容器无网）；其次环境变量；最后 URL。  
- **与 `load_from` 区别**：`pretrained` 多在 **backbone 构建时** `init_weights`；`load_from` / `resume` 是 **整网 checkpoint + 优化器状态**，用于续训或官方全权重。

---

## 6. 训练时终端「会看到什么」、分别怎么来的

### 6.1 框架默认行（MMEngine `LoggerHook` + `LogProcessor`）

- 形如：`Epoch(train) [ep][i/N] lr: ... eta: ... time: ... memory: ... loss: ... decode.xxx ...`  
- **来源**：每个 iter 模型 `train_step` 把 loss 字典写入 **`message_hub`**；`LogProcessor.get_log_after_iter` 拼字符串。  
- **与课题表的关系**：这里的 **`decode.acc_seg`** 等是 **训练像素精度**，**不是** 验证集上那张「IoU(fg)/F1/…」表；不要混为一谈。

### 6.2 自定义彩色行与验证表（`ConsoleSummaryHook`）

- **文件**：TransNeXt → `segmentation/mask2former/train.py`；Mamba → `semantic_segmentation/training_viz_hooks.py`（逻辑对齐）。  
- **训练过程中**：`after_train_iter` 按 `interval` 打 `[训练] epoch=... | batch=... | Loss | LR | GPU Mem | ...`。  
- **每个 epoch 最后一个 train iter**：`_print_train_block` 打「本轮训练结束」+ 小表（Epoch / data_num / Loss / LR / …）。  
- **每个 epoch 训练段结束**：`after_train_epoch` 打 `[Epoch] ... 随后验证集…`。  
- **验证结束后**：`after_val_epoch` 打 `[验证]` + 两行表头/数值：  
  **`data_num | IoU(fg) | F1 | Precision | Recall | aAcc`**  
  数值来自 **`metrics` 字典**，用 `_pick_metric` 兼容 `val/IoU` 等带前缀的键；百分比由 `fmt_pct` 统一（≤1 则 ×100）。

### 6.3 曲线与 CSV（`PlotMetricsHook`）

- **采样**：按 `sample_interval` 从 `message_hub` 取 `train/loss`、LR 等；验证后追加 **val 行** 到 **`val_metrics.csv`** 并更新 **png**。  
- **注意**：`PlotMetricsHook` 里若用 `logging.INFO` 等，需 **`import logging`**，否则 `after_train_epoch` 保存图时会在 **验证之后** 的路径上报 `NameError`（已踩坑记录）。

---

## 7. TransNeXt 线（基线）— 复现时抓什么

| 项 | 建议 |
|------|------|
| 工程根 | `TransNeXt-main/.../segmentation/mask2former/`（日常 `cd` 此处）。 |
| 入口 | `train.py` / `test.py`（与 MMSeg 注册、`_base_` 路径一致）。 |
| 主干 | **TransNeXt-Tiny**；无 `swattention` 时走 **`transnext_native`**。 |
| 配置 | `configs/mask2former_transnext_tiny_dataa_512x512_iou.py`、`*_datab_*.py`。 |
| 依赖 | 以该目录 `requirements.txt` 为准；**MMCV 与 torch 版本需按官方 wheel 表对齐**。 |

详细环境版本见仓库根 **`README.md`**。

---

## 8. MambaVision 线 — 与基线对齐时抓什么

| 项 | 路径/说明 |
|------|-----------|
| 工程根 | `MambaVision-main/MambaVision-main/`（文档 **`MambaVision.md`**）。 |
| 训练 `cd` | **`semantic_segmentation/`**（所有 `python tools/train.py` 相对此目录）。 |
| 入口 | `tools/train.py`、`test.py`、`infer_one.py`、`smoke_test_wire_data.py`。 |
| 配置 | `configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py`、`*_datab_*.py`。 |
| 指标与 Hook | `binary_fg_metrics.py`、`wire_seg_hooks.py`、`training_viz_hooks.py`；`train.py` 里 **`import`** 以注册类并调用 `apply_wire_seg_training_options`。 |
| `sys.path` | `train.py`/`test.py` 已插入 **`semantic_segmentation` 根** + **`tools/`**，避免 `binary_fg_metrics` / `mamba_vision` 导入失败。 |
| 预训练 | `checkpoint/pretrained/mambavision_tiny_1k.pth.tar` 或环境变量；脚本 **`tools/download_mambavision_pretrained.py`**。 |

**与 TransNeXt 数字对齐的示例（可在 `MambaVision.md` 里维护结果表）**：

- `max_epochs=200`，`val_interval=1`，**早停 patience=50**，**`save_best='val/IoU'`**。

---

## 9. 命令模板（复制后替换路径/配置名）

```bash
# Conda（若 activate 失败先 source .../etc/profile.d/conda.sh）
conda activate <环境名>
```

**冒烟（不训练）**

```bash
cd <工程根>/<...>/semantic_segmentation   # 或 mask2former
python tools/smoke_test_wire_data.py <config.py>   # Mamba
# TransNeXt: python tools/smoke_test_dataa.py 等，以实际脚本为准
```

**训练**

```bash
cd <工程根>/.../semantic_segmentation
python tools/train.py configs/mamba_vision/mamba_vision_tiny_dataa_512x512_wire_iou.py
```

**测试**

```bash
python tools/test.py <config.py> <best_xxx.pth> --work-dir <输出目录>
```

**续训（恢复优化器）**

```bash
python tools/train.py <config.py> --resume --cfg-options load_from=<checkpoint.pth>
```

---

## 10. Git 与复现包

- **`.gitignore`**：必须忽略 **`checkpoint/`**、**`*.pth`**、**`*.pth.tar`**、**`DataA-B/`**、**`.venv/`**。  
- **提交物**：配置、自写 Hook/指标、**`MambaVision.md` / 本模板**、小脚本；不提交权重与数据。  
- **复现说明**：在 PR / README 中写清 **Python / torch / mmcv / mmseg 版本** 与 **一条完整 train 命令**。

---

## 11. 迁移到新项目时的检查表（Checklist）

- [ ] 数据目录与 `metainfo`、`pipeline`、`num_classes` 一致。  
- [ ] `val_evaluator` / `test_evaluator` 使用 **`BinaryForegroundIoUMetric`**，`prefix` 与 `save_best` 前缀一致。  
- [ ] `default_hooks.checkpoint`：`save_best`、`rule`、`out_dir`、`filename_tmpl`。  
- [ ] `custom_hooks`：`CheckpointToLogDirHook`、`ValLossPatienceEarlyStopHook`（monitor/rule/patience）、`ConsoleSummaryHook`、`PlotMetricsHook`。  
- [ ] `train.py`（或 Runner 入口）**最先**完成 **`sys.path`** 与 **`import binary_fg_metrics`**，再 **`Config.fromfile`**（否则 `custom_imports` 失败）。  
- [ ] 跑 1 epoch：终端是否出现 **验证表**；`work_dir` 下是否生成 **`best_*.pth`** 与 **`val_metrics.csv`**。  
- [ ] `.gitignore` 与远程仓库体积。  

---

## 12. 已知坑（节省下次时间）

| 现象 | 原因 | 处理 |
|------|------|------|
| `cfg.pretty_text` / yapf 报语法错 | 配置里把 **`os.environ.get` 赋给变量** 再进 `cfg`，序列化出 `<bound method ...>`。 | 用 **`os.getenv(...)` 内联** 或只保留字符串常量。 |
| 容器下载 HF 权重 `errno 99` | 外网/IPv6/代理问题。 | **本地下载**放到 `checkpoint/pretrained/`，或 `MAMBAVISION_TINY_PRETRAINED`。 |
| `torch.load` zip central directory | 权重文件损坏或非完整下载。 | `unzip -t` 检查；重新下载。 |
| `No module named 'binary_fg_metrics'`（test） | 仅把 `tools` 加进 `path`，配置在 **`semantic_segmentation` 根**。 | **`train.py`/`test.py` 同时 insert 仓库根**。 |
| `No module named 'ftfy'` | mmseg visualizer 依赖。 | `pip install ftfy`。 |
| mmcv `_ext` undefined symbol | **wheel 与 torch 主版本不匹配**。 | 按官方表重装 **匹配** 的 mmcv；必要时放宽 mmseg 版本断言或降级 torch（需整体评估）。 |

---

## 13. 相关文件索引（本仓库）

| 内容 | 路径 |
|------|------|
| TransNeXt 总备忘 | `README.md` |
| Mamba 课题备忘 | `MambaVision-main/MambaVision-main/MambaVision.md` |
| 本复现模板 | `EXPERIMENT_REPRODUCTION_TEMPLATE.md`（本文） |

---

*模板版本：随课题迭代可增删「检查表」与「已知坑」；数值结果表建议只在 `README.md` / `MambaVision.md` 中维护一处，本文保持方法不变。*
