# 项目备忘

本目录含：导师提供的参考文献 PDF、数据集 `DataA-B`，以及 **`TransNeXt-main` 内的 Mask2Former + TransNeXt 分割工程**（作为主要开发与训练目录）。数据不外传，本文记录路径、环境与 Git 习惯。

---

## 项目总结

**任务与路线**：面向 **DataA / DataB** 的二类语义分割（细小前景 / 类不平衡场景），以 **Mask2Former + TransNeXt-Tiny** 为主干，在 **MMSeg 1.0 + MMDet 3.0** 上训练与评测；默认走 **`transnext_native`**（不编译 `swattention`），便于 Windows / 无自定义 CUDA 算子环境跑通。

**数据与配置**：`DataA`、`DataB` 位于 `DataA-B`；主要配置为 `mask2former/configs/mask2former_transnext_tiny_dataa_512x512_iou.py` 与 `mask2former_transnext_tiny_datab_512x512_iou.py`。指标为二分类下的 **前景 IoU**（日志中记为 `val/IoU` / `IoU_fg`）、F1、Precision、Recall、aAcc。

### 训练与测试结果（按验证集 IoU 选优，`checkpoints1`）

以下 run 均保存在 `segmentation/mask2former/data/checkpoints1/`，`save_best='val/IoU'`；测试使用对应 **best** 权重。

| 数据集 | 配置 | 训练目录 | 验证集最佳 IoU（前景） | 测试集 |
|--------|------|----------|------------------------|--------|
| **DataA** | `mask2former_transnext_tiny_dataa_512x512_iou.py` | `train_20260416_111438/` | **73.42%**（第 121 epoch） | IoU **65.52%**，F1 **79.17%**，P 73.78%，R 85.40%，aAcc 99.61%（`test_20260416_124510/test_20260416_124510.json`） |
| **DataB** | `mask2former_transnext_tiny_datab_512x512_iou.py` | `train_20260416_125132/` | **80.16%**（第 34 epoch） | IoU **78.84%**，F1 **88.17%**，P 88.94%，R 87.41%，aAcc 99.00%（`test_20260416_155628/test_20260416_155628.json`） |

**小结**：同一套 Mask2Former + TransNeXt-Tiny 下，**DataB 验证与测试 IoU 均明显高于 DataA**（DataA 更小目标/更难，验证–测试差距也更大）。逐 epoch 曲线见各 `train_*` 目录内 `val_metrics.csv` 与 `train_curves.png`。

### 历史实验（按验证 loss 选优，`checkpoints2`）

曾用 `*_val_loss.py` 配置、以验证 **loss** 为主监控做对照；指标摘录见 `segmentation/mask2former/EXPERIMENT_METRICS.md`（例如 DataA：验证 IoU 峰值约 **67.88% / 69.38%**，对应测试 IoU 约 **58.70% / 60.09%**；DataB 有短训 **75.18%** 验证 IoU 记录）。与上表 IoU 选优策略 **不可直接横向比大小**，仅作方法/监控项对照备忘。

---

## 目录约定

| 说明 | 路径 |
|------|------|
| 本备忘与数据根目录（本机原路径） | `C:\Users\34977\Desktop\大二下学习\sci\` |
| 仓库根目录（Linux 工作区示例） | `/root/sci/sci/`（本仓库内 `sci.md` 与 `TransNeXt-main` 同级） |
| **分割代码（日常在此改）** | `sci\TransNeXt-main\TransNeXt-main\segmentation\mask2former\` |
| 依赖版本文件 | 同上目录下的 `requirements.txt` |

**为何直接在 `TransNeXt-main` 里改**：与 MMSeg 注册表、`train.py` / `test.py`、config 相对路径一致，**少配 PYTHONPATH**，单人课题迭代更快。官方仓库更新不频繁时，合并上游不是刚需。

**上传 GitHub**：在此目录初始化或沿用已有 git，**新建你自己的远程仓库**（或先 fork 再只推你的分支即可）；与本地是否叫 `TransNeXt-main` 文件夹名无关，远程可以命名为例如 `uav-small-target-seg`。

**习惯建议**：

- 用独立分支做课题改动（如 `feat/small-target`），主分支保留可运行的基线。  
- **不要**把 `DataA-B`、大权重 `.pth` 提交进仓库；用 `.gitignore` 排除。  
- 若需标注「哪些是你改的」，在 README 里写一节 **Changes** 或保持 commit 信息清晰即可。

（若曾创建 `small_target_lab`，可删除或改作杂项目录，**不再作为必选结构**。）

---

## 运行策略：TransNeXt Native（纯 PyTorch）

- **不要**安装、不要编译 `swattention`（`swattention_extension`）。  
- `train.py` / `test.py` 在未检测到 `swattention` 包时，会自动加载 **`transnext_native`**，速度较慢但**不依赖 CUDA 自定义算子**，适合 **Windows** 先跑通。  
- 当前数据量级约 **160 张**：单卡迭代次数少时，用 native 版通常**可接受**；若以后有 Linux + 编译环境，再考虑 CUDA 扩展加速。

---

## Windows 环境配置（版本互锁）

以下与 `segmentation/mask2former/requirements.txt` 一致，并满足 **MMCV 2.0 + PyTorch 2.0** 的官方预编译 wheel 组合。

**建议：Python 3.10（64 位）。** Python 3.11 可能与旧版 mmcv/mmseg 轮子不完全匹配，未列在此方案中。

**显卡驱动**：安装较新的 NVIDIA 驱动即可；PyTorch 2.0.1 的 **cu118** 安装包自带 CUDA 运行时，**不要求**本机单独装完整 CUDA Toolkit（除非你要编译 swattention）。

**推荐用 Anaconda**：环境建在 `D:\Software\Anaconda\envs\` 下即可，与现有 `transformer`、`u_net` 并列；**环境名不要用已占用的名字**，例如新建名为 **`transnext`**（你已有 `transformer` 环境，避免混用）。

### 1. 用 Conda 新建环境并激活（CMD 或 Anaconda Prompt）

```text
conda create -n transnext python=3.10 -y
conda activate transnext
python -m pip install --upgrade pip
```

### 2. 安装 PyTorch 2.0.1 + torchvision 0.15.2（CUDA 11.8）

```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**仅 CPU 调试**（无 NVIDIA 或先验证 import）可用：

```powershell
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安装 MMCV 2.0.0（必须用 OpenMMLab 预编译 wheel，与 cu118 + torch2.0 对齐）

```powershell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

若你第 2 步用的是 **CPU 版** PyTorch，请改用 CPU 索引（无 GPU 时训练会很慢，仅建议做环境自检）：

```powershell
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
```

### 4. 安装其余依赖（与 requirements.txt 一致）

```powershell
pip install mmengine==0.7.3 mmsegmentation==1.0.0 mmdet==3.0.0 timm==0.5.4 "matplotlib>=3.5.0"
pip install numpy==1.26.4
pip install yapf==0.32.0
```

（`mmdet` 为 Mask2Former 配置里 `custom_imports` 所需。`numpy==1.26.4` 可避免 NumPy 2.x 与当前 torch/timm 组合的告警；若与已装的 `opencv-python` 冲突，可改为 `pip install "opencv-python<4.10"` 或卸掉独立 opencv，依赖 mmcv 自带接口。）  
若训练启动时报 `FormatCode() got an unexpected keyword argument 'verify'`，为 **yapf 过新** 与 mmengine 不兼容，执行：`pip install yapf==0.32.0`。

**不要**执行 `cd swattention_extension && pip install -e .`，以免装上 CUDA 版扩展；保持未安装 `swattention` 即可走 **transnext_native**。

### 5. 自检

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import mmcv; import mmseg; print('mmcv', mmcv.__version__)"
python -c "import importlib.util; print('swattention installed:', importlib.util.find_spec('swattention') is not None)"
```

最后一行应为 **`swattention installed: False`**，即使用 **transnext_native**。若为 `True`，说明误装了 CUDA 扩展，与本文「仅 native」策略不一致。

### 日常使用（训练前）

```text
conda activate transnext
cd /d C:\Users\34977\Desktop\大二下学习\sci\TransNeXt-main\TransNeXt-main\segmentation\mask2former
python train.py ...
```

PowerShell 里把第二行改成：`cd "C:\Users\34977\Desktop\大二下学习\sci\TransNeXt-main\TransNeXt-main\segmentation\mask2former"`。

**说明**：不必在项目里再建 `.venv`，**Conda 环境与 venv 二选一即可**；本文按 Anaconda 一条线写完，少折腾。

---

## 版本对照表（锁定兼容）

| 组件 | 版本 |
|------|------|
| Python | 3.10.x（推荐） |
| torch | 2.0.1 |
| torchvision | 0.15.2 |
| mmcv | 2.0.0（cu118/torch2.0 或 cpu/torch2.0 预编译） |
| mmengine | 0.7.3 |
| mmsegmentation | 1.0.0 |
| mmdet | 3.0.0（Mask2Former 依赖） |
| numpy | 建议 `1.26.4`（与 torch/timm 组合更稳） |
| timm | 0.5.4 |
| TransNeXt 注意力 | **native**（不装 swattention） |

---

## 后续（环境就绪后）

- 将 `DataA-B` 配成 MMSeg **CustomDataset** 或统一格式，在 **本仓库内** 新增/修改 config 的 `data_root`。  
- 训练在 Windows 上用 **`python train.py ...`**（参数以当前 `train.py` 为准），勿依赖未安装的 `bash dist_train.sh`，除非使用 Git Bash/WSL。  
- **验证集最优权重**：`train.py` 会在未写 `save_best` 的配置里默认补上 `save_best='mIoU'`（越大越好），训练过程中生成 `best_mIoU*.pth`。最终测试推荐：在 `mask2former` 目录执行  
  `python test.py <你的config.py> --best`  
  会在 `work_dir` 与 `CheckpointHook.out_dir` 下自动选用**修改时间最新**的 `best_*.pth`（即训练结束时的最优 val 权重）。  
- **DataA 数据集**：使用 `configs/mask2former_transnext_tiny_dataa_512x512_iou.py`（数据段已内联；勿将 `dataa.py` 再作为第二 `_base_`，否则会与 ADE 配置**重复键**报错）。在 `mask2former` 下：  
  `python train.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py`  
  `data_root` 默认为 `../../../../DataA-B/DataA`（相对 `mask2former` 指向 `sci\DataA-B\DataA`）。  
  **DataA 当前为按 epoch 训练**：在配置中改 `_DATAA_MAX_EPOCHS`、`_DATAA_VAL_INTERVAL_EPOCHS`、`_DATAA_TRAIN_BATCH`；**不按固定间隔存盘**，仅 **`save_best=mIoU`** 更新 `best_mIoU*.pth`，**不存 last**。  
- **冒烟测试**（不启动训练，只检查配置与样本数）：  
  `python tools/smoke_test_dataa.py`  
  期望输出含 `train dataset len`、`val dataset len`（例如 160 / 20）。

---

## 课题方向（备忘）

- 任务倾向：**语义分割**（细小目标 / 类不平衡可参考 PLE-Net 等：BCE+Dice、多尺度与 skip 侧增强等）。  
- 对比框架：导师要求 CNN / Transformer / Mamba 等多类方法时，本工程先作为 **Transformer 主干 + Mask2Former** 基线，再扩展其它主干或损失。
