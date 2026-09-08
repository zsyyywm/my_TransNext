# TransNeXt Mask2Former 电线二分类复现环境（2026-09-08）

> 本文写的是今天这次复现实际可用的环境、权重放置、训练与测试命令。
> 工程根：`/root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former/`

## 1. Conda 与 Python

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n transnext python=3.10 -y
conda activate transnext
python -m pip install --upgrade pip setuptools wheel
```

## 2. PyTorch 与基础依赖

```bash
pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

## 3. MMCV / MMEngine / MMSeg / MMDet

```bash
pip install mmcv==2.0.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmengine==0.7.3 mmsegmentation==1.0.0 mmdet==3.0.0 \
  timm==0.5.4 numpy==1.26.4 "matplotlib>=3.5.0" yapf==0.32.0
```

进入工程根后：

```bash
cd /root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former
pip install -r requirements.txt
```

## 4. 预训练权重

下载地址：

```text
https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true
```

文件名：

```text
transnext_tiny_224_1k.pth
```

放置位置：

```text
/root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former/transnext_tiny_224_1k.pth
```

另一个训练用 checkpoint：

```text
https://huggingface.co/DaiShiResearch/mask2former-transnext-tiny-ade/resolve/main/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth?download=true
```

文件名：

```text
mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth
```

放置位置：

```text
/root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth
```

## 5. 数据集位置

约定为：

```text
/root/sci/DataA-B/DataA
/root/sci/DataA-B/DataB
```

每个数据集内结构为：

```text
image/train|val|test
mask/train|val|test
```

## 6. 自检

```bash
cd /root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former
conda activate transnext
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import mmcv; import mmseg; print('mmcv', mmcv.__version__, 'mmseg', mmseg.__version__)"
python -c "import importlib.util; print('swattention installed:', importlib.util.find_spec('swattention') is not None)"
```

## 7. 训练命令

### DataA

```bash
cd /root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former
conda activate transnext
python train.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py
```

### DataB

```bash
python train.py configs/mask2former_transnext_tiny_datab_512x512_iou.py
```

## 8. 测试命令

### DataA

```bash
cd /root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former
conda activate transnext
python test.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py --best
```

### DataB

```bash
python test.py configs/mask2former_transnext_tiny_datab_512x512_iou.py --best
```

### 指定权重测试

```bash
python test.py \
  configs/mask2former_transnext_tiny_dataa_512x512_iou.py \
  /root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former/data/checkpoints1/train_<时间戳>/best_val_IoU_*.pth
```

`train_<时间戳>` 是训练时生成的目录名；`best_val_IoU_*.pth` 则是该次训练保存下来的最优权重文件。

## 9. 结果目录

训练输出：

```text
/root/sci/my_TransNext-main/my_TransNext-main/segmentation/mask2former/data/checkpoints1/train_<时间戳>/
```

当前文件里会有：

- `best_val_IoU_*.pth`
- `train_*.log`
- `val_metrics.csv`
- `train_curves.png`
- `val_foreground_trends.png`
