# 项目备忘

本目录含：导师提供的参考文献 PDF、数据集 `DataA-B`，以及 **`TransNeXt-main` 内的 Mask2Former + TransNeXt 分割工程**（主要开发与训练目录）。

如果你是来复现环境，先读 [`segmentation/mask2former/SETUP.md`](segmentation/mask2former/SETUP.md)；如果你是来直接看结果，读 [`segmentation/mask2former/RESULTS.md`](segmentation/mask2former/RESULTS.md)。

---

## 文档入口

| 文档 | 用途 |
|------|------|
| [`README.md`](README.md) | 项目总览、目录约定、入口链接 |
| [`segmentation/mask2former/SETUP.md`](segmentation/mask2former/SETUP.md) | 复现环境、权重放置、训练/测试命令 |
| [`segmentation/mask2former/RESULTS.md`](segmentation/mask2former/RESULTS.md) | 历史结果与测试指标 |

---

## 项目概况

- 任务：DataA / DataB 二类语义分割
- 主干：Mask2Former + TransNeXt-Tiny
- 框架：MMSeg 1.0 + MMDet 3.0
- 数据约定：`/root/sci/DataA-B/DataA`、`/root/sci/DataA-B/DataB`
- 运行方式：默认 `transnext_native`，不编译 `swattention`

---

## 目录约定

| 说明 | 路径 |
|------|------|
| 分割工程根 | `segmentation/mask2former/` |
| 训练产物 | `segmentation/mask2former/data/checkpoints1/train_<时间戳>/` |
| 测试产物 | `segmentation/mask2former/data/checkpoints1/test_<时间戳>/` |
| 预训练权重 | 工程根下的 `transnext_tiny_224_1k.pth`、`mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth` |

---

## Citation

```bibtex
@InProceedings{shi2023transnext,
  author    = {Dai Shi},
  title     = {TransNeXt: Robust Foveal Visual Perception for Vision Transformers},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {17773-17783}
}
```
