# DataA + 同时按「验证集平均 loss」存 best（与仅 IoU 的 dataa 配置分离，避免混用）
#
# 与 ``mask2former_transnext_tiny_dataa_512x512_iou.py`` 完全一致，仅多开关 + 独立工作目录：
# - 本配置 **work_dir**：``data/checkpoints2``（训练日志、val 曲线、**仅**按 val/loss 的 best 权重在此，不与 IoU 单轨的 checkpoints1 混放）
# - 仅按验证 loss 存 best：``data/checkpoints2/train_<时间>/best_val_loss_*.pth``（与日志同目录）
#
# 用法（在 mask2former 目录）::
#   python train.py configs/mask2former_transnext_tiny_dataa_512x512_val_loss.py

_base_ = ['./mask2former_transnext_tiny_dataa_512x512_iou.py']

work_dir = 'data/checkpoints2'

# train.py 据此注入 ValAverageLossHook + 第二套 CheckpointHook(save_best=val/loss)
mask2former_enable_val_loss_best = True

# 按 val/loss 训练：总轮数上限 + 早停（连续若干轮验证 loss 未刷新最优则停，见 train.py）
_DATAA_MAX_EPOCHS = 200
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=_DATAA_MAX_EPOCHS,
        by_epoch=True)
]
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=_DATAA_MAX_EPOCHS,
    val_interval=1)
mask2former_val_loss_early_stop_patience = 50
