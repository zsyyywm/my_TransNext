# DataB + 同时按「验证集平均 loss」存 best（与仅 IoU 的 datab 配置分离）
#
# 与 ``mask2former_transnext_tiny_datab_512x512_iou.py`` 完全一致，仅多开关 + ``data/checkpoints2``：
# - 仅按验证 loss 存 best：``data/checkpoints2/train_<时间>/best_val_loss_*.pth``（与日志同目录）
#
# 用法（在 mask2former 目录）::
#   python train.py configs/mask2former_transnext_tiny_datab_512x512_val_loss.py

_base_ = ['./mask2former_transnext_tiny_datab_512x512_iou.py']

work_dir = 'data/checkpoints2'

mask2former_enable_val_loss_best = True

_DATAB_MAX_EPOCHS = 200
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=_DATAB_MAX_EPOCHS,
        by_epoch=True)
]
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=_DATAB_MAX_EPOCHS,
    val_interval=1)
mask2former_val_loss_early_stop_patience = 50
