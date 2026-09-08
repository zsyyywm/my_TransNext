# Mask2Former + TransNeXt-Tiny，在本地 DataA（二分类）上微调；**仅按验证集 IoU 存 best**
# 用法（在 mask2former 目录）：python train.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py
# 测试：python test.py configs/mask2former_transnext_tiny_dataa_512x512_iou.py --best
#
# 与 ``mask2former_transnext_tiny_dataa_512x512_val_loss.py`` 成对：后者继承本配置并额外按 val/loss 存 best。
#
# 训练方式：按 epoch；最多 200 epoch；连续 50 次验证 val/IoU 未刷新最优则早停（见 train.py ``_inject_iou_early_stop``）。
# 权重：不按固定间隔存盘；按验证集「前景 IoU」更新 best_IoU*.pth（不另存 last）。输出在 data/checkpoints1/train_<时间戳>/。
#
# 注意：MMEngine 不允许两个 _base_ 同时定义同名键，故 DataA 数据配置写在本文件内。

_base_ = ['./mask2former_transnext_tiny_160k_ade20k-512x512.py']

custom_imports = dict(
    imports=['mmdet.models', 'binary_fg_metrics'],
    allow_failed_imports=False)

# Windows：多进程 DataLoader 用 spawn；单卡下 gloo 比 nccl 更通用
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))

# ----- DataA：jpg + png mask，0/1；data_root 相对 mask2former -----
_dataa_root = '../../../../DataA-B/DataA'
_dataa_meta = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]])
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations', reduce_zero_label=False)],
            [dict(type='PackSegInputs')]
        ])
]

# 按 epoch 训练须用「有限长」的 epoch：DefaultSampler + shuffle，勿用 InfiniteSampler
_DATAA_TRAIN_BATCH = 4
train_dataloader = dict(
    batch_size=_DATAA_TRAIN_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_dataa_root,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_dataa_meta,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_dataa_root,
        data_prefix=dict(img_path='image/val', seg_map_path='mask/val'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_dataa_meta,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_dataa_root,
        data_prefix=dict(img_path='image/test', seg_map_path='mask/test'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_dataa_meta,
        pipeline=test_pipeline))
val_evaluator = dict(
    type='BinaryForegroundIoUMetric',
    foreground_index=1,
    iou_metrics=['mIoU', 'mFscore'],
    nan_to_num=0,
    prefix='val')
test_evaluator = val_evaluator

# ----- 二分类头 -----
num_classes = 2
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 0.1])))

# ----- 按 epoch 训练；train.py 在「非 val_loss 模式」下读取早停轮数 -----
_DATAA_MAX_EPOCHS = 200
_DATAA_VAL_INTERVAL_EPOCHS = 1
# 连续若干次验证 val/IoU 未刷新最优则早停（与 val_loss 版逻辑一致，监控键不同）
mask2former_iou_early_stop_patience = 50
_ckpt_time = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
# 根目录；单次训练子目录 train_<时间> 由 train.py 写入 cfg.timestamp
work_dir = 'data/checkpoints1'

log_processor = dict(by_epoch=True)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=_DATAA_MAX_EPOCHS,
        by_epoch=True)
]
# 父链含 IterBasedTrainLoop 的 max_iters；深度合并会残留 max_iters，导致 EpochBasedTrainLoop 报错
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=_DATAA_MAX_EPOCHS,
    val_interval=_DATAA_VAL_INTERVAL_EPOCHS)

# 仅保留 val 前景 IoU 最优：interval 取极大 → 不按轮次额外存盘；save_last=False
# 写全 default_hooks，避免与 _base_ 合并时丢 Timer/ParamScheduler 等
# 注：train.py 会注册 CheckpointToLogDirHook，将权重实际保存到 work_dir/时间戳/（与 .log、曲线图同目录）
_CKPT_NO_PERIODIC = 10**9
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=_CKPT_NO_PERIODIC,
        save_best='val/IoU',
        rule='greater',
        save_last=False,
        # 实际保存路径由 train.py 的 CheckpointToLogDirHook 改到 log_dir（train_<时间>/）
        out_dir='data/checkpoints1',
        filename_tmpl=(
            f'finetune_transnext_tiny_dataa_{_ckpt_time}_epoch{{}}.pth')),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
