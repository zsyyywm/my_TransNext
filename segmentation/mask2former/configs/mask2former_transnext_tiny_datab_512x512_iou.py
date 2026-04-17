# Mask2Former + TransNeXt-Tiny，在本地 DataB（二分类）上微调；**仅按验证集 IoU 存 best**
# 用法（在 mask2former 目录）：python train.py configs/mask2former_transnext_tiny_datab_512x512_iou.py
# 测试：python test.py configs/mask2former_transnext_tiny_datab_512x512_iou.py --best
#
# 与 DataA IoU 版（``mask2former_transnext_tiny_dataa_512x512_iou.py``）骨干/训练协议一致；唯 data_root 为 DataB。
# 成对配置：``mask2former_transnext_tiny_datab_512x512_val_loss.py``（另存 val/loss best）。

_base_ = ['./mask2former_transnext_tiny_160k_ade20k-512x512.py']

custom_imports = dict(
    imports=['mmdet.models', 'binary_fg_metrics'],
    allow_failed_imports=False)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))

# ----- DataB：目录结构与 DataA 相同（image/train|val|test + mask/...），jpg + png，0/1 -----
_datab_root = '../../../../DataA-B/DataB'
_datab_meta = dict(
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

_DATAB_TRAIN_BATCH = 4
train_dataloader = dict(
    batch_size=_DATAB_TRAIN_BATCH,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_datab_root,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_datab_meta,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_datab_root,
        data_prefix=dict(img_path='image/val', seg_map_path='mask/val'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_datab_meta,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=_datab_root,
        data_prefix=dict(img_path='image/test', seg_map_path='mask/test'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=_datab_meta,
        pipeline=test_pipeline))
val_evaluator = dict(
    type='BinaryForegroundIoUMetric',
    foreground_index=1,
    iou_metrics=['mIoU', 'mFscore'],
    nan_to_num=0,
    prefix='val')
test_evaluator = val_evaluator

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

_DATAB_MAX_EPOCHS = 200
_DATAB_VAL_INTERVAL_EPOCHS = 1
mask2former_iou_early_stop_patience = 50
_ckpt_time = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = 'data/checkpoints1'

log_processor = dict(by_epoch=True)
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
    val_interval=_DATAB_VAL_INTERVAL_EPOCHS)

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
        out_dir='data/checkpoints1',
        filename_tmpl=(
            f'finetune_transnext_tiny_datab_{_ckpt_time}_epoch{{}}.pth')),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
