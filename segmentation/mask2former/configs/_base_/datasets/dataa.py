# 【已弃用为独立 _base_】MMEngine 不允许与 ade20k 并列作第二 _base_（重复键会报错）。
# 请使用 configs/mask2former_transnext_tiny_dataa_512x512_iou.py（数据段已内联）。
# 以下为备份参考，勿再在 _base_ 中引用本文件。
#
# DataA：image/{train,val,test} 下为 .jpg，mask/{train,val,test} 下为同名 .png（像素 0=背景, 1=前景）
# data_root 相对训练时工作目录 mask2former：向上 4 级到 sci，再进 DataA-B/DataA
# sci/TransNeXt-main/TransNeXt-main/segmentation/mask2former -> ../../../../DataA-B/DataA

dataset_type = 'BaseSegDataset'
data_root = '../../../../DataA-B/DataA'
metainfo = dict(
    classes=('background', 'foreground'),
    palette=[[0, 0, 0], [255, 255, 255]])
crop_size = (512, 512)
img_suffix = '.jpg'
seg_map_suffix = '.png'

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

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='image/train', seg_map_path='mask/train'),
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='image/val', seg_map_path='mask/val'),
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        metainfo=metainfo,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='image/test', seg_map_path='mask/test'),
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        metainfo=metainfo,
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
