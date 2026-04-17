"""一键冒烟：加载 DataA 配置、构建数据集、可选 1 iter 训练（--train）。"""
import argparse
import os
import os.path as osp
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/mask2former_transnext_tiny_dataa_512x512_iou.py',
        help='config path relative to mask2former cwd')
    parser.add_argument(
        '--train',
        action='store_true',
        help='run 1 training iter (needs GPU + checkpoint)')
    args = parser.parse_args()

    root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from mmengine.config import Config
    from mmseg.registry import DATASETS

    cfg = Config.fromfile(args.config)
    print('[ok] config loaded, num_classes=', cfg.model.decode_head.num_classes)

    dr = cfg.train_dataloader.dataset.get('data_root', '')
    abs_dr = osp.abspath(dr)
    print('[ok] train data_root ->', abs_dr, 'exists=', osp.isdir(abs_dr))

    ds_cfg = cfg.train_dataloader.dataset.copy()
    ds_cfg['pipeline'] = []
    ds = DATASETS.build(ds_cfg)
    print('[ok] train dataset len =', len(ds))

    val_cfg = cfg.val_dataloader.dataset.copy()
    val_cfg['pipeline'] = []
    val_ds = DATASETS.build(val_cfg)
    print('[ok] val dataset len =', len(val_ds))

    if args.train:
        import binary_fg_metrics  # noqa: F401
        import transnext_native  # noqa: F401, register TransNeXt backbone
        from mmengine.runner import Runner
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=1, val_interval=1)
        cfg.default_hooks = dict(
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=999999),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='SegVisualizationHook'))
        cfg.train_dataloader['batch_size'] = 1
        cfg.val_dataloader['batch_size'] = 1
        runner = Runner.from_cfg(cfg)
        runner.train()
        print('[ok] 1-iter train finished')


if __name__ == '__main__':
    main()
