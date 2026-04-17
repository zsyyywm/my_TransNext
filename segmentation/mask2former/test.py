# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import logging
import importlib.util
import os
import os.path as osp


def is_installed(package_name):
    return importlib.util.find_spec(package_name) is not None


if is_installed('swattention'):
    print('swattention package found, loading CUDA version of TransNeXt')
    import transnext_cuda
else:
    print('swattention package not found, loading PyTorch native version of TransNeXt')
    import transnext_native

import binary_fg_metrics  # noqa: F401 — 与训练配置一致，注册 BinaryForegroundIoUMetric

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner


def _patch_mmengine_runner_use_cfg_timestamp():
    """与 train.py 相同：使 ``Runner.log_dir`` 使用 ``cfg.timestamp``（如 ``test_<时间>``）。"""
    if getattr(Runner.setup_env, '_mask2former_cfg_timestamp_patch', False):
        return
    _orig = Runner.setup_env

    def setup_env(self, env_cfg):
        _orig(self, env_cfg)
        cfg_ts = None
        if getattr(self, 'cfg', None) is not None:
            cfg_ts = self.cfg.get('timestamp')
        if cfg_ts is not None and str(cfg_ts).strip():
            self._timestamp = str(cfg_ts).strip()

    setup_env._mask2former_cfg_timestamp_patch = True
    Runner.setup_env = setup_env  # type: ignore[method-assign]


def _collect_best_checkpoints(search_root):
    """在目录顶层及一层子目录中查找 MMEngine 保存的 best 权重。"""
    if not search_root or not osp.isdir(search_root):
        return []
    roots = [search_root]
    hits = []
    patterns = (
        'best_IoU*.pth',
        'best_mIoU*.pth',
        'best_mDice*.pth',
        'best_*.pth',
        'best*.pth')
    for root in roots:
        for pat in patterns:
            hits.extend(glob.glob(osp.join(root, pat)))
        for sub in glob.glob(osp.join(root, '*')):
            if osp.isdir(sub):
                for pat in patterns:
                    hits.extend(glob.glob(osp.join(sub, pat)))
    # 去重、仅文件
    out = []
    seen = set()
    for p in hits:
        p = osp.abspath(p)
        if osp.isfile(p) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _filter_iou_best_paths(paths):
    """排除按 val/loss 保存的目录（``val_loss_best`` 或旧版 ``checkpoints_val_loss``），避免 ``--best`` 误选。"""
    out = []
    for p in paths:
        norm = osp.normpath(p).replace('\\', '/')
        if 'checkpoints_val_loss' in norm or 'val_loss_best' in norm:
            continue
        out.append(p)
    if not out:
        return paths
    iouish = [
        p for p in out
        if ('IoU' in osp.basename(p)) or ('mIoU' in osp.basename(p))
        or ('mDice' in osp.basename(p))]
    return iouish if iouish else out


def find_best_checkpoint_path(cfg):
    """在 cfg.work_dir 与 CheckpointHook.out_dir 下取修改时间最新的 best_*.pth（如按 val 前景 IoU 保存的 best_IoU*.pth）。"""
    dirs = []
    wd = cfg.get('work_dir')
    if wd:
        dirs.append(osp.abspath(wd))
    dh = cfg.get('default_hooks')
    if dh is not None:
        ch = dh.get('checkpoint') if isinstance(dh, dict) else None
        if isinstance(ch, dict) and ch.get('out_dir'):
            od = osp.abspath(ch['out_dir'])
            if od not in dirs:
                dirs.append(od)
    hits = []
    for d in dirs:
        hits.extend(_collect_best_checkpoints(d))
    hits = _filter_iou_best_paths(hits)
    if not hits:
        loc = ', '.join(dirs) if dirs else '(work_dir 未设置)'
        raise FileNotFoundError(
            f'未找到验证最优权重（best_mIoU*.pth 等）。已搜索: {loc}')
    hits.sort(key=lambda p: osp.getmtime(p), reverse=True)
    return hits[0]


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='checkpoint .pth；与 --best 二选一（或同时写路径时以 --best 为准）')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument(
        '--best',
        action='store_true',
        help='自动选用验证集最优权重：在 work_dir 与 checkpoint out_dir 下找最新的 best_*.pth（如 save_best=IoU 时为 best_IoU*.pth）')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if not args.best and not args.checkpoint:
        parser.error('请提供 checkpoint 路径，或使用 --best 自动选用验证最优权重')
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visulizer = cfg.visualizer
            visulizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 评测输出与日志：IoU 轨默认 checkpoints1；以 loss 为依据的权重/val_loss 配置一律归 checkpoints2
    _mask2former_root = osp.dirname(osp.abspath(__file__))
    _ckpt2_root = osp.join(_mask2former_root, 'data', 'checkpoints2')
    _test_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.work_dir is not None:
        cfg.work_dir = osp.abspath(args.work_dir)
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(_mask2former_root, 'data', 'checkpoints1')
    else:
        wd = cfg.work_dir
        cfg.work_dir = (
            osp.abspath(wd) if osp.isabs(wd) else osp.join(_mask2former_root, wd))
    # 未显式 --work-dir 时：loss best 权重路径或 val_loss 配置 → 强制与训练一致落在 checkpoints2
    if args.work_dir is None:
        force_ckpt2 = bool(cfg.get('mask2former_enable_val_loss_best'))
        if not force_ckpt2 and not args.best and args.checkpoint:
            ck_abs = osp.abspath(args.checkpoint)
            ck_norm = ck_abs.replace('\\', '/').lower()
            base = osp.basename(ck_abs)
            if 'best_val_loss' in base or '/data/checkpoints2/' in ck_norm + '/':
                force_ckpt2 = True
        if force_ckpt2:
            cfg.work_dir = _ckpt2_root
            print_log(
                f'test.py: 评测输出目录已设为 checkpoints2（与 loss 轨一致）:\n  {cfg.work_dir}',
                logger='current',
                level=logging.INFO)
    if not cfg.get('timestamp'):
        cfg.timestamp = f'test_{_test_ts}'

    _patch_mmengine_runner_use_cfg_timestamp()

    if args.best:
        if args.checkpoint:
            print_log(
                '已指定 --best，将忽略命令行中的 checkpoint 路径。',
                logger='current',
                level=logging.WARNING)
        ckpt = find_best_checkpoint_path(cfg)
        print_log(f'--best: 使用验证最优权重（按文件时间最新）\n  {ckpt}', logger='current')
        cfg.load_from = ckpt
    else:
        cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    print_log(
        '评测指标以配置为准；DataA 二分类配置下为前景 IoU、F1、Precision、Recall、aAcc。',
        logger='current')
    runner.test()


if __name__ == '__main__':
    main()
