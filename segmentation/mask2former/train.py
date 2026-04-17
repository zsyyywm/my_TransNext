# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import csv
import logging
import numbers
import importlib.util
import os
import os.path as osp

import torch


def is_installed(package_name):
    return importlib.util.find_spec(package_name) is not None


if is_installed('swattention'):
    print('swattention package found, loading CUDA version of TransNeXt')
    import transnext_cuda
else:
    print('swattention package not found, loading PyTorch native version of TransNeXt')
    import transnext_native

import binary_fg_metrics  # noqa: F401 — 注册 BinaryForegroundIoUMetric

from mmengine.config import Config, DictAction
from mmengine.dataset.utils import pseudo_collate
from mmengine.dist import is_main_process
from mmengine.fileio import FileClient, get_file_backend
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def _patch_mmengine_runner_use_cfg_timestamp():
    """MMEngine 的 ``Runner.setup_env`` 会用 ``strftime`` 覆盖 ``_timestamp``，导致
    ``log_dir`` 变成纯 ``YYYYMMDD_HHMMSS``，与配置里的 ``cfg.timestamp='train_…'`` 不一致。

    在原生 ``setup_env`` 执行完后，若 ``cfg`` 中带有非空 ``timestamp`` 字符串，则写回
    ``self._timestamp``，使 ``work_dir / train_<时间>`` 与文档约定一致。
    """
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


@HOOKS.register_module()
class CheckpointToLogDirHook(Hook):
    """将各 ``CheckpointHook`` 的 ``out_dir`` 指到 ``runner.log_dir``（与 .log、vis_data 同目录）。

    启用 ``mask2former_enable_val_loss_best`` 时仅保留按 ``val/loss`` 存 best，不再单独使用
    ``val_loss_best`` 子目录。
    """

    priority = 'LOWEST'

    def before_train(self, runner):
        logd = runner.log_dir
        for hook in runner.hooks:
            if hook.__class__.__name__ != 'CheckpointHook':
                continue
            sb = getattr(hook, 'save_best', None)
            target = logd
            if isinstance(sb, str) and sb.strip() == 'val/loss' and is_main_process():
                print_log(
                    f'CheckpointToLogDirHook: 按 val/loss 的 best 权重将保存到\n  {target}',
                    logger='current',
                    level=logging.INFO)
            hook.out_dir = target
            hook.file_client = FileClient.infer_client(
                hook.file_client_args, hook.out_dir)
            if hook.file_client_args is None:
                hook.file_backend = get_file_backend(
                    hook.out_dir, backend_args=hook.backend_args)
            else:
                hook.file_backend = hook.file_client  # 与 CheckpointHook.before_train 一致
        if is_main_process():
            print_log(
                f'CheckpointToLogDirHook: 周期 checkpoint（若有）与带 save_best 的 best 将保存到\n  {logd}',
                logger='current',
                level=logging.INFO)


@HOOKS.register_module()
class ValAverageLossHook(Hook):
    """在 **同一次** ValLoop 内对每批调用 ``model.loss``，与训练一致；epoch 末写入 ``val/loss``。

    需在 ``CheckpointHook(save_best='val/loss')`` 之前把标量放进 ``metrics``（本 Hook 使用
    ``HIGHEST`` 优先级，确保先于 ``CheckpointHook`` 的 ``after_val_epoch``）。
    """

    priority = 'HIGHEST'

    def before_val_epoch(self, runner):
        self._sum = 0.0
        self._count = 0
        self._warned = False

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, 'module') else model

    @staticmethod
    def _normalize_val_data_batch(data_batch):
        """ValLoop 的 batch 有时为 ``list``/``tuple`` of dict，需 ``pseudo_collate`` 成单 dict。"""
        if data_batch is None:
            return None
        if isinstance(data_batch, dict):
            return data_batch
        if isinstance(data_batch, (list, tuple)) and data_batch and isinstance(
                data_batch[0], dict):
            return pseudo_collate(list(data_batch))
        return None

    @staticmethod
    def _batch_inputs_tensor(inputs):
        """无 ``data_preprocessor`` 时的兜底：``list[Tensor]`` → (N,C,H,W)。"""
        if isinstance(inputs, torch.Tensor):
            return inputs
        if isinstance(inputs, (list, tuple)) and inputs:
            if all(isinstance(x, torch.Tensor) for x in inputs):
                return torch.stack(list(inputs), dim=0)
        return inputs

    @staticmethod
    def _total_loss_from_dict(loss_dict):
        if not isinstance(loss_dict, dict):
            return 0.0
        total = 0.0
        for v in loss_dict.values():
            if isinstance(v, torch.Tensor):
                total += float(v.detach().mean().cpu())
        return total

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        data_batch = self._normalize_val_data_batch(data_batch)
        if data_batch is None or not isinstance(data_batch, dict):
            return
        inputs = data_batch.get('inputs')
        samples = data_batch.get('data_samples')
        if inputs is None or samples is None:
            return
        if not isinstance(samples, list):
            samples = [samples]
        batch_for_prep = {'inputs': inputs, 'data_samples': samples}
        model = runner.model
        try:
            with torch.no_grad():
                # 与 train_step/val_step 一致：先 preprocessor（float 化、归一化、pad），再 loss
                prep = None
                if hasattr(model, 'data_preprocessor') and getattr(
                        model, 'data_preprocessor', None) is not None:
                    prep = model.data_preprocessor(batch_for_prep, training=False)
                if prep is not None and isinstance(prep, dict):
                    p_in = prep.get('inputs')
                    p_sp = prep.get('data_samples')
                else:
                    p_in = self._batch_inputs_tensor(inputs)
                    p_sp = samples
                if hasattr(model, 'loss'):
                    loss_dict = model.loss(p_in, p_sp)
                else:
                    loss_dict = self._unwrap(model).loss(p_in, p_sp)
            self._sum += self._total_loss_from_dict(loss_dict)
            self._count += 1
        except Exception as exc:
            if is_main_process() and not self._warned:
                print_log(
                    f'ValAverageLossHook: 计算 val batch loss 失败: {exc}',
                    logger='current',
                    level=logging.WARNING)
                self._warned = True

    def after_val_epoch(self, runner, metrics=None):
        s, c = float(self._sum), int(self._count)
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                ws = getattr(runner, 'world_size', 1)
                if ws and ws > 1:
                    dev = next(runner.model.parameters()).device
                    t = torch.tensor(
                        [s, float(max(c, 0))],
                        dtype=torch.float64,
                        device=dev)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    s, c = float(t[0].item()), int(round(t[1].item()))
        except Exception:
            pass
        if c <= 0:
            if is_main_process():
                print_log(
                    'ValAverageLossHook: 未累计到任何 val batch，跳过写入 val/loss',
                    logger='current',
                    level=logging.WARNING)
            return
        denom = max(1, c)
        avg = s / denom
        if metrics is not None and isinstance(metrics, dict):
            metrics['val/loss'] = float(avg)
        hub = getattr(runner, 'message_hub', None)
        if hub is not None and hasattr(hub, 'update_scalar'):
            try:
                hub.update_scalar('val/loss', float(avg))
            except TypeError:
                try:
                    hub.update_scalar('val/loss', float(avg), count=1)
                except Exception:
                    pass
        if is_main_process():
            print_log(
                f'ValAverageLossHook: 已写入 metrics/message_hub: val/loss = {avg:.6f} '
                f'(sum_batches={c})',
                logger='current',
                level=logging.INFO)


@HOOKS.register_module()
class ValLossPatienceEarlyStopHook(Hook):
    """按验证集 ``val/loss`` 早停：连续 ``patience`` 个 epoch 验证 loss 未刷新最优则结束训练。

    依赖 ``ValAverageLossHook``（更高优先级）先写入 ``metrics['val/loss']``。通过将
    ``runner.train_loop.stop_training = True`` 结束 ``EpochBasedTrainLoop``（与 MMEngine
    ``EarlyStoppingHook`` 所用机制一致）。
    """

    priority = 'NORMAL'

    def __init__(self, monitor='val/loss', patience=50, rule='less', min_delta=0.0):
        self.monitor = monitor
        self.patience = int(patience)
        self.rule = str(rule).lower()
        self.min_delta = float(min_delta)
        self._best = None
        self._epochs_no_improve = 0

    def after_val_epoch(self, runner, metrics=None):
        if not isinstance(metrics, dict) or self.monitor not in metrics:
            return
        try:
            cur = float(metrics[self.monitor])
        except (TypeError, ValueError):
            return
        if self._best is None:
            self._best = cur
            self._epochs_no_improve = 0
            return
        if self.rule == 'less':
            improved = cur < (self._best - self.min_delta)
        else:
            improved = cur > (self._best + self.min_delta)
        if improved:
            self._best = cur
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
        if self._epochs_no_improve < self.patience:
            return
        tl = getattr(runner, 'train_loop', None)
        if tl is not None and hasattr(tl, 'stop_training'):
            tl.stop_training = True
        if is_main_process():
            print_log(
                f'ValLossPatienceEarlyStopHook: 连续 {self.patience} 次验证 '
                f'「{self.monitor}」未优于当前最优 {self._best:.6f}，触发早停。',
                logger='current',
                level=logging.WARNING)


def _inject_val_loss_early_stop(cfg):
    """在 ``mask2former_enable_val_loss_best`` 时注入 ``ValLossPatienceEarlyStopHook``。"""
    if not cfg.get('mask2former_enable_val_loss_best'):
        return
    custom = list(cfg.get('custom_hooks') or [])
    if any(
            isinstance(h, dict)
            and h.get('type') == 'ValLossPatienceEarlyStopHook' for h in custom):
        return
    patience = int(cfg.get('mask2former_val_loss_early_stop_patience', 50))
    custom.append(
        dict(
            type='ValLossPatienceEarlyStopHook',
            monitor='val/loss',
            patience=patience,
            rule='less'))
    cfg.custom_hooks = custom


def _inject_iou_early_stop(cfg):
    """按 ``mask2former_iou_early_stop_patience`` 对主 ``save_best`` 指标（如 ``val/IoU``）做早停。

    与 val_loss 模式互斥（后者已有独立早停）；复用 ``ValLossPatienceEarlyStopHook``，仅改 monitor/rule。
    """
    if cfg.get('mask2former_enable_val_loss_best'):
        return
    patience = cfg.get('mask2former_iou_early_stop_patience')
    if patience is None:
        return
    patience = int(patience)
    if patience <= 0:
        return
    ck = (cfg.get('default_hooks') or {}).get('checkpoint')
    monitor = 'val/IoU'
    rule = 'greater'
    if isinstance(ck, dict):
        monitor = ck.get('save_best') or monitor
        rule = str(ck.get('rule', rule)).lower()
    custom = list(cfg.get('custom_hooks') or [])
    if any(
            isinstance(h, dict)
            and h.get('type') == 'ValLossPatienceEarlyStopHook'
            and h.get('monitor') == monitor for h in custom):
        return
    custom.append(
        dict(
            type='ValLossPatienceEarlyStopHook',
            monitor=monitor,
            patience=patience,
            rule=rule))
    cfg.custom_hooks = custom


@HOOKS.register_module()
class ConsoleSummaryHook(Hook):
    """训练：按间隔打印 loss/lr 等，并标明 epoch / batch 或 iter / max_iters；验证后打印指标表。"""

    priority = 'LOW'
    _MAX_REASONABLE_EPOCH_LEN = 100000

    def __init__(self, interval=1):
        self.interval = interval
        self._epoch_total = None
        self._max_epochs_cfg = None
        self._max_iters_cfg = None
        self._train_blocks = 0
        self._cache = {}

    @staticmethod
    def _c(text, color):
        colors = {
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'magenta': '\033[95m',
            'bold': '\033[1m',
            'white': '\033[97m',
        }
        end = '\033[0m'
        return f"{colors.get(color, '')}{text}{end}"

    def _cell(self, text, width, color):
        s = str(text)
        if len(s) > width:
            s = s[: max(1, width - 2)] + '..'
        s = s.ljust(width)
        return self._c(s, color)

    @staticmethod
    def _safe_scalar(message_hub, key):
        scalar = message_hub.get_scalar(key)
        if scalar is None:
            return None
        return scalar.current()

    @staticmethod
    def _safe_lr(runner):
        if not hasattr(runner, 'optim_wrapper'):
            return None
        lrs = runner.optim_wrapper.get_lr()
        if isinstance(lrs, dict) and len(lrs) > 0:
            first = next(iter(lrs.values()))
            if isinstance(first, (list, tuple)) and len(first) > 0:
                return float(first[0])
            if isinstance(first, (int, float)):
                return float(first)
        return None

    @staticmethod
    def _safe_image_size(data_batch):
        if not isinstance(data_batch, dict):
            return None
        inputs = data_batch.get('inputs', None)
        if isinstance(inputs, torch.Tensor):
            return int(inputs.shape[-1])
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0 and isinstance(
                inputs[0], torch.Tensor):
            return int(inputs[0].shape[-1])
        return None

    @staticmethod
    def _pick_metric(metrics, keys):
        for key in keys:
            if key in metrics:
                return metrics[key]
        for key in keys:
            for metric_key, metric_val in metrics.items():
                if metric_key.endswith(key):
                    return metric_val
        return None

    def before_train(self, runner):
        self._epoch_total = None
        self._max_epochs_cfg = None
        self._max_iters_cfg = None
        tl_obj = getattr(runner, 'train_loop', None)
        if tl_obj is not None:
            self._max_epochs_cfg = getattr(tl_obj, 'max_epochs', None)
            self._max_iters_cfg = getattr(tl_obj, 'max_iters', None)
        try:
            tl = len(runner.train_dataloader)
            mi = getattr(runner.train_loop, 'max_iters', None)
            if tl and 0 < tl < self._MAX_REASONABLE_EPOCH_LEN and mi is not None:
                self._epoch_total = max(1, (int(mi) + tl - 1) // tl)
        except (TypeError, ValueError, AttributeError):
            pass

    def _train_loop_type(self, runner):
        tl = getattr(runner, 'train_loop', None)
        return type(tl).__name__ if tl is not None else ''

    def _progress_tags(self, runner, batch_idx=None):
        """返回 (无颜色) 进度标签，便于训练行 / 验证行统一格式。"""
        loop_type = self._train_loop_type(runner)
        it = runner.iter + 1
        tags = []

        if loop_type == 'EpochBasedTrainLoop':
            ep = runner.epoch + 1
            me = self._max_epochs_cfg
            tags.append(f'epoch={ep}' + (f'/{me}' if me is not None else ''))
            if batch_idx is not None:
                try:
                    n = len(runner.train_dataloader)
                    if n and n < self._MAX_REASONABLE_EPOCH_LEN:
                        tags.append(f'batch={batch_idx + 1}/{n}')
                except TypeError:
                    pass
            tags.append(f'global_iter={it}')
        elif loop_type == 'IterBasedTrainLoop':
            mi = self._max_iters_cfg
            tags.append(
                f'iter={it}' + (f'/{mi}' if mi is not None else ''))
            me = self._max_epochs_cfg
            if me is not None:
                tags.append(f'epoch={runner.epoch + 1}/{me}')
        else:
            tags.append(f'global_iter={it}')

        return ' | '.join(tags)

    def _epoch_label(self, runner):
        cur = runner.epoch + 1
        if self._epoch_total:
            return f'{cur}/{self._epoch_total}'
        return str(cur)

    def _print_train_block(self, runner, batch_idx=None):
        c = self._cache
        if self._train_blocks > 0:
            print(flush=True)

        w_ep, w_dn, w_mem, w_loss, w_lr, w_img = 10, 12, 14, 14, 14, 10
        row1 = (
            self._cell('Epoch', w_ep, 'green')
            + self._cell('data_num', w_dn, 'yellow')
            + self._cell('GPU Mem', w_mem, 'yellow')
            + self._cell('Loss', w_loss, 'yellow')
            + self._cell('LR', w_lr, 'yellow')
            + self._cell('Image_size', w_img, 'yellow'))

        loss = c.get('loss')
        lr = c.get('lr')
        loss_str = f'{loss:.8f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        img = c.get('image_size')
        img_str = str(img) if img is not None else '?'
        mem = c.get('gpu_mem', 0.0)
        dn = c.get('data_num', '?/?')
        ep = self._epoch_label(runner)
        row2 = (
            self._cell(ep, w_ep, 'bold')
            + self._cell(dn, w_dn, 'white')
            + self._cell(f'{mem:.2f} MB', w_mem, 'white')
            + self._cell(loss_str, w_loss, 'white')
            + self._cell(lr_str, w_lr, 'white')
            + self._cell(img_str, w_img, 'white'))
        tag = self._progress_tags(runner, batch_idx=batch_idx)
        print(self._c(f'[本轮训练结束] {tag}', 'cyan'), flush=True)
        print(row1, flush=True)
        print(row2, flush=True)
        self._train_blocks += 1

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        if not is_main_process():
            return

        message_hub = runner.message_hub
        loss = self._safe_scalar(message_hub, 'train/loss')
        lr = self._safe_lr(runner)
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            gpu_mem = 0.0

        data_num = '?/?'
        total = None
        if hasattr(runner, 'train_dataloader'):
            try:
                total = len(runner.train_dataloader)
                data_num = f'{batch_idx + 1}/{total}'
            except TypeError:
                data_num = f'{batch_idx + 1}/?'

        image_size = self._safe_image_size(data_batch)
        self._cache = dict(
            loss=loss,
            lr=lr,
            gpu_mem=gpu_mem,
            data_num=data_num,
            image_size=image_size,
        )

        epoch_end = (
            total is not None and 0 < total < self._MAX_REASONABLE_EPOCH_LEN
            and (batch_idx + 1) == total)
        if epoch_end:
            self._print_train_block(runner, batch_idx=batch_idx)
            return

        if not self.interval or self.interval <= 0:
            return
        if (runner.iter + 1) % self.interval != 0:
            return
        image_size_str = str(image_size) if image_size is not None else '?'
        loss_str = f'{loss:.6f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        prog = self._progress_tags(runner, batch_idx=batch_idx)
        print(
            f"{self._c('[训练]', 'green')} {prog} | "
            f"{self._c('batch进度', 'yellow')} {data_num} | "
            f"{self._c('GPU Mem', 'magenta')} {gpu_mem:.2f} MB | "
            f"{self._c('Loss', 'red')} {loss_str} | "
            f"{self._c('LR', 'yellow')} {lr_str} | "
            f"{self._c('Img', 'blue')} {image_size_str}",
            flush=True)

    def after_val_epoch(self, runner, metrics=None):
        if not is_main_process() or not isinstance(metrics, dict):
            return

        vtag = self._progress_tags(runner, batch_idx=None)
        print(self._c(f'[验证] {vtag}', 'red'), flush=True)

        w_dn, w_iou, w_f1, w_p, w_r, w_a = 12, 12, 10, 12, 12, 10
        row3 = (
            self._cell('data_num', w_dn, 'red')
            + self._cell('IoU(fg)', w_iou, 'red')
            + self._cell('F1', w_f1, 'red')
            + self._cell('Precision', w_p, 'red')
            + self._cell('Recall', w_r, 'red')
            + self._cell('aAcc', w_a, 'red'))

        def fmt_pct(v):
            if not isinstance(v, numbers.Real):
                return 'N/A'
            x = float(v)
            if x <= 1.0 + 1e-6:
                x = x * 100.0
            return f'{x:.2f}'

        miou = self._pick_metric(metrics, ['IoU', 'mIoU'])
        mf1 = self._pick_metric(metrics, ['F1', 'mFscore', 'mF1'])
        mp = self._pick_metric(metrics, ['Precision', 'mPrecision'])
        mr = self._pick_metric(metrics, ['Recall', 'mRecall'])
        aacc = self._pick_metric(metrics, ['aAcc'])

        val_total = None
        try:
            val_total = len(runner.val_dataloader)
        except TypeError:
            pass
        dn_val = f'{val_total}/{val_total}' if val_total else '?/?'

        row4 = (
            self._cell(dn_val, w_dn, 'white')
            + self._cell(fmt_pct(miou), w_iou, 'white')
            + self._cell(fmt_pct(mf1), w_f1, 'white')
            + self._cell(fmt_pct(mp), w_p, 'white')
            + self._cell(fmt_pct(mr), w_r, 'white')
            + self._cell(fmt_pct(aacc), w_a, 'white'))
        print(row3, flush=True)
        print(row4, flush=True)
        print(flush=True)

    def after_train_epoch(self, runner, metrics=None):
        """按 epoch 训练时，每轮结束再打一行小结（紧接着 Runner 会跑验证）。"""
        if not is_main_process():
            return
        if self._train_loop_type(runner) != 'EpochBasedTrainLoop':
            return
        hub = runner.message_hub
        loss = self._safe_scalar(hub, 'train/loss')
        lr = self._safe_lr(runner)
        loss_str = f'{loss:.6f}' if loss is not None else '?'
        lr_str = f'{lr:.8f}' if lr is not None else '?'
        ep_done = runner.epoch + 1
        me = self._max_epochs_cfg
        ep_tag = f'{ep_done}/{me}' if me is not None else str(ep_done)
        it = runner.iter + 1
        print(
            self._c('[Epoch]', 'magenta')
            + f' 第 {ep_tag} 轮训练阶段结束 | global_iter={it} | '
            f'loss≈{loss_str} | lr={lr_str} | 随后验证集…',
            flush=True)
        print(flush=True)


@HOOKS.register_module()
class PlotMetricsHook(Hook):
    """采样训练 loss；每次验证追加 val_metrics.csv，并更新前景 IoU/F1/P/R 趋势图（单文件）。"""

    priority = 'LOW'

    def __init__(self, sample_interval=50):
        self.sample_interval = max(1, int(sample_interval))
        self._t_iters = []
        self._t_loss = []
        self._t_lr = []
        self._t_loss_mask = []
        self._t_loss_dice = []
        self._t_loss_cls = []
        self._v_epoch = []
        self._v_step = []
        self._v_iou = []
        self._v_f1 = []
        self._v_precision = []
        self._v_recall = []

    def before_train(self, runner):
        self._t_iters.clear()
        self._t_loss.clear()
        self._t_lr.clear()
        self._t_loss_mask.clear()
        self._t_loss_dice.clear()
        self._t_loss_cls.clear()
        self._v_epoch.clear()
        self._v_step.clear()
        self._v_iou.clear()
        self._v_f1.clear()
        self._v_precision.clear()
        self._v_recall.clear()

    @staticmethod
    def _log_dir(runner):
        ld = getattr(runner, '_log_dir', None)
        if ld:
            return ld
        ts = getattr(runner, 'timestamp', None)
        if ts and getattr(runner, 'work_dir', None):
            return osp.join(runner.work_dir, ts)
        return getattr(runner, 'work_dir', '.')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not is_main_process():
            return
        if (runner.iter + 1) % self.sample_interval != 0:
            return
        hub = runner.message_hub
        loss = ConsoleSummaryHook._safe_scalar(hub, 'train/loss')
        lr = ConsoleSummaryHook._safe_lr(runner)
        it = runner.iter + 1
        self._t_iters.append(it)
        self._t_loss.append(float(loss) if loss is not None else float('nan'))
        self._t_lr.append(float(lr) if lr is not None else float('nan'))
        lm = ConsoleSummaryHook._safe_scalar(hub, 'train/decode.loss_mask')
        ld = ConsoleSummaryHook._safe_scalar(hub, 'train/decode.loss_dice')
        lc = ConsoleSummaryHook._safe_scalar(hub, 'train/decode.loss_cls')
        self._t_loss_mask.append(float(lm) if lm is not None else float('nan'))
        self._t_loss_dice.append(float(ld) if ld is not None else float('nan'))
        self._t_loss_cls.append(float(lc) if lc is not None else float('nan'))

    def after_train_epoch(self, runner):
        """按 epoch 记录时，epoch 末也可更新曲线（仅训练部分），不必等验证。"""
        if not is_main_process():
            return
        self._save_figure(runner)

    def _append_val_csv(self, runner, metrics):
        logd = self._log_dir(runner)
        os.makedirs(logd, exist_ok=True)
        path = osp.join(logd, 'val_metrics.csv')
        ep = runner.epoch + 1
        step = runner.iter + 1
        iou = ConsoleSummaryHook._pick_metric(metrics, ['IoU', 'mIoU'])
        f1 = ConsoleSummaryHook._pick_metric(metrics, ['F1', 'mFscore', 'mF1'])
        pr = ConsoleSummaryHook._pick_metric(metrics, ['Precision', 'mPrecision'])
        rc = ConsoleSummaryHook._pick_metric(metrics, ['Recall', 'mRecall'])
        aacc = ConsoleSummaryHook._pick_metric(metrics, ['aAcc'])
        vl = metrics.get('val/loss')
        row = [
            ep,
            step,
            float(iou) if isinstance(iou, numbers.Real) else '',
            float(f1) if isinstance(f1, numbers.Real) else '',
            float(pr) if isinstance(pr, numbers.Real) else '',
            float(rc) if isinstance(rc, numbers.Real) else '',
            float(aacc) if isinstance(aacc, numbers.Real) else '',
            float(vl) if isinstance(vl, numbers.Real) else '',
        ]
        write_header = not osp.isfile(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    'epoch', 'global_iter', 'IoU_fg', 'F1', 'Precision', 'Recall', 'aAcc',
                    'val_loss',
                ])
            w.writerow(row)

    def after_val_epoch(self, runner, metrics=None):
        if not is_main_process():
            return
        if not isinstance(metrics, dict):
            metrics = {}
        step = runner.iter + 1
        ep = runner.epoch + 1
        iou = ConsoleSummaryHook._pick_metric(metrics, ['IoU', 'mIoU'])
        f1 = ConsoleSummaryHook._pick_metric(metrics, ['F1', 'mFscore', 'mF1'])
        pr = ConsoleSummaryHook._pick_metric(metrics, ['Precision', 'mPrecision'])
        rc = ConsoleSummaryHook._pick_metric(metrics, ['Recall', 'mRecall'])
        self._append_val_csv(runner, metrics)
        self._v_epoch.append(ep)
        self._v_step.append(step)
        self._v_iou.append(float(iou) if isinstance(iou, numbers.Real) else float('nan'))
        self._v_f1.append(float(f1) if isinstance(f1, numbers.Real) else float('nan'))
        self._v_precision.append(float(pr) if isinstance(pr, numbers.Real) else float('nan'))
        self._v_recall.append(float(rc) if isinstance(rc, numbers.Real) else float('nan'))
        self._save_figure(runner)

    def after_train(self, runner):
        if is_main_process():
            self._save_figure(runner)

    def _save_figure(self, runner):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print_log(
                'PlotMetricsHook: 未安装 matplotlib，跳过绘图。可 pip install matplotlib',
                logger='current',
                level=logging.WARNING)
            return

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        exp = getattr(runner, 'experiment_name', 'train')
        fig.suptitle(f'Train (sampled) — {exp}', fontsize=12)

        def _plot_xy(ax, xs, ys, title, ylabel, xlabel='global_iter', style='-'):
            if xs and any(v == v for v in ys):
                ax.plot(xs, ys, style, lw=1, alpha=0.88)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        if self._t_iters:
            _plot_xy(axes[0, 0], self._t_iters, self._t_loss, 'Train loss', 'loss')
            _plot_xy(axes[0, 1], self._t_iters, self._t_lr, 'Learning rate', 'lr', style='g-')
            _plot_xy(
                axes[1, 0], self._t_iters, self._t_loss_mask,
                'Train decode.loss_mask', 'loss', style='c-')
            ax = axes[1, 1]
            ax.plot(
                self._t_iters, self._t_loss_dice, '-',
                color='darkorange', lw=1, alpha=0.9, label='loss_dice')
            ax.plot(
                self._t_iters, self._t_loss_cls, '-',
                color='purple', lw=1, alpha=0.9, label='loss_cls')
            ax.set_title('Train decode.loss_dice / loss_cls')
            ax.set_xlabel('global_iter')
            ax.set_ylabel('loss')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'no train samples yet', ha='center', va='center')

        plt.tight_layout()
        logd = self._log_dir(runner)
        os.makedirs(logd, exist_ok=True)
        overview_png = osp.join(logd, 'train_curves.png')
        plt.savefig(overview_png, dpi=150, bbox_inches='tight')
        plt.close()

        if self._v_step:
            fig2, ax2 = plt.subplots(2, 2, figsize=(10, 8))
            fig2.suptitle('Val: foreground IoU / F1 / Precision / Recall (%)', fontsize=11)
            xsv = self._v_step
            ax2[0, 0].plot(xsv, self._v_iou, 'r-o', ms=3, lw=1)
            ax2[0, 0].set_title('IoU (fg)')
            ax2[0, 0].set_xlabel('global_iter @ val')
            ax2[0, 0].grid(True, alpha=0.3)
            ax2[0, 1].plot(xsv, self._v_f1, 'm-s', ms=3, lw=1)
            ax2[0, 1].set_title('F1 (fg)')
            ax2[0, 1].set_xlabel('global_iter @ val')
            ax2[0, 1].grid(True, alpha=0.3)
            ax2[1, 0].plot(xsv, self._v_precision, 'b-^', ms=3, lw=1)
            ax2[1, 0].set_title('Precision (fg)')
            ax2[1, 0].set_xlabel('global_iter @ val')
            ax2[1, 0].grid(True, alpha=0.3)
            ax2[1, 1].plot(xsv, self._v_recall, 'g-d', ms=3, lw=1)
            ax2[1, 1].set_title('Recall (fg)')
            ax2[1, 1].set_xlabel('global_iter @ val')
            ax2[1, 1].grid(True, alpha=0.3)
            plt.tight_layout()
            val_png = osp.join(logd, 'val_foreground_trends.png')
            plt.savefig(val_png, dpi=150, bbox_inches='tight')
            plt.close()

        print_log(
            f'PlotMetricsHook: 已写入 {overview_png}；验证指标见 val_metrics.csv',
            logger='current',
            level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=None,
        help='print training logs every N iterations, e.g. 10 for near real-time')
    parser.add_argument(
        '--log-by-epoch',
        action='store_true',
        default=False,
        help='log metrics by epoch style for easier reading during training')
    parser.add_argument(
        '--clean-console',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='print concise, readable training lines in terminal')
    parser.add_argument(
        '--clean-console-interval',
        type=int,
        default=1,
        help='每 N 次迭代打印一行 [训练]（含 epoch/batch 与 global_iter）；默认 1 便于观察变化，'
        '长训可调大（如 10）')
    parser.add_argument(
        '--plot-curves',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='在 log 目录保存 training_metrics.png / training_curves.png（需 matplotlib）')
    parser.add_argument(
        '--plot-sample-interval',
        type=int,
        default=50,
        help='sample train loss/lr every N iters for the plot')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='resume from a specific checkpoint .pth (full state if the file contains it)')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _is_full_mmengine_train_checkpoint(path):
    """仅当 checkpoint 内含 ``message_hub`` 等完整字段时，``Runner.resume`` 才合法。"""
    try:
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location='cpu')
    except Exception:
        return False
    return isinstance(obj, dict) and 'message_hub' in obj


def _apply_resume_from_cli(cfg, resume_from):
    """设置 ``cfg.load_from`` / ``cfg.resume``。

    MMEngine 在 ``resume=True`` 且 ``load_from`` 非空时，会用 **load_from** 的路径调用 ``resume()``；
    不能把断点路径只写在 ``cfg.resume`` 字符串里。``best_*.pth`` 多为精简权重，无 ``message_hub``，
    只能 ``load_from`` + ``resume=False``，否则会 ``KeyError: 'message_hub'``。
    """
    rp = osp.abspath(resume_from)
    if not osp.isfile(rp):
        raise FileNotFoundError(f'--resume-from 文件不存在: {rp}')
    if _is_full_mmengine_train_checkpoint(rp):
        cfg.load_from = rp
        cfg.resume = True
    else:
        print_log(
            '所选文件不是完整训练断点（无 message_hub 等），已改为仅 '
            '`load_from` 加载权重；优化器与 epoch 计数将重新开始。'
            '若要保留优化器与轮次续训，请使用同目录下的 last.pth / epoch_*.pth，'
            '或在 checkpoint 配置中启用 save_last。',
            logger='current',
            level=logging.WARNING)
        cfg.load_from = rp
        cfg.resume = False


def _ensure_val_best_checkpoint(cfg):
    """若未设置 save_best，则默认 ``mIoU``（通用分割）；DataA 等配置里应显式写 ``save_best='IoU'``。"""
    if cfg.get('default_hooks') is None:
        return
    dh = cfg.default_hooks
    ckpt = dh.get('checkpoint') if isinstance(dh, dict) else None
    if not isinstance(ckpt, dict):
        return
    if 'save_best' not in ckpt:
        ckpt['save_best'] = 'mIoU'
        ckpt.setdefault('rule', 'greater')
    if 'save_last' not in ckpt:
        ckpt['save_last'] = True


def _inject_val_loss_checkpoint_pipeline(cfg):
    """在 ``cfg.mask2former_enable_val_loss_best`` 为真时：关闭主 ``CheckpointHook`` 的 ``save_best``（不再写 IoU best），仅注入 ``val/loss`` + 第二套 ``CheckpointHook``。"""
    if not cfg.get('mask2former_enable_val_loss_best'):
        return
    dh = cfg.get('default_hooks')
    if not isinstance(dh, dict):
        return
    base_ckpt = dh.get('checkpoint')
    if not isinstance(base_ckpt, dict):
        return
    custom = list(cfg.get('custom_hooks') or [])
    if any(isinstance(h, dict) and h.get('type') == 'ValAverageLossHook' for h in custom):
        return
    if any(
            isinstance(h, dict) and h.get('type') == 'CheckpointHook'
            and h.get('save_best') == 'val/loss' for h in custom):
        return
    loss_ckpt = copy.deepcopy(base_ckpt)
    loss_ckpt['save_best'] = 'val/loss'
    loss_ckpt['rule'] = 'less'
    loss_ckpt['save_last'] = True
    loss_ckpt.setdefault('type', 'CheckpointHook')
    # 同一 train_* 目录下只保留按验证 loss 的 best，不再与 val/IoU 双文件并存
    base_ckpt['save_best'] = None
    new_entries = [dict(type='ValAverageLossHook'), loss_ckpt]
    insert_at = len(custom)
    for i, h in enumerate(custom):
        if isinstance(h, dict) and h.get('type') == 'CheckpointToLogDirHook':
            insert_at = i
            break
    cfg.custom_hooks = custom[:insert_at] + new_entries + custom[insert_at:]


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # make terminal logs easier to follow during training
    if args.log_interval is not None:
        if 'default_hooks' not in cfg:
            cfg.default_hooks = dict()
        if 'logger' not in cfg.default_hooks:
            cfg.default_hooks.logger = dict(type='LoggerHook')
        cfg.default_hooks.logger['interval'] = args.log_interval
        cfg.default_hooks.logger['log_metric_by_epoch'] = True

    if args.log_by_epoch:
        if 'log_processor' not in cfg:
            cfg.log_processor = dict()
        cfg.log_processor['by_epoch'] = True

    if args.clean_console:
        if 'default_hooks' not in cfg:
            cfg.default_hooks = dict()
        # Keep the raw logger but make it sparse to avoid noisy terminal output.
        if 'logger' not in cfg.default_hooks:
            cfg.default_hooks.logger = dict(type='LoggerHook')
        _tc = cfg.get('train_cfg')
        _epoch_loop = isinstance(_tc, dict) and _tc.get('type') == 'EpochBasedTrainLoop'
        if _epoch_loop:
            # 按 epoch 训练时保留配置中的 LoggerHook.interval（如每轮记一次），勿强行拉到 100000 iter
            cfg.default_hooks.logger.setdefault('log_metric_by_epoch', True)
        else:
            cfg.default_hooks.logger['interval'] = max(
                cfg.default_hooks.logger.get('interval', 50), 100000)

        custom_hooks = list(cfg.get('custom_hooks', []))
        custom_hooks.append(
            dict(type='ConsoleSummaryHook', interval=args.clean_console_interval))
        cfg.custom_hooks = custom_hooks

    if args.plot_curves:
        if 'custom_hooks' not in cfg or cfg.custom_hooks is None:
            cfg.custom_hooks = []
        cfg.custom_hooks = list(cfg.custom_hooks)
        cfg.custom_hooks.append(
            dict(
                type='PlotMetricsHook',
                sample_interval=args.plot_sample_interval))

    # 默认归档到 data/checkpoints1；显式 val_loss 配置使用 checkpoints2（见各 configs 的 work_dir）
    _mask2former_root = osp.dirname(osp.abspath(__file__))
    _run_ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.work_dir is not None:
        cfg.work_dir = osp.abspath(args.work_dir)
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(_mask2former_root, 'data', 'checkpoints1')
    else:
        wd = cfg.work_dir
        cfg.work_dir = (
            osp.abspath(wd) if osp.isabs(wd) else osp.join(_mask2former_root, wd))
    # 目录名：train_<时间>（MMEngine 默认会忽略 cfg.timestamp，见 _patch_mmengine_runner_use_cfg_timestamp）
    if not cfg.get('timestamp'):
        cfg.timestamp = f'train_{_run_ts}'

    _patch_mmengine_runner_use_cfg_timestamp()

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume: explicit path > --resume (auto latest) > cfg / --cfg-options
    if args.resume_from is not None:
        _apply_resume_from_cli(cfg, args.resume_from)
    elif args.resume:
        cfg.resume = True

    _ensure_val_best_checkpoint(cfg)
    if cfg.get('mask2former_enable_val_loss_best'):
        _inject_val_loss_checkpoint_pipeline(cfg)
        _inject_val_loss_early_stop(cfg)
        _dh_sync = cfg.get('default_hooks')
        if isinstance(_dh_sync, dict) and isinstance(
                _dh_sync.get('checkpoint'), dict):
            cfg.default_hooks.checkpoint['out_dir'] = cfg.work_dir
    else:
        _inject_iou_early_stop(cfg)

    # 单次训练归档：权重写入 runner.log_dir（与日志、曲线图同文件夹）
    _ch = cfg.get('custom_hooks')
    if _ch is None:
        _ch = []
    _ch = list(_ch)
    if not any(
            isinstance(h, dict) and h.get('type') == 'CheckpointToLogDirHook'
            for h in _ch):
        _ch.append(dict(type='CheckpointToLogDirHook'))
    cfg.custom_hooks = _ch

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # LocalVisBackend 使用 log_dir/vis_data（与 MMEngine Visualizer 一致）
    vis_data = osp.join(runner.log_dir, 'vis_data')
    os.makedirs(vis_data, exist_ok=True)

    if is_main_process():
        ck_cfg = cfg.get('default_hooks') or {}
        ck = ck_cfg.get('checkpoint') if isinstance(ck_cfg, dict) else None
        if cfg.get('mask2former_enable_val_loss_best'):
            sb, rule = 'val/loss', 'less'
        else:
            sb = (ck.get('save_best') if isinstance(ck, dict) else None) or '未设置'
            rule = (ck.get('rule') if isinstance(ck, dict) else None) or 'greater'
        msg = (
            f'本次 run 主输出目录（日志、val 指标、按「{sb}」的 best）: {runner.log_dir}\n'
            f'  每次验证后: 指标键「{sb}」与 rule={rule} → 更新 best 权重；'
            f'DataA 二分类下 IoU = 前景类 IoU。')
        if cfg.get('mask2former_enable_val_loss_best'):
            msg += '\n（val_loss 模式：主 checkpoint 不再按 IoU 另存 best，仅 val/loss。）'
        print_log(msg, logger='current', level=logging.INFO)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
