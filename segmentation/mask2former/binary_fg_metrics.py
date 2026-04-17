# Copyright (c) OpenMMLab-style. 二分类前景（如输电线）评测：主指标为前景 IoU，并给出 F1/Precision/Recall。
import os.path as osp
from collections import OrderedDict
from typing import Dict

import numpy as np
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.evaluation.metrics.iou_metric import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class BinaryForegroundIoUMetric(IoUMetric):
    """在 per-class 统计基础上，只把「前景类」的 IoU/F1/Precision/Recall 写入 metrics（0–100），并保留 aAcc。

    需在 ``iou_metrics`` 中同时包含 ``mIoU`` 与 ``mFscore``，以便得到逐类 IoU 与 P/R/F。
    """

    def __init__(self, foreground_index: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.foreground_index = int(foreground_index)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.dataset_meta['classes']
        fg = self.foreground_index

        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        iou_arr = ret_metrics.get('IoU')
        fscore_arr = ret_metrics.get('Fscore')
        precision_arr = ret_metrics.get('Precision')
        recall_arr = ret_metrics.get('Recall')

        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        out = OrderedDict()
        if 'aAcc' in ret_metrics_summary:
            out['aAcc'] = float(ret_metrics_summary['aAcc'])

        if iou_arr is not None and 0 <= fg < len(iou_arr):
            v = float(np.nan_to_num(iou_arr[fg], nan=0.0))
            out['IoU'] = float(np.round(v * 100, 2))

        if fscore_arr is not None and 0 <= fg < len(fscore_arr):
            v = float(np.nan_to_num(fscore_arr[fg], nan=0.0))
            out['F1'] = float(np.round(v * 100, 2))
        if precision_arr is not None and 0 <= fg < len(precision_arr):
            v = float(np.nan_to_num(precision_arr[fg], nan=0.0))
            out['Precision'] = float(np.round(v * 100, 2))
        if recall_arr is not None and 0 <= fg < len(recall_arr):
            v = float(np.nan_to_num(recall_arr[fg], nan=0.0))
            out['Recall'] = float(np.round(v * 100, 2))

        return out
