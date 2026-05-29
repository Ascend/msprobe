# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import itertools
import os
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import create_directory, write_df_to_csv
from msprobe.core.monitor.anomaly_processor import AnomalyDataFactory, AnomalyTurbulence, AnomalyScanner
from msprobe.pytorch.common.log import logger


class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class WriterInput:
    path: str
    ad_rules: list
    job_id: str
    anomaly_factory: AnomalyDataFactory = None
    ndigits: int = 6
    step_count_per_record: int = 1


class BaseWriterWithAD:
    def __init__(self, writer_input: WriterInput):
        self.tag2scalars = {}
        self.ad_rules = writer_input.ad_rules
        self.job_id = writer_input.job_id
        self.anomaly_factory = writer_input.anomaly_factory
        self.anomalies = []
        self.ndigits = writer_input.ndigits
        self.beta = 0.99

    @staticmethod
    def stack_tensors(tensor_list):
        """
        Torch not support stack cpu and xpu tensors. Group the tensors into cpu_group and xpu_group,
        stack them separately, migrate xpu_group to cpu, and then restore in the order of input.

        :param tensor_list: [tensor(-1.6165), tensor(-1.0985), tensor(-1.7777), tensor(-1.8408, device='npu:0')]
        :return: result: list of float
        """
        cpu_tensors = []
        xpu_tensors = []

        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor) and tensor.device.type != 'cpu':
                # 将device上的tensor先stack后to cpu
                xpu_tensors.append(tensor)
            else:
                cpu_tensors.append(tensor)

        xpu_stack = torch.stack(xpu_tensors).cpu() if xpu_tensors else torch.tensor([])

        # 按照输入的顺序恢复
        result = []
        cpu_tensors_idx, xpu_tensors_idx = 0, 0
        for tensor in tensor_list:
            if isinstance(tensor, torch.Tensor) and tensor.device.type != 'cpu':
                result.append(xpu_stack[xpu_tensors_idx])
                xpu_tensors_idx += 1
            else:
                result.append(cpu_tensors[cpu_tensors_idx])
                cpu_tensors_idx += 1

        return result

    def get_anomalies(self):
        """返回已检测到的异常列表"""
        return self.anomalies

    def clear_anomalies(self):
        self.anomalies.clear()

    def add_scalar(self, tag, scalar_value, global_step=None):
        """If an anomaly is detected, the anomaly information is recorded and added to self.anomalies.
        Args:
            tag (tuple): tuple of tag_name and tag like ('0:1.post_attention_norm.weight/rank0/pre_grad', 'min').
            scalar_value (float): scalar_value.
            global_step (int): global_step.
        Returns:
            None
        """
        if not self.ad_rules or tag[-1] in ["shape", "dtype"]:
            return
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.item()
        avg = self._update_tag2scalars(tag, scalar_value)
        detected, rule_name = self._ad(scalar_value, history=avg)
        if detected:
            if rule_name == AnomalyTurbulence.name and tag[-1] not in ["norm", "mean"]:
                return
            exception_message = (
                f"Rule {rule_name} reports anomaly signal in {tag} at step {global_step}, "
                f"current value {scalar_value}, history mean {avg}."
            )
            logger.info(f"{BCOLORS.WARNING}> {exception_message}{BCOLORS.ENDC}")
            # append to self.anomalies for dump
            if self.anomaly_factory:
                self.anomalies.append(self.anomaly_factory.create(tag, exception_message, global_step))

    def write_metrics(self, ops, metric_value, step, prefix=''):
        if not metric_value:
            return
        tensors = []
        tags = list(itertools.product(metric_value.keys(), ops))
        for op2tensor in metric_value.values():
            tensors.extend(op2tensor.values())
        if not tensors:
            return

        n_slices = len(tensors) // MonitorConst.SLICE_SIZE
        with torch.no_grad():
            for i in range(n_slices + 1):
                begin = i * MonitorConst.SLICE_SIZE
                end = (i + 1) * MonitorConst.SLICE_SIZE
                if begin == len(tensors):
                    continue
                metric_list = self.stack_tensors(tensors[begin:end])
                for tag, metric in zip(tags[begin:end], metric_list):
                    self.add_scalar(tag, metric, step)

    def _ad(self, scalar_value, history):
        return AnomalyScanner.scan(self.ad_rules, history, cur=scalar_value)

    def _update_tag2scalars(self, tag, scalar_value):
        """Update the average and count of a scalar value associated with a tag.

        This method is used to maintain a running average of scalar values for each tag.


        Args:
            tag (str): The tag identifier.
            scalar_value (float): The scalar value to be added.

        Returns:
            float: The average value before update.
        """
        abs_scalar_value = abs(scalar_value)
        if tag not in self.tag2scalars:
            self.tag2scalars[tag] = {'avg': abs_scalar_value, 'count': 0}
        avg = self.tag2scalars[tag]['avg']
        self.tag2scalars[tag]['avg'] = self.beta * avg + (1 - self.beta) * abs_scalar_value
        self.tag2scalars[tag]['count'] += 1
        return avg


class CSVWriterWithAD(BaseWriterWithAD):
    def __init__(self, writer_input: WriterInput):
        super().__init__(writer_input)

        path = writer_input.path
        self.log_dir = path
        create_directory(path)
        self.context_dict = defaultdict(list)
        self.header = []
        self.step_count_per_record = writer_input.step_count_per_record

    def get_step_interval(self, step):
        count = step // self.step_count_per_record
        return count * self.step_count_per_record, (count + 1) * self.step_count_per_record - 1

    def write_csv(self, prefix, step):
        """
        Args:
            prefix[str]: prefix of output csv file e.g. grad_unreduced
            step[int]
        """
        if len(self.context_dict) == 0:
            return

        ster_start, step_end = self.get_step_interval(step)
        filepath = os.path.join(self.log_dir, f'{prefix}_{ster_start}-{step_end}.csv')
        if not os.path.exists(filepath):
            data_frame = pd.DataFrame(columns=self.header)
            write_df_to_csv(data_frame, filepath)

        new_data = []
        for name, metric_value in self.context_dict.items():
            new_line = name.split(MonitorConst.NAME_SEP) + metric_value
            new_line.insert(2, step)
            new_data.append(new_line)
        new_data = pd.DataFrame(new_data).round(self.ndigits).fillna("nan")
        write_df_to_csv(new_data, filepath, mode='a+', header=False)
        self.context_dict = defaultdict(list)

    def add_scalar(self, tag, scalar_value, global_step=None):
        """
        ('0:1.post_attention_norm.weight/rank0/pre_grad', 'min')
        """
        super().add_scalar(tag, scalar_value, global_step)

        name = tag[0].split('/')[0]
        if isinstance(scalar_value, torch.Tensor):
            value = scalar_value.item()
        elif isinstance(scalar_value, torch.Size):
            value = list(scalar_value)
        else:
            value = scalar_value
        self.context_dict[name].append(value)

    def write_metrics(self, ops, metric_value, step, prefix='', **kwargs):
        super().write_metrics(ops, metric_value, step, prefix='')

        if prefix in [MonitorConst.ACTV, MonitorConst.ACTVGRAD] or kwargs.get("use_micro_step"):
            self.header = MonitorConst.CSV_HEADER_MICRO_STEP + ops
        else:
            self.header = MonitorConst.CSV_HEADER + ops
        self.write_csv(prefix, step)

    def close(self):
        pass


class SummaryWriterWithAD(SummaryWriter, BaseWriterWithAD):
    def __init__(self, writer_input: WriterInput):
        path = writer_input.path
        if not os.path.exists(path):
            create_directory(path)
        try:
            super(SummaryWriter, self).__init__(writer_input)  # pylint: disable=bad-super-call
            super().__init__(path)
        except Exception as e:
            logger.error(f'error when init summary writer at {path}: {e}')
            raise ValueError("Init summary writer error.") from e

    def add_scalar(self, tag, scalar_value, global_step=None):
        super(SummaryWriter, self).add_scalar(tag, scalar_value, global_step)  # pylint: disable=bad-super-call
        tag = f'{tag[0]}_{tag[1]}'
        super().add_scalar(tag, scalar_value, global_step)
