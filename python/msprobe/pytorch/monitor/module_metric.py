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

# pylint: disable=duplicate-code  # 跨框架（mindspore/pytorch）实现相似是设计决定

import re

import torch

from msprobe.pytorch.monitor.features import get_max, get_min, get_zeros, get_nans, get_norm, get_mean
from msprobe.pytorch.monitor.features import (
    cal_entropy,
    cal_stable_rank,
    cal_router_weight_similarity,
    cal_per_token_expert_entropy,
)
from msprobe.pytorch.monitor.utils import get_nan_tensor


def get_summary_writer_tag_name(module_or_param_name: str, tag: str, rank):
    if rank is None:
        return f"{module_or_param_name}/{tag}"
    else:
        return f"{module_or_param_name}/rank{rank}/{tag}"


def squash_param_name(param_name, enable=True):
    if not enable:
        return param_name
    name = ''
    for pattern in [
        r'^.*\.(layers?\..*)',
        r'^.*\.(embeddings?\..*)',
        r'^.*\.(final.*)',
        r'^.*\.(output.*)',
        r'^.*\.(norm.*)',
    ]:
        match = re.findall(pattern, param_name)
        if match:
            name += match[0]
            break
    if name == '':
        name = param_name
    return name


# 用于存储所有metric实现类的注册表
config_metric_registry = {}


def register_config_metric(key, cls=None):
    """装饰器 用于注册Metric的实现类"""
    if cls is None:
        # 无参数时，返回装饰器函数
        return lambda cls_: register_config_metric(key, cls_)
    config_metric_registry[key] = cls()
    return cls


class TensorMetrics:
    fun_map = {"norm": get_norm, "max": get_max, "min": get_min, "mean": get_mean}

    def __init__(self) -> None:
        self.metrics = {}  # tensor_tag --> []
        self.cur_idx = {}

    def stat_insert(self, tensor, stat_ops, module_name, tensor_name, rank):
        """get stats and insert into metrics dictionary"""
        prefix = get_summary_writer_tag_name(module_name, tensor_name, rank)
        for stat_op in stat_ops:
            y = TensorMetrics.fun_map[stat_op](tensor)
            key = f"{prefix}_{stat_op}"
            if key not in self.metrics:
                self.metrics[key] = []
                self.cur_idx[key] = 0
            self.metrics[key].append(y)

    def flush(self, tb_writer):
        for key, metric_list in self.metrics.items():
            start = self.cur_idx[key]
            for v in metric_list[start:]:
                tb_writer.add_scalar(key, v.item(), global_step=self.cur_idx[key])
                self.cur_idx[key] += 1


class Metric:
    @staticmethod
    def get_metric_value(tensor, eps):
        raise NotImplementedError

    def get_metric(self, tensor, eps):
        try:
            return self.get_metric_value(tensor, eps)
        except RuntimeError:
            return torch.tensor(torch.nan).to(tensor.device)


@register_config_metric("min")
class MinMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_min(tensor)


@register_config_metric("mean")
class MeanMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_mean(tensor)


@register_config_metric("max")
class MaxMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_max(tensor)


@register_config_metric("norm")
class NormMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_norm(tensor)


@register_config_metric("zeros")
class ZerosMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_zeros(tensor, eps)


@register_config_metric("nans")
class NaNsMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return get_nans(tensor)


@register_config_metric("id")
class IdentMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        if tensor.dim() != 0:
            return None
        return tensor


@register_config_metric("shape")
class ShapeMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return tensor.shape


@register_config_metric("dtype")
class DtypeMetric(Metric):
    @staticmethod
    def get_metric_value(tensor, eps):
        return tensor.dtype


def get_metrics(ops, tag2tensor, eps, out_dict=None):
    """
    :param ops: ["op1", "op2"]
    :param tag2tensor: {
    '0:fc.input:0/actv': torch.randn([3, 4]),
    '0:fc.output:0/actv': torch.randn([3, 3])
    }
    :param eps: float 1e-8
    :param out_dict:{
    '0:fc.input:0/actv': {"op1": op1(torch.randn([3, 4])), "op2": op2(torch.randn([3, 4]))}
    '0:fc.output:0/actv': {"op1": op1(torch.randn([3, 3])), "op2": op2(torch.randn([3, 3]))}
    }
    :return: out_dict
    """
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if tag not in out_dict:
            out_dict[tag] = {}
        if not torch.is_tensor(tensor):
            # Non-tensor in/output filled with nan.
            out_dict[tag].update({metric_name: get_nan_tensor() for metric_name in ops})
            continue
        for metric_name in ops:
            fun_metric = config_metric_registry.get(metric_name)
            out_dict[tag][metric_name] = fun_metric.get_metric(tensor, eps)
    return out_dict


def get_sr_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if "sr" not in tag:
            continue
        if tag not in out_dict:
            out_dict[tag] = {}
        sr, eig = cal_stable_rank(tensor)
        out_dict[tag]['sr'] = sr
        out_dict[tag]['kernel_norm'] = eig


def get_entropy_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if tag not in out_dict:
            out_dict[tag] = {}
        entropy, softmax_max = cal_entropy(tensor)
        out_dict[tag]['entropy'] = entropy
        out_dict[tag]['softmax_max'] = softmax_max


def get_moe_router_weight_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if "router_weight" not in tag:
            continue
        if tag not in out_dict:
            out_dict[tag] = {}
        out_dict[tag]['router_weight_similarity'] = cal_router_weight_similarity(tensor)


def get_moe_router_logit_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if "router_logit" not in tag:
            continue
        if tag not in out_dict:
            out_dict[tag] = {}
        out_dict[tag]['per_token_expert_entropy'] = cal_per_token_expert_entropy(tensor)
