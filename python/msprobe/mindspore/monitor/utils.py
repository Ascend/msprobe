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

from mindspore import dtype as mstype, Tensor

from msprobe.mindspore.monitor.features import (
    FUNC_MAP,
    cal_entropy,
    cal_stable_rank,
    cal_router_weight_similarity,
    cal_per_token_expert_entropy,
)


def get_single_metrics(op_list, tag, tensor, eps=1e-8, output=None):
    if output is None:
        output = {}
    if tag not in output:
        output[tag] = {}
    for op in op_list:
        func = FUNC_MAP.get(op)
        if op == "zeros":
            statistic = func(tensor, eps)
        else:
            statistic = func(tensor)
        if hasattr(statistic, "dtype") and statistic.dtype == mstype.bfloat16:
            statistic = float(statistic)
            statistic = Tensor(statistic)
        if isinstance(statistic, Tensor):
            output[tag][op] = statistic.astype(mstype.float32)
        else:
            output[tag][op] = statistic


def get_metrics(op_list, tag2tensor, eps, output=None):
    if output is None:
        output = {}
    for tag, tensor in tag2tensor.items():
        if tag not in output:
            output[tag] = {}
        get_single_metrics(op_list, tag, tensor, eps, output)
    return output


def get_summary_writer_tag_name(module_or_param_name: str, tag: str, rank):
    if rank is None:
        return f"{module_or_param_name}/{tag}"
    else:
        return f"{module_or_param_name}/rank{rank}/{tag}"


def step_accumulates_one(context, micro_batch_number):
    """
    :param context: ModuleHookContext
    :param micro_batch_number: mbs of training model.
    :return:
    """
    context.micro_step += 1
    if context.micro_step == micro_batch_number:
        context.micro_step = 0
        context.step += 1


def is_skip_step(step, start_step, step_interval, has_collect_times=0, collect_times=1e8):
    """
    If current step less than start_step or not reach step_interval, skip current step.
    :param step: current training step, int
    :param start_step: int
    :param step_interval: int
    :return: whether skip or not, bool
    """
    return step < start_step or (step - start_step) % step_interval != 0 or has_collect_times >= collect_times


def get_entropy_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if tag not in out_dict:
            out_dict[tag] = {}
        entropy, softmax = cal_entropy(tensor)
        out_dict[tag]["entropy"] = entropy
        out_dict[tag]["softmax"] = softmax
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
        out_dict[tag]["sr"] = sr
        out_dict[tag]["eig"] = eig
    return out_dict


def get_moe_router_weight_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if "router_weight" not in tag:
            continue
        if tag not in out_dict:
            out_dict[tag] = {}
        out_dict[tag]["router_weight_similarity"] = cal_router_weight_similarity(tensor)
    return out_dict


def get_moe_router_logit_metric(tag2tensor, out_dict=None):
    if out_dict is None:
        out_dict = {}
    for tag, tensor in tag2tensor.items():
        if "router_logit" not in tag:
            continue
        if tag not in out_dict:
            out_dict[tag] = {}
        out_dict[tag]["per_token_expert_entropy"] = cal_per_token_expert_entropy(tensor)
    return out_dict
