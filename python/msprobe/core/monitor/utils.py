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

import os
import re

from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import wraps

from msprobe.core.common.const import MonitorConst
from msprobe.core.common.file_utils import check_file_or_directory_path
from msprobe.core.common.log import logger
from msprobe.core.common.utils import is_int

beijing_tz = timezone(timedelta(hours=8))
MVResult = namedtuple('MVResult', ("exp_avg", "exp_avg_sq", "update", "ratio"))


class MsgConst:
    """
    Class for log messages const
    """

    SPECIAL_CHAR = ["\n", "\r", "\u007f", "\b", "\f", "\t", "\u000b", "%08", "%0a", "%0b", "%0c", "%0d", "%7f"]


def get_output_base_dir():
    return os.getenv(MonitorConst.MONITOR_OUTPUT_DIR, MonitorConst.DEFAULT_MONITOR_OUTPUT_DIR)


def filter_special_chars(func):
    @wraps(func)
    def func_level(msg):
        for char in MsgConst.SPECIAL_CHAR:
            msg = msg.replace(char, '_')
        return func(msg)

    return func_level


def validate_ops(ops):
    if not isinstance(ops, list):
        raise TypeError("ops should be a list")
    valid_ops = []
    for op in ops:
        if op not in MonitorConst.OP_LIST:
            logger.warning(f"op {op} is not supported. Optional ops: {MonitorConst.OP_LIST}")
            continue
        valid_ops.append(op)
    if not valid_ops:
        default_op = MonitorConst.OP_LIST[0]
        valid_ops.append(default_op)
        logger.info_on_rank_0(f"There is no valid ops, default op {default_op} is used")
    # 增加默认shape和dtype参数
    if "shape" not in valid_ops:
        valid_ops.append("shape")
    if "dtype" not in valid_ops:
        valid_ops.append("dtype")
    return valid_ops


def validate_ndigits(ndigits):
    if not ndigits:
        return
    if not is_int(ndigits) or ndigits <= 0:
        raise ValueError(f"ndigits({ndigits}) is not a positive integer, current is: {ndigits}.")
    if ndigits > MonitorConst.MAX_NDIGITS:
        raise ValueError(f"The maximum supported ndigits is {MonitorConst.MAX_NDIGITS}, current value: {ndigits}.")


def validate_ranks(ranks):
    if not isinstance(ranks, list):
        raise TypeError("module_ranks should be a list")
    for rank in ranks:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError(f"element in module_ranks should be a int, get {type(rank)}")


def validate_targets(targets):
    if not isinstance(targets, dict):
        raise TypeError('targets in config.json should be a dict')
    for module_name, field in targets.items():
        if not isinstance(module_name, str):
            raise TypeError('key of targets should be module_name[str] in config.json')
        if not isinstance(field, dict):
            raise TypeError('values of targets should be cared filed e.g. {"input": "tensor"} in config.json')


def validate_l2_targets(targets):
    if not isinstance(targets, dict):
        raise TypeError('l2_targets in config.json should be a dict')
    for hook_name, target_list in targets.items():
        if hook_name not in MonitorConst.L2_HOOKS:
            raise TypeError(f'key of l2_targtes must be in {MonitorConst.L2_HOOKS}, got {hook_name}')
        if not isinstance(target_list, list):
            raise TypeError('values of l2_targets should be a list in config.json')
        for item in target_list:
            if not isinstance(item, str):
                raise TypeError(f'item of "{hook_name}" in l2_targets should be module_name[str] in config.json')


def validate_recording_l2_features(recording_l2_features):
    if not isinstance(recording_l2_features, bool):
        raise TypeError("recording_l2_features should be a bool")


def validate_sa_order(sa_order):
    if isinstance(sa_order, str):
        sa_order = sa_order.replace(' ', '')
    if sa_order not in MonitorConst.SA_ORDERS:
        raise TypeError(f'sa_order must be in {MonitorConst.SA_ORDERS}, got {sa_order}')


def validate_print_struct(print_struct):
    if not isinstance(print_struct, bool):
        raise TypeError("print_struct should be a bool")


def validate_ur_distribution(ur_distribution):
    if not isinstance(ur_distribution, bool):
        raise TypeError('ur_distribution should be a bool')


def validate_xy_distribution(xy_distribution):
    if not isinstance(xy_distribution, bool):
        raise TypeError('xy_distribution should be a bool')


def validate_wg_distribution(wg_distribution):
    if not isinstance(wg_distribution, bool):
        raise TypeError('wg_distribution should be a bool')


def validate_mg_distribution(mg_distribution):
    if not isinstance(mg_distribution, bool):
        raise TypeError('mg_distribution should be a bool')


def validate_param_distribution(param_distribution):
    if not isinstance(param_distribution, bool):
        raise TypeError('param_distribution should be a bool')


def validate_cc_distribution(cc_distribution):
    if not isinstance(cc_distribution, dict):
        raise TypeError('cc_distribution should be a dictionary')
    for key, value in cc_distribution.items():
        if key == 'enable':
            if not isinstance(value, bool):
                raise TypeError('cc_distribution enable should be a bool')
        elif key == 'cc_codeline':
            if not isinstance(value, list):
                raise TypeError('cc_distribution cc_codeline should be a list')
        elif key == 'cc_pre_hook':
            if not isinstance(value, bool):
                raise TypeError('cc_distribution cc_pre_hook should be a bool')
        elif key == 'cc_log_only':
            if not isinstance(value, bool):
                raise TypeError('cc_distribution cc_log_only should be a bool')
        else:
            raise TypeError(f'{key} of cc_distribution is not supported.')


def validate_squash_name(squash_name):
    if not isinstance(squash_name, bool):
        raise TypeError('squash_name should be a bool')


def validate_alert(alert):
    if not isinstance(alert, dict):
        raise TypeError('alert should be a dictionary')
    rules = alert.get('rules')
    if rules and isinstance(rules, list):
        for rule in rules:
            rule_name = rule.get("rule_name")
            if rule_name and rule_name not in MonitorConst.RULE_NAME:
                raise TypeError(f"{rule_name} is not supported")
            args = rule.get("args")
            if args and isinstance(args, dict):
                threshold = args.get("threshold")
                if not isinstance(threshold, (float, int)) or threshold < 0:
                    raise TypeError('threshold must be float and not less than 0')
    dump = alert.get('dump')
    if dump and not isinstance(dump, bool):
        raise TypeError('dump must be bool.')


def validate_step_count_per_record(step_count_per_record):
    if not is_int(step_count_per_record):
        raise TypeError('step_count_per_record must be int.')
    if step_count_per_record < 1:
        raise ValueError("step_count_per_record must greater than 0")
    if step_count_per_record > 1e6:
        raise ValueError("step_count_per_record must smaller than 1e6")


def validate_dynamic_on(dynamic_on):
    if not isinstance(dynamic_on, bool):
        raise TypeError('dynamic_on should be a bool')


def validate_monitor_mbs_grad(monitor_mbs_grad):
    if not isinstance(monitor_mbs_grad, bool):
        logger.warning(f'monitor_mbs_grad should be a bool, actual value is {monitor_mbs_grad}.')
        return False
    return monitor_mbs_grad


def validate_append_output(append_output):
    if not isinstance(append_output, list):
        raise TypeError('append_output should be a list')
    if len(append_output) > 0 and len(append_output) != 2:
        raise ValueError('append_output should be empty or contain exactly 2 elements')


def validate_config(config):
    config['ops'] = validate_ops(config.get('ops', []))

    ndigits = config.get('ndigits')
    validate_ndigits(ndigits)

    eps = config.get('eps', 1e-8)
    if not isinstance(eps, float):
        raise TypeError("eps should be a float")

    ranks = config.get("module_ranks", [])
    validate_ranks(ranks)

    targets = config.get("targets", {})
    validate_targets(targets)

    l2_targets = config.get("l2_targets", {})
    validate_l2_targets(l2_targets)

    recording_l2_features = config.get("recording_l2_features", False)
    validate_recording_l2_features(recording_l2_features)

    sa_order = config.get("sa_order", "s,b,h,d")
    validate_sa_order(sa_order)

    print_struct = config.get('print_struct', False)
    validate_print_struct(print_struct)

    ur_distribution = config.get('ur_distribution', False)
    validate_ur_distribution(ur_distribution)

    xy_distribution = config.get('xy_distribution', False)
    validate_xy_distribution(xy_distribution)

    wg_distribution = config.get('wg_distribution', False)
    validate_wg_distribution(wg_distribution)

    mg_distribution = config.get('mg_distribution', False)
    validate_mg_distribution(mg_distribution)

    param_distribution = config.get('param_distribution', False)
    validate_param_distribution(param_distribution)

    cc_distribution = config.get('cc_distribution', {})
    validate_cc_distribution(cc_distribution)

    alert = config.get('alert', {})
    validate_alert(alert)

    step_count_per_record = config.get('step_count_per_record', 1)
    validate_step_count_per_record(step_count_per_record)

    config["start_step"] = validate_int_arg(
        config.get("start_step"), "start_step", MonitorConst.DEFAULT_START_STEP, MonitorConst.DEFAULT_START_STEP
    )
    config["collect_times"] = validate_int_arg(
        config.get("collect_times"),
        "collect_times",
        MonitorConst.DEFAULT_MIN_COLLECT_TIMES,
        MonitorConst.DEFAULT_MAX_COLLECT_TIMES,
    )
    config["step_interval"] = validate_int_arg(
        config.get("step_interval"),
        "step_interval",
        MonitorConst.DEFAULT_STEP_INTERVAL,
        MonitorConst.DEFAULT_STEP_INTERVAL,
    )

    squash_name = config.get('squash_name', False)
    validate_squash_name(squash_name)

    time_tags = config.get("append_output", [])
    validate_append_output(time_tags)

    config["monitor_mbs_grad"] = validate_monitor_mbs_grad(config.get('monitor_mbs_grad', False))

    dynamic_on = config.get('dynamic_on', False)
    validate_dynamic_on(dynamic_on)

    if not targets:
        if xy_distribution:
            config["all_xy"] = True
        config["targets"] = {"": {}}


def time_str2time_digit(time_str):
    time_format = '%b%d_%H-%M-%S'
    if not isinstance(time_str, str):
        raise TypeError(f"time_str:{time_str} should be a str")
    try:
        time_digit = datetime.strptime(time_str, time_format)
    except Exception as e:
        raise RuntimeError(
            f"illegal timestamp: {time_str}, timestamp should be prefix \
                           of existing output dirpath, like 'Dec03_21-34-40'."
        ) from e
    return time_digit


def get_target_output_dir(monitor_path, time_start, time_end):
    check_file_or_directory_path(monitor_path, isdir=True)
    time_start = time_str2time_digit(time_start) if time_start is not None else time_start
    time_end = time_str2time_digit(time_end) if time_end is not None else time_end
    if time_start and time_end and time_start > time_end:
        raise ValueError(f"time_start({time_start}) greater than time_end({time_end})")
    result = {}
    for dirname in os.listdir(monitor_path):
        match = re.match(MonitorConst.OUTPUT_DIR_PATTERN, dirname)
        if not match:
            continue
        time_tag = match.group(1)
        rank = match.group(2)
        target_time = time_str2time_digit(time_tag)
        start_ok = time_start is None or target_time >= time_start
        end_ok = time_end is None or target_time <= time_end
        if start_ok and end_ok:
            result[rank] = os.path.join(monitor_path, dirname)
    return result


def validate_set_monitor(grad_acc_steps, start_iteration):
    """
    validate parameters of set_monitor.
    """
    grad_acc_steps = validate_int_arg(
        grad_acc_steps, "grad_acc_steps", MonitorConst.DEFAULT_GRAD_ACC_STEPS, MonitorConst.DEFAULT_GRAD_ACC_STEPS
    )

    start_iteration = validate_int_arg(
        start_iteration, "start_iteration", MonitorConst.DEFAULT_START_ITERATION, MonitorConst.DEFAULT_START_ITERATION
    )
    return grad_acc_steps, start_iteration


def validate_int_arg(value, name, minimum, default_value):
    """Validate int args, if any exception occurs, use the default value."""
    if value is None:
        return default_value
    try:
        if not is_int(value):
            raise TypeError(f"{name} must be int")
        if value < minimum:
            raise ValueError(f"{name} must greater than {minimum}")
    except Exception as e:
        value = default_value
        logger.warning(f"Validate {name} failed, {e}, replaced with default value {value}.")
    return value
