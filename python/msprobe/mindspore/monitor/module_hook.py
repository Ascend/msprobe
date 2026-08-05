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

import os
import re
import uuid
from collections import defaultdict
from datetime import datetime

import pytz
import pandas as pd
import mindspore
from mindspore import Tensor, mint
from mindspore import nn, _no_grad

from msprobe.core.common.log import logger
from msprobe.core.common.const import MonitorConst, Const
from msprobe.core.common.file_utils import load_json, save_json, create_directory
from msprobe.core.monitor.utils import validate_config, get_output_base_dir, get_target_output_dir
from msprobe.core.monitor.anomaly_processor import AnomalyScanner, AnomalyDataFactory, AnomalyDataWriter
from msprobe.mindspore.common.utils import is_mindtorch
from msprobe.mindspore.monitor.common_func import (
    is_valid_instance,
    get_parameters,
    get_submodules,
    get_rank,
)
from msprobe.mindspore.monitor.utils import (
    get_summary_writer_tag_name,
    step_accumulates_one,
    is_skip_step,
    get_metrics,
    get_entropy_metric,
    get_sr_metric,
    get_moe_router_weight_metric,
    get_moe_router_logit_metric,
)
from msprobe.mindspore.monitor.optimizer_collect import OptimizerMonFactory
from msprobe.mindspore.monitor.data_writers import CSVWriterWithAD, BaseWriterWithAD, WriterInput
from msprobe.mindspore.monitor.distributed.wrap_distributed import api_register, create_hooks, op_aggregate
from msprobe.mindspore.monitor.features import cal_qkt
from msprobe.core.common.file_utils import write_df_to_csv
from msprobe.core.common.utils import analyze_api_call_stack

FORMAT_MAPPING = {MonitorConst.CSV: CSVWriterWithAD, MonitorConst.API: BaseWriterWithAD}


def get_param_struct(param):
    res = {}
    if isinstance(param, (tuple, list)):
        res['config'] = f'{type(param).__name__}[{len(param)}]'
        for i, x in enumerate(param):
            res[i] = f'size={tuple(x.shape)}, dtype={x.dtype}' if isinstance(x, Tensor) else f'{type(x)}'
    elif isinstance(param, Tensor):
        res['config'] = 'tensor'
        res['tensor'] = f'size={tuple(param.shape)}, dtype={param.dtype}'
    else:
        res['config'] = f'{type(param)}'
        logger.warning(f'Not support type({type(param)}) now, please check the type of param {param}')
    return res


def param_is_not_tensor_parallel_duplicate(param, tp_group):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
        mint.distributed.get_rank(group=tp_group) == 0
    )


def param_is_data_parallel_duplicate(dp_group):
    return mint.distributed.get_rank(group=dp_group) != 0


def squash_param_name(param_name):
    for pattern in [
        r'^.*\.(layers?\..*)',
        r'^.*\.(embeddings?\..*)',
        r'^.*\.(final.*)',
        r'^.*\.(output.*)',
        r'^.*\.(norm.*)',
    ]:
        match = re.findall(pattern, param_name)
        if match:
            return match[0]
    return param_name


def is_recording_module(module_name, l2_targets, vpp_stage):
    if len(l2_targets) > 0:
        for pattern in [vpp_stage + squash_param_name(module_name), vpp_stage + module_name]:
            if pattern in l2_targets:
                return pattern
    return ""


# Used For Module Forward & Backward Collect
class ModuleHookContext:
    def __init__(self, module_name) -> None:
        self.step = 0
        self.micro_step = 0
        self.actv = defaultdict(dict)
        self.actvgrad = []
        self.module_name = module_name
        self.struct = {}
        self.stack = ""

    def reset(self):
        self.actv.clear()
        self.actvgrad.clear()


class FeatureHookContext:
    def __init__(self, module_name):
        self.step = 0
        self.micro_step = 0
        self.attention_feature = {}
        self.linear_feature = {}
        self.moe_router_weight_feature = {}
        self.moe_router_logit_feature = {}
        # 独立的 micro_step 计数器，避免同一 module 上多个 hook 共享计数导致时序错乱
        self.moe_router_weight_micro_step = 0
        self.moe_router_logit_micro_step = 0
        self.module_name = module_name

    def reset(self):
        self.attention_feature.clear()
        self.linear_feature.clear()
        self.moe_router_weight_feature.clear()
        self.moe_router_logit_feature.clear()
        self.moe_router_weight_micro_step = 0
        self.moe_router_logit_micro_step = 0


start_step = 0


# Used For Optimizer Weight Grad & M/V Collect
class OptimizerContext:
    def __init__(self) -> None:
        self.step = start_step
        self.param_mg_direction = defaultdict(float)
        self.param_adam_update = defaultdict()
        self.param_adam_ratio = defaultdict()
        self.param_weight_grad = defaultdict()
        self.param_exp_avg = defaultdict()
        self.exp_avg_metric = {}
        self.param_exp_avg_sq = defaultdict()
        self.exp_avg_sq_metric = {}
        self.metric_dict = {}
        self.param_metric = {}

    def reset(self) -> None:
        self.param_mg_direction.clear()
        self.param_adam_update.clear()
        self.param_adam_ratio.clear()
        self.param_weight_grad.clear()
        self.param_exp_avg.clear()
        self.exp_avg_metric.clear()
        self.param_exp_avg_sq.clear()
        self.exp_avg_sq_metric.clear()
        self.metric_dict.clear()
        self.param_metric.clear()


# Used For Weight Grad Collect
class GradContext:
    def __init__(self) -> None:
        self.pre = {}
        self.post = {}
        self.actv = {}

    def reset(self):
        self.pre.clear()
        self.post.clear()
        self.actv.clear()


class CommunicationContext:
    def __init__(self) -> None:
        self.data = {}

    @staticmethod
    def _agg(data):
        aggregated_data = {}
        for tag, op2tensorlist in data.items():
            aggregated_data[tag] = {}
            for op, tensorlist in op2tensorlist.items():
                aggregated_data[tag][op] = op_aggregate(op, tensorlist)
        return aggregated_data

    def reset(self):
        self.data = {}

    def aggregate(self):
        self.data = self._agg(self.data)


class TrainerMon:
    def __init__(self, config_file_path, process_group=None, params_have_main_grad=True) -> None:
        # TYPE1: 只在这里初始化的变量, 不会随着训练中途config配置改变而重置
        self.config_file_path = config_file_path
        self.process_group = process_group
        self.params_have_main_grad = params_have_main_grad
        self.is_mindtorch = is_mindtorch()
        self.config_timestamp = (
            0  # 后面有校验时间戳, 首次监测无需为了更新config文件时间戳而去改, 可通过dynamic_on开关直接打开
        )
        self.config = load_json(config_file_path)
        validate_config(self.config)

        local_tz = pytz.timezone("Asia/Shanghai")  # 根据需要调整为目标时区
        cur_time = datetime.now(local_tz).strftime('%b%d_%H-%M-%S')
        self.unique_id = str(uuid.uuid4())[:8]
        self.output_base_dir = get_output_base_dir()
        time_tags = self.config.get("append_output", [])
        try:
            self.rank = get_rank()
            if time_tags:
                output_append_dirs = get_target_output_dir(self.output_base_dir, time_tags[0], time_tags[1])
                if str(self.rank) in output_append_dirs:
                    self.tensorboard_dir = output_append_dirs[str(self.rank)]
                    logger.info(f"Append rank({self.rank}) result to {self.tensorboard_dir}")
            else:
                self.tensorboard_dir = os.path.join(
                    self.output_base_dir, f"{cur_time}-rank{self.rank}-{self.unique_id}"
                )
        except Exception:
            self.rank = 0
            self.tensorboard_dir = os.path.join(self.output_base_dir, f"{cur_time}-rank{self.rank}-{self.unique_id}")

        self.pp_stage = 0
        self.group_mates = [0]

        # TYPE2: 只会在set_monitor()主调中赋值的变量
        self.model = None
        self.vpp = False
        self.dp_group = None
        self.tp_group = None
        self.micro_batch_number = 1
        self.optimizer_mon = None

        # TYPE3: 会随着训练中途config配置更新或监测状态改变而重置的变量
        self.module_fwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.module_bwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.feature_hook_context_by_module = defaultdict(FeatureHookContext)
        self.optimizer_context = defaultdict(OptimizerContext)
        self.cc_context = defaultdict(CommunicationContext)
        self.grad_context = GradContext()
        self.handles = defaultdict(list)
        self.param2name = defaultdict(str)
        self.name2index = defaultdict()
        self.name2indices = defaultdict()
        self.name2param = {}
        self.duplicate_param = {}
        self.name2tag = {}
        self.param_name_call_id = {}
        self.call_id = 0
        self.module_struct = defaultdict(dict)
        self.weight_hooked = False
        self.optimizer_hooked = False
        self.param_registered = False
        self.struct_printed = False
        self.pre_step_hooks = []
        self.post_step_hooks = []

        # 动静态区分
        self.dynamic_enable = os.getenv("DYNAMIC_MONITOR", 'False').lower() == 'true'
        if self.dynamic_enable:
            logger.warning(
                f"DYNAMIC_MONITOR is set, "
                f"please make sure you have 'dynamic_on' and 'collect_times' in {self.config_file_path}"
            )
            self.monitoring = False
        else:
            self.set_config()
            # 静态且collect_times>0时在第0步self.monitoring就可以True, 动态默认在下一步开启
            if self.collect_times > 0:
                self.monitoring = True

    @staticmethod
    def get_linear_hook_target(module):
        if isinstance(module, nn.Embedding):
            return ''
        if hasattr(module, "num_embeddings") or hasattr(module, "vocab_start_index"):
            return ''
        for weight_name in ["weight", "wg"]:
            if hasattr(module, weight_name) and isinstance(getattr(module, weight_name), Tensor):
                if getattr(module, weight_name).dim() == 2:
                    return weight_name
        return ''

    def set_config(self):
        self.start_step = self.config.get("start_step", 0)
        self.collect_times = self.config.get("collect_times", 100000000)  # 默认大值, 目的是一直采集
        self.step_interval = self.config.get("step_interval", 1)
        self.has_collect_times = 0  # 重设采集计数器
        self.print_struct = self.config.get("print_struct", False)
        self.targets = self.config.get("targets", None)
        self.module_rank_list = self.config.get("module_ranks", [])
        self.format = self.config.get('format', MonitorConst.CSV)  # only csv supported in mindspore
        self.eps = self.config.get('eps', 1e-8)
        self.ops = self.config.get('ops', [])  # monitor mean/max/norm/min/nan...
        self.ndigits = self.config.get('ndigits', 6)
        self.all_xy = self.config.get('all_xy', False)
        self.xy_distribution = self.config.get('xy_distribution', False)
        self.forward_only = self.config.get('forward_only', False)
        self.backward_only = self.config.get('backward_only', False)
        self.ur_distribution = self.config.get('ur_distribution', False)  # vector and ratio vector of adam
        self.mv_distribution = self.config.get("mv_distribution", False)  # m/v of adam
        self.wg_distribution = self.config.get("wg_distribution", False)
        self.param_distribution = self.config.get("param_distribution", False)
        self.mg_direction = self.config.get('mg_direction', False)  # main grad direction
        self.cc_distribution = self.config.get("cc_distribution", {})  # communication ops
        self.stack_info = self.config.get('stack_info', False)
        self.monitor_mbs_grad = self.config.get('monitor_mbs_grad', False)
        self.recording_l2_features = self.config.get('recording_l2_features', False)
        self.sa_order = self.config.get('sa_order', "s,b,h,d")

        if not self.cc_distribution.get('enable', False):
            self.cc_log_only = False
        else:
            self.cc_codeline = self.cc_distribution.get('cc_codeline', [])
            self.cc_log_only = self.cc_distribution.get('cc_log_only', False)
            self.cc_logged_stack = defaultdict(set)
            self.cc_pre_hook = self.cc_distribution.get('cc_pre_hook', False)
        self.common_info()

        # 初始化AnomalyData工厂
        alert_setting = self.config.get('alert', {"rules": []})
        self.alert_rules = AnomalyScanner.load_rules(alert_setting["rules"])
        self.anomaly_data_factory = None
        if alert_setting.get('dump', False):
            self.anomaly_data_factory = AnomalyDataFactory(self.rank, self.pp_stage, self.group_mates)

        # 初始化writer, 创建输出目录
        if self.format not in FORMAT_MAPPING:
            logger.error(f"Unsupported format: {self.format}, use default format: {MonitorConst.CSV}")
            self.format = MonitorConst.CSV
        self.step_count_per_record = self.config.get('step_count_per_record', 1)
        if not self.module_rank_list or (self.rank in self.module_rank_list):
            writer = FORMAT_MAPPING[self.format]
            self.summary_writer = writer(
                WriterInput(
                    self.tensorboard_dir,
                    self.alert_rules,
                    self.unique_id,
                    self.anomaly_data_factory,
                    self.ndigits,
                    self.step_count_per_record,
                )
            )

            # 初始化anomaly detected文件目录
            if self.anomaly_data_factory:
                self.anomaly_data_writer = AnomalyDataWriter(
                    os.path.join(self.output_base_dir, "anomaly_detected"), self.rank
                )
                self.anomaly_data_writer.init_detected_json()

    def common_info(self):
        if not self.xy_distribution:
            logger.info("> module input/output input_grad/output_grad is not monitored. ")
        if self.forward_only:
            logger.info("> only module forward is monitored. ")
        if not self.ur_distribution:
            logger.info("> update vector and ratio vector of adam is not monitored. ")
        if not self.mv_distribution:
            logger.info("> momentum and variance of adam is not monitored. ")
        if not self.wg_distribution:
            logger.info("> weight grad of specified module is not monitored. ")
        if not self.recording_l2_features:
            logger.info("> l2 features of specified module is not monitored. ")
        if not self.mg_direction:
            logger.info('> grad and momentum direction will not be compared.')
        if not self.cc_distribution.get('enable', False):
            logger.info("> cc operator is not monitored.")

    def set_monitor(self, model, optimizer, grad_acc_steps=1, tp_group=None, dp_group=None, start_iteration=0):
        global start_step
        start_step = start_iteration
        self.micro_batch_number = grad_acc_steps
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.optimizer_mon = OptimizerMonFactory.create_optimizer_mon(optimizer)
        self.hook_step_final(optimizer)
        if not isinstance(model, list):
            model = [model]
        self.model = model
        if len(model) > 1:
            self.vpp = True
            logger.info('vpp enabled')
        if not self.dynamic_enable:
            self.register_hooks(optimizer)

    def hook_step_final(self, optimizer):
        def step_final_hook(optimizer, *args, **kwargs):
            context = self.optimizer_context[optimizer]
            # 静态在第0步就可以保存, 动态在第0步不可以, 因为动态设计的就是重置后下一步开启, 第0步的self.monitoring还是False
            if self.monitoring:
                module_rank_valid = self.is_target_rank()
                step_condition = (
                    context.step >= self.start_step and (context.step - self.start_step) % self.step_interval == 0
                )
                if module_rank_valid and step_condition:
                    self.has_collect_times += 1

                    if self.anomaly_data_factory:
                        self.anomaly_data_factory.set_call_id(self.param_name_call_id)
                    self.write_xy_tb(context.step)
                    self.write_grad_tb(context.step)
                    self.write_mv_tb(context)
                    self.write_param_tb(context)
                    self.write_features_tb(context.step)
                    if self.stack_info:
                        self.write_stack_info()
                        self.stack_info = False  # pylint: disable=attribute-defined-outside-init
                        for handle in self.handles["stack"]:
                            handle.remove()
                        self.handles["stack"].clear()

                    if context.metric_dict:
                        self.summary_writer.write_metrics(self.ops, context.metric_dict, context.step, 'other')
                    context.metric_dict.clear()

                    if self.anomaly_data_factory:
                        self.anomaly_data_writer.write_detected_json(self.summary_writer.get_anomalies())
                    self.summary_writer.clear_anomalies()

                    self.call_id = 0
                    self.param_name_call_id.clear()

                    if self.has_collect_times >= self.collect_times:
                        self._remove_all_hooks_final(optimizer)

            context.step += 1
            self.dynamic_monitor(optimizer)

        def patch_step(func, optimizer):
            def wrapper(*args, **kwargs):
                for hook in self.pre_step_hooks:
                    hook(optimizer, args, kwargs)
                out = func(*args, **kwargs)
                for hook in self.post_step_hooks:
                    hook(optimizer, args, kwargs)
                step_final_hook(optimizer, args, kwargs)
                return out

            return wrapper

        if self.is_mindtorch:
            optimizer.__class__.step = patch_step(optimizer.__class__.step, optimizer)
        else:
            optimizer.__class__.construct = patch_step(optimizer.__class__.construct, optimizer)

    def dynamic_monitor(self, optimizer):
        """
        If dynamic monitor enabled and config.json updated,
        remove hooks and register new hooks according to new configuration.
        """
        context = self.optimizer_context[optimizer]
        if not self.dynamic_enable:
            return
        try:
            # 如果文件时间戳没变, 可以不读取节省时间
            config_timestamp = os.path.getmtime(self.config_file_path)
            if config_timestamp == self.config_timestamp:
                return
            # 更新config文件最新修改时间戳
            self.config_timestamp = config_timestamp
            config = load_json(self.config_file_path)
        except Exception as e:
            logger.error(f"get config.json wrong because {e}, not updated, please check!!!")
            return

        if config.get("dynamic_on", False):
            try:
                validate_config(config)
                self.config = config
                self.set_config()
                self.start_step = context.step  # pylint: disable=attribute-defined-outside-init
                logger.warning(
                    f"config is updated at step{context.step - 1}, will start new hook at step{context.step}."
                )
            except Exception as e:
                logger.error(f"set config wrong because {e}, not updated, please check!!!")
                return

            self._remove_all_hooks(optimizer)
            self.register_hooks(optimizer)

    def register_hooks(self, optimizer):
        self._register_param_name()
        self.hook_modules()
        self.hook_optimizer(optimizer)
        self._patch_grad_sync()
        if self.cc_distribution.get('enable', False):
            self.handles['cc'] = api_register.initialize_hook(*create_hooks(context=self.cc_context, monitor=self))
            api_register.redirect_api()
        self.monitoring = True

    def hook_modules(self):
        if not self.is_target_rank():
            return
        module_in_all_stage = [key for key in self.targets.keys() if MonitorConst.NAME_SEP not in key]

        for key in module_in_all_stage:
            struct = self.targets.pop(key)
            self.targets.update(
                {f'{vpp_stage}{MonitorConst.NAME_SEP}{key}': struct for vpp_stage in range(len(self.model))}
            )

        hooked_count = 0
        for vpp_stage, model_chunk in enumerate(self.model):
            if not is_valid_instance(model_chunk):
                logger.info("Target Model is not Cell")
                continue
            vpp_stage = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            targets = [x for x, _ in get_submodules(model_chunk)] if self.print_struct else self.targets.keys()
            l2_target_names = self.config.get('l2_targets', {})
            hooked_count += self._hook_module(targets, l2_target_names, model_chunk, vpp_stage)
        logger.info(f"> {hooked_count} modules are monitored.")

    def hook_optimizer(self, optimizer):
        def optimizer_pre_step_hook(opt, *args, **kwargs):
            context = self.optimizer_context[opt]
            if (
                self.print_struct
                and not all(value == {} for value in self.module_struct.values())
                and not self.struct_printed
            ):
                self._save_module_struct()
                if not self.cc_log_only:
                    raise RuntimeError("exit after first monitor step when print model struct")
            if is_skip_step(
                context.step, self.start_step, self.step_interval, self.has_collect_times, self.collect_times
            ):
                return

            grad_dict = {}
            if self.wg_distribution:
                grad_dict = self.optimizer_mon.fetch_grad(self, self.param2name)

            if self.mv_distribution or self.ur_distribution or self.mg_direction:
                if self.is_mindtorch:
                    (
                        context.param_exp_avg,
                        context.param_exp_avg_sq,
                        context.param_adam_update,
                        context.param_adam_ratio,
                    ) = self.optimizer_mon.fetch_mv(self, self.param2name)
                else:
                    context.param_exp_avg, context.param_exp_avg_sq = self.get_mv_for_ms(optimizer)

            self.generate_wgrad_metrics(grad_dict)
            self.generate_mv_metrics(context)
            self.generate_param_metrics(context, MonitorConst.PRE_PARAM)

            metric_dict = {}
            for cc in self.cc_context.values():
                cc.aggregate()
                metric_dict.update(cc.data)
                cc.reset()

            if not metric_dict:
                return
            context.metric_dict = metric_dict
            return

        def optimizer_post_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            self.generate_param_metrics(context, MonitorConst.POST_PARAM)

        if self.optimizer_hooked or not self.is_target_rank():
            return

        self.pre_step_hooks.append(optimizer_pre_step_hook)
        self.post_step_hooks.append(optimizer_post_step_hook)
        self.optimizer_hooked = True
        return

    def generate_wgrad_metrics(self, grad_dict):
        if not self.wg_distribution:
            return

        get_metrics(self.ops, grad_dict, self.eps, self.grad_context.post)

    def generate_param_map(self, tag, param_tensor):
        metrics = {}
        if not self.is_mindtorch:
            return param_tensor
        for name in self.param2name.values():
            key = get_summary_writer_tag_name(name, tag, self.rank)
            self.register_param_call_id("optimizer_pre_step_hook", key)
            if name not in param_tensor or param_tensor[name] is None:
                continue
            metrics[key] = param_tensor[name]
        return metrics

    def generate_param_metrics(self, opt_context, stage=MonitorConst.PRE_PARAM):
        if not self.param_distribution:
            return
        tag2param = {
            self.name2tag.get(name, {}).get(stage): param
            for name, param in self.name2param.items()
            if param.numel() != 0
        }
        get_metrics(self.ops, tag2param, self.eps, opt_context.param_metric)

    def get_mv_for_ms(self, opt):
        if not self.mv_distribution:
            return {}, {}
        common_opt = opt
        if not is_valid_instance(opt):
            common_opt = getattr(opt, 'optimizer')
            if not is_valid_instance(common_opt):
                logger.warning("Optimizer is not valid, please check usage")
                return {}, {}
        m_dict = {}
        v_dict = {}
        for name, param in get_parameters(common_opt):
            if MonitorConst.EXP_AVG_SQ in name:
                v_dict[name] = param
            elif MonitorConst.EXP_AVG in name:
                m_dict[name] = param
        return m_dict, v_dict

    def generate_mv_metrics(self, opt_context):
        if not self.mv_distribution:
            return
        opt_context.exp_avg_metric = {}
        opt_context.exp_avg_sq_metric = {}
        m_tag_tensor_map = self.generate_param_map(MonitorConst.EXP_AVG, opt_context.param_exp_avg)
        v_tag_tensor_map = self.generate_param_map(MonitorConst.EXP_AVG_SQ, opt_context.param_exp_avg_sq)
        get_metrics(self.ops, m_tag_tensor_map, self.eps, opt_context.exp_avg_metric)
        get_metrics(self.ops, v_tag_tensor_map, self.eps, opt_context.exp_avg_sq_metric)

    def write_stack_info(self):
        stack_data = []
        header = ["module_name", "stack_info"]
        stack_data.append(header)
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            stack_data.append([fwd_context.module_name, fwd_context.stack])
        filepath = os.path.join(self.tensorboard_dir, 'stack_info.csv')
        if not os.path.exists(filepath):
            data_frame = pd.DataFrame(columns=stack_data)
            write_df_to_csv(data_frame, filepath)

    def write_xy_tb(self, step):
        if not self.xy_distribution:
            return
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            if len(fwd_context.actv) == 0:
                continue
            self.summary_writer.write_metrics(self.ops, fwd_context.actv, step, MonitorConst.ACTV)
            fwd_context.actv.clear()
        if self.grad_context.actv:
            self.summary_writer.write_metrics(self.ops, self.grad_context.actv, step, MonitorConst.ACTVGRAD)

    def write_param_tb(self, opt_context):
        if not self.param_distribution:
            return
        param_metrics = {k: v for k, v in opt_context.param_metric.items() if MonitorConst.PRE_PARAM in k}
        updated_param_metrics = {k: v for k, v in opt_context.param_metric.items() if MonitorConst.POST_PARAM in k}
        self.summary_writer.write_metrics(self.ops, param_metrics, opt_context.step, MonitorConst.PRE_PARAM)
        self.summary_writer.write_metrics(self.ops, updated_param_metrics, opt_context.step, MonitorConst.POST_PARAM)

    def write_mv_tb(self, opt_context):
        if not self.mv_distribution:
            return
        self.summary_writer.write_metrics(self.ops, opt_context.exp_avg_metric, opt_context.step, MonitorConst.EXP_AVG)
        self.summary_writer.write_metrics(
            self.ops, opt_context.exp_avg_sq_metric, opt_context.step, MonitorConst.EXP_AVG_SQ
        )

    def write_grad_tb(self, step):
        if not self.wg_distribution:
            return

        self.summary_writer.write_metrics(
            self.ops, self.grad_context.pre, step, 'grad_unreduced', use_micro_step=self.monitor_mbs_grad
        )
        self.summary_writer.write_metrics(self.ops, self.grad_context.post, step, 'grad_reduced')

    def write_metrics_if_not_empty(self, features, metrics, step, hook_name):
        if not features or len(features) == 0:
            return
        use_micro_step = hook_name not in ["linear_hook", "moe_router_weight_hook"]
        self.summary_writer.write_metrics(metrics, features, step, hook_name, use_micro_step=use_micro_step)
        features.clear()

    def write_features_tb(self, step):
        if not self.recording_l2_features:
            return
        for context in self.feature_hook_context_by_module.values():
            num_features = (
                len(context.attention_feature)
                + len(context.linear_feature)
                + len(context.moe_router_weight_feature)
                + len(context.moe_router_logit_feature)
            )
            if num_features == 0:
                continue
            self.write_metrics_if_not_empty(context.attention_feature, ["entropy", "softmax"], step, "attention_hook")
            self.write_metrics_if_not_empty(context.linear_feature, ["sr", "kernel_norm"], step, "linear_hook")
            self.write_metrics_if_not_empty(
                context.moe_router_weight_feature, ["router_weight_similarity"], step, "moe_router_weight_hook"
            )
            self.write_metrics_if_not_empty(
                context.moe_router_logit_feature, ["per_token_expert_entropy"], step, "moe_router_logit_hook"
            )

    def is_target_rank(self):
        if self.module_rank_list and (self.rank not in self.module_rank_list):
            return False
        return True

    def build_tbtag_tensor_map(self, module_name, suffix, tag, tensor):
        """
        :param module_name: str of module name
        :param suffix:
        :param tag:
        :param tensor: torch.tensor or tuple/list of torch.tensor
        :return: tensor_map
        """
        tensor_map = {}
        if isinstance(tensor, Tensor):
            tensor = [tensor]
        if isinstance(tensor, (tuple, list)):
            if len(tensor) == 1:
                key = get_summary_writer_tag_name(module_name + suffix, tag, self.rank)
                self.register_param_call_id("_hook_module", key)
                tensor_map[key] = tensor[0]
            else:
                for i, tensor_i in enumerate(tensor):
                    key = get_summary_writer_tag_name(module_name + f"_{i}" + suffix, tag, self.rank)
                    self.register_param_call_id("_hook_module", key)
                    tensor_map[key] = tensor_i
        return tensor_map

    def register_param_call_id(self, hook_name: str, key: str):
        """
        :param hook_name:
        :param key: str, '0:relu_0/output_grad'
        :return:
        """
        logger.debug(f"{hook_name} {key}: {self.call_id}")
        self.param_name_call_id[key] = self.call_id
        self.call_id += 1

    def _register_param_name(self):
        for vpp_stage, model_chunk in enumerate(self.model):
            prefix = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            self._register_chunk(model_chunk, prefix)

    def _register_chunk(self, model_chunk, prefix):
        index = 0
        for param_name, param in get_parameters(model_chunk):
            if not param.requires_grad:
                continue
            if self._is_target_param(param_name, param, prefix):
                name = prefix + squash_param_name(param_name)
                if name in self.param2name.values():
                    name = prefix + param_name
                self.param2name[param] = name
                self.name2param[name] = param
                self.name2index[name] = index

                if self.tp_group and not param_is_not_tensor_parallel_duplicate(param, self.tp_group):
                    self.duplicate_param[name] = True
                if self.dp_group and param_is_data_parallel_duplicate(self.dp_group):
                    self.duplicate_param[name] = True
                keywords = [
                    MonitorConst.PRE_GRAD,
                    MonitorConst.POST_GRAD,
                    MonitorConst.PRE_PARAM,
                    MonitorConst.POST_PARAM,
                ]
                self.name2tag[name] = {k: get_summary_writer_tag_name(name, k, self.rank) for k in keywords}
                index += 1

    def _save_module_struct(self):
        output_dir = os.path.join(get_output_base_dir(), 'module_struct', f'rank{self.rank}')
        create_directory(output_dir)
        module_struct_file = os.path.realpath(os.path.join(output_dir, 'module_struct.json'))
        save_json(module_struct_file, self.module_struct, indent=2)
        logger.info(f"> save module struct to {module_struct_file}")
        self.struct_printed = True

    def _hook_module(self, target_names, l2_target_names, module, vpp_stage=''):
        if not is_valid_instance(module):
            # nothing to hook
            return 0

        def fwd_hook_fun(module, args, kwargs, module_output, name):
            module_input = [tensor for tensor in args if isinstance(tensor, Tensor)]
            if kwargs:
                kwargs_tensors = [tensor for tensor in kwargs.values() if isinstance(tensor, Tensor)]
                module_input.extend(kwargs_tensors)

            if module not in self.module_fwd_hook_context_by_module:
                self.module_fwd_hook_context_by_module[module] = ModuleHookContext(name)
            context: ModuleHookContext = self.module_fwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    Const.INPUT: get_param_struct(module_input),
                    Const.OUTPUT: get_param_struct(module_output),
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return
            if not module.training:
                return
            if is_skip_step(
                context.step, self.start_step, self.step_interval, self.has_collect_times, self.collect_times
            ):
                step_accumulates_one(context, self.micro_batch_number)
                return

            tbtag_tensor_map = {}
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.INPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV,
                    module_input,
                )
            )
            module_output = (
                [tensor for tensor in module_output if isinstance(tensor, Tensor)]
                if isinstance(module_output, tuple)
                else module_output
            )
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.OUTPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV,
                    module_output,
                )
            )
            try:
                get_metrics(self.ops, tbtag_tensor_map, self.eps, context.actv)
            except Exception:
                logger.warning("An error occurred while generating forward activation metrics")

            step_accumulates_one(context, self.micro_batch_number)
            return

        def bwd_hook_fun(module, input_grad, output_grad):
            context: ModuleHookContext = self.module_bwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    MonitorConst.ACTVGRAD_IN: get_param_struct(input_grad),
                    MonitorConst.ACTVGRAD_OUT: get_param_struct(output_grad),
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return

            if is_skip_step(
                context.step, self.start_step, self.step_interval, self.has_collect_times, self.collect_times
            ):
                step_accumulates_one(context, self.micro_batch_number)
                return

            valid_input_grad = [tensor for tensor in input_grad if isinstance(tensor, Tensor)]
            tbtag_tensor_map = {}
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.INPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTVGRAD,
                    valid_input_grad,
                )
            )

            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.OUTPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTVGRAD,
                    output_grad,
                )
            )

            if context.micro_step == 0 and context.actvgrad:
                logger.warning(
                    f"actvgrad context of {context.module_name} is not empty when first micro_step, "
                    f"maybe something wrong happened. Now clear it."
                )
                context.actvgrad.clear()
            try:
                get_metrics(self.ops, tbtag_tensor_map, self.eps, self.grad_context.actv)
            except Exception as e:
                logger.warning(f"An error occurred while generating backward activation metrics: {e}")

            step_accumulates_one(context, self.micro_batch_number)
            return

        def fwd_hook_register(module, fwd_hook_fun, name):
            from packaging import version

            if version.parse(mindspore.__version__) >= version.parse('2.6.0'):

                def wrapper(module, args, kwargs, module_output):
                    return fwd_hook_fun(module, args, kwargs, module_output, name)

                return module.register_forward_hook(wrapper, with_kwargs=True)

            else:

                def wrapper(module, args, module_output):
                    return fwd_hook_fun(module, args, None, module_output, name)

                return module.register_forward_hook(wrapper)

        def extract_attention_feature_hook(module, args, kwargs, module_output, name):
            module_input = [tensor for tensor in args if isinstance(tensor, Tensor)]
            if kwargs:
                kwargs_tensors = [tensor for tensor in kwargs.values() if isinstance(tensor, Tensor)]
                module_input.extend(kwargs_tensors)

            if module not in self.feature_hook_context_by_module:
                self.feature_hook_context_by_module[module] = FeatureHookContext(name)
            context: FeatureHookContext = self.feature_hook_context_by_module[module]

            tbtag_tensor_map = {}
            if len(module_input) < 2:
                logger.warning(
                    "Calculate attention feature failed, the length of module_input in attention hook's module should "
                    "be greater than or equal to 2."
                )

            q_h = module_input[0]
            k_h = module_input[1]
            qkt = cal_qkt(q_h, k_h, order=self.sa_order)
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.attention', f'{MonitorConst.NAME_SEP}{context.micro_step}', 'qkt', qkt
                )
            )
            get_entropy_metric(tbtag_tensor_map, context.attention_feature)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1

        def extract_linear_sr_hook(module, args, kwargs, module_output, name):
            weight_name = self.get_linear_hook_target(module)
            if weight_name == "":
                return
            if module not in self.feature_hook_context_by_module:
                self.feature_hook_context_by_module[module] = FeatureHookContext(name)
            context: FeatureHookContext = self.feature_hook_context_by_module[module]

            if context.micro_step == self.micro_batch_number - 1:
                tbtag_tensor_map = {}
                value = module.weight.data
                tbtag_tensor_map.update(
                    self.build_tbtag_tensor_map(
                        f'{context.module_name}.linear', f'{MonitorConst.NAME_SEP}{context.micro_step}', 'sr', value
                    )
                )

                get_sr_metric(tbtag_tensor_map, context.linear_feature)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
            return

        def extract_router_weight_similarity_hook(module, args, kwargs, module_output, name):
            if not module.training:
                return
            weight_name = self.get_linear_hook_target(module)
            if weight_name == "":
                return
            if module not in self.feature_hook_context_by_module:
                self.feature_hook_context_by_module[module] = FeatureHookContext(name)
            context: FeatureHookContext = self.feature_hook_context_by_module[module]

            if context.moe_router_weight_micro_step == self.micro_batch_number - 1:
                tbtag_tensor_map = {}
                value = getattr(module, weight_name).data
                tbtag_tensor_map.update(
                    self.build_tbtag_tensor_map(f'{context.module_name}.moe_router', '', 'router_weight', value)
                )
                get_moe_router_weight_metric(tbtag_tensor_map, context.moe_router_weight_feature)

            context.moe_router_weight_micro_step += 1
            if context.moe_router_weight_micro_step == self.micro_batch_number:
                context.moe_router_weight_micro_step = 0
            return

        def extract_per_token_expert_entropy_hook(module, args, kwargs, module_output, name):
            if not module.training:
                return
            logit_tensor = module_output
            if isinstance(logit_tensor, (list, tuple)):
                logit_tensor = next((t for t in logit_tensor if isinstance(t, Tensor)), None)
            if not isinstance(logit_tensor, Tensor):
                logger.warning(
                    f"moe_router_logit hook ({name}) got non-tensor output: {type(module_output)}. Skipping."
                )
                return
            if logit_tensor.dim() < 2:
                logger.warning(
                    f"moe_router_logit hook ({name}) got unexpected logit shape: {logit_tensor.shape}. Skipping."
                )
                return

            if module not in self.feature_hook_context_by_module:
                self.feature_hook_context_by_module[module] = FeatureHookContext(name)
            context: FeatureHookContext = self.feature_hook_context_by_module[module]

            tbtag_tensor_map = {}
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.moe_router',
                    f'{MonitorConst.NAME_SEP}{context.moe_router_logit_micro_step}',
                    'router_logit',
                    logit_tensor,
                )
            )
            get_moe_router_logit_metric(tbtag_tensor_map, context.moe_router_logit_feature)

            context.moe_router_logit_micro_step += 1
            if context.moe_router_logit_micro_step == self.micro_batch_number:
                context.moe_router_logit_micro_step = 0
            return

        def stack_hook(module, args, kwargs, module_output, name):
            if module not in self.module_fwd_hook_context_by_module:
                self.module_fwd_hook_context_by_module[module] = ModuleHookContext(name)
            context: ModuleHookContext = self.module_fwd_hook_context_by_module[module]
            context.stack = analyze_api_call_stack(name)

        if self.backward_only and self.forward_only:
            logger.warning('not enable backward_only and forward_only simultaneously')
        hooked_count = 0

        for module_name, submodule in get_submodules(module):
            if self.stack_info:
                name = vpp_stage + squash_param_name(module_name)
                handle = fwd_hook_register(submodule, stack_hook, name=name)
                self.handles["stack"].append(handle)
            name = self._is_target_module(module_name, target_names, vpp_stage)
            if not name:
                continue
            if self.xy_distribution or self.print_struct:
                if not self.backward_only:
                    handle = fwd_hook_register(submodule, fwd_hook_fun, name=name)
                    self.handles['xy'].append(handle)
                if not self.forward_only:
                    handle = submodule.register_backward_hook(bwd_hook_fun)
                    self.handles['xy'].append(handle)
                    self.module_bwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                logger.info(f"> {name} is monitored successfully")
                hooked_count += 1

        if not self.print_struct and self.recording_l2_features:
            for module_name, submodule in get_submodules(module):
                func_map = {
                    "attention_hook": extract_attention_feature_hook,
                    "linear_hook": extract_linear_sr_hook,
                    "moe_router_weight_hook": extract_router_weight_similarity_hook,
                    "moe_router_logit_hook": extract_per_token_expert_entropy_hook,
                }
                for hook, func in func_map.items():
                    if hook in l2_target_names:
                        temp_names = l2_target_names[hook]
                        name = is_recording_module(module_name, temp_names, vpp_stage)
                        if name:
                            handle = fwd_hook_register(submodule, func, name=name)
                            print_feature_name = hook.split('_', maxsplit=1)[0]
                            logger.info_on_rank_0(
                                f'> {print_feature_name} features of {name} is monitored successfully'
                            )
                            self.handles["L2_features"].append(handle)
                            hooked_count += 1
        return hooked_count

    def _patch_grad_sync(self):
        if not self.wg_distribution:
            return
        self._hook_weights()

    def _hook_weights(self):
        @_no_grad()
        def param_hook(grad, param, name):
            key = name
            if self.monitor_mbs_grad:
                key += f'{MonitorConst.NAME_SEP}{param.micro_step}'
            key = get_summary_writer_tag_name(key, 'acc_grad', self.rank)
            self.register_param_call_id("param_hook", key)
            param.micro_step += 1
            if self.monitor_mbs_grad or (param.micro_step == self.micro_batch_number):
                get_metrics(self.ops, {key: grad}, self.eps, self.grad_context.pre)
            if param.micro_step == self.micro_batch_number:
                param.micro_step = 0

        def param_hook_wrapper(param_hook, param, name):
            def wrapper(grad):
                return param_hook(grad, param, name)

            return wrapper

        logger.info("hooking weight grads.")
        for param, name in self.param2name.items():
            setattr(param, 'micro_step', 0)
            handle = param.register_hook(param_hook_wrapper(param_hook, param=param, name=name))
            self.handles['wgrads'].append(handle)
        self.weight_hooked = True

    def _is_target_param(self, param_name, param, prefix):
        if not self.targets:
            return True
        squash_name = prefix + squash_param_name(param_name)
        name = prefix + param_name
        for target in self.targets.keys():
            if param_name.startswith(target) or squash_name.startswith(target) or name.startswith(target):
                setattr(param, "zero_out_wgrad", True)
                return True
        return False

    def _is_target_module(self, module_name, targets, vpp_stage):
        if self.all_xy or self.print_struct:
            return vpp_stage + squash_param_name(module_name)
        for pattern in [
            vpp_stage + squash_param_name(module_name),
            vpp_stage + module_name,
        ]:
            if pattern in targets:
                return pattern
        return ""

    def _remove_all_hooks(self, optimizer):
        # 清空hook handle
        for handle in self.handles['xy']:
            handle.remove()
        self.handles['xy'].clear()
        for handle in self.handles['L2_features']:
            handle.remove()
        self.handles['L2_features'].clear()
        # 清空对应context缓存
        for fwd_context in self.module_fwd_hook_context_by_module.values():
            fwd_context.reset()
        for bwd_context in self.module_bwd_hook_context_by_module.values():
            bwd_context.reset()
        for feature_context in self.feature_hook_context_by_module.values():
            feature_context.reset()
        self.grad_context.reset()  # 权重梯度和激活值梯度都在这

        for handle in self.handles['wgrads']:
            handle.remove()
        self.handles['wgrads'].clear()
        self.weight_hooked = False

        if self.optimizer_hooked:
            self.pre_step_hooks.clear()
            self.post_step_hooks.clear()
        for _, context in self.optimizer_context.items():
            context.reset()
        self.optimizer_hooked = False

        for handle in self.handles['cc']:
            handle.remove()
        self.handles['cc'].clear()
        api_register.restore_api()
        for _, context in self.cc_context.items():
            context.reset()

        # 清空节点缓存
        self.param2name.clear()
        self.name2index.clear()
        self.name2indices.clear()
        self.name2param.clear()
        self.duplicate_param.clear()
        self.name2tag.clear()
        self.module_struct.clear()

        # 关闭采集状态
        self.monitoring = False

    def _remove_all_hooks_final(self, optimizer):
        if self.dynamic_enable:
            # 结束后自动重置dynamic_on为False等待用户手动开启
            try:
                config = load_json(self.config_file_path)
                config['dynamic_on'] = False
                save_json(self.config_file_path, config, indent=2)
                config_timestamp = os.path.getmtime(self.config_file_path)
                self.config_timestamp = config_timestamp
                logger.info(
                    "Finish monitor, set config'dynamic_on=False, will restart by set it to True and update config"
                )
            except Exception as e:
                logger.warning(f"Finish monitor, set config'dynamic_on=False fail because {e}, please check!!!")
        logger.info("Finish monitor")
        self._remove_all_hooks(optimizer)
