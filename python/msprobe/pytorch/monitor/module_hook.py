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

import json
import os
import uuid
import importlib
from collections import defaultdict
from datetime import datetime
from functools import partial

import pytz
import torch
import torch.distributed as dist
import pandas as pd
from torch.utils.hooks import BackwardHook
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from msprobe.core.common.const import MonitorConst, Const
from msprobe.core.common.file_utils import load_json, save_json, create_directory
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.monitor.anomaly_processor import AnomalyScanner, AnomalyDataFactory, AnomalyDataWriter
from msprobe.core.common.file_utils import write_df_to_csv
from msprobe.core.common.utils import analyze_api_call_stack
from msprobe.core.monitor.utils import (
    validate_config,
    validate_ops,
    get_output_base_dir,
    get_target_output_dir,
    validate_set_monitor,
)
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import is_recomputation
from msprobe.pytorch.monitor.utils import get_param_struct
from msprobe.pytorch.monitor.data_writers import CSVWriterWithAD, WriterInput
from msprobe.pytorch.monitor.distributed.wrap_distributed import (
    api_register,
    create_hooks,
    op_aggregate,
    get_process_group,
)
from msprobe.pytorch.monitor.features import get_sign_matches, cal_qkt
from msprobe.pytorch.monitor.module_metric import (
    get_metrics,
    get_summary_writer_tag_name,
    TensorMetrics,
    squash_param_name,
    get_entropy_metric,
    get_sr_metric,
)
from msprobe.pytorch.monitor.optimizer_collect import OptimizerMonFactory
from msprobe.pytorch.monitor.visualizer import HeatmapVisualizer

torch_version_above_or_equal_2 = torch.__version__.split('+')[0] >= '2.0'
if not torch_version_above_or_equal_2:
    raise ValueError("monitor require torch>=2.0")

FORMAT_MAPPING = {
    MonitorConst.CSV: CSVWriterWithAD,
}
start_step = 0


def param_is_not_tensor_parallel_duplicate(param, tp_group):
    return (hasattr(param, 'tensor_model_parallel') and param.tensor_model_parallel) or (
        torch.distributed.get_rank(group=tp_group) == 0
    )


def param_is_data_parallel_duplicate(dp_group):
    return torch.distributed.get_rank(group=dp_group) != 0


class ModuleHookContext:
    def __init__(self, module_name) -> None:
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
        self.module_name = module_name

    def reset(self):
        self.attention_feature.clear()
        self.linear_feature.clear()


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

    def reset(self):
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


class GradContext:
    def __init__(self) -> None:
        self.pre = {}
        self.post = {}
        self.actv = {}

    def reset(self):
        self.pre.clear()
        self.post.clear()
        self.actv.clear()


class TrainerMon:
    tensor_metrics = TensorMetrics()

    # 保留原opt_ty参数, 兼容msprobe1.2.2前旧版本
    def __init__(self, config_file_path, process_group=None, params_have_main_grad=True, opt_ty=None) -> None:
        # TYPE1: 只在这里初始化的变量, 不会随着训练中途config配置改变而重置
        self.config_file_path = config_file_path
        self.process_group = get_process_group(process_group)
        self.params_have_main_grad = params_have_main_grad
        self.update_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.ratio_heatmap_visualizer = defaultdict(HeatmapVisualizer)
        self.fsdp_post_backward_hook = None
        self.fsdp2_foreach_reduce = None
        self.config_timestamp = (
            0  # 后面有校验时间戳, 首次监测无需为了更新config文件时间戳而去改, 可通过dynamic_on开关直接打开
        )
        self.config = load_json(config_file_path)
        validate_config(self.config)

        self.squash_name = self.config.get('squash_name', False)  # 不允许修改防止前后名字对不上
        local_tz = pytz.timezone("Asia/Shanghai")  # 根据需要调整为目标时区
        cur_time = datetime.now(local_tz).strftime('%b%d_%H-%M-%S')
        self.unique_id = str(uuid.uuid4())[:8]
        self.output_base_dir = get_output_base_dir()
        time_tags = self.config.get("append_output", [])
        if dist.is_initialized():
            self.rank = dist.get_rank()
            if time_tags:
                output_append_dirs = get_target_output_dir(self.output_base_dir, time_tags[0], time_tags[1])
                if str(self.rank) in output_append_dirs:
                    self.tensorboard_dir = output_append_dirs[str(self.rank)]
                    logger.info(f"append rank({self.rank}) result to {self.tensorboard_dir}")
            else:
                self.tensorboard_dir = os.path.join(
                    self.output_base_dir, f"{cur_time}-rank{self.rank}-{self.unique_id}"
                )
            self.pp_stage = dist.get_group_rank(self.process_group, self.rank)
            self.group_mates = dist.get_process_group_ranks(self.process_group)
        else:
            self.rank = 0
            self.tensorboard_dir = os.path.join(self.output_base_dir, f"{cur_time}-rank{self.rank}-{self.unique_id}")
            self.pp_stage = 0
            self.group_mates = [0]

        # TYPE2: 只会在set_monitor()主调中赋值的变量
        self.model = None
        self.vpp = False
        self.dp_group = None
        self.tp_group = None
        self.enable_megatron = False
        self.enable_deepspeed = False
        self.fsdp_wrapped_module = False
        self.fsdp2_wrapped_module = False
        self.micro_batch_number = 1
        self.optimizer_mon = None
        self.optimizer_trans = None

        # TYPE3: 会随着训练中途config配置更新或监测状态改变而重置的变量
        self.module_fwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.module_bwd_hook_context_by_module = defaultdict(ModuleHookContext)
        self.feature_hook_context_by_module = defaultdict(FeatureHookContext)
        self.optimizer_context = defaultdict(OptimizerContext)
        self.cc_context = defaultdict(CommunicationContext)
        self.grad_context = GradContext()
        self.handles = defaultdict(list)
        self.param2name = defaultdict(str)
        self.name2indices = defaultdict()
        self.name2param = {}
        self.origin2squash = {}
        self.duplicate_param = {}
        self.name2tag = {}
        self.param_name_call_id = {}
        self.fsdp_param_name_map = {}
        self.call_id = 0
        self.module_struct = defaultdict(dict)
        self.grad_accs = []
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

    def __del__(self):
        if hasattr(self, "summary_writer"):
            self.summary_writer.close()

    @property
    def ops(self):
        return self._ops

    @ops.setter
    def ops(self, value):
        self._ops = validate_ops(value)

    @staticmethod
    def has_register_backward_hook(module_name, module):
        if (
            hasattr(module, '_backward_hooks')
            and len(module._backward_hooks) > 0
            and module._is_full_backward_hook is False
        ):
            logger.warning(
                f"The {module_name} has registered deprecated register_backward_hook,"
                f"which may cause abnormal data dump. The backward input/output for this module will be skipped."
            )
            return True
        return False

    @staticmethod
    def generate_cc_metrics(cc_name, cc_tensor):
        metrics = defaultdict(dict)
        rank = dist.get_rank() if dist.is_initialized() else None
        for op, tag2tensor in cc_tensor.data.items():
            for tag, tensor in tag2tensor.items():
                key = get_summary_writer_tag_name(cc_name, tag, rank)
                metrics[op].update({key: tensor})
        cc_tensor.reset()
        return metrics

    @staticmethod
    def get_linear_hook_target(module):
        if isinstance(module, torch.nn.Embedding):
            return ''
        if hasattr(module, "num_embeddings") or hasattr(module, "vocab_start_index"):
            return ''
        for weight_name in ["weight", "wg"]:
            if hasattr(module, weight_name) and isinstance(getattr(module, weight_name), torch.Tensor):
                if getattr(module, weight_name).dim() == 2:
                    return weight_name
        return ''

    def set_config(self):
        logger.info(f"current config: {self.config}")
        self.start_step = self.config.get("start_step", 0)
        self.collect_times = self.config.get("collect_times", 100000000)  # 默认大值, 目的是一直采集
        self.step_interval = self.config.get("step_interval", 1)
        self.has_collect_times = 0  # 重设采集计数器
        self.print_struct = self.config.get("print_struct", False)
        self.module_rank_list = self.config.get("module_ranks", [])
        self.format = self.config.get('format', MonitorConst.CSV)
        self.eps = self.config.get('eps', 1e-8)
        self.ops = self.config.get('ops', [])
        self.ndigits = self.config.get('ndigits', 6)
        self.all_xy = self.config.get('all_xy', False)
        self.xy_distribution = self.config.get('xy_distribution', False)
        self.forward_only = self.config.get('forward_only', False)
        self.backward_only = self.config.get('backward_only', False)
        self.ur_distribution = self.config.get('ur_distribution', False)
        self.mv_distribution = self.config.get("mv_distribution", False)
        self.wg_distribution = self.config.get("wg_distribution", False)
        self.param_distribution = self.config.get("param_distribution", False)
        self.mg_direction = self.config.get('mg_direction', False)
        self.cc_distribution = self.config.get("cc_distribution", {})
        self.stack_info = self.config.get('stack_info', False)
        self.monitor_mbs_grad = self.config.get('monitor_mbs_grad', False)
        self.recording_l2_features = self.config.get("recording_l2_features", False)
        self.sa_order = self.config.get("sa_order", "s,b,h,d")

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
            logger.warning(f"Unsupported format: {self.format}, use default format: {MonitorConst.CSV}")
            self.format = MonitorConst.CSV

        if self.ur_distribution:
            logger.warning("ur_distribution is not supported in CSV format, cancel ur_distribution")
            self.ur_distribution = False

        writer = FORMAT_MAPPING[self.format]
        self.step_count_per_record = self.config.get('step_count_per_record', 1)

        if (self.rank in self.module_rank_list) or len(self.module_rank_list) == 0:
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
            logger.info_on_rank_0("> module input/output input_grad/output_grad is not monitored. ")
        if self.forward_only:
            logger.info_on_rank_0("> only module forward is monitored. ")
        if not self.ur_distribution:
            logger.info_on_rank_0("> update vector and ratio vector of adam is not monitored. ")
        if not self.mv_distribution:
            logger.info_on_rank_0("> momentum and variance of adam is not monitored. ")
        if not self.wg_distribution:
            logger.info_on_rank_0("> weight grad of specified module is not monitored. ")
        if not self.recording_l2_features:
            logger.info_on_rank_0("> l2 features of specified module is not monitored. ")
        if not self.mg_direction:
            logger.info_on_rank_0('> grad and momentum direction will not be compared.')
        if not self.cc_distribution.get('enable', False):
            logger.info_on_rank_0("> cc operator is not monitored.")

    # 保留原接口, 兼容msprobe1.2.2前旧版本
    def monitor_gnorm_with_ad(
        self, model, optimizer=None, grad_acc_steps=1, tp_group=None, dp_group=None, start_iteration=0
    ):
        if optimizer is None:
            optimizer = getattr(self, "optimizer_trans", None)  # 兼容老版本可传None的情况, 从set_wrapped_optimizer获取
            if optimizer is None:
                logger.error("monitor_gnorm_with_ad: please set_wrapped_optimizer before it or input optimizer!=None")
                return
        self.set_monitor(model, optimizer, grad_acc_steps, tp_group, dp_group, start_iteration)

    # 保留原接口, 兼容msprobe1.2.2前旧版本
    def set_wrapped_optimizer(self, optimizer):
        self.optimizer_trans = optimizer

    def set_monitor(self, model, optimizer, grad_acc_steps=1, tp_group=None, dp_group=None, start_iteration=0):
        """External interface"""
        grad_acc_steps, start_iteration = validate_set_monitor(grad_acc_steps, start_iteration)
        global start_step
        start_step = start_iteration
        logger.info(f'grad acc steps {grad_acc_steps}')
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
            self._smallest_rank_print('vpp enabled')
        if not self.dynamic_enable:
            self.register_hooks(optimizer)

    def register_hooks(self, optimizer):
        self._register_param_name()
        self.hook_optimizer(optimizer)
        self._patch_grad_sync()
        self.hook_modules()
        if self.cc_distribution.get('enable', False):
            self.handles['cc'] = api_register.initialize_hook(*create_hooks(context=self.cc_context, monitor=self))
            api_register.redirect_api()
        self.monitoring = True

    def adhoc_check(self, target_tensor: torch.tensor, module_name: str, tensor_name: str, rank_list, ops_list):
        rank = None
        if dist.is_initialized():
            rank = dist.get_rank()
            if (rank not in rank_list) and len(rank_list) != 0:
                return
        self.tensor_metrics.stat_insert(target_tensor, ops_list, module_name, tensor_name, rank)

    def build_tbtag_tensor_map(self, module_name, suffix, tag, tensor):
        """
        :param module_name: str of module name
        :param suffix:
        :param tag:
        :param tensor: torch.tensor or tuple/list of torch.tensor
        :return: tensor_map
        """
        tensor_map = {}
        if isinstance(tensor, torch.Tensor):
            tensor = [tensor]
        if isinstance(tensor, (list, tuple)):
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

    def generate_param_map(self, tag, param_tensor):
        metrics = {}
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

    def generate_mv_metrics(self, opt_context):
        if not self.mv_distribution:
            return
        opt_context.exp_avg_metric = {}
        opt_context.exp_avg_sq_metric = {}
        m_tag_tensor_map = self.generate_param_map(MonitorConst.EXP_AVG, opt_context.param_exp_avg)
        v_tag_tensor_map = self.generate_param_map(MonitorConst.EXP_AVG_SQ, opt_context.param_exp_avg_sq)
        get_metrics(self.ops, m_tag_tensor_map, self.eps, opt_context.exp_avg_metric)
        get_metrics(self.ops, v_tag_tensor_map, self.eps, opt_context.exp_avg_sq_metric)

    def generate_wgrad_metrics(self, post_grad_dict):
        if not self.wg_distribution:
            return {}, {}

        get_metrics(self.ops, post_grad_dict, self.eps, self.grad_context.post)
        return self.grad_context.post, self.grad_context.pre

    def generate_xy_metrics(self):
        actv = {}
        for fwd_context in self.module_fwd_hook_context_by_module.values():
            actv.update(fwd_context.actv)

        actv_grad = self.grad_context.actv
        return actv, actv_grad

    def reload_xy(self, xy_distribution=False):
        logger.warning(
            "reload_xy() is deprecated and will be removed in a future version. Use DYNAMIC_MONITOR instead."
        )
        self.xy_distribution = xy_distribution  # pylint: disable=attribute-defined-outside-init

        for handle in self.handles['xy']:
            handle.remove()
        self.handles['xy'].clear()
        self.hook_modules()
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            fwd_context.actv.clear()

    def write_adhoc_check(self, step):
        self.tensor_metrics.flush(self.summary_writer)

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

    def write_metrics_if_not_empty(self, features, metrics, step, hook_name):
        if not features or len(features) == 0:
            return
        use_micro_step = hook_name not in ["linear_hook"]
        self.summary_writer.write_metrics(metrics, features, step, hook_name, use_micro_step=use_micro_step)
        features.clear()

    def write_features_tb(self, step):
        if not self.recording_l2_features:
            return
        for context in self.feature_hook_context_by_module.values():
            num_features = len(context.attention_feature) + len(context.linear_feature)
            if num_features == 0:
                continue
            self.write_metrics_if_not_empty(
                context.attention_feature, ["entropy", "softmax_max"], step, "attention_hook"
            )
            self.write_metrics_if_not_empty(context.linear_feature, ["sr", "kernel_norm"], step, "linear_hook")

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

    def hook_optimizer(self, optimizer):
        # in DDP by default use params_have_main_grad
        def optimizer_pre_step_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]

            if (
                self.print_struct
                and not all(value == {} for value in self.module_struct.values())
                and not self.struct_printed
            ):
                self._save_module_struct()
                if not self.cc_log_only:
                    raise Exception("exit after first monitor step when print model struct")  # pylint: disable=broad-exception-raised
            if self.cc_log_only and context.step > 0:
                self._smallest_rank_print("> Used communication ops and corresponding stack")
                self._smallest_rank_print(
                    json.dumps({k: [i.split(';') for i in v] for k, v in self.cc_logged_stack.items()})
                )
                raise Exception("exit after first step when print cc stack")  # pylint: disable=broad-exception-raised

            # skip generate metrics
            if context.step < self.start_step or (context.step - self.start_step) % self.step_interval != 0:
                return

            grad_dict = {}
            if self.wg_distribution:
                grad_dict = self.optimizer_mon.fetch_grad(self, self.param2name)

            mv_result = None
            if self.mv_distribution or self.ur_distribution or self.mg_direction:
                mv_result = self.optimizer_mon.fetch_mv(self, self.param2name)
            if mv_result:
                context.param_exp_avg = mv_result.exp_avg
                context.param_exp_avg_sq = mv_result.exp_avg_sq
                context.param_adam_update = mv_result.update
                context.param_adam_ratio = mv_result.ratio

            _, _ = self.generate_wgrad_metrics(grad_dict)
            self.generate_mv_metrics(context)
            self.generate_param_metrics(context, MonitorConst.PRE_PARAM)

            tbtag_tensor_map = {}
            if self.mg_direction:
                for param, name in self.param2name.items():
                    grad = param.main_grad if self.params_have_main_grad else param.grad
                    if grad is None:
                        logger.warning(f"grad is None: {name}, maybe something wrong happened.")
                        continue
                    if context.step == 0:
                        same_direction_ratio = torch.tensor(1.0)
                    else:
                        same_direction_ratio = get_sign_matches(grad, context.param_exp_avg[name])
                    context.param_mg_direction[name] = same_direction_ratio
                tbtag_tensor_map.update(self.generate_param_map('mg_direction', context.param_mg_direction))

            metric_dict = {}
            get_metrics(self.ops, tbtag_tensor_map, self.eps, metric_dict)
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

        if self.optimizer_hooked:
            return

        self.pre_step_hooks.append(optimizer_pre_step_hook)
        self.post_step_hooks.append(optimizer_post_step_hook)

        self.optimizer_hooked = True
        return

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
                # 动态启停时不受原start_step影响，永远从下一步开始
                self.start_step = context.step  # pylint: disable=attribute-defined-outside-init
                logger.warning(
                    f"config is updated at step{context.step - 1}, will start new hook at step{context.step}."
                )
            except Exception as e:
                logger.error(f"set config wrong because {e}, not updated, please check!!!")
                return

            self._remove_all_hooks(optimizer)
            self.register_hooks(optimizer)

    def hook_step_final(self, optimizer):
        def step_final_hook(optimizer, args, kwargs):
            context = self.optimizer_context[optimizer]
            rank = dist.get_rank() if dist.is_initialized() else None
            # 静态在第0步就可以保存, 动态在第0步不可以, 因为动态设计的就是重置后下一步开启, 第0步的self.monitoring还是False
            if self.monitoring:
                module_rank_valid = not self.module_rank_list or (
                    dist.is_initialized() and dist.get_rank() in self.module_rank_list
                )
                step_condition = (
                    context.step >= self.start_step and (context.step - self.start_step) % self.step_interval == 0
                )
                if module_rank_valid and step_condition:
                    self.has_collect_times += 1

                    if self.anomaly_data_factory:
                        self.anomaly_data_factory.set_call_id(self.param_name_call_id)
                    self.write_xy_tb(context.step)
                    self.write_features_tb(context.step)
                    self.write_grad_tb(context.step)
                    self.write_mv_tb(context)
                    self.write_param_tb(context)
                    self.write_adhoc_check(context.step)
                    if self.stack_info:
                        self.write_stack_info()
                        self.stack_info = False  # pylint: disable=attribute-defined-outside-init
                        for handle in self.handles["stack"]:
                            handle.remove()
                        self.handles["stack"].clear()

                    if self.ur_distribution:
                        for param_name, _ in context.param_adam_update.items():
                            self.update_heatmap_visualizer[param_name].visualize(
                                get_summary_writer_tag_name(param_name, 'adam_update', rank),
                                context.step,
                                self.summary_writer,
                            )
                        for param_name, _ in context.param_adam_ratio.items():
                            self.ratio_heatmap_visualizer[param_name].visualize(
                                get_summary_writer_tag_name(param_name, 'adam_ratio', rank),
                                context.step,
                                self.summary_writer,
                            )

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

        optimizer.step = patch_step(optimizer.step, optimizer)

    def hook_modules(self):
        if self.module_rank_list and (self.rank not in self.module_rank_list):
            return

        targets = self.config['targets']
        module_in_all_stage = [key for key in targets.keys() if MonitorConst.NAME_SEP not in key]
        for key in module_in_all_stage:
            struct = targets.pop(key)
            targets.update({f'{vpp_stage}{MonitorConst.NAME_SEP}{key}': struct for vpp_stage in range(len(self.model))})

        hooked_count = 0
        for vpp_stage, model_chunk in enumerate(self.model):
            vpp_stage = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            targets = (
                [x for x, _ in model_chunk.named_modules()] if self.print_struct else self.config['targets'].keys()
            )
            l2_target_names = self.config.get('l2_targets', '')
            hooked_count += self._hook_module(targets, l2_target_names, model_chunk, vpp_stage)

        logger.info_on_rank_0(f"> {hooked_count} modules are monitored.")

        @recursion_depth_decorator('msprobe.pytorch.monitor.clone_if_tensor')
        def clone_if_tensor(args):
            if isinstance(args, tuple):
                return tuple(clone_if_tensor(arg) for arg in args)
            elif isinstance(args, torch.Tensor):
                return args.clone()
            else:
                return args

        @torch.no_grad
        def wrap_hook_setup(setup):
            def wrapped_setup(*args, **kwargs):
                args = setup(*args, **kwargs)
                args = clone_if_tensor(args)
                return args

            return wrapped_setup

        BackwardHook.setup_input_hook = wrap_hook_setup(BackwardHook.setup_input_hook)
        BackwardHook.setup_output_hook = wrap_hook_setup(BackwardHook.setup_output_hook)
        return

    def register_param_call_id(self, hook_name: str, key: str):
        """
        :param hook_name:
        :param key: str, '0:relu_0/output_grad'
        :return:
        """
        logger.debug(f"{hook_name} {key}: {self.call_id}")
        self.param_name_call_id[key] = self.call_id
        self.call_id += 1

    def _remove_all_hooks(self, optimizer):
        # 清空hook handle
        for handle in self.handles['xy']:
            handle.remove()
        self.handles['xy'].clear()
        for handle in self.handles['L2_features']:
            handle.remove()
        self.handles['L2_features'].clear()
        # 清空对应context缓存
        for _, fwd_context in self.module_fwd_hook_context_by_module.items():
            fwd_context.reset()
        for _, bwd_context in self.module_bwd_hook_context_by_module.items():
            bwd_context.reset()
        self.grad_context.reset()  # 权重梯度和激活值梯度都在这

        # 还原梯度patch
        self.optimizer_mon.restore_grad_sync(self)
        if self.fsdp_post_backward_hook:  # fsdp
            torch.distributed.fsdp._runtime_utils._post_backward_hook = self.fsdp_post_backward_hook
            logger.info("remove patch_post_backward_hook in fsdp.")
        if self.fsdp2_foreach_reduce:  # fsdp2
            torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce = self.fsdp2_foreach_reduce
            importlib.reload(torch.distributed.fsdp._fully_shard._fsdp_param_group)
            logger.info("remove patch_foreach_reduce_hook in fsdp2.")
        else:  # not megatron and not fsdp
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
        self.name2indices.clear()
        self.name2param.clear()
        self.duplicate_param.clear()
        self.name2tag.clear()
        self.module_struct.clear()
        self.grad_accs.clear()

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

    def _smallest_rank_print(self, msg):
        if dist.is_initialized():
            if self.module_rank_list:
                if dist.get_rank() == min(self.module_rank_list):
                    logger.info(msg)
            else:
                if dist.get_rank() == 0:
                    logger.info(msg)
        else:
            logger.info(msg)

    def _save_module_struct(self):
        output_dir = os.path.join(get_output_base_dir(), 'module_struct', f'rank{self.rank}')
        create_directory(output_dir)
        module_struct_file = os.path.realpath(os.path.join(output_dir, 'module_struct.json'))
        save_json(module_struct_file, self.module_struct, indent=2)
        logger.info(f"> save module struct to {module_struct_file}")
        self.struct_printed = True

    def _is_target_param(self, param_name, param, prefix):
        name = prefix + param_name
        squash_name = prefix + squash_param_name(param_name, self.squash_name)
        for target in self.config['targets'].keys():
            if param_name.startswith(target) or squash_name.startswith(target) or name.startswith(target):
                return True

        return False

    def _register_chunk(self, model_chunk, prefix):
        if isinstance(model_chunk, FSDP):
            if not model_chunk._use_orig_params:
                raise ValueError("Only Support fsdp1 with use_orig_params=True")
            self.fsdp_wrapped_module = True
        for param_name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            if not self.fsdp2_wrapped_module and param.__class__.__name__ == "DTensor":
                self.fsdp2_wrapped_module = True
            if self.fsdp_wrapped_module:  # FSDP1需要记录完整的不被target限制的flat权重名，以供后续对flat解包
                self.fsdp_param_name_map[id(param)] = param_name

            if self._is_target_param(param_name, param, prefix):
                name = prefix + squash_param_name(param_name, self.squash_name)
                if name in self.param2name.values():
                    name = prefix + param_name
                self.param2name[param] = name
                self.name2param[name] = param
                self.origin2squash[param_name] = name

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

    def _register_param_name(self):
        for vpp_stage, model_chunk in enumerate(self.model):
            prefix = f'{vpp_stage}{MonitorConst.NAME_SEP}'
            self._register_chunk(model_chunk, prefix)

    def _is_target_module(self, module_name, targets, vpp_stage):
        if self.all_xy or self.print_struct:
            return vpp_stage + squash_param_name(module_name, self.squash_name)
        for pattern in [
            vpp_stage + squash_param_name(module_name, self.squash_name),
            vpp_stage + module_name,
        ]:
            if pattern in targets:
                return pattern
        return ""

    def _is_recording_module(self, module_name, l2_targets, vpp_stage, hook_name):
        if len(l2_targets) > 0:
            for pattern in [
                vpp_stage + squash_param_name(module_name, self.squash_name),
                vpp_stage + module_name,
            ]:
                if pattern in l2_targets:
                    return pattern
        elif hook_name in ["linear_hook"]:
            return vpp_stage + squash_param_name(module_name, self.squash_name)
        return ""

    def _hook_module(self, target_names, l2_target_names, module: torch.nn.Module, vpp_stage=''):
        if '_modules' not in module.__dict__:
            # nothing to hook
            return 0

        def fwd_hook_fun(module, args, kwargs, module_output, name):
            if not module.training or is_recomputation():
                # 1 only monitor training stage.
                # 2 when open recompute, skip recomputed forward stage.
                return

            module_input = [tensor for tensor in args if torch.is_tensor(tensor)]
            if kwargs:
                kwargs_tensors = [tensor for tensor in kwargs.values() if torch.is_tensor(tensor)]
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

            tbtag_tensor_map = {}
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.INPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV,
                    module_input,
                )
            )
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.OUTPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTV,
                    module_output,
                )
            )

            get_metrics(self.ops, tbtag_tensor_map, self.eps, context.actv)
            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
            return

        def bwd_hook_fun(module, input_grad, output_grad):
            context: ModuleHookContext = self.module_bwd_hook_context_by_module[module]
            if not context.struct:
                context.struct = {
                    MonitorConst.INPUT_GRAD: get_param_struct(input_grad),
                    MonitorConst.OUTPUT_GRAD: get_param_struct(output_grad),
                }
            if self.print_struct:
                self.module_struct[context.module_name].update(context.struct)
                return

            tbtag_tensor_map = {}
            tbtag_tensor_map.update(
                self.build_tbtag_tensor_map(
                    f'{context.module_name}.{Const.INPUT}',
                    f'{MonitorConst.NAME_SEP}{context.micro_step}',
                    MonitorConst.ACTVGRAD,
                    input_grad,
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

            get_metrics(self.ops, tbtag_tensor_map, self.eps, self.grad_context.actv)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
            return

        def extract_attention_feature_hook(module, module_input, module_output, name):
            if is_recomputation() or not module.training:
                return

            if module not in self.feature_hook_context_by_module:
                self.feature_hook_context_by_module[module] = FeatureHookContext(name)
            context: FeatureHookContext = self.feature_hook_context_by_module[module]
            tbtag_tensor_map = {}
            if len(module_input) < 2:
                logger.warning(
                    f"Length of module_input in attention hook ({name}) is {len(module_input)}, "
                    "expected >= 2. Skipping feature extraction for this module."
                )
                return
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
            return

        def extract_linear_sr_hook(module, module_input, module_output, name):
            if is_recomputation() or not module.training:
                return
            weight_name = self.get_linear_hook_target(module)
            if weight_name == '':
                return

            if module not in self.feature_hook_context_by_module:
                self.feature_hook_context_by_module[module] = FeatureHookContext(name)
            context: FeatureHookContext = self.feature_hook_context_by_module[module]

            if context.micro_step == (self.micro_batch_number - 1):
                tbtag_tensor_map = {}
                value = getattr(module, weight_name).data
                tbtag_tensor_map.update(self.build_tbtag_tensor_map(f'{context.module_name}.linear', '', 'sr', value))
                get_sr_metric(tbtag_tensor_map, context.linear_feature)

            context.micro_step += 1
            if context.micro_step == self.micro_batch_number:
                context.micro_step = 0
                context.step += 1
            return

        def stack_hook(module, args, kwargs, module_output, name):
            if module not in self.module_fwd_hook_context_by_module:
                self.module_fwd_hook_context_by_module[module] = ModuleHookContext(name)
            context: ModuleHookContext = self.module_fwd_hook_context_by_module[module]
            context.stack = analyze_api_call_stack(name)

        if self.backward_only and self.forward_only:
            logger.warning('not enable backward_only and forward_only simultaneously')

        hooked_count = 0
        for module_name, submodule in module.named_modules():
            if self.stack_info:
                name = vpp_stage + squash_param_name(module_name, self.squash_name)
                handle = submodule.register_forward_hook(partial(stack_hook, name=name), with_kwargs=True)
                self.handles['stack'].append(handle)
            name = self._is_target_module(module_name, target_names, vpp_stage)
            if not name:
                continue
            if submodule.__class__.__name__ == "FullyShardedDataParallel":
                continue
            if self.xy_distribution or self.print_struct:
                if not self.backward_only:
                    handle = submodule.register_forward_hook(partial(fwd_hook_fun, name=name), with_kwargs=True)
                    self.handles['xy'].append(handle)
                if not self.forward_only and not self.has_register_backward_hook(name, submodule):
                    handle = submodule.register_full_backward_hook(bwd_hook_fun)
                    self.handles['xy'].append(handle)
                    self.module_bwd_hook_context_by_module[submodule] = ModuleHookContext(name)
                logger.info_on_rank_0(f"> {name} is monitored successfully")
                hooked_count += 1
        if not self.print_struct and self.recording_l2_features:
            for module_name, submodule in module.named_modules():
                func_map = {
                    "attention_hook": extract_attention_feature_hook,
                    "linear_hook": extract_linear_sr_hook,
                }
                for hook_name in func_map:  # pylint: disable=consider-using-dict-items
                    if hook_name not in l2_target_names:
                        continue
                    temp_names = l2_target_names[hook_name]
                    name = self._is_recording_module(module_name, temp_names, vpp_stage, hook_name)
                    if name:
                        handle = submodule.register_forward_hook(partial(func_map[hook_name], name=name))
                        print_feature_name = hook_name.split('_')[0]
                        logger.info_on_rank_0(f'> {print_feature_name} features of {name} is monitored successfully')
                        self.handles["L2_features"].append(handle)
                        hooked_count += 1
                continue

        return hooked_count

    def _patch_grad_sync(self):
        if not self.wg_distribution:
            return

        if self.fsdp_wrapped_module:
            # patch fsdp _runtime_utils._post_backward_hook
            self._patch_fsdp_post_backward_hook()
            return

        if self.fsdp2_wrapped_module:
            # patch fsdp2 _fully_shard._fsdp_collectives.foreach_reduce
            self._patch_fsdp2_foreach_reduce()
            return

        if self.monitor_mbs_grad:
            self._hook_weights()
            return

        self.optimizer_mon.patch_grad_sync(self)

        if self.enable_megatron or self.enable_deepspeed:
            return

        # default hook weights
        self._hook_weights()

    def _patch_fsdp_post_backward_hook(self):
        """
        FSDP runtime 需要处理整个forward和backward计算和通信的流程，通过override nn.Module的forward，定义相应的逻辑。
        对AccumulateGrad对象注册hook，可以在backward计算grad后立刻执行，在reduce_scatter操作前采集梯度累计后，通信聚合前的梯度。
        每个forward阶段，fsdp对AccumulateGrad重复注册hook方法，monitor工具内注册hook无法生效，
        因此对_post_backward_hook进行patch，在backward后，reduce_scatter前采集梯度。
        """

        def patch_post_backward_hook(_post_backward_hook):
            def wrapper(state, handle, *unused):
                if not hasattr(handle.flat_param, 'micro_step'):
                    setattr(handle.flat_param, 'micro_step', 0)

                handle_not_in_any_model = all(
                    model._all_handles is None or handle not in model._all_handles for model in self.model
                )
                if handle_not_in_any_model:
                    logger.debug(f"Skip handle not along to the monitor model list, handle id: {id(handle)}.")
                elif self.monitor_mbs_grad or (handle.flat_param.micro_step + 1 == self.micro_batch_number):
                    grad_dict = {}
                    offsets = handle._get_flat_param_offsets()
                    shapes = handle.flat_param._shapes
                    params = handle.flat_param._params
                    for param, (start, end), local_shape in zip(params, offsets, shapes):
                        param_name = self.fsdp_param_name_map.get(id(param), "")
                        tag = self.name2tag.get(self.origin2squash.get(param_name, ""), {}).get(MonitorConst.PRE_GRAD)
                        if tag is None:
                            continue
                        grad = handle.flat_param.grad[start : end + 1].reshape(local_shape)
                        if self.monitor_mbs_grad:
                            tag = tag.replace('/', f'{MonitorConst.NAME_SEP}{handle.flat_param.micro_step}/', 1)
                        grad_dict[tag] = grad
                        self.register_param_call_id("_post_backward_hook", tag)
                    get_metrics(self.ops, grad_dict, self.eps, self.grad_context.pre)
                handle.flat_param.micro_step = (handle.flat_param.micro_step + 1) % self.micro_batch_number
                out = _post_backward_hook(state, handle, *unused)
                return out

            return wrapper

        logger.info("Patch fsdp _post_backward_hook, collect pre_grad metrics.")
        self.fsdp_post_backward_hook = torch.distributed.fsdp._runtime_utils._post_backward_hook
        torch.distributed.fsdp._runtime_utils._post_backward_hook = patch_post_backward_hook(
            torch.distributed.fsdp._runtime_utils._post_backward_hook
        )

    def _patch_fsdp2_foreach_reduce(self):
        def patch_foreach_reduce(foreach_reduce):
            def wrapper(fsdp_params, unsharded_grads, *unused):
                if not hasattr(fsdp_params[0], 'micro_step'):
                    setattr(fsdp_params[0], 'micro_step', 0)

                if self.monitor_mbs_grad or (fsdp_params[0].micro_step + 1 == self.micro_batch_number):
                    grad_dict = {}
                    for param, grad in zip(fsdp_params, unsharded_grads):
                        tag = self.name2tag.get(self.origin2squash.get(param._param_fqn, ""), {}).get(
                            MonitorConst.PRE_GRAD
                        )
                        if tag is None:
                            continue
                        if self.monitor_mbs_grad:
                            tag = tag.replace('/', f'{MonitorConst.NAME_SEP}{fsdp_params[0].micro_step}/', 1)
                        grad_dict[tag] = grad
                        self.register_param_call_id("foreach_reduce", tag)
                    get_metrics(self.ops, grad_dict, self.eps, self.grad_context.pre)
                fsdp_params[0].micro_step = (fsdp_params[0].micro_step + 1) % self.micro_batch_number
                out = foreach_reduce(fsdp_params, unsharded_grads, *unused)
                return out

            return wrapper

        logger.info("Patch fsdp2 foreach_reduce, collect pre_grad metrics.")
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as _fsdp_param_group  # pylint: disable=consider-using-from-import
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_collectives  # pylint: disable=consider-using-from-import

        self.fsdp2_foreach_reduce = _fsdp_collectives.foreach_reduce
        _fsdp_collectives.foreach_reduce = patch_foreach_reduce(_fsdp_collectives.foreach_reduce)
        importlib.reload(_fsdp_param_group)  # 关键操作，不然会因为torch一开始就import foreach_reduce导致patch失效

    def _hook_weights(self):
        """
        遍历参数的梯度生成函数（grad_acc），并挂载hook，以便在该参数所有梯度计算后，采集通信聚合前梯度数据。
        """

        @torch.no_grad
        def param_hook(*args, param, name):
            key = name
            if self.monitor_mbs_grad:
                key += f'{MonitorConst.NAME_SEP}{param.micro_step}'

            key = get_summary_writer_tag_name(key, 'acc_grad', self.rank)
            self.register_param_call_id("param_hook", key)
            if self.monitor_mbs_grad or (param.micro_step + 1 == self.micro_batch_number):
                if self.params_have_main_grad:
                    grad = param.main_grad
                else:
                    grad = param.grad
                get_metrics(self.ops, {key: grad.clone()}, self.eps, self.grad_context.pre)
            param.micro_step = (param.micro_step + 1) % self.micro_batch_number

        logger.info("hooking weight grads.")
        for param, name in self.param2name.items():
            setattr(param, 'micro_step', 0)
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            handle = grad_acc.register_hook(partial(param_hook, param=param, name=name))
            self.grad_accs.append(grad_acc)
            self.handles['wgrads'].append(handle)
        self.weight_hooked = True
