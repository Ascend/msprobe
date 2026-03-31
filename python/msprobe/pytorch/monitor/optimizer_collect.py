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

from abc import abstractmethod

import torch

from msprobe.pytorch.common.log import logger
from msprobe.core.monitor.utils import MVResult
from msprobe.pytorch.monitor.module_metric import get_metrics
from msprobe.core.common.const import MonitorConst


class OptimizerMon(object):
    def __init__(self, torch_opt) -> None:
        self.fp16_to_fp32_param = {}
        self.torch_opt = torch_opt
        self.state = {}
        self.origin_funcs = []
        self.bucket_class = None

    def narrow_from_flatten(self, param, flatten_state):
        return flatten_state

    def get_state(self, torch_opt):
        if hasattr(torch_opt, 'chained_optimizers'):
            for opt in torch_opt.chained_optimizers:
                self._get_single_state(opt)
        else:
            self._get_single_state(torch_opt)

    def fetch_grad(self, monitor, params2name):
        if not self.fp16_to_fp32_param:
            self.map_fp16_to_fp32_param(self.torch_opt)

        grad_dict = {}
        first_param = True
        for param, name in params2name.items():
            if monitor.duplicate_param.get(name, False):
                continue
            if self.fp16_to_fp32_param and param not in self.fp16_to_fp32_param:
                continue
            grad = param.main_grad if monitor.params_have_main_grad else param.grad
            if grad.__class__.__name__ == 'DTensor':
                grad = grad.to_local()
            element_in_cur_partition = self.fp16_to_fp32_param.get(param, param).numel()
            if param.numel() != element_in_cur_partition:
                if first_param:
                    grad = grad.flatten()[-element_in_cur_partition:]
                else:  # supposed to be the last one
                    grad = grad.flatten()[:element_in_cur_partition]
            first_param = False

            if grad is None:
                if not monitor.fsdp_wrapped_module:
                    logger.warning(f"grad is None: {name}, maybe something wrong happened.")
                continue
            tag = monitor.name2tag.get(name, {}).get(MonitorConst.POST_GRAD)
            monitor.register_param_call_id("hook_optimizer", tag)
            grad_dict[tag] = grad
        return grad_dict

    def map_fp16_to_fp32_param(self, torch_opt):
        pass

    def fetch_mv(self, monitor, params2name):
        if not self.fp16_to_fp32_param:
            self.map_fp16_to_fp32_param(self.torch_opt)
        if not self.state:
            self.get_state(self.torch_opt)

        exp_avg_dict = {}
        exp_avg_sq_dict = {}
        update_dict = {}
        ratio_dict = {}

        if not self.state:
            logger.warning('optimizer state can not accessed')
            return MVResult(exp_avg=exp_avg_dict, exp_avg_sq=exp_avg_sq_dict, update=update_dict, ratio=ratio_dict)

        for lp_param, name in params2name.items():
            if lp_param in self.fp16_to_fp32_param:
                hp_param = self.fp16_to_fp32_param[lp_param]
            else:
                hp_param = lp_param

            if hp_param in self.state:
                state_param = self.state.get(hp_param, {})
                exp_avg = self.narrow_from_flatten(lp_param, state_param.get("exp_avg", None))
                exp_avg_sq = self.narrow_from_flatten(lp_param, state_param.get("exp_avg_sq", None))
                if monitor.mv_distribution:
                    exp_avg_dict[name] = exp_avg
                    exp_avg_sq_dict[name] = exp_avg_sq
                if monitor.mg_direction:
                    exp_avg_dict[name] = exp_avg
                if monitor.ur_distribution:
                    if len(self.torch_opt.param_groups) > 1:
                        logger.info(f"the length of torch_opt.param_groups is {len(self.torch_opt.param_groups)}.")
                    if 'step' in state_param:
                        step = state_param['step']  # Optimizer from pytorch or FusedAdam from apex(used by megatron)
                    elif 'step' in self.torch_opt.param_groups[0]:
                        step = self.torch_opt.param_groups[0]['step']  # AdamW from mindspeed
                    else:
                        logger.warning(f"step of {name} is None, maybe something wrong happened.")
                        continue
                    if exp_avg is None or exp_avg_sq is None:
                        logger.warning(f"exp_avg or exp_avg_sq of {name} is None, skip calculation.")
                        continue
                    exp_avg_hat = exp_avg / (1 - self.torch_opt.defaults['betas'][0] ** step)
                    exp_avg_sq_hat = exp_avg_sq / (1 - self.torch_opt.defaults['betas'][1] ** step)
                    update_dict[name] = exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + self.torch_opt.defaults['eps'])
                    ratio_dict[name] = exp_avg_hat / torch.sqrt(exp_avg_sq_hat)
                    monitor.update_heatmap_visualizer[name].pre_cal(update_dict[name])
                    monitor.ratio_heatmap_visualizer[name].pre_cal(ratio_dict[name])
        return MVResult(exp_avg=exp_avg_dict, exp_avg_sq=exp_avg_sq_dict, update=update_dict, ratio=ratio_dict)

    def patch_grad_sync(self, monitor):
        def patch_sync(sync_grad_func):
            def wrapper(bucket):
                grad_dict = {}
                # Megatron between core_r0.6.0 and core_r0.8.0, this bucket is Bucket.
                # When megatron is core_r0.9.0, this bucket is _ParamAndGradBucketGroup.
                # In megatron version core_r0.9.0, func start_grad_sync from Bucket moved to _ParamAndGradBucketGroup.
                if hasattr(bucket, "params"):
                    param_attr = bucket.params
                elif hasattr(bucket, "params_list"):
                    param_attr = bucket.params_list
                else:
                    raise AttributeError(
                        f"class {bucket.__class__.__name__} has no attribute 'params' or 'params_list', please check!")
                bucket_params_id_list = [id(params) for params in param_attr]
                for param, name in monitor.param2name.items():
                    if id(param) not in bucket_params_id_list:
                        continue
                    grad = param.main_grad if monitor.params_have_main_grad else param.grad
                    if grad is None:
                        logger.warning(f"grad is None: {name}, maybe something wrong happened.")
                        continue
                    tag = monitor.name2tag.get(name, {}).get(MonitorConst.PRE_GRAD)
                    if tag is None:
                        continue
                    grad_dict[tag] = grad
                    monitor.register_param_call_id("sync_grad_func", tag)
                get_metrics(monitor.ops, grad_dict, monitor.eps, monitor.grad_context.pre)
                out = sync_grad_func(bucket)
                return out

            return wrapper

        try:
            from megatron.core.distributed.param_and_grad_buffer import Bucket
            self.origin_funcs.append(Bucket.start_grad_sync)
            self.bucket_class = Bucket
            Bucket.start_grad_sync = patch_sync(Bucket.start_grad_sync)
            monitor.enable_megatron = True
            logger.info("megatron version is >= core_r0.6.0 <= core_r0.8.0")
        except ImportError:
            monitor.enable_megatron = False

        try:
            from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup
            self.origin_funcs.append(_ParamAndGradBucketGroup.start_grad_sync)
            self.bucket_class = _ParamAndGradBucketGroup
            _ParamAndGradBucketGroup.start_grad_sync = patch_sync(_ParamAndGradBucketGroup.start_grad_sync)
            monitor.enable_megatron = True
            logger.info("megatron version is > core_r0.8.0 <= core_r0.9.0")
        except ImportError:
            monitor.enable_megatron = False | monitor.enable_megatron

    def restore_grad_sync(self, monitor):
        if not monitor.enable_megatron:
            return

        self.bucket_class.start_grad_sync = self.origin_funcs[0]

    def _get_single_state(self, torch_opt):
        state = {}
        if hasattr(torch_opt, 'param_to_cpu_states_map'):
            state = torch_opt.param_to_cpu_states_map
        elif hasattr(torch_opt, 'state'):
            state = torch_opt.state
        elif hasattr(torch_opt, 'optimizer') and hasattr(torch_opt.optimizer, 'state'):
            state = torch_opt.optimizer.state
        self.state.update(state)


class MegatronMixPrecisionOptimizerMon(OptimizerMon):
    """
    混合精度优化器监测类。在混合精度训练中监测和管理优化器。
    混合精度训练通过适当降低某些计算的精度来加速训练过程并减少内存消耗。
    """

    def map_fp16_to_fp32_param(self, torch_opt):
        if not (hasattr(torch_opt, "float16_groups") and hasattr(torch_opt, "fp32_from_float16_groups")):
            logger.warning(
                "megatron class Float16OptimizerWithFloat16Params should have float16_groups and "
                "fp32_from_float16_groups, please check Megatron-LM version.")
        else:
            for fp16_group, fp32_group in zip(torch_opt.float16_groups, torch_opt.fp32_from_float16_groups):
                for fp16_param, fp32_param in zip(fp16_group, fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = fp32_param


class MegatronDistributedOptimizerMon(OptimizerMon):
    def map_fp16_to_fp32_param(self, torch_opt):
        if not (hasattr(torch_opt, "model_float16_groups") and hasattr(torch_opt, "shard_fp32_from_float16_groups")):
            logger.warning(
                "megatron class DistributedOptimizer should have model_float16_groups and "
                "shard_fp32_from_float16_groups, please check Megatron-LM version.")
        else:
            for fp16_group, shard_fp32_group in zip(torch_opt.model_float16_groups,
                                                    torch_opt.shard_fp32_from_float16_groups):
                for fp16_param, shard_fp32_param in zip(fp16_group, shard_fp32_group):
                    self.fp16_to_fp32_param[fp16_param] = shard_fp32_param


class MegatronChainedDistributedOptimizerMon(MegatronDistributedOptimizerMon):
    def map_fp16_to_fp32_param(self, torch_opt):
        for opt in torch_opt.chained_optimizers:
            super().map_fp16_to_fp32_param(opt)


class MegatronChainedMixPrecisionOptimizerMon(MegatronMixPrecisionOptimizerMon):
    def map_fp16_to_fp32_param(self, torch_opt):
        for opt in torch_opt.chained_optimizers:
            super().map_fp16_to_fp32_param(opt)


class DeepSpeedZeroOptimizerMon(OptimizerMon):
    """
    Base monitor class for DeepSpeed ZeRO optimizer.
    ZeRO stage 0 no partition
    ZeRO stage 1 partitions optimizer states across data parallel processes.
    ZeRO stage 2 additionally partitions gradients.
    ZeRO stage 3 additionally partitions parameters.

    This class provides monitoring capabilities for ZeRO optimizers by:
    - Handling gradient collection for different ZeRO stages
    - Managing optimizer state access for monitoring
    """

    def __init__(self, torch_opt):
        super().__init__(torch_opt)
        self.stage = ''
        self.bit16_groups = []
        self.fp32_flat_groups = []
        self.param2group = ()
        self.param2index = []
        self.group_offset = {}

    @abstractmethod
    def get_grad_for_param(self, lp_param, group_idx, param_id):
        raise NotImplementedError

    def param_not_in_partition(self, lp_param, group_idx):
        param_slice_mapping = self.torch_opt.state_dict()['param_slice_mappings'][group_idx]
        hp_address = param_slice_mapping.get(self.torch_opt.param_names.get(lp_param))
        return hp_address is None

    def get_position(self, lp_param, group_idx):
        param_slice_mapping = self.torch_opt.state_dict()['param_slice_mappings'][group_idx]
        hp_address = param_slice_mapping.get(self.torch_opt.param_names.get(lp_param))
        return hp_address.start, hp_address.numel

    def get_group_index(self):
        param2group = {}
        for group_idx, bit16_group in enumerate(self.bit16_groups):
            for param in bit16_group:
                param2group[param] = group_idx
        return param2group

    def get_param_index(self, lp_param, group_idx):
        if not self.param2index:
            for group in self.bit16_groups:
                param2index = {}
                for index, param in enumerate(group):
                    param2index[param] = index
                self.param2index.append(param2index)

        return self.param2index[group_idx][lp_param]

    def narrow_from_flatten(self, param, flatten_state):
        if flatten_state is None:
            return flatten_state
        group_idx = self.param2group[param]
        if self.param_not_in_partition(param, group_idx):
            return None
        start, numel = self.get_position(param, group_idx)
        return flatten_state.narrow(0, start, numel)

    def map_fp16_to_fp32_param(self, torch_opt):
        for group_idx, group in enumerate(self.bit16_groups):
            for param in group:
                self.fp16_to_fp32_param[param] = self.fp32_flat_groups[group_idx]

    def fetch_grad(self, monitor, params2name):
        grad_dict = {}
        for lp_param, name in params2name.items():
            group_idx = self.param2group[lp_param]
            param_id = self.get_param_index(lp_param, group_idx)
            if self.param_not_in_partition(lp_param, group_idx):
                continue
            if self.stage == '1or2':
                param_id = param_id - self.group_offset[group_idx] - 1
            grad = self.get_grad_for_param(lp_param, group_idx, param_id)
            tag = monitor.name2tag.get(name, {}).get(MonitorConst.POST_GRAD)
            monitor.register_param_call_id("hook_optimizer", tag)
            grad_dict[tag] = grad

        return grad_dict

    def patch_grad_sync(self, monitor):
        pass

    def restore_grad_sync(self, monitor):
        pass


class DeepSpeedZeroOptimizerStage0Mon(DeepSpeedZeroOptimizerMon):
    def __init__(self, torch_opt):
        super().__init__(torch_opt)
        self.stage = '0'
        self.bit16_groups = torch_opt.bf16_groups
        self.fp32_flat_groups = torch_opt.fp32_groups_flat_partition
        self.param2group = self.get_group_index()

    def get_grad_for_param(self, lp_param, group_idx, param_id):
        return self.torch_opt.fp32_groups_gradient_dict[group_idx][param_id]


class DeepSpeedZeroOptimizerStage1or2Mon(DeepSpeedZeroOptimizerMon):
    def __init__(self, torch_opt):
        super().__init__(torch_opt)
        self.stage = '1or2'
        self.bit16_groups = torch_opt.bit16_groups
        self.fp32_flat_groups = torch_opt.single_partition_of_fp32_groups
        self.param2group = self.get_group_index()
        self.group_offset = {}
        self.get_group_offset()

    def get_grad_for_param(self, lp_param, group_idx, param_id):
        if getattr(self.torch_opt, "cpu_offload", False):
            grads = self.torch_opt.single_partition_of_fp32_groups[group_idx].grad
            start, numel = self.get_position(lp_param, group_idx)
            grad = grads.narrow(0, start, numel)
        else:
            grad = self.torch_opt.averaged_gradients[group_idx][param_id]
        return grad

    def get_group_offset(self):
        for group_idx, group in enumerate(self.bit16_groups):
            self.group_offset[group_idx] = -1
            for lp_param in group:
                if self.param_not_in_partition(lp_param, group_idx):
                    self.group_offset[group_idx] = self.get_param_index(lp_param, group_idx)
                else:
                    break

    def patch_grad_sync(self, monitor):
        def patch_sync(reduce_func):
            def wrapper(zero_optimizer, *args, **kwargs):
                grad_dict = {}
                for i, param, _ in zero_optimizer.params_in_ipg_bucket:
                    if isinstance(param, int):  # for ds >= 0.17.0
                        param = zero_optimizer.bit16_groups[i][param]
                    name = monitor.param2name[param]
                    tag = monitor.name2tag.get(name, {}).get(MonitorConst.PRE_GRAD)
                    grad_dict[tag] = zero_optimizer.get_gradient_for_reduction(param)
                    monitor.register_param_call_id("sync_grad_func", tag)
                get_metrics(monitor.ops, grad_dict, monitor.eps, monitor.grad_context.pre)
                out = reduce_func(zero_optimizer, *args, **kwargs)
                return out

            return wrapper

        try:
            from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
            self.origin_funcs = [
                DeepSpeedZeroOptimizer.average_tensor,
                DeepSpeedZeroOptimizer.buffered_reduce_fallback
            ]
            DeepSpeedZeroOptimizer.average_tensor = patch_sync(DeepSpeedZeroOptimizer.average_tensor)
            DeepSpeedZeroOptimizer.buffered_reduce_fallback = \
                patch_sync(DeepSpeedZeroOptimizer.buffered_reduce_fallback)
            monitor.enable_deepspeed = True
            logger.info('deepspeed enabled')
        except Exception as e:
            monitor.enable_deepspeed = False | monitor.enable_deepspeed
            logger.warning('Seems using deepspeed zero 1 or 2. But patch average tensor failed')

    def restore_grad_sync(self, monitor):
        if not monitor.enable_deepspeed:
            return

        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        DeepSpeedZeroOptimizer.average_tensor = self.origin_funcs[0]
        DeepSpeedZeroOptimizer.buffered_reduce_fallback = self.origin_funcs[1]


class DeepSpeedZeroOptimizerStage3Mon(DeepSpeedZeroOptimizerMon):
    def __init__(self, torch_opt):
        super().__init__(torch_opt)
        self.stage = '3'
        self.bit16_groups = torch_opt.fp16_groups
        self.fp32_flat_groups = torch_opt.fp32_partitioned_groups_flat
        self.param2group = self.get_group_index()

    def param_not_in_partition(self, lp_param, group_idx):
        """Each param partioned across all zero ranks"""
        return False

    def get_position(self, lp_param, group_idx):
        param_id = self.torch_opt.get_param_id(lp_param)
        return self.torch_opt.grad_position[param_id][1:]

    def get_grad_for_param(self, lp_param, group_idx, param_id):
        return self.torch_opt.averaged_gradients[group_idx][param_id]


class OptimizerMonFactory:
    _optimizer_mon_map = {
        "FP32Optimizer": OptimizerMon,
        "Float16OptimizerWithFloat16Params": MegatronMixPrecisionOptimizerMon,
        "DistributedOptimizer": MegatronDistributedOptimizerMon,
        "SwapDistributedOptimizer": MegatronDistributedOptimizerMon,
        "ChainedDistributedOptimizer": MegatronChainedDistributedOptimizerMon,
        "ChainedSwapDistributedOptimizer": MegatronChainedDistributedOptimizerMon,
        "ChainedFloat16OptimizerWithFloat16Params": MegatronChainedMixPrecisionOptimizerMon,
        "BF16_Optimizer": DeepSpeedZeroOptimizerStage0Mon,
        "DeepSpeedZeroOptimizer": DeepSpeedZeroOptimizerStage1or2Mon,
        "DeepSpeedZeroOptimizer_Stage3": DeepSpeedZeroOptimizerStage3Mon,
        "Adam": OptimizerMon
    }

    @staticmethod
    def create_optimizer_mon(optimizer):
        # auto replace opt_ty
        optimizer_class = optimizer.__class__.__name__
        if optimizer_class == "ChainedOptimizer":
            optimizer_class = "Chained" + optimizer.chained_optimizers[0].__class__.__name__
        logger.info(f'The optimizer type is {optimizer_class}')

        optimizer_mon_class = OptimizerMonFactory._optimizer_mon_map.get(optimizer_class, OptimizerMon)
        return optimizer_mon_class(optimizer)
