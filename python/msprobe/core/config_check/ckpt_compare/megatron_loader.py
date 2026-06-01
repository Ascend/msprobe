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
from collections import defaultdict
from typing import Dict
import numpy as np
from msprobe.core.common.log import logger
from msprobe.core.common.decorator import recursion_depth_decorator
from msprobe.core.common.const import Const
from msprobe.core.common.file_utils import FileOpen, load_yaml, check_link
from msprobe.core.common.framework_adapter import FmkAdp

# both weights and bias are partitioned in column parallel
COLUMN_PARALLEL_PARAMS = ['linear_qkv', 'linear_fc1', 'word_embeddings.weight', 'output_layer.weight']
# only weights are partitioned in column parallel
ROW_PARALLEL_PARAMS = ['linear_fc2.weight', 'linear_proj.weight']
ARGS = 'args'
LAYER_IDX_PATTERN = re.compile(r'layers\.(\d+)\.')
EXPERT_IDX_PATTERN = re.compile(r'experts\.(\d+)\.')
ITER_DIR_PATTERN = re.compile(r'iter_(\d{7})')


@recursion_depth_decorator('')
def _get_parameter(weights, prefix=''):
    for k, v in weights.items():
        name = Const.SEP.join([prefix, k]).strip(Const.SEP)
        if isinstance(v, dict):
            yield from _get_parameter(v, prefix=name)
        elif FmkAdp.is_tensor(v):
            yield name, FmkAdp.asnumpy(v)


def _map_to_mcore_local_names(param_name: str) -> str:
    """Map parameter names to mcore + local transformer implementation names."""
    mcore_local_map = load_yaml(os.path.join(os.path.dirname(__file__), 'name_mapping.yaml'))
    for other_name, mcore_local_name in mcore_local_map.items():
        param_name = param_name.replace(other_name, mcore_local_name)

    return param_name


def _parse_real_layer_idx(param_name, num_layers_per_stage, pp_size, pp_rank):
    """Map local (virtual) pipeline stage layer index to global layer index.

    For virtual pipeline parallel, each pipeline stage is further divided into virtual stages.
    The global layer index needs to account for both pipeline stage and virtual stage.

    Args:
        param_name (str): Parameter name containing layer index: layers.x.<submodule_name>/<vpp_stage>
        num_layers_per_stage (int): Number of layers per pipeline stage
        pp_size (int): Pipeline parallel size

    Returns:
        int: Global layer index accounting for both pipeline and virtual pipeline stages
    """
    # Extract local layer index from parameter name
    layer_match = re.search(LAYER_IDX_PATTERN, param_name)
    param_name, vpp_stage = param_name.split(Const.SCOPE_SEPARATOR)
    if not layer_match:
        return param_name

    local_layer_idx = int(layer_match.group(1))
    vpp_stage = int(vpp_stage)

    # Calculate global layer index based on pipeline stage and virtual stage
    real_layer_idx = local_layer_idx + (pp_size * vpp_stage + pp_rank) * num_layers_per_stage

    return param_name.replace(f'layers.{local_layer_idx}', f'layers.{real_layer_idx}')


def _parse_real_expert_idx(param_name, num_experts_per_rank, exp_rank):
    """Map local expert index to global expert index.

    For expert parallel, experts are distributed across ranks. This function maps
    the local expert index on a rank to its global index across all ranks.

    Args:
        param_name (str): Parameter name containing local expert index
        num_experts_per_rank (int): Number of experts on each rank
        exp_rank (int): Expert parallel rank

    Returns:
        str: Parameter name with local expert index replaced by global expert index
    """
    # Extract local layer index from parameter name
    expert_match = re.search(EXPERT_IDX_PATTERN, param_name)
    if not expert_match:
        return param_name

    local_expert_idx = int(expert_match.group(1))
    # Calculate global layer index based on pipeline stage and virtual stage
    real_experts_idx = local_expert_idx + exp_rank * num_experts_per_rank

    return param_name.replace(f'experts.{local_expert_idx}', f'experts.{real_experts_idx}')


def _consolidate_tp_weights(weights: Dict) -> Dict:
    """Consolidate weights from different tensor parallel ranks into combined tensors.

    Args:
        weights: Dictionary of weights with rank information in keys

    Returns:
        Dict: Consolidated weights without rank information
    """
    consolidated = {}
    for key, tensors in weights.items():
        if any(name in key for name in COLUMN_PARALLEL_PARAMS):
            # Column parallel - concatenate along input dimension (dim 0)
            combined = np.concatenate(tensors, axis=0)
        elif any(name in key for name in ROW_PARALLEL_PARAMS):
            # Row parallel - concatenate along output dimension (dim 1)
            combined = np.concatenate(tensors, axis=1)
        else:
            # For other params, verify identical and use first
            if not all(np.allclose(tensors[0], t) for t in tensors[1:]):
                logger.warning(f"Inconsistent values for {key} across TP ranks")
            combined = tensors[0]

        consolidated[key] = combined
    return consolidated


def _parse_num_layers_per_stage(tp_partition):
    match = [re.findall(LAYER_IDX_PATTERN, key) for key in tp_partition.keys()]
    layer_idx = [int(i[0]) for i in match if i]
    if not layer_idx:
        return 1
    num_layers_per_pipeline_stage = max(layer_idx) + 1

    return num_layers_per_pipeline_stage


def parse_parallel_size(checkpoint_dir: str):
    """Parse tensor, pipeline and expert parallel sizes from checkpoint filenames.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files

    Returns:
        Namespace
    """
    # Find all rank directories
    rank_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('mp_rank_')]

    if not rank_dirs:
        raise ValueError(f"No checkpoint rank directories found in {checkpoint_dir}")

    ckpt = FmkAdp.load_checkpoint(
        os.path.join(checkpoint_dir, rank_dirs[0], 'model_optim_rng.pt'), to_cpu=True, weights_only=False
    )
    args = ckpt[ARGS]
    return (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.num_experts,
    )


def parse_iteration(checkpoint_path: str) -> Dict:
    """
    Parse the checkpoint iteration directory from a given checkpoint path.

    If the path is a top-level checkpoint directory, this function reads the
    'latest_checkpointed_iteration.txt' file to determine the latest iteration.
    If the path is already an iteration directory (e.g., 'iter_0000005'), it extracts
    the iteration number from the path.

    Args:
        checkpoint_path (str): Path to the checkpoint directory or iteration directory.

    Returns:
        str: The full path to the checkpoint directory for the determined iteration.

    Raises:
        ValueError: If the checkpoint directory for the determined iteration does not exist.
    """
    iteration = None
    tracker_file = os.path.join(checkpoint_path, "latest_checkpointed_iteration.txt")
    if os.path.exists(tracker_file):
        with FileOpen(tracker_file, 'r') as f:
            latest_iteration = f.read().strip()
            if latest_iteration != 'release':
                try:
                    iteration = int(latest_iteration)
                except Exception:
                    logger.warning(
                        f"The latest_checkpointed_iteration is supposed to be `release` or an int. \
                        But {latest_iteration} is found."
                    )
            checkpoint_path = os.path.join(checkpoint_path, f'iter_{iteration:07d}')
    else:
        match = re.findall(ITER_DIR_PATTERN, checkpoint_path)
        if match:
            iteration = int(match[0])

    # Checkpoint directory for this iteration
    logger.info(f"Loaded checkpoint from iteration {iteration}")

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_path}")

    return checkpoint_path


def get_weights_from_state_dict(state_dict):
    weights = {}
    vpp_stage = 0
    if 'model' in state_dict:
        model_weights = state_dict['model']

        for key, value in _get_parameter(model_weights):
            key = _map_to_mcore_local_names(key)
            weights[f"{key}{Const.SCOPE_SEPARATOR}{vpp_stage}"] = value

    elif 'model0' in state_dict:
        # vpp enabled
        while f'model{vpp_stage}' in state_dict:
            model_weights = state_dict[f'model{vpp_stage}']
            for key, value in _get_parameter(model_weights):
                key = _map_to_mcore_local_names(key)
                weights[f"{key}{Const.SCOPE_SEPARATOR}{vpp_stage}"] = value
            vpp_stage += 1
    return weights


def load_megatron_weights(checkpoint_path: str) -> Dict:
    """Load Megatron parallel checkpoint weights into a single dictionary.

    Args:
        checkpoint_path (str): Base checkpoint directory path

    Returns:
        combined_weights: Dict with weights from all ranks, keys include rank info
    """
    try:
        import megatron  # noqa: F401
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("No module named 'megatron', which is required to load a megatron ckpt") from e

    # Find latest iteration if not specified
    checkpoint_path = parse_iteration(checkpoint_path)

    # Parse parallel sizes from checkpoint directory structure
    tp_size, pp_size, exp_size, num_experts = parse_parallel_size(checkpoint_path)
    combined_weights = {}

    # Load checkpoints from all ranks
    for exp_rank in range(exp_size):
        num_layers_per_pipeline_stage = 0
        for pp_rank in range(pp_size):
            tp_partition = defaultdict(list)
            for tp_rank in range(tp_size):
                # Construct checkpoint path based on parallel ranks
                if pp_size > 1:
                    rank_dir = f'mp_rank_{tp_rank:02d}_{pp_rank:03d}'
                else:
                    rank_dir = f'mp_rank_{tp_rank:02d}'

                if exp_size > 1:
                    rank_dir = f'{rank_dir}_{exp_rank:03d}'

                ckpt_file = os.path.join(checkpoint_path, rank_dir, 'model_optim_rng.pt')
                check_link(ckpt_file)
                try:
                    state_dict = FmkAdp.load_checkpoint(ckpt_file, to_cpu=True, weights_only=False)
                    partition = get_weights_from_state_dict(state_dict)
                    for key, weight in partition.items():
                        tp_partition[key].append(weight)

                except Exception as load_error:
                    logger.warning(f"Error loading {ckpt_file}: {load_error}")

            if not tp_partition:
                raise ValueError('No state loaded.')

            if not num_layers_per_pipeline_stage:
                num_layers_per_pipeline_stage = _parse_num_layers_per_stage(tp_partition)

            consolidated_weight = _consolidate_tp_weights(tp_partition)
            for key, value in consolidated_weight.items():
                key = _parse_real_layer_idx(key, num_layers_per_pipeline_stage, pp_size, pp_rank)
                if num_experts:
                    key = _parse_real_expert_idx(key, num_experts // exp_size, exp_rank)
                combined_weights[key] = value

    logger.info(f"Found {len(combined_weights)} total parameters across all ranks")

    return combined_weights
