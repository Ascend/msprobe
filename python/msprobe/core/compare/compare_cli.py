# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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


from msprobe.core.common.utils import CompareException
from msprobe.core.common.log import logger
from msprobe.core.compare.atb_data_compare import compare_atb_mode
from msprobe.core.compare.auto_compare import compare_auto_mode
from msprobe.core.compare.offline_data_compare import compare_offline_data_mode
from msprobe.core.compare.torchair_acc_cmp import compare_torchair_mode
from msprobe.infer.offline.compare.msquickcmp.main import compare_offline_model_mode


MODE_DISPATCHER = {
    'auto': compare_auto_mode,
    'offline_data': compare_offline_data_mode,
    'torchair': compare_torchair_mode,
    'offline_model': compare_offline_model_mode,
    'atb': compare_atb_mode
}

VALID_ARGS_MAP = {
    'auto': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o', 
        '--xlsx',
        '--fuzzy_match', '-fm', '--cell_mapping', '-cm', '--api_mapping', '-am', '--data_mapping',
        '-dm', '--layer_mapping', '-lm', '--diff_analyze', '-da', '--rank', '--step',
        '--is_print_compare_log', '-tensor_log', '--consistent_check', '--backend'
    ],
    'offline_data': [
        "-m", "-tp", "-gp", "-fr", "-qfr", "-cfr", "-o", "--mode", "--target_path", "--golden_path", 
        "--fusion_rule_file", "--quant_fusion_rule_file", "--close_fusion_rule_file", "--output_path"
    ],
    'torchair': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o'
    ],
    'offline_model': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o',
        '--rank', '--input_data', '--input_shape', '--output_size', '--dym_shape_range',
        '-ofs', '--onnx_fusion_switch'
    ],
    'atb': [
        '--mode', '-m', '--target_path', '-tp', '--golden_path', '-gp', '--output_path', '-o'
    ],
}


def check_valid_args(sys_argv, mode):
    valid_args = VALID_ARGS_MAP.get(mode)
    for arg in sys_argv:
        if arg.startswith("-") and arg not in valid_args:
            logger.error(f"Invalid argument '{arg}' for mode '{mode}'")
            raise CompareException(CompareException.INVALID_PARAM_ERROR)


def compare_cli(args, sys_argv):
    """
    Dispatch comparison based on mode parameter
    """
    mode = getattr(args, 'mode', 'auto')
    # Get the appropriate function based on mode
    compare_func = MODE_DISPATCHER.get(mode)
    if compare_func is None:
        logger.error(f"Invalid mode '{mode}'. Available modes: {list(MODE_DISPATCHER.keys())}")
        raise CompareException(CompareException.INVALID_COMPARE_MODE)
    check_valid_args(sys_argv, mode)
    # Execute the comparison function
    return compare_func(args)
