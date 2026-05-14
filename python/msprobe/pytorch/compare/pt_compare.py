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
import numpy as np

from msprobe.core.common.const import CompareConst, Const
from msprobe.core.compare.acc_compare import Comparator, ModeConfig, MappingConfig, setup_comparison
from msprobe.core.compare.utils import split_tensors
from msprobe.pytorch.compare.utils import read_pt_data


def need_cat(backend: str, data_name: str) -> bool:
    rule = CompareConst.BACKEND_CAT_RULES.get(backend)
    if rule is None:
        return False
    return rule(data_name)


def read_real_data(data_path_dict: dict, cross_frame, backend) -> tuple:
    """
    PT 版本：
    - cross_frame: 为了和 MS 版本接口统一保留，不使用
    - backend: 用于区分 fsdp / megatron

    npu_data_name/bench_data_name存在两种情况:
        单tensor：'xxx.pt'
        多tensor：'xx1.pt;xx2.pt;xx3.pt'，tensor名以;分割
    """
    npu_dir = data_path_dict.get("npu_dir")
    bench_dir = data_path_dict.get("bench_dir")
    npu_data_name = data_path_dict.get("npu_data_name")
    bench_data_name = data_path_dict.get("bench_data_name")

    if backend == Const.FSDP:
        if need_cat(backend, npu_data_name):
            npu_tensors_list = split_tensors(npu_data_name)
            if '.input.' in npu_data_name:
                npu_value = read_pt_data(npu_dir, npu_tensors_list[0])
            elif '.parameters.' in npu_data_name:
                npu_tensors = [read_pt_data(npu_dir, t) for t in npu_tensors_list]
                npu_value = np.concatenate(npu_tensors, axis=0)
            else:
                npu_tensors = [read_pt_data(npu_dir, t) for t in npu_tensors_list]
                npu_value = np.concatenate(npu_tensors, axis=1)
        else:
            npu_value = read_pt_data(npu_dir, npu_data_name)

        bench_value = read_pt_data(bench_dir, bench_data_name)
    else:
        if need_cat(backend, bench_data_name):
            bench_tensors_list = split_tensors(bench_data_name)
            if '.input.' in bench_data_name:
                bench_value = read_pt_data(bench_dir, bench_tensors_list[0])
            elif '.parameters.' in bench_data_name:
                bench_tensors = [read_pt_data(bench_dir, t) for t in bench_tensors_list]
                bench_value = np.concatenate(bench_tensors, axis=0)
            else:
                bench_tensors = [read_pt_data(bench_dir, t) for t in bench_tensors_list]
                bench_value = np.concatenate(bench_tensors, axis=1)
        else:
            bench_value = read_pt_data(bench_dir, bench_data_name)

        npu_value = read_pt_data(npu_dir, npu_data_name)

    return npu_value, bench_value


def pt_compare(input_param, output_path, **kwargs):
    config = setup_comparison(input_param, output_path, **kwargs)

    config_dict = {
        'stack_mode': config.stack_mode,
        'fuzzy_match': config.fuzzy_match,
        'xlsx': config.xlsx,
        'dump_mode': config.dump_mode,
        'first_diff_analyze': config.first_diff_analyze,
        'compared_file_type': config.compared_file_type,
        'consistent_check': config.consistent_check,
        'backend': config.backend
    }
    mode_config = ModeConfig(**config_dict)
    mapping_config = MappingConfig(cell_mapping=config.cell_mapping, data_mapping=config.data_mapping)
    pt_comparator = Comparator(read_real_data, mode_config, mapping_config)
    pt_comparator.compare_core(input_param, output_path, suffix=config.suffix)
