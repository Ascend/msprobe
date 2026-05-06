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


from msprobe.core.compare.acc_compare import Comparator, ModeConfig, MappingConfig, setup_comparison
from msprobe.core.compare.layer_mapping import generate_data_mapping_by_layer_mapping
from msprobe.mindspore.compare.utils import read_npy_data, check_cross_framework


def read_real_data(data_path_dict: dict, cross_frame, backend) -> tuple:
    """
    MS 版本：
    - cross_frame: 是否跨框架
    - backend: 为了和 PT 版本接口统一保留，不使用
    """
    npu_dir = data_path_dict.get("npu_dir")
    bench_dir = data_path_dict.get("bench_dir")
    npu_data_name = data_path_dict.get("npu_data_name")
    bench_data_name = data_path_dict.get("bench_data_name")

    n_value = read_npy_data(npu_dir, npu_data_name)
    if cross_frame:
        from msprobe.pytorch.compare.utils import read_pt_data
        b_value = read_pt_data(bench_dir, bench_data_name)
    else:
        b_value = read_npy_data(bench_dir, bench_data_name)
    return n_value, b_value


def ms_compare(input_param, output_path, **kwargs):
    config = setup_comparison(input_param, output_path, **kwargs)

    if config.layer_mapping:
        config.data_mapping = generate_data_mapping_by_layer_mapping(input_param, config.layer_mapping, output_path)

    is_cross_framework = check_cross_framework(input_param.get('bench_path'))

    config_dict = {
        'stack_mode': config.stack_mode,
        'fuzzy_match': config.fuzzy_match,
        'xlsx': config.xlsx,
        'dump_mode': config.dump_mode,
        'compared_file_type': config.compared_file_type
    }
    mode_config = ModeConfig(**config_dict)
    mapping_config = MappingConfig(config.cell_mapping, config.api_mapping, config.data_mapping)
    ms_comparator = Comparator(read_real_data, mode_config, mapping_config, is_cross_framework)
    ms_comparator.compare_core(input_param, output_path, suffix=config.suffix)
