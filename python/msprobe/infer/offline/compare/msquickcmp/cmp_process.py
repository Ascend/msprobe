# -*- coding: utf-8 -*-
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

"""
Function:
This class mainly involves the main function.
"""

import csv
import os
import stat
import time

import pandas as pd

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import create_directory, write_df_to_csv
from msprobe.infer.offline.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.atc import atc_utils
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_npy_to_bin
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException, \
    get_shape_to_directory_name, OPTYPE_WHITWLIST
from msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare import NetCompare
from msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data import NpuDumpData
from msprobe.infer.offline.compare.msquickcmp.npu.om_parser import OmParser
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.check.rule import Rule
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE

try:
    import acl
except ImportError:
    acl = None
    logger.error("Please verify that the CANN environment is properly configured.")

WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
MAX_MEMORY_USE = 6 * 1024 * 1024 * 1024
COSINE_SIMILARITY = 0.99
RELATIVE_EUCLIDEAN_DISTANCE = 0.05
KULLBACK_LEIBLER_DIVERGENCE = 0.005
ROOT_MEAN_SQUARE_ERROR = 1.0
MEAN_RELATIVE_ERROR = 1.0
YES = "YES"
NO = "NO"


def _generate_golden_data_model(args, npu_dump_npy_path):
    model_name, extension = utils.get_model_name_and_extension(args.golden_path)
    if ".onnx" == extension:
        from msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData
        return OnnxDumpData(args, npu_dump_npy_path), extension
    elif ".om" == extension:
        return NpuDumpData(arguments=args, is_golden=True), extension
    else:
        logger.error("Only model files whose names end with .om or .onnx are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        logger.info(f"swap the {left_index} and {right_index} item in golden_net_output_info!")


def _check_output_node_name_mapping(original_net_output_node, golden_net_output_info):
    for left_index, node_name in original_net_output_node.items():
        match = False
        for right_index, dump_file_path in golden_net_output_info.items():
            dump_file_name = os.path.basename(dump_file_path)
            if dump_file_name.startswith(node_name.replace("/", "_").replace(":", ".")):
                match = True
                _correct_the_wrong_order(left_index, right_index, golden_net_output_info)
                break
        if not match:
            logger.warning(f"the original name: {node_name} of net output maybe not correct!")
            break


def _get_single_csv_in_folder(csv_path):
    for file_name in os.listdir(csv_path):
        if file_name.endswith('.csv'):
            return os.path.join(csv_path, file_name)
    raise IOError(f"None csv file exists in folder {csv_path}")


def _read_and_process_csv(csv_path, process_func, node_output_show_list):
    if Rule.input_file().check(csv_path):
        with ms_open(csv_path, 'r', max_size=TENSOR_MAX_SIZE) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        if len(rows) < 1:
            logger.error("csv is empty, please check.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_EMPTY_CSV_ERROR)
        header = rows[0]
        rows = process_func(header, rows, node_output_show_list)
    return rows


def _process_is_npu_and_is_precision_error_ops(header, rows, node_output_name_list):
    ground_truth_col = header.index("GroundTruth")
    optype_col = header.index("OpType")
    cosine_similarity_col = header.index("CosineSimilarity")
    relative_euclidean_distance_col = header.index("RelativeEuclideanDistance")
    kullback_leibler_divergence_col = header.index("KullbackLeiblerDivergence")
    root_mean_square_error_col = header.index("RootMeanSquareError")
    mean_relative_error_col = header.index("MeanRelativeError")

    header.append('IsNpuOps')
    header.append('IsOutputNode')
    header.append('IsPrecisionError')
    for row in rows[1:]:
        try:
            is_npu_ops = YES if row[ground_truth_col] == "*" else NO
            row.append(is_npu_ops)

            optype = row[optype_col]
            cosine_similarity = row[cosine_similarity_col]
            relative_euclidean_distance = float(row[relative_euclidean_distance_col])
            kullback_leibler_divergence = float(row[kullback_leibler_divergence_col])
            root_mean_square_error = float(row[root_mean_square_error_col])
            mean_relative_error = float(row[mean_relative_error_col])
            if _is_output_node(row[ground_truth_col], node_output_name_list):
                row.append(YES)
            else:
                row.append(NO)
            if optype in OPTYPE_WHITWLIST or cosine_similarity.lower() == 'nan':
                row.append(NO)
                continue
            if _is_row_precision_error(cosine_similarity,
                                       relative_euclidean_distance,
                                       kullback_leibler_divergence,
                                       root_mean_square_error,
                                       mean_relative_error):
                row.append(YES)
            else:
                row.append(NO)

        except ValueError as e:
            logger.warning(f"Skipping row due to invalid data: {row}. Error: {e}")
            continue
    return rows


def _is_row_precision_error(cosine_similarity,
                            relative_euclidean_distance,
                            kullback_leibler_divergence,
                            root_mean_square_error,
                            mean_relative_error):
    return (float(cosine_similarity) < COSINE_SIMILARITY or
            relative_euclidean_distance > RELATIVE_EUCLIDEAN_DISTANCE or
            kullback_leibler_divergence > KULLBACK_LEIBLER_DIVERGENCE or
            root_mean_square_error > ROOT_MEAN_SQUARE_ERROR or
            mean_relative_error > MEAN_RELATIVE_ERROR)


def _is_output_node(groundtruth, onnxnode_output_name_list):
    return groundtruth.strip() in onnxnode_output_name_list


def _append_column_to_csv(csv_path, node_output_show_list=None):
    if node_output_show_list is None:
        node_output_show_list = []
    csv_path = _get_single_csv_in_folder(csv_path)
    rows = _read_and_process_csv(csv_path, _process_is_npu_and_is_precision_error_ops, node_output_show_list)

    rows_pd = pd.DataFrame(rows[1:], columns=rows[0])  # _read_and_process_csv 内已校验rows非空
    write_df_to_csv(rows_pd, csv_path, malicious_check=True)


def cmp_process(args: CmpArgsAdapter):
    args.golden_path = os.path.realpath(args.golden_path)
    args.target_path = os.path.realpath(args.target_path)
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_data = convert_npy_to_bin(args.input_data)
    try:
        check_and_run(args)
    except utils.AccuracyCompareException as error:
        raise error


def run(args: CmpArgsAdapter, input_shape, original_out_path):
    if input_shape:
        args.input_shape = input_shape
        args.output_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))
    run_om_model_compare(args)


def run_om_model_compare(args):
    # whether use aipp
    output_json_path = atc_utils.convert_model_to_json(args.cann_path, args.target_path, args.output_path)
    golden_json_path = None
    if args.golden_path.endswith('.om'):
        golden_json_path = atc_utils.convert_model_to_json(args.cann_path, args.golden_path, args.output_path)

    temp_om_parser = OmParser(output_json_path)
    use_aipp = True if temp_om_parser.get_aipp_config_content() else False

    npu_dump = NpuDumpData(args, is_golden=False)
    # generate npu inputs data
    npu_dump.generate_inputs_data(use_aipp=use_aipp)
    # generate npu dump data
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data()
    npu_dump_npy_path = ""

    # generate onnx inputs data
    golden_dump, model_extension = _generate_golden_data_model(args, npu_dump_npy_path)
    expect_net_output_node = npu_dump.get_expect_output_name()
    # generate dump data by golden model
    golden_dump.generate_inputs_data(npu_dump_data_path, use_aipp)
    if isinstance(golden_dump, NpuDumpData):
        golden_dump_data_path, _ = golden_dump.generate_dump_data(npu_dump_npy_path, npu_dump.om_parser)
    else:
        golden_dump_data_path = golden_dump.generate_dump_data(npu_dump_npy_path, npu_dump.om_parser)
    golden_net_output_info = golden_dump.get_net_output_info()

    # if it's dynamic batch scenario, golden data files should be renamed
    utils.handle_ground_truth_files(npu_dump.om_parser, npu_dump_data_path, golden_dump_data_path)

    # compare the entire network
    net_compare = NetCompare(npu_dump_data_path, golden_dump_data_path, output_json_path, args, golden_json_path)
    net_compare.accuracy_network_compare()

    # Check and correct the mapping of net output node name.
    if len(expect_net_output_node) == 1:
        _check_output_node_name_mapping(expect_net_output_node, golden_net_output_info)

    node_output_show_list = None
    if model_extension == ".onnx":
        node_output_show_list = _get_model_output_node_name_list(golden_dump.model_with_inputs_session,
                                                                 golden_dump.origin_model)
    _append_column_to_csv(args.output_path, node_output_show_list)


def _get_model_output_node_name_list(model_with_inputs_session, origin_model):
    net_output_node_name_list = [item.name for item in model_with_inputs_session.get_outputs()]
    node_output_show_list = []
    for node_name in net_output_node_name_list:
        pre_node = _find_previous_node(origin_model.graph, node_name)
        if pre_node is None:
            return None
        node_output_show_list.append(pre_node)
    return node_output_show_list


def _find_previous_node(graph, output_name):
    # 遍历所有节点
    for node in graph.node:
        if output_name in [output for output in node.output]:
            # 找到目标输出节点
            return node.name
    return None


def check_and_run(args: CmpArgsAdapter):
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.output_path, time_dir))
    args.output_path = original_out_path
    create_directory(args.output_path)

    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        run(args, input_shape, original_out_path)
    if args.dym_shape_range:
        csv_sum(original_out_path)


def csv_sum(original_out_path):
    """
    Function:
        Summarize csv files under different shapes
        generate a xlsx file
    """
    csv_file_list = []
    sheet_name_list = []

    for files in os.listdir(original_out_path):
        if not os.path.isdir(os.path.join(original_out_path, files)):
            continue
        for sub_file in os.listdir(os.path.join(original_out_path, files)):
            if sub_file.endswith(".csv"):
                csv_file_list.append(os.path.join(original_out_path, files, sub_file))
                sheet_name_list.append(files)

    xlsx_file_summary = os.path.join(original_out_path, "result_summary.xlsx")

    if os.path.exists(xlsx_file_summary):
        logger.error(f"Error, file {xlsx_file_summary} already exists!")
        os.remove(xlsx_file_summary)

    with ms_open(xlsx_file_summary, 'wb') as fp_write:
        with pd.ExcelWriter(xlsx_file_summary) as writer:
            for i, csv_file in enumerate(csv_file_list):
                if Rule.input_file().check(csv_file):
                    data = pd.read_csv(csv_file, na_values=['NAN'])
                    data.to_excel(writer, sheet_name=sheet_name_list[i], index=False, na_rep='NAN')
