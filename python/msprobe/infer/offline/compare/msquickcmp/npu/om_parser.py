# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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
This class is used to om fusion parser.
"""

import itertools
import json
from typing import Union, List

import numpy as np

from msprobe.core.common.log import logger
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.dynamic_argument_bean import DynamicArgumentEnum
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.utils.util import load_file_to_read_common_check
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE


GRAPH_OBJECT = "graph"
OP_OBJECT = "op"
NAME_OBJECT = "name"
TYPE_OBJECT = "type"
INPUT_DESC_OBJECT = "input_desc"
INPUT_OBJECT = "input"
ATTR_OBJECT = "attr"
SHAPE_OBJECT = "shape"
SHAPE_RANGE_OBJECT = "shape_range"
DIM_OBJECT = "dim"
DATA_OBJECT = "Data"
NET_OUTPUT_OBJECT = "NetOutput"
ATC_CMDLINE_OBJECT = "atc_cmdline"
INPUT_SHAPE = "--input_shape"
INPUT_SHAPE_RANGE = "--input_shape_range"
LIST_LIST_INT_OBJECT = 'list_list_int'
LIST_LIST_I_OBJECT = 'list_list_i'
LIST_I_OBJECT = 'list_i'
LIST_OBJECT = 'list'
KEY_OBJECT = "key"
VALUE_OBJECT = "value"
SUBGRAPH_NAME = 'subgraph_name'
S_OBJECT = "s"
DTYPE_OBJECT = "dtype"
DTYPE_MAP = {
    "DT_FLOAT": np.float32,
    "DT_FLOAT16": np.float16,
    "DT_DOUBLE": np.float64,
    "DT_INT8": np.int8,
    "DT_INT16": np.int16,
    "DT_INT32": np.int32,
    "DT_INT64": np.int64,
    "DT_UINT8": np.uint8,
    "DT_UINT16": np.uint16,
    "DT_UINT32": np.uint32,
    "DT_UINT64": np.uint64,
    "DT_BOOL": np.bool_,
}
OUT_NODES_NAME = "attr_model_out_nodes_name"
AIPP_CONFIG_PATH = "aipp_config_path"
LAYOUT_OBJECT = "layout"
# special ops
SPECIAL_OPS_TYPE = ("Cast", "TransData")


class OmParser(object):  # pylint: disable=useless-object-inheritance
    """
    This class is used to parse om model.
    """

    def __init__(self, output_json_path):
        self.json_object = self._load_json_file(output_json_path)
        self.subgraph_name = self.get_sub_graph_name()
        self.shape_range = self._is_input_shape_range()
        self.contain_negative_1 = False
        self.special_op_attr = self._parse_special_op_attr()

    @staticmethod
    def _load_json_file(json_file_path):
        """
        Function Description:
            load json file
        Parameter:
            json_file_path: json file path
        Return Value:
            json object
        Exception Description:
            when invalid json file path throw exception
        """
        try:
            json_file_path = load_file_to_read_common_check(json_file_path)
            with ms_open(json_file_path, "r", max_size=TENSOR_MAX_SIZE) as input_file:
                try:
                    return json.load(input_file)
                except Exception as exc:
                    logger.error(f"Load Json {json_file_path} failed, {str(exc)}")
                    raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR) from exc
        except IOError as input_file_open_except:
            logger.error('Failed to open"' + json_file_path + '", ' + str(input_file_open_except))
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_OPEN_FILE_ERROR) from input_file_open_except

    @staticmethod
    def _get_prefix(input_obj):
        return input_obj.split(':')[0]

    @staticmethod
    def _parse_net_output_node_attr(operator):
        net_output_info = {}
        if INPUT_DESC_OBJECT in operator:
            input_index = 0
            for input_object in operator.get(INPUT_DESC_OBJECT):
                shape = []
                data_type = DTYPE_MAP.get(input_object.get(DTYPE_OBJECT))
                if not input_object.get(SHAPE_OBJECT):
                    # no shape info, assumed to be scalar
                    net_output_info[input_index] = [data_type, [1]]
                    continue
                for num in input_object.get(SHAPE_OBJECT).get(DIM_OBJECT):
                    shape.append(num)
                net_output_info[input_index] = [data_type, shape]
                input_index += 1
        return net_output_info

    def get_shape_size(self):
        """
        Get shape size for input
        """
        input_desc_array = self._get_data_input_desc()
        # extracts the input shape value
        return self._process_inputs(input_desc_array)

    def get_shape_list(self):
        """
        Get shape list for input
        """
        input_desc_array = self._get_data_input_desc()
        # extracts the input shape value
        return self._process_inputs_to_list(input_desc_array)

    def get_net_output_count(self):
        """
        Get net output count
        """
        count = 0
        is_dynamic_scenario, dym_arg = self.get_dynamic_scenario_info()
        if not is_dynamic_scenario or dym_arg is DynamicArgumentEnum.DYM_SHAPE:
            op_iterator = self._gen_operator_list()
        else:
            op_iterator = self._gen_operator_list_from_subgraph()
        for operator in op_iterator:
            if NET_OUTPUT_OBJECT == operator.get(TYPE_OBJECT) and INPUT_DESC_OBJECT in operator:
                count += len(operator.get(INPUT_DESC_OBJECT))
        return count

    def get_atc_cmdline(self):
        for attr in self.json_object.get(ATTR_OBJECT):
            if KEY_OBJECT in attr and attr.get(KEY_OBJECT) == ATC_CMDLINE_OBJECT:
                if VALUE_OBJECT in attr and S_OBJECT in attr.get(VALUE_OBJECT):
                    atc_cmd = attr.get(VALUE_OBJECT).get(S_OBJECT)
                    return self._replace_input_shape_with_input_shape_range_in_atc_cmd_if_dynamic(atc_cmd)
        return ''

    def get_expect_net_output_name(self):
        """
        Get the expected output tensor corresponding to Net_output.
        """
        net_output_names = []
        if ATTR_OBJECT not in self.json_object:
            return {}
        for attr in self.json_object.get(ATTR_OBJECT):
            if not (KEY_OBJECT in attr and attr.get(KEY_OBJECT) == OUT_NODES_NAME):
                continue
            if not (VALUE_OBJECT in attr and LIST_OBJECT in attr.get(VALUE_OBJECT)):
                continue
            list_object = attr.get(VALUE_OBJECT).get(LIST_OBJECT)
            if S_OBJECT in list_object:
                net_output_names = list_object.get(S_OBJECT)
        return dict(enumerate(net_output_names))

    def get_net_output_data_info(self, dump_data_path):
        """
        get_net_output_data_info
        """
        net_output_list = []
        for operator in self._gen_operator_list():
            if NET_OUTPUT_OBJECT == operator.get(TYPE_OBJECT):
                net_output_list.append(operator)
        if len(net_output_list) == 1:
            return self._parse_net_output_node_attr(net_output_list[0])
        # if it's dynamic batch scenario, the net output node should be identified by batch index
        _, scenario = self.get_dynamic_scenario_info()
        if scenario in [DynamicArgumentEnum.DYM_BATCH, DynamicArgumentEnum.DYM_DIMS]:
            if not dump_data_path:
                for operator in net_output_list:
                    return self._parse_net_output_node_attr(operator)
            cur_batch_index = utils.get_batch_index(dump_data_path)
            for operator in net_output_list:
                batch_index_in_operator = utils.get_batch_index_from_name(operator.get(NAME_OBJECT))
                if cur_batch_index == batch_index_in_operator:
                    return self._parse_net_output_node_attr(operator)
        logger.error("get npu output node info failed.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_PARSER_JSON_FILE_ERROR)

    def get_dynamic_scenario_info(self):
        atc_cmd = self.get_atc_cmdline()
        for dym_arg in DynamicArgumentEnum:
            if dym_arg.value.atc_arg in atc_cmd:
                return True, dym_arg
        return False, None

    def get_aipp_config_content(self) -> Union[str, List]:
        aipp_configs = []
        for graph in self.json_object.get(GRAPH_OBJECT):
            for op in graph.get(OP_OBJECT):
                for attr in op.get(ATTR_OBJECT):
                    if KEY_OBJECT in attr and attr.get(KEY_OBJECT) == AIPP_CONFIG_PATH:
                        aipp_configs.append(attr.get(VALUE_OBJECT).get(S_OBJECT))
        if aipp_configs:
            return aipp_configs
        else:
            return []

    def get_sub_graph_name(self):
        subgraph_name = []
        for graph in self.json_object.get(GRAPH_OBJECT):
            for operator in graph.get(OP_OBJECT):
                if SUBGRAPH_NAME in operator:
                    subgraph_name += operator.get(SUBGRAPH_NAME)
        return subgraph_name

    def _replace_input_shape_with_input_shape_range_in_atc_cmd_if_dynamic(self, atc_cmd):
        # `--input_shape_range` is deprecated, and replaced with `--input_shape` in atc.
        # but `--input_shape_range` is currently used in parsing model as dynamic shape one,
        # It's more convinient for using and doesn't matter in comparing process.
        # Thus `--input_shape` args is converted to `--input_shape_range` in
        # this function for further comparing usage.
        atc_cmd_split = []
        for cmd_token in atc_cmd.split():
            atc_cmd_split += cmd_token.split("=")  # Could be in format like "--input_shape=input:1,3,224,224"

        # Return if any dynamic args already involved
        for dym_arg in DynamicArgumentEnum:
            if dym_arg.value.atc_arg in atc_cmd_split:
                return atc_cmd

        # Check if `--input_shape` in atc_cmd line. Return if not
        input_shape_pos = -1
        for pos, cmd_token in enumerate(atc_cmd_split):
            if cmd_token == INPUT_SHAPE:
                input_shape_pos = pos

        if input_shape_pos == -1 or len(atc_cmd_split) <= input_shape_pos + 1:
            return atc_cmd

        # Check if dynamic `--input_shape`
        input_shape_args = atc_cmd_split[input_shape_pos + 1]
        if "-1" not in input_shape_args and "~" not in input_shape_args:
            return atc_cmd

        # replace `--input_shape` with `--input_shape_range` if dynamic
        input_shape_range_args_split = []
        for inputs in input_shape_args.split(';'):
            inputs_split = inputs.split(':')
            input_name, input_shape = ":".join(inputs_split[:-1]), inputs_split[-1]
            input_shape_range_args_split.append('{}:[{}]'.format(input_name, input_shape))
        input_shape_range_args = ';'.join(input_shape_range_args_split)

        atc_cmd_with_range_split = atc_cmd_split[:input_shape_pos]
        atc_cmd_with_range_split += [INPUT_SHAPE_RANGE, input_shape_range_args]
        atc_cmd_with_range_split += atc_cmd_split[input_shape_pos + 2 :]
        atc_cmd_with_range = " ".join(atc_cmd_with_range_split)

        if not self.shape_range:  # Only print once, just if shape_range is False
            logger.info(
                f"Convert atc arg '{INPUT_SHAPE} {input_shape_args}' -> '{INPUT_SHAPE_RANGE} {input_shape_range_args}'"
            )
        self.shape_range = True
        return atc_cmd_with_range

    def _gen_operator_list(self):
        _, scenario = self.get_dynamic_scenario_info()
        for graph in self.json_object.get(GRAPH_OBJECT):
            if graph.get(NAME_OBJECT) in self.subgraph_name and scenario not in [
                DynamicArgumentEnum.DYM_BATCH,
                DynamicArgumentEnum.DYM_DIMS,
            ]:
                continue
            for operator in graph.get(OP_OBJECT):  # pylint: disable=use-yield-from
                yield operator  # pylint: disable=use-yield-from

    def _gen_operator_list_from_subgraph(self):
        for graph in self.json_object.get(GRAPH_OBJECT):
            if graph.get(NAME_OBJECT) in self.subgraph_name:
                for operator in graph.get(OP_OBJECT):  # pylint: disable=use-yield-from
                    yield operator  # pylint: disable=use-yield-from
                return

    def _get_data_input_desc(self):
        input_desc_list = []
        for operator in self._gen_operator_list():
            if DATA_OBJECT == operator.get(TYPE_OBJECT):
                if len(operator.get(INPUT_DESC_OBJECT)) != 0:
                    for item in operator.get(INPUT_DESC_OBJECT):
                        input_desc_list.append(item)
        return input_desc_list

    def _parse_special_op_attr(self):
        special_op_attr = {}
        for operator in self._gen_operator_list():
            if operator.get(TYPE_OBJECT) in SPECIAL_OPS_TYPE:
                special_op_attr[operator.get(NAME_OBJECT)] = operator.get(INPUT_OBJECT)
        return special_op_attr

    def _is_input_shape_range(self):
        if ATTR_OBJECT not in self.json_object:
            return False
        for attr in self.json_object.get(ATTR_OBJECT):
            if KEY_OBJECT in attr and attr.get(KEY_OBJECT) == ATC_CMDLINE_OBJECT:
                if VALUE_OBJECT in attr and S_OBJECT in attr.get(VALUE_OBJECT):
                    if INPUT_SHAPE_RANGE in attr.get(VALUE_OBJECT).get(S_OBJECT):
                        return True
        return False

    def _get_shape_list(self, list_list_int_object, shape_list):
        if LIST_LIST_I_OBJECT in list_list_int_object:
            for list_list_i in list_list_int_object.get(LIST_LIST_I_OBJECT):
                if LIST_I_OBJECT in list_list_i:
                    list_i = list_list_i.get(LIST_I_OBJECT)
                    if -1 in list_i:
                        self.contain_negative_1 = True
                        return
                    if len(list_i) != 2:
                        continue
                    shape_list.append(list(range(list_i[0], list_i[1] + 1)))
        return

    def _get_range_shape_size_list(self, input_object):
        range_shape_size_list = []
        if ATTR_OBJECT not in input_object:
            return range_shape_size_list
        shape_list = []
        for attr in input_object.get(ATTR_OBJECT):
            if KEY_OBJECT in attr and attr.get(KEY_OBJECT) == SHAPE_RANGE_OBJECT:
                if VALUE_OBJECT in attr and attr.get(VALUE_OBJECT) and LIST_LIST_INT_OBJECT in attr.get(VALUE_OBJECT):
                    list_list_int_object = attr.get(VALUE_OBJECT).get(LIST_LIST_INT_OBJECT)
                    self._get_shape_list(list_list_int_object, shape_list)
                    if self.contain_negative_1:
                        return []
        shape_list_all = list(itertools.product(*shape_list))
        for item in shape_list_all:
            item_sum = 1
            for num in item:
                item_sum *= num
            range_shape_size_list.append(item_sum)
        return range_shape_size_list

    def _process_inputs(self, input_desc_array):
        value = []
        for input_object in input_desc_array:
            if SHAPE_OBJECT not in input_object:
                value.append(0)
                continue
            data_type = DTYPE_MAP.get(input_object.get(DTYPE_OBJECT))
            if not data_type:
                logger.error(f"The dtype attribute does not support {input_object[DTYPE_OBJECT]} value.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_KEY_ERROR)
            data_type_size = np.dtype(data_type).itemsize
            if self.shape_range:
                range_shape_size_list = self._get_range_shape_size_list(input_object)
                for item in range_shape_size_list:
                    value.append(item * data_type_size)
            else:
                item_sum = 1
                for num in input_object.get(SHAPE_OBJECT).get(DIM_OBJECT):
                    item_sum *= num
                value.append(item_sum * data_type_size)
        return value

    def _process_inputs_to_list(self, input_desc_array):
        shape_lists = []
        for input_object in input_desc_array:
            shape_list = []
            if SHAPE_OBJECT not in input_object:
                logger.error("Please specify the input shape of om model through -is param")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            data_type = DTYPE_MAP.get(input_object.get(DTYPE_OBJECT))
            if not data_type:
                logger.error(f"The dtype attribute does not support {input_object[DTYPE_OBJECT]} value.")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_KEY_ERROR)
            if self.shape_range:
                logger.error("Please specify the input shape of om model through -is param")
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            else:
                input_format = input_object.get(LAYOUT_OBJECT, "NCHW")
                input_format_index = [input_format.index(x) for x in "NCHW"]
                for num in input_object.get(SHAPE_OBJECT).get(DIM_OBJECT):
                    shape_list.append(num)
                shape_list = list(np.array(shape_list)[input_format_index])
                shape_lists.append(shape_list)
        return shape_lists
