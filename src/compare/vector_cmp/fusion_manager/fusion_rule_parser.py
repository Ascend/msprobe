# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
FusionRuleParser class.
This class mainly involves the analysis_fusion_rule function.
"""

import uuid

from cmp_utils import utils, utils_type
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.file_utils import FileUtils
from cmp_utils.constant.compare_error import CompareError
from vector_cmp.fusion_manager.quant_filter import QuantFilter
from vector_cmp.fusion_manager.fusion_op import OpAttr, FusionOp, OutputDesc
from dump_parse import dump


def make_left_and_right_string(ground_truth_to_my_output_map: dict) -> (str, str):
    """
    Convert my output ops and ground truth ops to string, split by ","
    :param ground_truth_to_my_output_map: the map for ground truth op to my output op list
    :return my output op names, ground truth op names
    """
    ground_truth_ops_str = ",".join(ground_truth_to_my_output_map.keys())
    my_output_op_name_list = []
    for op_list in ground_truth_to_my_output_map.values():
        for fusion_op in op_list:
            my_output_op_name_list.append(fusion_op.op_name)
    my_output_ops_str = ",".join(my_output_op_name_list)
    return my_output_ops_str, ground_truth_ops_str


def get_relation_for_fusion(fusion_op_list: list) -> utils_type.FusionRelation:
    """
    Get the relation for fusion op list
    :param fusion_op_list: the fusion op list
    :return FusionRelation
    """
    if len(fusion_op_list) == 1:
        if len(fusion_op_list[0].attr.original_op_names) > 1:
            return utils_type.FusionRelation.MultiToOne
        return utils_type.FusionRelation.OneToOne
    if len(fusion_op_list) > 1:
        if fusion_op_list[0].attr.l1_fusion_no != "":
            return utils_type.FusionRelation.L1Fusion
        if len(fusion_op_list[0].attr.original_op_names) == 1:
            return utils_type.FusionRelation.OneToMulti
        return utils_type.FusionRelation.MultiToMulti
    return utils_type.FusionRelation.OneToOne


def make_right_to_left_multi_map(fusion_op_list: list) -> dict:
    """
    make map {original_op_name_string, op_list} for any to multi
    :param fusion_op_list: the fusion op list
    :return the map {original_op_name_string, op_list}
    """
    # if multi to multi, the map is{'C,D,F': ['A', 'B']}
    # if one to multi, {'G': ['H', 'I']}
    # if splitter, the value of return is like {A: [A_1, A_2], B: [B_1, B_2]}
    ground_truth_to_my_output_map = {}
    for operator in fusion_op_list:
        original_names = ','.join(operator.attr.original_op_names)
        if original_names in ground_truth_to_my_output_map:
            ground_truth_to_my_output_map.get(original_names).append(operator)
        else:
            ground_truth_to_my_output_map[original_names] = [operator]
    return ground_truth_to_my_output_map


def _get_original_op_names_by_op_list(op_list: list, tmp_original_op_names: list) -> None:
    for fusion_op in op_list:
        for original_op_name in fusion_op.attr.original_op_names:
            if original_op_name not in tmp_original_op_names:
                tmp_original_op_names.append(original_op_name)


def _get_original_op_names_before_quant(fusion_op_info: FusionOp, quant_fusion_rule: any) -> list:
    tmp_original_op_names = []
    for op_name in fusion_op_info.attr.original_op_names:
        if op_name in quant_fusion_rule.op_name_to_fusion_op_name_map:
            op_list = quant_fusion_rule.fusion_op_name_to_op_map.get(
                quant_fusion_rule.op_name_to_fusion_op_name_map.get(op_name))
            _get_original_op_names_by_op_list(op_list, tmp_original_op_names)
    return tmp_original_op_names


def _make_new_output_desc(desc: OutputDesc, origin_format: str, origin_shape: list) -> None:
    if origin_format != '':
        desc.origin_format = origin_format
        desc.origin_shape = origin_shape


def _make_output_desc_by_index(output_desc: OutputDesc, op_list: list, tmp_output_desc: list) -> None:
    for operator in op_list:
        if operator.op_name != output_desc.origin_name:
            continue
        if output_desc.origin_output_index >= len(operator.output_desc):
            tmp_output_desc.append(OutputDesc("", None, "", []))
        else:
            desc = operator.output_desc[output_desc.origin_output_index]
            _make_new_output_desc(desc, output_desc.origin_format, output_desc.origin_shape)
            tmp_output_desc.append(desc)


def _make_output_desc_by_op_list(output_desc: OutputDesc, op_list: list, tmp_output_desc: list) -> None:
    origin_output_index = output_desc.origin_output_index
    # get origin output desc
    if origin_output_index is None:
        for item in op_list:
            for desc in item.output_desc:
                _make_new_output_desc(desc, output_desc.origin_format, output_desc.origin_shape)
                tmp_output_desc.append(desc)
    else:
        _make_output_desc_by_index(output_desc, op_list, tmp_output_desc)


def _get_output_desc_before_quant(fusion_op_info: FusionOp, quant_fusion_rule: any) -> list:
    tmp_output_desc = []
    for output_desc in fusion_op_info.output_desc:
        origin_name = output_desc.origin_name
        # skip empty origin name
        if origin_name == "":
            continue
        # skip origin name not in quant
        if origin_name not in quant_fusion_rule.op_name_to_fusion_op_name_map:
            log.print_warn_log('The name "%s" is not in quant fusion rule.' % origin_name)
            continue
        op_list = quant_fusion_rule.fusion_op_name_to_op_map.get(
            quant_fusion_rule.op_name_to_fusion_op_name_map.get(origin_name))
        _make_output_desc_by_op_list(output_desc, op_list, tmp_output_desc)
    return tmp_output_desc


def merge_fusion_rule(offline_fusion_rule: any, quant_fusion_rule: any) -> any:
    """
    Merge offline fusion rule and quant fusion rule
    :param offline_fusion_rule: the offline fusion rule
    :param quant_fusion_rule: the quant fusion rule
    :return the merged fusion rule
    """
    merge_map = {}
    for (_, fusion_op_list) in list(
            offline_fusion_rule.fusion_op_name_to_op_map.items()):
        for fusion_op in fusion_op_list:
            # get the quant fusion rule original op names, and deduplication
            tmp_original_op_names = _get_original_op_names_before_quant(
                fusion_op, quant_fusion_rule)
            # replace original op names
            if len(tmp_original_op_names) > 0:
                fusion_op.attr.original_op_names = list(tmp_original_op_names)

            # replace output desc
            tmp_output_desc = _get_output_desc_before_quant(fusion_op, quant_fusion_rule)
            if len(tmp_output_desc) > 0:
                fusion_op.output_desc = tmp_output_desc

            # replace fusion op name by original_op_names
            offline_fusion_rule.make_fusion_op_name(fusion_op.op_name, "", fusion_op.attr.original_op_names)
            fusion_op_name = offline_fusion_rule.op_name_to_fusion_op_name_map.get(fusion_op.op_name)
            if fusion_op_name in merge_map:
                fusion_op.op_id = merge_map.get(fusion_op_name)[0].op_id
                merge_map.get(fusion_op_name).append(fusion_op)
            else:
                fusion_op.op_id = len(merge_map)
                merge_map[fusion_op_name] = [fusion_op]
    offline_fusion_rule.fusion_op_name_to_op_map = merge_map
    return offline_fusion_rule


def _make_open_fusion_original_op_names(fusion_op: FusionOp, close_fusion_origin_to_op_map: dict) -> list:
    if len(fusion_op.attr.original_op_names) == 1 and fusion_op.attr.original_op_names[0] == '':
        return ['']
    original_op_names = []
    for origin_op_name in fusion_op.attr.original_op_names:
        if origin_op_name in close_fusion_origin_to_op_map:
            op_name = close_fusion_origin_to_op_map.get(origin_op_name)
            if op_name not in original_op_names:
                original_op_names.append(op_name)
        else:
            log.print_warn_log('There is no original operator associated with the operator "%s" '
                               'in original op names.' % origin_op_name)
    return original_op_names


def _get_close_fusion_origin_output_index(op_list: list, output_desc: OutputDesc) -> bool:
    for close_op in op_list:
        for index, close_output_desc in enumerate(close_op.output_desc):
            if output_desc.origin_name == close_output_desc.origin_name \
                    and output_desc.origin_output_index == close_output_desc.origin_output_index:
                output_desc.origin_output_index = index
                return True
    return False


def _make_open_fusion_output_desc(fusion_op: FusionOp, close_fusion_origin_to_op_map: dict,
                                  close_fusion_rule: any) -> None:
    for output_index, output_desc in enumerate(fusion_op.output_desc):
        if not output_desc.origin_name:
            continue
        if output_desc.origin_name in close_fusion_origin_to_op_map:
            op_name = close_fusion_origin_to_op_map.get(output_desc.origin_name)
            op_list = close_fusion_rule.fusion_op_name_to_op_map.get(
                close_fusion_rule.op_name_to_fusion_op_name_map.get(op_name))
            if not _get_close_fusion_origin_output_index(op_list, output_desc):
                log.print_warn_log('There is no valid output desc associated with "%s:output:%d".'
                                   % (fusion_op.op_name, output_index))
            output_desc.origin_name = op_name
        else:
            log.print_warn_log('There is no valid output desc associated with "%s:output:%d".'
                               % (fusion_op.op_name, output_index))


def _check_unity_onnx_op(delete_open_fusion_op_name_to_op_name: list, delete_close_fusion_op_name_to_op_name: list,
                         fusion_op: FusionOp, close_fusion_origin_to_op_map: list) -> bool:
    if fusion_op.attr.original_op_names[0] in delete_open_fusion_op_name_to_op_name:
        return False
    delete_open_fusion_op_name_to_op_name.append(fusion_op.attr.original_op_names[0])
    if fusion_op.attr.original_op_names[0] not in close_fusion_origin_to_op_map:
        return True
    if close_fusion_origin_to_op_map.get(fusion_op.attr.original_op_names[0]) in delete_close_fusion_op_name_to_op_name:
        return False
    delete_close_fusion_op_name_to_op_name.append(close_fusion_origin_to_op_map.get(
        fusion_op.attr.original_op_names[0]))
    return True




def merge_close_and_open_fusion_rule(open_fusion_rule: any, close_fusion_rule: any) -> any:
    """
    Merge close fusion rule and open fusion rule
    :param open_fusion_rule: the open fusion rule
    :param close_fusion_rule: the close fusion rule
    :return the merged fusion rule
    """
    merged_fusion_rule = FusionRuleParser('')
    delete_fusion_op_name_list = []
    delete_open_fusion_op_name_to_op_name = []
    delete_close_fusion_op_name_to_op_name = []
    close_fusion_origin_to_op_map = close_fusion_rule.get_origin_name_to_op_name_map()
    for key, value in reversed(list(open_fusion_rule.op_name_to_fusion_op_name_map.items())):
        if value in delete_fusion_op_name_list:
            continue
        delete_fusion_op_name_list.append(value)
        merged_fusion_rule.op_name_to_fusion_op_name_map[key] = value
        for fusion_op in open_fusion_rule.fusion_op_name_to_op_map.get(value):
            if not _check_unity_onnx_op(delete_open_fusion_op_name_to_op_name, delete_close_fusion_op_name_to_op_name,
                                        fusion_op, close_fusion_origin_to_op_map):
                continue
            # make new original op names
            fusion_op.attr.original_op_names = _make_open_fusion_original_op_names(fusion_op,
                                                                                   close_fusion_origin_to_op_map)
            # make new output desc
            _make_open_fusion_output_desc(fusion_op, close_fusion_origin_to_op_map, close_fusion_rule)
            if value in merged_fusion_rule.fusion_op_name_to_op_map:
                merged_fusion_rule.fusion_op_name_to_op_map.get(value).append(fusion_op)
            else:
                merged_fusion_rule.fusion_op_name_to_op_map[value] = [fusion_op]
            merged_fusion_rule.op_list.insert(0, fusion_op)
    return merged_fusion_rule


class FusionRuleParser:
    """
    the class for parse fusion rule.
    """

    def __init__(self: any, path: str) -> None:
        self.json_path = path
        self.json_object = None
        self.fusion_op_name_to_op_map = {}
        self.op_name_to_fusion_op_name_map = {}
        self.op_list = []
        self.input_nodes = []

    @staticmethod
    def _check_key_exist(json_object: any, key: str) -> None:
        if key not in json_object:
            log.print_warn_log('There is no "%s" element in fusion rule file.' % key)
            raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    @staticmethod
    def _make_output_desc(output_desc_list: list, name: str) -> None:
        if len(output_desc_list) == 0:
            output_desc = OutputDesc(name, None, "", [])
            output_desc_list.append(output_desc)
        else:
            for (index, _) in enumerate(output_desc_list):
                if output_desc_list[index].origin_name == "":
                    output_desc_list[index].origin_name = name

    @staticmethod
    def _process_ffts_op_name(item):
        if ConstManager.FFTS_MANUAL_MODE_FIELD in item:
            item = dump.process_op_name(item)
        return item

    def analysis_fusion_rule(self: any) -> None:
        """
        Analysis fusion json file
        """
        self.json_object = FileUtils.load_json_file(self.json_path)
        self._parse_fusion_op_json_object()
        # analysis and filter the parsed op list
        filtering = QuantFilter(self.op_list)
        filtering.process_filtering()

    def make_fusion_op_name(self: any, name: str, l1_fusion_no: str, original_op_names: list) -> None:
        """
        Make fusion op name by group op name and original op names
        :return the fusion op name
        """
        # the fusion op name priority:
        # l1_fusion_no -> original_op_names -> name
        if l1_fusion_no != "":
            # the l1_fusion_no is not empty,
            # the fusion op name is the l1_fusion_no
            self.op_name_to_fusion_op_name_map[name] = l1_fusion_no
            return

        if original_op_names:
            if len(original_op_names) == 1:
                # There is one original op name
                if original_op_names[0] == '':
                    # the original name is empty, the fusion op name is op name
                    self.op_name_to_fusion_op_name_map[name] = name
                else:
                    # the original name is not empty,
                    # the fusion op name is original op name
                    self.op_name_to_fusion_op_name_map[name] = original_op_names[0]
            else:
                # The original op name more than one,
                # the fusion op name is uuid names
                self.op_name_to_fusion_op_name_map[name] = uuid.uuid3(uuid.NAMESPACE_DNS, ''.join(original_op_names))
        else:
            self.op_name_to_fusion_op_name_map[name] = name

    def get_origin_name_to_op_name_map(self: any) -> dict:
        """
        Get origin name to op name map
        :return: the map
        """
        origin_name_to_op_name_map = {}
        for fusion_op in self.op_list:
            for origin_name in fusion_op.attr.original_op_names:
                origin_name_to_op_name_map[origin_name] = fusion_op.op_name
        return origin_name_to_op_name_map

    def check_array_object_valid(self: any, json_object: any, key: str) -> None:
        """
        Check array object valid
        :param json_object:the json object
        :param key : key in json
        """
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], list):
            log.print_error_log('The content of the json file "%r" is invalid. The "%s" element is not an array.'
                                % (self.json_path, key))
            raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def check_string_object_valid(self: any, json_object: any, key: str) -> None:
        """
        Check string object valid
        :param json_object:the json object
        :param key : key in json
        """
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], str):
            log.print_error_log('The content of the json file "%r" is invalid. The "%s" element is not a string.'
                                % (self.json_path, key))
            raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def get_fusion_op_list(self: any, op_name: str) -> (list, FusionOp):
        """
        Get the fusion op list by op name
        :param op_name: the op name
        :return :the fusion op list, the fusion op by name
        """
        if op_name not in self.op_name_to_fusion_op_name_map:
            message = 'There is no "%s" in the fusion rule file.' % op_name
            log.print_warn_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR, message)
        fusion_op_name = self.op_name_to_fusion_op_name_map.get(op_name)
        fusion_op_list = self.fusion_op_name_to_op_map.get(fusion_op_name, [])

        # get fusion op in list by op name
        fusion_op_info = None
        for operator in fusion_op_list:
            if operator.op_name == op_name:
                fusion_op_info = operator
                break
        if fusion_op_info is None:
            message = 'There is no "%s" in the fusion rule file.' % op_name
            log.print_warn_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR, message)
        return fusion_op_list, fusion_op_info

    def _adjust_rename_node(self: any) -> None:
        for _, fusion_op_list in self.fusion_op_name_to_op_map.items():
            if len(fusion_op_list) == 1 and self._is_rename_node(fusion_op_list[0]):
                self._make_output_desc(fusion_op_list[0].output_desc, fusion_op_list[0].attr.original_op_names[0])

    def _parse_fusion_op_json_object(self: any) -> None:
        # check graph element in json file
        self.check_array_object_valid(self.json_object, ConstManager.GRAPH_OBJECT)
        for graph in self.json_object[ConstManager.GRAPH_OBJECT]:
            # check op element in graph value
            self.check_array_object_valid(graph, ConstManager.OP_OBJECT)
            for operator in graph[ConstManager.OP_OBJECT]:
                self._parse_op_object(operator)

            # adjust the output desc for the rename node
            self._adjust_rename_node()

        self.op_list.sort(key=lambda x: x.attr.get_op_sequence())

    def _parse_input_nodes(self, op_object):
        if ConstManager.DATA_OBJECT == op_object.get(ConstManager.TYPE_OBJECT):
            if op_object.get(ConstManager.NAME_OBJECT):
                self.input_nodes.append(op_object.get(ConstManager.NAME_OBJECT))

    def _parse_input(self: any, op_object: any) -> list:
        input_list = []
        # data layer has no input layer
        if ConstManager.INPUT_OBJECT in op_object:
            if not isinstance(op_object[ConstManager.INPUT_OBJECT], list):
                log.print_error_log('The content of the json file "%r" is invalid. The "%s" element is not '
                                    'an array.' % (self.json_path, ConstManager.INPUT_OBJECT))
                raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)
            for item in op_object[ConstManager.INPUT_OBJECT]:
                if item == "" and len(op_object[ConstManager.INPUT_OBJECT]) == 1:
                    break
                # skip control edge
                if not item.endswith(':-1') and item != "":
                    item = self._process_ffts_op_name(item)
                    input_list.append(item)
        return input_list

    def _parse_output_desc_in_attr(self: any, output_desc_attr: any, default_index: int) -> OutputDesc:
        origin_name = self._get_string_value_in_attr(output_desc_attr, ConstManager.ORIGIN_NAME_OBJECT)
        origin_output_index = self._get_int_value_in_attr(output_desc_attr, ConstManager.ORIGIN_OUTPUT_INDEX_OBJECT)
        if origin_output_index is None:
            origin_output_index = default_index
        origin_output_format = self._get_string_value_in_attr(output_desc_attr, ConstManager.ORIGIN_FORMAT_OBJECT)
        if origin_output_format == '':
            origin_output_format = self._get_string_value_in_attr(
                output_desc_attr, ConstManager.GE_ORIGIN_FORMAT_OBJECT)
        origin_output_shape = self._get_origin_shape_in_attr(output_desc_attr)
        return OutputDesc(origin_name, origin_output_index, origin_output_format, origin_output_shape)

    def _parse_output_desc(self: any, op_object: any) -> list:
        output_desc_list = []
        # get output desc
        if ConstManager.OUTPUT_DESC_OBJECT in op_object:
            default_index = 0
            for output_desc_object in op_object[ConstManager.OUTPUT_DESC_OBJECT]:

                d_type = ""
                if ConstManager.D_TYPE in output_desc_object:
                    d_type = output_desc_object.get(ConstManager.D_TYPE)

                if ConstManager.ATTR_OBJECT in output_desc_object:
                    output_desc = self._parse_output_desc_in_attr(
                        output_desc_object[ConstManager.ATTR_OBJECT], default_index)
                    output_desc.set_data_type(d_type)
                    output_desc_list.append(output_desc)
                default_index += 1
        return output_desc_list

    def _is_rename_node(self: any, fusion_op: FusionOp) -> bool:
        return len(fusion_op.attr.original_op_names) == 1 and \
               self.op_name_to_fusion_op_name_map.get(fusion_op.op_name) == fusion_op.attr.original_op_names[0]

    def _parse_attr(self: any, op_object: any, op_name: str) -> (OpAttr, bool):
        # check attr element is valid
        if ConstManager.ATTR_OBJECT not in op_object:
            attr_array = []
        else:
            self.check_array_object_valid(op_object, ConstManager.ATTR_OBJECT)
            attr_array = op_object[ConstManager.ATTR_OBJECT]
        is_multi_op = self._get_bool_value_in_attr(attr_array, ConstManager.IS_MULTI_OP)
        # get l1_fusion_sub_graph_no
        l1_fusion_no = self._get_string_value_in_attr(attr_array, ConstManager.L1_FUSION_SUB_GRAPH_NO_OBJECT)
        # get original op names
        original_op_names, have_origin = self._get_original_op_names_in_attr(attr_array, op_name)
        op_sequence = self._parse_id_object(op_object)
        return OpAttr(original_op_names, l1_fusion_no, is_multi_op, op_sequence), have_origin

    def _parse_id_object(self: any, op_object: any) -> int:
        op_sequence = 0
        if ConstManager.ID_OBJECT in op_object:
            self._check_int_object_valid(op_object, ConstManager.ID_OBJECT)
            op_sequence = op_object[ConstManager.ID_OBJECT]
        return op_sequence

    def _parse_op_object(self: any, op_object: dict) -> None:
        # check name element is valid
        self.check_string_object_valid(op_object, ConstManager.NAME_OBJECT)
        name = op_object[ConstManager.NAME_OBJECT]
        if ConstManager.FFTS_MANUAL_MODE_FIELD in name:
            name = dump.process_op_name(name)
        # check type element is valid
        self.check_string_object_valid(op_object, ConstManager.TYPE_OBJECT)
        self._parse_input_nodes(op_object)

        input_list = self._parse_input(op_object)

        output_desc_list = self._parse_output_desc(op_object)
        attr, have_origin = self._parse_attr(op_object, name)
        if not have_origin:
            self._make_output_desc(output_desc_list, name)

        self.make_fusion_op_name(name, attr.l1_fusion_no, attr.original_op_names)
        fusion_op_name = self.op_name_to_fusion_op_name_map.get(name)
        fusion_op = FusionOp(0, name, input_list, op_object[ConstManager.TYPE_OBJECT], output_desc_list, attr)
        if fusion_op_name in self.fusion_op_name_to_op_map:
            fusion_op.op_id = self.fusion_op_name_to_op_map.get(fusion_op_name)[0].op_id
            fusion_op_name_list = \
                [_fusion_op.op_name for _fusion_op in self.fusion_op_name_to_op_map.get(fusion_op_name)]
            if name not in fusion_op_name_list:
                self.fusion_op_name_to_op_map.get(fusion_op_name).append(fusion_op)
                self.op_list.append(fusion_op)
        else:
            fusion_op.op_id = len(self.fusion_op_name_to_op_map)
            self.fusion_op_name_to_op_map[fusion_op_name] = [fusion_op]
            self.op_list.append(fusion_op)

    def _get_string_value_in_attr(self: any, attr_array: list, key: str) -> str:
        value = ""
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self.check_string_object_valid(value_value, ConstManager.STRING_TYPE_OBJECT)
                value = value_value[ConstManager.STRING_TYPE_OBJECT]
                break
        return value

    def _get_int_value_in_attr(self: any, attr_array: list, key: str) -> int:
        value = None
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self._check_int_object_valid(value_value, ConstManager.INT_TYPE_OBJECT)
                value = value_value[ConstManager.INT_TYPE_OBJECT]
                break
        return value

    def _get_origin_shape_in_attr(self: any, attr_array: list) -> list:
        value = []
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == ConstManager.GE_ORIGIN_SHAPE_OBJECT:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self._check_key_exist(value_value, ConstManager.LIST_TYPE_OBJECT)
                if ConstManager.INT_TYPE_OBJECT in value_value[ConstManager.LIST_TYPE_OBJECT]:
                    self.check_array_object_valid(
                        value_value[ConstManager.LIST_TYPE_OBJECT], ConstManager.INT_TYPE_OBJECT)
                    value = value_value[ConstManager.LIST_TYPE_OBJECT][ConstManager.INT_TYPE_OBJECT]
                break
        return value

    def _get_bool_value_in_attr(self: any, attr_array: list, key: str) -> bool:
        value = False
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == key:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value_value = attr[ConstManager.VALUE_OBJECT]
                self._check_bool_object_valid(value_value, ConstManager.BOOL_TYPE_OBJECT)
                value = value_value[ConstManager.BOOL_TYPE_OBJECT]
                break
        return value

    def _get_original_op_names_in_attr(self: any, attr_array: list, op_name: str) -> (list, bool):
        array = []
        match = False
        for attr in attr_array:
            self.check_string_object_valid(attr, ConstManager.KEY_OBJECT)
            key_value = attr[ConstManager.KEY_OBJECT]
            if key_value == ConstManager.ORIGINAL_OP_NAMES_OBJECT:
                self._check_key_exist(attr, ConstManager.VALUE_OBJECT)
                value = attr[ConstManager.VALUE_OBJECT]
                self._check_key_exist(value, ConstManager.LIST_TYPE_OBJECT)
                if ConstManager.STRING_TYPE_OBJECT not in value[ConstManager.LIST_TYPE_OBJECT]:
                    array = ['']
                else:
                    self.check_array_object_valid(value[ConstManager.LIST_TYPE_OBJECT], ConstManager.STRING_TYPE_OBJECT)
                    array = value[ConstManager.LIST_TYPE_OBJECT][ConstManager.STRING_TYPE_OBJECT]
                match = True
                break
        if not match:
            array.append(op_name)
        return array, match

    def _check_int_object_valid(self: any, json_object: any, key: str) -> None:
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], int):
            log.print_error_log('The content of the json file "%r" is invalid. The "%s" element is not a integer.'
                                % (self.json_path, key))
            raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)

    def _check_bool_object_valid(self: any, json_object: any, key: str) -> None:
        self._check_key_exist(json_object, key)
        if not isinstance(json_object[key], bool):
            log.print_error_log('The content of the json file "%r" is invalid. The "%s" element is not a bool.'
                                % (self.json_path, key))
            raise CompareError(CompareError.MSACCUCMP_PARSER_JSON_FILE_ERROR)
