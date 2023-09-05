# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
Quant Filter class, process the fusion json file
"""
import re

from cmp_utils import log


class QuantFilter:
    """
    the class for recognize and filter the quant/dequant ops
    """

    NORMAL_OP = 0
    MIDDLE_OP = 1
    QUANT_OP = 2
    DEQUANT_OP = 3
    QUANT_DEQUANT_OP = 5

    def __init__(self: any, op_list: list) -> None:
        """
        :param op_list: the list of parsed ops
        """
        self._op_list = op_list
        self._op_output_type_map = {}  # structure like {op_name: [out1_type, out2_type]}
        self._quant_name_pattern = r"_quant_layer|/AscendQuant|/AscendWeightQuant|\.quant|\.weight_quant"
        self._dequant_name_pattern = \
            r"_dequant_layer|_anti_quant_layer|/AscendDequant|/AntiQuant|\.dequant|\.anti_quant"

    @staticmethod
    def _check_out_type(op_name: str, name_type: int, in_pairs: bool, output_node: any) -> int:
        data_type = output_node.data_type
        _out_type = QuantFilter._match_existing_type(name_type, data_type, in_pairs)

        # Not matched type, treat as a normal op
        if _out_type is None:
            log.print_warn_log("[{}] this op looks like a quantization op, but it does not have proper output type"
                               .format(op_name))
            if in_pairs:
                _out_type = QuantFilter.MIDDLE_OP
            else:
                _out_type = QuantFilter.NORMAL_OP
        return _out_type

    @staticmethod
    def _match_existing_type(name_type: int, data_type: str, in_pairs: bool) -> int:
        _matched_type = None
        if name_type == QuantFilter.NORMAL_OP:
            if in_pairs:
                _matched_type = QuantFilter.MIDDLE_OP
            else:
                _matched_type = QuantFilter.NORMAL_OP
        elif name_type == QuantFilter.QUANT_OP:
            if data_type == "DT_INT8":
                _matched_type = QuantFilter.QUANT_OP
        elif name_type == QuantFilter.DEQUANT_OP:
            if data_type == "DT_FLOAT16":
                _matched_type = QuantFilter.DEQUANT_OP
        elif name_type == QuantFilter.QUANT_DEQUANT_OP:
            # 5--quant & dequant
            if data_type == "DT_INT8":
                _matched_type = QuantFilter.QUANT_OP
            if data_type == "DT_FLOAT16":
                _matched_type = QuantFilter.DEQUANT_OP

        return _matched_type

    def process_filtering(self: any) -> None:
        """
        check through the op list, mark in attr.is_quant_filter the ops should be filtered.
        """
        self._op_output_type_map = {}

        # fetch ops in order
        for op in self._op_list:

            _op_name = op.op_name

            # check name type using regex
            _name_type = self._check_name_type(op)

            # check inputs to determine if in middle of pairs
            _in_pairs = self._check_in_pairs(op)

            # store the output type of each output desc
            self._op_output_type_map[_op_name] = []

            for output in op.output_desc:
                # combine the conditions above to determine filtering or not
                _out_type = QuantFilter._check_out_type(_op_name, _name_type, _in_pairs, output)
                self._op_output_type_map[_op_name].append(_out_type)

            # store filter result in result set
            self._add_filter_list(op, _in_pairs)

    def get_out_type_map(self: any) -> dict:
        """
        get output type map

        :return: output map
        """
        return self._op_output_type_map

    def _check_in_pairs(self: any, op: any) -> bool:
        _inputs = op.input_list

        _is_in_pair = False
        for _input in _inputs:
            if ':' not in _input:
                continue
            last_colon_index = _input.rfind(':')
            _input_name = _input[:last_colon_index]
            if _input[last_colon_index + 1:].isdigit():
                _input_out_index = int(_input[last_colon_index + 1:])
            else:
                log.print_warn_log("[{}] the input operator of op with index {} is invalid."
                                   .format(op.op_name, _input))
                continue
            _out_list = self._op_output_type_map.get(_input_name)
            if _out_list is None or len(_out_list) <= _input_out_index:
                log.print_warn_log("[{}] the input operator of op {} does not exist in previous graph."
                                   .format(op.op_name, _input_name))
                continue
            _input_out_type = _out_list[_input_out_index]
            if _input_out_type in (QuantFilter.QUANT_OP, QuantFilter.MIDDLE_OP):
                _is_in_pair = True
                break

        return _is_in_pair

    def _add_filter_list(self: any, op: any, in_pairs: bool) -> None:
        _outputs = self._op_output_type_map.get(op.op_name)
        _is_dequant = True
        for _output in _outputs:
            if _output != QuantFilter.DEQUANT_OP:
                _is_dequant = False
                break
        _need_filter = in_pairs and not _is_dequant

        op.attr.quant_filter = _need_filter

    def _check_name_type(self, op) -> int:
        """
        check the op type with name, using regex

        :param op: the op name need to be checked
        :return: type of this op, Quant=2/Dequant=3/Quant&Dequant=5/Normal=0
        """

        _op_type = QuantFilter.NORMAL_OP

        # All the keywords of quant/dequant ops:
        # caffe: "_quant_layer", "_dequant_layer", "_anti_quant_layer"
        # tensorflow: "/AscendQuant", "/AscendWeightQuant", "/AscendDequant", "/AntiQuant"
        # onnx: ".quant", ".weight_quant", ".dequant", ".anti_quant"

        # find quant keywords
        if re.search(self._quant_name_pattern, op.op_name):
            _op_type += QuantFilter.QUANT_OP
        elif re.search(self._quant_name_pattern, ''.join(op.attr.original_op_names)):
            _op_type += QuantFilter.QUANT_OP

        # find dequant keywords
        if re.search(self._dequant_name_pattern, op.op_name):
            _op_type += QuantFilter.DEQUANT_OP
        elif re.search(self._dequant_name_pattern, ''.join(op.attr.original_op_names)):
            _op_type += QuantFilter.DEQUANT_OP

        return _op_type
