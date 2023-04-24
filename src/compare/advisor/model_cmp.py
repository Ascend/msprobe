
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2012-2022. All rights reserved.

import json
import os
import sys

from src.compare.mscmp_advisor import CompareAdvisor


CLASS_TYPE = {'op': '0', 'model': '1'}
ERROR_CODE = {'success': '0', 'optimized': '1'}
EXTEND_TYPE = {'list': '0', 'table': '1', 'sourcedata': '2'}
EXTEND_DATA_TYPE = {'str': '0', 'int': '1', 'double': '2'}


class ExtendResult:
    def __init__(self):
        self.type = '0'
        self.extend_title = ""
        self.data_type = []  # table type is an array with multiple elements, list type with only one element
        self.key = []  # this field is only used for table type result
        self.value = []  # table type is a two-dimensional array, list type is a one-dimensional array


class Result:
    def __init__(self):
        self.class_type = CLASS_TYPE.get('op')
        self.error_code = ERROR_CODE.get('success')
        self.summary = ""
        self.extend_result = []

    def generate(self):
        extend_data = []
        for item in self.extend_result:
            data = {
                "type": item.type,
                "extendTitle": item.extend_title,
                "dataType": item.data_type,
                "key": item.key,
                "value": item.value
            }
            extend_data.append(data)
        res = {
            "classType": self.class_type,
            "errorCode": self.error_code,
            "summary": self.summary,
            "extendResult": extend_data
        }
        outputstr = json.dumps(res)
        return outputstr


def evaluate(data_path, parameter):
    """
    interface function called by msadvisor
    Args:
        data_path: string data_path
        input_nodes: input nodes list
    Returns:
        json string of result info
        result must by ad_result
    """
    # do evaluate work by file data
    # my code begin
    input_file = os.path.realpath(data_path)

    parameters = json.loads(parameter)
    input_nodes = parameters.get("input_nodes")
    input_nodes = [] if input_nodes is None else input_nodes

    compare_advisor = CompareAdvisor(input_file, input_nodes)
    advisor_result = compare_advisor.advisor()
    advisor_result.print_advisor_log()
    result_dict = {
        "Detection Type": advisor_result.advisor_type,
        "Operator Index": advisor_result.operator_index,
        "Expert Advice": advisor_result.advisor_message
    }

    # fill result
    result = Result()
    result.class_type = CLASS_TYPE.get('model')
    result.error_code = ERROR_CODE.get('success')
    result.summary = "Suggestions for accuracy comparison of the entire network"
    extend_result = ExtendResult()
    # the value of extend_result.type cat be 'list' 'table' 'sourcedata'
    extend_result.type = EXTEND_TYPE.get('table')

    # list type result
    if extend_result.type == EXTEND_TYPE.get('list'):
        extend_result.extend_title = "Recommendations of Ops_Not_Support_Heavy_Format"
        extend_result.data_type.append(EXTEND_DATA_TYPE.get('str'))
        extend_result.value.append("Modify the operation to support light format")
        extend_result.value.append("Modify the operation to support heavy format")
        result.extend_result.append(extend_result)

    # table type result
    elif extend_result.type == EXTEND_TYPE.get('table'):
        extend_result.extend_title = "suggestions"
        extend_result.key.append("Detection Type")
        extend_result.key.append("Operator Index")
        extend_result.key.append("Expert Advice")

        extend_result.data_type.append(EXTEND_DATA_TYPE.get('str'))
        extend_result.data_type.append(EXTEND_DATA_TYPE.get('str'))
        extend_result.data_type.append(EXTEND_DATA_TYPE.get('str'))

        value = []
        value.append(result_dict.get("Detection Type"))
        value.append(result_dict.get("Operator Index"))
        value.append(result_dict.get("Expert Advice"))
        extend_result.value.append(value)

        result.extend_result.append(extend_result)
    elif extend_result.type == EXTEND_TYPE.get('sourcedata'):
        extend_result.extend_title = "sourcedatapath"
        extend_result.data_type.append(EXTEND_DATA_TYPE.get('str'))
        extend_result.value.append("/home/datapath/")
        result.extend_result.append(extend_result)
    return result.generate()


if __name__ == "__main__":
    DATA_PATH = sys.argv[1]
    my_input_nodes = []
    param = json.dumps({"inputs_nodes": my_input_nodes})
    ret = evaluate(DATA_PATH, param)
