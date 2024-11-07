
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves xxxx function.
"""
import os
import re
import time

from cmp_utils import log
from cmp_utils.file_utils import FileUtils
from cmp_utils import path_check as path_utils
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError
from compare_vector import VectorComparison
from vector_cmp.fusion_manager.fusion_rule_parser import FusionRuleParser


class BatchCompare:
    """
    The class for batch compare
    """
    DUMP_FILE_PATH_FORMAT = "dump_path/time/device_id/model_name/model_id/dump_step/dump_file"

    def __init__(self: any) -> None:
        self.model_name_to_json_map = {}
        self.json_path_to_dump_path_map = {}

    @staticmethod
    def _check_path_valid(path: str) -> None:
        ret = path_utils.check_path_valid(path, True, False, path_utils.PathType.Directory)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            raise CompareError(ret)

    def check_fusion_rule_json_dir(self: any, fusion_rule_json_dir: str) -> bool:
        """
        Check fusion rule is dir or not
        :param fusion_rule_json_dir: the path for fusion rule json
        :return:bool
        """
        fusion_rule_path = os.path.realpath(fusion_rule_json_dir)
        if os.path.isfile(fusion_rule_path):
            return False
        self._check_path_valid(fusion_rule_json_dir)
        return True

    def check_argument_valid(self: any, arguments: any) -> None:
        """
        Check argument valid
        """
        if arguments.op_name:
            log.print_error_log(
                "the {} single operator comparison is not supported for batch network comparison.".format(
                    arguments.op_name))
            raise CompareError(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

        npu_dump_path = os.path.realpath(arguments.my_dump_path)
        self._check_path_valid(npu_dump_path)
        # check whether the path ends with a timestamp.
        path = os.path.split(npu_dump_path)
        # path[1]:last level directory
        if not path[1].isdigit():
            log.print_error_log("The {} path must end with a timestamp.".format(npu_dump_path))
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def compare(self: any, arguments: any) -> int:
        """
        Compare the entire network in batches
        :param arguments: the command parameters
        :return: the compare finish status code
        """
        self.check_argument_valid(arguments)
        self._make_model_name_to_json_map(os.path.realpath(arguments.fusion_rule_file))
        self._make_json_path_to_dump_path_map(os.path.realpath(arguments.my_dump_path))
        return self._execute_batch_compare(arguments)

    def _make_model_name_to_json_map(self: any, json_dir_path: str) -> None:
        for json_file_name in os.listdir(json_dir_path):
            json_file_path = os.path.join(json_dir_path, json_file_name)
            if json_file_path.endswith(".json"):
                self._parse_json_file(json_file_path)

        if not self.model_name_to_json_map:
            log.print_error_log('There is no fusion rule json file in "%r".' % json_dir_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def _parse_json_file(self: any, json_file_path: str) -> None:
        fusion_rule_parse = FusionRuleParser(json_file_path)
        json_object = FileUtils.load_json_file(json_file_path)
        fusion_rule_parse.check_array_object_valid(json_object, ConstManager.GRAPH_OBJECT)
        for graph in json_object[ConstManager.GRAPH_OBJECT]:
            fusion_rule_parse.check_string_object_valid(graph, ConstManager.NAME_OBJECT)
            model_name = graph[ConstManager.NAME_OBJECT]
            self.model_name_to_json_map[model_name] = json_file_path

    def _make_map_for_inconsistent_timestamp(self: any, graph_name: str, dump_file_path_map: dict,
                                             npu_dump_dir: str) -> None:
        # 1.get the number list from graph name.
        # graph_name:'ge_default_20210420113943_73',graph_name_number_list:['20210420113943', '73'].
        graph_name_number_list = re.findall(r"\d+", graph_name)
        for _, model_name in enumerate(dump_file_path_map):
            # 1.get the number list from model name.for example:
            # model_name:'ge_default_20210420113944_73',model_name_number_list:['20210420113944', '73'].
            model_name_number_list = re.findall(r"\d+", model_name)
            # check graph_name and model_name inconsistent timestamps
            if len(graph_name_number_list) != 2 or len(graph_name_number_list) != len(model_name_number_list):
                log.print_npu_path_valid_message(npu_dump_dir, self.DUMP_FILE_PATH_FORMAT)
                raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
            if graph_name_number_list[1] == model_name_number_list[1]:
                dump_file_path = dump_file_path_map.get(model_name)
                log.print_warn_log(
                    "The {0} and in {1} dump data are not generated at the same time. "
                    "The result may be incorrect.".format(graph_name, ",".join(dump_file_path)))
                self.json_path_to_dump_path_map[self.model_name_to_json_map.get(graph_name)] = dump_file_path

    def _make_json_path_to_dump_path_map(self: any, npu_dump_dir: str) -> None:
        dump_file_path_map = self._get_npu_dump_file_map(npu_dump_dir)
        graph_name_list = self.model_name_to_json_map.keys()
        for graph_name in graph_name_list:
            dump_file_path = dump_file_path_map.get(graph_name)
            if dump_file_path:
                self.json_path_to_dump_path_map[self.model_name_to_json_map.get(graph_name)] = dump_file_path
            else:
                self._make_map_for_inconsistent_timestamp(graph_name, dump_file_path_map, npu_dump_dir)

        if not self.json_path_to_dump_path_map:
            log.print_error_log("The {} not match fusion rule file".format(npu_dump_dir))
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def _execute_batch_compare(self: any, arguments: any) -> int:
        if os.path.islink(os.path.abspath(arguments.output_path)):
            log.print_error_log('The path "%r" is a softlink, not permitted.' % arguments.output_path)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        output_path = os.path.realpath(arguments.output_path)
        my_dump_path = os.path.realpath(arguments.my_dump_path)
        ret = CompareError.MSACCUCMP_NONE_ERROR
        for key, value in self.json_path_to_dump_path_map.items():
            for dump_path in value:
                path_data = dump_path.replace(my_dump_path, "").split("/")
                # remove model_id from path_data,path_data:['0', 'ge_default_20210420113943_71', '10', '1']
                model_id_index = len(path_data) - 2
                path_data.pop(model_id_index)
                if "" in path_data:
                    path_data.remove("")
                result_csv = 'result_%s.csv' % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
                # generate result file,for example:'0_ge_default_20210420113943_71_1_result_20210715022552.csv'
                file_name = "".join(["_".join(path_data), "_", result_csv])
                arguments.my_dump_path = dump_path
                arguments.fusion_rule_file = key
                compare = VectorComparison(arguments)
                compare.set_output_path(os.path.join(output_path, file_name))
                ret = compare.compare()
        return ret

    def _get_npu_dump_file_map_by_model_id(self: any, model_name_dir: str, dump_file_path_map: dict) -> None:
        for model_id in os.listdir(model_name_dir):
            model_id_dir = os.path.join(model_name_dir, model_id)
            self._check_path_valid(model_id_dir)
            model_id_list = os.listdir(model_id_dir)
            model_name = os.path.basename(model_name_dir)
            if len(model_id_list) != 0:
                step = max(model_id_list) if len(model_id_list) > 1 else model_id_list[0]
                dump_step_path = os.path.join(model_id_dir, step)
                if dump_file_path_map.get(model_name):
                    dump_file_path_map.get(model_name).append(dump_step_path)
                else:
                    dump_file_path_map[model_name] = [dump_step_path]

    def _get_npu_dump_file_map(self: any, npu_dump_dir: str) -> dict:
        dump_file_path_map = {}
        for device_id in os.listdir(npu_dump_dir):
            device_id_dir = os.path.join(npu_dump_dir, device_id)
            self._check_path_valid(device_id_dir)
            for model_name in os.listdir(device_id_dir):
                model_name_dir = os.path.join(device_id_dir, model_name)
                self._check_path_valid(model_name_dir)
                self._get_npu_dump_file_map_by_model_id(model_name_dir, dump_file_path_map)
        if not dump_file_path_map:
            log.print_npu_path_valid_message(npu_dump_dir, self.DUMP_FILE_PATH_FORMAT)
            raise CompareError(CompareError.MSACCUCMP_INVALID_PATH_ERROR)
        return dump_file_path_map
