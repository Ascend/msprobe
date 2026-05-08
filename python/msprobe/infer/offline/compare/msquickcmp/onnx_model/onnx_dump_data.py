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
This class is used to generate GUP dump data of the ONNX model.
"""
import os
import re
import time

import onnx
import onnxruntime
import numpy as np

from msprobe.core.common.log import logger
from msprobe.core.common.file_utils import check_file_or_directory_path, create_directory
from msprobe.infer.offline.compare.msquickcmp.common.dump_data import DumpData
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException
from msprobe.infer.offline.compare.msquickcmp.common.utils import InputShapeError, load_npy_from_buffer
from msprobe.infer.offline.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_bin_file_to_npy
from msprobe.infer.utils.file_open_check import ms_open
from msprobe.infer.utils.util import load_file_to_read_common_check
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE


NODE_TYPE_TO_DTYPE_MAP = {
    "tensor(int)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.double,
    "tensor(bool)": np.bool_,
    "tensor(complex64)": np.complex64,
    "tensor(complex128)": np.complex128,
}
MAX_PROTOBUF = 2000000000
ONNX_DTYPE = {1: np.float32, 2: np.float64}
MAX_FILE_NAME_LEN = 255


class OnnxDumpData(DumpData):
    """
    This class is used to generate dump data of the ONNX model.
    """

    def __init__(self, arguments: CmpArgsAdapter, npu_dump_npy_path=None):
        super().__init__()
        self.model_path = arguments.golden_path
        self.output_path = arguments.output_path
        self.input_data = arguments.input_data
        self.input_shape = arguments.input_shape
        self.dym_shape_range = arguments.dym_shape_range
        self.onnx_fusion_switch = arguments.onnx_fusion_switch
        self.cann_path = arguments.cann_path

        self._check_path_exists(self.model_path, extentions="onnx")

        self.input_shapes = utils.parse_input_shape(self.input_shape)
        self.data_dir, self.onnx_dump_data_dir, self.model_dir = self._create_dir()

        self.net_output, self.inputs_map, self.extend_inputs_map = {}, {}, {}
        self.origin_model, origin_model_contents = self._load_onnx(self.model_path)

        self.dump_model_with_inputs_path = self._new_model_save_path(self.model_path)
        self.model_with_inputs = self.origin_model
        self.model_with_inputs_session = self._load_session(self.model_path)

    @staticmethod
    def _check_input_shape_fix_value(op_name, model_shape, input_shape):
        message = "fixed input tensor dim not equal to model input dim." "tensor_name:%s, %s vs %s" % (
            op_name,
            str(input_shape),
            str(model_shape),
        )
        if len(model_shape) != len(input_shape):
            logger.error(message)
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)
        for index, value in enumerate(model_shape):
            if value is None or isinstance(value, str):
                continue
            if input_shape[index] != value:
                logger.error(message)
                raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR)

    def generate_inputs_data(self, npu_dump_data_path=None, use_aipp=False):
        inputs_tensor_info = self._get_inputs_tensor_info()
        if use_aipp:
            if not npu_dump_data_path:
                logger.error("find no aipp op in dump data")
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            self._check_path_exists(npu_dump_data_path)
            self.inputs_map = self._get_inputs_data_aipp(self.data_dir, inputs_tensor_info, npu_dump_data_path)
        else:
            self.inputs_map = self._get_inputs_data(inputs_tensor_info)

    def get_output_map(self, output_list):
        output_map, res_idx, output_list_size = {}, 0, len(output_list)
        for node in self.origin_model.graph.node:
            for node_output in node.output:
                output_map[node_output] = output_list[res_idx]
                res_idx += 1
                if res_idx >= output_list_size:
                    return output_map
        return output_map

    def get_input_map(self, input_map, output_list):
        output_map = self.get_output_map(output_list)
        input_map = self.augment_input_map(input_map, output_map)
        return input_map

    def augment_input_map(self, input_map, output_map):
        for temp in self.origin_model.graph.initializer:
            npy_data = load_npy_from_buffer(temp.raw_data, ONNX_DTYPE.get(temp.data_type), temp.dims)
            input_map[temp.name] = npy_data
        input_map = {**input_map, **output_map}
        return input_map
    
    def generate_dump_data(self, npu_dump_path=None, om_parser=None):
        self._modify_model_add_outputs_nodes(
            self.model_with_inputs, self.dump_model_with_inputs_path
        )
        session = self._load_session(self.dump_model_with_inputs_path)
        dump_bins = self._run_model(session, self.inputs_map)
        augment_inputs_map = self.get_input_map(self.inputs_map, dump_bins)
        net_output_node = [output_item.name for output_item in self.model_with_inputs_session.get_outputs()]
        self._save_dump_data(dump_bins, self.model_with_inputs, net_output_node, augment_inputs_map)

        return self.onnx_dump_data_dir

    def _load_session(self, model_path):
        options = onnxruntime.SessionOptions()
        if not self.onnx_fusion_switch:
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            model_path = load_file_to_read_common_check(model_path)
            infersession = onnxruntime.InferenceSession(model_path, options)
        except Exception as e:
            logger.error(f"Please check onnx model can run in local env. Error: {e}")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)
        return infersession

    def _load_onnx(self, model_path):
        # model_path str -> read as bytes -> deserialize to onnx_model
        #                                 -> onnxruntime load as session
        model_path = load_file_to_read_common_check(model_path)
        with ms_open(model_path, "rb", max_size=TENSOR_MAX_SIZE) as ff:
            model_contents = ff.read()
        onnx_model = onnx.load_model(model_path)
        unique_index = 999
        name_set = set((node.name for node in onnx_model.graph.node))

        def get_unique_name(ori_name, index):
            new_name = f"{ori_name}_{index}"
            if new_name not in name_set:
                return new_name, index + 1
            else:
                return get_unique_name(ori_name, index + 1)

        for index, node in enumerate(onnx_model.graph.node):
            if node.name:
                continue
            new_name = node.op_type + "_" + str(index)
            if new_name in name_set:
                new_name, unique_index = get_unique_name(new_name, unique_index)
            name_set.add(new_name)
            node.name = new_name
        return onnx_model, model_contents

    def _new_model_save_path(self, origin_path):
        save_name = "new_" + os.path.basename(origin_path)
        return os.path.join(self.model_dir, save_name)

    def _create_dir(self):
        # create input directory
        data_dir = os.path.join(self.output_path, "input")
        create_directory(data_dir)

        # create dump_data/onnx directory
        onnx_dump_data_dir = os.path.join(self.output_path, "dump_data/onnx")
        create_directory(onnx_dump_data_dir)

        # create model directory
        model_dir = ""
        if self.dym_shape_range:
            model_relative_name = "../model"
        else:
            model_relative_name = "model"
            model_dir = os.path.join(self.output_path, model_relative_name)
            create_directory(model_dir)
        return data_dir, onnx_dump_data_dir, model_dir

    def _modify_model_add_outputs_nodes(self, onnx_model, save_path):
        del onnx_model.graph.output[:]

        onnx_model.graph.output.extend(
            onnx.ValueInfoProto(name=tensor_name)
            for node in onnx_model.graph.node
            for tensor_name in node.output
        )
        
        model_size = onnx_model.ByteSize()
        save_external_flag = model_size < 0 or model_size > MAX_PROTOBUF
        
        logger.debug(f"Modified model has size over 2G: {save_external_flag}")
        
        onnx.save_model(
            onnx_model, 
            save_path,
            save_as_external_data=save_external_flag
        )
        
        logger.info(f"Modified model has being saved successfully at: {os.path.abspath(save_path)}")
        
    def _get_inputs_tensor_info(self):
        inputs_tensor_info = []
        input_tensor_names = [item.name for item in self.model_with_inputs_session.get_inputs()]
        for _, tensor_name in enumerate(self.input_shapes):
            utils.check_input_name_in_model(input_tensor_names, tensor_name)
        for input_item in self.model_with_inputs_session.get_inputs():
            tensor_name = input_item.name
            tensor_type = input_item.type
            tensor_shape = tuple(input_item.shape)
            # skip extend inputs add by custom op
            if tensor_name in self.extend_inputs_map.keys():
                continue

            if utils.check_dynamic_shape(tensor_shape):
                if not self.input_shapes:
                    logger.error(
                        f"The dynamic shape {tensor_shape} are not supported. "
                        f"Please set '--input_shape' to fix the dynamic shape.")
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)
            if self.input_shapes and tensor_name in self.input_shapes:
                input_shape = self.input_shapes.get(tensor_name)
                try:
                    number_shape = [int(dim) for dim in input_shape]
                except (ValueError, TypeError) as error:
                    logger.error(utils.get_shape_not_match_message(InputShapeError.FORMAT_NOT_MATCH, self.input_shape))
                    raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR) from error
                self._check_input_shape_fix_value(tensor_name, tensor_shape, number_shape)
                tensor_info = {"name": tensor_name, "shape": tuple(number_shape), "type": tensor_type}
                logger.info(f"Fix dynamic input shape of {tensor_name} to {number_shape}")
            else:
                tensor_info = {"name": tensor_name, "shape": tensor_shape, "type": tensor_type}
            inputs_tensor_info.append(tensor_info)
        logger.info(f"model inputs tensor info:\n{inputs_tensor_info}\n")
        return inputs_tensor_info

    def _get_inputs_data(self, inputs_tensor_info):
        names = [ii["name"] for ii in inputs_tensor_info]
        shapes = [ii["shape"] for ii in inputs_tensor_info]
        dtypes = [self._convert_to_numpy_type(ii["type"]) for ii in inputs_tensor_info]

        bin_file_path_array = []
        if "" == self.input_data:
            check_file_or_directory_path(os.path.realpath(self.data_dir), True)
            input_bin_files = os.listdir(self.data_dir)
            if len(input_bin_files) == 0:
                return self._generate_random_input_data(self.data_dir, names, shapes, dtypes)
            input_bin_files.sort(key=lambda file: int((re.findall("\\d+", file))[0]))
            bin_file_path_array = [os.path.join(self.data_dir, item) for item in input_bin_files]

        else:
            input_initial_path = self.input_data.split(",")
            for input_item in input_initial_path:
                input_item_path = os.path.realpath(input_item)
                if input_item_path.endswith('.bin'):
                    bin_file_path_array.append(input_item_path)
                else:
                    utils.get_input_path(input_item_path, bin_file_path_array)

        self._check_input_data_path(bin_file_path_array, inputs_tensor_info)
        return self._read_input_data(bin_file_path_array, names, shapes, dtypes)

    def _get_inputs_data_aipp(self, data_dir, inputs_tensor_info, npu_dump_data_path):
        inputs_map = {}
        aipp_input = []
        for bin_file in os.listdir(npu_dump_data_path):
            if bin_file.startswith("Aipp"):
                aipp_input.append(os.path.join(npu_dump_data_path, bin_file))
        if len(aipp_input) != len(inputs_tensor_info):
            logger.error("lengths of aipp_input and input_tensor_info unequal, please check.")
            raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INDEX_OUT_OF_BOUNDS_ERROR)
        for i, tensor_info in enumerate(inputs_tensor_info):
            convert_bin_file_to_npy(aipp_input[i], os.path.join(self.output_path, "input"), self.cann_path)
            aipp_output_path = (os.path.join(self.output_path, "input", aipp_input[i].rsplit("/", 1)[1]) +
                                ".output.0.npy")
            aipp_output_path = load_file_to_read_common_check(aipp_output_path)
            aipp_output = np.load(aipp_output_path)
            nchw_prod = np.prod(tensor_info["shape"])
            nchwc_prod_without_c1 = np.prod(aipp_output.shape[:-1])
            try:
                c0 = int(nchw_prod / nchwc_prod_without_c1)
            except ZeroDivisionError as e:
                logger.error(f"Aipp output has wrong shape, file path: {aipp_output_path}")
                raise utils.AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_DATA_ERROR) from e
            onnx_input = aipp_output[..., :c0].transpose((0, 4, 2, 3, 1)).squeeze(-1).astype(np.float32)
            inputs_map[tensor_info["name"]] = onnx_input
        return inputs_map

    def _convert_to_numpy_type(self, tensor_type):
        numpy_data_type = NODE_TYPE_TO_DTYPE_MAP.get(tensor_type)
        if numpy_data_type:
            return numpy_data_type
        else:
            logger.error(f"unsupported tensor type: {tensor_type}")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_TENSOR_TYPE_ERROR)

    def _run_model(self, session, inputs_map):
        outputs_name = [node.name for node in session.get_outputs()]
        return session.run(outputs_name, inputs_map)

    def _save_dump_data(self, dump_bins, old_onnx_model, net_output_node, input_map):
        res_idx = 0
        file_name_map = []
        for node in old_onnx_model.graph.node:
            #存储onnx的输入dump数据
            for i, node_input in enumerate(node.input):
                file_name = self._generate_dump_data_file_name("input_" + node.name, i)
                if len(file_name) > MAX_FILE_NAME_LEN:
                    new_file_name = str(round(time.time() * 1e6)) + str(len(file_name_map)) + ".npy"
                    file_name_map.append(f"{new_file_name},{file_name}\n")
                    file_name = new_file_name
                file_path = os.path.join(self.onnx_dump_data_dir, file_name)
                if input_map.get(node_input) is not None:
                    np.save(file_path, input_map.get(node_input))
            for j, output in enumerate(node.output):
                file_name = self._generate_dump_data_file_name(node.name, j)
                if len(file_name) > MAX_FILE_NAME_LEN:
                    new_file_name = str(round(time.time() * 1e6)) + str(len(file_name_map)) + ".npy"
                    file_name_map.append(f"{new_file_name},{file_name}\n")
                    file_name = new_file_name

                file_path = os.path.join(self.onnx_dump_data_dir, file_name)
                if output in net_output_node:
                    self.net_output[net_output_node.index(output)] = file_path
                if res_idx <= len(dump_bins) - 1:
                    np.save(file_path, dump_bins[res_idx])
                else:
                    logger.error("res_idx out of bounds of dump_bins and can not save, please check.")
                res_idx += 1

        if len(file_name_map) > 0:
            mapping_file_path = os.path.join(self.onnx_dump_data_dir, "mapping.csv")
            with ms_open(mapping_file_path, mode="w") as map_file:
                map_file.writelines(file_name_map)

        for key, value in self.net_output.items():
            logger.info(f"net_output node is:{key}, file path is {value}")
        logger.info("dump data success")
