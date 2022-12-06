import json

import numpy as np
import utils
from dump_data_pb2 import DumpData


class DumpTensor:
    """
    The class of DumpTensor, replace the class of DD.DumpData.input or output.
    Include the data detail: index, data_type, tensor_format, shape, data, size, original_shape
    """

    def __init__(self: any, index: int = None, data_type: int = None,
                 tensor_format: int = None, shape: list = None, data: np.ndarray = None,
                 size: int = None, original_shape: list = None, address: int = None, sub_format: int = 0) -> None:

        self.index = index
        self.data_type = data_type
        self.tensor_format = tensor_format
        self.shape = shape if shape else []
        self.data = data
        self.size = size
        self.original_shape = original_shape
        self.address = address
        self.sub_format = sub_format


class DumpDataObj:
    """
    The class of DumpDataObject, replace the class DD.DumpData.
    Include dump_file information
    """
    def __init__(self: any, dump_data: DumpData = DumpData()) -> None:
        self.version = dump_data.version
        self.op_name = dump_data.op_name
        self.dump_time = dump_data.dump_time
        self.buffer = dump_data.buffer
        self.attr = dump_data.attr
        self.input_data = [_input_data for _input_data in dump_data.input]
        self.output_data = [_output_data for _output_data in dump_data.output]
        self.ffts_info = {}

    @staticmethod
    def _build_dump_tensor(dump_data_object_data: list) -> None:
        """
        replace the input or output object of DD.DumpData to DumpyTensor
        @param dump_data_object_data: input or output object of DD.DumpData
        @return: None
        """
        for index, tensor in enumerate(dump_data_object_data):
            data_to_np = utils.deserialize_dump_data_to_array(tensor)
            dump_tensor = DumpTensor(index, tensor.data_type, tensor.format, list(tensor.shape.dim),
                                     data_to_np, tensor.size, list(tensor.original_shape.dim),
                                     tensor.address, tensor.sub_format)
            dump_data_object_data[index] = dump_tensor

    def build_input_dump_tensor(self: any) -> None:
        """
        Get input DumpTensor
        @return: None
        """
        self._build_dump_tensor(self.input_data)

    def build_output_dump_tensor(self: any) -> None:
        """
        Get output DumpTensor
        @return: None
        """
        self._build_dump_tensor(self.output_data)

    @property
    def get_output_data(self):
        """
        Get output data
        @return: list of output data
        """
        return [output.data for output in self.output_data]

    def parser_ffts_attr(self):
        if self.attr:
            data_attr = json.loads(self.attr[0].value)
            self.attr = data_attr

    @property
    def get_thread_num(self):
        return self.attr["slice_instance_num"]

    @property
    def get_cut_axis_manual(self):
        cut_axis = []
        for output in self.attr["outputCutList"]:
            _ = []
            for index, value in enumerate(output):
                if value != 1:
                    _.append(index)
            cut_axis.append(_)
        return cut_axis

    def get_cut_axis_auto(self):
        pass

    def calculate_auto_mode_shape(self):
        output_shape = []
        for output in self.attr["output_tensor_slice"]:
            _ = []



    @property
    def get_ffts_mode(self):
        return self.attr["threadMode"]
#
#
# class ParserFfts:
#     def __init__(self: any, dump_file_list, dump_data_list):
#         self.dump_file_list = dump_file_list
#         self.dump_data_list = dump_data_list
#
#     def get_base_info(self):
#         base_dump_file_path = self.dump_file_list[0]
#         base_dump_data = self.dump_data_list[0]
