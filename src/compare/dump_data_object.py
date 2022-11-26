import numpy as np
import utils
from dump_data_pb2 import DumpData


class DumpTensor:
    """
    The class of DumpTensor, replace the class of DD.DumpData.input or output.
    Include the data detail: index, data_tyoe, tensor_foramt, shape, data, size, orginal_shape
    """
    def __init__(self: any, index: int = None, data_type: int = None,
                 tensor_format: int = None, shape: list = None, data: np.ndarray = None,
                 size: int = None, original_shape: list = None) -> None:
        self.index = index
        self.data_type = data_type
        self.tensor_format = tensor_format
        self.shape = shape if shape else []
        self.data = data
        self.size = size
        self.original_shape = original_shape


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
                                     data_to_np, tensor.size, list(tensor.original_shape.dim))
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
