
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
Pytorch CompareData class. This class mainly involves the function of parse dump_data.
"""
import numpy as np

from cmp_utils import utils_type
from cmp_utils import log
from pytorch_cmp import hdf5_parser
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError


class DataType:
    """
    The class for pytorch dump data DataType
    """
    data_type = {
        "Float": 1,
        "Byte": 2,
        "Char": 3,
        "Short": 5,
        "Int": 6,
        "Long": 7,
        "Bool": 9,
        "Half": 10,
        "Double": 11
    }
    equivalent_type = [
        ("Int", "Long"),
        ("Float", "Half")
    ]

    @classmethod
    def get_name(cls: any, value: int) -> str:
        """
        Get the type name by the type value.
        :value: the code of data type
        """
        return list(DataType.data_type.keys())[list(DataType.data_type.values()).index(value)]

    @classmethod
    def get_value(cls: any, name: str) -> int:
        """
        Get the type value by the type name.
        :name: the name of data type
        """
        return DataType.data_type.get(name)


class CompareMap:
    """
    The class is used for op mapping.
    """
    def __init__(self: any) -> None:
        self.op_map = {}
        self.param_map = {}
        self._init_map_table()

    def get_mapping_opname(self: any, opname: str) -> list:
        """
        Get mapping op name in map table.
        :opname: the op name before mapping
        """
        if opname in list(self.op_map.keys()):
            return self.op_map.get(opname)
        return []

    def get_mapping_param(self: any, opname: str, param: str) -> str:
        """
        Get mapping param in map table.
        :opname: the param name before mapping
        """
        if opname not in list(self.param_map.keys()):
            return ''
        item = self.param_map.get(opname)
        if param in item.keys():
            return item[param]
        return ''

    def get_mapping_opname_all(self: any) -> list:
        """
        Get all the mapped opname.
        """
        return list(self.op_map.values())

    def _init_map_table(self: any) -> None:
        self.op_map = {
            'CudnnBatchNormBackward': ['NativeBatchNormBackward'],
            'NativeBatchNormBackward': ['CudnnBatchNormBackward'],
            'CudnnConvolutionBackward': ['NpuConvolutionBackward'],
            'ThnnConvDepthwise2DBackward': ['NpuConvolutionBackward'],
            'NpuConvolutionBackward': ['CudnnConvolutionBackward', 'ThnnConvDepthwise2DBackward']
        }

        self.param_map = {
            'CudnnBatchNormBackward': {'epsilon': 'eps'},
            'NativeBatchNormBackward': {'eps': 'epsilon'},
            'CudnnConvolutionBackward': {'self': 'input'},
            'ThnnConvDepthwise2DBackward': {'self': 'input'},
            'NpuConvolutionBackward': {'input': 'self'}
        }


class CompareData:
    """
    The class for compare data, left dump data and right dump data.
    """
    NPU_PREFIX = "Npu"
    GPU_PREFIX = "Cudnn"
    CPU_PREFIX = "Thnn"

    def __init__(self: any, my_dump_path: str, golden_dump_path: str) -> None:
        self.mapping = CompareMap()
        self.my_dump = hdf5_parser.Hdf5Parser(my_dump_path, hdf5_parser.Hdf5Parser.MY_DUMP_FILE,
                                              self.mapping.get_mapping_opname_all())
        self.golden_dump = hdf5_parser.Hdf5Parser(golden_dump_path, hdf5_parser.Hdf5Parser.GOLDEN_DUMP_FILE,
                                                  self.mapping.get_mapping_opname_all())
        self.orders_num = 0

    @staticmethod
    def get_original_opname(ext_opname: str) -> str:
        """
        Get original op name in ext_opname.
        :ext_opname: extend op name. such as cov2d:2
        """
        op_name, _ = ext_opname.split(ConstManager.DELIMITER, 1)
        return op_name

    @staticmethod
    def _is_equivalent_type(my_dump_data_type: int, golden_dump_data_type: int, type_info: str) -> bool:
        equivalent_type = set()
        for item in DataType.equivalent_type:
            equivalent_type.clear()
            for type_name in item:
                equivalent_type.add(DataType.get_value(type_name))
            if my_dump_data_type in equivalent_type \
                    and golden_dump_data_type in equivalent_type:
                message = 'The DataType on both sides are Compatible, {}'.format(type_info)
                log.print_info_log(message)
                return True
        return False

    @staticmethod
    def _check_stride(dataset_path: str, shape: list, stride: list) -> bool:
        message = "shape={},stride={},stride is invalid". \
            format(tuple(shape), tuple(stride))

        if len(shape) != len(stride):
            log.print_warn_log("[{}]:{}".format(dataset_path, message))
            return False

        expect_data_num = 0
        real_data_num = 1
        for (index, _) in enumerate(shape):
            expect_data_num += ((shape[index] - 1) * stride[index])
            real_data_num *= shape[index]
        expect_data_num += 1
        if expect_data_num > real_data_num:
            log.print_warn_log("[{}]:{}".format(dataset_path, message))
            return False
        return True

    def get_golden_dataset(self: any, ext_opname: str, my_dump_dataset_path: str) -> (bool, str, str):
        """
        Get the golden_dataset_path that matches the my_dump_dataset_path.
        :ext_opname: the extend op name. such as cov2d:2
        :my_dump_dataset_path: the my dump dataset path that get from dump data
        """
        message = 'No data match with {} in golden dump data.' \
            .format(my_dump_dataset_path)
        if ext_opname in self.golden_dump.ext_opname_dataset_map.keys():
            golden_ext_opname = ext_opname
        else:
            golden_ext_opname = self._opname_map(ext_opname, self.golden_dump.device_type)
            if golden_ext_opname not in self.golden_dump.ext_opname_dataset_map.keys():
                return False, '', message

        golden_dataset_path = self._construct_dataset_path(
            ext_opname, my_dump_dataset_path, self.golden_dump, golden_ext_opname)
        if golden_dataset_path and self.golden_dump.have_dataset(golden_ext_opname, golden_dataset_path):
            return True, golden_dataset_path, ''
        log.print_warn_log(message)
        return False, '', message

    def get_dump_data(self: any, my_dataset_path: str, golden_dataset_path: str) -> (any, any, str):
        """
        Get my dump data and golden dump data.
        :my_dataset_path: my dataset path in my dump data
        :golden_dataset_path: golden dataset path in golden dump data
        """
        success, message = self._check_data_type(my_dataset_path, golden_dataset_path)
        if not success:
            log.print_warn_log(message)
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, message)

        my_dump_data = self.my_dump.get_dump_data(my_dataset_path)
        golden_dump_data = self.golden_dump.get_dump_data(golden_dataset_path)
        converted_golden_dump_data = self._converted_stride(
            golden_dump_data, golden_dataset_path)
        return my_dump_data, converted_golden_dump_data, ''

    def get_not_matched_golden_datasets(self: any) -> list:
        """
        Find all the dataset that does not match with my dump in the golden file.
        """
        not_matched_info = []
        # Unmatched items. To display them at the end of the result file after sorting,
        # order must be the maximum value plus 1.
        for golden_ext_opname, _ in self.golden_dump.ext_opname_dataset_map.items():
            all_not_matched = True
            failed_info = []
            golden_dataset = ''
            for golden_dataset in self.golden_dump.ext_opname_dataset_map[golden_ext_opname]:
                if golden_ext_opname in self.my_dump.ext_opname_dataset_map.keys():
                    my_dump_ext_opname = golden_ext_opname
                else:
                    my_dump_ext_opname = self._opname_map(golden_ext_opname)
                if not self._reverse_match_non_load_mode(my_dump_ext_opname, golden_dataset, failed_info):
                    continue
                if not self._reverse_match_process(my_dump_ext_opname, golden_ext_opname,
                                                   golden_dataset, failed_info):
                    continue
                all_not_matched = False

            if all_not_matched:
                failed_info.clear()
                _, op_name, order, _ = golden_dataset.split('/', 3)
                message = 'No data match with /{}/{} in my dump data.'.format(op_name, order)
                failed_info.append([max(self.my_dump.get_all_orders()) + 1, 'NaN',
                                    '/{}/{}'.format(op_name, order), message])
            not_matched_info.extend(failed_info)

        return not_matched_info

    def set_compare_input_flag(self: any) -> None:
        """
        Sets whether to compare inputs.
        """
        if self.my_dump.need_compare_input:
            self.golden_dump.need_compare_input = True

    def check_my_dump_file_valid(self: any) -> None:
        """
        Check whether the device type is NPU.
        """
        if not self.my_dump.ext_opname_dataset_map:
            log.print_warn_log('my dump file is empty!')
            raise CompareError(
                CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

        if self.my_dump.device_type != utils_type.DeviceType.NPU.value:
            log.print_error_log('My dump file is not the dump data of the model'
                                ' executed on the AI processor, please check -m param!')
            raise CompareError(
                CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def parse_dump_file(self: any) -> int:
        """
        Parsing the data to be compared.
        """
        ret = self.my_dump.parse_dump_file()
        if ret == CompareError.MSACCUCMP_NONE_ERROR:
            self.check_my_dump_file_valid()
            self.set_compare_input_flag()
            self.orders_num = len(self.my_dump.get_all_orders())
            return self.golden_dump.parse_dump_file()
        return ret

    def open_file(self: any, model: str) -> None:
        """
        Open the my dump and golden dump files.
        """
        self.my_dump.open_file(model)
        self.golden_dump.open_file(model)

    def close_file(self: any) -> None:
        """
        Close the my dump and golden dump files.
        """
        self.my_dump.close_file()
        self.golden_dump.close_file()

    def get_all_orders(self: any) -> list:
        """
        Get all orders incrementally.
        """
        return sorted(self.my_dump.get_all_orders())

    def get_ext_opname_by_order(self: any, order: int) -> list:
        """
        Get the list of extend op name corresponding to the order.
        :order: the order of Op execution
        """
        return self.my_dump.get_ext_opname_group_by_order(order)

    def get_my_dump_datasets(self: any, ext_opname: str) -> list:
        """
        Get the dump datasets(include input and output) by ext_opname.
        :ext_opname: the extend op name. such as cov2d:2
        """
        if ext_opname in self.my_dump.ext_opname_dataset_map.keys():
            return self.my_dump.ext_opname_dataset_map[ext_opname]
        return []

    def _construct_dataset_path(self: any, src_ext_opname: str, src_dataset_path: str, dst_dump_data: any,
                                dst_ext_opname: str) -> str:
        """
        Construct the dataset path based on the peer path.
        :ext_opname: the extend op name. such as cov2d:2
        :dataset_path: the base dataset_path
        :gen_golden_path: if the Dataset is the expect type(golden)
        """
        dst_order = dst_dump_data.get_order_by_ext_opname(dst_ext_opname)

        # e.g. dataset_path:/AddmmBackward/4/input/mat1,dataset_name: input/mat1
        _, _, src_order, direction, src_dataset_name = src_dataset_path.split('/', 4)

        # in load comparison mode, inputs do not need to be compared, and
        # the dataset paths of my dump and golden dump must be the same order.
        if dst_dump_data.is_load_mode():
            dst_order = src_order

        dst_opname = self.get_original_opname(dst_ext_opname)

        dst_dataset_path = "/{}/{}/{}/{}".format(dst_opname, dst_order, direction, src_dataset_name)
        if dst_dump_data.have_dataset(dst_ext_opname, dst_dataset_path):
            return dst_dataset_path

        src_opname = self.get_original_opname(src_ext_opname)
        # get param name at suffix
        src_param_name = src_dataset_path.split('/')[-1]
        mapping_param_name = self.mapping.get_mapping_param(src_opname, src_param_name)
        if mapping_param_name:
            # replace param name at suffix
            return "{}{}".format(dst_dataset_path[:-len(src_param_name)], mapping_param_name)

        return ''

    def _opname_map_by_map_table(self:any, ext_opname: str, device_type: int) -> str:
        replaced_ext_opname = ''
        opname = self.get_original_opname(ext_opname)
        mapping_opname_list = self.mapping.get_mapping_opname(opname)
        if not mapping_opname_list:
            return replaced_ext_opname
        for mapping_opname in mapping_opname_list:
            replaced_ext_opname = ext_opname.replace(opname, mapping_opname, 1)
            if device_type == utils_type.DeviceType.NPU.value:
                if replaced_ext_opname in self.my_dump.ext_opname_dataset_map.keys():
                    return replaced_ext_opname
            else:
                if replaced_ext_opname in self.golden_dump.ext_opname_dataset_map.keys():
                    return replaced_ext_opname
        return ''

    def _opname_map(self: any, ext_opname: str, device_type: int = utils_type.DeviceType.NPU.value) -> str:
        """
        Processes mappings between NPUs, GPNs, and CPU operators.
        :ext_opname: the extend op name. such as cov2d:2
        :device_type: the device_type at my dump side
        """
        mapped_opname = self._opname_map_by_map_table(ext_opname, device_type)
        if mapped_opname:
            return mapped_opname

        if ext_opname.startswith(self.NPU_PREFIX):
            if device_type == utils_type.DeviceType.CPU.value:
                return ext_opname.replace(self.NPU_PREFIX, self.CPU_PREFIX, 1)
            if device_type == utils_type.DeviceType.GPU.value:
                return ext_opname.replace(self.NPU_PREFIX, self.GPU_PREFIX, 1)
        # reverse lookup
        if ext_opname.startswith(self.CPU_PREFIX):
            return ext_opname.replace(self.CPU_PREFIX, self.NPU_PREFIX, 1)
        if ext_opname.startswith(self.GPU_PREFIX):
            return ext_opname.replace(self.GPU_PREFIX, self.NPU_PREFIX, 1)
        return ext_opname

    def _check_data_type(self: any, my_dataset_path: str, golden_dataset_path: str) -> (bool, str):
        """
        Check the DataType of my dump data and golden dump data.
        :my_dataset_path: the my dump dataset path
        :golden_dataset_path: the golden dump dataset path
        """
        message = "my_dataset_path is {}, golden_dataset_path" \
                  " is {}.".format(my_dataset_path, golden_dataset_path)
        attr_ok, my_dump_data_type = self.my_dump.get_dump_data_attr(
            my_dataset_path, utils_type.DatasetAttr.DataType.name)
        if not attr_ok:
            return False, "Get the attr 'DataType' of {} failed! {}".format("my_dump", message)

        attr_ok, golden_dump_data_type = self.golden_dump.get_dump_data_attr(
            golden_dataset_path, utils_type.DatasetAttr.DataType.name)
        if not attr_ok:
            return False, "Get the attr 'DataType' of {} failed! {}".format("golden_dump", message)

        type_info = 'my dump data path is {}, the data type of both sides is({},{}).' \
            .format(my_dataset_path, DataType.get_name(my_dump_data_type),
                    DataType.get_name(golden_dump_data_type))
        if my_dump_data_type == golden_dump_data_type:
            return True, ''

        if self._is_equivalent_type(my_dump_data_type, golden_dump_data_type, type_info):
            return True, ''

        message = 'The DataType on both sides are different, {}'.format(type_info)
        return False, message

    def _converted_stride(self: any, dump_data: any, dataset_path: str) -> any:
        # convert GPU/CPU stride
        attr_ok, device_type = self.golden_dump.get_dump_data_attr(dataset_path,
                                                                   utils_type.DatasetAttr.DeviceType.name)
        if attr_ok and device_type in (utils_type.DeviceType.GPU.value, utils_type.DeviceType.CPU.value):
            have_stride_attr, stride_attr = self.golden_dump.get_dump_data_attr(dataset_path,
                                                                                utils_type.DatasetAttr.Stride.name)
            if have_stride_attr and self._check_stride(dataset_path, list(dump_data.shape), list(stride_attr)):
                dump_data_flatten = dump_data.flatten()
                real_stride = (dump_data_flatten.strides[0] * i for i in list(stride_attr))
                return np.lib.stride_tricks.as_strided(
                    dump_data_flatten, shape=dump_data.shape, strides=real_stride)
        return dump_data

    def _reverse_match_non_load_mode(self: any, my_dump_ext_opname: str, golden_dataset: str,
                                     failed_info: list) -> bool:
        message = 'No data match with {} in my dump data.'.format(golden_dataset)
        if not self.my_dump.is_load_mode() \
                and my_dump_ext_opname not in self.my_dump.ext_opname_dataset_map.keys():
            failed_info.append([max(self.my_dump.get_all_orders()) + 1, 'NaN', golden_dataset, message])
            return False
        return True

    def _reverse_match_process(self: any, my_dump_ext_opname: str, golden_ext_opname: str,
                               golden_dataset: str, failed_info: list) -> bool:
        my_dump_dataset_path = self._construct_dataset_path(
            golden_ext_opname, golden_dataset, self.my_dump, my_dump_ext_opname)
        if not my_dump_dataset_path or not self.my_dump.have_dataset(my_dump_ext_opname, my_dump_dataset_path):
            order = self.my_dump.get_order_by_ext_opname(my_dump_ext_opname)
            op_name = self.get_original_opname(my_dump_ext_opname)
            message = 'No data match with {} in my dump data.'.format(golden_dataset)
            failed_info.append([order, op_name, golden_dataset, message])
            return False
        return True
