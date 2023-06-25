
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
"""
Function:
This file is used to parse the HDF5 file format.
"""
import re
from enum import Enum
import collections

from cmp_utils import utils, utils_type, path_check
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError


class DataSetType(Enum):
    """
    The enum for pytorch dump data type
    """
    TENSOR = 0
    VEC_TENSOR = 1
    VEC_I64 = 2
    I64 = 3
    BOOL = 4
    DOUBLE = 5
    SCALER = 6
    TYPE_AND_SIZED = 7
    OPT_INT64 = 8
    OPT_SCALER = 9
    VEC_VEC_I64 = 10
    GEOMETRY = 11
    SIZE = 12
    SCALER_TYPE = 13


def _open_h5py_file(file_path: str, model: str = 'r') -> any:
    """
    Open the HDF5 file
    """
    import h5py

    return h5py.File(file_path, model)


class Hdf5Parser:
    """
    The class for HDF5 file parse
    """
    MAX_OP_NUM = 1000000
    DATASET_PATH_PATTERN = r"(/[a-zA-Z0-9_]*)(/[0-9]*)(/[a-z]*)"
    INPUT = "/input"
    GOLDEN_DUMP_FILE = "GOLDEN_DUMP"
    MY_DUMP_FILE = "MY_DUMP"

    def __init__(self: any, file_path: str, dump_file_type: str, mapping_list: list) -> None:
        self.file_handle = None
        self.dump_file_type = dump_file_type
        self.file_path = file_path
        self.order_ext_opname_map = collections.defaultdict(list)
        self.ext_opname_dataset_map = collections.defaultdict(list)
        self.device_type = None
        self.need_compare_input = False
        self._mapping_list = mapping_list

    @staticmethod
    def _is_multimap(op_name: str, mapping_set: list) -> bool:
        if op_name in mapping_set:
            return True
        return False

    def open_file(self: any, model: str) -> int:
        """
        Open file and check exception
        """
        try:
            self.file_handle = _open_h5py_file(self.file_path, model)
            return CompareError.MSACCUCMP_NONE_ERROR
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError):
            self.file_handle = None
            log.print_error_log("Open {} failed!".format(self.file_path))
            return CompareError.MSACCUCMP_OPEN_FILE_ERROR
        finally:
            pass

    def close_file(self: any) -> None:
        """
        Close the file by file_handle
        """
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                    MemoryError):
                log.print_error_log("Close {} failed!".format(self.file_path))
                return
            self.file_handle = None

    def get_dump_data_attr(self: any, dataset_path: str, attr_type: str) -> any:
        """
        Get the dump data attribute by dataset_path
        :dataset_path: the path of dataset
        :attr_type:
        """
        if self.file_handle is None:
            log.print_error_log("the handle of {} is invalid!".format(self.file_path))
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        try:
            attrs = self.file_handle[dataset_path].attrs
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as err:
            log.print_error_log("Read dataset attr:{} failed!"
                                .format(dataset_path))
            raise CompareError(CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR) from err
        if attrs:
            return True, attrs.get(attr_type)
        return False, ''

    def get_dump_data(self: any, dataset_path: str) -> any:
        """
        Get the dump data by dataset_path
        :dataset_path: the path of dataset
        """
        if self.file_handle is None:
            log.print_error_log("the handle of {} is invalid!".format(self.file_path))
            raise CompareError(CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)
        try:
            if self._dataset_path_valid(dataset_path):
                return self.file_handle[dataset_path][()]
            return []
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError) as err:
            log.print_error_log("Read dataset:{} failed!"
                                .format(dataset_path))
            raise CompareError(CompareError.MSACCUCMP_PARSE_DUMP_FILE_ERROR) from err
        finally:
            pass

    def get_all_orders(self: any) -> list:
        """
        Get all orders string used to compare dump data.
        """
        return list(self.order_ext_opname_map.keys())

    def get_order_by_ext_opname(self: any, ext_opname: str) -> int:
        """
        Get the order corresponding to the ext_opname.
        """
        for order, ext_opname_group in self.order_ext_opname_map.items():
            if ext_opname in ext_opname_group:
                return order
        # Unmatched items. To display them at the end of the result file after sorting,
        # order must be the maximum value plus 1.
        if self.get_all_orders():
            return max(self.get_all_orders()) + 1
        return 0

    def get_ext_opname_group_by_order(self: any, order: int) -> list:
        """
        Get the list of op names corresponding to the order.
        """
        return self.order_ext_opname_map.get(order, [])

    def parse_dump_file(self: any) -> int:
        """
        Entry for parsing dump files.
        """
        ret = self.open_file('r')
        if ret == CompareError.MSACCUCMP_NONE_ERROR:
            self._generate_order_ext_opname_map()
            self._parse_all_dataset()
        return ret

    def have_dataset(self: any, ext_opname: str, dataset_path: str) -> bool:
        """
        Check whether the dataset path is included.
        :ext_opname: extended op name
        :dataset_path: dataset path in dump file
        """
        if dataset_path in self.ext_opname_dataset_map.get(ext_opname, []):
            return True
        op_name, _ = ext_opname.split(ConstManager.DELIMITER, 1)
        # get 4 from '/AddmmBackward/4/input/mat1'

        _, _, order, _ = dataset_path.split('/', 3)
        all_ext_opname = self.get_ext_opname_group_by_order(int(order))
        for item in all_ext_opname:
            if item.startswith("{}:".format(op_name)) and \
                    dataset_path in self.ext_opname_dataset_map.get(item, []):
                return True
        return False

    def file_is_empty(self: any) -> bool:
        """
        Check whether the file is empty.
        """
        return self.file_handle is None \
               or self.file_handle.keys()

    def is_load_mode(self: any) -> bool:
        """
        Check whether the load comparison mode.
        """
        return not self.need_compare_input

    def _check_value(self: any, op_order_list: list) -> None:
        if len(op_order_list) > self.MAX_OP_NUM:
            log.print_error_log("The number of ops in {} exceeds the range!"
                                .format(self.file_path))
            raise CompareError(CompareError.MSACCUCMP_INDEX_OUT_OF_BOUNDS_ERROR)

    def _get_mapping_set(self: any, op_name: str) -> list:
        for mapping_set in self._mapping_list:
            if op_name in mapping_set and len(mapping_set) > 1:
                return mapping_set
        return []

    def _is_parsed(self: any, op_name: str) -> bool:
        for ext_op_name_list in self.order_ext_opname_map.values():
            for ext_op_name in ext_op_name_list:
                if ext_op_name.startswith(op_name):
                    return True
        return False

    def _gen_ext_opname_map_special(self: any, op_name: str, multimap_set: list) -> dict:
        order_ext_opname_map = collections.defaultdict(list)
        if self._is_parsed(op_name):
            return order_ext_opname_map
        op_order_list = []
        for op in multimap_set:
            if op in self.file_handle.keys():
                op_order_list.extend(list(map(int, self.file_handle[op].keys())))
        op_order_list.sort()
        for index, op_order in enumerate(op_order_list):
            for op in multimap_set:
                if op in self.file_handle.keys() and str(op_order) in self.file_handle[op].keys():
                    order_ext_opname_map[op_order] = ["{}:{}".format(op, index)]
        return order_ext_opname_map

    def _gen_single_order_ext_opname_map(self: any, op_name: str) -> dict:
        """
        Get the map of order to extended op name.
        """
        order_ext_opname_map = collections.defaultdict(list)
        mapping_set = self._get_mapping_set(op_name)
        if self.file_handle is None:
            return order_ext_opname_map
        try:
            if self._is_multimap(op_name, mapping_set):
                return self._gen_ext_opname_map_special(op_name, mapping_set)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError):
            log.print_error_log("construct order_ext_map by {} failed:!"
                                .format(op_name))
            return order_ext_opname_map

        # sort by the order of the op executed multi times
        op_order_list = list(map(int, self.file_handle[op_name].keys()))
        op_order_list.sort()
        self._check_value(op_order_list)
        for index, op_order in enumerate(op_order_list):
            order_ext_opname_map[op_order] = ["{}:{}".format(op_name, index)]
        return order_ext_opname_map

    def _generate_order_ext_opname_map(self: any) -> None:
        """
        Generate the mapping between order and extended operator names in dump data.
        """
        if self.file_handle is None:
            return
        for op_name in self.file_handle.keys():
            order_op_map = self._gen_single_order_ext_opname_map(op_name)
            utils.merge_dict(self.order_ext_opname_map, order_op_map)

    def _dataset_path_valid(self: any, dataset_path: str) -> bool:
        match_result = re.search(self.DATASET_PATH_PATTERN, dataset_path)
        if match_result is not None:
            return True
        return False

    def _data_path_is_input(self: any, curr_group_path: str) -> bool:
        """
        Check whether the data path contains the input flag.
        :curr_group_path: the path of of group or dataset
        """
        match_result = re.search(self.DATASET_PATH_PATTERN, curr_group_path)
        if match_result is not None and match_result.group(3) == self.INPUT:
            return True
        return False

    def _parse_one_dataset(self: any, data_type: int, ext_opname: str, curr_group_path: str) -> None:
        # compare Tensor data only.
        if data_type != DataSetType.TENSOR.value:
            return
        self.ext_opname_dataset_map[ext_opname].append(curr_group_path)
        if self.device_type and self.device_type != utils_type.DeviceType.CPU.value:
            return
        attr_ok, device_type = self.get_dump_data_attr(curr_group_path,
                                                       utils_type.DatasetAttr.DeviceType.name)
        if attr_ok:
            self.device_type = device_type

    def _need_compare(self: any, curr_group_path: str) -> bool:
        if not self._data_path_is_input(curr_group_path):
            return True
        if self.dump_file_type == self.MY_DUMP_FILE:
            self.need_compare_input = True
            return True
        if self.dump_file_type == self.GOLDEN_DUMP_FILE and not self.need_compare_input:
            return False
        return True

    def _parse_dataset_recursively(self: any, ext_opname: str, curr_group_path: str) -> None:
        """
        Get the data_set path of one op recursively
        :ext_opname: example. Adamm_1, 1 is the execution order of the op Adamm in the model.
        :curr_group_path: current hierarchical path found recursively
        """
        if self.file_handle is None:
            self.ext_opname_dataset_map.clear()

        if not self._need_compare(curr_group_path):
            return
        # only dataset have the attribute 'Type'. The group dose not.
        have_attr, data_type = self.get_dump_data_attr(curr_group_path, utils_type.DatasetAttr.Type.name)
        if have_attr:
            self._parse_one_dataset(data_type, ext_opname, curr_group_path)
            return
        try:
            # find all the datasets recursively by path
            if self.file_handle[curr_group_path].keys():
                for next_group_name in self.file_handle[curr_group_path].keys():
                    next_group_path = "{}/{}".format(curr_group_path, next_group_name)
                    self._parse_dataset_recursively(ext_opname, next_group_path)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError,
                MemoryError, KeyError, IOError):
            log.print_error_log("parse_dataset failed, group path is:{}!"
                                .format(curr_group_path))
        finally:
            pass

    def _parse_all_dataset(self: any) -> None:
        """
        Parse the hdf5 dump file.
        Obtain all dataset paths from this file.
        """
        if self.file_handle is None:
            return

        for order in self.order_ext_opname_map.keys():
            ext_opname_group = self.order_ext_opname_map.get(order, [])
            for ext_opname in ext_opname_group:
                op_name, _ = ext_opname.split(ConstManager.DELIMITER, 1)
                group_path = "/{}/{}".format(op_name, order)
                self._parse_dataset_recursively(ext_opname, group_path)
