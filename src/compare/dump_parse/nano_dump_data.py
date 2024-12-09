# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the nano dump function.
"""
import os
import struct
import warnings
from typing.io import BinaryIO
from enum import Enum

import dump_data_pb2 as DD

from cmp_utils import path_check
from cmp_utils import log
from cmp_utils.constant.const_manager import ConstManager
from cmp_utils.constant.compare_error import CompareError
from cmp_utils.tlv_parse import TLV


NANO_DUMP_DATA_MAGIC_NUM = 0x5a5a5a5a
TLV_TYPE_INPUT_DESC = 4
TLV_TYPE_OUTPUT_DESC = 5

INPUT_DESC_TLV_TYPE_SHAPE_DIMS = 0
INPUT_DESC_TLV_TYPE_ORI_SHAPE_DIMS = 1

OUTPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS = 0
OUTPUT_DESC_L3_TLV_TYPE_ORI_SHAPE_DIMS = 1
OUTPUT_DESC_L3_TLV_TYPE_ORI_NAME = 2

INPUT_INFO_CONFIG_TAB = [
    {'Name': 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'format', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'address_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'address', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT64'},
    {'Name': 'offset', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT64'},
    {'Name': 'size', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT64'},
    {'Name': 'tlv_list_len', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
    {'Name': 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64',
     'Tag': INPUT_DESC_TLV_TYPE_SHAPE_DIMS
     },
    {'Name': 'original_shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64',
     'Tag': INPUT_DESC_TLV_TYPE_ORI_SHAPE_DIMS
     }
]

INPUTS_LIST_CONFIG_TAB = [
    {'Name': 'inputs_num', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
    {'Name': 'inputs', 'TLV_Type': 'NV', 'Ele_Type': INPUT_INFO_CONFIG_TAB, 'N':'inputs_num'}
]

INPUTS_CONFIG_TAB = [
    {'Name': 'input_desc', 'TLV_Type': 'TLV', 'Ele_Type': INPUTS_LIST_CONFIG_TAB, 'Tag': TLV_TYPE_INPUT_DESC}
]

OUTPUT_INFO_CONFIG_TAB = [
    {'Name': 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'format', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'address_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'original_index', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'original_data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'original_format', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'address', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT64'},
    {'Name': 'offset', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT64'},
    {'Name': 'size', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT64'},
    {'Name': 'tlv_list_len', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
    {'Name': 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64',
     'Tag': OUTPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS
     },
    {'Name': 'original_shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64',
     'Tag': OUTPUT_DESC_L3_TLV_TYPE_ORI_SHAPE_DIMS
     },
    {'Name': 'origin_name', 'TLV_Type': 'TLNV', 'Ele_Type': 'CHAR',
     'Tag': OUTPUT_DESC_L3_TLV_TYPE_ORI_NAME
     }
]

OUTPUTS_LIST_CONFIG_TAB = [
    {'Name': 'outputs_num', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
    {'Name': 'outputs', 'TLV_Type': 'NV', 'Ele_Type': OUTPUT_INFO_CONFIG_TAB, 'N':'outputs_num'}
]

OUTPUTS_CONFIG_TAB = [
    {'Name': 'output_desc', 'TLV_Type': 'TLV', 'Ele_Type': OUTPUTS_LIST_CONFIG_TAB, 'Tag': TLV_TYPE_OUTPUT_DESC}
]


class NanoDataType(Enum):
    DT_FLOAT = 0
    DT_FLOAT16 = 1
    DT_INT8 = 2
    DT_INT16 = 6
    DT_UINT16 = 7
    DT_UINT8 = 4
    DT_INT32 = 3
    DT_INT64 = 9
    DT_UINT32 = 8
    DT_UINT64 = 10
    DT_BOOL = 12
    DT_DOUBLE = 11
    DT_STRING = 13
    DT_DUAL_SUB_INT8 = 14
    DT_DUAL_SUB_UINT8 = 15
    DT_COMPLEX64 = 16
    DT_COMPLEX128 = 17
    DT_QINT8 = 18
    DT_QINT16 = 19
    DT_QINT32 = 20
    DT_QUINT8 = 21
    DT_QUINT16 = 22,
    DT_RESOURCE = 23
    DT_STRING_REF = 24
    DT_DUAL = 25
    DT_VARIANT = 26
    DT_BF16 = 27
    DT_UNDEFINED = 28
    DT_INT4 = 29
    DT_UINT1 = 30
    DT_INT2 = 31
    DT_UINT2 = 32
    DT_MAX = 33


NANO_DATA_TYPE_TO_PROTO_DATA_TYPE = {
    NanoDataType.DT_FLOAT.value: DD.DT_FLOAT,
    NanoDataType.DT_FLOAT16.value: DD.DT_FLOAT16,
    NanoDataType.DT_DOUBLE.value: DD.DT_DOUBLE,
    NanoDataType.DT_INT8.value: DD.DT_INT8,
    NanoDataType.DT_INT16.value: DD.DT_INT16,
    NanoDataType.DT_INT32.value: DD.DT_INT32,
    NanoDataType.DT_INT64.value: DD.DT_INT64,
    NanoDataType.DT_UINT8.value: DD.DT_UINT8,
    NanoDataType.DT_UINT16.value: DD.DT_UINT16,
    NanoDataType.DT_UINT32.value: DD.DT_UINT32,
    NanoDataType.DT_UINT64.value: DD.DT_UINT64,
    NanoDataType.DT_BOOL.value: DD.DT_BOOL,
    NanoDataType.DT_COMPLEX64.value: DD.DT_COMPLEX64,
    NanoDataType.DT_COMPLEX128.value: DD.DT_COMPLEX128,
    NanoDataType.DT_BF16.value: DD.DT_BF16,
    NanoDataType.DT_UINT1.value: DD.DT_UINT1,
    NanoDataType.DT_RESOURCE.value: DD.DT_INT64,
}


class EmptyObj:
    pass


class NanoDumpData:
    def __init__(self, op_type="", op_name="", dump_time=0) -> None:
        self.magic_num = None
        self.version_id = 0
        self.op_type = op_type
        self.op_name = op_name
        self.dump_time = dump_time
        self.inputs = []
        self.outputs = []

    @staticmethod
    def _unpack_value(aux: bytes, value_type: str) -> (bytes, any):
        type_para = ConstManager.UNPACK_FORMAT.get(value_type)
        _fmt, _size = type_para.get('FMT'), type_para.get('SIZE')

        if len(aux) < _size:
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)

        value = struct.unpack(_fmt, aux[:_size])[0]
        return aux[_size:], value

    def parse_tlv_head(self, content: bytes) -> None:
        aux = content

        # magic_num
        aux, self.magic_num = self._parse_uint32(aux)
        if self.magic_num != NANO_DUMP_DATA_MAGIC_NUM:
            # parse data from bin file stream
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)

        # version_id
        aux, self.version_id = self._parse_uint32(aux)

        # inputs
        intputs_desc_obj = EmptyObj()
        tlv = TLV(INPUTS_CONFIG_TAB)
        try:
            aux, nested_obj = tlv.parse_tlv_by_cfg_tab(aux, intputs_desc_obj)
        except CompareError as error:
            log.print_error_log('Failed to decode inputs info %s', error)
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from error

        if nested_obj.input_desc is not None:
            self.inputs = nested_obj.input_desc.inputs

        # outputs
        outputs_desc_obj = EmptyObj()
        tlv = TLV(OUTPUTS_CONFIG_TAB)
        try:
            aux, nested_obj = tlv.parse_tlv_by_cfg_tab(aux, outputs_desc_obj)
        except CompareError as error:
            log.print_error_log('Failed to decode outputs info %s' % error)
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from error

        if nested_obj.output_desc is not None:
            self.outputs = nested_obj.output_desc.outputs

        self._convert_data_type_to_proto_data_type()

    def _parse_uint32(self, aux: bytes) -> (bytes, any):
        return self._unpack_value(aux, 'UINT32')

    def _convert_data_type_to_proto_data_type(self):
        for index, item in enumerate(self.inputs):
            self.inputs[index].data_type = NANO_DATA_TYPE_TO_PROTO_DATA_TYPE.get(item.data_type)

        for index, item in enumerate(self.outputs):
            self.outputs[index].data_type = NANO_DATA_TYPE_TO_PROTO_DATA_TYPE.get(item.data_type)
            self.outputs[index].original_data_type = NANO_DATA_TYPE_TO_PROTO_DATA_TYPE.get(item.original_data_type)


class NanoDumpDataParser:
    """
    The class for big dump data parser
    """
    warnings.filterwarnings("ignore")

    def __init__(self: any, dump_file_path: str) -> None:
        self.dump_file_path = dump_file_path
        self.file_size = os.path.getsize(self.dump_file_path)
        self.tlv_header_length = 0

        file_name = os.path.basename(self.dump_file_path)
        op_name = file_name.split('.')[1]
        dump_time = file_name.split('.')[4]
        self.nano_dump_data = NanoDumpData(op_name, dump_time)

    def parse(self: any) -> NanoDumpData:
        """
        Parse the dump file path by nano dump data format

        file format:
        |tlv head length| tlv head |input0 data |input1 data |...|output0 data |output1 data |...|
        |uint64         | tlv head |input0 data |input1 data |...|output0 data |output1 data |...|

        tlv head format:
        |——uint32 magicnum  //0x5A5A5A5A
        |——uint32 version_id
        |——list of tensor(input和output)

        :return: NanoDumpData
        :exception when read or parse file error
        """
        with open(self.dump_file_path, 'rb') as dump_file:
            self._read_header_length(dump_file)
            self._read_tlv_head_data(dump_file)
            self._check_size_match()
            self._read_input_data(dump_file)
            self._read_output_data(dump_file)
            return self.nano_dump_data

    def _check_size_match(self: any) -> None:
        input_data_size = 0
        for item in self.nano_dump_data.inputs:
            input_data_size += item.size
        output_data_size = 0
        for item in self.nano_dump_data.outputs:
            output_data_size += item.size

        if self.tlv_header_length + input_data_size + output_data_size + \
                ConstManager.UINT64_SIZE != self.file_size:
            log.print_error_log(
                'The file size (%d) of %r is not equal to %d (tlv_header_length)'
                '+ %d(the sum of input data) + %d(the sum of output data) '
                '. Please check the dump file.'
                % (self.file_size, self.dump_file_path,
                   self.tlv_header_length, input_data_size, output_data_size))
            raise CompareError(CompareError.MSACCUCMP_UNMATCH_STANDARD_DUMP_SIZE)

    def _read_header_length(self: any, dump_file: BinaryIO) -> None:
        # read tlv header length
        tlv_header_length = dump_file.read(ConstManager.UINT64_SIZE)
        self.tlv_header_length = struct.unpack(ConstManager.UINT64_FMT, tlv_header_length)[0]

        if self.tlv_header_length > self.file_size - ConstManager.UINT64_SIZE:
            log.print_warn_log(
                'The header content size (%d) of %r must be less than or'
                ' equal to %d (file size) - %d (tlv header length).'
                ' Please check the dump file.'
                % (self.tlv_header_length, self.dump_file_path, self.file_size, ConstManager.UINT64_SIZE))
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

    def _read_tlv_head_data(self: any, dump_file: BinaryIO) -> None:
        content = dump_file.read(self.tlv_header_length)
        try:
            self.nano_dump_data.parse_tlv_head(content)
        except CompareError as de_error:
            log.print_warn_log(
                'Failed to parse the serialized header content of %r. '
                'Please check the dump file. %s '
                % (self.dump_file_path, str(de_error)))
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from de_error

    def _read_input_data(self: any, dump_file: BinaryIO) -> None:
        for data_input in self.nano_dump_data.inputs:
            setattr(data_input, 'data', dump_file.read(data_input.size))

    def _read_output_data(self: any, dump_file: BinaryIO) -> None:
        for data_output in self.nano_dump_data.outputs:
            setattr(data_output, 'data', dump_file.read(data_output.size))


class NanoDumpDataHandler:
    """
    Handle dump data
    """

    def __init__(self: any, dump_file_path: str) -> None:
        self.dump_file_path = dump_file_path
        self.file_size = 0

    def check_is_nano_dump_format(self):
        ret = path_check.check_path_valid(self.dump_file_path, True)
        if ret != CompareError.MSACCUCMP_NONE_ERROR:
            return False

        self.file_size = os.path.getsize(self.dump_file_path)
        head_len = ConstManager.UINT64_SIZE + ConstManager.UINT32_SIZE
        if self.file_size > head_len:
            with open(self.dump_file_path, 'rb') as dump_file:
                # read tlv_head_len
                _ = dump_file.read(ConstManager.UINT64_SIZE)

                # read magicnum
                magicnum = dump_file.read(ConstManager.UINT32_SIZE)
                magicnum = struct.unpack(ConstManager.UINT32_FMT, magicnum)[0]
                return True if magicnum == NANO_DUMP_DATA_MAGIC_NUM else False
        return False

    def check_argument_valid(self: any) -> None:
        """
        check argument valid
        :exception when invalid
        """
        file_name = os.path.basename(self.dump_file_path)

        if len(file_name.split('.')) < 5:
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR, "The file name has no dump time")

        if not self.check_is_nano_dump_format():
            raise CompareError(CompareError.MSACCUCMP_INVALID_DUMP_DATA_ERROR)

        head_len = ConstManager.UINT64_SIZE * 2 + ConstManager.UINT32_SIZE * 2
        if self.file_size < head_len:
            log.print_error_log(
                'The file size (%d) of %r must to bigger than%d (head_len).'
                'Please check the dump file.'
                % (self.file_size, self.dump_file_path, head_len))

        if self.file_size > ConstManager.ONE_GB:
            log.print_warn_log(
                'The size (%d) of %r exceeds 1GB, it may task more time to run, please wait.'
                % (self.file_size, self.dump_file_path))

    def parse_dump_data(self: any, dump_version: int) -> NanoDumpData:
        """
        Parse dump file
        :param dump_version: the dump version
        :return: DumpData
        """
        self.check_argument_valid()
        try:
            nano_dump_parser = NanoDumpDataParser(self.dump_file_path)
            return nano_dump_parser.parse()
        except CompareError as error:
            message = 'Failed to parse the dump file %r, type is nano dump format. Please check the dump file. %s' \
                      % (self.dump_file_path, str(error))
            log.print_error_log(message)
            raise error


