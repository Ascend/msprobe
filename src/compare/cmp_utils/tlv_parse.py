# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the tlv info parse function.

How to use:

1、Tag map:

TLV_CONFIG_TAB = [
    {'Name': 'data_type',  'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'data_id',    'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    {'Name': 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64', 'Tag': 1}
]

2、parse an bytes:
from cmp_utils.tlv_parse import TLV

class ParseInfo:
    def __init__(self):
        self.data_type = 0
        self.data_id = 0
        self.shape_dims = []

aux = b'0x5a\0x5a\0x5a\0x5a' + \
      b'0x11\0x11\0x11\0x11' + \
      b'0x01\0x00\0x00\0x00' + \
      b'0x10\0x00\0x00\0x00' + \
      b'0x0a\0x0a\0x0a\0x0b\0x0b\0x0b\0x0b\0x0b' + \
      b'0x0b\0x0b\0x0b\0x0b\0x0b\0x0b\0x0b\0x0b'

input = ParseInfo()
tlv = TLV(TLV_CONFIG_TAB)
aux, input = tlv.parse_tlv_by_cfg_tab(aux, input)

"""
import struct

from cmp_utils import log
from cmp_utils.constant.compare_error import CompareError

CHAR_SIZE = 1
INT16_SIZE = 2
UINT16_SIZE = 2
INT32_SIZE = 4
UINT32_SIZE = 4
INT64_SIZE = 8
UINT64_SIZE = 8

_UNPACK_FORMAT = {
    'CHAR':   {'FMT': 's', 'SIZE': 1},
    'UINT16': {'FMT': 'H', 'SIZE': 2},
    'INT16':  {'FMT': 'h', 'SIZE': 2},
    'UINT32': {'FMT': 'I', 'SIZE': 4},
    'INT32':  {'FMT': 'i', 'SIZE': 4},
    'UINT64': {'FMT': 'Q', 'SIZE': 8},
    'INT64':  {'FMT': 'q', 'SIZE': 8},
}

TLV_TYPE = ['ATOM', 'TLV', 'TLNV', 'NV']


class NestedObj:
    pass


class TLV:
    def __init__(self, tlv_tab, tag_type="UINT32", len_type="UINT32") -> None:
        self.tlv_tab = tlv_tab
        self.tag_type = tag_type
        self.len_type = len_type

    @staticmethod
    def _unpack_single_element(aux: bytes, ele_type) -> (bytes, any):
        type_para = _UNPACK_FORMAT.get(ele_type)
        ele_fmt, ele_size = type_para.get('FMT'), type_para.get('SIZE')

        if len(aux) < ele_size:
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)
        try:
            value = struct.unpack(ele_fmt, aux[:ele_size])[0]
        except struct.error as error:
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from error

        if ele_type == 'CHAR' and isinstance(value, bytes):
            value = value.decode(encoding="utf-8")

        return aux[ele_size:], value

    @staticmethod
    def _check_tl(tag, length, tlv_cfg):
        tag_in_cfg, ele_type = tlv_cfg.get("Tag"), tlv_cfg.get('Ele_Type')

        if tag_in_cfg != tag:
            log.print_error_log(
                'Failed to parse of tag_in_cfg: %s, tag_in_stream: %s.' % (tag_in_cfg, tag))
            raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)

        if not isinstance(ele_type, list):
            ele_size = _UNPACK_FORMAT.get(ele_type).get('SIZE')
            if length < ele_size:
                log.print_error_log(
                    'Failed to parse of tag: %s. length:%d < single ele_size:%d' % (tag_in_cfg, length, ele_size))
                raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)

    def parse_tlv_by_cfg_tab(self, content: bytes, x: object) -> (bytes, object):
        aux = content
        self._check_cfg_tab(self.tlv_tab)

        for tlv_cfg in self.tlv_tab:
            tlv_type = tlv_cfg.get('TLV_Type')
            if tlv_type == 'ATOM':
                aux, x = self._unpack_atom_value(aux, tlv_cfg, x)
            elif tlv_type == 'TLV':
                aux, x = self._unpack_tlv_value(aux, tlv_cfg, x)
            elif tlv_type == 'TLNV':
                aux, x = self._unpack_tlnv_value(aux, tlv_cfg, x)
            elif tlv_type == 'NV':
                num_attr_name = tlv_cfg.get('N')
                num = getattr(x, num_attr_name)
                aux, x = self._unpack_nv_value(aux, tlv_cfg, x, num)
        return aux, x

    def _check_cfg_tab(self, tlv_tab):
        for tlv_cfg in tlv_tab:
            tlv_type, ele_type = tlv_cfg.get('TLV_Type'), tlv_cfg.get('Ele_Type')
            if tlv_type not in TLV_TYPE:
                log.print_error_log('%s invalid tlv_type: %s.' % (tlv_cfg.get('Name'), tlv_type))
                raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)

            # Nested tlv
            if isinstance(ele_type, list):
                self._check_cfg_tab(ele_type)
            elif _UNPACK_FORMAT.get(ele_type) is None :
                log.print_error_log('invalid ele_type: %s.' % ele_type)
                raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)

            if tlv_type == 'NV':
                num_attr_name = tlv_cfg.get('N')
                if num_attr_name is None:
                    log.print_error_log("is tlv_tpye is NV, 'N' must be given!")
                    raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR)
        return

    def _unpack_atom_value(self, aux: bytes, tlv_cfg, x: object) -> (bytes, object):
        ele_type, ele_name = tlv_cfg.get('Ele_Type'), tlv_cfg.get('Name')

        if isinstance(ele_type, list):
            # parse Nested tlv value
            sub_tlv_tab = ele_type
            sub_tlv = TLV(sub_tlv_tab)
            nested_obj = NestedObj()
            aux, value = sub_tlv.parse_tlv_by_cfg_tab(aux, nested_obj)
        else:
            try:
                # parse single element
                aux, value = self._unpack_single_element(aux, ele_type)
            except CompareError as error:
                log.print_error_log('Failed to parse of ele name: %s, value_type: %s, '
                                    'Please check the dump file:%s.' % (tlv_cfg.get('Name'), ele_type, str(error)))
                raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from error

        setattr(x, ele_name, value)
        return aux, x

    def _parse_tag_len(self, aux: bytes) -> (bytes, any, any):
        aux, tag = self._unpack_single_element(aux, self.tag_type)
        aux, length = self._unpack_single_element(aux, self.len_type)
        return aux, tag, length

    def _unpack_tlv_value(self, aux: bytes, tlv_cfg, x: object) -> (bytes, object):
        ele_type, ele_name = tlv_cfg.get('Ele_Type'), tlv_cfg.get('Name')

        # tag, len
        try:
            aux_try, tag, length = self._parse_tag_len(aux)
        except CompareError:
            setattr(x, ele_name, None)
            log.print_info_log('Failed to parse of ele name: %s, ele_type: %s' % (ele_name, ele_type))
            return aux, x
        try:
            self._check_tl(tag, length, tlv_cfg)
        except CompareError:
            log.print_info_log('Failed to parse of ele name: %s, ele_type: %s' % (ele_name, ele_type))
            setattr(x, ele_name, None)
            return aux, x

        aux = aux_try
        return self._unpack_atom_value(aux, tlv_cfg, x)

    def _unpack_tlnv_value(self, aux: bytes, tlv_cfg, x: object) -> (bytes, object):
        tag_in_cfg, ele_type, ele_name = tlv_cfg.get("Tag"), tlv_cfg.get('Ele_Type'), tlv_cfg.get('Name')

        # tag, len
        try:
            aux_try, tag, length = self._parse_tag_len(aux)
        except CompareError:
            setattr(x, ele_name, None)
            log.print_info_log('skip to parse tlv of ele name: %s, ele_type: %s' % (ele_name, ele_type))
            return aux, x
        try:
            self._check_tl(tag, length, tlv_cfg)
        except CompareError:
            log.print_info_log('skip to parse tlv of ele name: %s, ele_type: %s' % (ele_name, ele_type))
            setattr(x, ele_name, None)
            return aux, x

        aux = aux_try

        # get list num of value
        if _UNPACK_FORMAT.get(ele_type) is not None:
            ele_size = _UNPACK_FORMAT.get(ele_type).get('SIZE')
            ele_num = int(length / ele_size)

            try:
                aux, x = self._unpack_nv_value(aux, tlv_cfg, x, ele_num)
            except CompareError as error:
                log.print_error_log('Failed to parse of ele name: %s, value_type: %s, '
                                    'Please check the dump file:%s.' % (tlv_cfg.get('Name'), ele_type, str(error)))
                raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from error

        elif isinstance(ele_type, list):
            value = []
            sub_aux = aux[:length]

            while (len(sub_aux) > 0):
                # Nested tlv
                sub_tlv_tab = ele_type
                sub_tlv, nested_obj = TLV(sub_tlv_tab), NestedObj()
                sub_aux, single_value = sub_tlv.parse_tlv_by_cfg_tab(sub_aux, nested_obj)
                value.append(single_value)

            setattr(x, tlv_cfg.get('Name'), value)
            aux = aux[length:]
        return aux, x

    def _unpack_nv_value(self, aux: bytes, tlv_cfg, x: object, ele_num) -> (bytes, object):
        ele_type, ele_name = tlv_cfg.get('Ele_Type'), tlv_cfg.get('Name')
        value = []

        for _ in range(ele_num):
            if isinstance(ele_type, list):
                # Nested tlv
                sub_tlv_tab = ele_type
                sub_tlv, nested_obj = TLV(sub_tlv_tab), NestedObj()
                aux, single_value = sub_tlv.parse_tlv_by_cfg_tab(aux, nested_obj)
                value.append(single_value)
            else:
                try:
                    # single element
                    aux, single_value = self._unpack_single_element(aux, ele_type)
                except CompareError as error:
                    log.print_error_log('Failed to parse of ele name: %s, value_type: %s, '
                                        'Please check the dump file:%s.' % (tlv_cfg.get('Name'), ele_type, str(error)))
                    raise CompareError(CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR) from error
                value.append(single_value)

        if ele_type == 'CHAR' and isinstance(value, list):
            value_str = "".join(value)
            setattr(x, tlv_cfg.get('Name'), value_str)
        else:
            setattr(x, tlv_cfg.get('Name'), value)
        return aux, x
