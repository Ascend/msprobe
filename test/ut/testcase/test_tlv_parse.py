# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
"""
Function:
This file mainly involves the tlv info parse function.
"""
import struct

import pytest

from cmp_utils.tlv_parse import TLV
from cmp_utils.constant.compare_error import CompareError


data_type = 0x5a5a5a5a
data_id = 0x12121212
tag_of_shape_dim = 1
length_of_shape = 16
shape_dim0 = 0x0a0a0a0a0a0a0a0a
shape_dim1 = 0x0b0b0b0b0b0b0b0b

TLV_CFG_KEY_NAME = 'Name'


class FakeInfo:
    pass


@pytest.fixture(scope="module", autouse=True)
def fake_aux():
    aux = struct.pack("2i2I2Q", data_type, data_id, tag_of_shape_dim, length_of_shape, shape_dim0, shape_dim1)
    yield aux


@pytest.fixture(scope="module", autouse=True)
def fake_obj():
    class ParseInfo:
        def __init__(self):
            self.data_type = 0
            self.data_id = 0
            self.shape_dims = []

    fake_obj = ParseInfo()
    yield fake_obj


@pytest.fixture(scope="module", autouse=True)
def fake_tab():
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64', 'Tag': tag_of_shape_dim}
    ]
    yield TLV_CONFIG_TAB


def test_tlv_given_ATOM_and_TLNV_when_any_then_pass(fake_aux, fake_obj, fake_tab):
    tlv = TLV(fake_tab)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(fake_aux, fake_obj)
    assert len(aux) == 0

    assert fake_obj.data_type == data_type
    assert fake_obj.data_id == data_id
    assert len(fake_obj.shape_dims) == 2
    assert fake_obj.shape_dims[0] == shape_dim0
    assert fake_obj.shape_dims[1] == shape_dim1


def test_tlv_given_ATOM_and_TLV_when_any_then_pass(fake_obj):
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dim', 'TLV_Type': 'TLV', 'Ele_Type': 'UINT64', 'Tag': tag_of_shape_dim}
    ]

    tlv = TLV(TLV_CONFIG_TAB)
    aux = struct.pack("2i2IQ", data_type, data_id, tag_of_shape_dim, 8, shape_dim0)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(aux, fake_obj)
    assert len(aux) == 0

    assert fake_obj.data_type == data_type
    assert fake_obj.data_id == data_id
    assert fake_obj.shape_dim == shape_dim0


def test_tlv_given_err_cfg_then_failed(fake_obj):
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATT', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dim', 'TLV_Type': 'TLV', 'Ele_Type': 'UINT64', 'Tag': tag_of_shape_dim}
    ]

    tlv = TLV(TLV_CONFIG_TAB)
    aux = struct.pack("2i2IQ", data_type, data_id, tag_of_shape_dim, 8, shape_dim0)
    with pytest.raises(CompareError) as err:
        aux, fake_obj = tlv.parse_tlv_by_cfg_tab(aux, fake_obj)

    assert err.value.args[0] == CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR


def test_tlv_given_err_cfg_then_failed2(fake_obj):
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'UNDEFINE'},
    ]

    tlv = TLV(TLV_CONFIG_TAB)
    aux = struct.pack("2i2IQ", data_type, data_id, tag_of_shape_dim, 8, shape_dim0)
    with pytest.raises(CompareError) as err:
        aux, fake_obj = tlv.parse_tlv_by_cfg_tab(aux, fake_obj)

    assert err.value.args[0] == CompareError.MSACCUCMP_PARSE_NANO_DUMP_FILE_ERROR


def test_tlv_given_more_aux_when_aux_left_then_pass(fake_aux, fake_obj, fake_tab):
    tlv = TLV(fake_tab)

    left_aux = b'01237678'
    my_aux = fake_aux + left_aux
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(my_aux, fake_obj)
    assert len(aux) == len(left_aux)


def test_tlv_given_ATOM_and_TLNV_when_tag_err_then_pass(fake_aux, fake_obj):
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64', 'Tag': 2}
    ]

    tlv = TLV(TLV_CONFIG_TAB)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(fake_aux, fake_obj)
    assert fake_obj.shape_dims is None


def test_tlv_given_TLNV_and_CHAR_when_any_then_pass(fake_aux, fake_obj):
    tag_of_origin_name = 3
    tag_of_origin_shape_dims = 4

    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64', 'Tag': tag_of_shape_dim},
        {TLV_CFG_KEY_NAME: 'origin_name', 'TLV_Type': 'TLNV', 'Ele_Type': 'CHAR', 'Tag': tag_of_origin_name},
        {TLV_CFG_KEY_NAME: 'origin_shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT32', \
            'Tag': tag_of_origin_shape_dims},
    ]

    origin_shape_dim0 = 0x01010101
    origin_shape_dim1 = 0x02020202

    origin_shape_dims_aux = struct.pack('2I' + '2I', tag_of_origin_shape_dims, 8, origin_shape_dim0, origin_shape_dim1)

    my_name = 'my_name'
    fake_name = my_name.encode(encoding="utf-8")
    fake_aux_with_name = fake_aux + \
                         struct.pack('2I' + '%ds' % len(fake_name), tag_of_origin_name, len(fake_name), fake_name) + \
                         origin_shape_dims_aux

    tlv = TLV(TLV_CONFIG_TAB)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(fake_aux_with_name, fake_obj)
    assert len(aux) == 0

    assert fake_obj.data_type == data_type
    assert fake_obj.data_id == data_id
    assert len(fake_obj.shape_dims) == 2
    assert fake_obj.shape_dims[0] == shape_dim0
    assert fake_obj.shape_dims[1] == shape_dim1
    assert fake_obj.origin_name == my_name

    assert len(fake_obj.origin_shape_dims) == 2
    assert fake_obj.origin_shape_dims[0] == origin_shape_dim0
    assert fake_obj.origin_shape_dims[1] == origin_shape_dim1


def test_tlv_given_TLNV_when_ATOM_last_then_pass(fake_obj):
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64', 'Tag': tag_of_shape_dim},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
    ]

    # 本机序，按原字节对齐
    my_aux = struct.pack("=i2I2Qi", data_type, tag_of_shape_dim, length_of_shape, shape_dim0, shape_dim1, data_id)

    tlv = TLV(TLV_CONFIG_TAB)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(my_aux, fake_obj)
    assert len(aux) == 0

    assert fake_obj.data_type == data_type
    assert fake_obj.data_id == data_id
    assert len(fake_obj.shape_dims) == 2
    assert fake_obj.shape_dims[0] == shape_dim0
    assert fake_obj.shape_dims[1] == shape_dim1


def test_tlv_given_NV_when_any_then_pass():
    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'input_num', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
        {TLV_CFG_KEY_NAME: 'inputs', 'TLV_Type': 'NV', 'Ele_Type': 'UINT64', 'N': 'input_num'}
    ]

    fake_obj = FakeInfo()

    input_num = 4
    fake_aux = struct.pack("=I%sQ" % input_num, input_num, shape_dim0, shape_dim1, shape_dim0, shape_dim1)

    tlv = TLV(TLV_CONFIG_TAB)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(fake_aux, fake_obj)
    assert len(aux) == 0
    assert fake_obj.input_num == input_num
    assert len(fake_obj.inputs) == input_num
    assert fake_obj.inputs[0] == shape_dim0
    assert fake_obj.inputs[1] == shape_dim1
    assert fake_obj.inputs[2] == shape_dim0
    assert fake_obj.inputs[3] == shape_dim1


@pytest.fixture(scope="module", autouse=True)
def fake_Nested_tab(fake_tab):
    LIST_NESTED_TLV_CONFIG_TAB = \
    [
        {TLV_CFG_KEY_NAME: 'input_num', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
        {TLV_CFG_KEY_NAME: 'inputs', 'TLV_Type': 'ATOM', 'Ele_Type': fake_tab}
    ]
    yield LIST_NESTED_TLV_CONFIG_TAB


@pytest.fixture(scope="module", autouse=True)
def fake_Nested_aux():
    sub_aux = struct.pack("2i2I2Q", data_type, data_id, tag_of_shape_dim, length_of_shape, shape_dim0, shape_dim1)
    input_num = 1

    aux = struct.pack("i%ds" % len(sub_aux), input_num, sub_aux)
    yield aux


def test_tlv_given_ATOM_Nested_when_any_then_pass(fake_Nested_aux, fake_Nested_tab):
    tlv = TLV(fake_Nested_tab)

    nested_obj = FakeInfo()

    aux, nested_obj = tlv.parse_tlv_by_cfg_tab(fake_Nested_aux, nested_obj)
    assert len(aux) == 0

    assert nested_obj.input_num == 1
    assert nested_obj.inputs.data_type == data_type
    assert nested_obj.inputs.shape_dims[0] == shape_dim0
    assert nested_obj.inputs.shape_dims[1] == shape_dim1


@pytest.fixture(scope="module", autouse=True)
def fake_Nested_NV_tab(fake_tab):
    LIST_NESTED_TLV_CONFIG_TAB = \
    [
        {TLV_CFG_KEY_NAME: 'input_num', 'TLV_Type': 'ATOM', 'Ele_Type': 'UINT32'},
        {TLV_CFG_KEY_NAME: 'inputs', 'TLV_Type': 'NV', 'Ele_Type': fake_tab, 'N': 'input_num'}
    ]
    yield LIST_NESTED_TLV_CONFIG_TAB


@pytest.fixture(scope="module", autouse=True)
def fake_Nested_NV_aux(fake_aux):
    input_num = 2
    aux = struct.pack("i%ds" % (len(fake_aux) * input_num), input_num, fake_aux + fake_aux)
    yield aux


def test_tlv_given_NV_Nested_when_any_then_pass(fake_Nested_NV_aux, fake_Nested_NV_tab):
    tlv = TLV(fake_Nested_NV_tab)

    nested_obj = FakeInfo()

    aux, nested_obj = tlv.parse_tlv_by_cfg_tab(fake_Nested_NV_aux, nested_obj)
    assert len(aux) == 0

    assert nested_obj.input_num == 2
    assert len(nested_obj.inputs) == 2

    assert nested_obj.inputs[0].data_type == data_type
    assert nested_obj.inputs[0].shape_dims[0] == shape_dim0
    assert nested_obj.inputs[0].shape_dims[1] == shape_dim1

    assert nested_obj.inputs[1].data_type == data_type
    assert nested_obj.inputs[1].shape_dims[0] == shape_dim0
    assert nested_obj.inputs[1].shape_dims[1] == shape_dim1


def test_tlv_given_TLV_cfg_when_tlv_not_exit_then_pass(fake_aux, fake_obj):
    tag_of_origin_name = 3
    tag_of_origin_shape_dims = 4
    tag_of_fake_tlv = 7

    TLV_CONFIG_TAB = [
        {TLV_CFG_KEY_NAME: 'data_type', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'data_id', 'TLV_Type': 'ATOM', 'Ele_Type': 'INT32'},
        {TLV_CFG_KEY_NAME: 'shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT64', 'Tag': tag_of_shape_dim},
        {TLV_CFG_KEY_NAME: 'origin_name', 'TLV_Type': 'TLNV', 'Ele_Type': 'CHAR', 'Tag': tag_of_origin_name},
        {TLV_CFG_KEY_NAME: 'fake_tlv', 'TLV_Type': 'TLV', 'Ele_Type': 'UINT64', 'Tag': tag_of_fake_tlv},
        {TLV_CFG_KEY_NAME: 'origin_shape_dims', 'TLV_Type': 'TLNV', 'Ele_Type': 'UINT32', \
            'Tag': tag_of_origin_shape_dims},
    ]

    origin_shape_dim0 = 0x01010101
    origin_shape_dim1 = 0x02020202

    origin_shape_dims_aux = struct.pack('2I' + '2I', tag_of_origin_shape_dims, 8, origin_shape_dim0, origin_shape_dim1)

    # skip origin_name
    fake_aux_with_name = fake_aux + origin_shape_dims_aux

    tlv = TLV(TLV_CONFIG_TAB)
    aux, fake_obj = tlv.parse_tlv_by_cfg_tab(fake_aux_with_name, fake_obj)
    assert len(aux) == 0

    assert fake_obj.data_type == data_type
    assert fake_obj.data_id == data_id
    assert len(fake_obj.shape_dims) == 2
    assert fake_obj.shape_dims[0] == shape_dim0
    assert fake_obj.shape_dims[1] == shape_dim1
    assert fake_obj.origin_name is None
    assert fake_obj.fake_tlv is None

    assert len(fake_obj.origin_shape_dims) == 2
    assert fake_obj.origin_shape_dims[0] == origin_shape_dim0
    assert fake_obj.origin_shape_dims[1] == origin_shape_dim1


def test_tlv_given_TLV_Nested_when_any_then_pass(fake_tab, fake_aux):
    tag_of_inputs_desc = 9
    LIST_NESTED_TLV_CONFIG_TAB = \
    [
        {TLV_CFG_KEY_NAME: 'inputs_desc', 'TLV_Type': 'TLV', 'Ele_Type': fake_tab, 'Tag': tag_of_inputs_desc}
    ]
    aux = struct.pack("2I", tag_of_inputs_desc, len(fake_aux)) + fake_aux

    tlv = TLV(LIST_NESTED_TLV_CONFIG_TAB)

    nested_tlv_obj = FakeInfo()
    aux, nested_tlv_obj = tlv.parse_tlv_by_cfg_tab(aux, nested_tlv_obj)
    assert len(aux) == 0

    assert nested_tlv_obj.inputs_desc.data_type == data_type
    assert nested_tlv_obj.inputs_desc.shape_dims[0] == shape_dim0
    assert nested_tlv_obj.inputs_desc.shape_dims[1] == shape_dim1

    assert nested_tlv_obj.inputs_desc.data_type == data_type
    assert nested_tlv_obj.inputs_desc.shape_dims[0] == shape_dim0
    assert nested_tlv_obj.inputs_desc.shape_dims[1] == shape_dim1


def test_tlv_given_TLV_Nested_when_Tag_not_exist_then_pass(fake_tab, fake_aux):
    tag_of_inputs_desc = 9
    LIST_NESTED_TLV_CONFIG_TAB = \
    [
        {TLV_CFG_KEY_NAME: 'inputs_desc', 'TLV_Type': 'TLV', 'Ele_Type': fake_tab, 'Tag': tag_of_inputs_desc}
    ]

    aux = struct.pack("I", 0)

    tlv = TLV(LIST_NESTED_TLV_CONFIG_TAB)

    nested_obj = FakeInfo()
    aux, nested_obj = tlv.parse_tlv_by_cfg_tab(aux, nested_obj)
    assert len(aux) == 4

    assert nested_obj.inputs_desc is None