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

import os
import unittest
from unittest import mock
from unittest.mock import patch 
from collections import namedtuple

import pytest
import numpy as np

from algorithm_manager.algorithm_manager import AlgorithmManager, AlgorithmManagerMain
from cmp_utils.constant.compare_error import CompareError


class Args:
    def __init__(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)


@pytest.fixture(scope="function")
def fake_arguments():
    return Args(
        my_dump_path='/home/a.npy',
        golden_dump_path='/home/b.npy',
        custom_script_path="",
        algorithm="all",
        algorithm_options="",
        output_path=None,
    )


def _mock_algorithm_manager(custom_script_path, select_algorithm="", algorithm_options=""):
    with mock.patch('cmp_utils.path_check.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.path.isfile', return_value=True):
                return AlgorithmManager(custom_script_path, select_algorithm, algorithm_options)


def test_make_select_algorithm_map_given_all_when_valid_then_pass():
    manager = AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options=";")
    assert len(manager.built_in_support_algorithm) == 10


def test_make_select_algorithm_map_given_empty_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="", algorithm_options="")


def test_algorithm_options_given_valid_when_any_then_pass():
    manager = AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="xx:a=b")
    assert len(manager.algorithm_param) == 1
    assert manager.algorithm_param.get('xx') == {'a': 'b'}


def test_algorithm_options_given_multi_when_valid_then_pass():
    manager = AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="xx:a=b;xx:a=c,d=e")
    assert len(manager.algorithm_param) == 1
    assert len(manager.algorithm_param.get('xx')) == 2
    assert manager.algorithm_param.get('xx') == {'a': 'c', 'd': 'e'}


def test_algorithm_options_given_index_when_valid_then_pass():
    manager = AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="0:a=b;1:a=c,d=e")
    assert len(manager.algorithm_param) == 2
    assert len(manager.algorithm_param.get('CosineSimilarity')) == 1
    assert len(manager.algorithm_param.get('MaxAbsoluteError')) == 2
    assert manager.algorithm_param.get('CosineSimilarity') == {'a': 'b'}
    assert manager.algorithm_param.get('MaxAbsoluteError') == {'a': 'c', 'd': 'e'}


def test_algorithm_options_given_colon_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options=":")


def test_algorithm_options_given_comma_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options=",")


def test_algorithm_options_given_name_only_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="xx")


def test_algorithm_options_given_name_colon_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="xx:a")


def test_algorithm_options_given_name_colon_equal_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="xx:a=")


def test_algorithm_options_given_name_equal_colon_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PARAM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="all", algorithm_options="xx:=b")


def test_select_algorithm_given_valid_when_any_then_pass():
    manager = AlgorithmManager(
        custom_script_path="", select_algorithm="0, CosineSimilarity, 5, cc", algorithm_options=""
    )
    assert len(manager.select_algorithm_list) == 2


def test_select_algorithm_given_invalid_when_any_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)):
        AlgorithmManager(custom_script_path="", select_algorithm="cc", algorithm_options="")


def test_custom_script_path_given_any_when_invalid_custom_script_path_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_PATH_ERROR)):
        AlgorithmManager(custom_script_path="/not_exists_path", select_algorithm="cc", algorithm_options="")


def test_custom_script_path_given_invalid_when_valid_custom_script_path_then_error():
    mock_stat_result = os.stat_result((16877, 2, 64768, 22, os.getuid(), 0, 0, 0, 0, 0))
    
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)):
        with mock.patch('os.stat', return_value=mock_stat_result), \
             mock.patch('os.path.exists', side_effect=[True, True, False]):
            AlgorithmManager(custom_script_path="/", select_algorithm="cc", algorithm_options="")


def test_custom_script_path_given_valid_when_valid_then_pass():
    manager = _mock_algorithm_manager(custom_script_path="", select_algorithm="MaxAbsoluteError")
    assert len(manager.built_in_support_algorithm) == 10
    assert len(manager.support_algorithm_map) == 1


def test_custom_script_path_given_not_match_when_valid_custom_script_path_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)):
        with mock.patch('os.listdir', return_value=['xxx.py']):
            _mock_algorithm_manager(custom_script_path="/tmp", select_algorithm="cc", algorithm_options="")


def test_custom_script_path_given_invalid_when_valid_custom_script_file_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)):
        with mock.patch('os.listdir', return_value=['alg_xxx.py']), mock.patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = 0o640
            mock_stat.return_value.st_uid = os.getuid()
            _mock_algorithm_manager(custom_script_path="/tmp", select_algorithm="cc", algorithm_options="")


def test_custom_script_path_given_not_match_group_when_valid_custom_script_path_then_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_DANGER_FILE_ERROR)):
        with mock.patch('os.listdir', return_value=['alg_xxx.py']), mock.patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = 0o640
            mock_stat.return_value.st_uid = os.getuid() + 1
            _mock_algorithm_manager(custom_script_path="/tmp", select_algorithm="cc", algorithm_options="")


def test_check_file_permission_when_others_have_write_permission_then_raise_error():
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_DANGER_FILE_ERROR)):
        with mock.patch('os.listdir', return_value=['alg_xxx.py']), mock.patch('os.stat') as mock_stat:
            mock_stat.return_value.st_mode = 0o666
            mock_stat.return_value.st_uid = os.getuid()
            _mock_algorithm_manager(custom_script_path="/tmp", select_algorithm="cc", algorithm_options="")


def test_algorithm_manager_main_given_any_when_unknown_error_then_error(fake_arguments):
    with pytest.raises(CompareError, match=str(CompareError.MSACCUCMP_UNKNOWN_ERROR)):
        with mock.patch('cmp_utils.path_check.check_path_valid', side_effect=[0, 1]):
            AlgorithmManagerMain(fake_arguments).process()


def test_algorithm_manager_main_given_any_when_invalid_shape_then_error(fake_arguments):
    dump_data1 = np.arange(2)
    dump_data2 = np.arange(6)

    with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
        with mock.patch('dump_parse.dump_utils.read_numpy_file', side_effect=[dump_data1, dump_data2]):
            ret = AlgorithmManagerMain(fake_arguments).process()
    assert ret == CompareError.MSACCUCMP_INVALID_SHAPE_ERROR


def test_algorithm_manager_main_given_valid_when_any_then_pass(fake_arguments):
    dump_data = np.arange(2)

    with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
        with mock.patch('dump_parse.dump_utils.read_numpy_file', side_effect=dump_data):
            ret = AlgorithmManagerMain(fake_arguments).process()
    assert ret == CompareError.MSACCUCMP_NONE_ERROR


def test_algorithm_manager_main_given_algorithm_when_valid_then_pass(fake_arguments):
    dump_data = np.arange(2)
    fake_arguments.algorithm = "5,1,0"
    with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
        with mock.patch('dump_parse.dump_utils.read_numpy_file', side_effect=dump_data):
            ret = AlgorithmManagerMain(fake_arguments).process()
    assert ret == CompareError.MSACCUCMP_NONE_ERROR


def test_algorithm_manager_main_given_zeros_when_valid_then_pass(fake_arguments):
    dump_data = np.zeros(5)
    with mock.patch('cmp_utils.path_check.check_path_valid', return_value=0):
        with mock.patch('dump_parse.dump_utils.read_numpy_file', side_effect=dump_data):
            ret = AlgorithmManagerMain(fake_arguments).process()
    assert ret == CompareError.MSACCUCMP_NONE_ERROR


def test_algorithm_manager_compare_given_bool_when_valid_then_pass():
    a_m = AlgorithmManager('', 'all', '')
    my_output_dump_data = namedtuple('aa', ['dtype', 'test'])(np.bool_, 'test')
    ground_truth_dump_data = namedtuple('aa', ['dtype', 'test'])(np.bool_, 'test')
    with mock.patch('algorithm_manager.algorithm_manager.AlgorithmManager._make_algorithm_param', return_value={}):
        a_m.compare(my_output_dump_data, ground_truth_dump_data, {})


def test_algorithm_manager_compare_given_none_dtype_when_valid_then_pass():
    a_m = AlgorithmManager('', 'all', '')
    my_output_dump_data = namedtuple('aa', ['dtype', 'test', "test_2"])(None, 1, 2)
    ground_truth_dump_data = namedtuple('aa', ['dtype', 'test', "test_2"])(None, 1, 2)
    with mock.patch('algorithm_manager.algorithm_manager.AlgorithmManager._check_data_size_valid'):
        with mock.patch('algorithm_manager.algorithm_manager.AlgorithmManager._make_algorithm_param', return_value={}):
            with mock.patch(
                    'algorithm_manager.algorithm_manager.AlgorithmManager._call_compare_function',
                    return_value=(123, '')):
                a_m.compare(my_output_dump_data, ground_truth_dump_data, {})


class TestAlgorithmManagerMain(unittest.TestCase):

    def setUp(self):
        self.args = mock.Mock
        self.args.my_dump_path = "a.npy"
        self.args.golden_dump_path = "b.npy"
        self.args.custom_script_path = ""
        self.args.algorithm = "all"
        self.args.algorithm_options = ""
        self.args.output_path = "path"

    @patch('os.path.realpath', return_value=True)
    @patch('os.path.islink', return_value=True)
    @patch("cmp_utils.log.print_error_log")
    def test_init_when_init_raise_error(self, mock_error_log, mock_islink, mock_realpath):
        with self.assertRaises(CompareError) as context:
            ret = AlgorithmManagerMain(self.args)
        mock_error_log.assert_called_once_with('The path "%r" is a softlink, not permitted.' % self.args.output_path)
        assert context.exception.args[0] == CompareError.MSACCUCMP_INVALID_PATH_ERROR
    
    @patch("cmp_utils.path_check.check_output_path_valid", return_value=9)
    def test_print_result_save_result_invalid(self, mock_check_output_path_valid):
        with self.assertRaises(CompareError) as context:
            ret = AlgorithmManagerMain(self.args)._print_result([], [], "path")
        assert context.exception.args[0] == CompareError.MSACCUCMP_OPEN_DIR_ERROR
    