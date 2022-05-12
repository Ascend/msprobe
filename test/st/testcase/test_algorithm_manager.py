import unittest
import pytest
import numpy as np
import utils

from algorithm_manager import AlgorithmManager
from algorithm_manager import AlgorithmManagerMain
from compare_error import CompareError
from unittest import mock


class TestUtilsMethods(unittest.TestCase):

    def test_make_select_algorithm_map1(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "", "")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map2(self):
        manager = AlgorithmManager("", "all", ";")
        self.assertEqual(len(manager.built_in_support_algorithm), 10)

    def test_make_select_algorithm_map3(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "all", ":")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map4(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "all", ",")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map5(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "all", "xx")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map6(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "all", "xx:a")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map7(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "all", "xx:a=")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map8(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "all", "xx:=b")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_make_select_algorithm_map9(self):
        manager = AlgorithmManager("", "all", "xx:a=b")
        self.assertEqual(len(manager.algorithm_param), 1)
        self.assertEqual(manager.algorithm_param['xx']['a'], 'b')

    def test_make_select_algorithm_map10(self):
        manager = AlgorithmManager("", "all", "xx:a=b;xx:a=c,d=e")
        self.assertEqual(len(manager.algorithm_param), 1)
        self.assertEqual(len(manager.algorithm_param['xx']), 2)
        self.assertEqual(manager.algorithm_param['xx']['a'], 'c')

    def test_make_select_algorithm_map11(self):
        manager = AlgorithmManager("", "all", "0:a=b;1:a=c,d=e")
        self.assertEqual(len(manager.algorithm_param), 2)
        self.assertEqual(len(manager.algorithm_param['CosineSimilarity']), 1)
        self.assertEqual(len(manager.algorithm_param['MaxAbsoluteError']), 2)
        self.assertEqual(manager.algorithm_param['CosineSimilarity']['a'], 'b')

    def test_make_select_algorithm_map12(self):
        manager = AlgorithmManager("", "0,  CosineSimilarity, 5, cc", "")
        self.assertEqual(len(manager.select_algorithm_list), 2)

    def test_make_select_algorithm_map13(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("", "cc", "")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)

    def test_make_select_algorithm_map14(self):
        with pytest.raises(CompareError) as error:
            AlgorithmManager("/xxx", "cc", "")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_PATH_ERROR)

    def test_make_select_algorithm_map15(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR):
                AlgorithmManager("/xxxxx", "cc", "")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)

    def test_make_select_algorithm_map16(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('os.listdir', return_value=['xxx.py']):
                AlgorithmManager("/xxxxx", "cc", "")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)

    def test_make_select_algorithm_map17(self):
        with pytest.raises(CompareError) as error:
            with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR), \
                 mock.patch('os.path.exists', return_value=True), \
                 mock.patch('os.path.isfile', return_value=True), \
                 mock.patch('os.listdir', return_value=['alg_xxx.py']):
                AlgorithmManager("/xxxxx", "cc", "")
        self.assertEqual(error.value.args[0], CompareError.MSACCUCMP_INVALID_ALGORITHM_ERROR)

    def test_make_select_algorithm_map18(self):
        with mock.patch('utils.check_path_valid', return_value=CompareError.MSACCUCMP_NONE_ERROR), \
             mock.patch('os.path.exists', return_value=True), \
             mock.patch('os.path.isfile', return_value=True):
            manager = AlgorithmManager("", "MaxAbsoluteError", "")
        self.assertEqual(len(manager.built_in_support_algorithm), 10)
        self.assertEqual(len(manager.support_algorithm_map), 1)

    def test_process1(self):
        args = mock.Mock()
        args.my_dump_path = '/home/a.npy'
        args.golden_dump_path = '/home/b.npy'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        dump_data = np.arange(2)
        with pytest.raises(CompareError) as error:
            with mock.patch('utils.check_path_valid', side_effect=[0, 1]):
                with mock.patch('utils.read_numpy_file', return_value=dump_data):
                    AlgorithmManagerMain(args).process()
        self.assertEqual(error.value.args[0], 1)

    def test_process2(self):
        args = mock.Mock()
        args.my_dump_path = '/home/a.npy'
        args.golden_dump_path = '/home/b.npy'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        dump_data1 = np.arange(2)
        dump_data2 = np.arange(6)
        with mock.patch('utils.check_path_valid', return_value=0):
            with mock.patch('utils.read_numpy_file', side_effect=[dump_data1, dump_data2]):
                ret = AlgorithmManagerMain(args).process()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_SHAPE_ERROR)

    def test_process3(self):
        args = mock.Mock()
        args.my_dump_path = '/home/a.npy'
        args.golden_dump_path = '/home/b.npy'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        dump_data = np.arange(2)
        with mock.patch('utils.check_path_valid', return_value=0):
            with mock.patch('utils.read_numpy_file', return_value=dump_data):
                AlgorithmManagerMain(args).process()

    def test_process4(self):
        args = mock.Mock()
        args.my_dump_path = '/home/a.npy'
        args.golden_dump_path = '/home/b.npy'
        args.custom_script_path = ''
        args.algorithm = '5,1,0'
        args.algorithm_options = ''
        dump_data = np.arange(2)
        with mock.patch('utils.check_path_valid', return_value=0):
            with mock.patch('utils.read_numpy_file', return_value=dump_data):
                AlgorithmManagerMain(args).process()

    def test_process5(self):
        args = mock.Mock()
        args.my_dump_path = '/home/a.npy'
        args.golden_dump_path = '/home/b.npy'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        dump_data = np.zeros(5)
        with mock.patch('utils.check_path_valid', return_value=0):
            with mock.patch('utils.read_numpy_file', return_value=dump_data):
                ret = AlgorithmManagerMain(args).process()
        self.assertEqual(ret, 0)

    def test_algorithmManager_compare1(self):
        a_m = AlgorithmManager('', 'all', '')
        my_output_dump_data = mock.Mock()
        my_output_dump_data.__len__ = self._get_len
        ground_truth_dump_data = mock.Mock()
        ground_truth_dump_data.__len__ = self._get_len
        my_output_dump_data.dtype = np.bool_
        ground_truth_dump_data.dtype = np.bool_
        with mock.patch(
                'algorithm_manager.AlgorithmManager._make_algorithm_param', return_value={}):
            a_m.compare(my_output_dump_data, ground_truth_dump_data, {})

    def test_algorithmManager_compare2(self):
        a_m = AlgorithmManager('', 'all', '')
        my_output_dump_data = mock.Mock()
        my_output_dump_data.__len__ = self._get_len
        ground_truth_dump_data = mock.Mock()
        ground_truth_dump_data.__len__ = self._get_len
        my_output_dump_data.dtype = None
        ground_truth_dump_data.dtype = None
        with mock.patch(
                'algorithm_manager.AlgorithmManager._check_data_size_valid'):
            with mock.patch(
                    'algorithm_manager.AlgorithmManager._make_algorithm_param', return_value={}):
                with mock.patch(
                        'algorithm_manager.AlgorithmManager._call_compare_function', return_value=(123, '')):
                    a_m.compare(my_output_dump_data, ground_truth_dump_data, {})

    def _get_len(self, args):
        return 4

if __name__ == '__main__':
    unittest.main()
