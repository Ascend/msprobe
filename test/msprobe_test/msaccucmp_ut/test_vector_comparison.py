import os
import unittest
from unittest import mock

from vector_cmp import vector_comparison as compare_vector
from cmp_utils.constant.compare_error import CompareError
from dump_parse.dump import DumpType
from cmp_utils.constant.const_manager import ConstManager


class TestProcessSingleOpMaxLineParameters(unittest.TestCase):
    def test_max_line_within_range(self):
        try:
            compare_vector.VectorComparison._process_single_op_max_line_parameters(50000)
        except Exception:
            self.fail("Should not raise for valid param")

    def test_max_line_below_min(self):
        with self.assertRaises(CompareError) as ctx:
            compare_vector.VectorComparison._process_single_op_max_line_parameters(100)
        self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_max_line_above_max(self):
        with self.assertRaises(CompareError) as ctx:
            compare_vector.VectorComparison._process_single_op_max_line_parameters(9999999)
        self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_max_line_at_min_boundary(self):
        try:
            compare_vector.VectorComparison._process_single_op_max_line_parameters(
                ConstManager.DETAIL_LINE_COUNT_RANGE_MIN)
        except Exception:
            self.fail("Should not raise at min boundary")

    def test_max_line_at_max_boundary(self):
        try:
            compare_vector.VectorComparison._process_single_op_max_line_parameters(
                ConstManager.DETAIL_LINE_COUNT_RANGE_MAX)
        except Exception:
            self.fail("Should not raise at max boundary")


class TestCheckBothDumpData(unittest.TestCase):
    def _make_args(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right',
                '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True):
                return compare_vector.VectorComparison()

    def test_both_offline_with_overflow(self):
        main = self._make_args()
        main.compare_data.left_dump_info.type = DumpType.Offline
        main.compare_data.right_dump_info.type = DumpType.Offline
        main.args['overflow_detection'] = True
        ret = main._check_both_dump_data()
        self.assertTrue(ret)

    def test_both_offline_no_overflow(self):
        main = self._make_args()
        main.compare_data.left_dump_info.type = DumpType.Offline
        main.compare_data.right_dump_info.type = DumpType.Offline
        main.args['overflow_detection'] = False
        ret = main._check_both_dump_data()
        self.assertTrue(ret)

    def test_neither_offline(self):
        main = self._make_args()
        main.compare_data.left_dump_info.type = DumpType.Numpy
        main.compare_data.right_dump_info.type = DumpType.Standard
        ret = main._check_both_dump_data()
        self.assertFalse(ret)

    def test_one_not_offline(self):
        main = self._make_args()
        main.compare_data.left_dump_info.type = DumpType.Offline
        main.compare_data.right_dump_info.type = DumpType.Numpy
        ret = main._check_both_dump_data()
        self.assertFalse(ret)


class TestProcessOutputPathParameter(unittest.TestCase):
    def _make_main_with_arguments(self):
        args = mock.MagicMock()
        args.mapping = False
        args.output_path = '/home/result'
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        main = compare_vector.VectorComparison(arguments=args)
        return main

    @mock.patch('os.path.exists', return_value=True)
    def test_normal_output_path(self, mock_exists):
        args = mock.MagicMock()
        args.mapping = False
        args.output_path = '/home/result'
        main = self._make_main_with_arguments()
        main._process_output_path_parameter(args)
        self.assertIn('result_', main.output_path)
        self.assertTrue(main.output_path.endswith('.csv'))

    @mock.patch('os.path.exists', return_value=True)
    def test_mapping_output_path(self, mock_exists):
        args = mock.MagicMock()
        args.mapping = True
        args.output_path = '/home/result'
        main = self._make_main_with_arguments()
        main._process_output_path_parameter(args)
        self.assertIn('mapping_', main.output_path)
        self.assertTrue(main.output_path.endswith('.csv'))


class TestProcessSingleOpParameters(unittest.TestCase):
    def setUp(self):
        self.args = mock.MagicMock()
        self.args.output_path = '/home/result'
        self.args.op_name = 'test_op'
        self.args.max_line = None
        self.args.topn = 20
        self.args.ignore_single_op_result = False
        self.args.input = None
        self.args.output = None
        self.args.fusion_rule_file = ''
        self.args.quant_fusion_rule_file = ''
        self.args.close_fusion_rule_file = ''
        self.args.my_dump_path = '/home/left'
        self.args.golden_dump_path = '/home/right'
        self.args.dump_version = ConstManager.OLD_DUMP_TYPE
        self.args.ffts = False
        self.args.csv = False
        self.args.custom_script_path = ''
        self.args.algorithm = 'all'
        self.args.algorithm_options = ''
        self.args.overflow_detection = False
        self.args.advisor = False
        self.args.range = None
        self.args.select = None
        self.args.max_cmp_size = 0
        self.args.mapping = False

    def test_single_op_input_tensor(self):
        self.args.input = '0'
        main = compare_vector.VectorComparison(arguments=self.args)
        self.assertIsNotNone(main.detail_info)
        self.assertEqual(main.output_path, '/home/result')

    def test_single_op_output_tensor(self):
        self.args.output = '0'
        main = compare_vector.VectorComparison(arguments=self.args)
        self.assertIsNotNone(main.detail_info)

    def test_single_op_max_line_out_of_range(self):
        self.args.max_line = 100
        with self.assertRaises(CompareError) as ctx:
            compare_vector.VectorComparison(arguments=self.args)
        self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)


class TestFilterLeftDumpIsNpyOverflow(unittest.TestCase):
    def _make_args(self):
        args = ['aaa.py', '-l', '/home/left', '-r', '/home/right',
                '-o', '/home/result', '-d', 'C2']
        with mock.patch('sys.argv', args):
            with mock.patch('os.path.exists', return_value=True):
                return compare_vector.VectorComparison()

    def test_filter_npy_overflow_disabled(self):
        main = self._make_args()
        main.args['overflow_detection'] = True
        main.compare_data.left_dump_info.op_name_to_file_map = {
            'op1': ['/path/to/op1.npy'],
        }
        main._filter_left_dump_is_npy_overflow()
        self.assertFalse(main.args['overflow_detection'])

    def test_filter_npy_overflow_not_npy(self):
        main = self._make_args()
        main.args['overflow_detection'] = True
        main.compare_data.left_dump_info.op_name_to_file_map = {
            'op1': ['/path/to/op1.bin'],
        }
        main._filter_left_dump_is_npy_overflow()
        self.assertTrue(main.args['overflow_detection'])

    def test_filter_npy_overflow_no_overflow(self):
        main = self._make_args()
        main.args['overflow_detection'] = False
        main.compare_data.left_dump_info.op_name_to_file_map = {
            'op1': ['/path/to/op1.npy'],
        }
        main._filter_left_dump_is_npy_overflow()
        self.assertFalse(main.args['overflow_detection'])

    def test_filter_npy_overflow_empty_map(self):
        main = self._make_args()
        main.args['overflow_detection'] = True
        main.compare_data.left_dump_info.op_name_to_file_map = {}
        main._filter_left_dump_is_npy_overflow()
        self.assertTrue(main.args['overflow_detection'])

    def test_filter_npy_overflow_first_op_empty(self):
        main = self._make_args()
        main.args['overflow_detection'] = True
        main.compare_data.left_dump_info.op_name_to_file_map = {
            'op1': [],
            'op2': ['/path/to/op2.npy'],
        }
        main._filter_left_dump_is_npy_overflow()
        self.assertFalse(main.args['overflow_detection'])

    def test_filter_npy_overflow_all_empty(self):
        main = self._make_args()
        main.args['overflow_detection'] = True
        main.compare_data.left_dump_info.op_name_to_file_map = {
            'op1': [],
            'op2': [],
        }
        main._filter_left_dump_is_npy_overflow()
        self.assertTrue(main.args['overflow_detection'])


class TestGetMaxProcessNum(unittest.TestCase):
    def setUp(self):
        self._orig_max_num = compare_vector.VectorComparison.MULTI_THREAD_MAX_NUM
        compare_vector.VectorComparison.MULTI_THREAD_MAX_NUM = 1
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)

    def tearDown(self):
        compare_vector.VectorComparison.MULTI_THREAD_MAX_NUM = self._orig_max_num

    @mock.patch('os.listdir', return_value=['file1', 'file2'])
    @mock.patch('os.path.getsize', return_value=1024)
    @mock.patch('multiprocessing.cpu_count', return_value=8)
    @mock.patch('psutil.virtual_memory')
    def test_max_process_num_normal(self, mock_vm, mock_cpu, mock_size, mock_listdir):
        compare_vector.VectorComparison.MULTI_THREAD_MAX_NUM = self._orig_max_num
        mock_vm.return_value.available = 1024 * 1024 * 1024
        ret = self.main._get_max_process_num()
        self.assertGreater(ret, 0)

    @mock.patch('os.listdir', return_value=['file1', 'file2'])
    @mock.patch('os.path.getsize', return_value=1024)
    @mock.patch('multiprocessing.cpu_count', return_value=16)
    @mock.patch('psutil.virtual_memory')
    def test_max_process_num_capped(self, mock_vm, mock_cpu, mock_size, mock_listdir):
        compare_vector.VectorComparison.MULTI_THREAD_MAX_NUM = self._orig_max_num
        mock_vm.return_value.available = 1024 * 1024 * 1024
        ret = self.main._get_max_process_num()
        self.assertGreater(ret, 0)


class TestGetResultList(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)

    def test_get_result_list_single(self):
        mock_result = mock.MagicMock()
        mock_item = mock.MagicMock()
        mock_result.result_list = [mock_item]
        res = [None, None, [mock_result]]
        ret = self.main._get_result_list(res)
        self.assertEqual(len(ret), 1)

    def test_get_result_list_multiple(self):
        mock_result1 = mock.MagicMock()
        mock_result1.result_list = ['a', 'b']
        mock_result2 = mock.MagicMock()
        mock_result2.result_list = ['c']
        res = [None, None, [mock_result1, mock_result2]]
        ret = self.main._get_result_list(res)
        self.assertEqual(ret, ['a', 'b', 'c'])


class TestPreHandleHeader(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)

    @mock.patch('cmp_utils.utils.dump_path_contains_npy')
    def test_pre_handle_header_neither_npy(self, mock_contains):
        mock_contains.return_value = False
        ret = self.main._pre_handle_header()
        self.assertIsInstance(ret, list)
        self.assertIn('Address', ret)

    @mock.patch('cmp_utils.utils.dump_path_contains_npy')
    def test_pre_handle_header_my_npy(self, mock_contains):
        def side_effect(path):
            return path == '/home/left'
        mock_contains.side_effect = side_effect
        ret = self.main._pre_handle_header()
        self.assertIsInstance(ret, list)
        self.assertIn('Address', ret)

    @mock.patch('cmp_utils.utils.dump_path_contains_npy')
    def test_pre_handle_header_golden_npy(self, mock_contains):
        def side_effect(path):
            return path == '/home/right'
        mock_contains.side_effect = side_effect
        ret = self.main._pre_handle_header()
        self.assertIsInstance(ret, list)
        self.assertIn('Address', ret)

    @mock.patch('cmp_utils.utils.dump_path_contains_npy')
    def test_pre_handle_header_both_npy(self, mock_contains):
        mock_contains.return_value = True
        ret = self.main._pre_handle_header()
        self.assertIsInstance(ret, list)
        self.assertNotIn('Address', ret)


class TestCompareFusionOps(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)
        self.main.compare_rule.fusion_info = mock.MagicMock()

    def test_compare_fusion_ops(self):
        with mock.patch.object(self.main, '_compare_by_fusion_op', return_value=(0, True, [])):
            ret = self.main._compare_fusion_ops(['op1'], lock=None)
            self.assertIsNotNone(ret)
            self.assertEqual(len(ret), 1)


class TestCompareByMultiProcess(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)
        self.main.compare_rule.fusion_info = mock.MagicMock()
        self.main.compare_rule.fusion_info.fusion_op_name_to_op_map = {}

    def test_compare_by_multi_process_no_ops(self):
        with mock.patch.object(self.main, '_get_max_process_num', return_value=1), \
                mock.patch.object(self.main, '_write_header_to_file', return_value=True), \
                mock.patch('multiprocessing.Pool'), \
                mock.patch('multiprocessing.Manager'):
            ret = self.main._compare_by_multi_process()
            self.assertEqual(ret[0], CompareError.MSACCUCMP_NONE_ERROR)
            self.assertEqual(ret[1], False)

    def test_compare_by_multi_process_write_header_fails(self):
        with mock.patch.object(self.main, '_write_header_to_file', return_value=False):
            ret = self.main._compare_by_multi_process()
            self.assertEqual(ret[0], CompareError.MSACCUCMP_OPEN_FILE_ERROR)
            self.assertEqual(ret[1], False)


class TestCompareVector(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)
        self.main.compare_rule.fusion_info = mock.MagicMock()

    def test_compare_vector_no_dump_match_range(self):
        self.main.args['range'] = '0-10'
        with mock.patch.object(self.main, '_compare_by_multi_process', return_value=(0, False)), \
                mock.patch('cmp_utils.utils.sort_result_file_by_index'):
            self.main.compare_data.left_dump_info.op_name_to_file_map = {}
            ret = self.main._compare_vector()
            self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_vector_no_dump_match_select(self):
        self.main.args['select'] = '0,1,2'
        with mock.patch.object(self.main, '_compare_by_multi_process', return_value=(0, False)), \
                mock.patch('cmp_utils.utils.sort_result_file_by_index'):
            self.main.compare_data.left_dump_info.op_name_to_file_map = {}
            ret = self.main._compare_vector()
            self.assertEqual(ret, CompareError.MSACCUCMP_NO_DUMP_FILE_ERROR)

    def test_compare_vector_with_advisor(self):
        self.main.args['advisor'] = True
        with mock.patch.object(self.main, '_compare_by_multi_process', return_value=(0, True)), \
                mock.patch('cmp_utils.utils.sort_result_file_by_index'), \
                mock.patch('os.path.exists', return_value=True), \
                mock.patch.object(self.main, '_do_advisor'):
            ret = self.main._compare_vector()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)


class TestDoAdvisor(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)

    def test_do_advisor_import_warning(self):
        self.main.args['advisor'] = True
        with mock.patch('builtins.__import__', side_effect=ImportError):
            try:
                self.main._do_advisor()
            except Exception:
                self.fail("Should not raise on import error")


class TestCompareDetail(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = 'C2'
        args.max_line = None
        args.topn = 20
        args.ignore_single_op_result = False
        args.input = None
        args.output = None
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = False
        self.main = compare_vector.VectorComparison(arguments=args)
        self.main.compare_rule.fusion_info = mock.MagicMock()

    def test_compare_detail_no_fusion_file_overflow(self):
        self.main.compare_rule.fusion_json_file_path = ''
        self.main.compare_rule.quant_fusion_rule_file_path = ''
        self.main.args['overflow_detection'] = True
        with mock.patch('vector_cmp.vector_comparison.DumpDetailComparison') as mock_cls:
            mock_cls.return_value.compare.return_value = CompareError.MSACCUCMP_NONE_ERROR
            ret = self.main._compare_detail()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_no_overflow_both_dump(self):
        self.main.compare_rule.fusion_json_file_path = ''
        self.main.compare_rule.quant_fusion_rule_file_path = ''
        self.main.args['overflow_detection'] = False
        with mock.patch('vector_cmp.vector_comparison.DumpDetailComparison') as mock_cls:
            mock_cls.return_value.compare.return_value = CompareError.MSACCUCMP_NONE_ERROR
            ret = self.main._compare_detail()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_with_overflow_detection(self):
        self.main.compare_rule.fusion_json_file_path = '/home/fusion.json'
        self.main.compare_rule.fusion_info.op_name_to_fusion_op_name_map = {'C2': 'fusion_C2'}
        self.main.args['overflow_detection'] = True
        with mock.patch.object(self.main, '_check_both_dump_data', return_value=False), \
                mock.patch('vector_cmp.vector_comparison.OverflowDetection'), \
                mock.patch('vector_cmp.vector_comparison.FusionOpComparison'), \
                mock.patch('vector_cmp.vector_comparison.DetailComparison') as mock_detail:
            mock_detail.return_value.compare.return_value = CompareError.MSACCUCMP_NONE_ERROR
            ret = self.main._compare_detail()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_detail_op_not_in_fusion(self):
        self.main.compare_rule.fusion_json_file_path = '/home/fusion.json'
        self.main.compare_rule.fusion_info.op_name_to_fusion_op_name_map = {}
        self.main.args['overflow_detection'] = False
        ret = self.main._compare_detail()
        self.assertEqual(ret, CompareError.MSACCUCMP_INVALID_PARAM_ERROR)

    def test_compare_detail_no_dump_file(self):
        self.main.compare_rule.fusion_json_file_path = ''
        self.main.compare_rule.quant_fusion_rule_file_path = ''
        self.main.compare_data.left_dump_info.op_name_to_file_map = {}
        self.main.compare_data.right_dump_info.op_name_to_file_map = {}
        self.main.args['overflow_detection'] = False
        with mock.patch('vector_cmp.vector_comparison.DumpDetailComparison') as mock_cls:
            mock_cls.return_value.compare.return_value = CompareError.MSACCUCMP_NONE_ERROR
            ret = self.main._compare_detail()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)


class TestMakeTable(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        args.mapping = True
        self.main = compare_vector.VectorComparison(arguments=args)
        self.main.compare_rule.fusion_info = mock.MagicMock()
        self.main.compare_rule.fusion_info.fusion_op_name_to_op_map = {}

    def test_make_table(self):
        with mock.patch.object(self.main, '_handle_multi_process', return_value=[]), \
                mock.patch('os.fdopen'), \
                mock.patch('os.open'), \
                mock.patch('os.path.exists', return_value=True), \
                mock.patch('csv.writer'):
            ret = self.main._make_table()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_make_table_ioerror(self):
        with mock.patch.object(self.main, '_handle_multi_process', return_value=[]), \
                mock.patch('os.open', side_effect=IOError):
            with self.assertRaises(CompareError) as ctx:
                self.main._make_table()
            self.assertEqual(ctx.exception.code, CompareError.MSACCUCMP_OPEN_FILE_ERROR)

    def test_make_table_compare_error(self):
        ret_obj = mock.MagicMock()
        ret_obj.ret_code = CompareError.MSACCUCMP_DUMP_FILE_ERROR
        self.main.compare_rule.get_mapping_op_info = mock.MagicMock(return_value=([], ret_obj))
        self.main.args['mapping'] = True
        self.main.compare_rule.fusion_info.fusion_op_name_to_op_map = {}
        with mock.patch.object(self.main, '_handle_multi_process', return_value=[]), \
                mock.patch('os.fdopen'), \
                mock.patch('os.open'), \
                mock.patch('os.path.exists', return_value=True), \
                mock.patch('csv.writer'):
            ret = self.main._make_table()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)


class TestCompare(unittest.TestCase):
    def setUp(self):
        args = mock.MagicMock()
        args.fusion_rule_file = ''
        args.quant_fusion_rule_file = ''
        args.close_fusion_rule_file = ''
        args.my_dump_path = '/home/left'
        args.golden_dump_path = '/home/right'
        args.dump_version = ConstManager.OLD_DUMP_TYPE
        args.ffts = False
        args.op_name = ''
        args.csv = False
        args.output_path = '/home/result'
        args.custom_script_path = ''
        args.algorithm = 'all'
        args.algorithm_options = ''
        args.mapping = False
        args.overflow_detection = False
        args.advisor = False
        args.range = None
        args.select = None
        args.max_cmp_size = 0
        self.args = args

    def test_compare_with_detail(self):
        self.args.op_name = 'C2'
        self.args.max_line = None
        self.args.topn = 20
        self.args.ignore_single_op_result = False
        self.args.input = None
        self.args.output = None
        main = compare_vector.VectorComparison(arguments=self.args)
        fusion_info = mock.MagicMock()
        fusion_info.input_nodes = []
        fusion_info.op_list = [mock.MagicMock()]
        main.compare_rule.fusion_info = fusion_info
        with mock.patch.object(main, 'check_arguments_valid'), \
                mock.patch.object(main.compare_rule, 'parse_fusion_rule'), \
                mock.patch.object(main, '_compare_detail',
                                  return_value=CompareError.MSACCUCMP_NONE_ERROR):
            ret = main.compare()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_with_mapping(self):
        self.args.mapping = True
        main = compare_vector.VectorComparison(arguments=self.args)
        fusion_info = mock.MagicMock()
        fusion_info.input_nodes = []
        fusion_info.op_list = [mock.MagicMock()]
        main.compare_rule.fusion_info = fusion_info
        with mock.patch.object(main, 'check_arguments_valid'), \
                mock.patch.object(main.compare_rule, 'parse_fusion_rule'), \
                mock.patch.object(main, '_make_table',
                                  return_value=CompareError.MSACCUCMP_NONE_ERROR):
            ret = main.compare()
            self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)

    def test_compare_with_no_range_manager(self):
 	    main = compare_vector.VectorComparison(arguments=self.args)
 	    fusion_info = mock.MagicMock()
 	    fusion_info.input_nodes = []
 	    fusion_info.op_list = [mock.MagicMock()]
 	    main.compare_rule.fusion_info = fusion_info
 	    with mock.patch.object(main, 'check_arguments_valid'), \
 	            mock.patch.object(main.compare_rule, 'parse_fusion_rule'), \
 	            mock.patch.object(main, '_compare_vector',
                                return_value=CompareError.MSACCUCMP_NONE_ERROR):
 	        ret = main.compare()
 	        self.assertEqual(ret, CompareError.MSACCUCMP_NONE_ERROR)


if __name__ == '__main__':
    unittest.main()