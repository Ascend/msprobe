# coding=utf-8
import multiprocessing
import os
import shutil
import threading
import unittest

import pandas as pd

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.acc_compare import ModeConfig
from msprobe.core.compare.multiprocessing_compute import check_accuracy, CompareRealData, ComparisonResult, CompareInfo
from msprobe.pytorch.compare.pt_compare import read_real_data
from .test_acc_compare import generate_dump_json, generate_pt, generate_stack_json

data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
         'torch.float32', 'torch.float32', [2, 2], [2, 2], True, True,
         '', '', '', '', '', '',
         1, 1, 1, 1, 1, 1, 1, 1,
         True, 'None', '', ['-1', '-1'],
         [-1, -1]]]
o_data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
           'torch.float32', 'torch.float32', [2, 2], [2, 2], True, True,
           'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported',
           1, 1, 1, 1, 1, 1, 1, 1,
           True, 'None', 'NPU does not have data file.', ['-1', '-1'],
           [-1, -1]]]
columns = CompareConst.COMPARE_RESULT_HEADER + [CompareConst.DATA_NAME, CompareConst.DIRTY_VALID_LEN]
result_df = pd.DataFrame(data, columns=columns)
o_result = pd.DataFrame(o_data, columns=columns)
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_cmp_multiprocessing_compute')
base_dir3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_acc_compare_data3')
pt_dir = os.path.join(base_dir3, f'dump_data_dir')


class TestUtilsMethods(unittest.TestCase):

    def test_check_accuracy(self):
        max_abs_err = ''

        cos_1 = CompareConst.SHAPE_UNMATCH
        result_1 = check_accuracy(cos_1, max_abs_err)
        self.assertEqual(result_1, CompareConst.ACCURACY_CHECK_UNMATCH)

        cos_2 = CompareConst.NONE
        result_2 = check_accuracy(cos_2, max_abs_err)
        self.assertEqual(result_2, CompareConst.NONE)

        cos_3 = 'N/A'
        result_3 = check_accuracy(cos_3, max_abs_err)
        self.assertEqual(result_3, CompareConst.ACCURACY_CHECK_NO)

        cos_4 = ''
        result_4 = check_accuracy(cos_4, max_abs_err)
        self.assertEqual(result_4, CompareConst.NONE)

        cos_5 = 0.95
        max_abs_err = 0.002
        result_5 = check_accuracy(cos_5, max_abs_err)
        self.assertEqual(result_5, CompareConst.ACCURACY_CHECK_NO)

        cos_6 = 0.85
        max_abs_err = 2
        result_6 = check_accuracy(cos_6, max_abs_err)
        self.assertEqual(result_6, CompareConst.ACCURACY_CHECK_NO)

        cos_7 = 0.95
        max_abs_err = 0.001
        result_7 = check_accuracy(cos_7, max_abs_err)
        self.assertEqual(result_7, CompareConst.ACCURACY_CHECK_YES)


class TestCompareRealData(unittest.TestCase):

    def setUp(self):
        self.result_df = pd.DataFrame([['']*8]*2, columns=[
            CompareConst.COSINE, CompareConst.EUC_DIST, CompareConst.MAX_ABS_ERR, CompareConst.MAX_RELATIVE_ERR,
            CompareConst.ONE_THOUSANDTH_ERR_RATIO, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO,
            CompareConst.ACCURACY, CompareConst.ERROR_MESSAGE
        ])
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(base_dir3, mode=0o750, exist_ok=True)
        os.makedirs(pt_dir, mode=0o750, exist_ok=True)
        self.lock = threading.Lock()

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        if os.path.exists(pt_dir):
            shutil.rmtree(pt_dir)
        if os.path.exists(base_dir3):
            shutil.rmtree(base_dir3)

    def test_read_dump_data(self):
        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        # normal
        result = compare_real_data.read_dump_data(result_df)
        self.assertEqual(result, {'Functional.linear.0.forward.input.0Functional.linear.0.forward.input.0': ['-1', '-1']})

        # index error
        with self.assertRaises(CompareException) as context:
            result = compare_real_data.read_dump_data(pd.DataFrame())
        self.assertEqual(context.exception.code, CompareException.INVALID_KEY_ERROR)

    def test_save_cmp_result_success(self):
        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        comparison_result = ComparisonResult(
            cos_result=[0.99, 0.98],
            max_err_result=[0.01, 0.02],
            max_relative_err_result=[0.001, 0.002],
            euc_dist_result=[0.5, 0.49],
            one_thousand_err_ratio_result=[0.1, 0.2],
            five_thousand_err_ratio_result=[0.05, 0.1],
            err_msgs=['', 'Error in comparison']
        )
        offset = 0
        updated_df = compare_real_data._save_cmp_result(offset, comparison_result, self.result_df, self.lock)

        self.assertEqual(updated_df.loc[0, CompareConst.COSINE], 0.99)
        self.assertEqual(updated_df.loc[1, CompareConst.COSINE], 0.98)
        self.assertEqual(updated_df.loc[1, CompareConst.ERROR_MESSAGE], 'Error in comparison')

    def test_save_cmp_result_index_error(self):
        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        comparison_result = ComparisonResult(
            cos_result=[0.99],
            max_err_result=[],
            max_relative_err_result=[0.001],
            euc_dist_result=[0.5],
            one_thousand_err_ratio_result=[0.1],
            five_thousand_err_ratio_result=[0.05],
            err_msgs=['']
        )
        with self.assertRaises(CompareException) as context:
            compare_real_data._save_cmp_result(0, comparison_result, self.result_df, self.lock)
        self.assertEqual(context.exception.code, CompareException.INDEX_OUT_OF_BOUNDS_ERROR)

    def test_compare_by_op_bench_normal(self):
        npu_op_name = 'Functional.linear.0.forward.input.0'
        bench_op_name = 'Functional.linear.0.forward.input.0'

        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        pt_name = '-1'
        npu_info = CompareInfo(npu_op_name, pt_name, -1)
        bench_info = CompareInfo(bench_op_name, pt_name, -1)
        input_param = {'npu_dump_data_dir': base_dir, 'bench_dump_data_dir': base_dir}
        result = compare_real_data.compare_by_op(npu_info, bench_info, input_param)
        self.assertEqual(result, ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported',
                                  'unsupported', 'NPU does not have data file.'])

        pt_name = 'Functional.linear.0.forward.input.0.pt'
        npu_info = CompareInfo(npu_op_name, pt_name, -1)
        bench_info = CompareInfo(bench_op_name, pt_name, -1)
        input_param = {'npu_dump_data_dir': base_dir, 'bench_dump_data_dir': base_dir}
        result = compare_real_data.compare_by_op(npu_info, bench_info, input_param)
        self.assertEqual(result, ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported',
                                  'unsupported', "Dump file: Functional.linear.0.forward.input.0.pt or Functional.linear.0.forward.input.0.pt not found or read failed."])

        generate_pt(base_dir)
        result = compare_real_data.compare_by_op(npu_info, bench_info, input_param)
        self.assertEqual(result, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, ''])

    def test_compare_by_op_bench_no_npu_real_data(self):
        npu_op_name = 'Functional.linear.0.forward.input.0'
        bench_op_name = 'N/A'
        input_param = {}

        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)
        npu_info = CompareInfo(npu_op_name, -1, -1)
        bench_info = CompareInfo(bench_op_name, -1, -1)
        result = compare_real_data.compare_by_op(npu_info, bench_info, input_param)
        self.assertEqual(result, ['unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported',
                                  'unsupported', 'NPU does not have data file.'])

    def test_compare_ops(self):
        generate_dump_json(base_dir3)
        generate_stack_json(base_dir3)
        generate_pt(pt_dir)
        dump_path = os.path.join(base_dir3, 'dump.json')
        stack_path = os.path.join(base_dir3, 'stack.json')
        input_param = {'npu_json_path': dump_path, 'bench_json_path': dump_path, 'stack_json_path': stack_path,
                       'is_print_compare_log': True, 'npu_dump_data_dir': pt_dir, 'bench_dump_data_dir': pt_dir}
        dump_path_dict = {'Functional.linear.0.forward.input.0Functional.linear.0.forward.input.0':
                              ['Functional.linear.0.forward.input.0.pt',
                               'Functional.linear.0.forward.input.0.pt']}
        result_df = pd.DataFrame(
            [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0', '', [21, 21]]],
            columns=[CompareConst.NPU_NAME, CompareConst.BENCH_NAME, CompareConst.ERROR_MESSAGE, CompareConst.DIRTY_VALID_LEN]
        )

        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        updated_df = compare_real_data.compare_ops(idx=0, dump_path_dict=dump_path_dict, result_df=result_df,
                                                   lock=self.lock, input_param=input_param)

        self.assertEqual(updated_df.loc[0, CompareConst.COSINE], 1.0)
        self.assertEqual(updated_df.loc[0, CompareConst.MAX_ABS_ERR], 0)

    def test_do_multi_process(self):
        data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                 'torch.float32', 'torch.float32', [2, 2], [2, 2], True, True,
                 '', '', '', '', '', '', 1, 1, 1, 1, 1, 1, 1, 1, True, 'None', '', ['-1', '-1'],
                 [21, 21]
                 ]]
        o_data = [['Functional.linear.0.forward.input.0', 'Functional.linear.0.forward.input.0',
                   'torch.float32', 'torch.float32', [2, 2], [2, 2], True, True,
                   'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported', 'unsupported',
                   1, 1, 1, 1, 1, 1, 1, 1, True, 'None', 'NPU does not have data file.', ['-1', '-1'],
                   [21, 21]
                   ]]
        columns = CompareConst.COMPARE_RESULT_HEADER + [CompareConst.DATA_NAME, CompareConst.DIRTY_VALID_LEN]
        result_df = pd.DataFrame(data, columns=columns)
        o_result = pd.DataFrame(o_data, columns=columns)
        generate_dump_json(base_dir)
        input_param = {'bench_json_path': os.path.join(base_dir, 'dump.json')}

        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        result = compare_real_data.do_multi_process(input_param, result_df)
        self.assertTrue(result.equals(o_result))

    def test_handle_multi_process(self):
        file_reader = read_real_data
        mode_config = ModeConfig(dump_mode=Const.ALL)
        cross_frame = False
        compare_real_data = CompareRealData(file_reader, mode_config, cross_frame)

        func = compare_real_data.compare_ops
        generate_dump_json(base_dir)
        input_param = {'bench_json_path': os.path.join(base_dir, 'dump.json')}
        lock = multiprocessing.Manager().RLock()
        result = compare_real_data._handle_multi_process(func, input_param, result_df, lock)
        self.assertTrue(result.equals(o_result))


if __name__ == '__main__':
    unittest.main()
