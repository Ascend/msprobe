import unittest
from unittest.mock import patch, MagicMock, Mock

import pandas as pd

from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import *
from msprobe.core.common.exceptions import FileCheckException
from msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare import _api_precision_compare_command, register_compare_func
from msprobe.core.common.const import CompareConst
from msprobe.pytorch.api_accuracy_checker.compare.compare_input import PrecisionCompareInput


class Args:
    def __init__(self, npu_csv_path=None, gpu_csv_path=None, out_path=None):
        self.npu_csv_path = npu_csv_path
        self.gpu_csv_path = gpu_csv_path
        self.out_path = out_path


class TestFileCheck(unittest.TestCase):
    def setUp(self):
        src_path = 'temp_path'
        create_directory(src_path)
        dst_path = 'compare_soft_link'
        os.symlink(src_path, dst_path)
        self.hard_path = os.path.abspath(src_path)
        self.soft_path = os.path.abspath(dst_path)
        csv_path = os.path.join(self.hard_path, 'test.csv')
        csv_data = [['1', '2', '3']]
        write_csv(csv_data, csv_path)
        self.hard_csv_path = os.path.abspath(csv_path)
        soft_csv_path = 'soft.csv'
        os.symlink(csv_path, soft_csv_path)
        self.soft_csv_path = os.path.abspath(soft_csv_path)
        self.empty_path = "empty_path"

    def tearDown(self):
        os.unlink(self.hard_csv_path)
        os.unlink(self.soft_csv_path)
        os.unlink(self.soft_path)
        for file in os.listdir(self.hard_path):
            os.remove(os.path.join(self.hard_path, file))
        os.rmdir(self.hard_path)

    def test_npu_path_soft_link_check(self):
        args = Args(npu_csv_path=self.soft_csv_path, gpu_csv_path=self.hard_csv_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_gpu_path_soft_link_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.soft_csv_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)

    def test_out_path_soft_link_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.hard_csv_path, out_path=self.soft_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.SOFT_LINK_ERROR)
    
    def test_npu_path_empty_check(self):
        args = Args(npu_csv_path=self.empty_path, gpu_csv_path=self.hard_csv_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_gpu_path_empty_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.empty_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_out_path_empty_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.hard_csv_path, out_path=self.empty_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.ILLEGAL_PATH_ERROR)
    
    def test_npu_path_invalid_type_check(self):
        args = Args(npu_csv_path=123, gpu_csv_path=self.hard_csv_path, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)
    
    def test_gpu_path_invalid_type_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=123, out_path=self.hard_path)
        
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)
    
    def test_out_path_invalid_type_check(self):
        args = Args(npu_csv_path=self.hard_csv_path, gpu_csv_path=self.hard_csv_path, out_path=123)
        with self.assertRaises(Exception) as context:
            _api_precision_compare_command(args)
            self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)


class TestApiPrecisionCompare(unittest.TestCase):

    def setUp(self):
        # Setup paths and mock data
        self.config = CompareConfig(
            npu_csv_path='mock_npu.csv',
            gpu_csv_path='mock_gpu.csv',
            result_csv_path='result.csv',
            details_csv_path='details.csv'
        )

        self.npu_data = pd.DataFrame({
            'API_NAME': ['torch.abs.0.forward.output.0', 'torch.matmul.0.forward.output.0'],
            'DEVICE_DTYPE': ['float32', 'float32'],
            'ERROR_RATE': ['0', '0.1'],
            'SMALL_VALUE_ERROR_RATE': ['0.01', '0.02'],
            'RMSE': ['0.1', '0.2'],
            'MAX_REL_ERR': ['0.1', '0.2'],
            'MEAN_REL_ERR': ['0.1', '0.2'],
            'EB': ['0.1', '0.2']
        })

        self.gpu_data = pd.DataFrame({
            'API_NAME': ['torch.abs.0.forward.output.0', 'torch.matmul.0.forward.output.0'],
            'DEVICE_DTYPE': ['float32', 'float32'],
            'ERROR_RATE': ['0', '0'],
            'SMALL_VALUE_ERROR_RATE': ['0.01', '0.01'],
            'RMSE': ['0.1', '0.1'],
            'MAX_REL_ERR': ['0.1', '0.1'],
            'MEAN_REL_ERR': ['0.1', '0.1'],
            'EB': ['0.1', '0.1']
        })
        
        self.test_data = {
            'API Name': ['torch.abs.0.forward.output.0'],
            'Shape': ['(2,3)'],
            'DEVICE Dtype': ['float32'],
            '小值域错误占比': ['0'],
            '均方根误差': ['0'],
            '相对误差最大值': ['0'],
            '相对误差平均值': ['0'],
            '误差均衡性': ['0'],
            '二进制一致错误率': ['0'],
            'inf/nan错误率': ['0'],
            '相对误差错误率': ['0'],
            '绝对误差错误率': ['0'],
            'ULP误差平均值': ['0'],
            'ULP误差大于阈值占比': ['0'],
            '双千指标': ['0.999'],
            'Message': ['error']
        }
        
        self.test_data_2 = {
            'API Name': ['torch.abs.0.forward.output.0', 'torch.matmul.0.forward.output.0', 
                         'torch.matmul.0.backward.output.0', 'torch.add.0.forward.output.0'],
            'DEVICE Dtype': ['torch.float32', 'torch.float32', 'torch.float32', 'torch.float64'],
            '小值域错误占比': ['0', '0', '0', '0'],
            '均方根误差': ['0', '0', '0', '0'],
            '相对误差最大值': ['0', '0', '0', '0'],
            '相对误差平均值': ['0', '0', '0', '0'],
            '误差均衡性': ['0', '0', '0', '0'],
            '二进制一致错误率': ['0', '0', '0', '0'],
            'inf/nan错误率': ['0', '0', '0', '0'],
            '相对误差错误率': ['0', '0', '0', '0'],
            '绝对误差错误率': ['0', '0', '0', '0'],
            'ULP误差平均值': ['0', '0', '0', '0'],
            'ULP误差大于阈值占比': ['0', '0', '0', '0'],
            '双千指标': ['0', '0', '0', '0'],
            'Message': ['error', 'pass', 'error', 'pass']
        }
        
        self.api_name = "test_api"
        self.npu_precision = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0', ApiPrecisionCompareColumn.REL_ERR_RATIO: '0',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0', ApiPrecisionCompareColumn.ERROR_RATE: '0', 
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE: '0.01', ApiPrecisionCompareColumn.RMSE: '0.1', 
            ApiPrecisionCompareColumn.MAX_REL_ERR: '0.1', ApiPrecisionCompareColumn.MEAN_REL_ERR: '0.1', 
            ApiPrecisionCompareColumn.EB: '0.1', ApiPrecisionCompareColumn.MEAN_ULP_ERR: '0.1', 
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION: '0.05', ApiPrecisionCompareColumn.SHAPE: '(2,3)'
        }
        self.gpu_precision = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0', ApiPrecisionCompareColumn.REL_ERR_RATIO: '0',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0', ApiPrecisionCompareColumn.ERROR_RATE: '0', 
            ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE: '0.01', ApiPrecisionCompareColumn.RMSE: '0.1', 
            ApiPrecisionCompareColumn.MAX_REL_ERR: '0.1', ApiPrecisionCompareColumn.MEAN_REL_ERR: '0.1', 
            ApiPrecisionCompareColumn.EB: '0.1', ApiPrecisionCompareColumn.MEAN_ULP_ERR: '0.2', 
            ApiPrecisionCompareColumn.ULP_ERR_PROPORTION: '0.06', ApiPrecisionCompareColumn.SHAPE: '(2,3)'
        }

        # 创建 DataFrame
        self.npu_data = pd.DataFrame(self.test_data)
        self.gpu_data = pd.DataFrame(self.test_data)
        
        self.npu_data_2 = pd.DataFrame(self.test_data_2)
        self.gpu_data_2 = pd.DataFrame(self.test_data_2)
    
        # 使用第一行数据作为测试用例
        self.row_npu = self.npu_data.iloc[0]
        self.row_gpu = self.gpu_data.iloc[0]

        # 添加 compare_column
        self.compare_column = MagicMock()
        self.compare_column.api_name = MagicMock(return_value="test_api")

        self.registry = register_compare_func()
        
        self.dtype = 'torch.float16'

        self.input_data = PrecisionCompareInput(self.row_npu, self.row_gpu, self.dtype, self.compare_column)

    def test_check_csv_columns(self):
        with self.assertRaises(Exception):
            check_csv_columns([], 'test_csv')

    def test_check_error_rate(self):
        result = check_error_rate('0')
        self.assertEqual(result, CompareConst.PASS)

        result = check_error_rate('0.1')
        self.assertEqual(result, CompareConst.ERROR)

    def test_get_api_checker_result(self):
        result = get_api_checker_result([CompareConst.PASS, CompareConst.ERROR])
        self.assertEqual(result, CompareConst.ERROR)

        result = get_api_checker_result([CompareConst.PASS, CompareConst.PASS])
        self.assertEqual(result, CompareConst.PASS)
    
    def test_print_test_success_success(self):
        with patch('msprobe.pytorch.common.log.logger.info') as mock_info:
            api_full_name = "test_api"
            forward_result = CompareConst.PASS
            backward_result = CompareConst.PASS
            print_test_success(api_full_name, forward_result, backward_result)
            mock_info.assert_called_once_with(f"running api_full_name {api_full_name} compare, "
                                              f"is_fwd_success: True, "
                                              f"is_bwd_success: True")

    def test_print_test_success_forward_failure(self):
        with patch('msprobe.pytorch.common.log.logger.info') as mock_info:
            api_full_name = "test_api"
            forward_result = CompareConst.ERROR
            backward_result = CompareConst.PASS
            print_test_success(api_full_name, forward_result, backward_result)
            mock_info.assert_called_once_with(f"running api_full_name {api_full_name} compare, "
                                              f"is_fwd_success: False, "
                                              f"is_bwd_success: True")

    def test_print_test_success_backward_failure(self):
        with patch('msprobe.pytorch.common.log.logger.info') as mock_info:
            api_full_name = "test_api"
            forward_result = CompareConst.PASS
            backward_result = CompareConst.ERROR
            print_test_success(api_full_name, forward_result, backward_result)
            mock_info.assert_called_once_with(f"running api_full_name {api_full_name} compare, "
                                              f"is_fwd_success: True, "
                                              f"is_bwd_success: False")

        result = get_api_checker_result([])
        self.assertEqual(result, CompareConst.SPACE)

        result = get_api_checker_result([CompareConst.SKIP])
        self.assertEqual(result, CompareConst.SKIP)

    def test_write_detail_csv(self):
        content = [1, 2, 3]
        save_path = "path/"
        create_directory(save_path)
        details_csv_path = os.path.join(save_path, "details.csv")
        write_detail_csv(content, details_csv_path)
        self.assertTrue(os.path.exists(details_csv_path))
        for filename in os.listdir(save_path):
            os.remove(os.path.join(save_path, filename))
        os.rmdir(save_path)

    def test_get_absolute_threshold_result_pass(self):
        row_npu = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0',
            ApiPrecisionCompareColumn.REL_ERR_RATIO: '0',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0'
        }
        result = get_absolute_threshold_result(row_npu)
        self.assertEqual(result['inf_nan_error_ratio'], 0.0)
        self.assertEqual(result['inf_nan_result'], CompareConst.PASS)
        self.assertEqual(result['rel_err_ratio'], 0.0)
        self.assertEqual(result['rel_err_result'], CompareConst.PASS)
        self.assertEqual(result['abs_err_ratio'], 0.0)
        self.assertEqual(result['abs_err_result'], CompareConst.PASS)
        self.assertEqual(result['absolute_threshold_result'], CompareConst.PASS)

    def test_get_absolute_threshold_result_error(self):
        row_npu = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: '0',
            ApiPrecisionCompareColumn.REL_ERR_RATIO: '0.1',
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: '0'
        }
        result = get_absolute_threshold_result(row_npu)
        self.assertEqual(result['inf_nan_error_ratio'], 0.0)
        self.assertEqual(result['inf_nan_result'], CompareConst.PASS)
        self.assertEqual(result['rel_err_ratio'], 0.1)
        self.assertEqual(result['rel_err_result'], CompareConst.ERROR)
        self.assertEqual(result['abs_err_ratio'], 0.0)
        self.assertEqual(result['abs_err_result'], CompareConst.PASS)
        self.assertEqual(result['absolute_threshold_result'], CompareConst.ERROR)

    def test_api_precision_compare(self):
        # 准备测试目录和文件
        base_path = 'test_compare_tmp'
        os.makedirs(base_path, exist_ok=True)
        
        # 创建测试用的CSV文件
        npu_csv = os.path.join(base_path, 'npu.csv')
        gpu_csv = os.path.join(base_path, 'gpu.csv')
        result_csv = os.path.join(base_path, 'result.csv')
        details_csv = os.path.join(base_path, 'details.csv')
        
        # 将测试数据写入CSV文件
        df = pd.DataFrame(self.test_data)
        df.to_csv(npu_csv, index=False)
        df.to_csv(gpu_csv, index=False)
        
        try:
            # 执比较操作
            config = CompareConfig(npu_csv, gpu_csv, result_csv, details_csv)
            api_precision_compare(config)
            
            # 验证结果文件是否生成
            self.assertTrue(os.path.exists(result_csv))
            self.assertTrue(os.path.exists(details_csv))
            
            # 读取并验证结果
            result_df = pd.read_csv(result_csv)
            self.assertFalse(result_df.empty)
            
            details_df = pd.read_csv(details_csv)
            self.assertFalse(details_df.empty)
            
        finally:
            # 清理测试文件
            for file_path in [npu_csv, gpu_csv, result_csv, details_csv]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            if os.path.exists(base_path):
                os.rmdir(base_path)

               
    def test_skip_due_to_empty_output(self):
        self.row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] = ' '
        api_name = "abs"
        result = get_api_status(self.row_npu, self.row_gpu, api_name, self.compare_column, self.registry)
        self.assertEqual(result, CompareConst.SKIP)

    def test_thousandth_standard(self):
        self.row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] = 'torch.float16'
        api_name = "conv2d"
        result = get_api_status(self.row_npu, self.row_gpu, api_name, self.compare_column, self.registry)
        self.assertEqual(result, CompareConst.PASS)

    def test_binary_consistency(self):
        self.row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] = 'torch.float16'
        api_name = "abs"
        result = get_api_status(self.row_npu, self.row_gpu, api_name, self.compare_column, self.registry)
        self.assertEqual(result, CompareConst.PASS)

    def test_absolute_threshold(self):
        self.row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] = 'torch.float16'
        api_name = "mul"
        result = get_api_status(self.row_npu, self.row_gpu, api_name, self.compare_column, self.registry)
        self.assertEqual(result, CompareConst.PASS)

    def test_ulp_standard(self):
        self.row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] = "torch.float16"
        api_name = "matmul"
        result = get_api_status(self.row_npu, self.row_gpu, api_name, self.compare_column, self.registry)
        self.assertEqual(result, CompareConst.PASS)

    def test_benchmark_compare(self):
        self.row_npu[ApiPrecisionCompareColumn.DEVICE_DTYPE] = "torch.float16"
        api_name = "mean"
        result = get_api_status(self.row_npu, self.row_gpu, api_name, self.compare_column, self.registry)
        self.assertEqual(result, CompareConst.PASS)

    def test_record_binary_consistency_result_pass(self):
        self.row_npu[ApiPrecisionCompareColumn.ERROR_RATE] = "0.0"
        self.compare_column.ERROR = CompareConst.PASS
        
        result = record_binary_consistency_result(self.input_data)
        
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(self.compare_column.compare_algorithm, "二进制一致法")

    def test_record_binary_consistency_result_error(self):
        self.row_npu[ApiPrecisionCompareColumn.ERROR_RATE] = "2.0"
        self.compare_column.ERROR = CompareConst.ERROR
        
        input_data = PrecisionCompareInput(self.row_npu, self.row_gpu, self.dtype, self.compare_column)
        result = record_binary_consistency_result(input_data)
        
        self.assertEqual(result, CompareConst.ERROR)
        self.assertIn("ERROR: 二进制一致错误率超过阈值\n", self.compare_column.compare_message)

    def test_record_absolute_threshold_result(self):
        row_npu = {
            ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO: "0.0",
            ApiPrecisionCompareColumn.REL_ERR_RATIO: "0.0",
            ApiPrecisionCompareColumn.ABS_ERR_RATIO: "0.0"
        }
        compare_column = MagicMock()
        
        input_data = PrecisionCompareInput(row_npu, self.row_gpu, self.dtype, compare_column)
        result = record_absolute_threshold_result(input_data)
        
        self.assertEqual(result, CompareConst.PASS)

    def test_record_benchmark_compare_result(self):
        bs = MagicMock()
        bs.get_result = MagicMock()
        bs.small_value_err_status = CompareConst.PASS
        bs.final_result = CompareConst.PASS
        compare_column = MagicMock()
        
        result = record_benchmark_compare_result(self.input_data)
        
        self.assertEqual(result, CompareConst.PASS)

    def test_record_ulp_compare_result(self):
        us = MagicMock()
        us.get_result = MagicMock()
        us.ulp_err_status = CompareConst.PASS
        compare_column = MagicMock()
        
        result = record_ulp_compare_result(self.input_data)
        
        self.assertEqual(result, CompareConst.PASS)

    def test_record_thousandth_threshold_result(self):
        self.row_npu = self.row_npu.astype(object)
        self.row_npu[ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH] = 0.999
        self.compare_column.rel_err_thousandth = 0.999
        self.compare_column.rel_err_thousandth_status = CompareConst.PASS
        
        input_data = PrecisionCompareInput(self.row_npu, self.row_gpu, self.dtype, self.compare_column)
        result = record_thousandth_threshold_result(input_data)
        
        self.assertEqual(result, CompareConst.PASS)
        self.assertEqual(self.compare_column.compare_message, "")

    @patch('msprobe.pytorch.api_accuracy_checker.compare.api_precision_compare.write_detail_csv')
    def test_analyse_csv(self, mock_write_detail_csv):
        analyse_csv(self.npu_data_2, self.gpu_data_2, self.config)
        mock_write_detail_csv.assert_called()

if __name__ == '__main__':
    unittest.main()
