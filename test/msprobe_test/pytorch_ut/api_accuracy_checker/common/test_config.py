import unittest
import os
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.common.config import Config, CheckerConfig, msCheckerConfig


class TestUtConfig():
    def __init__(self):
        self.white_list = ['api1', 'api2']
        self.black_list =  ['api3']
        self.error_data_path = '/path/to/error_data'


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        self.yaml_file = os.path.join(self.input_dir, "config.yaml")
        self.cfg = Config(self.yaml_file)
        self.task_config = TestUtConfig()

    def test_validate_valid_data(self):
        for key, val in self.cfg.config.items():
            validated_type = self.cfg.validate(key, val)
            self.assertEqual(validated_type, val)

    def test_validate_should_raise_when_invalid_type(self):
        with self.assertRaises(ValueError):
            self.cfg.validate('error_data_path', True)

    def test_validate_should_raise_when_invalid_key(self):
        with self.assertRaises(ValueError):
            self.cfg.validate('invalid_key', 'mock_value')

    def test_validate_precision(self):
        self.assertEqual(self.cfg.validate('precision', 1), 1)

        with self.assertRaises(ValueError):
            self.cfg.validate('precision', -1)
            
        with self.assertRaises(ValueError):
            self.cfg.validate('precision', True)

    def test_validate_white_list(self):
        validate_white_list = ['conv1d', 'max_pool1d', 'dropout', '__add__']
        self.assertEqual(self.cfg.validate('white_list', validate_white_list), validate_white_list)

        with self.assertRaises(Exception):
            self.cfg.validate('white_list', ['invalid_api1', 'invalid_api2'])
            
    def test_validate_quantization_api_list(self):
        validate_quantization_api_list = ['npu_rms_norm_dynamic_quant', 'npu_dynamic_quant', 'npu_quant_lightning_indexer']
        self.assertEqual(self.cfg.validate('quantization_api_list', validate_quantization_api_list), validate_quantization_api_list)

        with self.assertRaises(Exception):
            self.cfg.validate('quantization_api_list', ['invalid_api1', 'invalid_api2'])

    def test_CheckerConfig_init_with_defaults(self):
        checker_config = CheckerConfig()
        self.assertEqual(checker_config.white_list, msCheckerConfig.white_list)
        self.assertEqual(checker_config.black_list, msCheckerConfig.black_list)
        self.assertEqual(checker_config.error_data_path, msCheckerConfig.error_data_path)


    def test_init_with_task_config(self):
        checker_config = CheckerConfig(self.task_config)
        self.assertEqual(checker_config.white_list, self.task_config.white_list)
        self.assertEqual(checker_config.black_list, self.task_config.black_list)
        self.assertEqual(checker_config.error_data_path, self.task_config.error_data_path)


    def test_load_config(self):
        checker_config = CheckerConfig()
        checker_config.load_config(self.task_config)


    def test_get_acc_check_config(self):
        forward_content = {'api1': 'data1', 'api2': 'data2'}
        backward_content = {'api3': 'data3'}
        result_csv_path = '/path/to/result.csv'
        details_csv_path = '/path/to/details.csv'
        save_error_data = True
        api_result_csv_path = '/path/to/api_result.csv'
        real_data_path = '/path/to/real_data'
        error_data_path = '/path/to/error_data'

        checker_config = CheckerConfig()
        
        config_params = {
            'forward_content': forward_content,
            'backward_content': backward_content,
            'result_csv_path': result_csv_path,
            'details_csv_path': details_csv_path,
            'save_error_data': save_error_data,
            'is_continue_acc_check': api_result_csv_path,
            'real_data_path': real_data_path,
            'error_data_path': error_data_path
        }

        acc_check_config = checker_config.get_acc_check_config(**config_params)

        self.assertEqual(acc_check_config.forward_content, forward_content)
        self.assertEqual(acc_check_config.backward_content, backward_content)
        self.assertEqual(acc_check_config.result_csv_path, result_csv_path)
        self.assertEqual(acc_check_config.details_csv_path, details_csv_path)
        self.assertEqual(acc_check_config.save_error_data, save_error_data)
        self.assertEqual(acc_check_config.is_continue_acc_check, api_result_csv_path)
        self.assertEqual(acc_check_config.real_data_path, real_data_path)
        self.assertEqual(acc_check_config.error_data_path, error_data_path)
