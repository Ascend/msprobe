# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import json
import csv
import unittest
from unittest.mock import patch

from msprobe.pytorch.api_accuracy_checker.common.utils import *
from msprobe.core.common.file_utils import write_csv


class TestUtils(unittest.TestCase):
    
    def setUp(self):
        # 创建一个临时目录用于保存测试文件
        self.save_path = "temp_save_path"
        create_directory(self.save_path)
        self.processor = UtDataProcessor(self.save_path)
        
    def tearDown(self):
        # 测试完成后删除临时目录
        for filename in os.listdir(self.save_path):
            os.remove(os.path.join(self.save_path, filename))
        os.rmdir(self.save_path)
        
    def test_save_tensors_in_element(self):
        # 测试保存张量
        api_name = "test_api"
        tensor = torch.randn(10, 10)
        self.processor.save_tensors_in_element(api_name, tensor)
        file_path = os.path.join(self.save_path, f'{api_name}.0.pt')
        self.assertTrue(os.path.exists(file_path))

    @patch.object(Const, "MAX_DEPTH", 50)
    def test_recursion_limit_error(self):
        tensor = torch.randn(10, 10)
        with self.assertRaises(DumpException) as context:
            self.processor._save_recursive("test_api", [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor,                                                         [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor,
                                                        [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor,
                                                        [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor,
                                                        [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor, [tensor,
                                                        [tensor, [tensor, [tensor, [tensor
                                                                                    ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]], 0)
        self.assertTrue(isinstance(context.exception, DumpException))
        self.assertEqual(context.exception.code, DumpException.RECURSION_LIMIT_ERROR)

    def test_save_recursive_non_tensor_types(self):
        api_name = "test_api"
        non_tensor = [None, [True, [42, [3.14, ["test", [slice(0, 10)]]]]]]
        self.processor.save_tensors_in_element(api_name, non_tensor)
        self.assertEqual(self.processor.index, 6)

    def test_save_recursive_list_or_tuple(self):
        api_name = "test_api"
        tensor = torch.randn(10, 10)
        list_element = [tensor, [tensor, [tensor, [tensor, [tensor, [tensor]]]]]]
        self.processor.save_tensors_in_element(api_name, list_element)
        self.assertEqual(self.processor.index, 6)

    def test_save_recursive_dict(self):
        api_name = "test_api"
        dict_element = {i: torch.randn(10, 10) for i in range(5)}
        self.processor.save_tensors_in_element(api_name, dict_element)
        self.assertEqual(self.processor.index, 5)

    def test_check_need_convert(self):
        self.assertEqual(check_need_convert('cross_entropy'), 'int32_to_int64')
        self.assertIsNone(check_need_convert('linear'))

    def test_check_object_type(self):
        try:
            check_object_type(123, int)
        except Exception as e:
            self.fail(f"check_object_type raised exception {e}")
    
    def test_check_object_type_error(self):
        with self.assertRaises(CompareException) as context:
            check_object_type(123, str)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)

    def test_api_info_preprocess_no_conversion_needed(self):
        api_name = 'linear'
        original_api_info = {'key': 'value'}
        convert_type, processed_api_info = api_info_preprocess(api_name, original_api_info.copy())
        self.assertIsNone(convert_type)
        self.assertEqual(original_api_info, processed_api_info)

    def test_api_info_preprocess_cross_entropy_positive(self):
        api_name = 'cross_entropy'
        api_info = {'input_args': [{'Name': 'logit'}, {'Name': 'labels', 'Min': 1}]}
        convert_type, processed_api_info = api_info_preprocess(api_name, api_info.copy())
        self.assertEqual(convert_type, 'int32_to_int64')
        self.assertEqual(processed_api_info, api_info)
        
    def test_cross_entropy(self):
        api_info = {'input_args': [{'Name': 'logit'}, {'Name': 'labels', 'Min': -1}]}
        processed_api_info = cross_entropy_process(api_info.copy())
        self.assertEqual(processed_api_info, {'input_args': [{'Name': 'logit'}, {'Name': 'labels', 'Min': 0}]})
    
    def test_initialize_save_path(self):
        save_path = 'initialize_save_path'
        dir_name = 'test_dir'
        data_path = initialize_save_path(save_path, dir_name)
        self.assertTrue(os.path.exists(data_path))
        os.rmdir(data_path)

    def test_get_full_data_path(self):
        real_data_path = 'get_full_data_path'
        real_data_path = os.path.realpath(real_data_path)
        data_path = 'test_data'
        full_data_path = get_full_data_path(data_path, real_data_path)
        self.assertEqual(full_data_path, os.path.join(real_data_path, data_path))
    
    def test_get_full_data_path_with_empty_data_path(self):
        data_path = None
        real_data_path = None
        full_data_path = get_full_data_path(data_path, real_data_path)
        self.assertEqual(full_data_path, None)
    
    def test_extract_basic_api_segments(self):
        api_full_name = 'torch.matmul.0'
        api_type, api_name = extract_basic_api_segments(api_full_name)
        self.assertEqual(api_type, 'torch')
        self.assertEqual(api_name, 'matmul')
        
        api_full_name = 'torch.linalg.vector_norm.0'
        api_type, api_name = extract_basic_api_segments(api_full_name)
        self.assertEqual(api_type, 'torch')
        self.assertEqual(api_name, 'linalg.vector_norm')
        
        api_full_name = 'torch.nn.linalg.vector_norm.0'
        api_type, api_name = extract_basic_api_segments(api_full_name)
        self.assertEqual(api_type, None)
        self.assertEqual(api_name, None)
        
    def test_extract_detailed_api_segments(self):
        api_full_name = 'torch.matmul.0.forward.output.0'
        api_name, full_api_name, direction_status = extract_detailed_api_segments(api_full_name)
        self.assertEqual(api_name, 'matmul')
        self.assertEqual(full_api_name, 'torch.matmul.0')
        self.assertEqual(direction_status, 'forward')
        
        api_full_name = 'torch.linalg.vector_norm.0.backward.output.0'
        api_name, full_api_name, direction_status = extract_detailed_api_segments(api_full_name)
        self.assertEqual(api_name, 'linalg.vector_norm')
        self.assertEqual(full_api_name, 'torch.linalg.vector_norm.0')
        self.assertEqual(direction_status, 'backward')
        
        api_full_name = 'torch.nn.functional.linear.0.input.0.1'
        api_name, full_api_name, direction_status = extract_detailed_api_segments(api_full_name)
        self.assertEqual(api_name, None)
        self.assertEqual(full_api_name, None)
        self.assertEqual(direction_status, None)
        
    def test_get_module_and_atttribute_name(self):
        attribute = 'torch.float32'
        module_name, attribute_name = get_module_and_atttribute_name(attribute)
        self.assertEqual(module_name, 'torch')
        self.assertEqual(attribute_name, 'float32')
        
        attribute = 'torch'
        with self.assertRaises(CompareException) as context:
            get_module_and_atttribute_name(attribute)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
        
    def test_get_attribute(self):
        module_name = 'torch'
        attribute_name = 'float32'
        attribute = get_attribute(module_name, attribute_name)
        self.assertEqual(attribute, torch.float32)

        module_name = 'json'
        attribute_name = 'loads'
        with self.assertRaises(CompareException) as context:
            get_attribute(module_name, attribute_name)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.code, CompareException.INVALID_DATA_ERROR)
        
        module_name = "torch"
        attribute_name = "float128"
        with self.assertRaises(CompareException) as context:
            get_attribute(module_name, attribute_name)
        self.assertTrue(isinstance(context.exception, CompareException))
        self.assertEqual(context.exception.code, CompareException.INVALID_ATTRIBUTE_ERROR)
        