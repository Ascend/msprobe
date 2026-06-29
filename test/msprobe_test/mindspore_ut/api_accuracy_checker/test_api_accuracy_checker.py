import unittest
import sys
import logging
import os
import json
import csv
import tempfile
import shutil


from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)

def delete_files_with_prefix(directory, prefix):
    for file_or_folder in os.listdir(directory):
        full_path = os.path.join(directory, file_or_folder)
        if os.path.isfile(full_path) and file_or_folder.startswith(prefix):
            os.remove(full_path)
            print(f"已删除文件: {full_path}")

def modify_tensor_api_info_json(json_file_path, modified_dump_data_dir):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    if 'dump_data_dir' in data:
        data['dump_data_dir'] = modified_dump_data_dir
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def check_csv(csv_path, row_num):
    with open(csv_path, 'r', encoding="utf-8") as f:
        csvreader = csv.reader(f)
        assert row_num == sum(1 for _ in csvreader)

def find_with_prefix(directory, prefix):
    entries = os.listdir(directory)
    target_files = [os.path.join(directory, entry) for entry in entries if entry.startswith(prefix) and os.path.isfile(os.path.join(directory, entry))]
    return target_files


class Args:
    def __init__(self, api_info_file=None, out_path=None, result_csv_path=None, save_error_data=False):
        self.api_info_file = api_info_file if api_info_file is not None else os.path.join(directory, "files", "api_info_statistics.json")
        self.out_path = out_path if out_path is not None else os.path.join(directory, "files")
        self.result_csv_path = result_csv_path if result_csv_path is not None else ""
        self.save_error_data = save_error_data


class TestApiAccuracyChecker(unittest.TestCase):
    def test_init_save_error_data(self):
        # 使用临时目录，不污染项目文件
        temp_dir = tempfile.mkdtemp()
        try:
            # 构造 args，只关注 out_path 和 save_error_data
            args = Args(out_path=temp_dir, save_error_data=True)
            config, dump_path_agg = ApiAccuracyChecker.init_save_error_data(args)

            # 1. config 字段检查
            self.assertEqual(config.execution_mode, "pynative")
            self.assertEqual(config.task, "tensor")
            self.assertEqual(config.dump_path, temp_dir)
            self.assertEqual(config.dump_tensor_data_dir, temp_dir)
            self.assertFalse(config.async_dump)

            # 2. error_data 目录已创建
            error_dir = os.path.join(temp_dir, "error_data")
            self.assertTrue(os.path.isdir(error_dir), f"{error_dir} should exist")

            # 3. dump_path_agg 路径检查
            self.assertEqual(dump_path_agg.dump_file_path, os.path.join(temp_dir, "dump.json"))
            self.assertEqual(dump_path_agg.stack_file_path, os.path.join(temp_dir, "stack.json"))
            self.assertEqual(dump_path_agg.dump_tensor_data_dir, error_dir)

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)

    def test_statistics_mode(self):
        api_info_statistics_path = os.path.join(directory, "files", "api_info_statistics.json")
        result_directory = os.path.join(directory, "files")

        # 初始化 Args 类，提供三个路径参数
        args = Args(api_info_file=api_info_statistics_path, out_path=result_directory)  # 在这里传入自定义的路径参数

        delete_files_with_prefix(result_directory, "accuracy_checking")
        api_accuracy_checker = ApiAccuracyChecker(args)
        api_accuracy_checker.parse(api_info_statistics_path)
        api_accuracy_checker.run_and_compare()

        detail_csv = find_with_prefix(result_directory, "accuracy_checking_detail")
        assert len(detail_csv) == 1
        check_csv(detail_csv[0], 2)

        result_csv = find_with_prefix(result_directory, "accuracy_checking_result")
        assert len(result_csv) == 1
        check_csv(result_csv[0], 2)
        delete_files_with_prefix(result_directory, "accuracy_checking")

    def test_tensor_mode(self):
        api_info_tensor_path = os.path.join(directory, "files", "api_info_tensor.json")
        result_directory = os.path.join(directory, "files")
        delete_files_with_prefix(result_directory, "accuracy_checking")
        modify_tensor_api_info_json(api_info_tensor_path, result_directory)

        args = Args(api_info_file=api_info_tensor_path, out_path=result_directory)

        api_accuracy_checker = ApiAccuracyChecker(args)
        api_accuracy_checker.parse(api_info_tensor_path)
        api_accuracy_checker.run_and_compare()

        detail_csv = find_with_prefix(result_directory, "accuracy_checking_detail")
        assert len(detail_csv) == 1
        check_csv(detail_csv[0], 2)

        result_csv = find_with_prefix(result_directory, "accuracy_checking_result")
        assert len(result_csv) == 1
        check_csv(result_csv[0], 2)
        modify_tensor_api_info_json(api_info_tensor_path, "")
        delete_files_with_prefix(result_directory, "accuracy_checking")

    def test_is_api_checkable(self):
        input_return_mapping = {"fake api": False, "MintFunctional.relu.0.forward": True, "Tensor.add_.0.forward": True,
                                "Tensor.new.add.0.forward": False}
        for api_name, target_result in input_return_mapping.items():
            result = ApiAccuracyChecker.is_api_checkable(api_name)
            assert result == target_result


if __name__ == '__main__':
    unittest.main()