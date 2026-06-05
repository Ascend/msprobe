import os
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock

from msprobe.core.compare.data2db import (
    DBImporter,
    load_mapping,
    validate_micro_step,
    validate_process_num,
    _data2db_service_parser,
    _data2db_command,
)


class TestValidateMicroStep(unittest.TestCase):
    """测试validate_micro_step函数"""

    def test_valid_true_lower(self):
        """输入 'true' 返回 True"""
        self.assertTrue(validate_micro_step('true'))

    def test_valid_false_lower(self):
        """输入 'false' 返回 False"""
        self.assertFalse(validate_micro_step('false'))

    def test_valid_true_upper(self):
        """输入 'True' 返回 True"""
        self.assertTrue(validate_micro_step('True'))

    def test_valid_false_upper(self):
        """输入 'False' 返回 False"""
        self.assertFalse(validate_micro_step('False'))

    def test_invalid_string(self):
        """输入非预期字符串，应返回 False"""
        self.assertFalse(validate_micro_step('invalid'))


class TestValidateProcessNum(unittest.TestCase):
    """测试validate_process_num函数"""

    def test_valid_positive_int(self):
        """合法的正整数值"""
        validate_process_num(4)  # should not raise

    def test_valid_min_boundary(self):
        """边界值1"""
        validate_process_num(1)  # should not raise

    def test_valid_max_boundary(self):
        """边界值128"""
        validate_process_num(128)  # should not raise

    def test_invalid_zero(self):
        """0应该抛出异常"""
        with self.assertRaises(ValueError):
            validate_process_num(0)

    def test_invalid_negative(self):
        """负数应该抛出异常"""
        with self.assertRaises(ValueError):
            validate_process_num(-1)

    def test_invalid_exceeds_max(self):
        """超过最大值应该抛出异常"""
        with self.assertRaises(ValueError):
            validate_process_num(129)

    def test_invalid_non_int(self):
        """非整数应该抛出异常"""
        with self.assertRaises(ValueError):
            validate_process_num("abc")


class TestLoadMapping(unittest.TestCase):
    """测试load_mapping函数"""

    @patch("msprobe.core.compare.data2db.load_json")
    def test_load_with_valid_path(self, mock_load_json):
        """有效的mapping路径返回解析后的dict"""
        mock_load_json.return_value = {"key": "value"}
        result = load_mapping("/path/to/mapping.json")
        mock_load_json.assert_called_once_with("/path/to/mapping.json")
        self.assertEqual(result, {"key": "value"})

    @patch("msprobe.core.compare.data2db.load_json")
    def test_load_with_none_path(self, mock_load_json):
        """None路径返回空dict"""
        result = load_mapping(None)
        mock_load_json.assert_not_called()
        self.assertEqual(result, {})

    def test_load_with_empty_string(self):
        """空字符串路径返回空dict"""
        result = load_mapping("")
        self.assertEqual(result, {})

    def test_load_with_non_string(self):
        """非字符串类型返回空dict"""
        result = load_mapping(123)
        self.assertEqual(result, {})


class TestDBImporterInit(unittest.TestCase):
    """测试DBImporter初始化"""

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_init_default_params(self, mock_create_dir, mock_check_path):
        """默认参数初始化"""
        importer = DBImporter(
            db_path="/tmp/test_db",
            data_path="/tmp/test_data",
            micro_step="true",
        )
        self.assertEqual(importer.db_path, "/tmp/test_db")
        self.assertEqual(importer.data_path, "/tmp/test_data")
        self.assertEqual(importer.format, "auto")
        self.assertIsNone(importer.mapping_path)
        self.assertTrue(importer.micro_step)
        self.assertEqual(importer.process_num, 1)

    @patch("msprobe.core.compare.data2db.load_json")
    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_init_with_mapping(self, mock_create_dir, mock_check_path, mock_load_json):
        """带mapping文件的初始化"""
        mock_load_json.return_value = {"old": "new"}
        importer = DBImporter(
            db_path="/tmp/test_db",
            data_path="/tmp/test_data",
            format="dump",
            mapping_path="/path/to/mapping.json",
            micro_step="false",
            process_num=4,
        )
        self.assertEqual(importer.format, "dump")
        self.assertEqual(importer.mapping, {"old": "new"})
        self.assertFalse(importer.micro_step)
        self.assertEqual(importer.process_num, 4)

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path",
           side_effect=ValueError("Invalid path"))
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_init_invalid_data_path(self, mock_create_dir, mock_check_path):
        """无效的数据路径抛出异常"""
        with self.assertRaises(ValueError):
            DBImporter(
                db_path="/tmp/test_db",
                data_path="/invalid/path",
            )

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_init_unsupported_format(self, mock_create_dir, mock_check_path):
        """不支持的format抛出异常"""
        with self.assertRaises(ValueError) as ctx:
            DBImporter(
                db_path="/tmp/test_db",
                data_path="/tmp/test_data",
                format="unsupported",
                micro_step="true",
            )
        self.assertIn("Unsupported format", str(ctx.exception))

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_init_micro_step_true(self, mock_create_dir, mock_check_path):
        """micro_step='true' 应解析为 True"""
        importer = DBImporter(
            db_path="/tmp/test_db", data_path="/tmp/test_data", micro_step="true")
        self.assertTrue(importer.micro_step)

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_init_micro_step_false(self, mock_create_dir, mock_check_path):
        """micro_step='false' 应解析为 False"""
        importer = DBImporter(
            db_path="/tmp/test_db", data_path="/tmp/test_data", micro_step="false")
        self.assertFalse(importer.micro_step)


class TestDBImporterEnsureDbFileClean(unittest.TestCase):
    """测试_ensure_db_file_clean方法"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_clean_when_file_not_exists(self, mock_create_dir, mock_check_path):
        """文件不存在时不应报错"""
        importer = DBImporter(db_path="/tmp/test_db", data_path="/tmp/test_data", micro_step="true")
        importer._ensure_db_file_clean(self.test_db)  # should not raise

    @patch("msprobe.core.compare.data2db.check_file_or_directory_path")
    @patch("msprobe.core.compare.data2db.create_directory")
    def test_clean_when_file_exists(self, mock_create_dir, mock_check_path):
        """文件存在时应删除旧文件"""
        # 创建测试文件
        with open(self.test_db, "w") as f:
            f.write("test")
        self.assertTrue(os.path.exists(self.test_db))

        importer = DBImporter(db_path="/tmp/test_db", data_path="/tmp/test_data", micro_step="true")
        importer._ensure_db_file_clean(self.test_db)
        self.assertFalse(os.path.exists(self.test_db))


class TestDBImporterImportDumpData(unittest.TestCase):
    """测试import_dump_data方法"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.temp_dir, "db")
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("msprobe.core.compare.data2db.DumpRecordBuilder")
    @patch("msprobe.core.compare.data2db.DumpDB")
    @patch("msprobe.core.compare.data2db.dump_scan_files")
    def test_import_dump_data_with_valid_ranks(
        self, mock_scan_files, mock_dump_db_class, mock_builder_class
    ):
        """有有效rank时正常导入"""
        mock_scan_files.return_value = {0: [(0, "/path/to/dump.json", "/path/to/construct.json")]}

        mock_db = MagicMock()
        mock_dump_db_class.return_value = mock_db

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder

        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, micro_step="true")
        importer.import_dump_data()

        mock_scan_files.assert_called_once_with(self.data_dir)
        mock_dump_db_class.assert_called_once()
        mock_builder_class.assert_called_once()
        mock_builder.import_data.assert_called_once()

    @patch("msprobe.core.compare.data2db.dump_scan_files")
    def test_import_dump_data_no_valid_ranks(self, mock_scan_files):
        """无有效rank时应警告并返回"""
        mock_scan_files.return_value = {}

        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, micro_step="true")
        importer.import_dump_data()

        mock_scan_files.assert_called_once_with(self.data_dir)


class TestDBImporterImportMonitorData(unittest.TestCase):
    """测试import_monitor_data方法"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.temp_dir, "db")
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("msprobe.core.compare.data2db.get_target_output_dir")
    @patch("msprobe.core.compare.data2db.monitor_import_data")
    def test_import_monitor_data_with_valid_dirs(
        self, mock_import_data, mock_get_dirs
    ):
        """有有效monitor目录时正常导入"""
        mock_get_dirs.return_value = {0: "/path/to/monitor/rank0"}
        mock_import_data.return_value = True

        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, micro_step="true")
        importer.import_monitor_data()

        mock_get_dirs.assert_called_once_with(self.data_dir, None, None)
        mock_import_data.assert_called_once()

    @patch("msprobe.core.compare.data2db.get_target_output_dir")
    def test_import_monitor_data_no_valid_dirs(self, mock_get_dirs):
        """无有效monitor目录时应警告并返回"""
        mock_get_dirs.return_value = {}

        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, micro_step="true")
        importer.import_monitor_data()

        mock_get_dirs.assert_called_once_with(self.data_dir, None, None)


class TestDBImporterImportData(unittest.TestCase):
    """测试import_data主入口方法"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.temp_dir, "db")
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.data_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_import_data_auto_format(self):
        """format='auto' 应同时尝试dump和monitor"""
        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, micro_step="true")
        with patch.object(importer, "import_dump_data") as mock_dump:
            with patch.object(importer, "import_monitor_data") as mock_monitor:
                importer.import_data()
                mock_dump.assert_called_once()
                mock_monitor.assert_called_once()

    def test_import_data_dump_format(self):
        """format='dump' 应只调用dump"""
        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, format="dump", micro_step="true")
        with patch.object(importer, "import_dump_data") as mock_dump:
            with patch.object(importer, "import_monitor_data") as mock_monitor:
                importer.import_data()
                mock_dump.assert_called_once()
                mock_monitor.assert_not_called()

    def test_import_data_monitor_format(self):
        """format='monitor' 应只调用monitor"""
        importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, format="monitor", micro_step="true")
        with patch.object(importer, "import_dump_data") as mock_dump:
            with patch.object(importer, "import_monitor_data") as mock_monitor:
                importer.import_data()
                mock_dump.assert_not_called()
                mock_monitor.assert_called_once()

    def test_import_data_unsupported_format(self):
        """不支持的format在初始化时就抛出异常"""
        with self.assertRaises(ValueError):
            DBImporter(db_path=self.db_dir, data_path=self.data_dir, format="invalid", micro_step="true")
        # 如果能构造成功，再测试import_data也不支持
        with patch.object(DBImporter, "_validate_parameters"):
            importer = DBImporter(db_path=self.db_dir, data_path=self.data_dir, format="invalid", micro_step="true")
            with patch.object(importer, "import_dump_data"):
                with patch.object(importer, "import_monitor_data"):
                    with self.assertRaises(ValueError):
                        importer.import_data()


class TestData2dbServiceParser(unittest.TestCase):
    """测试_data2db_service_parser"""

    def test_parser_arguments(self):
        """验证命令行参数注册"""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        data2db_parser = subparsers.add_parser('data2db')
        _data2db_service_parser(data2db_parser)

        # 解析测试参数
        args = parser.parse_args([
            'data2db',
            '--db', '/tmp/db',
            '--data', '/tmp/data',
            '--format', 'dump',
            '--mapping', '/tmp/mapping.json',
            '--micro_step', 'false',
            '--process_num', '4',
        ])
        self.assertEqual(args.db, '/tmp/db')
        self.assertEqual(args.data, '/tmp/data')
        self.assertEqual(args.format, 'dump')
        self.assertEqual(args.mapping, '/tmp/mapping.json')
        self.assertEqual(args.micro_step, 'false')
        self.assertEqual(args.process_num, 4)

    def test_parser_default_values(self):
        """验证命令行参数默认值"""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        data2db_parser = subparsers.add_parser('data2db')
        _data2db_service_parser(data2db_parser)

        args = parser.parse_args([
            'data2db',
            '--db', '/tmp/db',
            '--data', '/tmp/data',
        ])
        self.assertEqual(args.format, 'auto')
        self.assertIsNone(args.mapping)
        self.assertEqual(args.micro_step, 'true')
        self.assertEqual(args.process_num, 1)


class TestData2dbCommand(unittest.TestCase):
    """测试_data2db_command函数"""

    def test_command_creates_importer_and_imports(self):
        """验证命令创建DBImporter并调用import_data"""
        args = MagicMock()
        args.db = "/tmp/db"
        args.data = "/tmp/data"
        args.format = "auto"
        args.mapping = None
        args.micro_step = "true"
        args.process_num = 1

        with patch("msprobe.core.compare.data2db.DBImporter") as mock_importer_class:
            mock_importer = MagicMock()
            mock_importer_class.return_value = mock_importer

            _data2db_command(args)

            mock_importer_class.assert_called_once_with(
                db_path="/tmp/db",
                data_path="/tmp/data",
                format="auto",
                mapping_path=None,
                micro_step="true",
                process_num=1,
            )
            mock_importer.import_data.assert_called_once()
