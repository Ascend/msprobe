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

import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock
from zipfile import ZipFile, ZipInfo

import pytest
from msprobe.core.common.file_utils import *


class TestFileChecks:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = str(tmp_path / "test_file.txt")
        self.test_dir = tmp_path / "test_dir"

        # Common mocks
        self.mock_stat = MagicMock()
        self.mock_stat.st_mode = 0o755
        self.mock_stat.st_uid = 1000

    def test_check_link(self):
        with patch('os.path.islink', return_value=False):
            check_link(self.test_file)

    def test_check_path_length(self):
        # Test normal path
        check_path_length(str(self.test_file))

        # Test too long path
        long_path = self.test_dir / ('a' * FileCheckConst.DIRECTORY_LENGTH)
        with pytest.raises(FileCheckException) as exc_info:
            check_path_length(str(long_path))
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    def test_check_path_exists(self):
        with patch('os.path.exists', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_exists(self.test_file)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    def test_check_path_not_exists(self):
        # path does not exist: should pass
        with patch('os.path.exists', return_value=False):
            check_path_not_exists(self.test_file)

        # path exists: should raise ILLEGAL_PATH_ERROR
        with patch('os.path.exists', return_value=True), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_not_exists(self.test_file)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    def test_check_path_readability(self):
        with patch('os.access', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_readability(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        with patch('os.access', return_value=True):
            check_path_readability(self.test_file)

    def test_check_path_writability(self):
        with patch('os.access', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_writability(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        with patch('os.access', return_value=True):
            check_path_writability(self.test_file)

    def test_check_path_executable(self):
        with patch('os.access', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_executable(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        with patch('os.access', return_value=True):
            check_path_executable(self.test_file)

    def test_check_path_pattern_valid(self):
        valid_paths = [
            self.test_dir / "file.txt",
            self.test_dir / "file-1.txt",
            self.test_dir / "file_1.txt",
            self.test_dir / "file.1.txt",
        ]

        invalid_paths = [
            self.test_dir / "file*.txt",
            self.test_dir / "file?.txt",
            self.test_dir / "file;.txt",
            self.test_dir / "file|.txt",
        ]

        for path in valid_paths:
            path = str(path)
            check_path_pattern_valid(path)

        for path in invalid_paths:
            path = str(path)
            with pytest.raises(FileCheckException) as exc_info:
                check_path_pattern_valid(path)
            assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    @pytest.mark.parametrize("file_size,max_size,should_raise", [
        (100, 200, False),
        (200, 100, True),
        (1024 * 1024, 1024 * 1024 - 1, True),
    ])
    def test_check_file_size(self, file_size, max_size, should_raise):
        with patch('os.path.getsize', return_value=file_size):
            if should_raise:
                with pytest.raises(FileCheckException) as exc_info:
                    check_file_size(self.test_file, max_size)
                assert exc_info.value.code == FileCheckException.FILE_TOO_LARGE_ERROR
            else:
                check_file_size(self.test_file, max_size)

    def test_check_file_size_os_error(self):
        # os.path.getsize 触发 OSError 时应抛出 INVALID_FILE_ERROR
        with patch('os.path.getsize', side_effect=OSError("stat failed")), \
                pytest.raises(FileCheckException) as exc_info:
            check_file_size(self.test_file, 1024)
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR


class TestFileOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = tmp_path / "test_file"
        self.test_dir = tmp_path / "test_dir"
        self.yaml_file = tmp_path / "test.yaml"
        self.json_file = tmp_path / "test.json"
        self.npy_file = tmp_path / "test.npy"
        self.excel_file = tmp_path / "test.xlsx"

    def test_check_common_file_size(self):
        with patch('os.path.isfile', return_value=True), \
                patch('os.path.getsize', return_value=100):
            check_common_file_size(str(self.test_file))
            check_common_file_size(str(self.test_file.with_suffix('.csv')))

        with patch('os.path.isfile', return_value=True), \
                patch('os.path.getsize', return_value=FileCheckConst.MAX_COMMON_FILE_SIZE + 1), \
                pytest.raises(FileCheckException) as exc_info:
            check_common_file_size(str(self.test_file))
        assert exc_info.value.code == FileCheckException.FILE_TOO_LARGE_ERROR

    def test_check_file_suffix(self):
        check_file_suffix(str(self.test_file.with_suffix('.txt')), '.txt')

        with pytest.raises(FileCheckException) as exc_info:
            check_file_suffix(str(self.test_file.with_suffix('.txt')), '.csv')
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR

        check_file_suffix((self.test_file.with_suffix('.txt')), None)

        # 非字符串 / list / tuple 类型的 file_suffix 应抛出 TypeError
        with pytest.raises(TypeError):
            check_file_suffix(str(self.test_file.with_suffix('.txt')), 123)

    def test_make_dir(self):
        with patch('os.path.isdir', return_value=False), \
                patch('os.makedirs') as mock_makedirs, \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker:
            mock_checker.return_value.common_check.return_value = None
            make_dir(self.test_dir)
            mock_makedirs.assert_called_once_with(
                str(self.test_dir),
                exist_ok=True
            )

    def test_load_yaml(self):
        yaml_content = """
        key: value
        list:
          - item1
          - item2
        """
        with patch('builtins.open', mock_open(read_data=yaml_content)), \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker, \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            mock_checker.return_value.common_check.return_value = str(self.yaml_file)
            result = load_yaml(str(self.yaml_file))
            assert result == {'key': 'value', 'list': ['item1', 'item2']}

        # Test load error
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content")), \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker, \
                pytest.raises(RuntimeError):
            mock_checker.return_value.common_check.return_value = str(self.yaml_file)
            load_yaml(str(self.yaml_file))

    def test_load_npy(self):
        mock_array = np.array([1, 2, 3])
        with patch('numpy.load', return_value=mock_array), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None):
            result = load_npy(str(self.npy_file))
            np.testing.assert_array_equal(result, mock_array)

        with patch('numpy.load', side_effect=Exception), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                pytest.raises(RuntimeError):
            load_npy(str(self.npy_file))

    def test_save_npy(self):
        mock_data = np.array([1, 2, 3])
        with patch('numpy.save') as mock_save, \
                patch('os.chmod') as mock_chmod:
            save_npy(mock_data, str(self.npy_file))
            mock_save.assert_called_once()

    def test_save_json(self):
        test_data = {'key': 'value'}
        mock_file = mock_open()

        with patch('builtins.open', mock_file), \
                patch('fcntl.flock') as mock_flock, \
                patch('json.dump') as mock_dump, \
                patch('os.chmod') as mock_chmod:
            save_json(self.json_file, test_data)
            mock_file.assert_called_once_with(str(self.json_file), 'w', encoding='utf-8')
            mock_dump.assert_called_once_with(test_data, mock_file(), indent=None)

    def test_load_json(self):
        test_data = '{"key": "value"}'
        mock_file = mock_open(read_data=test_data)

        with patch('builtins.open', mock_file), \
                patch('fcntl.flock') as mock_flock, \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = load_json(str(self.json_file))
            mock_file.assert_called_once_with(str(self.json_file), 'r', encoding='utf-8')
            assert mock_flock.call_count == 2
            assert result == {'key': 'value'}

    def test_save_yaml(self):
        test_data = {'key': 'value'}
        mock_file = mock_open()

        with patch('builtins.open', mock_file), \
                patch('fcntl.flock') as mock_flock, \
                patch('yaml.dump') as mock_dump, \
                patch('os.chmod') as mock_chmod:
            save_yaml(str(self.yaml_file), test_data)
            mock_file.assert_called_once_with(str(self.yaml_file), 'w', encoding='utf-8')
            assert mock_flock.call_count == 2
            mock_dump.assert_called_once_with(test_data, mock_file(), sort_keys=False)

        # 保存过程中异常被包装为 RuntimeError
        with patch('builtins.open', mock_file), \
                patch('yaml.dump', side_effect=Exception("yaml fail")), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None), \
                pytest.raises(RuntimeError):
            save_yaml(str(self.yaml_file), test_data)

    def test_save_excel_tiny(self):
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with patch('pandas.DataFrame.to_excel') as mock_to_excel, \
                patch('pandas.ExcelWriter') as mock_writer, \
                patch('os.chmod') as mock_chmod:
            save_excel(self.excel_file, df)
            mock_to_excel.assert_called_once_with(mock_writer().__enter__(), sheet_name='Sheet1', index=False)

    def test_save_excel_large(self):
        df = pd.DataFrame({'col1': list(range(1500000)), 'col2': list(range(1500000, 0, -1))})
        with patch('pandas.DataFrame.to_excel') as mock_to_excel, \
                patch('pandas.ExcelWriter') as mock_writer, \
                patch('os.chmod') as mock_chmod:
            save_excel(self.excel_file, df)
            mock_to_excel.assert_called_with(mock_writer().__enter__(), sheet_name='part_1', index=False)

    def test_move_file(self):
        dst_file = self.test_dir / "moved_file"
        with patch('shutil.move') as mock_move, \
                patch('os.chmod') as mock_chmod, \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_path_before_create', return_value=None):
            move_file(str(self.test_file), str(dst_file))
            mock_move.assert_called_once_with(str(self.test_file), str(dst_file))

        with patch('shutil.move', side_effect=Exception), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_path_before_create', return_value=None), \
                pytest.raises(RuntimeError):
            move_file(self.test_file, dst_file)


class TestCSVOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.csv_file = tmp_path / "test.csv"
        self.test_data = [['header1', 'header2'], ['value1', 'value2']]
        self.test_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    def test_write_csv(self):
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
                patch('csv.writer') as mock_writer, \
                patch('os.chmod') as mock_chmod:
            write_csv(self.test_data, self.csv_file)
            mock_file.assert_called_once_with(
                str(self.csv_file), 'a+', encoding='utf-8-sig'
            )
            mock_writer.return_value.writerows.assert_called_once_with(self.test_data)

    def test_write_csv_malicious_check(self):
        test_data = [['normal', '=1+1']]  # Formula injection attempt
        with pytest.raises(RuntimeError):
            write_csv(test_data, self.csv_file, malicious_check=True)

    def test_read_csv(self):
        # Test pandas read
        with patch('pandas.read_csv', return_value=self.test_df), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None):
            result = read_csv(str(self.csv_file), as_pd=True)
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, self.test_df)

        # Test standard csv read
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
                patch('csv.reader', return_value=self.test_data), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = read_csv(self.csv_file, as_pd=False)
            assert result == self.test_data

    def test_write_df_to_csv(self):
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
                patch('os.chmod') as mock_chmod:
            write_df_to_csv(self.test_df, str(self.csv_file))
            mock_to_csv.assert_called_once_with(
                str(self.csv_file),
                mode='w',
                header=True,
                index=False,
                encoding='utf-8'
            )

        # Test invalid data type
        with pytest.raises(ValueError):
            write_df_to_csv([1, 2, 3], self.csv_file)

        # Test malicious check
        df_with_formula = pd.DataFrame({'col1': ['=1+1']})
        with pytest.raises(RuntimeError):
            write_df_to_csv(df_with_formula, self.csv_file)


class TestPathOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_path = tmp_path / "test_path"
        self.test_dir = tmp_path / "test_dir"
        self.test_file = tmp_path / "test_file.txt"

    def test_check_path_type(self):
        # Test file type
        with patch('os.path.isfile', return_value=True):
            check_path_type(self.test_file, FileCheckConst.FILE)

        with patch('os.path.isfile', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_type(self.test_file, FileCheckConst.FILE)
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR

        # Test directory type
        with patch('os.path.isdir', return_value=True):
            check_path_type(self.test_dir, FileCheckConst.DIR)

        with patch('os.path.isdir', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_type(self.test_dir, FileCheckConst.DIR)
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR

    def test_create_directory(self):
        with patch('os.path.isdir', return_value=True), \
                patch('os.makedirs') as mock_makedirs, \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker:
            mock_checker.return_value.common_check.return_value = None
            create_directory(str(self.test_dir))

    def test_check_path_before_create(self):
        # Test valid path
        check_path_before_create(self.test_path)

        # Test path length exceeds limit
        long_path = self.test_dir / ('a' * FileCheckConst.DIRECTORY_LENGTH)
        with pytest.raises(FileCheckException) as exc_info:
            check_path_before_create(long_path)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

        # Test invalid characters
        invalid_path = self.test_dir / "test*file"
        with pytest.raises(FileCheckException) as exc_info:
            check_path_before_create(invalid_path)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR


class TestUtilityOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = tmp_path / "test_file"
        self.test_dir = tmp_path / "test_dir"
        self.npy_file = tmp_path / "test.npy"
        self.txt_file = tmp_path / "test.txt"
        self.workbook_file = tmp_path / "test.xlsx"

    def test_save_npy_to_txt(self):
        test_data = np.array([1, 2, 3, 4])

        with patch('os.path.exists', return_value=False), \
                patch('numpy.savetxt') as mock_savetxt:
            # Test without alignment
            save_npy_to_txt(test_data, self.txt_file)
            mock_savetxt.assert_called_once()

        # Test with alignment
        with patch('os.path.exists', return_value=False), \
                patch('numpy.savetxt') as mock_savetxt:
            save_npy_to_txt(test_data, self.txt_file, align=3)
            mock_savetxt.assert_called_once()

    def test_remove_path(self):
        # Test remove file
        with patch('os.path.exists', return_value=True), \
                patch('os.path.islink', return_value=False), \
                patch('os.path.isfile', return_value=True), \
                patch('os.remove') as mock_remove:
            remove_path("/test_remove_path/test/test.txt")
            mock_remove.assert_called_once_with("/test_remove_path/test/test.txt")

        # Test remove directory
        with patch('os.path.exists', return_value=True), \
                patch('os.path.islink', return_value=False), \
                patch('os.path.isfile', return_value=False), \
                patch('shutil.rmtree') as mock_rmtree:
            remove_path("/test_remove_path/test")
            mock_rmtree.assert_called_once_with("/test_remove_path/test")

    def test_get_json_contents(self):
        json_content = '{"key": "value"}'
        with patch('builtins.open', mock_open(read_data=json_content)), \
                patch('os.path.exists', return_value=True), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = get_json_contents(str(self.test_file))
            assert result == {'key': 'value'}

        # Test invalid JSON
        with patch('builtins.open', mock_open(read_data='invalid json')), \
                patch('os.path.exists', return_value=True), \
                pytest.raises(FileCheckException) as exc_info:
            get_json_contents(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

    def test_get_file_content_bytes(self):
        test_content = b'test content'
        with patch('builtins.open', mock_open(read_data=test_content)), \
                patch('os.path.exists', return_value=True), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = get_file_content_bytes(self.test_file)
            assert result == test_content

    def test_os_walk_for_files(self):
        mock_walk_data = [
            (str(self.test_dir), ['dir1'], ['file1.txt']),
            (str(self.test_dir / 'dir1'), [], ['file2.txt'])
        ]

        with patch('os.walk', return_value=mock_walk_data), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path'):
            # Test with depth 1
            result = os_walk_for_files(str(self.test_dir), 2)
            assert len(result) == 2
            assert result[0]['file'] == 'file1.txt'
            assert result[1]['file'] == 'file2.txt'

            # Test with depth 0
            result = os_walk_for_files(str(self.test_dir), 1)
            assert len(result) == 1
            assert result[0]['file'] == 'file1.txt'


class TestDirectoryChecks:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_dir = tmp_path / "test_dir"
        self.test_file = tmp_path / "test_file"

    def test_check_file_or_directory_path(self):
        with patch('msprobe.core.common.file_utils.FileChecker') as mock_checker:
            mock_checker.return_value.common_check.return_value = None
            # Test file path
            check_file_or_directory_path(self.test_file, isdir=False)
            # Test directory path
            check_file_or_directory_path(self.test_dir, isdir=True)


cur_dir = os.path.dirname(os.path.realpath(__file__))
zip_dir = os.path.join(cur_dir, 'test_temp_zip_file')


class TestCheckZipFile(unittest.TestCase):
    def setUp(self):
        os.makedirs(zip_dir, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(zip_dir):
            shutil.rmtree(zip_dir)

    @staticmethod
    def create_fake_zip_with_sizes(file_sizes):
        """创建临时 zip 文件，file_sizes 为每个文件的大小列表，伪造一个具有 file_size=size 的 ZIP 条目"""
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=zip_dir)
        os.close(tmp_fd)
        with ZipFile(tmp_path, 'w', allowZip64=True) as zipf:
            for i, size in enumerate(file_sizes):
                info = ZipInfo(f"file_{i}.bin")
                zipf.writestr(info, b'')  # 实际内容为空，但声明文件大小为 size
                info.file_size = size
        return tmp_path

    def test_valid_zip(self):
        file_sizes = [100, 200, 300]
        zip_path = self.create_fake_zip_with_sizes(file_sizes)
        try:
            check_zip_file(zip_path)
        finally:
            os.remove(zip_path)

    def test_single_file_too_large(self):
        file_sizes = [FileCheckConst.MAX_FILE_IN_ZIP_SIZE + 1]
        zip_path = self.create_fake_zip_with_sizes(file_sizes)
        try:
            with self.assertRaises(ValueError) as cm:
                check_zip_file(zip_path)
            self.assertIn("is too large to extract", str(cm.exception))
        finally:
            os.remove(zip_path)

    def test_total_size_too_large(self):
        count = 20
        size_each = (FileCheckConst.MAX_ZIP_SIZE // count) + 1
        file_sizes = [size_each] * count
        zip_path = self.create_fake_zip_with_sizes(file_sizes)
        try:
            with self.assertRaises(ValueError) as cm:
                check_zip_file(zip_path)
            self.assertIn("Total extracted size exceeds the limit", str(cm.exception))
        finally:
            os.remove(zip_path)


class TestMoveDirectory:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.src_dir = tmp_path / "src_dir"
        self.dst_dir = tmp_path / "dst_dir"
        self.src_dir.mkdir()
        self.test_file = self.src_dir / "test_file.txt"
        self.test_file.write_text("test content")

    def test_move_directory_success(self):
        """Test successful directory move"""
        with patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_path_before_create', return_value=None), \
                patch('os.chmod') as mock_chmod:
            move_directory(str(self.src_dir), str(self.dst_dir))

            assert not self.src_dir.exists()
            assert self.dst_dir.exists()
            assert (self.dst_dir / "test_file.txt").exists()
            assert (self.dst_dir / "test_file.txt").read_text() == "test content"


class TestCreateFileWithList:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = tmp_path / "test_file.txt"
        self.test_data = ["line1", "line2", "line3"]

    def test_create_file_with_list_success(self):
        """Test successful file creation with list content"""
        with patch('builtins.open', mock_open()) as mock_file, \
                patch('os.chmod') as mock_chmod, patch('fcntl.flock') as mock_flock:
            create_file_with_list(self.test_data, str(self.test_file))

            mock_file.assert_called_once_with(str(self.test_file), 'w', encoding='utf-8')


class TestCreateFileWithContent:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = tmp_path / "test_file.txt"
        self.test_content = "test file content"

    def test_create_file_with_content_success(self):
        """Test successful file creation with content"""
        with patch('builtins.open', mock_open()) as mock_file, \
                patch('os.chmod') as mock_chmod, \
                patch('fcntl.flock') as mock_flock:
            create_file_with_content(self.test_content, str(self.test_file))

            mock_file.assert_called_once_with(str(self.test_file), 'w', encoding='utf-8')
            mock_file().write.assert_called_once_with(self.test_content)


class TestAddFileToZip:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.zip_file = tmp_path / "test.zip"
        self.content_file = tmp_path / "content.txt"
        self.content_file.write_text("test content")

    def test_add_file_to_zip_success(self):
        """Test successful file addition to zip"""
        with patch('zipfile.ZipFile') as mock_zip_class, \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None):
            mock_zip = MagicMock()
            mock_zip_class.return_value.__enter__.return_value = mock_zip

            add_file_to_zip(str(self.zip_file), str(self.content_file), "archive_name.txt")

            mock_zip_class.assert_called_once_with(str(self.zip_file), 'a')
            mock_zip.write.assert_called_once_with(str(self.content_file), "archive_name.txt")

    def test_add_file_to_zip_size_exceeded(self):
        """当 ZIP 总大小超过上限时抛出 RuntimeError"""
        with patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_suffix', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_whether_exist_or_not', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_size', return_value=None), \
                patch('os.path.exists', return_value=True), \
                patch('os.path.getsize',
                      side_effect=[FileCheckConst.MAX_ZIP_SIZE, 1]):
            with pytest.raises(RuntimeError) as exc_info:
                add_file_to_zip(str(self.zip_file), str(self.content_file), "archive_name.txt")
            assert "ZIP file size exceeds the limit" in str(exc_info.value)

    def test_add_file_to_zip_runtime_error_on_write(self):
        """写 ZIP 过程中异常应被包装为 RuntimeError"""
        with patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_suffix', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_whether_exist_or_not', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_size', return_value=None), \
                patch('os.path.exists', return_value=False), \
                patch('os.path.getsize', return_value=0), \
                patch('zipfile.ZipFile', side_effect=Exception("zip fail")):
            with pytest.raises(RuntimeError) as exc_info:
                add_file_to_zip(str(self.zip_file), str(self.content_file), "archive_name.txt")
            assert "add file to zip" in str(exc_info.value)


class TestCreateFileInZip:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.zip_file = tmp_path / "test.zip"
        self.file_content = "test file content"

    def test_create_file_in_zip_success(self):
        """Test successful file creation in zip"""
        with patch('zipfile.ZipFile') as mock_zip_class:
            mock_zip = MagicMock()
            mock_zip_class.return_value.__enter__.return_value = mock_zip

            create_file_in_zip(str(self.zip_file), "internal_file.txt", self.file_content)

            mock_zip_class.assert_called_once()
            mock_zip.writestr.assert_called_once()

    def test_create_file_in_zip_size_exceeded(self):
        """ZIP 大小超过上限时抛出 RuntimeError"""
        with patch('msprobe.core.common.file_utils.check_file_suffix', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_whether_exist_or_not', return_value=None), \
                patch('os.path.exists', return_value=True), \
                patch('os.path.getsize', return_value=FileCheckConst.MAX_ZIP_SIZE), \
                patch('msprobe.core.common.file_utils.sys.getsizeof', return_value=1):
            with pytest.raises(RuntimeError) as exc_info:
                create_file_in_zip(str(self.zip_file), "internal_file.txt", self.file_content)
            assert "ZIP file size exceeds the limit" in str(exc_info.value)

    def test_create_file_in_zip_runtime_error_on_write(self):
        """写 ZIP 内容异常时应抛出 RuntimeError"""
        with patch('msprobe.core.common.file_utils.check_file_suffix', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_whether_exist_or_not', return_value=None), \
                patch('os.path.exists', return_value=False), \
                patch('os.path.getsize', return_value=0), \
                patch('msprobe.core.common.file_utils.sys.getsizeof', return_value=0), \
                patch('builtins.open', mock_open()) as mock_file, \
                patch('fcntl.flock'), \
                patch('zipfile.ZipFile', side_effect=Exception("zip fail")):
            with pytest.raises(RuntimeError) as exc_info:
                create_file_in_zip(str(self.zip_file), "internal_file.txt", self.file_content)
            assert "Save content to file" in str(exc_info.value)


class TestCheckOutputDirPath:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_dir = tmp_path / "test_output_dir"
        self.test_dir.mkdir(exist_ok=True)
        
    def test_check_output_dir_path_existing_dir(self):
        """Test check_output_dir_path with existing directory"""
        existing_dir = str(self.test_dir)
        
        with patch('msprobe.core.common.file_utils.check_path_length') as mock_check_length, \
                patch('msprobe.core.common.file_utils.check_path_pattern_valid') as mock_check_pattern, \
                patch('msprobe.core.common.file_utils.check_path_writability') as mock_check_writability, \
                patch('os.path.exists', return_value=True):
            
            check_output_dir_path(existing_dir)
            
            # Should call all current checks for existing path
            mock_check_length.assert_called_once()
            mock_check_pattern.assert_called_once()
            mock_check_writability.assert_called_once()


class TestExtractZip:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.zip_file = tmp_path / "test.zip"
        self.extract_dir = tmp_path / "extracted"

    def test_extract_zip_success(self):
        """Test successful zip extraction"""
        with patch('zipfile.ZipFile') as mock_zip_class, \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_output_dir_path', return_value=None):
            mock_zip = MagicMock()
            mock_zip_class.return_value.__enter__.return_value = mock_zip

            extract_zip(str(self.zip_file), str(self.extract_dir))

            mock_zip_class.assert_called()
            mock_zip.extractall.assert_called_once_with(str(self.extract_dir))

    def test_extract_zip_check_zip_failed(self):
        """check_zip_file 抛异常时应包装为 RuntimeError"""
        with patch('msprobe.core.common.file_utils.check_file_suffix', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.create_directory', return_value=None), \
                patch('msprobe.core.common.file_utils.check_zip_file', side_effect=Exception("invalid zip")):
            with pytest.raises(RuntimeError) as exc_info:
                extract_zip(str(self.zip_file), str(self.extract_dir))
            assert "Save content to file" in str(exc_info.value)

    def test_extract_zip_extract_failed(self):
        """解压过程中失败时应抛出 RuntimeError"""
        with patch('msprobe.core.common.file_utils.check_file_suffix', return_value=None), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.create_directory', return_value=None), \
                patch('msprobe.core.common.file_utils.check_zip_file', return_value=None), \
                patch('zipfile.ZipFile') as mock_zip_class:
            mock_zip = MagicMock()
            mock_zip.extractall.side_effect = Exception("extract fail")
            mock_zip_class.return_value.__enter__.return_value = mock_zip

            with pytest.raises(RuntimeError) as exc_info:
                extract_zip(str(self.zip_file), str(self.extract_dir))
            assert "extract zip file" in str(exc_info.value)


class TestSplitZipFilePath:
    def test_split_zip_file_path_valid(self):
        """Test splitting valid zip file path"""
        zip_path = "path/to/archive.zip"
        result_zip, result_internal = split_zip_file_path(zip_path)
        assert result_internal == "archive.zip"


class TestDeserializationScanner:
    def test_deserialization_scanner_creation(self):
        """Test DeserializationScanner creation"""
        scanner = DeserializationScanner()

        assert scanner.DANGEROUS_METHODS is not None
        assert scanner.DANGEROUS_MODULES is not None
        assert isinstance(scanner.DANGEROUS_METHODS, set)
        assert isinstance(scanner.DANGEROUS_MODULES, set)

    def test_scan_pickle_content_safe_data(self):
        """Test scanning safe pickle content"""
        scanner = DeserializationScanner()

        # Create safe pickle data (just a simple string)
        safe_data = b"test string data"
        with patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None), \
                patch('builtins.open', mock_open(read_data=safe_data)):
            result = scanner.scan_pickle_content(safe_data)

        # Safe data should not raise any warnings
        assert result is True  # Method doesn't return anything, just logs warnings

    def test_scan_pickle_content_with_dangerous_methods(self):
        """Test scanning pickle content with dangerous methods"""
        scanner = DeserializationScanner()

        # Create pickle data that contains a dangerous method name
        # This is a simplified test - in reality, we'd need actual pickle data
        dangerous_content = b"os.system"

        with patch('msprobe.core.common.file_utils.logger') as mock_logger, \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None), \
                patch('builtins.open', mock_open(read_data=dangerous_content)):
            result = scanner.scan_pickle_content(dangerous_content)

            assert result is False


class TestReadXlsx:
    """Test class for read_xlsx function"""
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_dir = tmp_path / "test_output_dir"
        self.test_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_read_xlsx_success(self):
        """Test read_xlsx function with successful reading"""
        test_file = str(self.test_dir / "test.xlsx")
        
        with patch('msprobe.core.common.file_utils.check_file_or_directory_path') as mock_check_path, \
                patch('msprobe.core.common.file_utils.check_zip_file') as mock_check_zip, \
                patch('pandas.read_excel') as mock_read_excel:
            
            # Mock the return value
            mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
            mock_read_excel.return_value = mock_df
            
            # Test without sheet_name
            result = read_xlsx(test_file)
            
            mock_check_path.assert_called_once_with(test_file)
            mock_check_zip.assert_called_once_with(test_file)
            mock_read_excel.assert_called_once_with(test_file, keep_default_na=False)
            assert result is mock_df
