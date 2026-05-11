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
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from msprobe.infer.utils.file_open_check import ms_open, FileStat, OpenException, SanitizeErrorType, sanitize_csv_value
from msprobe.infer.utils.file_open_check import PERMISSION_NORMAL, PERMISSION_KEY, RAW_INPUT_PATH
from msprobe.core.common.log import logger


@pytest.fixture(scope="function")
def not_exists_file_name():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_not_exists")
        if os.path.exists(file_name):
            os.remove(file_name)
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture(scope="function")
def file_name_which_content_is_abcd():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_abcd")
        with ms_open(file_name, "w") as aa:
            aa.write("abcd")
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture(scope="function")
def file_name_which_permission_777():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_permission_777")
        with ms_open(file_name, "w") as aa:
            aa.write("abcd")
        os.chmod(file_name, 0o777)
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture(scope="function")
def file_name_which_is_softlink():
    with tempfile.TemporaryDirectory() as dp:
        os.chmod(dp, 0o750)
        file_name = os.path.join(dp, ".test_open_file_softlink")
        Path(f"{file_name}_src").touch()
        os.symlink(f"{file_name}_src", file_name)
        yield file_name
        if os.path.exists(file_name):
            os.remove(file_name)


def test_msopen_given_mode_w_plus_when_write_4_lettle_then_file_writed_and_read_case(not_exists_file_name):
    with ms_open(not_exists_file_name, "w+") as aa:
        aa.write("1234")
        aa.seek(os.SEEK_SET)
        content = aa.read()
    assert content == "1234"
    assert FileStat(not_exists_file_name).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(not_exists_file_name).is_owner


def test_msopen_given_mode_w_when_write_4_lettle_then_file_writed_case(not_exists_file_name):
    with ms_open(not_exists_file_name, "w") as aa:
        aa.write("1234")

    assert FileStat(not_exists_file_name).file_size == 4
    assert FileStat(not_exists_file_name).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(not_exists_file_name).is_owner


def test_msopen_given_mode_w_when_exists_file_and_write_4_lettle_then_file_writed_and_read_case(
    file_name_which_content_is_abcd,
):
    with ms_open(file_name_which_content_is_abcd, "w+") as aa:
        aa.write("1234")
        aa.seek(os.SEEK_SET)
        content = aa.read()
    assert content == "1234"
    assert FileStat(file_name_which_content_is_abcd).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(file_name_which_content_is_abcd).is_owner


def test_msopen_given_mode_x_when_write_4_lettle_then_file_writed_case(not_exists_file_name):
    with ms_open(not_exists_file_name, "x") as aa:
        aa.write("1234")

    assert FileStat(not_exists_file_name).file_size == 4
    assert FileStat(not_exists_file_name).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(not_exists_file_name).is_owner


def test_msopen_given_mode_x_when_exists_file_then_file_writed_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "x") as aa:
        aa.write("1234")


def test_msopen_given_mode_r_when_none_then_file_read_out_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "r", max_size=100) as aa:
        content = aa.read()
    assert content == "abcd"


def test_msopen_given_mode_r_plus_when_none_then_file_read_out_and_write_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "r+", max_size=100) as aa:
        content = aa.read()
        assert content == "abcd"
        aa.write("1234")


def test_msopen_given_mode_a_when_none_then_file_writed_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "a", max_size=100) as aa:
        aa.write("1234")

    assert FileStat(file_name_which_content_is_abcd).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(file_name_which_content_is_abcd).is_owner

    with ms_open(file_name_which_content_is_abcd, "r", max_size=100) as aa:
        content = aa.read()
        assert content == "abcd1234"


def test_msopen_given_mode_a_plus_when_none_then_file_write_and_read_out_case(file_name_which_content_is_abcd):
    with ms_open(file_name_which_content_is_abcd, "a+", max_size=100) as aa:
        aa.write("1234")
        aa.seek(os.SEEK_SET)
        content = aa.read()
    assert content == "abcd1234"
    assert FileStat(file_name_which_content_is_abcd).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
    assert FileStat(file_name_which_content_is_abcd).is_owner


def test_msopen_given_mode_r_when_file_not_exits_then_file_read_failed_case(not_exists_file_name):
    try:
        with ms_open(not_exists_file_name, "r", max_size=100) as aa:
            aa.read()
            assert False
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_r_max_size_2_when_none_then_file_failed_read_out_case(file_name_which_content_is_abcd):
    try:
        with ms_open(file_name_which_content_is_abcd, mode="r", max_size=3) as aa:
            assert False
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_w_when_file_permission_777_then_file_delete_before_write_case(
    file_name_which_permission_777,
):
    with ms_open(file_name_which_permission_777, mode="w") as aa:
        aa.write("1234")

    assert FileStat(file_name_which_permission_777).permission | PERMISSION_NORMAL == PERMISSION_NORMAL


def test_msopen_given_mode_a_when_file_permission_777_then_file_chmod_before_write_case(file_name_which_permission_777):
    with ms_open(file_name_which_permission_777, mode="a") as aa:
        aa.write("1234")

    assert FileStat(file_name_which_permission_777).permission | PERMISSION_NORMAL == PERMISSION_NORMAL


def test_msopen_given_mode_w_when_file_softlink_then_file_delete_before_write_case(file_name_which_is_softlink):
    try:
        with ms_open(file_name_which_is_softlink, mode="w") as aa:
            aa.write("1234")
        assert FileStat(file_name_which_is_softlink).permission | PERMISSION_NORMAL == PERMISSION_NORMAL
        assert not FileStat(file_name_which_is_softlink).is_softlink
    except OpenException as err:
        assert "Softlink is not allowed to be opened" in str(err)


def test_msopen_given_mode_a_when_file_softlink_then_write_failed_case(file_name_which_is_softlink):
    try:
        with ms_open(file_name_which_is_softlink, mode="a") as aa:
            aa.write("1234")
    except OpenException as ignore:
        assert True


def test_msopen_given_mode_w_p_600_when_file_softlink_then_file_delete_before_write_case(file_name_which_is_softlink):
    try:
        with ms_open(file_name_which_is_softlink, mode="w", write_permission=PERMISSION_KEY) as aa:
            aa.write("1234")
        assert FileStat(file_name_which_is_softlink).permission | PERMISSION_KEY == PERMISSION_KEY
        assert not FileStat(file_name_which_is_softlink).is_softlink
    except OpenException as err:
        assert "Softlink is not allowed to be opened" in str(err)


def test_msopen_given_mode_r_when_file_softlink_whitelist_empty_then_file_read_failed_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="r", max_size=100, softlink=True) as aa:
        content = aa.read()
    os.environ[RAW_INPUT_PATH] = ""

    assert not FileStat(file_name_which_is_softlink).check_basic_permission()


def test_msopen_given_mode_r_when_file_softlink_target_right_then_file_read_succeed_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="r", max_size=100, softlink=True) as aa:
        content = aa.read()
    os.environ[RAW_INPUT_PATH] = os.path.abspath(os.path.dirname(os.readlink(file_name_which_is_softlink)))

    assert FileStat(file_name_which_is_softlink).check_basic_permission()


def test_msopen_given_mode_r_when_file_softlink_target_wrong_then_file_read_failed_case(file_name_which_is_softlink):
    with ms_open(file_name_which_is_softlink, mode="r", max_size=100, softlink=True) as aa:
        content = aa.read()
    os.environ[RAW_INPUT_PATH] = "1234"

    assert not FileStat(file_name_which_is_softlink).check_basic_permission()


def test_msopen_given_other_w_parent_dir_then_file_read_failed_case():
    try:
        with tempfile.TemporaryDirectory() as dp:
            os.chmod(dp, 0o702)
            fp = os.path.join(dp, "test_file")

            with ms_open(fp, mode="w") as aa:
                aa.write("no way")
    except OpenException as ignore:
        assert True


@pytest.mark.parametrize("value, errors, expected", [
    # 非字符串类型直接返回
    (123, SanitizeErrorType.strict.value, 123),
    (45.67, SanitizeErrorType.replace.value, 45.67),
    (None, SanitizeErrorType.ignore.value, None),
    
    # 可转数值的字符串
    ("123", SanitizeErrorType.strict.value, "123"),
    ("-45.67", SanitizeErrorType.strict.value, "-45.67"),
    ("+3.14", SanitizeErrorType.strict.value, "+3.14"),
    
    # 安全字符串
    ("hello", SanitizeErrorType.strict.value, "hello"),
    ("123abc", SanitizeErrorType.strict.value, "123abc"),
])
def test_sanitize_csv_value_normal(value, errors, expected):
    assert sanitize_csv_value(value, errors) == expected


@pytest.mark.parametrize("value", [
    "=;+exploit",
    "+;=injection",
    "-;-dangerous",
    "@;@malicious"
])
def test_sanitize_csv_value_strict_raises(value):
    with pytest.raises(ValueError):
        sanitize_csv_value(value, SanitizeErrorType.strict.value)


@pytest.fixture
def mock_self():
    """模拟包含 file_stat 的 self 对象"""
    obj = MagicMock()
    obj.file_stat = MagicMock(st_uid=1000)  # 默认文件所有者 UID=1000
    return obj


def test_owner_match(mock_self, caplog):
    """测试当前用户是文件所有者的情况"""
    with patch("os.getuid", return_value=1000):  # 模拟当前用户 UID=1000
        with patch.object(logger, "warning") as mock_warning:
            result = FileStat.check_owner_or_root(mock_self)

    assert result is None
    mock_warning.assert_not_called()
    assert "operating this tool using the root" not in caplog.text
    assert "file owner is not consistent" not in caplog.text


def test_root_user(mock_self, caplog):
    """测试 root 用户操作的情况"""
    with patch("os.getuid", return_value=0):  # 模拟 root 用户
        with patch.object(logger, "warning") as mock_warning:
            result = FileStat.check_owner_or_root(mock_self)

    assert result is None
    mock_warning.assert_called_once_with(
        "You are currently operating this tool using the root user. Please be aware of the risk of privilege escalation."
    )


def test_unauthorized_user(mock_self, caplog):
    """测试未授权用户的情况"""
    with patch("os.getuid", return_value=1001):  # 模拟未授权用户
        with patch.object(logger, "warning") as mock_warning:
            result = FileStat.check_owner_or_root(mock_self)

    assert result is None
    mock_warning.assert_called_once_with(
        "The file owner is not consistent with the current user. "
        "Please be aware of the risk of owner inconsistency."
    )