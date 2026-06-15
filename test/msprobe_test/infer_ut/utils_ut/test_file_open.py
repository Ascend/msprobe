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

import pytest

from msprobe.infer.utils.file_open_check import ms_open, OpenException


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


def test_msopen_given_other_w_parent_dir_then_file_read_failed_case():
    try:
        with tempfile.TemporaryDirectory() as dp:
            os.chmod(dp, 0o702)
            fp = os.path.join(dp, "test_file")

            with ms_open(fp, mode="w") as aa:
                aa.write("no way")
    except OpenException as ignore:
        assert True