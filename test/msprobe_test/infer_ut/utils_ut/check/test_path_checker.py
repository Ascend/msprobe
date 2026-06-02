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

import unittest
import os
import tempfile

from msprobe.infer.utils.check import path_checker


class TestPathChecker(unittest.TestCase):

    def setUp(self) -> None:
        self.fp = tempfile.NamedTemporaryFile()
        self.dp = tempfile.TemporaryDirectory()

        self.pc = path_checker.PathChecker()

    def test_exists(self):
        self.assertEqual(str(self.pc.exists().check(self.fp.name)), "pass")

    def test_not_exists(self):
        temp_name = "lsiwmcv"
        self.assertRegex(str(self.pc.exists().check(temp_name)), "No such file or directory")

    def test_is_file(self):
        self.assertEqual(str(self.pc.is_file().check(self.fp.name)), "pass")

    def test_not_file(self):
        self.assertRegex(str(self.pc.is_file().check(self.dp.name)), "Not a file")

    def test_is_dir(self):
        self.assertEqual(str(self.pc.is_dir().check(self.dp.name)), "pass")

    def test_not_dir(self):
        self.assertRegex(str(self.pc.is_dir().check(self.fp.name)), "Not a directory")

    @unittest.skipIf(os.getuid() == 0, "any file is readable to root")
    def test_is_readable(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o400)
            self.assertEqual(str(self.pc.is_readable().check(fp.name)), "pass")

    @unittest.skipIf(os.getuid() == 0, "any file is readable to root")
    def test_is_writeable(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o600)
            self.assertEqual(str(self.pc.is_writeable().check(fp.name)), "pass")

    @unittest.skipIf(os.getuid() == 0, "any file is readable to root")
    def test_is_executable(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o500)
            self.assertEqual(str(self.pc.is_executable().check(fp.name)), "pass")

    def test_higher_perm(self):
        self.assertEqual(str(self.pc.max_perm(0o777).check(self.fp.name)), "pass")

    def test_lower_perm(self):
        self.assertRegex(str(self.pc.max_perm(0o000).check(self.fp.name)), "should not have")

    def test_invalid_perm(self):
        self.assertRaises(ValueError, self.pc.max_perm(-1).check, self.fp.name)

    def test_smaller_size(self):
        self.assertEqual(str(self.pc.max_size(2048).check(self.fp.name)), "pass")

    def test_larger_size(self):
        self.assertRegex(str(self.pc.max_size(0).check(self.dp.name)), "File size larger than expected")

    def test_right_extension(self):
        self.assertEqual(str(self.pc.check_extensions("").check(self.fp.name)), "pass")

    def test_right_extension(self):
        self.assertRegex(str(self.pc.check_extensions(".bin").check(self.fp.name)), "Wrong file suffix")

    def test_file_name_too_long(self):
        self.assertRegex(str(self.pc.exists().check("s" * 256)), "File name too long")

    def test_file_name_too_long_raise(self):
        with self.assertRaises(ValueError) as cm:
            self.pc.exists().check("s" * 256, True)

        self.assertRegex(str(cm.exception), "File name too long")

    def test_file_not_found(self):
        self.assertRegex(str(self.pc.exists().check("s" * 240)), "No such file or directory")

    def test_file_not_found_raise(self):
        with self.assertRaises(ValueError) as cm:
            self.pc.exists().check("s" * 240, True)

        self.assertRegex(str(cm.exception), "No such file or directory")

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_no_perm(self):
        self.assertRegex(str(self.pc.exists().check("/root/a")), "Permission denied")

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_no_perm_raise(self):
        with self.assertRaises(ValueError) as cm:
            self.pc.exists().check("/root/a", True)

        self.assertRegex(str(cm.exception), "Permission denied")

    def test_error_type_raise(self):
        with self.assertRaises(TypeError):
            self.pc.exists().check(2, True)

    def tearDown(self) -> None:
        self.fp.close()
        self.dp.cleanup()
