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

import os
import unittest
from msprobe.visualization.utils import (load_json_file, load_data_json_file, str2float, check_directory_content,
                                         GraphConst, SerializableArgs)


class TestMappingConfig(unittest.TestCase):

    def setUp(self):
        self.yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapping.yaml")
        self.input = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")

    def test_load_json_file(self):
        result = load_json_file(self.yaml_path)
        self.assertEqual(result, {})

    def test_load_data_json_file(self):
        result = load_data_json_file(self.yaml_path)
        self.assertEqual(result, {})

    def test_str2float(self):
        result = str2float('23.4%')
        self.assertAlmostEqual(result, 0.234)
        result = str2float('2.3.4%')
        self.assertAlmostEqual(result, 0)

    def test_check_directory_content(self):
        input_type = check_directory_content(self.input)
        self.assertEqual(input_type, GraphConst.STEPS)

        input_type = check_directory_content(os.path.join(self.input, "step0"))
        self.assertEqual(input_type, GraphConst.RANKS)

        with self.assertRaises(ValueError):
            check_directory_content(os.path.join(self.input, "step1"))

        input_type = check_directory_content(os.path.join(self.input, "step0", "rank0"))
        self.assertEqual(input_type, GraphConst.FILES)

    def test_serializable_args(self):
        class TmpArgs:
            def __init__(self, a, b, c):
                self.a = a
                self.b = b
                self.c = c
        input_args1 = TmpArgs('a', 123, [1, 2, 3])
        serializable_args1 = SerializableArgs(input_args1)
        self.assertEqual(serializable_args1.__dict__, input_args1.__dict__)
        input_args2 = TmpArgs('a', 123, lambda x: print(x))
        serializable_args2 = SerializableArgs(input_args2)
        self.assertNotEqual(serializable_args2.__dict__, input_args2.__dict__)




if __name__ == '__main__':
    unittest.main()
