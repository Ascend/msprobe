#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import unittest
import sys
import os
from unittest.mock import patch, mock_open

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))


class TestWriteCSV(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.ms_fmk_transplt.utils.trans_utils import write_csv
        cls.write_csv = write_csv

    def test_absolute_path_error(self):
        with self.assertRaises(ValueError) as context:
            TestWriteCSV.write_csv([], '/mock/dir', '/absolute/path.csv', ['col1', 'col2'])
        self.assertEqual(str(context.exception), "csv_name /absolute/path.csv should not be an absolute path")


if __name__ == '__main__':
    unittest.main()
