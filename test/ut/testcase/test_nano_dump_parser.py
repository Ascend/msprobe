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
import unittest
import stat

from resource.concatV2D import binary_stream_concatV2D
import pytest

from cmp_utils.constant.compare_error import CompareError
from dump_parse import dump_utils

OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
OPEN_MODES = stat.S_IWUSR | stat.S_IRUSR


class TestNanoDataDump(unittest.TestCase):
    def test_parse1(self):
        file_path = './ConcatV2D.concatv2.0.0.550288648'
        with os.fdopen(os.open(file_path, OPEN_FLAGS, OPEN_MODES), 'w') as fout:
            os.write(fout.fileno(), binary_stream_concatV2D)

        dump_data = dump_utils.parse_dump_file(file_path, 2)
        self.assertEqual(dump_data.op_name, "550288648")
        os.remove(file_path)

    def test_parse2(self):
        file_path = './ConcatV2D.concatv2.0.0.550288648'
        with pytest.raises(CompareError) as err:
            dump_data = dump_utils.parse_dump_file(file_path, 2)

        self.assertEqual(err.value.args[0],
                         CompareError.MSACCUCMP_INVALID_PATH_ERROR)