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