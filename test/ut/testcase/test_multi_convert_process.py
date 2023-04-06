import unittest
from unittest import mock

from src.compare.cmp_utils.multi_process.multi_convert_process import MultiConvertProcess


class TestUtilsMethods(unittest.TestCase):

    @staticmethod
    def _process_func(path):
        return 0, path

    def test_get_max_file_size(self):
        process = MultiConvertProcess(self._process_func, "/home/input", "/honme/output")
        self.assertTrue(process.get_max_file_size() > 0)

    def test_handle_multi_process1(self):
        with mock.patch("os.listdir", return_value=["/home/a.txt", "mapping.csv"]):
            with mock.patch("os.path.isfile", return_value=True):
                with mock.patch("os.path.getsize", return_value=10000000000000):
                    process = MultiConvertProcess(self._process_func, ["/home/input"], "/home/output")
                    ret = process.process()
        self.assertEqual(ret, 0)

    def test_handle_multi_process2(self):
        with mock.patch("os.listdir", return_value=["/home/a.txt"]):
            with mock.patch("os.path.isfile", return_value=True):
                with mock.patch("os.path.getsize", return_value=100):
                    process = MultiConvertProcess(self._process_func, ["/home/input"], "/home/output")
                    ret = process.process()
        self.assertEqual(ret, 0)

    def test_handle_multi_process3(self):
        with mock.patch("os.path.isfile", return_value=True):
            with mock.patch("os.path.getsize", return_value=100):
                process = MultiConvertProcess(self._process_func, ["/home/input3/a.bin", "/home/input2/2.bin"],
                                              "/home/output")
                ret = process.process()
        self.assertEqual(ret, 0)


if __name__ == '__main__':
    unittest.main()
