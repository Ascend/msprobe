import unittest
import dump_data_pb2 as DD
from dump_parse.dump_data_object import build_dump_tensor


class TestUtilsMethods(unittest.TestCase):
    @staticmethod
    def make_dump_data_without_shape():
        return DD.OpInput(data=b"\000\312\232;", size=4)

    def test_build_dump_tensor(self):
        dump_data = [self.make_dump_data_without_shape()]
        try:
            build_dump_tensor(dump_data_object_data=dump_data, is_input=True, is_ffts=False)
        except Exception as e:
            self.fail(f"build dump tensor failed: {e}")
        self.assertEqual(dump_data[0].size, 4)
        self.assertEqual(dump_data[0].shape, [4])

