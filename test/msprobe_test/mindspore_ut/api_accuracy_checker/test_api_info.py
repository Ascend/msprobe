import unittest
import sys
import logging
import os
import mindspore
import numpy as np
from packaging.version import parse as version_parse

from msprobe.mindspore.api_accuracy_checker.api_info import ApiInfo
from msprobe.mindspore.api_accuracy_checker.utils import global_context

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)

class TestApiInfo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        class level setup_class
        """
        global_context.init(False, os.path.join(directory, "files"), "mindspore")

    def test_get_kwargs_with_null(self):
        # first load forward backward api_info
        only_kwargs_api_info_dict = {
            "input_kwargs": {
                "approximate": None,
            }
        }
        api_info = ApiInfo("only_input_kwargs_api")
        api_info.load_forward_info(only_kwargs_api_info_dict)

        self.assertTrue(api_info.check_forward_info())
        kwargs_compute_element_dict = api_info.get_kwargs()
        self.assertEqual(kwargs_compute_element_dict.get("approximate").get_parameter(), None)

    @unittest.skipIf(version_parse(np.__version__) >= version_parse("2.0.0"),
                     "mindspore is not compatible with numpy >= 2.0")
    def test_get_compute_element_list(self):
        # first load forward backward api_info
        forward_api_info_dict = {
            "input_args": [
                {
                    "type": "mindspore.Tensor",
                    "dtype": "Float32",
                    "shape": [
                        2,
                        3
                    ],
                    "Max": 3.0,
                    "Min": 1.0,
                    "data_name": "2_3_input.npy",
                }
            ],
            "input_kwargs": {
                "approximate": {
                    "type": "str",
                    "value": "tanh",
                }
            },
            "output": [
                                {
                    "type": "mindspore.Tensor",
                    "dtype": "Float32",
                    "shape": [
                        2,
                        3
                    ],
                    "Max": 3.0,
                    "Min": 1.0,
                    "data_name": "2_3_input.npy",
                }
            ],
        }

        api_info = ApiInfo("MintFuntional.gelu.0")
        api_info.load_forward_info(forward_api_info_dict)

        self.assertTrue(api_info.check_forward_info())
        self.assertFalse(api_info.check_backward_info())

        input_compute_element_list = api_info.get_compute_element_list("forward", "input")
        parameter_real = input_compute_element_list[0].get_parameter()
        parameter_target = mindspore.Tensor([1., 2., 3.])
        self.assertTrue((parameter_real == parameter_target).all())

        kwargs_compute_element_dict = api_info.get_kwargs()
        self.assertEqual(kwargs_compute_element_dict.get("approximate").get_parameter(), "tanh")

if __name__ == '__main__':
    unittest.main()
