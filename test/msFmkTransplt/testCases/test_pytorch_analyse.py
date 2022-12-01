#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

import os
import shutil
import sys
import unittest
import unittest.mock as mock

sys.path.append(os.path.abspath("../../../"))
sys.path.append(os.path.abspath("../../../src/ms_fmk_transplt"))

ANALYSE_ERROR = 1


class Args:
    def __init__(self, input_path, output_path, version='1.8.1', mode='torch_apis'):
        self.input = input_path
        self.output = output_path
        self.version = version
        self.mode = mode


def run(mock_args):
    from analysis.pytorch_analyse import PyTorchAnalyse
    from src.ms_fmk_transplt.utils import trans_utils as utils
    try:
        utils.refresh_parso_cache = mock.Mock(side_effect=mock_refresh_parso_cache())
        analyse = PyTorchAnalyse()
        analyse._PyTorchAnalyse__parse_command = mock_args
        return analyse.main()
    except Exception as exp:
        print(repr(exp))
        return ANALYSE_ERROR


def mock_refresh_parso_cache():
    pass


class TestPyTorchAnalyse(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.ms_fmk_transplt.utils import trans_utils as utils
        utils.IS_JEDI_INSTALLED = True

    def setUp(self):
        self.abs_input_path = os.path.abspath('../resources/net')
        shutil.rmtree("../test_result/", ignore_errors=True)
        os.makedirs("../test_result/analyse_result", exist_ok=True)
        self.abs_output_path = os.path.join(os.path.abspath("../test_result"), "analyse_result")
        self.has_error = False

    def test_analysis(self):
        mock_args = mock.Mock(return_value=Args(os.path.join(self.abs_input_path, "barlowtwins_amp"),
                                                self.abs_output_path))

        self.assertNotEqual(run(mock_args), ANALYSE_ERROR)

        mock_args = mock.Mock(return_value=Args(os.path.join(self.abs_input_path, "ID0329_CarPeting_Pytorch_FD-GAN"),
                                                self.abs_output_path, mode='third_party'))

        self.assertNotEqual(run(mock_args), ANALYSE_ERROR)

