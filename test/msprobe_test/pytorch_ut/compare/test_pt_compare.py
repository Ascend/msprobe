# coding=utf-8
import os
import shutil
import unittest

import torch

from msprobe.core.common.utils import CompareException
from msprobe.pytorch.compare.pt_compare import pt_compare
from ...core_ut.compare.test_acc_compare import generate_dump_json, generate_stack_json

base_dir1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare1')
base_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_pt_compare2')


def generate_bf16_pt(base_dir):
    data_path = os.path.join(base_dir, 'bf16.pt')
    data = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
    torch.save(data, data_path)


def generate_dict_pt(base_dir):
    data_path = os.path.join(base_dir, 'dict.pt')
    torch.save({1: 0}, data_path)


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir1, mode=0o750, exist_ok=True)
        os.makedirs(base_dir2, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir1):
            shutil.rmtree(base_dir1)
        if os.path.exists(base_dir2):
            shutil.rmtree(base_dir2)

    def test_compare(self):
        generate_dump_json(base_dir2)
        generate_stack_json(base_dir2)

        dump_path = os.path.join(base_dir2, 'dump.json')

        input_param = {
            'npu_path': dump_path,
            'bench_path': dump_path,
            'is_print_compare_log': True
        }
        output_path = base_dir2

        pt_compare(input_param, output_path)
        output_files = os.listdir(output_path)
        self.assertTrue(any(f.endswith(".csv") for f in output_files))

        input_param2 = {
            'npu_path': '',
            'bench_path': dump_path,
            'is_print_compare_log': True
        }
        with self.assertRaises(CompareException) as context:
            pt_compare(input_param2, output_path)
        self.assertEqual(context.exception.code, 1)
