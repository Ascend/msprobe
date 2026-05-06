# coding=utf-8
import unittest
import os
import shutil
from msprobe.pytorch.compare.distributed_compare import compare_distributed
from msprobe.core.common.utils import CompareException
from ...core_ut.compare.test_acc_compare import generate_dump_json, generate_stack_json


base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'test_compare_distributed', f'rank')
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'output')


class TestUtilsMethods(unittest.TestCase):

    def setUp(self):
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(output_path, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        if os.path.exists(os.path.dirname(base_dir)):
            shutil.rmtree(os.path.dirname(base_dir))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

    def test_compare_distributed_fail(self):
        npu_dump_dir = ''
        bench_dump_dir = ''
        output_path = ''
        with self.assertRaises(CompareException) as context:
            compare_distributed(npu_dump_dir, bench_dump_dir, output_path, suffix=True)
        self.assertEqual(context.exception.code, CompareException.INVALID_PARAM_ERROR)

    def test_compare_distributed_success(self):
        generate_dump_json(base_dir)
        generate_stack_json(base_dir)
        base_dir_parent = os.path.dirname(base_dir)
        npu_dump_dir = base_dir_parent
        bench_dump_dir = base_dir_parent
        compare_distributed(npu_dump_dir, bench_dump_dir, output_path)
        output_files = os.listdir(output_path)
        self.assertTrue(any(f.endswith(".csv") for f in output_files))
