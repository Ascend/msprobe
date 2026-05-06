# coding=utf-8
import csv
import os
import shutil
import unittest
import argparse
import numpy as np
from msprobe.mindspore.compare.ms_graph_compare import GraphMSComparator

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        f'test_ms_graph_compare')


def generate_npy_data(path):
    data_path = os.path.join(path, 'rank_0/mnist/0/0')
    os.makedirs(data_path, exist_ok=True)
    array = np.full((10, 10), 44, dtype=np.float16)
    file_name = os.path.join(data_path, "op_type.op_1.0.0.1721724968854345.input.0.DefaultFormat.npy")

    np.save(file_name, array)


def generate_csv_data(path):
    data_path = os.path.join(path, 'rank_0/mnist/0/0')
    os.makedirs(data_path, exist_ok=True)
    file_name = os.path.join(data_path, f"statistic.csv")

    name_list = ['Op Type', 'Op Name', 'Task ID', 'Stream ID', 'Timestamp', 'IO', 'Slot', 'Data Size', 'Data Type',
                 'Shape', 'Max Value', 'Min Value', 'Avg Value', 'L2Norm Value']

    data = ['RefData', 'Default_Switch-op1_kernel_graph1_Data_86', 185, 41, 1724813943439680, 'output', 0, 16384,
            'float32', -4096, 1.00008, 0.999916, 1, 63.9995]

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(name_list)
        writer.writerow(data)


class TestMsGraphCompare(unittest.TestCase):

    def setUp(self):
        self.npu_data_path = os.path.join(base_dir, 'npu_data')
        self.bench_data_path = os.path.join(base_dir, 'bench_data')
        self.output_path = os.path.join(base_dir, "compare_result")
        os.makedirs(base_dir, mode=0o750, exist_ok=True)
        os.makedirs(self.npu_data_path, mode=0o750, exist_ok=True)
        os.makedirs(self.bench_data_path, mode=0o750, exist_ok=True)
        os.makedirs(self.output_path, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.npu_data_path):
            shutil.rmtree(self.npu_data_path)
        if os.path.exists(self.bench_data_path):
            shutil.rmtree(self.bench_data_path)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)

    def test_npy_mode(self):
        generate_npy_data(self.npu_data_path)
        generate_npy_data(self.bench_data_path)

        args = argparse.Namespace(
            target_path=self.npu_data_path,
            golden_path=self.bench_data_path,
            output_path=self.output_path,
            xlsx=False,
            rank=[],
            step=[]
        )
        ms_graph_comparator = GraphMSComparator(args)
        ms_graph_comparator.compare_core()

        compare_result_db, mode = ms_graph_comparator.compare_process(0, 0)
        compare_result_db = ms_graph_comparator.compare_ops(compare_result_db, mode)
        result = str(compare_result_db.values.tolist())

        files = os.listdir(self.output_path)
        op_name = 'op_type.op_1.0.0.1721724968854345.input.0.DefaultFormat.npy'
        npu_file_path = os.path.join(self.npu_data_path, f'rank_0/mnist/0/0/{op_name}')
        bench_file_path = os.path.join(self.bench_data_path, f'rank_0/mnist/0/0/{op_name}')

        result_correct = (
            f"[['{npu_file_path}', '{bench_file_path}', dtype('float16'), dtype('float16'), (10, 10), (10, 10), "
            f"44.0, 44.0, 44.0, inf, 44.0, 44.0, 44.0, inf, 'Yes', '', 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]")

        self.assertNotEqual(len(files), 0)
        self.assertEqual(result, result_correct)

    def test_statistic_mode(self):
        generate_csv_data(self.npu_data_path)
        generate_csv_data(self.bench_data_path)

        args = argparse.Namespace(
            target_path=self.npu_data_path,
            golden_path=self.bench_data_path,
            output_path=self.output_path,
            xlsx=False,
            rank=[],
            step=[]
        )
        ms_graph_comparator = GraphMSComparator(args)
        compare_result_db, mode = ms_graph_comparator.compare_process(0, 0)
        compare_result_db = ms_graph_comparator.compare_ops(compare_result_db, mode)
        result = compare_result_db.values.tolist()

        op_name = 'Default_Switch-op1_kernel_graph1_Data_86.185.output.0'
        npu_file_path = os.path.join(self.npu_data_path, 'rank_0/mnist/0/0/statistic.csv')
        bench_file_path = os.path.join(self.bench_data_path, 'rank_0/mnist/0/0/statistic.csv')
        npu_name = f'{op_name}'
        bench_name = f'{op_name}'
        npu_csv_file = f'{npu_file_path}'
        bench_csv_file = f'{bench_file_path}'

        result_correct = [
            [npu_name, bench_name, 'float32', 'float32', '-4096', '-4096', 1.0000799894332886, 0.9999160170555115, 1.0,
             63.9995002746582, 1.0000799894332886, 0.9999160170555115, 1.0, 63.9995002746582, 'Yes', '', 0.0, 0.0, 0.0,
             0.0, '0.0%', '0.0%', '0.0%', '0.0%',
             npu_csv_file, bench_csv_file
             ]
        ]

        self.assertListEqual(result, result_correct)
