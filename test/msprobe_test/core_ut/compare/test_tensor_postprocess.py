#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import tempfile
from unittest import TestCase, mock

import numpy as np
import pytest

from msprobe.core.compare.tensor_postprocess.processor import (
    BaseTensorPostprocessor,
    LeftMatmulPostprocessor,
    LeftRightMatmulPostprocessor,
    RightMatmulPostprocessor,
    TensorPostprocessManager,
    _build_reverse_map,
    _load_tensor_as_numpy,
    _try_matmul,
)


class TestLoadTensorAsNumpy(TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _tmp_path(self, name):
        return os.path.join(self._tmp_dir.name, name)

    def test_load_npy(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = self._tmp_path("test.npy")
        np.save(path, data)
        result = _load_tensor_as_numpy(path)
        np.testing.assert_array_equal(result, data)

    def test_load_pt(self):
        data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        path = self._tmp_path("test.pt")
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor.load_pt_file"
        ) as mock_load_pt:
            mock_tensor = mock.MagicMock()
            mock_tensor.detach.return_value = mock_tensor
            mock_tensor.cpu.return_value = mock_tensor
            mock_tensor.numpy.return_value = data
            mock_load_pt.return_value = mock_tensor

            result = _load_tensor_as_numpy(path)
            mock_load_pt.assert_called_once_with(path, to_cpu=True)
            np.testing.assert_array_equal(result, data)

    def test_load_pth(self):
        data = np.array([7.0, 8.0], dtype=np.float32)
        path = self._tmp_path("test.pth")
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor.load_pt_file"
        ) as mock_load_pt:
            mock_tensor = mock.MagicMock()
            mock_tensor.detach.return_value = mock_tensor
            mock_tensor.cpu.return_value = mock_tensor
            mock_tensor.numpy.return_value = data
            mock_load_pt.return_value = mock_tensor

            result = _load_tensor_as_numpy(path)
            mock_load_pt.assert_called_once_with(path, to_cpu=True)
            np.testing.assert_array_equal(result, data)

    def test_unsupported_format_raises(self):
        path = self._tmp_path("test.txt")
        with open(path, "w") as f:
            f.write("hello")
        with self.assertRaises(ValueError) as ctx:
            _load_tensor_as_numpy(path)
        self.assertIn("Unsupported tensor file format", str(ctx.exception))


class TestRightMatmulPostprocessor(TestCase):

    def test_no_match_returns_original(self):
        processor = RightMatmulPostprocessor({"target_tensor_map": {"/path/to/mat.npy": ["op_a"]}})
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = processor.process(n_value, b_value, "op_b", "op_c")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_empty_tensor_map_returns_original(self):
        processor = RightMatmulPostprocessor({})
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = processor.process(n_value, b_value, "op_a", "op_a")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_none_tensor_map_returns_original(self):
        processor = RightMatmulPostprocessor({"target_tensor_map": None, "golden_tensor_map": None})
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = processor.process(n_value, b_value, "op_a", "op_a")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_both_sides_same_data_name(self):
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ) as mock_load:
            processor = RightMatmulPostprocessor({
                "target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
                "golden_tensor_map": {"/path/to/mat.npy": ["op_a"]},
            })
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "op_a", "op_a")

            self.assertEqual(mock_load.call_count, 2)
            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32))

    def test_npu_only_configured(self):
        npu_mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=npu_mat,
        ) as mock_load:
            processor = RightMatmulPostprocessor({"target_tensor_map": {"/path/to/npu_mat.npy": ["npu_op"]}})
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "npu_op", "bench_op")

            mock_load.assert_called_once_with("/path/to/npu_mat.npy")
            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)

    def test_bench_only_configured(self):
        bench_mat = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=bench_mat,
        ) as mock_load:
            processor = RightMatmulPostprocessor({"golden_tensor_map": {"/path/to/bench_mat.npy": ["bench_op"]}})
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "npu_op", "bench_op")

            mock_load.assert_called_once_with("/path/to/bench_mat.npy")
            np.testing.assert_array_equal(n_out, n_value)
            np.testing.assert_array_equal(b_out, np.array([[15.0, 18.0], [21.0, 24.0]], dtype=np.float32))

    def test_npu_and_bench_different_mats(self):
        npu_m = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        bench_m = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)

        def load_side_effect(path):
            if "npu" in path:
                return npu_m
            if "bench" in path:
                return bench_m
            raise ValueError("unexpected path")

        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            side_effect=load_side_effect,
        ) as mock_load:
            processor = RightMatmulPostprocessor({
                "target_tensor_map": {"/path/to/npu.npy": ["npu_op"]},
                "golden_tensor_map": {"/path/to/bench.npy": ["bench_op"]},
            })
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "npu_op", "bench_op")

            self.assertEqual(mock_load.call_count, 2)
            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[15.0, 18.0], [21.0, 24.0]], dtype=np.float32))

    def test_right_matmul_2d(self):
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            processor = RightMatmulPostprocessor({
                "target_tensor_map": {"/path/to/matrix.npy": ["op_a"]},
                "golden_tensor_map": {"/path/to/matrix.npy": ["op_a"]},
            })
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "op_a", "op_a")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32))


class TestTensorPostprocessManager(TestCase):

    def test_process_with_right_matmul_config(self):
        config = {
            "right_matmul": {
                "target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
                "golden_tensor_map": {"/path/to/mat.npy": ["op_a"]},
            }
        }
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            manager = TensorPostprocessManager(config)
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = manager.process(n_value, b_value, "op_a", "op_a")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32))

    def test_process_unconfigured_data_name_passthrough(self):
        config = {
            "right_matmul": {
                "target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
                "golden_tensor_map": {},
            }
        }
        manager = TensorPostprocessManager(config)
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = manager.process(n_value, b_value, "op_b", "op_c")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_process_npu_and_bench_different_data_names(self):
        config = {
            "right_matmul": {
                "target_tensor_map": {"/path/to/npu_mat.npy": ["npu_op"]},
                "golden_tensor_map": {"/path/to/bench_mat.npy": ["bench_op"]},
            }
        }
        npu_m = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        bench_m = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)

        def load_side_effect(path):
            if "npu" in path:
                return npu_m
            if "bench" in path:
                return bench_m
            raise ValueError("unexpected path")

        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            side_effect=load_side_effect,
        ):
            manager = TensorPostprocessManager(config)
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = manager.process(n_value, b_value, "npu_op", "bench_op")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[15.0, 18.0], [21.0, 24.0]], dtype=np.float32))

    def test_empty_config_all_processors_present(self):
        manager = TensorPostprocessManager()
        self.assertEqual(len(manager._processors), 3)
        self.assertIsInstance(manager._processors[0], RightMatmulPostprocessor)
        self.assertIsInstance(manager._processors[1], LeftMatmulPostprocessor)
        self.assertIsInstance(manager._processors[2], LeftRightMatmulPostprocessor)

    def test_empty_config_passthrough(self):
        manager = TensorPostprocessManager()
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = manager.process(n_value, b_value, "op_a", "op_a")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_one_tensor_multiple_data_names(self):
        config = {
            "right_matmul": {
                "target_tensor_map": {"/path/to/mat.npy": ["op_a", "op_b"]},
                "golden_tensor_map": {},
            }
        }
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            manager = TensorPostprocessManager(config)
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

            n_out, b_out = manager.process(n_value, b_value, "op_a", "bench_op")
            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)

            n_out, b_out = manager.process(n_value, b_value, "op_b", "bench_op")
            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)

            n_out, b_out = manager.process(n_value, b_value, "op_c", "bench_op")
            np.testing.assert_array_equal(n_out, n_value)
            np.testing.assert_array_equal(b_out, b_value)

    def test_mixed_modes_process(self):
        right_mat = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        left_mat = np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32)

        def load_side_effect(path):
            if "right" in path:
                return right_mat
            if "left" in path:
                return left_mat
            raise ValueError("unexpected path")

        config = {
            "right_matmul": {
                "target_tensor_map": {"/path/to/right.npy": ["op_a"]},
            },
            "left_matmul": {
                "target_tensor_map": {"/path/to/left.npy": ["op_a"]},
            },
        }
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            side_effect=load_side_effect,
        ):
            manager = TensorPostprocessManager(config)
            n_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            b_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            n_out, b_out = manager.process(n_value, b_value, "op_a", "bench_op")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)


class TestBaseTensorPostprocessor(TestCase):

    def test_cannot_instantiate_abstract(self):
        with self.assertRaises(TypeError):
            BaseTensorPostprocessor()

    def test_subclass_must_implement_process(self):
        class IncompleteProcessor(BaseTensorPostprocessor):
            pass

        with self.assertRaises(TypeError):
            IncompleteProcessor()

    def test_valid_subclass(self):
        class ValidProcessor(BaseTensorPostprocessor):
            def process(self, n_value, b_value, npu_data_name, bench_data_name):
                return n_value, b_value

        processor = ValidProcessor()
        n_out, b_out = processor.process(1, 2, "npu_op", "bench_op")
        self.assertEqual(n_out, 1)
        self.assertEqual(b_out, 2)


class TestLeftMatmulPostprocessor(TestCase):

    def test_no_match_returns_original(self):
        processor = LeftMatmulPostprocessor({"target_tensor_map": {"/path/to/mat.npy": ["op_a"]}})
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = processor.process(n_value, b_value, "op_b", "op_c")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_is_effective_empty_config(self):
        processor = LeftMatmulPostprocessor({})
        self.assertFalse(processor.is_effective())

    def test_is_effective_with_config(self):
        processor = LeftMatmulPostprocessor({"target_tensor_map": {"/path/to/mat.npy": ["op_a"]}})
        self.assertTrue(processor.is_effective())

    def test_left_matmul_2d(self):
        mat = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            processor = LeftMatmulPostprocessor({
                "target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
                "golden_tensor_map": {"/path/to/mat.npy": ["op_a"]},
            })
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "op_a", "op_a")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [9.0, 12.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[10.0, 12.0], [21.0, 24.0]], dtype=np.float32))

    def test_npu_only_configured(self):
        npu_mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=npu_mat,
        ):
            processor = LeftMatmulPostprocessor({"target_tensor_map": {"/path/to/npu_mat.npy": ["npu_op"]}})
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "npu_op", "bench_op")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)


class TestLeftRightMatmulPostprocessor(TestCase):

    def test_no_match_returns_original(self):
        processor = LeftRightMatmulPostprocessor({
            "left_target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
        })
        n_value = np.array([1.0, 2.0])
        b_value = np.array([3.0, 4.0])
        n_out, b_out = processor.process(n_value, b_value, "op_b", "op_c")
        np.testing.assert_array_equal(n_out, n_value)
        np.testing.assert_array_equal(b_out, b_value)

    def test_is_effective_empty_config(self):
        processor = LeftRightMatmulPostprocessor({})
        self.assertFalse(processor.is_effective())

    def test_is_effective_with_config(self):
        processor = LeftRightMatmulPostprocessor({
            "left_target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
        })
        self.assertTrue(processor.is_effective())

    def test_left_right_matmul(self):
        left_mat = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        right_mat = np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32)

        def load_side_effect(path):
            if "left" in path:
                return left_mat
            if "right" in path:
                return right_mat
            raise ValueError("unexpected path")

        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            side_effect=load_side_effect,
        ):
            processor = LeftRightMatmulPostprocessor({
                "left_target_tensor_map": {"/path/to/left.npy": ["op_a"]},
                "right_target_tensor_map": {"/path/to/right.npy": ["op_a"]},
                "left_golden_tensor_map": {"/path/to/left.npy": ["op_a"]},
                "right_golden_tensor_map": {"/path/to/right.npy": ["op_a"]},
            })
            n_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            b_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "op_a", "op_a")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32))

    def test_left_only(self):
        mat = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            processor = LeftRightMatmulPostprocessor({
                "left_target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
            })
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "op_a", "bench_op")

            np.testing.assert_array_equal(n_out, np.array([[2.0, 4.0], [9.0, 12.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)

    def test_right_only(self):
        mat = np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            processor = LeftRightMatmulPostprocessor({
                "right_target_tensor_map": {"/path/to/mat.npy": ["op_a"]},
            })
            n_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            b_value = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
            n_out, b_out = processor.process(n_value, b_value, "op_a", "bench_op")

            np.testing.assert_array_equal(n_out, np.array([[1.0, 6.0], [3.0, 12.0]], dtype=np.float32))
            np.testing.assert_array_equal(b_out, b_value)


class TestBuildReverseMap(TestCase):

    def test_empty_map(self):
        self.assertEqual(_build_reverse_map({}), {})

    def test_none_map(self):
        self.assertEqual(_build_reverse_map(None), {})

    def test_reverse_map(self):
        result = _build_reverse_map({
            "/path/to/mat.npy": ["op_a", "op_b"],
            "/path/to/mat2.npy": ["op_c"],
        })
        self.assertEqual(result, {
            "op_a": "/path/to/mat.npy",
            "op_b": "/path/to/mat.npy",
            "op_c": "/path/to/mat2.npy",
        })

    def test_invalid_data_names_skipped(self):
        result = _build_reverse_map({
            "/path/to/mat.npy": "not_a_list",
        })
        self.assertEqual(result, {})


class TestTryMatmul(TestCase):

    def test_none_mat_path_returns_original(self):
        value = np.array([1.0, 2.0])
        result = _try_matmul(value, None, "op_a", "right", "target")
        np.testing.assert_array_equal(result, value)

    def test_right_matmul_1d(self):
        mat = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            result = _try_matmul(value, "/path/to/mat.npy", "op_a", "right", "target")
            np.testing.assert_array_equal(result, np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))

    def test_left_matmul_1d(self):
        mat = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            return_value=mat,
        ):
            value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            result = _try_matmul(value, "/path/to/mat.npy", "op_a", "left", "target")
            np.testing.assert_array_equal(result, np.array([[2.0, 4.0], [9.0, 12.0]], dtype=np.float32))

    def test_load_failure_returns_original(self):
        with mock.patch(
            "msprobe.core.compare.tensor_postprocess.processor._load_tensor_as_numpy",
            side_effect=Exception("load failed"),
        ):
            value = np.array([1.0, 2.0])
            result = _try_matmul(value, "/path/to/bad.npy", "op_a", "right", "target")
            np.testing.assert_array_equal(result, value)


if __name__ == "__main__":
    unittest.main()