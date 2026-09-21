# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
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

import unittest

import torch

from msprobe.pytorch.api_accuracy_checker.acc_check.acc_check_utils import fp8_to_fp32


class TestFp8ToFp32(unittest.TestCase):

    @unittest.skipUnless(
        hasattr(torch, "float8_e4m3fn"),
        "torch does not support float8_e4m3fn",
    )
    def test_fp8_to_fp32_e4m3_matches_torch_cast(self):
        # 选取在 FP8 可表示范围内的一些值，避免饱和
        x = torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32)
        try:
            x_fp8 = x.to(torch.float8_e4m3fn)
        except RuntimeError as err:
            if "fill_empty_deterministic_" in str(err):
                self.skipTest("float8 CPU kernels not implemented in this torch build")
            raise
        y = fp8_to_fp32(x_fp8)
        y_expected = x_fp8.to(torch.float32)

        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.allclose(y, y_expected, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(
        hasattr(torch, "float8_e5m2"),
        "torch does not support float8_e5m2",
    )
    def test_fp8_to_fp32_e5m2_matches_torch_cast(self):
        x = torch.linspace(-4.0, 4.0, steps=16, dtype=torch.float32)
        try:
            x_fp8 = x.to(torch.float8_e5m2)
        except RuntimeError as err:
            if "fill_empty_deterministic_" in str(err):
                self.skipTest("float8 CPU kernels not implemented in this torch build")
            raise
        y = fp8_to_fp32(x_fp8)
        y_expected = x_fp8.to(torch.float32)

        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.allclose(y, y_expected, atol=1e-5, rtol=1e-5))

    def test_fp8_to_fp32_unsupported_dtype_raises(self):
        x = torch.ones(4, dtype=torch.float16)
        with self.assertRaises(ValueError):
            fp8_to_fp32(x)


if __name__ == "__main__":
    unittest.main()

