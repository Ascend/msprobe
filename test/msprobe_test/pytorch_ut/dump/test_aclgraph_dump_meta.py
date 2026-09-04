# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
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

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import torch

_MODULE_NAME = "msprobe.pytorch.aclgraph_dump._meta"
_EXPECTED_OP_NAMES = [
    "my_ns::acl_save",
    "my_ns::acl_tensor_save",
    "my_ns::acl_stat",
]


def _load_meta_module():
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module
    module_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "..",
        "python", "msprobe", "pytorch", "aclgraph_dump", "_meta.py",
    )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, os.path.realpath(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


class TestAclGraphDumpMeta(unittest.TestCase):
    def setUp(self):
        self.module = _load_meta_module()

    def test_register_fake_called_with_ns_op_names(self):
        """register_fake must use 'ns::op' naming, not 'ns.op'."""
        recorded_names = []

        def fake_register(op_name):
            recorded_names.append(op_name)

            def decorator(func):
                return func

            return decorator

        with patch.object(torch.library, "register_fake", side_effect=fake_register):
            self.module._register_meta()

        self.assertEqual(recorded_names, _EXPECTED_OP_NAMES)
        for op_name in recorded_names:
            self.assertNotIn(".", op_name)
            self.assertIn("::", op_name)

    def test_fake_implementations_behaviour(self):
        """Captured fake functions should return valid meta results."""
        captured = {}

        def fake_register(op_name):
            def decorator(func):
                captured[op_name] = func
                return func

            return decorator

        with patch.object(torch.library, "register_fake", side_effect=fake_register):
            self.module._register_meta()

        self.assertEqual(set(captured), set(_EXPECTED_OP_NAMES))

        x = torch.randn(4, 3)
        result = captured["my_ns::acl_save"](x, "/tmp/path")
        self.assertEqual(result.device.type, "meta")
        self.assertEqual(tuple(result.size()), tuple(x.size()))
        self.assertEqual(tuple(result.stride()), tuple(x.stride()))
        self.assertEqual(result.dtype, x.dtype)

        stats = torch.zeros(4)
        switch = torch.tensor([1.0])
        self.assertIs(captured["my_ns::acl_tensor_save"](x, "/tmp/path", "api", False, switch), x)
        self.assertIs(captured["my_ns::acl_stat"](x, stats, "tag", switch), x)

    def test_register_failure_logs_warning_instead_of_raising(self):
        def fake_register(op_name):
            raise RuntimeError(f"schema not found for {op_name}")

        with patch.object(torch.library, "register_fake", side_effect=fake_register), \
                patch.object(self.module, "logger") as mock_logger:
            self.module._register_meta()

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("fake", warning_msg)
        self.assertIn("schema not found", warning_msg)


if __name__ == "__main__":
    unittest.main()
