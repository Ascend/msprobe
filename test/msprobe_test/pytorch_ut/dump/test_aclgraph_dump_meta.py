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
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


class TestAclGraphDumpMeta(unittest.TestCase):
    def test_import_uses_existing_meta_kernels(self):
        # Model the dispatch registrations installed by the C++ extension.
        # A unique namespace keeps this test independent of any loaded extension.
        namespace = "aclgraph_meta_test_" + uuid4().hex
        library = torch.library.Library(namespace, "DEF")
        self.addCleanup(library._destroy)
        library.define("acl_save(Tensor x, str path) -> Tensor")
        library.define(
            "acl_tensor_save(Tensor x, str path, str api_name, "
            "bool is_call_start=False, Tensor? switch=None) -> Tensor"
        )
        library.define("acl_stat(Tensor x, Tensor? stats, str tag, Tensor? switch=None) -> Tensor")
        library.impl("acl_save", lambda x, path: torch.empty_like(x, device="meta"), "Meta")
        library.impl("acl_tensor_save", lambda x, *args: x, "Meta")
        library.impl("acl_stat", lambda x, *args: x, "Meta")
        ops = getattr(torch.ops, namespace)

        module_name = "msprobe.pytorch.aclgraph_dump"
        module_path = (
            Path(__file__).resolve().parents[4]
            / "python/msprobe/pytorch/aclgraph_dump/__init__.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        fake_lib = types.ModuleType("msprobe.lib")
        fake_lib.aclgraph_dump_ext = types.ModuleType("aclgraph_dump_ext")
        fake_log = types.ModuleType("msprobe.core.common.log")
        fake_log.logger = MagicMock()
        modules = {
            module_name: module,
            "msprobe.lib": fake_lib,
            "msprobe.core.common.log": fake_log,
            "torch_npu": types.ModuleType("torch_npu"),
        }
        with patch.dict(sys.modules, modules), \
                patch.object(torch.ops, "my_ns", ops), \
                patch("torch.fx.node.has_side_effect") as mark_side_effect, \
                patch.object(torch.library, "register_fake") as register_fake:
            spec.loader.exec_module(module)
            register_fake.assert_not_called()
            fake_log.logger.warning.assert_not_called()
            self.assertEqual(
                [call.args[0] for call in mark_side_effect.call_args_list],
                [ops.acl_save.default, ops.acl_tensor_save.default, ops.acl_stat.default],
            )
            self._check_outputs(module, torch.empty(4, 3, device="meta"))
            with FakeTensorMode():
                self._check_outputs(module, torch.empty(4, 3))

    def _check_outputs(self, module, x):
        for op, args in (
            (module.acl_save, ("unused",)),
            (module.acl_tensor_save, ("unused", "api")),
            (module.acl_stat, ("tag",)),
        ):
            with self.subTest(op=op.__name__, device=x.device):
                result = op(x, *args)
                self.assertEqual(result.shape, x.shape)
                self.assertEqual(result.stride(), x.stride())
                self.assertEqual(result.dtype, x.dtype)
                self.assertEqual(result.device, x.device)


if __name__ == "__main__":
    unittest.main()
