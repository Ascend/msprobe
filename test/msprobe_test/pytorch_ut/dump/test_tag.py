# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
#  Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
#  MindStudio is licensed under Mulan PSL v2.
#  You can use this software according to the terms and conditions of the Mulan PSL v2.
#  You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest

from msprobe.core.common.exceptions import MsprobeException
from msprobe.core.dump.custom_tag import CustomTagManager
from msprobe.pytorch.tag import tag


class TestTag(unittest.TestCase):
    def setUp(self):
        self._clean_state()

    def tearDown(self):
        self._clean_state()

    @staticmethod
    def _clean_state():
        CustomTagManager.reset()
        if hasattr(CustomTagManager._tls, "stack"):
            del CustomTagManager._tls.stack
        if hasattr(CustomTagManager._tls, "depth"):
            del CustomTagManager._tls.depth

    # ------------------------------------------------------------ 标签校验
    def test_valid_labels_accepted(self):
        for label in ("a", "attn_t1", "layer-0_1", "x" * 128):
            with tag(label):
                self.assertEqual(CustomTagManager.get_current_tag(), label)

    def test_empty_label_rejected(self):
        with self.assertRaises(MsprobeException):
            tag("")

    def test_non_string_label_rejected(self):
        with self.assertRaises(MsprobeException):
            tag(123)
        with self.assertRaises(MsprobeException):
            tag(None)

    def test_invalid_characters_rejected(self):
        for label in ("a b", "a.b", "a/b", "a:b", "标签"):
            with self.assertRaises(MsprobeException):
                tag(label)

    def test_too_long_label_rejected(self):
        with self.assertRaises(MsprobeException):
            tag("x" * 129)

    # ------------------------------------------------------- with 上下文
    def test_context_manager_pops_on_exit(self):
        with tag("t1"):
            self.assertEqual(CustomTagManager.get_current_tag(), "t1")
        self.assertIsNone(CustomTagManager.get_current_tag())

    def test_context_manager_pops_on_exception(self):
        with self.assertRaises(ValueError):
            with tag("t1"):
                raise ValueError("boom in tag scope")
        self.assertIsNone(CustomTagManager.get_current_tag())  # 异常路径不泄漏标签

    def test_nested_context_inner_wins_and_restores(self):
        with tag("outer"):
            with tag("inner"):
                self.assertEqual(CustomTagManager.get_current_tag(), "inner")
            self.assertEqual(CustomTagManager.get_current_tag(), "outer")
        self.assertIsNone(CustomTagManager.get_current_tag())

    # ------------------------------------------------------------ 锚点语义
    def test_tag_is_non_recursive(self):
        # 锚点语义（唯一行为）：嵌套子模块内标签不可见
        with tag("anchor"):
            CustomTagManager.enter_module_call()
            self.assertIsNone(CustomTagManager.get_current_tag())
            CustomTagManager.leave_module_call()
            self.assertEqual(CustomTagManager.get_current_tag(), "anchor")

    def test_recursive_option_removed(self):
        # 递归开关已移除：传入 recursive 参数直接 TypeError，用户无法打开递归
        with self.assertRaises(TypeError):
            tag("dye", recursive=True)
        with self.assertRaises(TypeError):
            tag("dye", recursive=False)

    def test_scope_restores_depth_snapshot_on_exit(self):
        with tag("t1"):
            CustomTagManager.enter_module_call()
        # with 正常退出后，depth 快照恢复（泄漏被自愈），标签弹出
        self.assertIsNone(CustomTagManager.get_current_tag())
        with tag("anchor"):
            self.assertEqual(CustomTagManager.get_current_tag(), "anchor")


if __name__ == "__main__":
    unittest.main()
