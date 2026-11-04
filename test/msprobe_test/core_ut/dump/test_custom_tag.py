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

import threading
import unittest

from msprobe.core.dump.custom_tag import CustomTagManager


class TestCustomTagManager(unittest.TestCase):
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

    # ---------------------------------------------------------- scope 栈
    def test_push_and_get_current_tag(self):
        self.assertIsNone(CustomTagManager.get_current_tag())
        CustomTagManager.push("tag_a")
        self.assertEqual(CustomTagManager.get_current_tag(), "tag_a")

    def test_pop_removes_top_tag(self):
        CustomTagManager.push("tag_a")
        CustomTagManager.pop()
        self.assertIsNone(CustomTagManager.get_current_tag())

    def test_pop_on_empty_stack_is_safe(self):
        CustomTagManager.pop()  # 不应抛出异常

    def test_nested_scopes_inner_tag_wins(self):
        CustomTagManager.push("outer")
        CustomTagManager.push("inner")
        self.assertEqual(CustomTagManager.get_current_tag(), "inner")

    def test_pop_restores_outer_tag(self):
        CustomTagManager.push("outer")
        CustomTagManager.push("inner")
        CustomTagManager.pop()
        self.assertEqual(CustomTagManager.get_current_tag(), "outer")

    # ------------------------------------------------------- sticky 登记表
    def test_record_and_get_sticky_tag(self):
        CustomTagManager.record_tags(
            ["Functional.gelu.0.forward", "Functional.gelu.0.backward"], "tag_a"
        )
        self.assertEqual(CustomTagManager.get_sticky_tag("Functional.gelu.0.forward"), "tag_a")
        self.assertEqual(CustomTagManager.get_sticky_tag("Functional.gelu.0.backward"), "tag_a")

    def test_get_sticky_tag_miss_returns_none(self):
        self.assertIsNone(CustomTagManager.get_sticky_tag("not_registered"))

    def test_record_tags_overwrites_same_name(self):
        CustomTagManager.record_tags(["name_a"], "tag_first")
        CustomTagManager.record_tags(["name_a"], "tag_second")
        self.assertEqual(CustomTagManager.get_sticky_tag("name_a"), "tag_second")

    def test_reset_clears_sticky_table(self):
        CustomTagManager.record_tags(["name_a", "name_b"], "tag_a")
        CustomTagManager.reset()
        self.assertIsNone(CustomTagManager.get_sticky_tag("name_a"))
        self.assertIsNone(CustomTagManager.get_sticky_tag("name_b"))

    # ------------------------------------------------------------ 线程模型
    def test_scope_stack_is_thread_local(self):
        CustomTagManager.push("main_thread_tag")
        result = {}

        def worker():
            result["outer_view"] = CustomTagManager.get_current_tag()
            CustomTagManager.push("worker_tag")
            result["self_view"] = CustomTagManager.get_current_tag()
            CustomTagManager.pop()

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        self.assertIsNone(result["outer_view"])  # 子线程看不到主线程的作用域栈
        self.assertEqual(result["self_view"], "worker_tag")  # 子线程自己的栈正常工作
        self.assertEqual(CustomTagManager.get_current_tag(), "main_thread_tag")  # 主线程不受影响

    def test_sticky_table_shared_across_threads(self):
        CustomTagManager.record_tags(["name_shared"], "tag_a")
        result = {}

        def worker():
            result["tag"] = CustomTagManager.get_sticky_tag("name_shared")

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(result["tag"], "tag_a")  # 登记表对其他线程可见

    # ------------------------------------------------------------ 递归控制
    def test_non_recursive_tag_hidden_inside_nested_module(self):
        CustomTagManager.push("anchor")
        self.assertEqual(CustomTagManager.get_current_tag(), "anchor")  # 窗口直接调用可见
        CustomTagManager.enter_module_call()
        self.assertIsNone(CustomTagManager.get_current_tag())  # 嵌套子模块内不可见
        CustomTagManager.leave_module_call()
        self.assertEqual(CustomTagManager.get_current_tag(), "anchor")  # 回到一级恢复可见

    def test_recursive_tag_visible_inside_nested_module(self):
        CustomTagManager.push("dye", recursive=True)
        CustomTagManager.enter_module_call()
        CustomTagManager.enter_module_call()
        self.assertEqual(CustomTagManager.get_current_tag(), "dye")  # 任意嵌套深度都可见
        CustomTagManager.leave_module_call()
        CustomTagManager.leave_module_call()
        self.assertEqual(CustomTagManager.get_current_tag(), "dye")

    def test_module_call_depth_counts_nesting(self):
        CustomTagManager.push("anchor")
        CustomTagManager.enter_module_call()
        CustomTagManager.enter_module_call()
        self.assertIsNone(CustomTagManager.get_current_tag())  # 二级嵌套不可见
        CustomTagManager.leave_module_call()
        self.assertIsNone(CustomTagManager.get_current_tag())  # 一级嵌套（depth=1）仍不可见
        CustomTagManager.leave_module_call()
        self.assertEqual(CustomTagManager.get_current_tag(), "anchor")

    def test_leave_module_call_never_goes_negative(self):
        CustomTagManager.leave_module_call()  # 多余的 leave 不应把 depth 减成负数
        CustomTagManager.push("anchor")
        self.assertEqual(CustomTagManager.get_current_tag(), "anchor")

    def test_pop_restores_outer_depth_snapshot(self):
        # 模拟 with 写在某个已 hook 的模块 forward 内：外层 depth=1
        CustomTagManager.enter_module_call()
        CustomTagManager.push("anchor")  # push 快照 depth=1 并清零
        self.assertEqual(CustomTagManager.get_current_tag(), "anchor")  # 窗口内重新计数
        CustomTagManager.pop()
        # 快照恢复：新 scope 里 enter 一次后 depth=2 > 0
        CustomTagManager.push("anchor2")
        CustomTagManager.enter_module_call()
        self.assertIsNone(CustomTagManager.get_current_tag())
        CustomTagManager.pop()
        CustomTagManager.leave_module_call()

    def test_pop_heals_depth_leak(self):
        # with 块内异常导致 leave 漏调（depth 泄漏）时，pop 直接恢复快照自愈
        CustomTagManager.push("anchor")
        CustomTagManager.enter_module_call()
        CustomTagManager.enter_module_call()  # 模拟泄漏
        CustomTagManager.pop()
        CustomTagManager.push("anchor2")
        self.assertEqual(CustomTagManager.get_current_tag(), "anchor2")  # 泄漏未影响新 scope
        CustomTagManager.pop()


if __name__ == "__main__":
    unittest.main()
