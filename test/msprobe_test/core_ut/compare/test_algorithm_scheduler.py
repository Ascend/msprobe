# coding=utf-8
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

import unittest
from unittest import mock

from msprobe.core.compare.algorithm.algorithm_scheduler import AlgorithmScheduler

EXPECTED_BUILTIN_ALGORITHMS = {
    "cosine_similarity",
    "euclidean_distance",
    "five_thousand_error",
    "max_absolute_error",
    "max_relative_error",
    "one_thousand_error",
}
EXPECTED_BUILTIN_COLUMN_NAMES = {
    "Cosine",
    "EucDist",
    "Five Thousandths Err Ratio",
    "MaxAbsErr",
    "MaxRelativeErr",
    "One Thousandth Err Ratio",
}
# 真实数据比对完整表头中由框架管理的固定列（BASIC_INFO + SUMMARY_INFO + EXTRACT_INDEX + 附加列）
EXPECTED_RESERVED_COLUMNS = {
    # BASIC_INFO
    "NPU Name", "Bench Name", "NPU Dtype", "Bench Dtype",
    "NPU Tensor Shape", "Bench Tensor Shape",
    "NPU Requires_grad", "Bench Requires_grad",
    # SUMMARY_INFO
    "NPU max", "NPU min", "NPU mean", "NPU l2norm",
    "Bench max", "Bench min", "Bench mean", "Bench l2norm",
    # EXTRACT_INDEX
    "Requires_grad Consistent", "Result", "Err_message",
    # 真实数据比对附加列
    "NPU_Stack_Info", "Data_name", "Dirty Valid Len",
}


def _two_param_func(n_value, b_value):
    return 0.0, ""


def _three_param_func(n_value, b_value, args):
    return 0.0, ""


def _one_param_func(n_value):
    return 0.0, ""


class TestAlgorithmScheduler(unittest.TestCase):
    """AlgorithmScheduler 单例/发现/校验/调度行为单测。"""

    def setUp(self):
        # 每个用例隔离单例状态，避免相互污染
        AlgorithmScheduler._instance = None

    def tearDown(self):
        AlgorithmScheduler._instance = None

    # ------------------------------------------------------------------
    # 辅助：构造一个未经 __init__ 的纯净实例，用于测试纯方法/受控状态
    # ------------------------------------------------------------------
    @staticmethod
    def _bare_instance():
        inst = object.__new__(AlgorithmScheduler)
        inst._module_cache = {}
        inst.build_in_support_algorithm = []
        inst.custom_support_algorithm = []
        inst.algorithm_names = []
        inst._column_name_algorithm_mapping = {}
        inst.algorithm_column_names = []
        return inst

    # ------------------------------------------------------------------
    # 单例行为
    # ------------------------------------------------------------------
    def test_get_instance_given_multiple_calls_when_any_then_same_instance(self):
        inst1 = AlgorithmScheduler.get_instance()
        inst2 = AlgorithmScheduler.get_instance()
        inst3 = AlgorithmScheduler.get_instance()
        self.assertIs(inst1, inst2)
        self.assertIs(inst2, inst3)

    def test_direct_new_given_any_when_invoke_then_runtime_error(self):
        with self.assertRaises(RuntimeError):
            AlgorithmScheduler()

    # ------------------------------------------------------------------
    # builtin 算法自动发现
    # ------------------------------------------------------------------
    def test_make_support_algorithm_given_builtin_dir_when_load_then_all_discovered(self):
        inst = AlgorithmScheduler.get_instance()
        self.assertEqual(set(inst.build_in_support_algorithm), EXPECTED_BUILTIN_ALGORITHMS)
        self.assertEqual(inst.custom_support_algorithm, [])

    def test_validate_column_names_given_builtin_when_load_then_no_duplicate(self):
        # __init__ 已执行 _validate_column_names，未抛异常即说明列名唯一
        inst = AlgorithmScheduler.get_instance()
        self.assertEqual(set(inst.algorithm_column_names), EXPECTED_BUILTIN_COLUMN_NAMES)
        self.assertEqual(len(inst.algorithm_column_names), len(set(inst.algorithm_column_names)))

    def test_get_algorithm_names_given_loaded_when_call_then_returns_all(self):
        inst = AlgorithmScheduler.get_instance()
        self.assertEqual(set(inst.get_algorithm_names()), EXPECTED_BUILTIN_ALGORITHMS)

    def test_get_algorithm_column_names_given_loaded_when_call_then_returns_all(self):
        inst = AlgorithmScheduler.get_instance()
        self.assertEqual(set(inst.get_algorithm_column_names()), EXPECTED_BUILTIN_COLUMN_NAMES)

    # ------------------------------------------------------------------
    # column_name 去重校验
    # ------------------------------------------------------------------
    def test_validate_column_names_given_duplicate_when_validate_then_value_error(self):
        # 跳过真实加载，构造受控状态后直接测试 _validate_column_names
        with mock.patch.object(AlgorithmScheduler, "_make_support_algorithm", lambda self: None):
            inst = AlgorithmScheduler.get_instance()
        inst.build_in_support_algorithm = ["alg_a", "alg_b"]
        inst.custom_support_algorithm = []
        inst.algorithm_names = ["alg_a", "alg_b"]
        inst._column_name_algorithm_mapping = {}
        dummy_module = mock.Mock()
        with mock.patch.object(inst, "_get_module", return_value=dummy_module), \
                mock.patch.object(inst, "_get_column_name", side_effect=lambda name, mod: "dup_col"):
            with self.assertRaisesRegex(ValueError, "duplicated"):
                inst._validate_column_names()

    def test_validate_column_names_given_unique_when_validate_then_mapping_built(self):
        with mock.patch.object(AlgorithmScheduler, "_make_support_algorithm", lambda self: None):
            inst = AlgorithmScheduler.get_instance()
        inst.build_in_support_algorithm = ["alg_a", "alg_b"]
        inst.custom_support_algorithm = []
        inst.algorithm_names = ["alg_a", "alg_b"]
        inst._column_name_algorithm_mapping = {}
        dummy_module = mock.Mock()
        col_names = {"alg_a": "ColA", "alg_b": "ColB"}
        with mock.patch.object(inst, "_get_module", return_value=dummy_module), \
                mock.patch.object(inst, "_get_column_name", side_effect=lambda name, mod: col_names[name]):
            inst._validate_column_names()
        self.assertEqual(inst._column_name_algorithm_mapping, {"ColA": "alg_a", "ColB": "alg_b"})

    # ------------------------------------------------------------------
    # 保留列名校验（真实数据比对固定列）
    # ------------------------------------------------------------------
    def test_get_reserved_column_names_given_const_when_call_then_returns_real_data_fixed_columns(self):
        reserved = AlgorithmScheduler._get_reserved_column_names()
        self.assertEqual(reserved, EXPECTED_RESERVED_COLUMNS)
        # 内置算法列属于算法列，不应被列为保留列
        self.assertFalse(reserved & EXPECTED_BUILTIN_COLUMN_NAMES)

    def test_validate_column_names_given_reserved_conflict_when_validate_then_value_error(self):
        # 跳过真实加载，构造受控状态后直接测试 _validate_column_names
        with mock.patch.object(AlgorithmScheduler, "_make_support_algorithm", lambda self: None):
            inst = AlgorithmScheduler.get_instance()
        inst.build_in_support_algorithm = ["alg_a"]
        inst.custom_support_algorithm = []
        inst.algorithm_names = ["alg_a"]
        inst._column_name_algorithm_mapping = {}
        dummy_module = mock.Mock()
        with mock.patch.object(inst, "_get_module", return_value=dummy_module), \
                mock.patch.object(inst, "_get_column_name", return_value="NPU Tensor Shape"):
            with self.assertRaisesRegex(ValueError, "conflicts with a reserved"):
                inst._validate_column_names()

    def test_validate_column_names_given_reserved_conflict_when_validate_then_error_shows_reserved(self):
        with mock.patch.object(AlgorithmScheduler, "_make_support_algorithm", lambda self: None):
            inst = AlgorithmScheduler.get_instance()
        inst.build_in_support_algorithm = ["alg_a"]
        inst.custom_support_algorithm = []
        inst.algorithm_names = ["alg_a"]
        inst._column_name_algorithm_mapping = {}
        dummy_module = mock.Mock()
        with mock.patch.object(inst, "_get_module", return_value=dummy_module), \
                mock.patch.object(inst, "_get_column_name", return_value="Data_name"):
            with self.assertRaises(ValueError) as ctx:
                inst._validate_column_names()
        self.assertIn("Reserved columns", str(ctx.exception))
        self.assertIn("Dirty Valid Len", str(ctx.exception))

    def test_validate_column_names_given_not_reserved_when_validate_then_no_conflict_error(self):
        # 非保留列名不触发保留列冲突，正常建立映射
        with mock.patch.object(AlgorithmScheduler, "_make_support_algorithm", lambda self: None):
            inst = AlgorithmScheduler.get_instance()
        inst.build_in_support_algorithm = ["alg_a"]
        inst.custom_support_algorithm = []
        inst.algorithm_names = ["alg_a"]
        inst._column_name_algorithm_mapping = {}
        dummy_module = mock.Mock()
        with mock.patch.object(inst, "_get_module", return_value=dummy_module), \
                mock.patch.object(inst, "_get_column_name", return_value="ColA"):
            inst._validate_column_names()
        self.assertEqual(inst._column_name_algorithm_mapping, {"ColA": "alg_a"})

    # ------------------------------------------------------------------
    # _is_real_number
    # ------------------------------------------------------------------
    def test_is_real_number_given_int_when_check_then_true(self):
        inst = self._bare_instance()
        self.assertTrue(inst._is_real_number(1))

    def test_is_real_number_given_float_when_check_then_true(self):
        inst = self._bare_instance()
        self.assertTrue(inst._is_real_number(1.5))

    def test_is_real_number_given_bool_when_check_then_false(self):
        inst = self._bare_instance()
        self.assertFalse(inst._is_real_number(True))
        self.assertFalse(inst._is_real_number(False))

    def test_is_real_number_given_non_number_when_check_then_false(self):
        inst = self._bare_instance()
        self.assertFalse(inst._is_real_number("1"))
        self.assertFalse(inst._is_real_number(None))
        self.assertFalse(inst._is_real_number([1]))

    # ------------------------------------------------------------------
    # _validate_return_value
    # ------------------------------------------------------------------
    def test_validate_return_value_given_custom_int_when_valid_then_true(self):
        inst = self._bare_instance()
        self.assertTrue(inst._validate_return_value(1, "alg"))

    def test_validate_return_value_given_custom_float_when_valid_then_true(self):
        inst = self._bare_instance()
        self.assertTrue(inst._validate_return_value(1.5, "alg"))

    def test_validate_return_value_given_custom_str_when_valid_then_true(self):
        inst = self._bare_instance()
        self.assertTrue(inst._validate_return_value("err", "alg"))

    def test_validate_return_value_given_custom_invalid_type_when_invalid_then_value_error(self):
        inst = self._bare_instance()
        for bad in (True, False, (1, "msg"), (1,), [1], {"a": 1}, None, object()):
            with self.assertRaises(ValueError):
                inst._validate_return_value(bad, "alg")

    def test_validate_return_value_given_builtin_when_call_then_skip_validate(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["alg"]
        for value in ((1, "msg"), 0.5, None, [1, 2]):
            self.assertTrue(inst._validate_return_value(value, "alg"))

    # ------------------------------------------------------------------
    # _get_module
    # ------------------------------------------------------------------
    def test_get_module_given_unsupported_name_when_call_then_value_error(self):
        inst = self._bare_instance()
        with self.assertRaisesRegex(ValueError, "is not supported"):
            inst._get_module("not_an_algorithm")

    def test_get_module_given_loaded_when_call_twice_then_cached(self):
        inst = AlgorithmScheduler.get_instance()
        inst._module_cache.clear()
        m1 = inst._get_module("cosine_similarity")
        m2 = inst._get_module("cosine_similarity")
        self.assertIs(m1, m2)
        self.assertIn("cosine_similarity", inst._module_cache)

    def test_get_module_given_missing_compare_when_load_then_attribute_error(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["fake_alg"]
        fake_module = mock.Mock(spec=[])  # 无 compare 属性
        with mock.patch("importlib.import_module", return_value=fake_module):
            with self.assertRaisesRegex(AttributeError, "must define a compare"):
                inst._get_module("fake_alg")

    def test_get_module_given_compare_not_callable_when_load_then_type_error(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["fake_alg"]
        fake_module = mock.Mock()
        fake_module.compare = "not_callable"
        with mock.patch("importlib.import_module", return_value=fake_module):
            with self.assertRaisesRegex(TypeError, "not a callable"):
                inst._get_module("fake_alg")

    def test_get_module_given_wrong_arg_count_when_load_then_value_error(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["fake_alg"]
        for func in (_one_param_func, _three_param_func):
            fake_module = mock.Mock()
            fake_module.compare = func
            with mock.patch("importlib.import_module", return_value=fake_module):
                with self.assertRaisesRegex(ValueError, "positional arguments"):
                    inst._get_module("fake_alg")
            inst._module_cache.clear()

    def test_get_module_given_valid_signature_when_load_then_cached(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["fake_alg"]
        fake_module = mock.Mock()
        fake_module.compare = _two_param_func
        with mock.patch("importlib.import_module", return_value=fake_module) as mock_import:
            module = inst._get_module("fake_alg")
            self.assertIs(module, fake_module)
            self.assertEqual(mock_import.call_count, 1)
            # 第二次命中缓存，不应再次 import
            inst._get_module("fake_alg")
            self.assertEqual(mock_import.call_count, 1)

    def test_get_module_given_import_failure_when_load_then_propagates(self):
        inst = self._bare_instance()
        inst.custom_support_algorithm = ["fake_alg"]
        with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("No module")):
            with self.assertRaises(ModuleNotFoundError):
                inst._get_module("fake_alg")

    def test_get_module_given_custom_when_load_then_uses_custom_path(self):
        inst = self._bare_instance()
        inst.custom_support_algorithm = ["fake_alg"]
        fake_module = mock.Mock()
        fake_module.compare = _two_param_func
        with mock.patch("importlib.import_module", return_value=fake_module) as mock_import:
            inst._get_module("fake_alg")
            called_module_name = mock_import.call_args[0][0]
            self.assertIn(AlgorithmScheduler.CUSTOM_ALGORITHM_DIR_NAME, called_module_name)

    def test_get_module_given_builtin_when_load_then_uses_builtin_path(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["fake_alg"]
        fake_module = mock.Mock()
        fake_module.compare = _two_param_func
        with mock.patch("importlib.import_module", return_value=fake_module) as mock_import:
            inst._get_module("fake_alg")
            called_module_name = mock_import.call_args[0][0]
            self.assertIn(AlgorithmScheduler.BUILT_IN_ALGORITHM_DIR_NAME, called_module_name)

    # ------------------------------------------------------------------
    # _get_column_name
    # ------------------------------------------------------------------
    def test_get_column_name_given_missing_func_when_call_then_attribute_error(self):
        inst = self._bare_instance()
        fake_module = mock.Mock(spec=[])
        with self.assertRaisesRegex(AttributeError, "must define a column_name"):
            inst._get_column_name("alg", fake_module)

    def test_get_column_name_given_not_callable_when_call_then_type_error(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.column_name = "not_callable"
        with self.assertRaisesRegex(TypeError, "not a callable"):
            inst._get_column_name("alg", fake_module)

    def test_get_column_name_given_func_raises_when_call_then_value_error(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.column_name = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertRaisesRegex(ValueError, "Error occurred"):
            inst._get_column_name("alg", fake_module)

    def test_get_column_name_given_empty_return_when_call_then_value_error(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.column_name = lambda: ""
        with self.assertRaisesRegex(ValueError, "non"):
            inst._get_column_name("alg", fake_module)

    def test_get_column_name_given_non_string_return_when_call_then_value_error(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.column_name = lambda: 123
        with self.assertRaises(ValueError):
            inst._get_column_name("alg", fake_module)

    def test_get_column_name_given_valid_when_call_then_returns_name(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.column_name = lambda: "ColA"
        self.assertEqual(inst._get_column_name("alg", fake_module), "ColA")

    # ------------------------------------------------------------------
    # _call_algorithm
    # ------------------------------------------------------------------

    def test_call_algorithm_given_custom_scalar_when_call_then_wrapped_with_empty_msg(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.compare = lambda n, b: 0.5
        with mock.patch.object(inst, "_get_module", return_value=fake_module):
            result = inst._call_algorithm("alg", 1, 2)
        self.assertEqual(result, (0.5, ""))

    def test_call_algorithm_given_custom_invalid_tuple_when_call_then_unsupported(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.compare = lambda n, b: (0.5, "")
        with mock.patch.object(inst, "_get_module", return_value=fake_module):
            result = inst._call_algorithm("alg", 1, 2)
        self.assertEqual(result, ("unsupported", ""))

    def test_call_algorithm_given_builtin_tuple_when_call_then_returns_as_is(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["alg"]
        fake_module = mock.Mock()
        fake_module.compare = lambda n, b: (0.5, "msg")
        with mock.patch.object(inst, "_get_module", return_value=fake_module):
            result = inst._call_algorithm("alg", 1, 2)
        self.assertEqual(result, (0.5, "msg"))

    def test_call_algorithm_given_builtin_scalar_when_call_then_not_wrapped(self):
        inst = self._bare_instance()
        inst.build_in_support_algorithm = ["alg"]
        fake_module = mock.Mock()
        fake_module.compare = lambda n, b: 0.5
        with mock.patch.object(inst, "_get_module", return_value=fake_module):
            result = inst._call_algorithm("alg", 1, 2)
        self.assertEqual(result, 0.5)

    def test_call_compare_given_algorithm_raise_when_call_then_unsupported_with_empty_msg(self):
        inst = self._bare_instance()
        fake_module = mock.Mock()
        fake_module.compare = mock.Mock(side_effect=RuntimeError("boom"))
        with mock.patch.object(inst, "_get_module", return_value=fake_module):
            result = inst._call_compare("alg", 1, 2)
        self.assertEqual(result, ("unsupported", ""))

    # ------------------------------------------------------------------
    # compare 编排
    # ------------------------------------------------------------------
    def test_compare_given_unknown_column_when_call_then_unsupported(self):
        inst = self._bare_instance()
        inst._column_name_algorithm_mapping = {"ColA": "alg_a", "ColB": "alg_b"}
        inst.algorithm_column_names = ["ColA", "ColB"]
        results, errors = inst.compare(1, 2, column_names=["Unknown"])
        self.assertEqual(results, ["unsupported"])
        self.assertIn("No available algorithm", errors[0])

    def test_compare_given_none_columns_when_call_then_uses_all(self):
        inst = self._bare_instance()
        inst._column_name_algorithm_mapping = {"ColA": "alg_a", "ColB": "alg_b"}
        inst.algorithm_column_names = ["ColA", "ColB"]
        with mock.patch.object(inst, "_call_algorithm", return_value=(0.0, "")) as mock_call:
            results, errors = inst.compare(1, 2, column_names=None)
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_call.call_count, 2)

    def test_compare_given_empty_columns_when_call_then_uses_all(self):
        inst = self._bare_instance()
        inst._column_name_algorithm_mapping = {"ColA": "alg_a", "ColB": "alg_b"}
        inst.algorithm_column_names = ["ColA", "ColB"]
        with mock.patch.object(inst, "_call_algorithm", return_value=(0.0, "")) as mock_call:
            results, errors = inst.compare(1, 2, column_names=[])
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_call.call_count, 2)

    def test_compare_given_subset_columns_when_call_then_only_subset_invoked(self):
        inst = self._bare_instance()
        inst._column_name_algorithm_mapping = {"ColA": "alg_a", "ColB": "alg_b"}
        inst.algorithm_column_names = ["ColA", "ColB"]
        with mock.patch.object(inst, "_call_algorithm", return_value=(0.0, "")) as mock_call:
            results, errors = inst.compare(1, 2, column_names=["ColA"])
        self.assertEqual(len(results), 1)
        self.assertEqual(mock_call.call_count, 1)

    def test_compare_given_error_msg_when_call_then_error_msgs_collected(self):
        inst = self._bare_instance()
        inst._column_name_algorithm_mapping = {"ColA": "alg_a"}
        inst.algorithm_column_names = ["ColA"]
        with mock.patch.object(inst, "_call_algorithm", return_value=(0.0, "some error")):
            results, errors = inst.compare(1, 2, column_names=["ColA"])
        self.assertEqual(errors, ["some error"])

    # ------------------------------------------------------------------
    # 自定义/内置算法目录加载（含空目录、非法文件名跳过）
    # ------------------------------------------------------------------
    def test_make_support_algorithm_from_dir_given_nonexistent_when_call_then_skip(self):
        inst = self._bare_instance()
        algos = []
        with mock.patch("os.path.exists", return_value=False):
            inst._make_support_algorithm_from_dir("/no/such/dir", algos)
        self.assertEqual(algos, [])

    def test_make_support_algorithm_from_dir_given_empty_when_call_then_no_algorithm(self):
        inst = self._bare_instance()
        algos = []
        with mock.patch("os.path.exists", return_value=True), \
                mock.patch("os.listdir", return_value=[]), \
                mock.patch("os.path.isfile", return_value=True):
            inst._make_support_algorithm_from_dir("/fake/dir", algos)
        self.assertEqual(algos, [])

    def test_make_support_algorithm_from_dir_given_mixed_names_when_call_then_only_valid_kept(self):
        inst = self._bare_instance()
        algos = []
        # alg_OK.py 含大写，不匹配 [a-z0-9_]；subdir 无 .py 后缀；not_match.py 不以 alg_ 开头
        files = ["alg_valid.py", "not_match.py", "alg_OK.py", "alg_another.py", "subdir"]
        with mock.patch("os.path.exists", return_value=True), \
                mock.patch("os.listdir", return_value=files), \
                mock.patch("os.path.isfile", return_value=True):
            inst._make_support_algorithm_from_dir("/fake/dir", algos)
        self.assertEqual(algos, ["valid", "another"])

    def test_add_algorithm_file_to_list_given_valid_name_when_call_then_appended(self):
        algos = []
        with mock.patch("os.path.isfile", return_value=True):
            result = AlgorithmScheduler._add_algorithm_file_to_list("/x/alg_test.py", algos)
        self.assertTrue(result)
        self.assertEqual(algos, ["test"])

    def test_add_algorithm_file_to_list_given_invalid_name_when_call_then_skipped(self):
        algos = []
        with mock.patch("os.path.isfile", return_value=True):
            result = AlgorithmScheduler._add_algorithm_file_to_list("/x/not_match.py", algos)
        self.assertFalse(result)
        self.assertEqual(algos, [])

    def test_add_algorithm_file_to_list_given_non_file_when_call_then_skipped(self):
        algos = []
        with mock.patch("os.path.isfile", return_value=False):
            result = AlgorithmScheduler._add_algorithm_file_to_list("/x/alg_test.py", algos)
        self.assertFalse(result)
        self.assertEqual(algos, [])

    def test_add_algorithm_file_to_list_given_uppercase_when_call_then_skipped(self):
        algos = []
        with mock.patch("os.path.isfile", return_value=True):
            result = AlgorithmScheduler._add_algorithm_file_to_list("/x/alg_OK.py", algos)
        self.assertFalse(result)
        self.assertEqual(algos, [])

    def test_add_algorithm_file_to_list_given_init_py_when_call_then_silent_skip(self):
        # __init__.py 属正常文件，跳过时不应产生 warning 日志
        algos = []
        with mock.patch("os.path.isfile", return_value=True), \
                mock.patch("msprobe.core.compare.algorithm.algorithm_scheduler.logger") as mock_logger:
            result = AlgorithmScheduler._add_algorithm_file_to_list("/x/__init__.py", algos)
        self.assertFalse(result)
        self.assertEqual(algos, [])
        mock_logger.warning.assert_not_called()

    def test_add_algorithm_file_to_list_given_bad_name_when_call_then_warns(self):
        # 非 __init__.py 的不匹配文件仍需告警，便于发现命名错误的算法文件
        algos = []
        with mock.patch("os.path.isfile", return_value=True), \
                mock.patch("msprobe.core.compare.algorithm.algorithm_scheduler.logger") as mock_logger:
            result = AlgorithmScheduler._add_algorithm_file_to_list("/x/not_match.py", algos)
        self.assertFalse(result)
        self.assertEqual(algos, [])
        mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
