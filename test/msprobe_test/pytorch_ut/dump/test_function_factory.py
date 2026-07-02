import unittest
from unittest.mock import patch, MagicMock

from msprobe.pytorch.dump.function_factory import Register, _resolve_func, get_fusion_config


class TestRegister(unittest.TestCase):
    def setUp(self):
        self.reg = Register()

    def test_len_empty(self):
        self.assertEqual(len(self.reg), 0)

    def test_set_and_get(self):
        def dummy():
            pass
        self.reg["test_op"] = dummy
        self.assertEqual(self.reg["test_op"], dummy)
        self.assertEqual(len(self.reg), 1)

    def test_contains(self):
        def dummy():
            pass
        self.reg["test_op"] = dummy
        self.assertIn("test_op", self.reg)
        self.assertNotIn("nonexistent", self.reg)

    def test_multiple_registrations(self):
        def fn1():
            pass

        def fn2():
            pass
        self.reg["op1"] = fn1
        self.reg["op2"] = fn2
        self.assertEqual(len(self.reg), 2)

    def test_keys_values_items(self):
        def fn():
            pass
        self.reg["op"] = fn
        self.assertEqual(list(self.reg.keys()), ["op"])
        self.assertEqual(list(self.reg.values()), [fn])
        self.assertEqual(list(self.reg.items()), [("op", fn)])

    def test_register_decorator(self):
        @self.reg.register
        def my_func():
            pass
        self.assertIn("my_func", self.reg)
        self.assertEqual(self.reg["my_func"], my_func)

    def test_register_callable_list(self):
        def fn1():
            pass

        def fn2():
            pass
        self.reg([fn1, fn2])
        self.assertIn("fn1", self.reg)
        self.assertIn("fn2", self.reg)

    def test_register_non_callable_raises(self):
        with self.assertRaises(Exception):
            self.reg.register("not_callable")


class TestResolveFunc(unittest.TestCase):
    def test_none_func_name(self):
        self.assertIsNone(_resolve_func(None, MagicMock()))

    def test_func_exists(self):
        mock_module = MagicMock()
        dummy = MagicMock()
        setattr(mock_module, "npu_test_op", dummy)
        result = _resolve_func("npu_test_op", mock_module)
        self.assertEqual(result, dummy)

    def test_func_not_exists(self):
        mock_module = MagicMock(spec=["npu_other"])
        result = _resolve_func("npu_nonexistent", mock_module)
        self.assertIsNone(result)


class TestGetFusionConfig(unittest.TestCase):
    def setUp(self):
        """每次测试前重置缓存"""
        import msprobe.pytorch.dump.function_factory as ff
        ff._FUSION_CONFIG_CACHE = None

    @patch("msprobe.pytorch.dump.function_factory.bench_functions")
    @patch("msprobe.pytorch.dump.function_factory.load_yaml")
    @patch("msprobe.pytorch.dump.function_factory.os.path.exists", return_value=True)
    def test_load_with_valid_config(self, mock_exists, mock_load_yaml, mock_bench_functions):
        mock_bench_functions.__file__ = "/path/bench_functions/__init__.py"
        mock_load_yaml.return_value = {
            "operators": {
                "npu_fast_gelu": {"forward": "npu_fast_gelu"},
                "npu_rms_norm": {"forward": "npu_rms_norm", "backward": "npu_rms_norm_backward"},
            }
        }
        operators = get_fusion_config()
        self.assertEqual(len(operators), 2)
        self.assertIn("npu_fast_gelu", operators)
        self.assertIn("npu_rms_norm", operators)

    @patch("msprobe.pytorch.dump.function_factory.bench_functions")
    @patch("msprobe.pytorch.dump.function_factory.os.path.exists", return_value=False)
    def test_load_config_not_found(self, mock_exists, mock_bench_functions):
        mock_bench_functions.__file__ = "/path/bench_functions/__init__.py"
        operators = get_fusion_config()
        self.assertEqual(operators, {})

    @patch("msprobe.pytorch.dump.function_factory.bench_functions")
    @patch("msprobe.pytorch.dump.function_factory.load_yaml")
    @patch("msprobe.pytorch.dump.function_factory.os.path.exists", return_value=True)
    def test_load_with_empty_operators(self, mock_exists, mock_load_yaml, mock_bench_functions):
        mock_bench_functions.__file__ = "/path/bench_functions/__init__.py"
        mock_load_yaml.return_value = {"operators": {}}
        operators = get_fusion_config()
        self.assertEqual(operators, {})

    @patch("msprobe.pytorch.dump.function_factory.bench_functions")
    @patch("msprobe.pytorch.dump.function_factory.load_yaml")
    @patch("msprobe.pytorch.dump.function_factory.os.path.exists", return_value=True)
    def test_cache_used_on_second_call(self, mock_exists, mock_load_yaml, mock_bench_functions):
        """验证第二次调用使用缓存，不再读取 YAML"""
        mock_bench_functions.__file__ = "/path/bench_functions/__init__.py"
        mock_load_yaml.return_value = {
            "operators": {"npu_test": {"forward": "npu_test"}}
        }
        operators1 = get_fusion_config()
        self.assertEqual(len(operators1), 1)

        # 修改 mock 返回值，验证缓存生效（第二次调用不应再走 YAML 加载）
        mock_load_yaml.return_value = {"operators": {"other": {"forward": "other"}}}
        operators2 = get_fusion_config()
        self.assertEqual(len(operators2), 1)
        self.assertIn("npu_test", operators2)


if __name__ == "__main__":
    unittest.main()
