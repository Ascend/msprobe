import unittest
from unittest import mock

import numpy as np
import torch

from msprobe.core.common.output_postprocess import processor


class TestProcessor(unittest.TestCase):
    def setUp(self):
        processor._get_rules.cache_clear()

    def test_should_postprocess_output(self):
        with mock.patch.object(
            processor,
            "_get_rules",
            return_value={
                "enabled_postprocess_api_names": {"target": {"api_a"}},
            },
        ):
            self.assertTrue(processor.should_postprocess_output("api_a", "target"))
            self.assertFalse(processor.should_postprocess_output("api_b", "target"))

    def test_should_postprocess_output_for_compare(self):
        with mock.patch.object(
            processor,
            "_get_rules",
            return_value={
                "enabled_compare_api_names": {"target": {"npu_grouped_matmul"}},
            },
        ):
            self.assertEqual(
                processor.should_postprocess_output_for_compare("xxx/npu_grouped_matmul_forward", "target"),
                (True, "npu_grouped_matmul"),
            )
            self.assertEqual(
                processor.should_postprocess_output_for_compare("xxx/other_api_forward", "target"), (False, None)
            )

    def test_postprocess_handler(self):
        api = "npu_grouped_matmul"
        out = torch.tensor([1, 2, 3])
        args = 0
        kwargs = {"group_list": torch.tensor([1, 0, 0])}
        backend = 'target'
        expected = torch.tensor([1, 0, 0])
        with (
            mock.patch.object(
                processor,
                "_get_rules",
                return_value={
                    "acc_check_handlers": {"target": {api: "x.py:f"}},
                },
            ),
            mock.patch.object(processor, "_run_acc_check_handler", return_value=expected) as m_handler,
        ):
            got = processor.postprocess_output(api, out, args, kwargs, backend)
        m_handler.assert_called_once_with("x.py:f", api, out, args, kwargs)
        self.assertTrue(torch.equal(got, expected))

    def test_extract_valid_len_handler(self):
        api, kwargs, expected = "api", {"x": 1}, 9
        args = 0
        backend = 'target'
        with (
            mock.patch.object(
                processor,
                "_get_rules",
                return_value={
                    "compare_handlers": {"target": {api: "x.py:f"}},
                },
            ),
            mock.patch.object(processor, "_run_compare_handler", return_value=expected) as m_handler,
        ):
            got = processor.extract_valid_len(api, args, kwargs, backend)
        m_handler.assert_called_once_with("x.py:f", api, args, kwargs)
        self.assertEqual(got, expected)

    def test_postprocess_no_rule(self):
        out = torch.tensor([1, 2])
        with mock.patch.object(
            processor,
            "_get_rules",
            return_value={
                "acc_check_handlers": {"target": {}},
            },
        ):
            got = processor.postprocess_output("none", out, 0, {}, 'target')
        self.assertTrue(torch.equal(got, out))

    def test_clean_by_group_key(self):
        out = torch.tensor([1, 2, 3])
        self.assertIs(processor._clean_by_group_key("api", "k", out, {}), out)
        self.assertIs(processor._clean_by_group_key("api", "k", out, {"k": 1}), out)
        with mock.patch.object(processor, "_clean_outputs", return_value="ok") as m:
            got = processor._clean_by_group_key("api", "k", out, {"k": torch.tensor([1, 0, 0])})
        m.assert_called_once_with(out, 1)
        self.assertEqual(got, "ok")

    def test_clean_padded_outputs(self):
        self.assertIsNone(processor._clean_outputs(None, 1))
        t = torch.tensor([1, 2, 3])
        with mock.patch.object(processor, "_clean_single_tensor", return_value="x") as m:
            self.assertEqual(processor._clean_outputs(t, 2), "x")
            self.assertEqual(processor._clean_outputs((t, t), 2), ("x", "x"))
            self.assertEqual(processor._clean_outputs([t, t], 2), ["x", "x"])
            self.assertEqual(m.call_count, 5)
        other = {"a": 1}
        self.assertIs(processor._clean_outputs(other, 2), other)

    def test_clean_single_tensor(self):
        self.assertIsNone(processor._clean_single_tensor(None, 1))
        s = "abc"
        self.assertIs(processor._clean_single_tensor(s, 1), s)
        e = torch.tensor([])
        self.assertIs(processor._clean_single_tensor(e, 2), e)
        c = torch.tensor(7)
        self.assertIs(processor._clean_single_tensor(c, 2), c)
        self.assertTrue(
            torch.equal(processor._clean_single_tensor(torch.tensor([1, 2, 3]), 2), torch.tensor([1, 2, 0]))
        )
        self.assertTrue(
            torch.equal(
                processor._clean_single_tensor(torch.tensor([[1, 2], [3, 4], [5, 6]]), 1),
                torch.tensor([[1, 2], [0, 0], [0, 0]]),
            )
        )

    def test_clean_single_tensor_numpy(self):
        tensor = np.array([1, 2, 3])
        result = processor.clean_single_tensor(tensor, 2)
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(np.array_equal(result, np.array([1, 2, 0])))

    def test_get_valid_len_from_group_key_guards(self):
        self.assertEqual(
            processor._get_valid_len_from_group_key("api", "k", {"k": torch.tensor([True, False, True])}), 2
        )
        self.assertIsNone(processor._get_valid_len_from_group_key("api", "k", {"k": torch.tensor([0.1, 0.2])}))
        self.assertIsNone(
            processor._get_valid_len_from_group_key("api", "k", {"k": torch.tensor([-1, 0], dtype=torch.int32)})
        )

    def test_acc_check_handler_flow(self):
        out = torch.tensor([1])
        with mock.patch.object(processor, "_load_callable", side_effect=ValueError("bad")):
            self.assertIs(processor._run_acc_check_handler("x.py:f", "api", out, (), {}), out)
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(side_effect=RuntimeError("boom"))):
            self.assertIs(processor._run_acc_check_handler("x.py:f", "api", out, (), {}), out)
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(return_value=None)):
            self.assertIs(processor._run_acc_check_handler("x.py:f", "api", out, (), {}), out)
        new_out = torch.tensor([9])
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(return_value=new_out)):
            self.assertIs(processor._run_acc_check_handler("x.py:f", "api", out, (), {}), new_out)

    def test_compare_handler_flow(self):
        with mock.patch.object(processor, "_load_callable", side_effect=ValueError("bad")):
            self.assertIsNone(processor._run_compare_handler("x.py:f", "api", (), {}))
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(side_effect=RuntimeError("boom"))):
            self.assertIsNone(processor._run_compare_handler("x.py:f", "api", (), {}))
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(return_value=None)):
            self.assertIsNone(processor._run_compare_handler("x.py:f", "api", (), {}))
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(return_value="bad")):
            self.assertIsNone(processor._run_compare_handler("x.py:f", "api", (), {}))
        with mock.patch.object(processor, "_load_callable", return_value=mock.Mock(return_value=7)):
            self.assertEqual(processor._run_compare_handler("x.py:f", "api", (), {}), 7)

    def test_resolve_handler_py_path(self):
        trusted = "/opt/trusted"
        with mock.patch.object(processor, "_DEFAULT_TRUSTED_HANDLER_DIRS", trusted):
            self.assertEqual(
                processor._resolve_handler_py_path("builtin_handlers.py"), "/opt/trusted/builtin_handlers.py"
            )
            self.assertEqual(processor._resolve_handler_py_path("/tmp/fake.py"), "/tmp/fake.py")

    def test_load_callable(self):
        with self.assertRaises(ValueError):
            processor._load_callable("bad")

        def handler(api_name, output, args, kwargs):
            return output

        module = mock.Mock()
        module.handler = handler
        with mock.patch.object(processor, "_load_module_from_py_path", return_value=module):
            self.assertIs(processor._load_callable("/tmp/fake.py:handler"), handler)

    def test_load_builtin_handler_callable(self):
        handler = processor._load_callable("builtin_handlers.py:postprocess_by_group_index")
        self.assertTrue(callable(handler))

    def test_load_module_from_py_path(self):
        checker = mock.Mock()
        checker.common_check.return_value = "/tmp/fake.py"

        with (
            mock.patch.object(processor, "FileChecker", return_value=checker),
            mock.patch.object(processor, "_ensure_path_in_trusted_dirs", return_value=None),
            mock.patch.object(processor.os.path, "isfile", return_value=True),
            mock.patch.object(processor.importlib.util, "spec_from_file_location", return_value=None),
        ):
            with self.assertRaises(ImportError):
                processor._load_module_from_py_path("/tmp/fake.py")

        fake_module = mock.Mock()
        fake_loader = mock.Mock()
        fake_spec = mock.Mock()
        fake_spec.loader = fake_loader
        with (
            mock.patch.object(processor, "FileChecker", return_value=checker),
            mock.patch.object(processor, "_ensure_path_in_trusted_dirs", return_value=None),
            mock.patch.object(processor.os.path, "isfile", return_value=True),
            mock.patch.object(processor.importlib.util, "spec_from_file_location", return_value=fake_spec),
            mock.patch.object(processor.importlib.util, "module_from_spec", return_value=fake_module),
        ):
            got = processor._load_module_from_py_path("/tmp/fake.py")
        fake_loader.exec_module.assert_called_once_with(fake_module)
        self.assertIs(got, fake_module)

    def test_trusted_path(self):
        trusted = "/opt/trusted"
        with (
            mock.patch.object(processor, "_DEFAULT_TRUSTED_HANDLER_DIRS", trusted),
            mock.patch.object(processor.os.path, "realpath", side_effect=lambda p: p),
        ):
            with self.assertRaises(PermissionError):
                processor._ensure_path_in_trusted_dirs("/opt/not_trusted/x.py")

    def test_get_rules_cache(self):
        data = {
            "acc_check_handlers": {"target": {"api_a": "demo.py:handle"}},
            "compare_handlers": {"target": {"api_b": "demo.py:extract"}},
        }
        with mock.patch.object(processor, "load_yaml", return_value=data) as m:
            first = processor._get_rules()
            second = processor._get_rules()
        self.assertEqual(m.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["acc_check_handlers"]["target"]["api_a"], "demo.py:handle")
        self.assertEqual(first["compare_handlers"]["target"]["api_b"], "demo.py:extract")
        self.assertEqual(first["enabled_postprocess_api_names"]["target"], {"api_a"})
        self.assertEqual(first["enabled_compare_api_names"]["target"], {"api_b"})


if __name__ == "__main__":
    unittest.main()
