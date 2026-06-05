import gc
import importlib
import sys
import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch

_MSPROBE_PATCHED = "__msprobe_patched__"
_MSPROBE_ORIGINAL = "__msprobe_original__"

from msprobe.pytorch.dump.api_dump.script_wrapper import (
    wrap_jit_script_func,
    set_current_service,
    get_current_service,
    patch_dynamo_compile,
    unpatch_dynamo_compile,
    patch_triton_jitfunction_run,
    unpatch_triton_jitfunction_run,
    adapt_megatron_distributed_mappings,
    wrap_script_func,
    preprocess_func,
)


class TestWrapJitScriptFunc(unittest.TestCase):
    def setUp(self):
        self.original_script = torch.jit.script
        self.mock_api_register = MagicMock()
        self.mock_api_register.all_api_registered = True
        self.mock_api_register.register_all_api = MagicMock()
        self.mock_api_register.restore_all_api = MagicMock()

    def tearDown(self):
        torch.jit.script = self.original_script

    @patch('torch.jit.script', new_callable=MagicMock)
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register', return_value=MagicMock())
    def test_patched_script(self, mock_get_api, mock_original_script):
        mock_original_script.return_value = "mocked_result"
        mock_get_api.return_value = self.mock_api_register

        wrap_jit_script_func()

        self.assertNotEqual(torch.jit.script, self.original_script)

        result = torch.jit.script("test_input")

        mock_original_script.assert_called_once_with("test_input")
        self.assertEqual(result, "mocked_result")

        self.mock_api_register.restore_all_api.assert_called_once()
        self.mock_api_register.register_all_api.assert_called_once()

    @patch('torch.jit.script', new_callable=MagicMock)
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register')
    def test_patched_script_not_registered(self, mock_get_api, mock_original_script):
        mock_original_script.return_value = "mocked_result"
        self.mock_api_register.all_api_registered = False
        mock_get_api.return_value = self.mock_api_register

        wrap_jit_script_func()

        self.assertEqual(torch.jit.script("test_input"), "mocked_result")
        self.mock_api_register.restore_all_api.assert_not_called()
        self.mock_api_register.register_all_api.assert_not_called()


class TestSetGetCurrentService(unittest.TestCase):
    def setUp(self):
        import msprobe.pytorch.dump.api_dump.script_wrapper as sw
        sw._service_ref = None

    def test_set_and_get_service(self):
        mock_service = MagicMock()
        set_current_service(mock_service)
        result = get_current_service()
        self.assertEqual(result, mock_service)

    def test_get_service_when_none(self):
        result = get_current_service()
        self.assertIsNone(result)

    def test_set_service_weakref(self):
        mock_service = MagicMock()
        set_current_service(mock_service)
        import msprobe.pytorch.dump.api_dump.script_wrapper as sw
        self.assertIsNotNone(sw._service_ref)
        self.assertIsInstance(sw._service_ref, weakref.ref)

    def test_weakref_service_deleted(self):
        class DummyService:
            pass

        service = DummyService()
        set_current_service(service)
        self.assertEqual(get_current_service(), service)
        del service
        gc.collect()
        result = get_current_service()
        self.assertIsNone(result)


class TestPatchDynamoCompile(unittest.TestCase):
    def setUp(self):
        self._cf_mod = None
        try:
            self._cf_mod = importlib.import_module("torch._dynamo.convert_frame")
        except ImportError:
            self.skipTest("torch._dynamo not available")

        self.original_compile = getattr(self._cf_mod, '_compile', None)

    def tearDown(self):
        if self._cf_mod and self.original_compile is not None:
            self._cf_mod._compile = self.original_compile

    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register')
    def test_patch_dynamo_compile(self, mock_get_api):
        mock_reg = MagicMock()
        mock_get_api.return_value = mock_reg

        original_fn = lambda *args, **kwargs: None
        if not hasattr(self._cf_mod, '_compile'):
            self._cf_mod._compile = original_fn

        patch_dynamo_compile()

        current = self._cf_mod._compile
        self.assertTrue(hasattr(current, _MSPROBE_PATCHED))
        self.assertTrue(hasattr(current, _MSPROBE_ORIGINAL))

    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register')
    def test_patch_dynamo_compile_idempotent(self, mock_get_api):
        mock_reg = MagicMock()
        mock_get_api.return_value = mock_reg

        original_fn = lambda *args, **kwargs: None
        if not hasattr(self._cf_mod, '_compile'):
            self._cf_mod._compile = original_fn

        patch_dynamo_compile()
        first_patched = self._cf_mod._compile

        patch_dynamo_compile()
        second_patched = self._cf_mod._compile

        self.assertEqual(first_patched, second_patched)

    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register')
    def test_patch_dynamo_compile_restore_on_error(self, mock_get_api):
        mock_reg = MagicMock()
        mock_get_api.return_value = mock_reg

        def failing_compile(*args, **kwargs):
            raise RuntimeError("compile failed")

        self._cf_mod._compile = failing_compile

        patch_dynamo_compile()

        with self.assertRaises(RuntimeError):
            self._cf_mod._compile()

        mock_reg.restore_all_api.assert_called()


class TestUnpatchDynamoCompile(unittest.TestCase):
    def setUp(self):
        self._cf_mod = None
        try:
            self._cf_mod = importlib.import_module("torch._dynamo.convert_frame")
        except ImportError:
            self.skipTest("torch._dynamo not available")

    def tearDown(self):
        if self._cf_mod and hasattr(self, 'original_compile') and self.original_compile is not None:
            self._cf_mod._compile = self.original_compile

    def test_unpatch_when_not_patched(self):
        if hasattr(self._cf_mod, '_compile'):
            self.original_compile = self._cf_mod._compile
        result = unpatch_dynamo_compile()
        self.assertFalse(result)

    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.get_api_register')
    def test_unpatch_after_patch(self, mock_get_api):
        mock_reg = MagicMock()
        mock_get_api.return_value = mock_reg

        self.original_compile = getattr(self._cf_mod, '_compile', None)
        if self.original_compile is None:
            self._cf_mod._compile = lambda *a, **k: None
            self.original_compile = self._cf_mod._compile

        patch_dynamo_compile()
        result = unpatch_dynamo_compile()
        self.assertTrue(result)


class TestPatchTritonJitfunctionRun(unittest.TestCase):
    def test_triton_not_available(self):
        with patch.dict(sys.modules, {'triton': None}):
            with patch('msprobe.pytorch.dump.api_dump.script_wrapper.importlib.import_module', side_effect=ImportError):
                patch_triton_jitfunction_run()

    def test_patch_with_mock_triton(self):
        mock_jit_cls = type('JITFunction', (), {'run': lambda self, *a, **k: None})
        mock_autotuner_cls = type('Autotuner', (), {'run': lambda self, *a, **k: None})

        mock_triton = MagicMock()
        mock_triton.__version__ = "3.0.0"
        sys.modules['triton'] = mock_triton
        sys.modules['triton.runtime'] = MagicMock()
        sys.modules['triton.runtime.jit'] = MagicMock()
        sys.modules['triton.runtime.autotuner'] = MagicMock()
        sys.modules['triton.runtime.heuristics'] = MagicMock()

        sys.modules['triton.runtime'].JITFunction = mock_jit_cls
        sys.modules['triton.runtime.jit'].JITFunction = mock_jit_cls
        sys.modules['triton.runtime.autotuner'].Autotuner = mock_autotuner_cls

        try:
            patch_triton_jitfunction_run()
            self.assertTrue(getattr(mock_jit_cls.run, _MSPROBE_PATCHED, False))
        finally:
            for key in ['triton', 'triton.runtime', 'triton.runtime.jit',
                        'triton.runtime.autotuner', 'triton.runtime.heuristics']:
                sys.modules.pop(key, None)

    def test_patch_already_patched(self):
        original_run = lambda self, *a, **k: None
        setattr(original_run, _MSPROBE_PATCHED, True)

        mock_jit_cls = type('JITFunction', (), {'run': original_run})

        mock_triton = MagicMock()
        mock_triton.__version__ = "3.0.0"
        sys.modules['triton'] = mock_triton
        sys.modules['triton.runtime'] = MagicMock()
        sys.modules['triton.runtime.jit'] = MagicMock()
        sys.modules['triton.runtime.autotuner'] = MagicMock()
        sys.modules['triton.runtime.heuristics'] = MagicMock()

        sys.modules['triton.runtime'].JITFunction = mock_jit_cls
        sys.modules['triton.runtime.jit'].JITFunction = mock_jit_cls

        try:
            patch_triton_jitfunction_run()
            self.assertIs(mock_jit_cls.run, original_run)
        finally:
            for key in ['triton', 'triton.runtime', 'triton.runtime.jit',
                        'triton.runtime.autotuner', 'triton.runtime.heuristics']:
                sys.modules.pop(key, None)


class TestUnpatchTritonJitfunctionRun(unittest.TestCase):
    def test_unpatch_when_not_patched(self):
        result = unpatch_triton_jitfunction_run()
        self.assertFalse(result)

    def test_unpatch_with_mock_triton(self):
        original_run = lambda self, *a, **k: None
        patched_run = lambda self, *a, **k: None
        setattr(patched_run, _MSPROBE_PATCHED, True)
        setattr(patched_run, _MSPROBE_ORIGINAL, original_run)

        mock_jit_cls = type('JITFunction', (), {'run': patched_run})

        sys.modules['triton'] = MagicMock()
        sys.modules['triton.runtime'] = MagicMock()
        sys.modules['triton.runtime.jit'] = MagicMock()
        sys.modules['triton.runtime.autotuner'] = MagicMock()
        sys.modules['triton.runtime.heuristics'] = MagicMock()

        sys.modules['triton.runtime'].JITFunction = mock_jit_cls
        sys.modules['triton.runtime.jit'].JITFunction = mock_jit_cls

        try:
            result = unpatch_triton_jitfunction_run()
            self.assertTrue(result)
            self.assertEqual(mock_jit_cls.run, original_run)
        finally:
            for key in ['triton', 'triton.runtime', 'triton.runtime.jit',
                        'triton.runtime.autotuner', 'triton.runtime.heuristics']:
                sys.modules.pop(key, None)


class TestAdaptMegatronDistributedMappings(unittest.TestCase):
    def test_megatron_not_installed(self):
        with patch('importlib.util.find_spec', return_value=None):
            adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_single_module_all_gather(self, mock_import, mock_find_spec):
        mock_module = MagicMock()
        mock_all_gather = MagicMock()
        mock_all_gather.__str__ = lambda self: '<function all_gather_into_tensor at 0x123>'
        mock_module.dist_all_gather_func = mock_all_gather
        mock_module.dist_reduce_scatter_func = None

        mock_import.return_value = mock_module

        with patch.object(torch.distributed, 'all_gather_into_tensor', MagicMock()):
            adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_single_module_reduce_scatter(self, mock_import, mock_find_spec):
        mock_module = MagicMock()
        mock_module.dist_all_gather_func = None
        mock_reduce_scatter = MagicMock()
        mock_reduce_scatter.__str__ = lambda self: '<function reduce_scatter_tensor at 0x456>'
        mock_module.dist_reduce_scatter_func = mock_reduce_scatter

        mock_import.return_value = mock_module

        with patch.object(torch.distributed, 'reduce_scatter_tensor', MagicMock()):
            adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module', side_effect=ImportError)
    def test_adapt_import_error(self, mock_import, mock_find_spec):
        adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_unexpected_error(self, mock_import, mock_find_spec):
        mock_import.side_effect = RuntimeError("unexpected error")
        adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_all_gather_base(self, mock_import, mock_find_spec):
        mock_module = MagicMock()
        mock_all_gather = MagicMock()
        mock_all_gather.__str__ = lambda self: '<function _all_gather_base at 0x789>'
        mock_module.dist_all_gather_func = mock_all_gather
        mock_module.dist_reduce_scatter_func = None

        mock_import.return_value = mock_module

        with patch.object(torch.distributed, '_all_gather_base', MagicMock()):
            adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_reduce_scatter_base(self, mock_import, mock_find_spec):
        mock_module = MagicMock()
        mock_module.dist_all_gather_func = None
        mock_reduce_scatter = MagicMock()
        mock_reduce_scatter.__str__ = lambda self: '<function _reduce_scatter_base at 0xabc>'
        mock_module.dist_reduce_scatter_func = mock_reduce_scatter

        mock_import.return_value = mock_module

        with patch.object(torch.distributed, '_reduce_scatter_base', MagicMock()):
            adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_no_matching_functions(self, mock_import, mock_find_spec):
        mock_module = MagicMock()
        mock_all_gather = MagicMock()
        mock_all_gather.__str__ = lambda self: '<function some_other_func at 0xdef>'
        mock_module.dist_all_gather_func = mock_all_gather
        mock_module.dist_reduce_scatter_func = None

        mock_import.return_value = mock_module
        adapt_megatron_distributed_mappings()

    @patch('importlib.util.find_spec', return_value=True)
    @patch('importlib.import_module')
    def test_adapt_no_dist_funcs(self, mock_import, mock_find_spec):
        mock_module = MagicMock()
        mock_module.dist_all_gather_func = None
        mock_module.dist_reduce_scatter_func = None

        mock_import.return_value = mock_module
        adapt_megatron_distributed_mappings()


class TestWrapScriptFunc(unittest.TestCase):
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.wrap_jit_script_func')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.patch_dynamo_compile')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.patch_triton_jitfunction_run')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.adapt_megatron_distributed_mappings')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.torch_version_above_or_equal_2', True)
    def test_wrap_script_func_calls_all(self, mock_jit, mock_dynamo, mock_triton, mock_adapt):
        wrap_script_func()
        mock_jit.assert_called_once()
        mock_dynamo.assert_called_once()
        mock_triton.assert_called_once()
        mock_adapt.assert_called_once()

    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.wrap_jit_script_func')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.patch_triton_jitfunction_run')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.adapt_megatron_distributed_mappings')
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.torch_version_above_or_equal_2', False)
    def test_wrap_script_func_no_dynamo(self, mock_jit, mock_triton, mock_adapt):
        wrap_script_func()
        mock_jit.assert_called_once()
        mock_triton.assert_called_once()
        mock_adapt.assert_called_once()


class TestPreprocessFunc(unittest.TestCase):
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.logger')
    def test_preprocess_func_no_device_constructors(self, mock_logger):
        preprocess_func()

    @patch('torch.utils._device._device_constructors', side_effect=ImportError)
    def test_preprocess_func_import_error(self, mock_constructors):
        preprocess_func()

    @patch('torch.utils._device._device_constructors', side_effect=RuntimeError("test error"))
    @patch('msprobe.pytorch.dump.api_dump.script_wrapper.logger')
    def test_preprocess_func_runtime_error(self, mock_logger, mock_constructors):
        preprocess_func()
        mock_logger.warning.assert_called()
