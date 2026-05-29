import os
import random
import shutil
import tempfile
import unittest

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from msprobe.pytorch.reproducibility.random_api_processor import GlobalRandomApiProcessor
from msprobe.pytorch.reproducibility.common import Const


def _save_original_apis():
    originals = {}
    p = GlobalRandomApiProcessor()
    for library_name, module in Const.API_MAPPING.items():
        func_names = p.api_dict.get(library_name, [])
        for name in func_names:
            if hasattr(module, name):
                originals[(library_name, name)] = getattr(module, name)
    return originals


def _restore_original_apis(originals):
    for (library_name, name), func in originals.items():
        module = Const.API_MAPPING[library_name]
        setattr(module, name, func)


def _reset_singleton():
    if GlobalRandomApiProcessor._instance is not None:
        GlobalRandomApiProcessor._instance._initialized = False
        GlobalRandomApiProcessor._instance = None
    GlobalRandomApiProcessor._has_fixed = False
    GlobalRandomApiProcessor._has_saved = False
    GlobalRandomApiProcessor._has_patched = False


class TestSingleton(unittest.TestCase):

    def setUp(self):
        _reset_singleton()

    def tearDown(self):
        _reset_singleton()

    def test_new_returns_same_instance(self):
        p1 = GlobalRandomApiProcessor()
        p2 = GlobalRandomApiProcessor()
        self.assertIs(p1, p2)

    def test_new_returns_none_when_no_instance(self):
        self.assertIsNone(GlobalRandomApiProcessor._instance)
        p = GlobalRandomApiProcessor()
        self.assertIs(GlobalRandomApiProcessor._instance, p)

    def test_init_only_runs_once(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = True
        p.state = "fake"
        p2 = GlobalRandomApiProcessor()
        self.assertTrue(p2.reset_state)
        self.assertEqual(p2.state, "fake")


class TestInit(unittest.TestCase):

    def setUp(self):
        _reset_singleton()

    def tearDown(self):
        _reset_singleton()

    def test_yaml_path_points_to_random_api_list(self):
        p = GlobalRandomApiProcessor()
        self.assertTrue(p.yaml_path.endswith("random_api_list.yaml"))
        self.assertTrue(os.path.isfile(p.yaml_path))

    def test_api_dict_loaded(self):
        p = GlobalRandomApiProcessor()
        self.assertIsInstance(p.api_dict, dict)
        self.assertIn("python_random", p.api_dict)
        self.assertIn("numpy_random", p.api_dict)
        self.assertIn("torch_random", p.api_dict)
        self.assertIn("tensor_random", p.api_dict)

    def test_default_state_values(self):
        _reset_singleton()
        p = GlobalRandomApiProcessor()
        self.assertFalse(p.reset_state)
        self.assertIsNone(p.state)
        self.assertFalse(p.enable_dump)
        self.assertIsNone(p.rank)
        self.assertIsNone(p.csv_path)
        self.assertEqual(dict(p.api_count), {})
        self.assertEqual(p._original_funcs, {})

    def test_initialized_flag_set(self):
        p = GlobalRandomApiProcessor()
        self.assertTrue(p._initialized)


class TestGetState(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        state = GlobalRandomApiProcessor._get_state()
        self.assertIn('python', state)
        self.assertIn('numpy', state)
        self.assertIn('torch_cpu', state)

    def test_python_state_is_tuple(self):
        state = GlobalRandomApiProcessor._get_state()
        self.assertIsInstance(state['python'], tuple)

    def test_numpy_state_is_tuple(self):
        state = GlobalRandomApiProcessor._get_state()
        self.assertIsInstance(state['numpy'], tuple)

    def test_torch_cpu_state_is_tensor(self):
        state = GlobalRandomApiProcessor._get_state()
        self.assertIsInstance(state['torch_cpu'], torch.Tensor)

    def test_state_reflects_current_rng(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        state = GlobalRandomApiProcessor._get_state()
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        state2 = GlobalRandomApiProcessor._get_state()
        self.assertNotEqual(state['python'], state2['python'])


class TestSetState(unittest.TestCase):

    def test_restores_python_state(self):
        random.seed(42)
        state = GlobalRandomApiProcessor._get_state()
        random.seed(99)
        GlobalRandomApiProcessor._set_state(state)
        random.seed(42)
        expected = random.random()
        random.seed(42)
        GlobalRandomApiProcessor._set_state(state)
        val_after = random.random()
        self.assertEqual(val_after, expected)

    def test_restores_numpy_state(self):
        np.random.seed(42)
        state = GlobalRandomApiProcessor._get_state()
        np.random.seed(99)
        GlobalRandomApiProcessor._set_state(state)
        np.random.seed(42)
        expected = np.random.rand()
        np.random.seed(42)
        GlobalRandomApiProcessor._set_state(state)
        val = np.random.rand()
        self.assertEqual(val, expected)

    def test_does_nothing_when_state_is_none(self):
        random.seed(42)
        val_before = random.random()
        GlobalRandomApiProcessor._set_state(None)
        val_after = random.random()
        self.assertNotEqual(val_before, val_after)

    def test_does_nothing_when_state_is_empty_dict(self):
        random.seed(42)
        val_before = random.random()
        GlobalRandomApiProcessor._set_state({})
        val_after = random.random()
        self.assertNotEqual(val_before, val_after)


class TestAnalyzeStack(unittest.TestCase):

    def test_returns_string(self):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        self.assertIsInstance(result, str)

    def test_result_contains_file_info_when_called_normally(self):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        if result:
            self.assertIn("File", result)

    @patch('msprobe.pytorch.reproducibility.random_api_processor.inspect.stack', side_effect=RuntimeError("stack error"))
    def test_handles_stack_exception(self, mock_stack):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        self.assertIn("Failed to get stack info", result)
        self.assertIn("test_api", result)

    @patch('msprobe.pytorch.reproducibility.random_api_processor.inspect.stack', return_value=[])
    def test_handles_empty_stack(self, mock_stack):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        self.assertEqual(result, "")

    @patch('msprobe.pytorch.reproducibility.random_api_processor.inspect.stack',
           return_value=[(None, "f.py", 1, "a", None, None)] * 4)
    def test_skips_frames_with_no_code(self, mock_stack):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        self.assertEqual(result, "")

    @patch('msprobe.pytorch.reproducibility.random_api_processor.inspect.stack',
           return_value=[(None, "f.py", 1, "a", None, None)] * 3 + [(None, "file.py", 10, "my_func", ["x = 1"], None)])
    def test_includes_frames_with_code(self, mock_stack):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        self.assertIn("file.py", result)
        self.assertIn("my_func", result)
        self.assertIn("x = 1", result)

    @patch('msprobe.pytorch.reproducibility.random_api_processor.inspect.stack',
           return_value=[(None, "f1.py", 1, "a", None, None)] * 3 +
                        [(None, "file1.py", 10, "func1", ["code1"], None),
                         (None, "file2.py", 20, "func2", ["code2"], None)])
    def test_includes_multiple_frames_with_code(self, mock_stack):
        result = GlobalRandomApiProcessor.analyze_stack("test_api")
        self.assertIn("file1.py", result)
        self.assertIn("file2.py", result)
        self.assertIn("code1", result)
        self.assertIn("code2", result)


class TestWriteStack(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_writes_to_csv(self):
        p = GlobalRandomApiProcessor()
        p.csv_path = os.path.join(self.temp_dir, "test.csv")
        from msprobe.core.common.file_utils import write_csv
        write_csv(Const.CSV_HEADER, p.csv_path, mode='w')
        p._write_stack("random", "python_random")
        self.assertTrue(os.path.isfile(p.csv_path))
        with open(p.csv_path, 'r') as f:
            content = f.read()
        self.assertIn("python_random.random", content)

    def test_increments_api_count(self):
        p = GlobalRandomApiProcessor()
        p.csv_path = os.path.join(self.temp_dir, "test.csv")
        from msprobe.core.common.file_utils import write_csv
        write_csv(Const.CSV_HEADER, p.csv_path, mode='w')
        p._write_stack("random", "python_random")
        self.assertEqual(p.api_count["python_random.random"], 1)
        p._write_stack("random", "python_random")
        self.assertEqual(p.api_count["python_random.random"], 2)

    @patch('msprobe.pytorch.reproducibility.random_api_processor.get_rank_id', return_value=5)
    @patch('msprobe.pytorch.reproducibility.random_api_processor.rename_csv')
    def test_updates_rank_when_none(self, mock_rename, mock_rank):
        p = GlobalRandomApiProcessor()
        p.csv_path = os.path.join(self.temp_dir, "test.csv")
        from msprobe.core.common.file_utils import write_csv
        write_csv(Const.CSV_HEADER, p.csv_path, mode='w')
        mock_rename.return_value = os.path.join(self.temp_dir, "random_rank5.csv")
        p._write_stack("random", "python_random")
        self.assertEqual(p.rank, 5)
        mock_rename.assert_called_once()

    @patch('msprobe.pytorch.reproducibility.random_api_processor.get_rank_id', return_value=None)
    def test_rank_stays_none_when_no_rank(self, mock_rank):
        p = GlobalRandomApiProcessor()
        p.csv_path = os.path.join(self.temp_dir, "test.csv")
        from msprobe.core.common.file_utils import write_csv
        write_csv(Const.CSV_HEADER, p.csv_path, mode='w')
        p._write_stack("random", "python_random")
        self.assertIsNone(p.rank)

    def test_api_name_format_with_count(self):
        p = GlobalRandomApiProcessor()
        p.csv_path = os.path.join(self.temp_dir, "test.csv")
        from msprobe.core.common.file_utils import write_csv
        write_csv(Const.CSV_HEADER, p.csv_path, mode='w')
        p._write_stack("random", "python_random")
        p._write_stack("random", "python_random")
        with open(p.csv_path, 'r') as f:
            content = f.read()
        self.assertIn("python_random.random.0", content)
        self.assertIn("python_random.random.1", content)


class TestCreateWrapper(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_wrapper_calls_origin_func(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = False
        mock_func = MagicMock(return_value=42)
        wrapper = p._create_wrapper("test", "python_random", mock_func)
        result = wrapper(1, 2, key=3)
        mock_func.assert_called_once_with(1, 2, key=3)
        self.assertEqual(result, 42)

    def test_wrapper_for_method_type(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = False
        mock_func = MagicMock(return_value=42)
        wrapper = p._create_wrapper("test", "tensor_random", mock_func)
        result = wrapper("self_obj", 1, 2)
        mock_func.assert_called_once_with("self_obj", 1, 2)
        self.assertEqual(result, 42)

    def test_wrapper_resets_state_when_enabled(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = True
        p.state = GlobalRandomApiProcessor._get_state()
        p.enable_dump = False
        mock_func = MagicMock(return_value=42)
        wrapper = p._create_wrapper("test", "python_random", mock_func)
        wrapper()
        self.assertEqual(mock_func.call_count, 1)

    def test_wrapper_writes_stack_when_dump_enabled(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = True
        p.csv_path = None
        mock_func = MagicMock(return_value=42)
        wrapper = p._create_wrapper("test", "python_random", mock_func)
        with patch.object(p, '_write_stack') as mock_write:
            wrapper()
            mock_write.assert_called_once_with("test", "python_random")

    def test_wrapper_does_not_write_stack_when_dump_disabled(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = False
        mock_func = MagicMock(return_value=42)
        wrapper = p._create_wrapper("test", "python_random", mock_func)
        with patch.object(p, '_write_stack') as mock_write:
            wrapper()
            mock_write.assert_not_called()

    def test_wrapper_preserves_func_name(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = False
        def my_original_func():
            pass
        wrapper = p._create_wrapper("test", "python_random", my_original_func)
        self.assertEqual(wrapper.__name__, "my_original_func")

    def test_wrapper_returns_origin_result(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = False
        mock_func = MagicMock(return_value="hello")
        wrapper = p._create_wrapper("test", "python_random", mock_func)
        result = wrapper()
        self.assertEqual(result, "hello")

    def test_wrapper_for_non_method_type_signature(self):
        p = GlobalRandomApiProcessor()
        p.reset_state = False
        p.enable_dump = False
        mock_func = MagicMock(return_value=42)
        wrapper = p._create_wrapper("test", "torch_random", mock_func)
        result = wrapper(1, 2)
        mock_func.assert_called_once_with(1, 2)
        self.assertEqual(result, 42)


class TestPatchFunctions(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_patches_module_attribute(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        p._patch_functions("python_random", ["random"], random)
        self.assertIsNot(random.random, origin_random)

    def test_saves_original_func(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        p._patch_functions("python_random", ["random"], random)
        self.assertIn(("python_random", "random"), p._original_funcs)
        self.assertIs(p._original_funcs[("python_random", "random")], origin_random)

    def test_skips_nonexistent_attribute(self):
        p = GlobalRandomApiProcessor()
        p._patch_functions("python_random", ["nonexistent_func_xyz"], random)
        self.assertNotIn(("python_random", "nonexistent_func_xyz"), p._original_funcs)

    def test_patches_multiple_functions(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        origin_uniform = random.uniform
        p._patch_functions("python_random", ["random", "uniform"], random)
        self.assertIsNot(random.random, origin_random)
        self.assertIsNot(random.uniform, origin_uniform)
        self.assertIn(("python_random", "random"), p._original_funcs)
        self.assertIn(("python_random", "uniform"), p._original_funcs)


class TestPatch(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_patch_sets_has_patched_flag(self):
        p = GlobalRandomApiProcessor()
        self.assertFalse(GlobalRandomApiProcessor._has_patched)
        p._patch()
        self.assertTrue(GlobalRandomApiProcessor._has_patched)

    def test_patch_only_runs_once(self):
        p = GlobalRandomApiProcessor()
        p._patch()
        origin_random = random.random
        p._patch()
        self.assertIs(random.random, origin_random)

    def test_patch_patches_all_categories(self):
        p = GlobalRandomApiProcessor()
        p._patch()
        self.assertTrue(len(p._original_funcs) > 0)
        libraries_in_originals = {key[0] for key in p._original_funcs}
        self.assertIn("python_random", libraries_in_originals)
        self.assertIn("numpy_random", libraries_in_originals)
        self.assertIn("torch_random", libraries_in_originals)
        self.assertIn("tensor_random", libraries_in_originals)


class TestFixRandomState(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_sets_reset_state_true(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(p.reset_state)

    def test_saves_state(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertIsNotNone(p.state)
        self.assertIn('python', p.state)
        self.assertIn('numpy', p.state)
        self.assertIn('torch_cpu', p.state)

    def test_sets_has_fixed_flag(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(GlobalRandomApiProcessor._has_fixed)

    def test_only_fixes_once(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        state1 = p.state
        random.seed(99)
        p.fix_random_state()
        self.assertIs(p.state, state1)

    def test_produces_deterministic_python_random(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        val1 = random.random()
        val2 = random.random()
        self.assertEqual(val1, val2)

    def test_produces_deterministic_numpy_random(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        val1 = np.random.rand()
        val2 = np.random.rand()
        self.assertEqual(val1, val2)

    def test_produces_deterministic_torch_random(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        val1 = torch.rand(1).item()
        val2 = torch.rand(1).item()
        self.assertEqual(val1, val2)

    def test_patches_apis(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(GlobalRandomApiProcessor._has_patched)


class TestSaveRandomApi(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sets_enable_dump_true(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertTrue(p.enable_dump)

    def test_creates_csv_file(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertIsNotNone(p.csv_path)
        self.assertTrue(os.path.isfile(p.csv_path))

    def test_sets_has_saved_flag(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertTrue(GlobalRandomApiProcessor._has_saved)

    def test_only_saves_once(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        csv_path1 = p.csv_path
        p.save_random_api(self.temp_dir)
        self.assertEqual(p.csv_path, csv_path1)

    def test_patches_apis(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertTrue(GlobalRandomApiProcessor._has_patched)


class TestUnpatchFunctions(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_restores_single_function(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        p._patch_functions("python_random", ["random"], random)
        self.assertIsNot(random.random, origin_random)
        p._unpatch_functions("python_random", ["random"], random)
        self.assertIs(random.random, origin_random)

    def test_removes_from_original_funcs(self):
        p = GlobalRandomApiProcessor()
        p._patch_functions("python_random", ["random"], random)
        self.assertIn(("python_random", "random"), p._original_funcs)
        p._unpatch_functions("python_random", ["random"], random)
        self.assertNotIn(("python_random", "random"), p._original_funcs)

    def test_skips_non_patched_function(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        p._unpatch_functions("python_random", ["random"], random)
        self.assertIs(random.random, origin_random)

    def test_restores_multiple_functions(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        origin_uniform = random.uniform
        p._patch_functions("python_random", ["random", "uniform"], random)
        p._unpatch_functions("python_random", ["random", "uniform"], random)
        self.assertIs(random.random, origin_random)
        self.assertIs(random.uniform, origin_uniform)


class TestUnpatch(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_unpatch_resets_has_patched_flag(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(GlobalRandomApiProcessor._has_patched)
        p._unpatch()
        self.assertFalse(GlobalRandomApiProcessor._has_patched)

    def test_unpatch_does_nothing_when_not_patched(self):
        p = GlobalRandomApiProcessor()
        origin_random = random.random
        p._unpatch()
        self.assertIs(random.random, origin_random)

    def test_unpatch_restores_all_categories(self):
        p = GlobalRandomApiProcessor()
        origin_python = random.random
        origin_numpy = np.random.rand
        origin_torch = torch.rand
        origin_tensor = torch.Tensor.uniform_
        p.fix_random_state()
        p._unpatch()
        self.assertIs(random.random, origin_python)
        self.assertIs(np.random.rand, origin_numpy)
        self.assertIs(torch.rand, origin_torch)
        self.assertIs(torch.Tensor.uniform_, origin_tensor)

    def test_unpatch_clears_original_funcs(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(len(p._original_funcs) > 0)
        p._unpatch()
        self.assertEqual(len(p._original_funcs), 0)


class TestUnfixRandomState(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_resets_reset_state(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(p.reset_state)
        p.unfix_random_state()
        self.assertFalse(p.reset_state)

    def test_clears_state(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertIsNotNone(p.state)
        p.unfix_random_state()
        self.assertIsNone(p.state)

    def test_resets_has_fixed_flag(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertTrue(GlobalRandomApiProcessor._has_fixed)
        p.unfix_random_state()
        self.assertFalse(GlobalRandomApiProcessor._has_fixed)

    def test_unpatches_when_no_save_active(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertIsNot(random.random, origin_random)
        p.unfix_random_state()
        self.assertIs(random.random, origin_random)

    def test_keeps_patch_when_save_active(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        temp_dir = tempfile.mkdtemp()
        try:
            p.save_random_api(temp_dir)
            p.unfix_random_state()
            self.assertIsNot(random.random, origin_random)
            self.assertTrue(GlobalRandomApiProcessor._has_patched)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_longer_deterministic_after_unfix(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.unfix_random_state()
        random.seed(100)
        val1 = random.random()
        val2 = random.random()
        self.assertNotEqual(val1, val2)


class TestUnsaveRandomApi(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resets_enable_dump(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertTrue(p.enable_dump)
        p.unsave_random_api()
        self.assertFalse(p.enable_dump)

    def test_clears_rank(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        p.unsave_random_api()
        self.assertIsNone(p.rank)

    def test_clears_csv_path(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        p.unsave_random_api()
        self.assertIsNone(p.csv_path)

    def test_resets_api_count(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        p.api_count["test"] = 5
        p.unsave_random_api()
        self.assertEqual(dict(p.api_count), {})

    def test_resets_has_saved_flag(self):
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertTrue(GlobalRandomApiProcessor._has_saved)
        p.unsave_random_api()
        self.assertFalse(GlobalRandomApiProcessor._has_saved)

    def test_unpatches_when_no_fix_active(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        self.assertIsNot(random.random, origin_random)
        p.unsave_random_api()
        self.assertIs(random.random, origin_random)

    def test_keeps_patch_when_fix_active(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        p.fix_random_state()
        p.unsave_random_api()
        self.assertIsNot(random.random, origin_random)
        self.assertTrue(GlobalRandomApiProcessor._has_patched)


class TestUnpatchPublic(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resets_all_flags(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.save_random_api(self.temp_dir)
        p.unpatch()
        self.assertFalse(GlobalRandomApiProcessor._has_fixed)
        self.assertFalse(GlobalRandomApiProcessor._has_saved)
        self.assertFalse(GlobalRandomApiProcessor._has_patched)

    def test_resets_all_instance_state(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.save_random_api(self.temp_dir)
        p.unpatch()
        self.assertFalse(p.reset_state)
        self.assertIsNone(p.state)
        self.assertFalse(p.enable_dump)
        self.assertIsNone(p.rank)
        self.assertIsNone(p.csv_path)
        self.assertEqual(dict(p.api_count), {})

    def test_restores_all_apis(self):
        origin_python = random.random
        origin_numpy = np.random.rand
        origin_torch = torch.rand
        origin_tensor = torch.Tensor.uniform_
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.save_random_api(self.temp_dir)
        p.unpatch()
        self.assertIs(random.random, origin_python)
        self.assertIs(np.random.rand, origin_numpy)
        self.assertIs(torch.rand, origin_torch)
        self.assertIs(torch.Tensor.uniform_, origin_tensor)

    def test_forcibly_unpatches_even_when_both_active(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.save_random_api(self.temp_dir)
        p.unpatch()
        self.assertIs(random.random, origin_random)

    def test_double_unpatch_is_safe(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.unpatch()
        p.unpatch()
        self.assertIs(random.random, origin_random)

    def test_unpatch_when_not_patched_is_noop(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.unpatch()
        self.assertIs(random.random, origin_random)

    def test_repatch_after_unpatch(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        self.assertIsNot(random.random, origin_random)
        p.unpatch()
        self.assertIs(random.random, origin_random)
        p.fix_random_state()
        self.assertIsNot(random.random, origin_random)

    def test_unfix_then_unsave_fully_unpatches(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.save_random_api(self.temp_dir)
        p.unfix_random_state()
        self.assertIsNot(random.random, origin_random)
        p.unsave_random_api()
        self.assertIs(random.random, origin_random)

    def test_unsave_then_unfix_fully_unpatches(self):
        origin_random = random.random
        p = GlobalRandomApiProcessor()
        p.save_random_api(self.temp_dir)
        p.fix_random_state()
        p.unsave_random_api()
        self.assertIsNot(random.random, origin_random)
        p.unfix_random_state()
        self.assertIs(random.random, origin_random)

    def test_python_random_behaves_normally_after_unpatch(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.unpatch()
        random.seed(100)
        val1 = random.random()
        val2 = random.random()
        self.assertNotEqual(val1, val2)

    def test_numpy_random_behaves_normally_after_unpatch(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.unpatch()
        np.random.seed(100)
        val1 = np.random.rand()
        val2 = np.random.rand()
        self.assertNotEqual(val1, val2)

    def test_torch_random_behaves_normally_after_unpatch(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.unpatch()
        torch.manual_seed(100)
        val1 = torch.rand(1).item()
        val2 = torch.rand(1).item()
        self.assertNotEqual(val1, val2)

    def test_tensor_random_behaves_normally_after_unpatch(self):
        p = GlobalRandomApiProcessor()
        p.fix_random_state()
        p.unpatch()
        torch.manual_seed(100)
        t = torch.empty(3)
        t.uniform_()
        val1 = t.clone()
        t.uniform_()
        val2 = t.clone()
        self.assertFalse(torch.equal(val1, val2))
