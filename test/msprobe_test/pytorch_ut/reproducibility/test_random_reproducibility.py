import os
import random
import shutil
import tempfile
import unittest

import numpy as np
import torch

from msprobe.core.common.exceptions import MsprobeException
from msprobe.pytorch.reproducibility.random_reproducibility import set_reproducibility, random_save
from msprobe.pytorch.reproducibility.random_api_processor import GlobalRandomApiProcessor
from msprobe.pytorch.reproducibility.common import Const


def _save_original_apis():
    originals = {}
    p = GlobalRandomApiProcessor()
    api_dict = p.api_dict
    for library_name, module in Const.API_MAPPING.items():
        func_names = api_dict.get(library_name, [])
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


class TestSetReproducibility(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()

    def test_sets_python_seed(self):
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=False)
        self.assertEqual(os.environ.get('PYTHONHASHSEED'), '42')

    def test_sets_numpy_seed(self):
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=False)
        np.random.seed(42)
        expected = np.random.rand()
        np.random.seed(42)
        val = np.random.rand()
        self.assertEqual(val, expected)

    def test_sets_torch_seed(self):
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=False)
        torch.manual_seed(42)
        expected = torch.rand(1).item()
        torch.manual_seed(42)
        val = torch.rand(1).item()
        self.assertEqual(val, expected)

    def test_does_not_fix_state_when_not_enhanced(self):
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=False)
        self.assertFalse(GlobalRandomApiProcessor._has_fixed)

    def test_fixes_state_when_enhanced(self):
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=True)
        self.assertTrue(GlobalRandomApiProcessor._has_fixed)
        self.assertTrue(GlobalRandomApiProcessor._has_patched)

    def test_raises_on_invalid_seed(self):
        with self.assertRaises(MsprobeException):
            set_reproducibility(seed=-1, is_deterministic=False, is_enhanced=False)

    def test_raises_on_non_bool_deterministic(self):
        with self.assertRaises(MsprobeException):
            set_reproducibility(seed=42, is_deterministic=1, is_enhanced=False)

    def test_raises_on_non_bool_enhanced(self):
        with self.assertRaises(MsprobeException):
            set_reproducibility(seed=42, is_deterministic=False, is_enhanced=1)

    def test_enhanced_mode_patches_apis(self):
        origin_random = random.random
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=True)
        self.assertIsNot(random.random, origin_random)

    def test_enhanced_mode_produces_deterministic_results(self):
        set_reproducibility(seed=42, is_deterministic=False, is_enhanced=True)
        p = GlobalRandomApiProcessor()
        val1 = random.random()
        val2 = random.random()
        self.assertEqual(val1, val2)


class TestRandomSave(unittest.TestCase):

    def setUp(self):
        _reset_singleton()
        self.originals = _save_original_apis()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        _restore_original_apis(self.originals)
        _reset_singleton()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_csv_file(self):
        random_save(self.temp_dir)
        p = GlobalRandomApiProcessor()
        self.assertIsNotNone(p.csv_path)
        self.assertTrue(os.path.isfile(p.csv_path))

    def test_sets_has_saved_flag(self):
        random_save(self.temp_dir)
        self.assertTrue(GlobalRandomApiProcessor._has_saved)

    def test_patches_apis(self):
        origin_random = random.random
        random_save(self.temp_dir)
        self.assertIsNot(random.random, origin_random)

    def test_raises_on_non_str_output_path(self):
        with self.assertRaises(MsprobeException) as ctx:
            random_save(12345)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_raises_on_none_output_path(self):
        with self.assertRaises(MsprobeException) as ctx:
            random_save(None)
        self.assertEqual(ctx.exception.code, MsprobeException.INVALID_PARAM_ERROR)
