import functools
import inspect
import os
import random
from collections import defaultdict
from typing import List, Callable, Any

import numpy as np
import torch

from msprobe.core.common.file_utils import load_yaml, write_csv
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.reproducibility.common import (
    Const,
    gpu_available,
    npu_available,
    create_csv,
    get_rank_id,
    rename_csv,
)


class GlobalRandomApiProcessor:
    _instance = None
    _has_fixed = False
    _has_saved = False
    _has_patched = False
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "random_api_list.yaml")
        self.api_dict = load_yaml(self.yaml_path)
        self.reset_state = False
        self.state = None
        self.enable_dump = False
        self.rank = None
        self.csv_path = None
        self.api_count = defaultdict(int)
        self._original_funcs = {}
        self._initialized = True

    @staticmethod
    def _get_state() -> Any:
        state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state()}
        if gpu_available:
            state['torch_gpu'] = torch.cuda.get_rng_state_all()
        if npu_available:
            state['torch_npu'] = torch.npu.get_rng_state_all()
        return state

    @staticmethod
    def _set_state(state: Any) -> None:
        if not state:
            return

        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        if gpu_available and 'torch_gpu' in state:
            torch.cuda.set_rng_state_all(state['torch_gpu'])
        if npu_available and 'torch_npu' in state:
            torch.npu.set_rng_state_all(state['torch_npu'])

    @staticmethod
    def analyze_stack(name) -> str:
        stack_str = []
        try:
            api_stack = inspect.stack()[3:]
        except Exception as e:
            failed_info = f"Failed to get stack info for {name}: {e}."
            logger.warning(failed_info)
            stack_str.append(failed_info)
            api_stack = None

        if api_stack:
            for _, path, line, func, code, _ in api_stack:
                if not code:
                    continue
                stack_line = f"File {path}, line {str(line)}, in {func}, \n {code[0].strip()} \n"
                stack_str.append(stack_line)

        return "".join(stack_str)

    def _write_stack(self, name: str, library: str) -> None:
        prefix_name = f"{library}.{name}"
        api_name = f"{prefix_name}.{self.api_count[prefix_name]}"
        stack_info = self.analyze_stack(api_name)
        if self.rank is None:
            cur_rank = get_rank_id()
            if cur_rank is not None:
                self.rank = cur_rank
                self.csv_path = rename_csv(self.csv_path, cur_rank)

        write_csv([[api_name, stack_info]], self.csv_path)
        self.api_count[prefix_name] += 1
        logger.debug(f"The {api_name} has been written to {self.csv_path}.")

    def _create_wrapper(self, api_name: str, library: str, origin_func: Callable) -> Callable:
        is_method = library in Const.METHOD_LIST
        if is_method:

            @functools.wraps(origin_func)
            def wrapper(obj, *args, **kwargs):
                if self.reset_state:
                    self._set_state(self.state)
                result = origin_func(obj, *args, **kwargs)
                if self.reset_state:
                    self._set_state(self.state)
                if self.enable_dump:
                    self._write_stack(api_name, library)
                return result
        else:

            @functools.wraps(origin_func)
            def wrapper(*args, **kwargs):
                if self.reset_state:
                    self._set_state(self.state)
                result = origin_func(*args, **kwargs)
                if self.reset_state:
                    self._set_state(self.state)
                if self.enable_dump:
                    self._write_stack(api_name, library)
                return result

        return wrapper

    def _patch_functions(self, library: str, func_names: List[str], module) -> None:
        for name in func_names:
            if hasattr(module, name):
                origin_func = getattr(module, name)
                self._original_funcs[(library, name)] = origin_func
                wrapped_func = self._create_wrapper(api_name=name, library=library, origin_func=origin_func)
                setattr(module, name, wrapped_func)

    def _patch(self) -> None:
        if GlobalRandomApiProcessor._has_patched:
            return
        for library_name, func_names in self.api_dict.items():
            self._patch_functions(library_name, func_names, Const.API_MAPPING[library_name])
        GlobalRandomApiProcessor._has_patched = True

    def fix_random_state(self) -> None:
        if GlobalRandomApiProcessor._has_fixed:
            return
        self.reset_state = True
        self.state = self._get_state()
        self._patch()
        GlobalRandomApiProcessor._has_fixed = True

    def save_random_api(self, output_path) -> None:
        if GlobalRandomApiProcessor._has_saved:
            return
        self.enable_dump = True
        self.rank = get_rank_id()
        self.csv_path = create_csv(output_path, self.rank)
        self._patch()
        GlobalRandomApiProcessor._has_saved = True

    def _unpatch_functions(self, library: str, func_names: List[str], module) -> None:
        for name in func_names:
            key = (library, name)
            if key in self._original_funcs:
                setattr(module, name, self._original_funcs[key])
                del self._original_funcs[key]

    def _unpatch(self) -> None:
        if not GlobalRandomApiProcessor._has_patched:
            return
        for library_name, func_names in self.api_dict.items():
            self._unpatch_functions(library_name, func_names, Const.API_MAPPING[library_name])
        GlobalRandomApiProcessor._has_patched = False

    def unfix_random_state(self) -> None:
        self.reset_state = False
        self.state = None
        GlobalRandomApiProcessor._has_fixed = False
        if not GlobalRandomApiProcessor._has_saved:
            self._unpatch()

    def unsave_random_api(self) -> None:
        self.enable_dump = False
        self.rank = None
        self.csv_path = None
        self.api_count = defaultdict(int)
        GlobalRandomApiProcessor._has_saved = False
        if not GlobalRandomApiProcessor._has_fixed:
            self._unpatch()

    def unpatch(self) -> None:
        self.reset_state = False
        self.state = None
        self.enable_dump = False
        self.rank = None
        self.csv_path = None
        self.api_count = defaultdict(int)
        GlobalRandomApiProcessor._has_fixed = False
        GlobalRandomApiProcessor._has_saved = False
        self._unpatch()
