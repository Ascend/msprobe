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

"""Tensor loader for module input override.

Loads previously-dumped module-level input tensors from a source dump directory
and uses them to override a model's actual module inputs during forward, to
isolate upstream accumulated errors during precision debugging.

Users specify exact dump entry names (matching dump .pt filenames) in
config.json load.modules, e.g.:
    "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
"""

import glob
import os
import re

import torch

from msprobe.core.common.const import Const
from msprobe.pytorch.common.log import logger
from msprobe.pytorch.common.utils import load_pt

# Regex for parsing module entry: Module.{dotted_path}.{ClassName}.forward.{N}
_MODULE_ENTRY_RE = re.compile(r"Module\.(.+)\.forward\.(\d+)$")


class TensorLoader:
    """Load previously-dumped module-level input tensors and override actual inputs.

    Created from a LoadConfig instance. The loader is injected into ModuleProcessor
    and invoked from forward_pre_hook before the dump capture, so overrides take
    effect even when dump is disabled (dump_after_load=false).
    """

    def __init__(self, load_config):
        """
        Args:
            load_config: LoadConfig instance with path/modules/step/rank/dump_after_load
        """
        self.path = load_config.path
        # modules: exact dump entry names, e.g.
        #   "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
        self.modules = set(load_config.modules)
        self.src_step = load_config.step  # list, [] = auto-align to current step
        self.src_rank = load_config.rank  # list, [] = auto-align to current rank
        self.dump_after_load = load_config.dump_after_load
        self._current_step = None
        self._current_rank = None
        # cache: (full_forward_name, category, suffix) -> loaded tensor
        self._cache = {}
        # active: whether load override should take effect for current step.
        # When load.step specifies specific steps and current step is not in range,
        # set to False so forward_pre_hook skips load logic entirely (no override,
        # no warning, no performance overhead).
        self.active = True
        # track which modules have been hit by should_override
        self._hit_modules = set()

    def validate_modules(self, model):
        """Validate module paths in load.modules against model's named_modules().

        Called after module hooks are mounted. Checks that the module path part
        (everything before .forward.N) exists in the model. Prints summary of
        valid/invalid modules.

        Args:
            model: the model passed to debugger.start(), used for named_modules()
        """

        # build set of module paths from named_modules()
        # named_modules() returns (name, module) where name is dotted path like "blocks.0.attn"
        # load.modules entries have format "Module.{path}.{ClassName}.forward.{N}"
        model_module_paths = set()
        model_class_names = {}
        for name, module in model.named_modules():
            model_module_paths.add(name)
            model_class_names[name] = module.__class__.__name__

        valid = []
        invalid = []
        for entry in sorted(self.modules):
            module_path, class_name = self._parse_module_entry(entry)
            if module_path is None:
                invalid.append(entry)
                continue

            if module_path in model_module_paths:
                actual_class = model_class_names.get(module_path, "")
                if actual_class != class_name:
                    invalid.append(entry)
                else:
                    valid.append(entry)
            else:
                invalid.append(entry)

        # check for multiple forward.N pointing to same module path
        path_count = {}
        for entry in valid:
            module_path, _ = self._parse_module_entry(entry)
            if module_path is not None:
                path_count[module_path] = path_count.get(module_path, 0) + 1

        logger.info_on_rank_0(f"[load] module validation: {len(valid)}/{len(self.modules)} modules valid in model")
        for entry in invalid:
            logger.warning_on_rank_0(f"[load] invalid module: {entry}")
        for path, count in path_count.items():
            if count > 1:
                logger.warning_on_rank_0(f"[load] module path '{path}' has {count} forward.N entries")
        logger.info_on_rank_0(
            "[load] note: forward.N call_index cannot be verified before runtime, "
            "mismatched call_index will result in source data missing warning during forward"
        )

    @staticmethod
    def _parse_module_entry(entry):
        """Parse a module entry into (module_path, class_name).

        Supports two formats:
        - Single model:  Module.{dotted_path}.{ClassName}.forward.{N}
        - List model:    Module.{index}.{dotted_path}.{ClassName}.forward.{N}

        Returns (module_path, class_name) or (None, None) if format is invalid.

        Example:
            "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
            -> ("blocks.0.attn", "MultiHeadSelfAttention")
            "Module.0.blocks.0.attn.MultiHeadSelfAttention.forward.0"
            -> ("blocks.0.attn", "MultiHeadSelfAttention")  (index stripped)
        """
        m = _MODULE_ENTRY_RE.match(entry)
        if not m:
            return None, None
        path_with_class = m.group(1)  # "blocks.0.attn.MultiHeadSelfAttention" or "0.blocks.0.attn..."
        # strip leading index for list models: "0.blocks.0.attn..." -> "blocks.0.attn..."
        parts = path_with_class.split(".", 1)
        if len(parts) == 2 and parts[0].isdigit():
            path_with_class = parts[1]
        # split into module_path and class_name
        parts = path_with_class.rsplit(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", parts[0]

    def update_step_rank(self, current_step, current_rank):
        """Called by service at each start() to inform loader of current step/rank.

        Clears the tensor cache so each step reloads from source.
        Also sets active flag based on whether current step is in load.step range.
        """
        self._current_step = current_step
        self._current_rank = current_rank
        self._cache.clear()
        # If load.step is specified (non-empty), only activate for steps in range.
        # Empty load.step means auto-align (active for all steps).
        if self.src_step:
            step_active = current_step in self.src_step
        else:
            step_active = True
        # If load.rank is specified (non-empty), only activate for ranks in range.
        if self.src_rank:
            rank_active = current_rank in self.src_rank
        else:
            rank_active = True
        self.active = step_active and rank_active

    def should_override(self, full_forward_name):
        """Check whether this module forward call should be overridden.

        Exact match against user-configured module names.
        Also tracks which modules have been hit for end-of-run reporting.

        Args:
            full_forward_name: the full forward name from hook, e.g.
                "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
        Returns:
            bool
        """
        hit = full_forward_name in self.modules
        if hit:
            self._hit_modules.add(full_forward_name)
        return hit

    def override_args(self, full_forward_name, args, kwargs):
        """Replace tensor-type positional args and kwargs with loaded tensors.

        Non-tensor args/kwargs are left unchanged (no .pt file exists for them in
        source dump). If a source .pt file is missing, warn and keep the original
        value.

        Dump-side naming (base.py analyze_forward_input):
            args:   {full_forward_name}.input.{i}.pt       (api_data_category="input")
            kwargs: {full_forward_name}.kwargs.{key}.pt    (api_data_category="kwargs")

        Args:
            full_forward_name: "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"
            args: tuple of positional args to module forward
            kwargs: dict of keyword args to module forward
        Returns:
            (new_args_tuple, new_kwargs_dict)
        """
        new_args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                loaded = self._load_override_tensor(full_forward_name, Const.INPUT, i, arg)
                if loaded is not None:
                    new_args[i] = loaded

        new_kwargs = dict(kwargs)
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                loaded = self._load_override_tensor(full_forward_name, Const.KWARGS, key, val)
                if loaded is not None:
                    new_kwargs[key] = loaded
        return tuple(new_args), new_kwargs

    def _load_override_tensor(self, full_forward_name, category, suffix, current_tensor):
        """Load a single tensor from source dump, with cache, shape/dtype check, and error handling.

        Returns the loaded tensor, or None if loading failed (warning logged, original kept).
        """
        cache_key = (full_forward_name, category, suffix)
        if cache_key in self._cache:
            return self._cache[cache_key]

        pt_path = self._build_pt_path(full_forward_name, category, suffix)
        if not os.path.exists(pt_path):
            logger.warning(f"[load] source data missing: {pt_path}, keep original value")
            return None

        try:
            loaded = load_pt(pt_path, map_location=current_tensor.device)
            if loaded.shape != current_tensor.shape or loaded.dtype != current_tensor.dtype:
                logger.warning(
                    f"[load] tensor mismatch for {full_forward_name}.{category}.{suffix}: "
                    f"source shape={list(loaded.shape)} dtype={loaded.dtype} "
                    f"vs current shape={list(current_tensor.shape)} dtype={current_tensor.dtype}, "
                    f"override may cause forward error"
                )
            self._cache[cache_key] = loaded
            logger.debug(f"[load] override {full_forward_name}.{category}.{suffix} <- {pt_path}")
            return loaded
        except Exception as e:
            logger.warning(f"[load] failed to load {pt_path}: {e}, keep original value")
            return None

    def _build_pt_path(self, full_forward_name, category, suffix):
        """Build the source tensor file path, matching dump-side naming exactly.

        Dump side (base.py:664 get_save_file_path):
            {current_api_or_module_name}.{api_data_category}{suffix}.pt
        where:
            current_api_or_module_name = full_forward_name
            api_data_category = category ("input" for args, "kwargs" for kwargs)
            suffix = arg_index (int) or kwargs key (str)
        So:
            args:   "{full_forward_name}.input.{i}.pt"
            kwargs: "{full_forward_name}.kwargs.{key}.pt"

        Directory naming matches dump-side _create_default_dirs:
            - rank is not None (distributed): rank{N}
            - rank is None (single-card): proc{pid}, auto-discovered via glob
              (source dump pid differs between runs)
        """
        step = self._current_step
        rank = self._current_rank
        if rank is not None:
            rank_subdir = f"{Const.RANK}{rank}"
        else:
            # single-card: find proc{pid} directory in source dump (pid differs between runs)
            step_dir = os.path.join(self.path, f"step{step}")
            proc_dirs = glob.glob(os.path.join(step_dir, f"{Const.PROC}*"))
            if not proc_dirs:
                logger.error(
                    f"[load] no proc* directory found in {step_dir}. "
                    "Please verify load.path is a valid dump directory with step{N}/proc{pid}/ structure. "
                    "Source dump pid differs from current process, auto-discovery failed."
                )
                rank_subdir = f"{Const.PROC}{os.getpid()}"  # fallback (will likely miss)
            else:
                rank_subdir = os.path.basename(proc_dirs[0])
                if len(proc_dirs) > 1:
                    logger.warning_on_rank_0(
                        f"[load] multiple proc* directories found in {step_dir}, using {rank_subdir}"
                    )
        filename = f"{full_forward_name}{Const.SEP}{category}{Const.SEP}{suffix}{Const.PT_SUFFIX}"
        return os.path.join(self.path, f"step{step}", rank_subdir, "dump_tensor_data", filename)

    def check_unhit(self):
        """Report modules that were configured but never hit during forward.

        Called at debugger.stop() time. Warns about configured modules that
        were never matched by should_override (e.g., wrong call_index or
        module not in model's forward path for current step).
        """
        unhit = self.modules - self._hit_modules
        if unhit:
            logger.warning(
                f"[load] {len(unhit)} module(s) were configured but never matched during this step: {sorted(unhit)}"
            )
