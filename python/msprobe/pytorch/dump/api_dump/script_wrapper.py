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

import os
import threading
import functools
import importlib
import types
import weakref

import torch

from msprobe.core.common.log import logger
from msprobe.pytorch.common.utils import torch_version_above_or_equal_2
from msprobe.pytorch.dump.api_dump.api_register import get_api_register
from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import ThreadSafe
from msprobe.core.dump.hook_manager import BaseHookManager
from msprobe.core.dump.data_dump.data_processor.base import ModuleForwardInputsOutputs
from msprobe.pytorch.dump.api_dump.hook_module import HOOKModule
from msprobe.core.common.const import Const


if torch_version_above_or_equal_2:
    from torch._dynamo.convert_frame import convert_frame as _orig_convert_frame, Hooks


def wrap_jit_script_func():
    def patched_script(*args, **kwargs):
        all_api_registered = api_register.all_api_registered
        if all_api_registered:
            api_register.restore_all_api()
        result = original_script(*args, **kwargs)
        if all_api_registered:
            api_register.register_all_api()
        return result

    original_script = torch.jit.script
    api_register = get_api_register()
    torch.jit.script = patched_script


def wrap_compile_script_func():
    def _patched_convert_frame(compiler_fn, hooks):
        """
        在调用原 convert_frame 生成的 _convert_frame 之前恢复 API，
        调用完之后再重新注册所有 API。
        """
        # 拿到原来 inner 版的 _convert_frame
        inner_convert = _orig_convert_frame(compiler_fn, hooks)  # pylint: disable=possibly-used-before-assignment

        def _wrapped(frame: types.FrameType, cache_size: int, hooks: Hooks, frame_state):  # pylint: disable=possibly-used-before-assignment
            reg = get_api_register()
            # 进入前 restore
            reg.restore_all_api()
            try:
                result = inner_convert(frame, cache_size, hooks, frame_state)
            except Exception:
                # 异常时也要确保 register
                reg.register_all_api()
                raise
            # 正常结束后 register
            reg.register_all_api()
            return result

        # 保留原属性以兼容
        _wrapped._torchdynamo_orig_callable = compiler_fn  # type: ignore[attr-defined]

        _wrapped._clone_with_backend = lambda backend: _patched_convert_frame(backend, hooks)  # type: ignore[attr-defined]

        return _wrapped

    import torch._dynamo.convert_frame as _cf_mod

    _cf_mod.convert_frame = _patched_convert_frame


def patch_dynamo_compile():
    cf = importlib.import_module("torch._dynamo.convert_frame")
    if not hasattr(cf, "_compile"):
        logger.info("No found torch._dynamo.convert_frame._compile")

    original = cf._compile
    if getattr(original, "__msprobe_patched__", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = None
        try:
            reg = get_api_register()
            reg.restore_all_api()
        except Exception as e:
            logger.info(f"[msprobe] Pre restore_all_api failed: {e}")
            return result

        try:
            result = original(*args, **kwargs)

        except Exception as e:
            logger.warning("[msprobe] _compile execution failed")
            # 当 PyTorch >= 2.7 msprobe 不支持 Dynamo
            if "Dynamo" in str(e) or "Unsupported" in str(e):
                raise RuntimeError(
                    "\n[msprobe ERROR]\n"
                    "Detected TorchDynamo execution.\n\n"
                    "msprobe is NOT compatible with TorchDynamo when PyTorch >= 2.7.\n"
                    "Please disable Dynamo before using msprobe.\n\n"
                    "Suggested solutions:\n"
                    "1. Disable torch.compile\n"
                    "2. Set environment variable:\n"
                    "   export TORCHDYNAMO_DISABLE=1\n"
                    "3. Or run model without compile mode\n"
                ) from e

            result = None
            raise
        finally:
            try:
                reg = get_api_register()
                reg.register_all_api()  # 改成注册hook
            except Exception as e:
                logger.warning(f"[msprobe] Post register_all_api failed: {e}")
        return result

    wrapped.__msprobe_patched__ = True
    wrapped.__msprobe_original__ = original
    cf._compile = wrapped


def unpatch_dynamo_compile() -> bool:
    # 预留取消patch接口
    cf = importlib.import_module("torch._dynamo.convert_frame")
    current = getattr(cf, "_compile", None)
    if current is None:
        return False
    original = getattr(current, "__msprobe_original__", None)
    if original is None:
        return False
    cf._compile = original
    return True


def preprocess_func():
    try:
        from torch.utils._device import _device_constructors

        _device_constructors()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to execute _device_constructors. Error Details: {str(e)}")


_service_ref = None


def set_current_service(service):
    """由 PytorchService 在初始化时注入，避免 script_wrapper 反向 import。"""
    global _service_ref
    _service_ref = weakref.ref(service)


def get_current_service():
    return _service_ref() if _service_ref else None


def patch_triton_jitfunction_run():
    def iter_triton_run_targets():
        try:
            import triton
        except Exception as e:
            logger.info(f"[msprobe] Triton not available, skip patch Triton.run: {e}")
            return
        logger.info(f"[msprobe] Triton detected, version={getattr(triton, '__version__', 'unknown')}")

        candidates = []
        try:
            from triton.runtime import JITFunction as RuntimeJITFunction

            candidates.append(RuntimeJITFunction)
        except Exception:  # nosec
            pass

        try:
            from triton.runtime.jit import JITFunction as JitJITFunction

            candidates.append(JitJITFunction)
        except Exception:  # nosec
            pass

        try:
            from triton.runtime.autotuner import Autotuner

            candidates.append(Autotuner)
        except Exception:  # nosec
            pass

        try:
            from triton.runtime.heuristics import Heuristics

            candidates.append(Heuristics)
        except Exception:  # nosec
            pass

        seen = set()
        for cls in candidates:
            if cls in seen:
                continue
            seen.add(cls)
            yield cls

    def get_triton_api_name(obj):
        fn = getattr(obj, "fn", None)
        fn_name = getattr(fn, "__name__", None)
        if isinstance(fn_name, str) and fn_name:
            return f"Triton.{fn_name}"
        obj_name = getattr(obj, "__name__", None)
        if isinstance(obj_name, str) and obj_name:
            return f"Triton.{obj_name}"
        return f"Triton.{type(obj).__name__}"

    def wrap_run_for_class(cls, suppress_nested):
        original_run = getattr(cls, "run", None)
        if original_run is None:
            return False
        if getattr(original_run, "__msprobe_patched__", False):
            return False

        @functools.wraps(original_run)
        def wrapped_run(self, *args, **kwargs):
            tid = threading.get_ident()
            if not Runtime.is_running:
                return original_run(self, *args, **kwargs)
            if BaseHookManager.inner_switch.get(tid, False):
                return original_run(self, *args, **kwargs)

            service = get_current_service()
            if service is None:
                return original_run(self, *args, **kwargs)
            if getattr(service, "should_stop_service", False):
                return original_run(self, *args, **kwargs)

            data_collector = service.data_collector
            pid = os.getpid()

            api_name = get_triton_api_name(self)
            count = HOOKModule.get_module_count(api_name)
            HOOKModule.add_module_count(api_name)
            full_name = f"{api_name}{Const.SEP}{count}{Const.SEP}{Const.FORWARD}"

            with ThreadSafe():
                BaseHookManager.inner_switch[tid] = True
                try:
                    data_collector.update_api_or_module_name(full_name)
                    module_io = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=None)
                    data_collector.forward_input_data_collect(full_name, self, pid, module_io)
                except Exception as e:
                    logger.info(f"[msprobe] Triton forward_input_data_collect failed: {e}")
                finally:
                    BaseHookManager.inner_switch[tid] = False

            if suppress_nested:
                BaseHookManager.inner_switch[tid] = True
            try:
                out = original_run(self, *args, **kwargs)
            finally:
                if suppress_nested:
                    BaseHookManager.inner_switch[tid] = False

            with ThreadSafe():
                BaseHookManager.inner_switch[tid] = True
                try:
                    data_collector.update_api_or_module_name(full_name)
                    module_io = ModuleForwardInputsOutputs(args=args, kwargs=kwargs, output=args)
                    data_collector.forward_output_data_collect(full_name, self, pid, module_io)
                except Exception as e:
                    logger.info(f"[msprobe] Triton forward_output_data_collect failed: {e}")
                finally:
                    BaseHookManager.inner_switch[tid] = False

            return out

        wrapped_run.__msprobe_patched__ = True
        wrapped_run.__msprobe_original__ = original_run
        setattr(cls, "run", wrapped_run)
        return True

    patched = False
    patched_targets = []
    skipped_targets = []
    for cls in iter_triton_run_targets():
        cls_name = getattr(cls, "__module__", "") + "." + getattr(cls, "__name__", str(cls))
        suppress_nested = cls_name.endswith(".Autotuner") or cls_name.endswith(".Heuristics")
        try:
            if getattr(getattr(cls, "run", None), "__msprobe_patched__", False):
                skipped_targets.append(f"{cls_name}.run(already_patched)")
                continue
            if wrap_run_for_class(cls, suppress_nested=suppress_nested):
                patched = True
                patched_targets.append(f"{cls_name}.run")
                logger.info(f"[msprobe] Patched {cls_name}.run successfully.")
        except Exception as e:
            logger.info(f"[msprobe] Patch {cls_name}.run failed: {e}")

    if not patched:
        logger.info("[msprobe] No Triton run target patched. Triton may be unavailable or API changed.")
    else:
        logger.info(f"[msprobe] Triton patch summary, patched={patched_targets}, skipped={skipped_targets}")


def unpatch_triton_jitfunction_run() -> bool:
    restored = False
    candidates = []
    try:
        from triton.runtime import JITFunction as RuntimeJITFunction

        candidates.append(RuntimeJITFunction)
    except Exception:  # nosec
        pass
    try:
        from triton.runtime.jit import JITFunction as JitJITFunction

        candidates.append(JitJITFunction)
    except Exception:  # nosec
        pass
    try:
        from triton.runtime.autotuner import Autotuner

        candidates.append(Autotuner)
    except Exception:  # nosec
        pass
    try:
        from triton.runtime.heuristics import Heuristics

        candidates.append(Heuristics)
    except Exception:  # nosec
        pass

    seen = set()
    for cls in candidates:
        if cls in seen:
            continue
        seen.add(cls)
        current = getattr(cls, "run", None)
        if current is None:
            continue
        original = getattr(current, "__msprobe_original__", None)
        if original is None:
            continue
        setattr(cls, "run", original)
        restored = True
    return restored


def adapt_megatron_distributed_mappings():
    """
    megatron框架定义的变量dist_all_gather_func和dist_reduce_scatter_func会随着import链路初始化，指向msprobe工具patch前的dist接口

    源码示例:
    -------------------------------------------------------------------------------------------------------------
    import ...
    from ... import ...
    if is_torch_min_version("1.13.0"):
        dist_all_gather_func = torch.distributed.all_gather_into_tensor
        dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
    else:
        dist_all_gather_func = torch.distributed._all_gather_base
        dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

    # 此时dist_all_gather_func和dist_reduce_scatter_func已初始化完成，指向msprobe工具patch前的dist接口
    ... ...

    dist_all_gather_func(...)
    dist_reduce_scatter_func(...) # 实际调用msprobe工具patch前的dist接口，无法dump数据
    -------------------------------------------------------------------------------------------------------------

    本函数将megatron涉及的模块变量dist_all_gather_func和dist_reduce_scatter_func重新指向msprobe工具patch后的dist接口，
    避免dist接口无法dump
    """

    # 到megatron core_v0.12.1版本共有6个模块涉及
    megatron_module_path = [
        "megatron.core.tensor_parallel.mappings",
        "megatron.core.tensor_parallel.utils",
        "megatron.core.tensor_parallel.layers",
        "megatron.core.distributed.param_and_grad_buffer",
        "megatron.core.utils",
        "megatron.core.timers",
    ]

    def adapt_single_module(module):
        """适配单个模块中的分布式函数映射"""
        # 更新all_gather函数映射
        all_gather_func = getattr(module, "dist_all_gather_func", None)
        if all_gather_func:
            if str(all_gather_func).startswith('<function all_gather_into_tensor'):
                module.dist_all_gather_func = torch.distributed.all_gather_into_tensor
            elif str(all_gather_func).startswith('<function _all_gather_base'):
                module.dist_all_gather_func = torch.distributed._all_gather_base

        # 更新reduce_scatter函数映射
        reduce_scatter_func = getattr(module, "dist_reduce_scatter_func", None)
        if reduce_scatter_func:
            if str(reduce_scatter_func).startswith('<function reduce_scatter_tensor'):
                module.dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
            elif str(reduce_scatter_func).startswith('<function _reduce_scatter_base'):
                module.dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

    if importlib.util.find_spec("megatron") is None:
        return

    for module_path in megatron_module_path:
        try:
            megatron_module = importlib.import_module(module_path)
            adapt_single_module(megatron_module)
        except ImportError:
            logger.info(f'Import {module_path} failed, skip mapping.')
            continue
        except Exception as e:
            logger.info(
                f'An unexpected error occurred in the function "adapt_megatron_distributed_mappings", skip mapping: {e}'
            )
            continue


def wrap_script_func():
    wrap_jit_script_func()
    if torch_version_above_or_equal_2:
        patch_dynamo_compile()
    patch_triton_jitfunction_run()
    adapt_megatron_distributed_mappings()
