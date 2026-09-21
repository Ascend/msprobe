# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
# bandit: disable=B110
# bandit: disable=B112

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from msprobe.core.common.const import Const, MonitorConst
from msprobe.core.common.framework_adapter import FmkAdp
from msprobe.core.common.log import logger
from msprobe.core.monitor_v2.base import BaseMonitorV2
from msprobe.core.monitor_v2.utils import build_param2name, get_vpp_stage_from_tag


class ParamMonitorV2(BaseMonitorV2, ABC):
    """
    Parameter distribution monitor that captures params before/after optimizer step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._optimizer: Any = None
        self._model: Any = None
        self._param2name: Dict[Any, str] = {}
        self._eps: float = 1e-8
        self.param_distribution: bool = True
        self._step_hook_handles: List[Any] = []
        self._orig_step = None
        self._step_patched = False

    @abstractmethod
    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]: ...

    def set_config(self, config: Dict[str, Any]) -> None:
        super().set_config(config)
        self.config = config
        self._eps = float(config.get("eps", 1e-8))
        self.param_distribution = bool(config.get("param_distribution", True))

    def start(self, model: Any = None, optimizer: Any = None, **context: Any) -> None:  # pylint: disable=arguments-differ
        self.set_context(**context)
        if self._optimizer is not None:
            return
        optimizer = optimizer or self._context.get("optimizer")
        model = model or self._context.get("model")
        if optimizer is None or model is None:
            raise ValueError("Model and optimizer must be provided to start ParamMonitorV2")
        self._optimizer = optimizer
        self._model = model
        self._param2name = build_param2name(model=model, optimizer=optimizer)
        self._patch_optimizer()

    def stop(self) -> None:
        self._restore_optimizer()
        self._optimizer = None
        self._model = None
        self._param2name.clear()
        self._rows = []

    def _patch_optimizer(self) -> None:
        if not self.param_distribution or self._optimizer is None:
            return
        pre_hook = getattr(self._optimizer, "register_step_pre_hook", None)
        post_hook = getattr(self._optimizer, "register_step_post_hook", None)
        if callable(pre_hook) and callable(post_hook):
            self._step_hook_handles.append(pre_hook(self._on_pre_step))  # pylint: disable=not-callable
            self._step_hook_handles.append(post_hook(self._on_post_step))  # pylint: disable=not-callable
            return
        if self._step_patched or not hasattr(self._optimizer, "step"):
            return
        self._orig_step = self._optimizer.step

        def patched_step(*args: Any, **kwargs: Any):
            self._record_param_distribution(MonitorConst.PRE_PARAM)
            out = self._orig_step(*args, **kwargs)
            self._record_param_distribution(MonitorConst.POST_PARAM)
            return out

        self._optimizer.step = patched_step  # type: ignore[assignment]
        self._step_patched = True

    def _restore_optimizer(self) -> None:
        for handle in self._step_hook_handles:
            try:
                handle.remove()
            except Exception:  # nosec B110
                pass
        self._step_hook_handles.clear()
        if self._step_patched and self._optimizer is not None and self._orig_step is not None:
            self._optimizer.step = self._orig_step  # type: ignore[assignment]
        self._orig_step = None
        self._step_patched = False

    def _on_pre_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        self._record_param_distribution(MonitorConst.PRE_PARAM)

    def _on_post_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        self._record_param_distribution(MonitorConst.POST_PARAM)

    def _record_param_distribution(self, scope: str) -> None:
        if not self.param_distribution or not self._param2name:
            return
        tag2tensor: Dict[str, Any] = {}
        for param, name in self._param2name.items():
            try:
                if param.numel() == 0:
                    continue
            except Exception:  # nosec B112
                continue
            if not FmkAdp.is_tensor(param):
                continue
            tensor = param
            if tensor.__class__.__name__ == "DTensor":
                try:
                    tensor = tensor.to_local()
                except Exception as exc:
                    logger.warning(f"[monitor_v2] Failed to convert DTensor param to local: {exc}")
                    continue
            tag2tensor[name] = tensor
        if not tag2tensor:
            return
        stats = self._compute_metrics(tag2tensor)
        for tag, op_dict in stats.items():
            vpp_stage = get_vpp_stage_from_tag(tag)
            row = {
                "module_name": tag,
                "scope": scope,
                "stats": {op: self._detach_stat(val) for op, val in op_dict.items()},
            }
            if vpp_stage is not None:
                row["vpp_stage"] = vpp_stage
            else:
                row["vpp_stage"] = MonitorConst.DEFAULT_STAGE
            self._rows.append(row)

    def _detach_stat(self, value: Any) -> Any:
        if value is None or not FmkAdp.is_tensor(value):
            return value
        if FmkAdp.fmk == Const.PT_FRAMEWORK:
            return value.detach()
        return value


class PyTorchParamMonitorV2(ParamMonitorV2):
    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.pytorch.monitor.module_metric import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})


class MindSporeParamMonitorV2(ParamMonitorV2):
    def _compute_metrics(self, tag2tensor: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        from msprobe.mindspore.monitor.utils import get_metrics

        return get_metrics(self._ops, tag2tensor, self._eps, {})
