# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025-2026 Huawei Technologies Co.,Ltd.
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

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, overload

import numpy as np

from msprobe.response_anomaly._vocab import extract_tokenizer_target
from msprobe.response_anomaly.detector import (
    DetectionResult,
    ILLDetector,
    _UNSET,
    _is_logprob_seq,
    _parse_batch_args,
)

_PKG_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_PKG_DIR, "configs", "config.yaml")
_DEFAULT_MTYPE_PATH = os.path.join(_PKG_DIR, "configs", "mtype_config.json")
_DEFAULT_TK2CAT_PATH = os.path.join(_PKG_DIR, "token2category")


def _is_single_token_sequence(tokens: Any) -> bool:
    if isinstance(tokens, np.ndarray):
        return tokens.ndim == 1
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        return False
    if len(tokens) == 0:
        return False
    first = tokens[0]
    return isinstance(first, (int, np.integer)) and not isinstance(first, (bool, np.bool_))


def _all_none_or_unset(topk_logprobs: Any) -> bool:
    if topk_logprobs is _UNSET or topk_logprobs is None:
        return True
    if isinstance(topk_logprobs, Sequence) and not isinstance(topk_logprobs, (str, bytes)):
        return len(topk_logprobs) > 0 and all(item is None for item in topk_logprobs)
    return False


@overload
def analyze_output_anomaly(
    tokens: Union[Sequence[int], Sequence[Sequence[int]], np.ndarray],
    model_configs: Any = None,
    *,
    tier: str = "intercept",
    think_end: Any = "auto",
    repetition_config: Any = None,
    rare_chars_model_path: Optional[str] = None,
    return_detail: Literal[False] = False,
) -> List[List[Any]]: ...


@overload
def analyze_output_anomaly(
    tokens: Union[Sequence[int], Sequence[Sequence[int]], np.ndarray],
    model_configs: Any = None,
    *,
    tier: str = "intercept",
    think_end: Any = "auto",
    repetition_config: Any = None,
    rare_chars_model_path: Optional[str] = None,
    return_detail: Literal[True],
) -> List[DetectionResult]: ...


@overload
def analyze_output_anomaly(
    topk_logprobs: Optional[List[Optional[List[Dict[int, float]]]]],
    tokens: Sequence[Sequence[int]],
    model_configs: Any = None,
    *,
    tier: str = "intercept",
    think_end: Any = "auto",
    repetition_config: Any = None,
    rare_chars_model_path: Optional[str] = None,
    return_detail: bool = False,
) -> Union[List[List[Any]], List[DetectionResult]]: ...


def analyze_output_anomaly(*args: Any, **kwargs: Any) -> Union[List[List[Any]], List[DetectionResult]]:
    """
    推理输出异常检测便捷入口。

    默认以纯 Token 序列模式执行（无需传入 topk_logprobs）；
    当传入 topk_logprobs 时，自动走兼容的 logprob 检测算法。
    """
    if "model_or_tokenizer" in kwargs:
        if "model_configs" in kwargs:
            raise TypeError("analyze_output_anomaly() got multiple values for 'model_configs'/'model_or_tokenizer'")
        kwargs["model_configs"] = kwargs.pop("model_or_tokenizer")

    tier = kwargs.pop("tier", "intercept")
    think_end = kwargs.pop("think_end", "auto")
    repetition_config = kwargs.pop("repetition_config", None)
    rare_chars_model_path = kwargs.pop("rare_chars_model_path", None)
    return_detail = kwargs.pop("return_detail", False)

    topk_logprobs, tokens, model_configs = _parse_batch_args("analyze_output_anomaly", *args, **kwargs)

    if isinstance(tokens, (str, bytes)):
        raise ValueError("tokens must be a 1D sequence or 2D batch of integer token IDs")

    # 支持直接传入单条 1D token 序列，自动包装为单元素批次（空序列 [] 在纯 Token 且带单配置时亦按单条空请求处理）
    if isinstance(tokens, np.ndarray):
        if tokens.ndim == 1:
            tokens = [tokens]
            if _is_logprob_seq(topk_logprobs):
                topk_logprobs = [topk_logprobs]
        elif tokens.ndim != 2:
            raise ValueError(f"tokens must be a 1D or 2D array, got shape {tokens.shape}")
    elif _is_single_token_sequence(tokens):
        tokens = [tokens]
        if _is_logprob_seq(topk_logprobs):
            topk_logprobs = [topk_logprobs]
    elif isinstance(tokens, Sequence) and len(tokens) == 0:
        has_single_cfg = model_configs is not None and not (
            isinstance(model_configs, (list, tuple)) and len(model_configs) == 0
        )
        if (topk_logprobs is _UNSET or topk_logprobs is None) and has_single_cfg:
            tokens = [tokens]

    # 纯 Token 批次共享同一个分词器目标时，直接在构造期绑定至 ILLDetector，避免加载默认 logprob 配置及逐条解析
    bound_target = None
    if _all_none_or_unset(topk_logprobs):
        candidate = (
            model_configs[0] if isinstance(model_configs, (list, tuple)) and len(model_configs) == 1 else model_configs
        )
        if not isinstance(candidate, (list, tuple)):
            bound_target, _ = extract_tokenizer_target(candidate)

    if bound_target is not None:
        detector = ILLDetector(
            bound_target,
            think_end=think_end,
            repetition_config=repetition_config,
            rare_chars_model_path=rare_chars_model_path,
        )
    else:
        detector = ILLDetector(
            config_path=_DEFAULT_CONFIG_PATH,
            mtype_path=_DEFAULT_MTYPE_PATH,
            tk2cat_path=_DEFAULT_TK2CAT_PATH,
            think_end=think_end,
            repetition_config=repetition_config,
            rare_chars_model_path=rare_chars_model_path,
        )

    if topk_logprobs is _UNSET:
        return detector.run(tokens, model_configs, tier=tier, return_detail=return_detail)
    return detector.run(
        tokens,
        model_configs,
        topk_logprobs=topk_logprobs,
        tier=tier,
        return_detail=return_detail,
    )


__all__ = ["DetectionResult", "ILLDetector", "analyze_output_anomaly"]
