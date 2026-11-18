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

"""推理输出异常检测统一调度器（默认纯 Token 序列模式，兼容 topk_logprobs 滑窗模式）。

统一承载单请求与批请求的双模式参数消歧、分词器目标解析缓存及检测路径路由：
  - 默认纯 Token 模式：无需采集 logprobs，协同调用 RareCharDetector（生僻字/UTF-8 乱码，ill_type = 1）
    与 RepetitionDetector（重复生成/兜圈推理，ill_type = 3），按最早触发前缀长度 fired_at 仲裁异常；
  - 兼容 logprob 模式：传入非空 topk_logprobs 时自动路由至传统滑窗算法，检测 logprobs 数值异常 NaN/Inf（4）、
    生僻字（1）、乱码（2）及 ACF/轨迹重复（3）。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import os
import re
from threading import RLock
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence, Tuple, Union, overload
import weakref

import numpy as np

from msprobe.core.common.file_utils import (
    check_file_exist,
    check_path_exists,
    listdir_path,
    load_json,
    load_yaml,
)
from msprobe.response_anomaly._rare_chars import RareCharDetector
from msprobe.response_anomaly._repetition import RepetitionDetector
import msprobe.response_anomaly._vocab as _vocab_mod
from msprobe.response_anomaly._vocab import (
    extract_tokenizer_target,
    is_yaml_path,
    load_vocab_bundle,
)


# 区分“调用方未传参”与“显式传入 None”的内部哨兵对象
_UNSET = object()
# 包目录与传统 logprob 模式的历史相对默认路径（未显式传参或工作目录不存在同名相对路径时回退至包内置文件）
_PKG_DIR = os.path.dirname(os.path.realpath(__file__))
_DEFAULT_CONFIG_REL = "./configs/config.yaml"
_DEFAULT_MTYPE_REL = "./configs/mtype_config.json"
_DEFAULT_TK2CAT_REL = "./token2category/"
# 单个 ILLDetector 实例按请求动态切换分词器时，缓存 (RareCharDetector, RepetitionDetector) 的 LRU 上限
_DYNAMIC_CACHE_MAX = 16


def _resolve_default_path(given: Any, default_rel: str, pkg_rel: str) -> str:
    """解析兼容模式默认配置路径：未显式传参时始终使用包内置路径，避免误读调用方工作目录同名文件。"""
    if given is _UNSET:
        return os.path.join(_PKG_DIR, pkg_rel)
    # 兼容外部显式传入历史默认相对路径（如 "./configs/config.yaml"）且当前工作目录无该文件的调用方式
    if given == default_rel and not os.path.exists(default_rel):
        return os.path.join(_PKG_DIR, pkg_rel)
    return given


def _default_cfg_paths(
    config_path: Any = _UNSET,
    mtype_path: Any = _UNSET,
    tk2cat_path: Any = _UNSET,
) -> Tuple[str, str, str]:
    """统一解析传统 logprob 模式所需的算法配置、模型映射表与 token2category 目录路径。"""
    return (
        _resolve_default_path(config_path, _DEFAULT_CONFIG_REL, "configs/config.yaml"),
        _resolve_default_path(mtype_path, _DEFAULT_MTYPE_REL, "configs/mtype_config.json"),
        _resolve_default_path(tk2cat_path, _DEFAULT_TK2CAT_REL, "token2category/"),
    )


def _is_empty_sequence(seq: Any) -> bool:
    """判断输入序列是否为空（兼容 None、空容器及长度为 0 的一维 NumPy 数组）。"""
    if seq is None:
        return True
    if isinstance(seq, np.ndarray):
        return seq.ndim == 1 and seq.size == 0
    return len(seq) == 0


def _has_logprobs(topk_logprobs: Any) -> bool:
    """判断 topk_logprobs 是否已传入且非空（非 _UNSET、非 None 且至少包含一个步长元素）。"""
    if topk_logprobs is _UNSET or topk_logprobs is None:
        return False
    if isinstance(topk_logprobs, np.ndarray):
        return topk_logprobs.size > 0
    return len(topk_logprobs) > 0


def _is_logprob_seq(obj: Any) -> bool:
    """判断输入是否为单条请求的 topk_logprobs 序列（首个非 None 元素为 {token_id: logprob} 字典）。"""
    if obj is None or isinstance(obj, (str, bytes, os.PathLike, dict)):
        return False
    if isinstance(obj, np.ndarray):
        return obj.dtype == object and obj.size > 0 and isinstance(obj.flat[0], dict)
    if isinstance(obj, Sequence):
        for item in obj:
            if item is not None:
                return isinstance(item, dict)
        return False
    return False


def _is_token_seq(obj: Any) -> bool:
    """判断输入是否为单条一维整数 token ID 序列（兼容空序列与一维 ndarray，排除布尔值、字符串与字典）。"""
    if obj is None or isinstance(obj, (str, bytes, os.PathLike, dict)):
        return False
    if isinstance(obj, np.ndarray):
        return obj.ndim == 1
    if isinstance(obj, Sequence):
        if len(obj) == 0:
            return True
        first = obj[0]
        return isinstance(first, (int, np.integer)) and not isinstance(first, (bool, np.bool_))
    return False


def _is_batch_logprobs(obj: Any) -> bool:
    """判断输入是否为批次级 topk_logprobs（元素为 None 或单条 logprob 序列，支持混合批次）。"""
    if obj is None or isinstance(obj, (str, bytes, os.PathLike, dict)):
        return False
    if isinstance(obj, np.ndarray):
        return obj.dtype == object and any(isinstance(x, dict) for x in obj.flat)
    if not isinstance(obj, Sequence):
        return False
    for req in obj:
        if req is None or _is_logprob_seq(req):
            return True
        if isinstance(req, (str, bytes)) or not isinstance(req, (Sequence, np.ndarray)):
            return False
        # 空子序列（如 []）无法单独区分空 logprobs 与空 tokens，继续向后扫描；非空且非字典序列则必为二维 tokens
        if len(req) > 0:
            return False
    return False


def _is_batch_tokens(obj: Any) -> bool:
    """判断输入是否为二维批次整数 token ID 序列（二维 ndarray 或由一维整数序列组成的列表/元组）。"""
    if obj is None or isinstance(obj, (str, bytes, os.PathLike, dict)):
        return False
    if isinstance(obj, np.ndarray):
        return obj.ndim == 2
    if not isinstance(obj, Sequence):
        return False
    if len(obj) == 0:
        return True
    first = obj[0]
    if isinstance(first, np.ndarray):
        return first.ndim == 1
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        if len(first) == 0:
            return True
        item = first[0]
        return isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_))
    return False


def _same_target(a: Any, b: Any) -> bool:
    """判断请求级分词器目标 a 是否与实例构造期绑定的目标 b 相同（同一对象引用或相同文件系统路径字面值）。"""
    if b is _UNSET:
        return False
    if a is b:
        return True
    if isinstance(a, (str, os.PathLike)) and isinstance(b, (str, os.PathLike)):
        return os.fspath(a) == os.fspath(b)
    return False


def _parse_dual_mode_args(
    func_name: str,
    mc_kw: str,
    two_arg_logprob_check: Any,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, Any, Any]:
    """统一解析纯 Token 默认签名 (tokens[, model_cfg]) 与传统签名 (topk_logprobs, tokens[, model_cfg])。

    位置参数消歧规则：
      - 1 个位置参数：视为 (tokens)（若同时以关键字传入 tokens=...，则该位置参数视为 topk_logprobs）；
      - 2 个位置参数：若 two_arg_logprob_check(a0, a1) 为真则视为 (topk_logprobs, tokens)，否则视为 (tokens, model_cfg)；
      - 3 个位置参数：视为 (topk_logprobs, tokens, model_cfg)。
    """
    kw_lp = kwargs.pop("topk_logprobs", _UNSET)
    kw_tok = kwargs.pop("tokens", _UNSET)
    kw_mc = kwargs.pop(mc_kw, _UNSET)
    if kwargs:
        raise TypeError(f"{func_name}() got unexpected keyword argument(s): {', '.join(sorted(kwargs))}")
    if len(args) > 3:
        raise TypeError(f"{func_name}() takes at most 3 positional arguments but {len(args)} were given")

    # 1) 显式传入关键字 topk_logprobs=... 时，位置参数依次对应 (tokens[, model_cfg])
    if kw_lp is not _UNSET:
        if len(args) == 3:
            raise TypeError(f"{func_name}() got multiple values for argument 'topk_logprobs'")
        if len(args) >= 1 and kw_tok is not _UNSET:
            raise TypeError(f"{func_name}() got multiple values for argument 'tokens'")
        if len(args) >= 2 and kw_mc is not _UNSET:
            raise TypeError(f"{func_name}() got multiple values for argument '{mc_kw}'")
        tokens = args[0] if len(args) >= 1 else kw_tok
        model_cfg = args[1] if len(args) >= 2 else (None if kw_mc is _UNSET else kw_mc)
        if tokens is _UNSET:
            raise TypeError(f"{func_name}() missing required argument: 'tokens'")
        return kw_lp, tokens, model_cfg

    # 2) 未传关键字 topk_logprobs 时，按位置参数数量结合形态探测消歧纯 Token 与传统签名
    if len(args) >= 2 and kw_tok is not _UNSET:
        raise TypeError(f"{func_name}() got multiple values for argument 'tokens'")
    if len(args) == 0:
        if kw_tok is _UNSET:
            raise TypeError(f"{func_name}() missing required argument: 'tokens'")
        return _UNSET, kw_tok, None if kw_mc is _UNSET else kw_mc
    if len(args) == 1:
        # 兼容传统混合调用 fn(topk_logprobs, tokens=...)；否则单位置参数默认为纯 Token 模式的 tokens
        if kw_tok is not _UNSET:
            return args[0], kw_tok, None if kw_mc is _UNSET else kw_mc
        return _UNSET, args[0], None if kw_mc is _UNSET else kw_mc
    if len(args) == 2:
        a0, a1 = args
        if two_arg_logprob_check(a0, a1):
            return a0, a1, None if kw_mc is _UNSET else kw_mc
        if kw_mc is not _UNSET:
            raise TypeError(f"{func_name}() got multiple values for argument '{mc_kw}'")
        return _UNSET, a0, a1

    if kw_mc is not _UNSET:
        raise TypeError(f"{func_name}() got multiple values for argument '{mc_kw}'")
    return args[0], args[1], args[2]


def _is_two_arg_single_logprob(a0: Any, a1: Any) -> bool:
    """单请求双参数场景下判断首参 a0 是否为 topk_logprobs（含 None 或空列表配合合法 token 序列的边界情况）。"""
    return a0 is None or _is_logprob_seq(a0) or (isinstance(a0, (list, tuple)) and len(a0) == 0 and _is_token_seq(a1))


def _is_two_arg_batch_logprob(a0: Any, a1: Any) -> bool:
    """批请求双参数场景下判断首参 a0 是否为批次 topk_logprobs（含 None 或全空子列表 [[]] 的兼容情况）。"""
    all_empty_reqs = (
        isinstance(a0, Sequence)
        and not isinstance(a0, (str, bytes))
        and all(isinstance(x, Sequence) and not isinstance(x, (str, bytes)) and len(x) == 0 for x in a0)
    )
    return a0 is None or _is_batch_logprobs(a0) or (all_empty_reqs and _is_batch_tokens(a1))


def _parse_detect_args(*args: Any, **kwargs: Any) -> Tuple[Any, Any, Any]:
    """解析单请求入口 ILLDetector.detector() 的位置与关键字参数，返回 (topk_logprobs, tokens, model_config)。"""
    return _parse_dual_mode_args("detector", "model_config", _is_two_arg_single_logprob, *args, **kwargs)


def _parse_batch_args(func_name: str, *args: Any, **kwargs: Any) -> Tuple[Any, Any, Any]:
    """解析批处理入口 ILLDetector.run() 与 analyze_output_anomaly() 的参数，返回 (topk_logprobs, tokens, model_configs)。"""
    return _parse_dual_mode_args(func_name, "model_configs", _is_two_arg_batch_logprob, *args, **kwargs)


def _normalize_model_configs(model_configs: Any, nums: int) -> List[Any]:
    """将批次 model_configs 广播或校验对齐为长度为 nums 的逐请求配置列表（支持 None、单对象及单元素列表广播）。"""
    if model_configs is None:
        return [None] * nums
    if not isinstance(model_configs, (list, tuple)):
        return [model_configs] * nums
    if len(model_configs) == nums:
        return list(model_configs)
    if len(model_configs) == 1 and nums > 1:
        return [model_configs[0]] * nums
    raise ValueError(f"model_configs length ({len(model_configs)}) does not match tokens batch size ({nums})")


# 单个请求的检测结果
@dataclass
class DetectionResult:
    """单条推理响应的异常检测结果。

    字段说明：
      - is_ill   : 是否检出异常；
      - ill_type : 异常编码（0 正常；1 生僻字或纯 Token 下的 UTF-8 乱码；2 logprob 乱码；3 重复/兜圈；4 logprob NaN/Inf）；
      - fired_at : 纯 Token 模式下首次触发告警的前缀 token 长度（1-based），未触发或 logprob 模式为 -1；
      - reason   : 纯 Token 模式下命中的检测规则描述，未触发或 logprob 模式为空字符串。
    """

    is_ill: bool = False
    ill_type: int = 0
    fired_at: int = -1
    reason: str = ""


# 重复检测的内部计数器
@dataclass
class RepetitionCounters:
    both: int = 0  # acf + trajectory 同时检出
    acf_only: int = 0  # 仅acf检出
    traj_only: int = 0  # 仅trajectory检出


# 生僻字/乱码检测的窗口结果
@dataclass
class RareGarbledResult:
    rare_flag: bool = False
    garbled_flag: bool = False


def _resolve_model_name(model_config: Any, know_models: set[str]) -> Optional[str]:
    """从 model_config 中解析出模型名, 与 know_models 做模糊匹配

    know_models 来自 mtype_config.json 的所有 key
    匹配规则: 忽略大小写、去掉所有分隔符后比较是否包含
    """
    if model_config is None:
        return None
    if isinstance(model_config, str):
        raw_name = model_config.lower()
    elif isinstance(model_config, dict):
        raw_name = model_config.get("model_name", "").lower()
    else:
        return None

    clean_raw = re.sub(r"[-_.\s]", "", raw_name)

    for known in know_models:
        clean_known = re.sub(r"[-_.\s]", "", known.lower())
        if clean_known in clean_raw or clean_raw in clean_known:
            return known  # 返回 mtype_config.json 里的原始 key 名
    return None


class ILLDetector:
    """推理输出异常检测器，支持纯 Token 序列实时/离线检测（默认）与 topk_logprobs 滑窗检测（兼容）。"""

    @overload
    def __init__(
        self,
        model_or_tokenizer: Any = ...,
        *,
        think_end: Any = "auto",
        repetition_config: Any = None,
        rare_chars_model_path: Optional[str] = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        config_path: str = "./configs/config.yaml",
        mtype_path: str = "./configs/mtype_config.json",
        tk2cat_path: str = "./token2category/",
    ) -> None: ...

    def __init__(
        self,
        model_or_tokenizer: Any = _UNSET,
        mtype_path: Any = _UNSET,
        tk2cat_path: Any = _UNSET,
        *,
        config_path: Any = _UNSET,
        think_end: Any = "auto",
        repetition_config: Any = None,
        rare_chars_model_path: Optional[str] = None,
    ) -> None:
        """初始化双模式检测器：可绑定分词器目标用于高频纯 Token 检测，或按传统三路径加载 logprob 滑窗配置。

        入参分派规则：
          - 传入模型目录、tokenizer.json 路径或内存分词器对象时，预构建并绑定生僻字与重复检测器（logprob 配置按需延迟加载）；
          - 无参调用、首参为 YAML 路径，或显式传入 config_path / mtype_path / tk2cat_path 时，立即初始化传统 logprob 配置。
        """
        # 1) 根据入参形态自动识别初始化模式，支持同时传入 config_path=... 与 model_or_tokenizer 的混合构造
        if config_path is not _UNSET:
            resolved_config = config_path
            resolved_model = model_or_tokenizer
            explicit_logprob = True
        elif mtype_path is not _UNSET or tk2cat_path is not _UNSET:
            resolved_config = model_or_tokenizer
            resolved_model = _UNSET
            explicit_logprob = True
        elif model_or_tokenizer is _UNSET:
            resolved_config = _UNSET
            resolved_model = _UNSET
            explicit_logprob = True
        elif is_yaml_path(model_or_tokenizer):
            resolved_config = model_or_tokenizer
            resolved_model = _UNSET
            explicit_logprob = True
        else:
            resolved_config = _UNSET
            resolved_model = model_or_tokenizer
            explicit_logprob = False

        # 2) 预校验 think_end 参数（仅接受 "auto"、None 或非负整数 token ID，排除布尔值）
        if not (think_end is None or (isinstance(think_end, str) and think_end == "auto")):
            bad_think = isinstance(think_end, (bool, np.bool_)) or not isinstance(think_end, (int, np.integer))
            if bad_think or int(think_end) < 0:
                raise ValueError(f"invalid think_end={think_end!r}")

        self._think_end_cfg = think_end
        self._repetition_config = repetition_config
        self._rare_chars_model_path = rare_chars_model_path
        self._bound_target = _UNSET
        self._dynamic_lock = RLock()
        self._dynamic_detectors: OrderedDict[Any, Any] = OrderedDict()
        self.think_end: Optional[int] = None
        self.vocab_size: int = 0
        self.rare_detector: Optional[RareCharDetector] = None
        self.repetition_detector: Optional[RepetitionDetector] = None
        self.topk: Optional[int] = None
        self._garbled_count: int = 0
        self.mtype2vocab: Dict[str, int] = {}
        self.tk2cat_path: Optional[str] = None

        # 3) 记录传统配置路径；纯 Token 构造时不主动读取 YAML/JSON 配置，直到实际收到非空 topk_logprobs 再惰性加载
        self._cfg_paths = _default_cfg_paths(resolved_config, mtype_path, tk2cat_path)
        if explicit_logprob:
            self._init_logprob_config(*self._cfg_paths)

        # 4) 若提供了分词器目标，预构建词表查表数组并绑定 RareCharDetector 与 RepetitionDetector 实例
        if resolved_model is not _UNSET:
            rare_tables, rep_tables, rare_model, resolved_think = load_vocab_bundle(
                resolved_model,
                think_end=think_end,
                model_path=rare_chars_model_path,
            )
            self._bound_target = resolved_model
            self.think_end = resolved_think
            self.vocab_size = len(rare_tables["rcls"])
            self.rare_detector = RareCharDetector(rare_tables, rare_model, think_end=resolved_think)
            self.repetition_detector = RepetitionDetector(
                rep_tables,
                think_end=resolved_think,
                config=repetition_config,
            )

    def _init_logprob_config(self, config_path: str, mtype_path: str, tk2cat_path: str) -> None:
        """加载并校验传统 logprob 模式所需的 YAML 阈值配置、模型 EOS 映射及 token2category 词表索引。"""
        # 加载配置文件
        config_data = load_yaml(config_path)
        # 验证配置参数的合法性
        self._validate_config(config_data)

        self.mtype2token = load_json(mtype_path)

        # token2category
        check_path_exists(tk2cat_path)  # 检查路径是否存在, 如果不存在, 报错!
        self._init_tk2cat(tk2cat_path)

        # 检测算法配置参数
        self.window_size: int = config_data["window_size"]
        self.stride: int = config_data["stride"]
        self.topk: Optional[int] = None

        _rare = config_data["rare_character"]
        self.rare_explogp_sum_thresh: float = _rare["explogp_sum_thresh"]
        self.rare_cat_thresh: int = _rare["category_thresh"]
        self.rare_top1_logp_thresh: float = _rare["top1_logp_thresh"]

        _garbled = config_data["garbled"]
        self.garbled_top1_logp_thresh: float = _garbled["top1_logp_thresh"]
        self.garbled_window_ratio: float = _garbled["window_ratio"]
        self.garbled_window_thresh: int = _garbled["window_thresh"]

        _repet = config_data["repetition"]
        _traj = _repet["trajectory"]
        self.repet_n: int = _traj["n"]
        self.repet_distinct_n_thresh: float = _traj["distinct_n_thresh"]
        self.repet_logp_thresh: float = _traj["logp_thresh"]

        _acf = _repet["acf"]
        self.w_std_threshold: float = 1e-12  # w_std_threshold: 1e-12
        self.acf_threshold: float = _acf["acf_threshold"]
        self.acf_harmonic_threshold: float = self.acf_threshold // 2  # 2倍/3倍处的谐波阈值，acf_threshold//2
        self.acf_logp_thresh: float = _acf["logp_thresh"]
        self.acf_min_period: int = 3
        self.acf_max_period: int = self.window_size // 3
        self.linalg_logp_thresh: float = 0.9

        self.single_window_thresh: int = _repet["single_window_thresh"]
        self.multi_window_thresh: int = _repet["multi_window_thresh"]

        # 符合乱码条件的窗口计数
        self._garbled_count: int = 0

        # 生僻字检测时需要过滤和删掉的类别
        self._FILTER_CATEGORIES = {"english_latin_space", "english_latin"}
        self._REMOVE_CATEGORIES = {"punctuation", "whitespace"}

    def _ensure_logprob_ready(self) -> None:
        """确保传统 logprob 配置就绪（纯 Token 方式构造的实例在首次处理非空 topk_logprobs 时按需惰性初始化）。"""
        if hasattr(self, "window_size"):
            return
        cfg_paths = getattr(self, "_cfg_paths", None) or _default_cfg_paths()
        self._init_logprob_config(*cfg_paths)

    def _validate_config(self, config_data: Dict):
        """
        验证配置参数的合法性
        """
        # ========== 基础参数校验 ==========
        window_size = config_data.get("window_size")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size 必须为正整数, 当前值: {window_size}")

        stride = config_data.get("stride")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError(f"stride 必须为正整数, 当前值: {stride}")

        if window_size < stride:
            raise ValueError(f"window_size ({window_size}) 必须大于或等于 stride ({stride})")

        # ========== rare_character 校验 ==========
        rare = config_data.get("rare_character", {})
        rare_explogp = rare.get("explogp_sum_thresh")
        if not isinstance(rare_explogp, (int, float)) or not (0 <= rare_explogp <= 1):
            raise ValueError(f"rare_character.explogp_sum_thresh 必须在 0-1 之间, 当前值为: {rare_explogp}")

        category_thresh = rare.get("category_thresh")
        if not isinstance(category_thresh, int) or category_thresh < 0:
            raise ValueError(f"rare_character.category_thresh 必须为非负整数, 当前值为: {category_thresh}")

        top1_logp_thresh = rare.get("top1_logp_thresh")
        if not isinstance(top1_logp_thresh, (int, float)) or top1_logp_thresh > 0:
            raise ValueError(f"rare_character.top1_logp_thresh 必须是负数或 0, 当前值为: {top1_logp_thresh}")

        # ========== garbled 校验 ==========
        garbled = config_data.get("garbled", {})

        garbled_top1 = garbled.get("top1_logp_thresh")
        if not isinstance(garbled_top1, (int, float)) or garbled_top1 > 0:
            raise ValueError(f"garbled.top1_logp_thresh 必须是负数或 0, 当前值为: {garbled_top1}")

        garbled_ratio = garbled.get("window_ratio")
        if not isinstance(garbled_ratio, (int, float)) or not (0 <= garbled_ratio <= 1):
            raise ValueError(f"garbled.window_ratio 必须在 0-1 之间, 当前值为: {garbled_ratio}")

        garbled_window_thresh = garbled.get("window_thresh")
        if not isinstance(garbled_window_thresh, int) or garbled_window_thresh < 0:
            raise ValueError(f"garbled.window_thresh 必须是非负整数, 当前值为: {garbled_window_thresh}")

        # ========== repetition 校验 ==========
        repet = config_data.get("repetition", {})

        traj = repet.get("trajectory", {})
        traj_n = traj.get("n")
        if not isinstance(traj_n, int) or traj_n <= 0:
            raise ValueError(f"repetition.trajectory.n 必须是正整数, 当前值为: {traj_n}")

        traj_distinct = traj.get("distinct_n_thresh")
        if not isinstance(traj_distinct, (int, float)) or not (0 <= traj_distinct <= 1):
            raise ValueError(f"repetition.trajectory.distinct_n_thresh 必须在 0-1 之间, 当前值为: {traj_distinct}")

        traj_logp_thresh = traj.get("logp_thresh")
        if not isinstance(traj_logp_thresh, (int, float)) or traj_logp_thresh > 0:
            raise ValueError(f"repetition.trajectory.logp_thresh 必须是负数, 当前值为: {traj_logp_thresh}")

        acf = repet.get("acf", {})
        acf_threshold = acf.get("acf_threshold")
        if not isinstance(acf_threshold, (int, float)) or not (0 <= acf_threshold <= 1):
            raise ValueError(f"repetition.acf.acf_threshold 应该在 0-1 之间, 当前值为: {acf_threshold}")

        acf_logp_thresh = acf.get("logp_thresh")
        if not isinstance(acf_logp_thresh, (int, float)) or acf_logp_thresh > 0:
            raise ValueError(f"repetition.acf.logp_thresh 必须是负数, 当前值为: {acf_logp_thresh}")

        single_window_thresh = repet.get("single_window_thresh")
        if not isinstance(single_window_thresh, int) or single_window_thresh < 0:
            raise ValueError(f"repetition.single_window_thresh 必须是非负整数, 当前值为: {single_window_thresh}")

        multi_window_thresh = repet.get("multi_window_thresh")
        if not isinstance(multi_window_thresh, int) or multi_window_thresh < 0:
            raise ValueError(f"repetition.multi_window_thresh 必须是非负整数, 当前值为: {multi_window_thresh}")

    def _init_tk2cat(self, tk2cat_path: str) -> None:
        """扫描 token2category 目录下的 <model_type>_<vocab_size>.json 静态词表文件并建立词表大小索引。"""
        # 过滤隐藏文件与非标准命名的无关文件，避免文件名拆分解析失败
        file_names = [
            val for val in listdir_path(tk2cat_path) if not val.startswith(".") and val.endswith(".json") and "_" in val
        ]

        # 模型类型与vocab_size的对应
        self.mtype2vocab: Dict[str, int] = {
            val.split("_")[0]: int(val.split("_")[1].split(".")[0]) for val in file_names
        }
        self.tk2cat_path: str = tk2cat_path

    # 滑窗
    def sliding_window(self, seq: List) -> Generator[Tuple[int, List], None, None]:
        """
        生成滑窗
        return:
            i: 窗口序列的起始索引
            window_seq: 窗口序列
        """
        for i in range(0, len(seq), self.stride):
            yield i, seq[i : min(i + self.window_size, len(seq))]

    # 根据model_name 和 token交叉验证model_type
    def get_tk2cat(self, eos_token: int, model_config: Any = None) -> Tuple[Optional[Dict[str, int]], Optional[int]]:
        # 判断从model_config取到的name是否在本地有预设的tokenizer信息，如果有，则对self.tk2cat和self.vocab_size赋值
        if model_config is None or self.tk2cat_path is None:
            return None, None

        name = _resolve_model_name(model_config, self.mtype2token.keys())
        if name is None:
            return None, None

        token_map = self.mtype2token.get(name)
        if token_map is None:
            return None, None

        eos_candidates = token_map.get("eos", [])
        if isinstance(eos_candidates, int):
            eos_candidates = [eos_candidates]
        if eos_token not in eos_candidates:
            return None, None

        vocab_size = self.mtype2vocab.get(name)
        if vocab_size is None:
            return None, None

        path = os.path.join(self.tk2cat_path, f"{name}_{vocab_size}.json")
        if not check_file_exist(path):
            return None, None

        tk2cat = load_json(path)
        return tk2cat, vocab_size

    # N-gram
    def get_ngrams(self, tokens: List[int]) -> List[Tuple[int, ...]]:
        n = self.repet_n
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []

    def get_distinct_n(self, tokens: List[int]) -> float:
        all_grams = self.get_ngrams(tokens)
        if not all_grams:
            return 1.0
        return len(set(all_grams)) / len(all_grams)

    # 乱码检测
    def _detect_garbled(
        self,
        window_topk_logprobs: List[Dict[int, float]],
        window_logprobs: np.ndarray,
        tk2cat: Dict[str, int],
        vocab_size: int,
    ) -> bool:
        seq_len = len(window_topk_logprobs)  # 当前序列长度

        # 1) 如果有词表信息,使用生僻字的检测方法
        if tk2cat is not None:
            flag, rare_character_count = self._detect_rare_character(
                window_topk_logprobs, window_logprobs, tk2cat, vocab_size
            )
            if flag and rare_character_count / seq_len > self.garbled_window_ratio:
                return True
            return False

        # 2) 如果没有词表信息
        dims = np.where(np.exp(window_logprobs).sum(-1) < self.rare_explogp_sum_thresh)[
            0
        ]  # 找出当前window里topk概率总和小于阈值的dims

        if dims.size == 0:
            return False

        if len(dims) / seq_len <= self.garbled_window_ratio:
            return False
        # window_logprobs以已降序排列, window_logprobs[:, 0]表示取序列中 top1 的 logprob
        log_dims = np.where(window_logprobs[:, 0] < self.garbled_top1_logp_thresh)[0]
        return len(log_dims) / seq_len > self.garbled_window_ratio

    # 维护_garbled_count 状态, 返回是否触发乱码警告
    def _update_garbled_state(self, garbled_flag: bool) -> bool:
        if garbled_flag:
            self._garbled_count += 1
        return self._garbled_count > self.garbled_window_thresh

    # 生僻字检测
    def _detect_rare_character(
        self,
        window_topk_logprobs: List[Dict[int, float]],
        window_logprobs: np.ndarray,
        tk2cat: Dict[str, int],
        vocab_size: int,
    ) -> Tuple[bool, int]:
        # 当tk2cat为None时，使用top1的logp < 阈值 来判定
        if tk2cat is None:
            # window_logprobs以已降序排列, window_logprobs[:, 0]表示取序列中 top1 的 logprob
            log_dims = np.where(window_logprobs[:, 0] < self.rare_top1_logp_thresh)[0]
            return len(log_dims) != 0, len(log_dims)

        dims = np.where(np.exp(window_logprobs).sum(-1) < self.rare_explogp_sum_thresh)[
            0
        ]  # 找出当前window里topk概率总和小于阈值的dims

        if dims.size == 0:
            return False, 0

        cat_hit_count = 0
        for dim in dims:
            categories = set(
                tk2cat[str(item)] for item in list(window_topk_logprobs[dim].keys()) if item <= vocab_size
            )  # 统计topk的token-id对应的类别
            # 剔除一些类别
            if len(categories.intersection(self._FILTER_CATEGORIES)) == 2:
                categories.discard("english_latin")
            categories -= self._REMOVE_CATEGORIES
            if len(categories) > self.rare_cat_thresh:
                cat_hit_count += 1
        return cat_hit_count > 0, cat_hit_count

    # 【单窗口】 轨迹检测 n-grams
    def _trajectory_detector(self, window_logprobs: List[float], window_tokens: List[int]) -> bool:
        distinct_n = self.get_distinct_n(window_tokens)
        if distinct_n >= self.repet_distinct_n_thresh:
            return False
        return np.min(window_logprobs) > self.repet_logp_thresh

    # 【单窗口】ACF检测
    def _acf_detector(self, window_logprobs: List[float]) -> bool:
        w_std = window_logprobs.std()
        if w_std < self.w_std_threshold:
            return False

        w_norm = (window_logprobs - window_logprobs.mean()) / w_std
        f = np.fft.rfft(w_norm, n=2 * self.window_size)[: self.window_size]
        acf = np.fft.irfft(f * np.conj(f)[: self.window_size])
        acf /= acf[0]

        acf_segment = acf[self.acf_min_period : self.acf_max_period + 1]
        if len(acf_segment) == 0:
            return False

        peak_lag = np.argmax(acf_segment) + self.acf_min_period
        peak_val = acf_segment[peak_lag - self.acf_min_period]
        if peak_val <= self.acf_threshold:
            return False

        harmonic_confirmed = any(
            h * peak_lag < self.window_size and acf[h * peak_lag] > self.acf_harmonic_threshold for h in [2, 3]
        )
        if not harmonic_confirmed or np.linalg.norm(acf[1:] - acf[:-1]) <= self.linalg_logp_thresh:
            return False

        return np.min(window_logprobs) > self.acf_logp_thresh

    # 【单窗口-总体判定】acf+轨迹检测
    def _detect_repetitions(
        self,
        window_logprobs: np.ndarray,
        window_tokens: List[int],
        count_res: RepetitionCounters,
    ) -> Tuple[bool, RepetitionCounters]:
        # acf
        res_acf = self._acf_detector(window_logprobs)
        # 轨迹检测
        res_traj = self._trajectory_detector(window_logprobs, window_tokens)

        # 总体判定
        if res_acf and res_traj:
            count_res.both += 1
        elif res_acf:
            count_res.acf_only += 1
        elif res_traj:
            count_res.traj_only += 1

        if (
            count_res.both > self.multi_window_thresh
            or count_res.acf_only > self.single_window_thresh
            or count_res.traj_only > self.single_window_thresh
        ):
            return True, count_res
        return False, count_res

    def _sort_and_truncate_topk(self, topk_logprobs):
        return [
            dict(sorted(topk_logprob.items(), key=lambda x: x[1], reverse=True)[: self.topk])
            for topk_logprob in topk_logprobs
        ]

    def _can_resolve_token_only(self, model_config: Any = None) -> bool:
        """判断当前请求是否具备纯 Token 检测条件（请求级 model_config 含有效分词器目标，或实例已绑定默认检测器）。"""
        target, _ = extract_tokenizer_target(model_config)
        if target is not None:
            return True
        return (
            getattr(self, "rare_detector", None) is not None and getattr(self, "repetition_detector", None) is not None
        )

    def _get_token_detectors(self, model_config: Any = None) -> Tuple[RareCharDetector, RepetitionDetector]:
        """按请求解析并获取 (RareCharDetector, RepetitionDetector) 实例对，优先复用构造期绑定实例与有界 LRU 缓存。"""
        target, _ = extract_tokenizer_target(model_config)
        bound_rare = getattr(self, "rare_detector", None)
        bound_rep = getattr(self, "repetition_detector", None)
        bound_target = getattr(self, "_bound_target", _UNSET)

        # 1) 请求未显式指定分词器目标（如为 None 或仅含裸模型名）：回退使用实例构造期预绑定的检测器
        if target is None:
            if bound_rare is not None and bound_rep is not None:
                return bound_rare, bound_rep
            raise ValueError(
                "Token-only detection (without topk_logprobs) requires a model directory, "
                "tokenizer.json path, or tokenizer instance via ILLDetector(...) or model_configs; "
                f"got model_config={model_config!r}."
            )

        # 2) 请求目标与实例预绑定目标相同时直接短路返回，省去词表缓存查找开销
        if bound_rare is not None and bound_rep is not None and _same_target(target, bound_target):
            return bound_rare, bound_rep

        # 3) 动态加载或命中模块级词表缓存；以查表数组对象 id 作为检测器 LRU 键，当内存分词器扩容重构查表时自动失效旧条目
        think_end = getattr(self, "_think_end_cfg", "auto")
        rare_path = getattr(self, "_rare_chars_model_path", None)
        rep_cfg = getattr(self, "_repetition_config", None)
        rare_tables, rep_tables, rare_model, resolved_think = load_vocab_bundle(
            target,
            think_end=think_end,
            model_path=rare_path,
        )
        cache = getattr(self, "_dynamic_detectors", None)
        lock = getattr(self, "_dynamic_lock", None)
        ckey = (id(rare_tables), id(rep_tables), id(rare_model), resolved_think)
        if cache is not None and lock is not None:
            with lock:
                entry = cache.get(ckey)
                if entry is not None and entry[0]() is rare_tables and entry[1]() is rep_tables:
                    if isinstance(cache, OrderedDict):
                        cache.move_to_end(ckey)
                    return (
                        RareCharDetector(rare_tables, rare_model, think_end=resolved_think),
                        RepetitionDetector(rep_tables, think_end=resolved_think, config=entry[2]),
                    )
                cache.pop(ckey, None)

        rare_det = RareCharDetector(rare_tables, rare_model, think_end=resolved_think)
        rep_det = RepetitionDetector(rep_tables, think_end=resolved_think, config=rep_cfg)
        bsize = _vocab_mod._bundle_size_bytes((rare_tables, rep_tables, resolved_think))
        if cache is not None and lock is not None and bsize <= _vocab_mod._BUNDLE_CACHE_MAX_BYTES:
            self_ref = weakref.ref(self)

            def _evict_cb(_ref: Any, _sr: Any = self_ref, _k: Any = ckey) -> None:
                inst = _sr()
                if inst is not None:
                    with inst._dynamic_lock:
                        if _k in inst._dynamic_detectors and _ref in inst._dynamic_detectors[_k][:2]:
                            inst._dynamic_detectors.pop(_k, None)

            try:
                refs = (weakref.ref(rare_tables, _evict_cb), weakref.ref(rep_tables, _evict_cb), rep_det.cfg)
            except TypeError:
                refs = None
            if refs is not None:
                with lock:
                    cache[ckey] = refs
                    if isinstance(cache, OrderedDict):
                        cache.move_to_end(ckey)
                        while len(cache) > _DYNAMIC_CACHE_MAX:
                            cache.popitem(last=False)
        return rare_det, rep_det

    def _detect_token_only(
        self,
        tokens: Sequence[int],
        model_config: Any = None,
        tier: str = "intercept",
    ) -> DetectionResult:
        """执行单条序列的纯 Token 异常检测，综合生僻字/乱码（ill_type=1）与重复/兜圈（ill_type=3）按最早触发位置仲裁。"""
        if tier not in ("intercept", "monitor"):
            raise ValueError(f"tier must be 'intercept' or 'monitor', got {tier!r}")
        if isinstance(tokens, (str, bytes)):
            raise ValueError("tokens must be a 1D sequence of integer token IDs")
        tok = np.asarray(tokens, dtype=np.int64)
        if tok.ndim != 1:
            raise ValueError(f"tokens must be a 1D sequence of token IDs, got shape {tok.shape}")
        if len(tok) == 0:
            return DetectionResult()
        rare_det, rep_det = self._get_token_detectors(model_config)
        rc_at, rc_reason = rare_det.detect(tok, tier=tier)
        rep_at, rep_reason = rep_det.detect(tok, tier=tier)
        # 同时命中生僻字/乱码 (1) 与重复 (3) 时返回更早触发位置 fired_at；同位置并列时优先返回生僻字/乱码 (1)
        if rc_at >= 0 and (rep_at < 0 or rc_at <= rep_at):
            return DetectionResult(is_ill=True, ill_type=1, fired_at=rc_at, reason=rc_reason)
        if rep_at >= 0:
            return DetectionResult(is_ill=True, ill_type=3, fired_at=rep_at, reason=rep_reason)
        return DetectionResult()

    def _detect_with_logprobs(
        self,
        topk_logprobs: List[Dict[int, float]],
        tokens: List[int],
        model_config: Any = None,
    ) -> DetectionResult:
        """执行传统兼容模式的滑窗异常检测：依次判定 logprobs NaN/Inf（4）、生僻字（1）、乱码（2）及重复（3）。"""
        # 当 topk_logprobs 或 tokens 为空时，直接返回
        if not (topk_logprobs and tokens):
            return DetectionResult()

        # 若实例以纯 Token 方式构造（未预加载 YAML 配置），在此惰性初始化传统滑窗阈值与词表索引
        self._ensure_logprob_ready()

        tk2cat, vocab_size = self.get_tk2cat(tokens[-1], model_config)  # 获取token ids to cagetory

        cur_topk = getattr(self, "topk", None)
        self.topk = min(len(logp) for logp in topk_logprobs) if cur_topk is None else cur_topk

        # 对 topk_logprobs 降序排列，并取 topk 数据
        topk_logprobs = self._sort_and_truncate_topk(topk_logprobs)
        # 包含排序后 topk logprobs数据
        logprobs = np.array([list(item.values()) for item in topk_logprobs])

        # 如果logp为 nan/inf 时，直接返回 NaN Value
        if np.isnan(logprobs).any() or np.isinf(logprobs).any():
            return DetectionResult(is_ill=True, ill_type=4)

        if len(tokens) < self.stride:  # 只检测生僻字
            rare_flag, _ = self._detect_rare_character(topk_logprobs, logprobs, tk2cat, vocab_size)
            if rare_flag:
                return DetectionResult(is_ill=True, ill_type=1)
            return DetectionResult()

        # 初始化状态
        self._garbled_count = 0
        rare_flag = 0
        repet_counters = RepetitionCounters()

        # 滑窗检测
        for start, window_tokens in self.sliding_window(tokens):
            end = start + len(window_tokens)
            window_topk_logprobs = topk_logprobs[start:end]

            window_logprobs = logprobs[start:end]  # 包含 topk logprobs数据
            window_top1_logprobs = window_logprobs[:, 0]  # 包含 top1 logprobs数据

            # 1) 生僻字
            rare_in_window, _ = self._detect_rare_character(window_topk_logprobs, window_logprobs, tk2cat, vocab_size)
            rare_flag = rare_in_window or rare_flag  # 即使检出生僻字，也不停止

            # 2） 乱码
            garbled = self._detect_garbled(window_topk_logprobs, window_logprobs, tk2cat, vocab_size)
            if self._update_garbled_state(garbled):
                return DetectionResult(is_ill=True, ill_type=2)

            # 最后一个不完整的窗口小于阈值时跳过重复检测
            if len(window_top1_logprobs) < self.stride:
                continue

            # 3) 重复检测
            flag_repet, repet_counters = self._detect_repetitions(window_top1_logprobs, window_tokens, repet_counters)
            if flag_repet:
                return DetectionResult(is_ill=True, ill_type=3)

        # 当检查到最后，除生僻字外无其他异常，则返回生僻字异常
        if rare_flag:
            return DetectionResult(is_ill=True, ill_type=1)
        return DetectionResult()

    def _route_single(
        self,
        topk_logprobs: Any,
        tokens: Any,
        model_config: Any = None,
        tier: str = "intercept",
    ) -> DetectionResult:
        """单请求双模式路由：有非空 logprobs 走滑窗算法；为 None/未传走纯 Token 算法；空列表按分词器可用性兜底。"""
        if _has_logprobs(topk_logprobs):
            if _is_empty_sequence(tokens):
                return DetectionResult()
            return self._detect_with_logprobs(topk_logprobs, tokens, model_config)
        if topk_logprobs is _UNSET or topk_logprobs is None:
            return self._detect_token_only(tokens, model_config=model_config, tier=tier)
        # 兼容传统调用 detector([], tokens)：若可用分词器则回退纯 Token 检测，否则保持旧版行为返回正常结果
        if _is_empty_sequence(tokens):
            return DetectionResult()
        if self._can_resolve_token_only(model_config):
            return self._detect_token_only(tokens, model_config=model_config, tier=tier)
        return DetectionResult()

    def detect(
        self,
        tokens: Sequence[int],
        tier: str = "intercept",
        *,
        model_config: Any = None,
    ) -> DetectionResult:
        """基于纯 Token ID 序列检测单个请求的生僻字/乱码（1）与重复/兜圈（3）异常，返回含 fired_at 与 reason 的结果。"""
        return self._detect_token_only(tokens, model_config=model_config, tier=tier)

    # 主流程（默认无 logprob，传入 topk_logprobs 时走传统检测流程）
    @overload
    def detector(
        self,
        tokens: Sequence[int],
        model_config: Any = None,
        *,
        tier: str = "intercept",
    ) -> DetectionResult: ...

    @overload
    def detector(
        self,
        topk_logprobs: Optional[List[Dict[int, float]]],
        tokens: Sequence[int],
        model_config: Any = None,
        *,
        tier: str = "intercept",
    ) -> DetectionResult: ...

    def detector(self, *args: Any, **kwargs: Any) -> DetectionResult:
        """单请求统一检测入口，兼容纯 Token 签名 (tokens[, model_config]) 与传统签名 (topk_logprobs, tokens[, model_config])。

        返回 DetectionResult（包含 is_ill、ill_type，以及纯 Token 模式下的首次触发位置 fired_at 与命中规则 reason）。
        """
        tier = kwargs.pop("tier", "intercept")
        if tier not in ("intercept", "monitor"):
            raise ValueError(f"tier must be 'intercept' or 'monitor', got {tier!r}")
        topk_logprobs, tokens, model_config = _parse_detect_args(*args, **kwargs)
        return self._route_single(topk_logprobs, tokens, model_config=model_config, tier=tier)

    # 批量处理多个请求
    @overload
    def run(
        self,
        tokens: Sequence[Sequence[int]],
        model_configs: Any = None,
        *,
        tier: str = "intercept",
        return_detail: Literal[False] = False,
    ) -> List[List[Any]]: ...

    @overload
    def run(
        self,
        tokens: Sequence[Sequence[int]],
        model_configs: Any = None,
        *,
        tier: str = "intercept",
        return_detail: Literal[True],
    ) -> List[DetectionResult]: ...

    @overload
    def run(
        self,
        topk_logprobs: Optional[List[Optional[List[Dict[int, float]]]]],
        tokens: Sequence[Sequence[int]],
        model_configs: Any = None,
        *,
        tier: str = "intercept",
        return_detail: bool = False,
    ) -> Union[List[List[Any]], List[DetectionResult]]: ...

    def run(self, *args: Any, **kwargs: Any) -> Union[List[List[Any]], List[DetectionResult]]:
        """批请求统一检测入口，支持整批纯 Token 序列、整批 topk_logprobs 或逐请求混合路由。

        默认返回兼容二维列表 [[is_ill, ill_type], ...]；传入 return_detail=True 时返回 List[DetectionResult]。
        """
        tier = kwargs.pop("tier", "intercept")
        return_detail = kwargs.pop("return_detail", False)
        if tier not in ("intercept", "monitor"):
            raise ValueError(f"tier must be 'intercept' or 'monitor', got {tier!r}")
        topk_logprobs, tokens, model_configs = _parse_batch_args("run", *args, **kwargs)

        # 校验二维批次输入形态（拒绝字符串/字节串与非二维 NumPy 数组），并对齐每条请求的 model_config
        if isinstance(tokens, (str, bytes)):
            raise ValueError("tokens must be a batch of token ID sequences")
        if isinstance(tokens, np.ndarray):
            if tokens.ndim != 2:
                raise ValueError(f"tokens batch must be 2D, got shape {tokens.shape}")
            nums = tokens.shape[0]
        else:
            nums = len(tokens)

        mc_list = _normalize_model_configs(model_configs, nums)
        # 未提供 topk_logprobs 时整批直走纯 Token 路径；否则逐请求按是否存在 logprobs 混合路由
        if topk_logprobs is _UNSET or topk_logprobs is None:
            results = [self._detect_token_only(tokens[i], model_config=mc_list[i], tier=tier) for i in range(nums)]
        else:
            if len(topk_logprobs) != nums:
                raise ValueError(f"topk_logprobs length ({len(topk_logprobs)}) does not match tokens length ({nums})")
            results = [
                self._route_single(topk_logprobs[i], tokens[i], model_config=mc_list[i], tier=tier) for i in range(nums)
            ]

        if return_detail:
            return results
        return [[res.is_ill, res.ill_type] for res in results]
