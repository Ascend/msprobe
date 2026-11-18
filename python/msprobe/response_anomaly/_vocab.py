# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

"""分词器词表解析、原始字节解码与两级查表缓存（_BUNDLE_CACHE / _OBJ_CACHE）。"""

from __future__ import annotations

from collections import OrderedDict
import itertools
import json
import os
import sys
from threading import RLock
from typing import Any, Optional, Tuple
import weakref

import numpy as np

from msprobe.response_anomaly._rare_chars import (
    build_rare_tables,
    check_read_dir,
    check_read_file,
    load_rare_model,
)
from msprobe.response_anomaly._repetition import build_repetition_tables


def _default_cache_max_bytes(default_mb: int = 256) -> int:
    raw = os.environ.get("MSPROBE_VOCAB_CACHE_MB", "").strip()
    return int(raw) * 1024 * 1024 if raw.isdigit() and int(raw) > 0 else default_mb * 1024 * 1024


THINK_MARKERS = ("</think>", "<|end_of_thought|>", "[/THINK]", "<|thought_end|>")
_TOKENIZER_DICT_KEYS = ("tokenizer", "model_or_tokenizer", "model_path", "tokenizer_path")
_BUNDLE_CACHE_MAX = 64
_BUNDLE_CACHE_MAX_BYTES = _default_cache_max_bytes()
_NONWEAK_CACHE_MAX = 16

# 进程级统一缓存锁与全局单调访问序号（路径缓存、弱引用对象缓存与非弱引用对象缓存共享同一字节预算按 LRU 逐出）
_CACHE_LOCK = RLock()
_CACHE_SEQ = itertools.count()
# 文件路径级缓存：bkey -> [bundle, size, seq]
_BUNDLE_CACHE: OrderedDict = OrderedDict()
# 内存分词器实例缓存：支持弱引用对象存 _OBJ_CACHE（宿主回收即释放），不支持弱引用对象存 _NONWEAK_OBJ_CACHE
_OBJ_CACHE: OrderedDict = OrderedDict()
_NONWEAK_OBJ_CACHE: OrderedDict = OrderedDict()


def _byte_decoder():
    """构造 GPT-2 ByteLevel BPE 字符到单字节（0..255）的逆映射（68 个控制/空白字节自 U+0100 起顺延）。"""
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    rem = [b for b in range(256) if b not in bs]
    return {chr(c): b for b, c in zip(bs + rem, bs + [256 + i for i in range(len(rem))])}


def _steps(node):
    """递归展开 tokenizer.json 中可能嵌套为 Sequence 的 decoder / pre_tokenizer 阶段节点。"""
    if isinstance(node, dict):
        if node.get("type") == "Sequence":
            items = node.get("decoders") or node.get("pre_tokenizers") or ()
        else:
            items = (node,)
        for sub in items:
            if sub is node:
                yield node
            else:
                yield from _steps(sub)


def _parse_spec(spec, extra_special=()):
    """解析 HuggingFace tokenizer.json 结构，还原每个 token ID 对应的原始 UTF-8 字节串与特殊标记集合。"""
    model = spec.get("model") if isinstance(spec, dict) else None
    vocab_raw = model.get("vocab") if isinstance(model, dict) else None
    # Unigram 模型的 vocab 为 [[piece, score], ...] 列表；BPE 模型通常为 {piece: id} 字典
    if isinstance(vocab_raw, list):
        id_p = {i: e[0] for i, e in enumerate(vocab_raw)}
    elif isinstance(vocab_raw, dict):
        id_p = {int(i): str(p) for p, i in vocab_raw.items()}
    else:
        raise ValueError("invalid tokenizer JSON model.vocab")
    p2i = {p: i for i, p in id_p.items()}
    sp_ids = {int(x) for x in extra_special}
    added = {}
    nxt = max(id_p, default=-1) + 1
    for tok in spec.get("added_tokens", ()):
        c = tok["content"]
        tid = p2i.get(c, int(tok["id"]) if "id" in tok else nxt)
        if c not in p2i:
            id_p[tid] = c
            p2i[c] = tid
            nxt = max(nxt, tid + 1)
        added[tid] = c
        if tok.get("special", False):
            sp_ids.add(tid)
    vocab_size = max(id_p, default=-1) + 1

    # 识别解码器方案：GPT-2 ByteLevel vs SentencePiece（Metaspace / Unigram / ▁ 替换 / <0xXX> ByteFallback）
    decs = list(_steps(spec.get("decoder")))
    dtypes = {d.get("type") for d in decs}
    ptypes = {p.get("type") for p in _steps(spec.get("pre_tokenizer"))}
    is_bl = ("ByteLevel" in dtypes) or (not dtypes and "ByteLevel" in ptypes)
    has_rep = any(d.get("type") == "Replace" and d.get("pattern", {}).get("String") == "\u2581" for d in decs)
    is_spm = bool(dtypes & {"Metaspace", "ByteFallback"}) or model.get("type") == "Unigram" or has_rep
    bf = "ByteFallback" in dtypes or (is_spm and "<0x00>" in p2i and "<0xFF>" in p2i)
    bd = _byte_decoder() if is_bl else None
    rep = next((d.get("replacement", "\u2581") for d in decs if d.get("type") == "Metaspace"), "\u2581")

    chunks = []
    piece_lens = np.zeros(vocab_size, dtype=np.int32)
    missing_ids = set()
    for i in range(vocab_size):
        if i not in id_p:
            missing_ids.add(i)
            sp_ids.add(i)
            chunks.append(b"")
            continue
        p = id_p[i]
        piece_lens[i] = len(p)
        if is_bl:
            try:
                b = bytes(bd[c] for c in p)
            except KeyError as e:
                # added_tokens / special_tokens 或 get_vocab() 宽松模式下可能直接存放字面 UTF-8 字符而非 ByteLevel 映射字符
                if i not in added and i not in sp_ids and not spec.get("_lenient"):
                    raise ValueError(f"token {i} outside ByteLevel alphabet") from e
                b = p.encode("utf-8")
        elif is_spm and i not in sp_ids and i not in added:
            b = None
            if bf and len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
                try:
                    b = bytes([int(p[3:5], 16)])
                except ValueError:
                    pass
            chunks.append(p.replace(rep, " ").encode("utf-8") if b is None else b)
            continue
        else:
            b = p.encode("utf-8")
        chunks.append(b)
    think_det = next((int(p2i[m]) for m in THINK_MARKERS if m in p2i), None)
    return chunks, piece_lens, missing_ids, sp_ids, think_det


def _parse_object(tok_obj):
    """按优先级解析内存分词器实例：tokenizers.Tokenizer -> HF Fast -> tiktoken -> 慢速 get_vocab()。"""
    extra = {int(x) for x in (getattr(tok_obj, "all_special_ids", None) or ())}
    if hasattr(tok_obj, "to_str") and callable(tok_obj.to_str) and hasattr(tok_obj, "get_vocab_size"):
        return _parse_spec(json.loads(tok_obj.to_str()), extra)
    if hasattr(tok_obj, "backend_tokenizer") and hasattr(tok_obj.backend_tokenizer, "to_str"):
        return _parse_spec(json.loads(tok_obj.backend_tokenizer.to_str()), extra)
    if hasattr(tok_obj, "mergeable_ranks") or hasattr(tok_obj, "_mergeable_ranks"):
        ranks = getattr(tok_obj, "mergeable_ranks", None) or getattr(tok_obj, "_mergeable_ranks")
        id_b = {int(v): (k if isinstance(k, bytes) else str(k).encode("utf-8")) for k, v in ranks.items()}
        p2i = {k.decode("utf-8", "ignore"): int(v) for k, v in ranks.items() if isinstance(k, bytes)}
        for s_tok, tid in (getattr(tok_obj, "_special_tokens", None) or {}).items():
            tid = int(tid)
            id_b[tid], p2i[s_tok] = s_tok.encode("utf-8"), tid
            extra.add(tid)
        vocab_size = max(id_b, default=-1) + 1
        missing_ids = {i for i in range(vocab_size) if i not in id_b}
        extra.update(missing_ids)
        chunks = [id_b.get(i, b"") for i in range(vocab_size)]
        lens = [len(b.decode("utf-8", "ignore")) if b else 0 for b in chunks]
        piece_lens = np.array(lens, dtype=np.int32)
        think_det = next((p2i[m] for m in THINK_MARKERS if m in p2i), None)
        return chunks, piece_lens, missing_ids, extra, think_det
    if hasattr(tok_obj, "get_vocab") and callable(tok_obj.get_vocab):
        vd = tok_obj.get_vocab()
        if not isinstance(vd, dict) or not vd:
            raise ValueError("empty vocabulary from get_vocab()")
        # 慢速分词器无 JSON decoder 描述：抽样前 2000 个词片嗅探 ▁ (U+2581) 或 Ġ (U+0120) 推断空格编码方式
        sample = [str(k) for k in list(vd)[:2000]]
        spm = any("\u2581" in p for p in sample)
        bl = (not spm) and ("\u0120" in "".join(sample))
        if bl:
            dec = {"type": "ByteLevel"}
        elif spm:
            dec = {"type": "Sequence", "decoders": [{"type": "Metaspace"}, {"type": "ByteFallback"}]}
        else:
            dec = None
        lenient_spec = {
            "model": {"type": "Unigram" if spm else "BPE", "vocab": vd},
            "decoder": dec,
            "_lenient": True,
        }
        return _parse_spec(lenient_spec, extra)
    raise ValueError(f"unsupported tokenizer target: {type(tok_obj).__name__}")


def _decode_and_build(chunks, piece_lens, missing_ids, sp_ids, rare_model):
    """对单 token 原始字节做独立 UTF-8 解码，标记无法完整解码的残缺字节 token，并构建两套查表。"""
    texts = []
    bad_u8 = np.zeros(len(chunks), dtype=bool)
    for b in chunks:
        try:
            texts.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            bad_u8[len(texts)] = True
            texts.append(b.decode("utf-8", "ignore"))
    rare_tables = build_rare_tables(chunks, texts, bad_u8, sp_ids, rare_model)
    rep_tables = build_repetition_tables(chunks, texts, bad_u8, piece_lens, missing_ids, sp_ids)
    return rare_tables, rep_tables


def _resolve_think(think_end, detected, vocab_size):
    if think_end == "auto" or think_end is None:
        return detected if think_end == "auto" else None
    bad_type = isinstance(think_end, (bool, np.bool_)) or not isinstance(think_end, (int, np.integer))
    if bad_type or not 0 <= int(think_end) < vocab_size:
        raise ValueError(f"invalid think_end={think_end!r} for vocab size {vocab_size}")
    return int(think_end)


def _dir_vocab_signature(abspath: str) -> Tuple[Tuple[str, int, int], ...]:
    entries = []
    with os.scandir(abspath) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                st = entry.stat()
                entries.append((entry.name, st.st_mtime_ns, st.st_size))
    entries.sort()
    return tuple(entries)


def _value_size_bytes(val: Any) -> int:
    if isinstance(val, np.ndarray):
        return max(sys.getsizeof(val), int(val.nbytes))
    nbytes = getattr(val, "nbytes", None)
    if isinstance(nbytes, int):
        return sys.getsizeof(val) + nbytes
    if isinstance(val, list):
        return sys.getsizeof(val) + sum(map(sys.getsizeof, val))
    return sys.getsizeof(val)


def _bundle_size_bytes(bundle: Tuple[Any, Any, Any]) -> int:
    rare_tables, rep_tables, think_det = bundle
    size = sys.getsizeof(bundle) + sys.getsizeof(think_det)
    for tables in (rare_tables, rep_tables):
        size += sys.getsizeof(tables) + sum(map(_value_size_bytes, tables.values()))
    return size


def _evict_caches_locked(evicted: list) -> None:
    while len(_NONWEAK_OBJ_CACHE) > _NONWEAK_CACHE_MAX:
        evicted.append(_NONWEAK_OBJ_CACHE.popitem(last=False))
    caches = (_BUNDLE_CACHE, _OBJ_CACHE, _NONWEAK_OBJ_CACHE)
    total = sum(v[2] for c in caches for v in c.values())
    while total > _BUNDLE_CACHE_MAX_BYTES or sum(map(len, caches)) > _BUNDLE_CACHE_MAX:
        heads = [(next(iter(c.values()))[3], next(iter(c.values()))[2], c) for c in caches if c]
        if not heads:
            break
        _, drop_size, victim = min(heads, key=lambda x: x[0])
        evicted.append(victim.popitem(last=False))
        total -= drop_size


def _get_bundle_cache(bkey: Any) -> Optional[Tuple[Any, Any, Any]]:
    with _CACHE_LOCK:
        entry = _BUNDLE_CACHE.get(bkey)
        if entry is not None:
            entry[3] = next(_CACHE_SEQ)
            _BUNDLE_CACHE.move_to_end(bkey)
            return entry[1]
        return None


def _put_bundle_cache(bkey: Any, bundle: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
    size = _bundle_size_bytes(bundle)
    src_id = (bkey[0], bkey[1], *bkey[3:]) if bkey[0] == "dir_fallback" else ("file", bkey[0], *bkey[2:])
    evicted: list = []
    with _CACHE_LOCK:
        if size > _BUNDLE_CACHE_MAX_BYTES:
            for k in [k for k, v in _BUNDLE_CACHE.items() if v[0] == src_id]:
                evicted.append(_BUNDLE_CACHE.pop(k))
            res = bundle
        elif bkey in _BUNDLE_CACHE:
            _BUNDLE_CACHE[bkey][3] = next(_CACHE_SEQ)
            _BUNDLE_CACHE.move_to_end(bkey)
            res = _BUNDLE_CACHE[bkey][1]
        else:
            for k in [k for k, v in _BUNDLE_CACHE.items() if v[0] == src_id]:
                evicted.append(_BUNDLE_CACHE.pop(k))
            _BUNDLE_CACHE[bkey] = [src_id, bundle, size, next(_CACHE_SEQ)]
            _BUNDLE_CACHE.move_to_end(bkey)
            _evict_caches_locked(evicted)
            res = bundle
    del evicted
    return res


def _object_vocab_sig(tok_obj: Any) -> Tuple[Any, ...]:
    """生成 O(1) 轻量化词表状态签名，自动感知 add_tokens() / add_special_tokens() 引起的词表变化。"""
    if hasattr(tok_obj, "get_vocab_size") and callable(tok_obj.get_vocab_size):
        size_part: Any = int(tok_obj.get_vocab_size())
    elif hasattr(tok_obj, "backend_tokenizer") and callable(getattr(tok_obj.backend_tokenizer, "get_vocab_size", None)):
        size_part = int(tok_obj.backend_tokenizer.get_vocab_size())
    elif hasattr(tok_obj, "mergeable_ranks") or hasattr(tok_obj, "_mergeable_ranks"):
        ranks = getattr(tok_obj, "mergeable_ranks", None) or getattr(tok_obj, "_mergeable_ranks")
        size_part = len(ranks)
    elif hasattr(tok_obj, "__len__"):
        size_part = len(tok_obj)
    elif hasattr(tok_obj, "get_vocab") and callable(tok_obj.get_vocab):
        size_part = len(tok_obj.get_vocab())
    else:
        size_part = None

    special = tuple(int(x) for x in (getattr(tok_obj, "all_special_ids", None) or ()))
    tik_sp = getattr(tok_obj, "_special_tokens", None)
    tik_part = tuple(sorted((str(k), int(v)) for k, v in tik_sp.items())) if isinstance(tik_sp, dict) else ()
    added_dec = getattr(tok_obj, "get_added_tokens_decoder", None)
    if callable(added_dec):
        added_len: Any = len(added_dec())
    elif hasattr(tok_obj, "added_tokens_encoder"):
        added_len = len(tok_obj.added_tokens_encoder)
    else:
        added_len = None
    return size_part, special, tik_part, added_len


def _load_object_bundle(target: Any, think_end: Any, rare_model: Any):
    sig = (rare_model["mkey"], _object_vocab_sig(target))
    okey = (id(target), rare_model["mkey"])
    cached = None
    with _CACHE_LOCK:
        for cache, is_weak in ((_OBJ_CACHE, True), (_NONWEAK_OBJ_CACHE, False)):
            entry = cache.get(okey)
            if entry is not None and (entry[0]() if is_weak else entry[0]) is target:
                entry[3] = next(_CACHE_SEQ)
                cache.move_to_end(okey)
                cached = entry[1]
                break

    if cached is None or cached[0] != sig:
        chunks, piece_lens, missing_ids, sp_ids, think_det = _parse_object(target)
        rare_tables, rep_tables = _decode_and_build(chunks, piece_lens, missing_ids, sp_ids, rare_model)
        cached = (sig, rare_tables, rep_tables, think_det)
        size = _bundle_size_bytes((rare_tables, rep_tables, think_det))
        evicted: list = []
        if size > _BUNDLE_CACHE_MAX_BYTES:
            with _CACHE_LOCK:
                for cache, is_weak in ((_OBJ_CACHE, True), (_NONWEAK_OBJ_CACHE, False)):
                    old = cache.get(okey)
                    if old is not None and ((old[0]() in (target, None)) if is_weak else (old[0] is target)):
                        evicted.append(cache.pop(okey))
        else:

            def _on_gc(_ref: Any, _k: Any = okey) -> None:
                dropped = None
                with _CACHE_LOCK:
                    if _k in _OBJ_CACHE and _OBJ_CACHE[_k][0] is _ref:
                        dropped = _OBJ_CACHE.pop(_k)
                del dropped

            try:
                owner: Any = weakref.ref(target, _on_gc)
                dest, is_weak = _OBJ_CACHE, True
            except TypeError:
                owner, dest, is_weak = target, _NONWEAK_OBJ_CACHE, False
            with _CACHE_LOCK:
                cur = dest.get(okey)
                if cur is not None and (cur[0]() if is_weak else cur[0]) is target and cur[1][0] == sig:
                    cur[3] = next(_CACHE_SEQ)
                    dest.move_to_end(okey)
                    cached = cur[1]
                else:
                    if cur is not None:
                        evicted.append(dest.pop(okey))
                    dest[okey] = [owner, cached, size, next(_CACHE_SEQ)]
                    dest.move_to_end(okey)
                    _evict_caches_locked(evicted)
        del evicted

    return cached[1], cached[2], rare_model, _resolve_think(think_end, cached[3], len(cached[1]["rcls"]))


def load_vocab_bundle(target, think_end="auto", model_path=None):
    """Parse and cache rare-character and repetition lookup tables for a tokenizer target."""
    rare_model = load_rare_model(model_path)
    if isinstance(target, os.PathLike):
        target = os.fspath(target)
    if not isinstance(target, str):
        return _load_object_bundle(target, think_end, rare_model)

    expanded = os.path.expanduser(target)
    extra, cfg_think = set(), None
    if os.path.isdir(expanded):
        abspath = check_read_dir(expanded)
        # 补充读取 tokenizer_config.json 中声明的特殊 token 与思考结束标记（部分模型未写入 tokenizer.json）
        cfg_candidate = os.path.join(abspath, "tokenizer_config.json")
        if os.path.isfile(cfg_candidate):
            cfg_path = check_read_file(cfg_candidate)
            with open(cfg_path, encoding="utf-8") as f:
                for tid_s, meta in (json.load(f).get("added_tokens_decoder") or {}).items():
                    if isinstance(meta, dict):
                        if meta.get("special", False):
                            extra.add(int(tid_s))
                        if cfg_think is None and meta.get("content") in THINK_MARKERS:
                            cfg_think = int(tid_s)
        json_candidate = os.path.join(abspath, "tokenizer.json")
        json_path = check_read_file(json_candidate) if os.path.isfile(json_candidate) else None
    else:
        json_path = check_read_file(expanded)

    if json_path is None:
        bkey = (
            "dir_fallback",
            abspath,
            _dir_vocab_signature(abspath),
            tuple(sorted(extra)),
            cfg_think,
            rare_model["mkey"],
        )
    else:
        bkey = (json_path, os.stat(json_path).st_mtime_ns, tuple(sorted(extra)), cfg_think, rare_model["mkey"])
    bundle = _get_bundle_cache(bkey)
    if bundle is None:
        if json_path is None:
            # 目录不含 tokenizer.json 时，仅限本地离线且禁用 trust_remote_code 加载内置慢速分词器提取词表
            from transformers import AutoTokenizer

            try:
                tok_obj = AutoTokenizer.from_pretrained(  # nosec B615
                    abspath,
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except ValueError as exc:
                if "trust_remote_code" in str(exc):
                    raise ValueError(
                        f"{abspath}: custom tokenizer code execution is disabled (trust_remote_code=False); "
                        "pass an initialized in-memory tokenizer instance (e.g., llm.get_tokenizer()) instead."
                    ) from exc
                raise
            chunks, piece_lens, missing_ids, sp_ids, think_det = _parse_object(tok_obj)
            sp_ids |= extra
        else:
            with open(json_path, encoding="utf-8") as f:
                chunks, piece_lens, missing_ids, sp_ids, think_det = _parse_spec(json.load(f), extra)
        rare_tables, rep_tables = _decode_and_build(chunks, piece_lens, missing_ids, sp_ids, rare_model)
        bundle = _put_bundle_cache(bkey, (rare_tables, rep_tables, cfg_think if think_det is None else think_det))
    rare_tables, rep_tables, think_det = bundle
    return rare_tables, rep_tables, rare_model, _resolve_think(think_end, think_det, len(rare_tables["rcls"]))


def is_yaml_path(value: Any) -> bool:
    """Return True when value looks like a YAML configuration file path."""
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value).lower().endswith((".yaml", ".yml"))
    return False


def looks_like_tokenizer_path(value: Any) -> bool:
    """Return True if value is a PathLike, existing path, or path-shaped string rather than a bare model name."""
    if isinstance(value, os.PathLike):
        return True
    if not isinstance(value, str):
        return False
    if os.path.exists(os.path.expanduser(value)):
        return True
    return os.sep in value or "/" in value or value.startswith((".", "~")) or value.lower().endswith(".json")


def extract_tokenizer_target(model_config: Any) -> Tuple[Optional[Any], bool]:
    """Extract an explicit tokenizer target (path or object) from model_config if present."""
    if model_config is None:
        return None, False
    if isinstance(model_config, dict):
        for key in _TOKENIZER_DICT_KEYS:
            val = model_config.get(key)
            if val is not None:
                return val, True
        return None, False
    if isinstance(model_config, (str, os.PathLike)):
        if looks_like_tokenizer_path(model_config):
            return model_config, True
        return None, False
    return model_config, True
