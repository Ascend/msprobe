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

"""基于纯 Token ID 序列的重复生成与兜圈推理检测器（映射至 ill_type = 3）。

按每 BLOCK = 64 个原始 token（以及非整块结尾 n）的因果块边界评估三类规则组件（见 configs/repetition.json）：
  - V   : 类似 vLLM 的精确周期片段循环检测（尾部长度 [mn, mx] 的 token 片段连续出现 c 次）；
  - RUN : 单一非空白字符跨连续 token 的历史最长连击字符数（run >= min_chars）；
  - M   : 滑窗统计阈值规则，涵盖 4/8-gram 重复率与周期性、新颖度守卫（nov）、分隔符占比守卫（guard）、
          词级重复率（wrep）、反思转折词密度（mark）及 4-seed MinHash 句级近似复述率（sdup）。
"""

import json
import os

import numpy as np

from msprobe.response_anomaly._rare_chars import _TableDict, check_read_file, load_safe_npz


# 词表归一化后为空（纯空白/标点）的分隔符标记，不进入归一化词流 x
SEP = -1
# 特征评估的原始 token 块步长
BLOCK = 64
# N-gram 历史统计的回看窗口长度（按归一化词位置计）
H = 4096
# 64 位复合键中低 24 位存归一化位置下标（最高支持 2^24 个归一化词），高 40 位存 N-gram 哈希
POSBITS = 24
POSMASK = (1 << POSBITS) - 1
# 单个位置的历史同 N-gram 计数封顶值
CCAP = 255
# 统计的归一化 N-gram 阶数（4-gram 与 8-gram）
NGS = (4, 8)
# 位置级 N-gram 标志：
#   A2/A3/A5 : 窗口 H 内此前已出现 >= 2/3/5 次（即当前至少是第 3/4/6 次出现）
#   P        : 等距周期标志（c >= 2 且前两次间距满足 |d2 - 2*d1| <= d1 >> 3）
#   N        : 新颖标志（窗口 H 内此前出现 0 次，死循环时趋近于 0）
FLAGS = ("A2", "A3", "A5", "P", "N")
# N-gram 标志比例滑窗长度（按归一化词位置计，不足窗口长度时记为 NaN）
WINS = (256, 512, 1024)
# 原始 token 维度的结构特征（sep / ws_frac / chars）回看窗口长度
RAW_W = 256
# 句级近似复述参数：计入统计的最短句长 4、强制切句上限 256、历史比较句数 512、MinHash 最少一致种子数 3/4
SENT_MIN, SENT_MAX, SENT_HIST, SENT_AGREE = 4, 256, 512, 3
SEEDS = np.array([0x51ED27, 0x9E3779, 0x7F4A7C, 0x2545F4], np.uint64)
SDUP_NS = (16,)
WREP_WINS = (1024, 2048)
MARK_WINS = (1024,)
# 每个块边界输出的 38 维特征列名顺序
FEATS = [
    *[f"{f}_n{ng}_W{w}" for ng in NGS for f in FLAGS for w in WINS],
    "sep",
    "ws_frac",
    "chars",
    "run",
    *[f"wrep_W{w}" for w in WREP_WINS],
    *[f"mark_W{w}" for w in MARK_WINS],
    *[f"sdup_N{n}" for n in SDUP_NS],
]
# 无进展思维链（CoT）绕圈推理常反复出现的中英文反思/转折词（归一化后全小写字母数字）
MARKERS = {
    "wait",
    "hmm",
    "alternatively",
    "actually",
    "maybe",
    "perhaps",
    "but",
    "however",
    "reconsider",
    "again",
    "no",
    "ok",
    "okay",
    "let",
    "不过",
    "或者",
    "等等",
    "嗯",
    "但是",
    "可能",
    "再",
    "重新",
    "还是",
    "对吗",
}
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIGS_DIR, "repetition.json")
VOCAB_FIELDS = {"norm", "letter", "chars", "ws", "run_char", "marker", "send"}
_NPZ_CACHE = {}
_CONFIG_CACHE = {}

# SplitMix64 64 位黄金比例步进常数与两级雪崩混合乘子（见 _mix）
_HASH_C1 = np.uint64(0x9E3779B97F4A7C15)
_HASH_C2 = np.uint64(0xBF58476D1CE4E5B9)
_HASH_C3 = np.uint64(0x94D049BB133111EB)
# 64 位 FNV 标准质数，用作 N-gram 多项式滚动哈希的乘法基数
_PRIME = np.uint64(0x100000001B3)
# 64 位复合键的高 40 位哈希掩码（清零低 POSBITS=24 位的位置槽）
_HI = np.uint64(~POSMASK & 0xFFFFFFFFFFFFFFFF)
# 最低位置 1 强制转为奇数（与 2^64 互质），保证模 2^64 线性哈希无退化碰撞
_SEED_MUL = SEEDS | np.uint64(1)


def _mix(x):
    """SplitMix64 64 位整数混淆函数，将离散词 ID 打散为均匀分布的无符号整数。"""
    x = np.asarray(x, np.int64).view(np.uint64) + _HASH_C1
    x = (x ^ (x >> np.uint64(30))) * _HASH_C2
    x = (x ^ (x >> np.uint64(27))) * _HASH_C3
    return x ^ (x >> np.uint64(31))


def _ngram_hi(ids_window, m, ng):
    """计算长度为 m + ng - 1 的窗口内 m 个连续 ng-gram 的多项式滚动哈希，并仅保留高 40 位。"""
    x = _mix(ids_window)
    h = np.zeros(m, np.uint64)
    with np.errstate(over="ignore"):
        for i in range(ng):
            end = i + m
            h = h * _PRIME + x[i:end]
    return h & _HI


def _flags(c, d1, d2):
    """将窗口内历史频次 c 与最近两次间距 d1、d2 转为位置标志字典（P 允许 d1 >> 3 即 1/8 周期抖动）。"""
    c, d1, d2 = (np.asarray(a, np.int64) for a in (c, d1, d2))
    return {
        "A2": c >= 2,
        "A3": c >= 3,
        "A5": c >= 5,
        "P": (c >= 2) & (np.abs(d2 - 2 * d1) <= (d1 >> 3)),
        "N": c == 0,
    }


def _boundaries(n):
    """生成长度为 n 的序列对应的评估块边界（64, 128, ...；若末尾不满 64 则补齐 n）。"""
    b = list(range(BLOCK, n + 1, BLOCK))
    if not b or b[-1] != n:
        b.append(n)
    return np.array(b, np.int64) if n else np.zeros(0, np.int64)


def _minhash(ids):
    """基于 4 组线性同余种子计算词集合的 MinHash 签名（句内重复词不影响最小值）。"""
    x = _mix(np.asarray(ids, np.int64))
    with np.errstate(over="ignore"):
        return (x[:, None] * _SEED_MUL + SEEDS).min(0)


def _sentences(norm, send, sep):
    """按句末符号或长度上限切分有效句子（>= SENT_MIN 词），返回 [(句末原始 token 下标, 是否近似复述)]。"""
    out, cur, hist = [], [], []
    for i, (n, e) in enumerate(zip(norm.tolist(), send.tolist())):
        if n != sep:
            cur.append(n)
        if e or len(cur) >= SENT_MAX:
            if len(cur) >= SENT_MIN:
                mh = _minhash(cur)
                # 与前 SENT_HIST 句比较，至少 SENT_AGREE (3/4) 个哈希分量一致即视为复述句
                hist_arr = np.array(hist[-SENT_HIST:], np.uint64).reshape(-1, len(SEEDS))
                out.append((i, bool(((hist_arr == mh).sum(1) >= SENT_AGREE).any())))
                hist.append(mh)
            cur = []
    return out


def _position_stats(x, ng, answer_at=None):
    """向量化统计每个归一化位置在回看窗口 H 内的同 N-gram 历史频次 c 及最近两次间距 d1、d2。"""
    m = len(x)
    c, d1, d2 = (np.zeros(m, np.int64) for _ in range(3))
    if m < ng:
        return c, d1, d2
    j = np.arange(ng - 1, m, dtype=np.int64)
    # 将高 40 位 N-gram 哈希与低 24 位位置下标 j 拼入单个 uint64，排序后同哈希元素自然按位置升序排列
    keys = _ngram_hi(x, m - ng + 1, ng) | j.astype(np.uint64)
    s = np.sort(keys)
    hi = np.searchsorted(s, keys)
    floor = np.maximum(j - H, 0)
    if answer_at is not None:
        # 隔离思考与回答阶段：回答段复述思考段内容属正常总结，因此 j >= answer_at 时回看下界不越过 answer_at
        floor = np.where(j >= answer_at, np.maximum(floor, answer_at), floor)
    lo = np.searchsorted(s, (keys & _HI) | floor.astype(np.uint64))
    cnt = hi - lo
    pos = (s & np.uint64(POSMASK)).astype(np.int64)
    a = j - pos[np.maximum(hi - 1, 0)]
    b = j - pos[np.maximum(hi - 2, 0)]
    start = ng - 1
    c[start:] = np.minimum(cnt, CCAP)
    d1[start:] = np.where(cnt >= 1, a, 0)
    d2[start:] = np.where(cnt >= 2, b, 0)
    return c, d1, d2


def _win(cs, m_arr, w):
    """通过前缀和数组 cs 计算各边界最近 w 个元素的均值；不足 w 个元素时返回 NaN。"""
    out = np.full(len(m_arr), np.nan)
    ok = m_arr >= w
    out[ok] = (cs[m_arr[ok]] - cs[m_arr[ok] - w]) / w
    return out


def _runs_max(rc, chars):
    """无循环计算截至每个 token 的单一非空白字符跨连续 token 历史最大累计字符数。"""
    n = len(rc)
    if n == 0:
        return np.zeros(0, np.int64)
    # 当单字符码点变化或遇到非纯单字符 token (rc < 0) 时开启新段，用分段前缀和相减得到当前连击长度
    brk = np.ones(n, bool)
    brk[1:] = (rc[1:] != rc[:-1]) | (rc[1:] < 0)
    seg = np.cumsum(brk) - 1
    val = np.where(rc >= 0, chars, 0).astype(np.int64)
    cs = np.cumsum(val)
    start = np.flatnonzero(brk)
    seg_base = np.r_[0, cs][start]
    run_here = cs - seg_base[seg]
    run_here[rc < 0] = 0
    return np.maximum.accumulate(run_here)


def _answer_start(tok, keep, think_end):
    """定位首个 think_end 标记对应的归一化流起始下标；无思考结束标记时返回 None。"""
    if think_end is None:
        return None
    hit = np.flatnonzero(tok == think_end)
    if not len(hit):
        return None
    end = int(hit[0])
    return int(np.count_nonzero(keep[:end]))


def _features(tok, tables, think_end):
    """在所有评估块边界上提取因果的 38 维重复与兜圈特征矩阵（边界 bnd[k] 仅读取前 bnd[k] 个 token）。"""
    tok = np.asarray(tok, np.int64)
    n = len(tok)
    bnd = _boundaries(n)

    # 1) 过滤纯标点/空白分隔符 (SEP)，构建归一化词流 x，并建立原始块边界到归一化词数的映射 m_arr
    norm = tables["norm"][tok]
    keep = norm != SEP
    x = norm[keep].astype(np.int64)
    m_arr = np.r_[0, np.cumsum(keep)][bnd]
    answer_at = _answer_start(tok, keep, think_end)
    cols = {}

    # 2) N-gram 滑窗特征（30 列）：按归一化窗口 WINS (256/512/1024) 统计 4/8-gram 各标志位占比
    for ng in NGS:
        fl = _flags(*_position_stats(x, ng, answer_at))
        for f in FLAGS:
            cs = np.r_[0, np.cumsum(fl[f])]
            for w in WINS:
                cols[f"{f}_n{ng}_W{w}"] = _win(cs, m_arr, w)

    # 3) 原始 token 窗口结构特征（最近 RAW_W=256 个 token 的分隔符率、空白字符率、字符总数）与全局最长单字符连击 run
    lo = np.maximum(bnd - RAW_W, 0)
    span = np.maximum(bnd - lo, 1)
    sep_cs = np.r_[0, np.cumsum(~keep)]
    ch = tables["chars"][tok].astype(np.int64)
    ws = tables["ws"][tok].astype(np.int64)
    ch_cs, ws_cs = np.r_[0, np.cumsum(ch)], np.r_[0, np.cumsum(ws)]
    cols["sep"] = (sep_cs[bnd] - sep_cs[lo]) / span
    chars = ch_cs[bnd] - ch_cs[lo]
    cols["ws_frac"] = np.where(chars > 0, (ws_cs[bnd] - ws_cs[lo]) / np.maximum(chars, 1), 0.0)
    cols["chars"] = chars.astype(np.float64)
    rm = _runs_max(tables["run_char"][tok].astype(np.int64), ch)
    cols["run"] = rm[bnd - 1].astype(np.float64) if n else np.zeros(0)

    # 4) 词级重复率 wrep（归一化窗口内非首次出现的词占比）与反思词密度 mark（原始 token 窗口占比）
    rep = np.ones(len(x), bool)
    rep[np.unique(x, return_index=True)[1]] = False
    rcs = np.r_[0, np.cumsum(rep)]
    for w in WREP_WINS:
        cols[f"wrep_W{w}"] = _win(rcs, m_arr, w)
    mcs = np.r_[0, np.cumsum(tables["marker"][tok])]
    for w in MARK_WINS:
        cols[f"mark_W{w}"] = _win(mcs, bnd, w)

    # 5) MinHash 句级复述率 sdup：仅纳入在当前边界前已闭合的句子（side="left" 保证句末下标 < bnd 的因果性）
    sf = _sentences(norm, tables["send"][tok], SEP)
    s_idx = np.searchsorted(np.array([e for e, _ in sf], np.int64), bnd, side="left")
    dcs = np.r_[0, np.cumsum([d for _, d in sf])]
    for nn in SDUP_NS:
        cols[f"sdup_N{nn}"] = _win(dcs, s_idx, nn)

    mat = np.stack([cols[f] for f in FEATS], 1).astype(np.float32) if n else np.zeros((0, len(FEATS)), np.float32)
    return bnd, mat


def _vllm_first_prefix(tok, mn, mx, c):
    """查找存在周期 p in [mn, mx] 连续重复 c 次的最短前缀长度 t；未触发返回 -1。"""
    tok = np.asarray(tok, np.int64)
    n, best = len(tok), -1
    for p in range(max(mn, 1), mx + 1):
        need = (c - 1) * p
        if c * p > n:
            break
        # eq[i] 对应 tok[i + p] == tok[i]；通过一阶差分定位首个长度 >= (c - 1) * p 的全 True 区间
        eq = tok[p:] == tok[:-p]
        d = np.diff(np.r_[0, eq.view(np.int8), 0])
        s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        k = (e - s) >= need
        if k.any():
            t = int(s[k][0]) + p + need
            best = t if best < 0 else min(best, t)
    return best


def _describe(comp):
    k = comp["kind"]
    if k == "M":
        parts = [f"{comp['feat']}>={comp['theta']}"]
        if comp.get("S", 1) > 1:
            parts.append(f" S{comp['S']}")
        if comp.get("guard") is not None:
            parts.append(f" sep<={comp['guard']}")
        if comp.get("nov"):
            parts.append(f" {comp['nov']['feat']}<={comp['nov']['max']}")
        for a in comp.get("and", []):
            op = ">=" if "min" in a else "<="
            parts.append(f" & {a['feat']}{op}{a.get('min', a.get('max'))}")
        return "".join(parts)
    if k == "V":
        return f"vllm[{comp['mn']},{comp['mx']}]x{comp['c']}"
    if k == "RUN":
        return f"run>={comp['min_chars']}"
    raise ValueError(f"unsupported repetition component kind: {k}")


def build_repetition_tables(chunks, texts, bad_u8, piece_lens, missing_ids, special_ids):
    """从解码后的词表构建重复检测所需的 7 列一维查表数组。

    各列含义：
      - norm     : 小写字母数字归一化词 ID（纯标点/空白为 SEP = -1；特殊 token 与残缺字节各自分配独立 ID）
      - letter   : 归一化形式是否含字母（残缺 UTF-8 字节 token 亦标为 True）
      - chars    : 解码字符数（int16 饱和截断）
      - ws       : 空白字符数（int16 饱和截断）
      - run_char : 若整个 token 由同一非空白字符重复组成（如 "===="），记录该字符码点，否则为 -1
      - marker   : 归一化词是否命中思维链反思词表 MARKERS
      - send     : 是否包含句末标点（. ? ! 。 ？ ！）或换行符
    """
    vocab_size = len(chunks)
    norm = np.full(vocab_size, SEP, np.int32)
    letter = np.zeros(vocab_size, bool)
    chars = np.zeros(vocab_size, np.int16)
    ws = np.zeros(vocab_size, np.int16)
    run_char = np.full(vocab_size, -1, np.int32)
    marker = np.zeros(vocab_size, bool)
    send = np.zeros(vocab_size, bool)
    ids = {}

    for i, (b, s, is_bad, plen) in enumerate(zip(chunks, texts, bad_u8.tolist(), piece_lens.tolist())):
        if i in missing_ids:
            continue
        if i in special_ids:
            norm[i] = ids.setdefault(("special", i), len(ids) + 1)
            chars[i] = min(int(plen), 32767)
            continue
        if is_bad:
            norm[i] = ids.setdefault(("bytes", b), len(ids) + 1)
            letter[i] = True
            chars[i] = 1
            continue
        chars[i] = min(len(s), 32767)
        send[i] = any(c in ".?!。？！\n" for c in s)
        ws[i] = min(sum(c.isspace() for c in s), 32767)
        if s and not s[0].isspace() and s == s[0] * len(s):
            run_char[i] = ord(s[0])
        key = "".join(c for c in s.lower() if c.isalnum())
        if key:
            norm[i] = ids.setdefault(key, len(ids) + 1)
            letter[i] = any(c.isalpha() for c in key)
            marker[i] = key in MARKERS
    return _TableDict(norm=norm, letter=letter, chars=chars, ws=ws, run_char=run_char, marker=marker, send=send)


def validate_repetition_tables(tables, source="<memory>"):
    """校验重复检测词表的 7 个必需字段均存在，且为一维等长的数值或布尔数组。"""
    missing = VOCAB_FIELDS - set(tables.keys())
    if missing:
        raise ValueError(f"{source}: not a repetition vocabulary: missing {', '.join(sorted(missing))}")
    vocab_size = len(tables["norm"])
    for k in VOCAB_FIELDS:
        if tables[k].ndim != 1 or len(tables[k]) != vocab_size or tables[k].dtype.kind not in "biuf":
            raise ValueError(f"{source}: invalid {k}")
    return tables


def load_vocab_npz(path):
    """安全加载并校验预构建的 7 列重复检测词表 .npz 文件（按路径与修改时间缓存）。"""
    abspath = check_read_file(path)
    key = (abspath, os.stat(abspath).st_mtime_ns)
    if key not in _NPZ_CACHE:
        tables = load_safe_npz(abspath)
        validate_repetition_tables(tables, source=abspath)
        tables["_path"] = abspath
        _NPZ_CACHE[key] = tables
    return _NPZ_CACHE[key]


def load_repetition_config(config=None):
    """加载并缓存重复检测规则组件配置（支持直接传入字典或 JSON 文件路径）。"""
    if isinstance(config, dict):
        return config
    abspath = check_read_file(config if config is not None else DEFAULT_CONFIG_PATH)
    key = (abspath, os.stat(abspath).st_mtime_ns)
    if key not in _CONFIG_CACHE:
        with open(abspath, encoding="utf-8") as f:
            _CONFIG_CACHE[key] = json.load(f)
    return _CONFIG_CACHE[key]


class RepetitionDetector:
    """基于纯 Token ID 序列的整条记录重复生成与兜圈推理检测器。"""

    def __init__(self, tables, think_end=None, config=None):
        self.tables = validate_repetition_tables(tables)
        self.think_end = think_end
        self.cfg = load_repetition_config(config)
        self.comps = self.cfg["components"]
        unsupported = {c["kind"] for c in self.comps} - {"V", "RUN", "M"}
        if unsupported:
            raise ValueError(f"unsupported repetition component kinds: {sorted(unsupported)}")

    @classmethod
    def from_npz(cls, path, think_end="auto", config=None):
        """从预构建的词表 .npz 文件直接构造检测器（think_end='auto' 时读取归档内的 chat / think_end 元数据）。"""
        tables = load_vocab_npz(path)
        vocab_size = len(tables["norm"])
        if think_end == "auto":
            if tables.get("chat") is not None and bool(tables["chat"]):
                resolved_think = None
            elif tables.get("think_end") is not None and int(tables["think_end"]) >= 0:
                resolved_think = int(tables["think_end"])
            else:
                raise ValueError("think_end='auto' requires bundle metadata in the .npz file")
        elif think_end is None:
            resolved_think = None
        else:
            bad_type = isinstance(think_end, (bool, np.bool_)) or not isinstance(think_end, (int, np.integer))
            if bad_type or not 0 <= int(think_end) < vocab_size:
                raise ValueError(f"invalid think_end={think_end!r} for vocab size {vocab_size}")
            resolved_think = int(think_end)
        return cls(tables, think_end=resolved_think, config=config)

    def _check_tokens(self, tokens):
        tok = np.asarray(tokens, np.int64)
        if tok.ndim != 1:
            raise ValueError(f"tokens must be a 1D sequence of token IDs, got shape {tok.shape}")
        vocab_size = len(self.tables["norm"])
        if len(tok) and (int(tok.min()) < 0 or int(tok.max()) >= vocab_size):
            raise ValueError(f"token id out of range [0, {vocab_size})")
        return tok

    def record_features(self, tokens):
        """校验输入 token 序列并提取各评估块边界上的 38 维重复与兜圈特征矩阵。"""
        tok = self._check_tokens(tokens)
        return _features(tok, self.tables, self.think_end)

    def detect(self, tokens, tier="intercept"):
        """检测重复生成或兜圈推理异常，返回 (最早触发块边界 fired_at, 命中组件描述)；正常时返回 (-1, "")。"""
        if tier not in ("intercept", "monitor"):
            raise ValueError(f"tier must be 'intercept' or 'monitor', got {tier!r}")
        tok = self._check_tokens(tokens)
        bnd, feat_mat = self.record_features(tok)
        col = {f: i for i, f in enumerate(FEATS)}
        best = (-1, "")
        for comp in self.comps:
            if tier == "intercept" and comp.get("tier", "intercept") == "monitor":
                continue
            k = comp["kind"]
            if k == "M":
                # 窗口尚未填满产生的 NaN 按保守方向替换（下界条件填 -1，上界条件填 2），确保不误触发
                cond = np.nan_to_num(feat_mat[:, col[comp["feat"]]].astype(np.float64), nan=-1) >= comp["theta"]
                if comp.get("guard") is not None:
                    cond &= feat_mat[:, col["sep"]].astype(np.float64) <= comp["guard"]
                if comp.get("nov"):
                    nov_col = col[comp["nov"]["feat"]]
                    cond &= np.nan_to_num(feat_mat[:, nov_col].astype(np.float64), nan=2) <= comp["nov"]["max"]
                for a in comp.get("and", []):
                    v = feat_mat[:, col[a["feat"]]].astype(np.float64)
                    cond &= np.nan_to_num(v, nan=-1) >= a["min"] if "min" in a else np.nan_to_num(v, nan=2) <= a["max"]
            elif k == "RUN":
                cond = feat_mat[:, col["run"]] >= comp["min_chars"]
            else:
                # V 组件（精确周期重复）与流式每块检查语义对齐：在 >= 首次重复前缀长度 t 的首个块边界上报
                t = _vllm_first_prefix(tok, comp["mn"], comp["mx"], comp["c"])
                cond = np.zeros(len(bnd), bool)
                if t >= 0:
                    idx = int(np.searchsorted(bnd, t))
                    cond[idx:] = True
            # 要求条件在连续 S 个评估块边界上均成立；多组件同边界命中时保留配置列表中靠前的组件
            streak, hit = 0, -1
            for i, ok in enumerate(cond.tolist()):
                streak = streak + 1 if ok else 0
                if streak >= comp.get("S", 1):
                    hit = int(bnd[i])
                    break
            if hit >= 0 and (best[0] < 0 or hit < best[0]):
                best = hit, _describe(comp)
        return best
