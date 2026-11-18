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

"""基于纯 Token ID 序列的生僻字与 UTF-8 乱码检测器（映射至 ill_type = 1）。

结合增量 UTF-8 解码校验、Unicode 文字体系分类、传统字符集编码层级以及内置轻量 GBDT 打分模型
（tables/rare_chars_model.npz，仅在少数文字字母 fs < 0.05 或字节损坏处门控求值），识别跨语种突兀乱码、
极罕见生僻字及残缺字节拼合故障。
"""

import codecs
from collections.abc import Sequence
import os
import unicodedata as unicd
import zipfile

import numpy as np

from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.log import logger


# 20 类 token 文字/符号分类编码（0..19）；序列首尾无邻居哨兵使用编码 31
_CLS_A = "space_punct latin digit han kana hangul cyrillic greek arabic hebrew thai"
_CLS_B = "indic other_letter symbol emoji partial replacement control mixed combining special"
CLASSES = f"{_CLS_A} {_CLS_B}".split()
CI = {c: i for i, c in enumerate(CLASSES)}
_LET_NAMES = "latin han kana hangul cyrillic greek arabic hebrew thai indic other_letter mixed".split()
# 标记 12 类文字字母（用于统计前缀同文字字母占比 fs 与局部窗口占比 fsw）
_LET = np.isin(np.arange(len(CLASSES)), [CI[c] for c in _LET_NAMES])
# K=16 种首尾邻接字符细分类别（见 _charcat）；WIN=64 为 fsw 回看字母数；GB_GATE=0.05 为少数文字 GBDT 门控阈值
K, WIN, GB_GATE = 16, 64, 0.05
_CJK_PUNCT_A = "\u3002\uff0c\u3001\uff1b\uff1a\uff1f\uff01\u201c\u201d\u2018\u2019\uff08\uff09"
_CJK_PUNCT_B = "\u3010\u3011\u300a\u300b\u2014\u2026\u00b7\u300c\u300d\u300e\u300f\u3008\u3009"
_CJK_PUNCT = set(_CJK_PUNCT_A + _CJK_PUNCT_B)
# Unicode 字母码点区间（成对的半开区间 [lo, hi)）到 10 类具体文字名称的映射
_LETTER_SCRIPTS = (
    ((0x0, 0x250, 0x1E00, 0x1F00, 0xFF21, 0xFF5B), "latin"),
    ((0x370, 0x400, 0x1F00, 0x2000), "greek"),
    ((0x400, 0x530), "cyrillic"),
    ((0x590, 0x600), "hebrew"),
    ((0x600, 0x700, 0x750, 0x780, 0xFB50, 0xFF00), "arabic"),
    ((0x900, 0xE00), "indic"),
    ((0xE00, 0xE80), "thai"),
    ((0x3040, 0x3100, 0x31F0, 0x3200, 0xFF66, 0xFFA0), "kana"),
    ((0xAC00, 0xD7B0, 0x1100, 0x1200, 0x3130, 0x3190), "hangul"),
    ((0x3400, 0xA000, 0xF900, 0xFB00, 0x20000, 0x31350), "han"),
)
# 非拉丁文字名称到 16 类邻接字符细分编码（见 _charcat）的归并映射
_SCRIPT_CAT = {
    "han": 9,
    "kana": 13,
    "hangul": 13,
    "greek": 11,
    "cyrillic": 11,
    "arabic": 11,
    "hebrew": 11,
    "thai": 11,
    "indic": 11,
    "other_letter": 11,
}
# GBDT 模型的 21 维输入特征：
#   cls/pcls/ncls : 当前/前/后 token 的文字类编码（序列外邻居取哨兵 31）
#   first/last    : 当前 token 首/尾字符的粗粒度类型（0..5，见 _kind）；lead 为是否以空白字节开头
#   pcat/ncat     : 前 token 尾字符 / 后 token 首字符的 16 类细分编码（0..15，见 _charcat）
#   fs / fsw      : 当前文字类在前缀全部字母 / 最近 64 个字母中的同类别占比
#   answer        : 是否已进入 think_end 之后的正式回答阶段；bad_utf8 为增量 UTF-8 解码是否损坏
#   u / u_mean    : token 内字符在参考语料中的最小 / 平均 log10 一元频率
#   cp / cn       : 首字符前接 pcat、尾字符后接 ncat 的平滑 log10 条件转移概率
#   c_max / c_mean: token 内字符传统字符集稀有层级（0..4，见 _codec_tier）的最大值 / 均值
#   c_len / c_rare / c_nscripts : 解码字符长度、是否含层级 >= 3 的生僻字符、非标点空白文字体系数
_GB_A = "cls pcls ncls first last lead pcat ncat fs fsw answer bad_utf8"
_GB_B = "u u_mean cp cn c_max c_mean c_len c_rare c_nscripts"
GB_INPUTS = f"{_GB_A} {_GB_B}".split()
# 仅依赖单 token 自身的 11 维静态特征列，在词表加载期预计算存入 tables["tf"]
_TOKF_NAMES = "cls first last lead u u_mean c_max c_mean c_len c_rare c_nscripts".split()
_TOKF_COLS = [GB_INPUTS.index(k) for k in _TOKF_NAMES]
_COL = {k: i for i, k in enumerate(GB_INPUTS)}
TABLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
DEFAULT_MODEL_PATH = os.path.join(TABLES_DIR, "rare_chars_model.npz")
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_NPZ_ENTRIES = 64

_TIER_CACHE, _MODEL_CACHE = {}, {}


def _warn_if_symlink(path):
    """逐级检查路径组件是否包含软链接；若命中则记录安全提示日志（不阻断读取）。"""
    cur = os.path.sep if os.path.isabs(path) else ""
    for part in os.path.abspath(path).split(os.path.sep):
        if not part:
            continue
        cur = os.path.join(cur, part) if cur else part
        if os.path.islink(cur):
            logger.warning(f"The path component [{cur}] is a soft link. Please ensure the file is trusted.")


def check_read_file(path, max_size=_MAX_FILE_BYTES):
    """校验输入文件的存在性、软链接提示、路径长度、常规文件类型、可读性及字节大小上限，返回规范化真实路径。"""
    raw = os.path.expanduser(os.fspath(path))
    if not os.path.exists(raw):
        raise FileNotFoundError(f"{path}: file not found")
    _warn_if_symlink(raw)
    real = os.path.realpath(raw)
    if len(real) > FileCheckConst.DIRECTORY_LENGTH or len(os.path.basename(real)) > FileCheckConst.FILE_NAME_LENGTH:
        raise ValueError(f"{path}: file path length exceeds limit")
    if not os.path.isfile(real):
        raise ValueError(f"{path}: expected a regular file")
    if not os.access(real, os.R_OK):
        raise PermissionError(f"{path}: file is not readable")
    if os.path.getsize(real) > max_size:
        raise ValueError(f"{path}: file size exceeds {max_size} bytes")
    return real


def check_read_dir(path):
    """校验输入目录的存在性、软链接提示、路径长度、目录类型及可读性，返回规范化真实路径。"""
    raw = os.path.expanduser(os.fspath(path))
    if not os.path.exists(raw):
        raise FileNotFoundError(f"{path}: directory not found")
    _warn_if_symlink(raw)
    real = os.path.realpath(raw)
    if len(real) > FileCheckConst.DIRECTORY_LENGTH:
        raise ValueError(f"{path}: directory path length exceeds limit")
    if not os.path.isdir(real):
        raise ValueError(f"{path}: expected a directory")
    if not os.access(real, os.R_OK):
        raise PermissionError(f"{path}: directory is not readable")
    return real


def load_safe_npz(real_path, max_size=_MAX_FILE_BYTES):
    """预检 .npz 归档的条目数、路径穿越风险与解压总大小（防 Zip 炸弹），并以禁用 pickle 方式安全加载数组。"""
    with zipfile.ZipFile(real_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_NPZ_ENTRIES:
            raise ValueError(f"{real_path}: too many entries in .npz archive")
        total = 0
        for member in infos:
            if os.path.isabs(member.filename) or ".." in member.filename.replace("\\", "/").split("/"):
                raise ValueError(f"{real_path}: unsafe archive entry {member.filename!r}")
            total += member.file_size
            if total > max_size:
                raise ValueError(f"{real_path}: uncompressed .npz size exceeds {max_size} bytes")
    with np.load(real_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _script(ch):
    """根据 Unicode 码点与通用类别，判定单个字符所属的文字或符号类别名称。"""
    o, cat = ord(ch), unicd.category(ch)
    if o == 0xFFFD:
        return "replacement"
    if cat in ("Cc", "Cf", "Co", "Cs", "Cn") and ch not in "\n\r\t":
        return "control"
    if cat[0] == "M":
        return "combining"
    if cat[0] == "N" and cat[1:] == "d":
        return "digit" if o < 128 else "other_letter"
    if cat[0] == "L":
        for rng, name in _LETTER_SCRIPTS:
            if any(rng[j] <= o < rng[j + 1] for j in range(0, len(rng), 2)):
                return name
        return "other_letter"
    if cat == "So" and (o >= 0x1F000 or 0x2600 <= o < 0x2800):
        return "emoji"
    return "symbol" if cat in ("So", "Sm", "Sc", "Sk") and o >= 128 else "space_punct"


def _classify(s):
    """聚合解码文本 s 中所有非空白标点字符的类别，确定整个 token 的主类别名称（多语种字母混杂记为 mixed）。"""
    kinds = {_script(ch) for ch in s} - {"space_punct"}
    for k in ("replacement", "control"):
        if k in kinds:
            return k
    letters = kinds - {"digit", "combining", "symbol", "emoji"}
    if letters:
        return "mixed" if len(letters) > 1 else next(iter(letters))
    return next((k for k in ("emoji", "symbol", "combining", "digit") if k in kinds), "space_punct")


def _kind(ch):
    """粗粒度字符类型：0 空/空白、1 小写字母、2 大写字母、3 数字、4 标点符号、5 无大小写字母（如汉字/假名/阿拉伯文）。"""
    if not ch or ch.isspace():
        return 0
    return 3 if ch.isdigit() else ((1 if ch.islower() else 2 if ch.isupper() else 5) if ch.isalpha() else 4)


def _charcat(ch):
    """邻接上下文的 16 类字符细分编码（0 空、1 空白、2 换行、3/4 大小写拉丁、5 数字、6..8/14/15 ASCII 标点、9..13 非拉丁）。"""
    if not ch or ch.isspace():
        return 0 if not ch else (2 if ch in "\n\r" else 1)
    o = ord(ch)
    if o < 128:
        if "a" <= ch <= "z":
            return 3
        if "A" <= ch <= "Z":
            return 4
        if "0" <= ch <= "9":
            return 5
        if ch in ".,;:!?":
            return 6
        if ch in "()[]{}\"'`":
            return 7
        if ch == "\\":
            return 14
        return 15 if ch == "_" else 8
    s = _script(ch)
    if s == "latin":
        return 4 if ch.isupper() else 3
    if s in _SCRIPT_CAT:
        return _SCRIPT_CAT[s]
    return 10 if (0x3000 <= o < 0x3040 or 0xFF00 <= o < 0xFF66 or unicd.category(ch)[0] == "P") else 12


def _enc(ch, codec):
    """尝试以指定传统字符集编码单个字符；若该字符不在编码表内则返回 None。"""
    try:
        return ch.encode(codec)
    except UnicodeEncodeError:
        return None


def _codec_tier(ch):
    """基于传统字符编码标准评估字符生僻度（0..4，数值越大越罕见）。

    0: 可打印 ASCII 与常规空白；
    1: GB2312 一级汉字（0xB0-0xD7）、Big5 常用字（0xA4-0xC6）、EUC-KR 常用谚文、平片假名、Latin-1、基础希腊及常用标点；
    2: GB2312 二级字、EUC-KR 次常用字、Latin-2、基础西里尔/阿拉伯/希伯来/天城文/TIS-620 泰文及 Emoji；
    3: GBK 扩展汉字、非常用字母变体及一般符号；
    4: 超出 GBK 的极生僻汉字、罕见扩展文字、控制/未分配字符及 U+FFFD。
    """
    t = _TIER_CACHE.get(ch)
    if t is not None:
        return t
    o, cat = ord(ch), unicd.category(ch)
    if o < 128:
        t = 0 if (ch >= " " or ch in "\n\r\t") else 4
    elif o == 0xFFFD or cat in ("Cc", "Cf", "Co", "Cs", "Cn"):
        t = 4
    else:
        s = _script(ch)
        if s == "han":
            b2 = _enc(ch, "gb2312")
            if b2:
                t = 1 if 0xB0 <= b2[0] <= 0xD7 else 2
            else:
                b5 = _enc(ch, "big5")
                t = (1 if 0xA4 <= b5[0] <= 0xC6 else 3) if b5 else (3 if _enc(ch, "gbk") else 4)
        elif s == "hangul":
            b = _enc(ch, "euc_kr")
            t = (1 if len(b) == 2 and 0xB0 <= b[0] <= 0xC8 else 2) if b else 4
        elif s == "kana":
            t = 1 if (0x3041 <= o <= 0x3096) or (0x30A1 <= o <= 0x30FA) or o in (0x30FC, 0x30FB) else 3
        elif s == "latin":
            t = 1 if _enc(ch, "iso8859_1") else (2 if _enc(ch, "iso8859_2") else 3)
        elif s == "greek":
            t = 1 if (0x0391 <= o <= 0x03A9 and o != 0x03A2) or (0x03B1 <= o <= 0x03C9) else 3
        elif s == "cyrillic":
            t = 2 if (0x0410 <= o <= 0x044F) or o in (0x0401, 0x0451) else 4
        elif s == "arabic":
            t = 2 if 0x0621 <= o <= 0x064A else 4
        elif s == "thai":
            t = 2 if _enc(ch, "tis_620") else 4
        elif s in ("hebrew", "indic"):
            t = 2 if (0x05D0 <= o <= 0x05EA) or (0x0904 <= o <= 0x0939) else 3
        elif ch in _CJK_PUNCT or (0x2000 <= o <= 0x206F) or (0x2190 <= o <= 0x22FF) or _enc(ch, "gb2312"):
            t = 1
        else:
            t = 2 if s == "emoji" else 3
    _TIER_CACHE[ch] = t
    return t


class _TableDict(dict):
    """支持弱引用的查表字典子类，便于实例级二级缓存在进程缓存逐出时随 GC 自动释放。"""


class _PackedBytes(Sequence):
    """将词表 bytes 列表紧凑打包为单一连续字节块与偏移数组，消除海量小对象碎片。"""

    __slots__ = ("_buf", "_off")

    def __init__(self, chunks):
        self._buf = b"".join(chunks)
        dtype = np.int32 if len(self._buf) <= np.iinfo(np.int32).max else np.int64
        self._off = np.r_[dtype(0), np.cumsum(np.fromiter(map(len, chunks), dtype=dtype, count=len(chunks)))]

    @property
    def nbytes(self):
        return len(self._buf) + int(self._off.nbytes)

    def __len__(self):
        return len(self._off) - 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        i = int(idx) + (len(self) if int(idx) < 0 else 0)
        if not 0 <= i < len(self):
            raise IndexError("packed token index out of range")
        return self._buf[int(self._off[i]) : int(self._off[i + 1])]


def load_rare_model(model_path=None):
    """加载并按 (真实路径, 修改时间) 缓存生僻字 GBDT 打分模型及字符频率/转移统计表。"""
    p = check_read_file(model_path if model_path is not None else DEFAULT_MODEL_PATH)
    mkey = (p, os.stat(p).st_mtime_ns)
    if mkey not in _MODEL_CACHE:
        d = load_safe_npz(p)
        inputs = [str(x) for x in d["gb_inputs"]]
        cols = [GB_INPUTS.index(k) for k in inputs]
        pad_row = np.full((1, K), np.float32(d["uni_default_c"]), dtype=np.float32)
        d["mkey"] = mkey
        d["ch_idx"] = {chr(int(c)): i for i, c in enumerate(d["uni_cp"].tolist())}
        d["pv_pad"] = np.vstack((np.asarray(d["uni_prev"], dtype=np.float32), pad_row))
        d["nx_pad"] = np.vstack((np.asarray(d["uni_next"], dtype=np.float32), pad_row))
        d["cols"] = None if cols == list(range(len(GB_INPUTS))) else np.array(cols)
        d["uses_fsw"] = "fsw" in inputs
        _MODEL_CACHE[mkey] = d
    return _MODEL_CACHE[mkey]


def build_rare_tables(chunks, texts, bad_u8, special_ids, model):
    """从解码后的词表预构建逐 token 生僻字检测查表字典（在线检测仅按 token ID 直接索引）。"""
    vocab_size = len(chunks)
    rcls = np.zeros(vocab_size, np.int8)
    first = np.zeros(vocab_size, np.int8)
    last = np.zeros(vocab_size, np.int8)
    lead = np.zeros(vocab_size, bool)
    u8 = np.asarray(bad_u8, dtype=bool).copy()
    fcat = np.zeros(vocab_size, np.int8)
    lcat = np.zeros(vocab_size, np.int8)
    def_u = float(model["uni_default_u"])
    ch_idx, u_arr = model["ch_idx"], model["uni_u"]
    default_ch = len(model["uni_prev"])
    ch_dtype = np.uint16 if default_ch <= np.iinfo(np.uint16).max else np.int32
    fch = np.full(vocab_size, default_ch, dtype=ch_dtype)
    lch = np.full(vocab_size, default_ch, dtype=ch_dtype)
    # 额外预留第 vocab_size 行作为序列外边界哨兵（索引 -1 时读取到类别码 31）
    tf = np.zeros((vocab_size + 1, len(_TOKF_COLS)), np.float32)
    tf[:vocab_size, 4:6], tf[vocab_size, 0] = def_u, 31
    sp_code, part_code = CI["special"], CI["partial"]

    for i, (b, s, is_bad) in enumerate(zip(chunks, texts, u8.tolist())):
        lead[i] = b[:1] in (b" ", b"\n", b"\t", b"\r")
        if is_bad:
            rcls[i] = sp_code if i in special_ids else part_code
        else:
            rcls[i] = sp_code if i in special_ids else CI[_classify(s)]
        first[i], last[i] = _kind(s[:1]), _kind(s[-1:])
        fcat[i], lcat[i] = _charcat(s[:1]), _charcat(s[-1:])
        u_v, um_v, cmx, cmn, cln, crr, cns = def_u, def_u, 0.0, 0.0, 0.0, 0.0, 0.0
        # 非特殊 token：残缺 UTF-8 字节赋予高稀有度默认分 3.5；正常文本提取字符一元频率、邻接转移与编码层级
        if rcls[i] != sp_code:
            if is_bad:
                cmx = cmn = 3.5
            elif s:
                us = [float(u_arr[ch_idx[ch]]) if ch in ch_idx else def_u for ch in s]
                u_v, um_v = float(np.float32(min(us))), float(np.float32(sum(us) / len(us)))
                if s[0] in ch_idx:
                    fch[i] = ch_idx[s[0]]
                if s[-1] in ch_idx:
                    lch[i] = ch_idx[s[-1]]
                tiers = [_codec_tier(ch) for ch in s]
                cmx, cmn = float(max(tiers)), float(np.float32(sum(tiers) / len(tiers)))
                cln, crr = float(len(s)), float(cmx >= 3)
                cns = float(len({_script(ch) for ch in s} - {"space_punct"}))
        tf[i] = (rcls[i], first[i], last[i], lead[i], u_v, um_v, cmx, cmn, cln, crr, cns)
    return _TableDict(
        rcls=rcls,
        first=first,
        last=last,
        lead=lead,
        u8=u8,
        bytes=_PackedBytes(chunks),
        # 末尾补 0 使 prv/nxt 为 -1（序列起止边界外）时自动映射至类别 0（无邻接字符）
        lcat_s=np.r_[lcat, np.int8(0)],
        fcat_s=np.r_[fcat, np.int8(0)],
        fch=fch,
        lch=lch,
        tf=tf,
    )


def _gb_score(m, x_mat):
    """在堆序完全二叉树数组上向量化计算 GBDT 决策函数分值（gb_base + 各树命中的叶子权重之和）。"""
    if not len(x_mat):
        return np.zeros(0)
    if m["cols"] is not None:
        x_mat = x_mat[:, m["cols"]]
    feat, thr, cat, mask, leaf = m["gb_feat"], m["gb_thr"], m["gb_cat"].astype(bool), m["gb_mask"], m["gb_leaf"]
    n, num_trees, depth = len(x_mat), len(feat), int(m["gb_depth"])
    node = np.zeros((n, num_trees), np.int64)
    rows, cols = np.arange(n)[:, None], np.arange(num_trees)
    for _ in range(depth):
        v = x_mat[rows, feat[cols, node]]
        # 类别分裂节点：mask 第 v 位为 1 走左子树（2*node+1），为 0 走右子树（2*node+2）；数值节点：v > thr 走右子树
        go_r = ((mask[cols, node] >> np.clip(v, 0, 31).astype(np.int64)) & 1) == 0
        node = 2 * node + 1 + np.where(cat[cols, node], go_r, v > thr[cols, node])
    leaf_idx = node - ((1 << depth) - 1)
    return float(m["gb_base"]) + np.ascontiguousarray(leaf[cols, leaf_idx]).sum(1)


def _bad_utf8(tok, tables):
    """流式增量 UTF-8 校验：识别跨 token 拼合失败产生的新增 U+FFFD（合法多字节拆分与流末尾未完结字节前缀不报错）。"""
    out = np.zeros(len(tok), bool)
    u8, by = tables["u8"], tables["bytes"]
    need = u8[tok]
    if not need.any():
        return out
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for i in range(int(np.argmax(need)), len(tok)):
        t = int(tok[i])
        if u8[t] or dec.getstate()[0]:
            b = by[t]
            # 仅当增量解码输出的替换符多于该 token 自身字面已有的 U+FFFD 时，才属于字节流拼合损坏
            out[i] = dec.decode(b).count("\ufffd") > b.decode("utf-8", "ignore").count("\ufffd")
    return out


def _share(cls, isl):
    """计算每个字母 token 的类别在截至当前的全部字母 token 中的累积占比（非字母位置恒为 1.0）。"""
    fs = np.ones(len(cls))
    li = np.flatnonzero(isl)
    if len(li):
        lc = cls[li]
        if not (lc == int(lc[0])).all():
            tot = np.arange(1, len(li) + 1)
            for c in np.unique(lc).tolist():
                m = lc == c
                fs[li[m]] = np.cumsum(m)[m] / tot[m]
    return fs


def _window_share(cls, isl, idx):
    """计算指定位置 idx 处的字母类别在其前序最多 WIN (64) 个字母中的平滑占比 (cnt + 1) / (win_len + 1)。"""
    out = np.ones(len(idx))
    letter_mask = isl[idx]
    if letter_mask.any():
        seq, j = cls[isl], (np.cumsum(isl) - isl)[idx]
        lo, c = np.maximum(j - WIN, 0), cls[idx]
        for cc in np.unique(c[letter_mask]).tolist():
            cum = np.r_[0, np.cumsum(seq == cc)]
            sel = letter_mask & (c == cc)
            out[sel] = (cum[j[sel]] - cum[lo[sel]] + 1) / (j[sel] - lo[sel] + 1)
    return out


class RareCharDetector:
    """基于纯 Token ID 序列的整条记录生僻字与 UTF-8 乱码检测器。"""

    def __init__(self, tables, model, think_end=None):
        """绑定预构建词表查表字典、GBDT 打分模型及可选的思考结束 token ID。"""
        self.tables = tables
        self.model = model
        self.think_end = think_end

    def _check_tokens(self, tokens):
        """将输入转为一维 int64 数组，并校验所有 token ID 均落在词表有效区间 [0, vocab_size) 内。"""
        tok = np.asarray(tokens, np.int64)
        if tok.ndim != 1:
            raise ValueError(f"tokens must be a 1D sequence of token IDs, got shape {tok.shape}")
        vocab_size = len(self.tables["rcls"])
        if len(tok) and (int(tok.min()) < 0 or int(tok.max()) >= vocab_size):
            raise ValueError(f"token id out of range [0, {vocab_size})")
        return tok

    def record_features(self, tokens):
        """提取整条 token 序列的语种占比、增量 UTF-8 损坏标记、跨语种粘连标记及门控 GBDT 异常分值。"""
        tok = self._check_tokens(tokens)
        n, tables = len(tok), self.tables
        if not n:
            return {
                "fs": np.zeros(0),
                "cls": np.zeros(0, np.int8),
                "letter": np.zeros(0, bool),
                "bad_utf8": np.zeros(0, bool),
                "caseless_after_latin": np.zeros(0, bool),
                "answer": np.zeros(0, bool),
                "gb": np.zeros(0),
            }

        # 1) 查表计算序列级基础特征：文字类别、前缀同文字占比 fs、拉丁后无空格粘连、思考/回答阶段与增量 UTF-8 校验
        cls = tables["rcls"][tok]
        isl = _LET[cls]
        fs = _share(cls, isl)
        first, last, lead = tables["first"][tok], tables["last"][tok], tables["lead"][tok]
        prev_last = np.r_[0, last[:-1]]
        # 前一个 token 以大小写拉丁字母结尾，且当前 token 无前导空白并以无大小写文字（如汉字/假名）开头
        caseless = (first == 5) & ~lead & ((prev_last == 1) | (prev_last == 2))
        answer = np.ones(n, bool) if self.think_end is None else np.zeros(n, bool)
        if self.think_end is not None:
            hit = np.flatnonzero(tok == self.think_end)
            if len(hit):
                first_hit = int(hit[0])
                answer[first_hit:] = True
        bad_utf8 = _bad_utf8(tok, tables)
        prv, nxt = np.r_[-1, tok[:-1]], np.r_[tok[1:], -1]
        pcat, ncat = tables["lcat_s"][prv], tables["fcat_s"][nxt]

        # 2) 门控 GBDT 评估：仅对少数文字字母（fs < 0.05）或 UTF-8 损坏位置组装 21 维特征并推理，其余位置分值保持 -inf
        gb = np.full(n, -np.inf)
        gate = np.flatnonzero((fs < GB_GATE) | bad_utf8)
        if len(gate):
            t, pc, nc = tok[gate], pcat[gate], ncat[gate]
            x_mat = np.empty((len(gate), len(GB_INPUTS)))
            x_mat[:, _TOKF_COLS] = tables["tf"][t]
            no_text = x_mat[:, _COL["c_len"]] == 0.0
            x_mat[no_text, _COL["u"]] = x_mat[no_text, _COL["u_mean"]] = float(self.model["uni_default_u"])
            x_mat[:, _COL["pcls"]] = tables["tf"][prv[gate], 0]
            x_mat[:, _COL["ncls"]] = tables["tf"][nxt[gate], 0]
            x_mat[:, _COL["pcat"]], x_mat[:, _COL["ncat"]] = pc, nc
            x_mat[:, _COL["cp"]] = self.model["pv_pad"][tables["fch"][t], pc]
            x_mat[:, _COL["cn"]] = self.model["nx_pad"][tables["lch"][t], nc]
            x_mat[:, _COL["fs"]] = fs[gate]
            x_mat[:, _COL["fsw"]] = _window_share(cls, isl, gate) if self.model["uses_fsw"] else 1.0
            x_mat[:, _COL["answer"]], x_mat[:, _COL["bad_utf8"]] = answer[gate], bad_utf8[gate]
            gb[gate] = _gb_score(self.model, x_mat)
        return {
            "fs": fs,
            "cls": cls,
            "letter": isl,
            "bad_utf8": bad_utf8,
            "caseless_after_latin": caseless,
            "answer": answer,
            "gb": gb,
        }

    def detect(self, tokens, tier="intercept"):
        """按灵敏度档位（intercept / monitor）检测生僻字或乱码异常，返回 (首次触发前缀长度 fired_at, 规则名)。"""
        if tier not in ("intercept", "monitor"):
            raise ValueError(f"tier must be 'intercept' or 'monitor', got {tier!r}")
        tok = self._check_tokens(tokens)
        n = len(tok)
        if not n:
            return -1, ""
        feat = self.record_features(tok)
        # 拦截档（intercept）：仅触发确定性 UTF-8 损坏与高置信度 GBDT 字母异常
        # 第三项为右邻居依赖偏移：bad_utf8 为纯因果逐字节校验（0 -> pos+1）；GBDT 特征依赖 pos+1（1 -> min(pos+2, n)）
        checks = [
            (feat["bad_utf8"], "bad_utf8", 0),
            ((feat["gb"] > 3.85) & feat["letter"], "gb>3.85 & letter", 1),
        ]
        if tier == "monitor":
            # 监控档（monitor）：追加极低占比跨语种无空格粘连规则，并放宽 GBDT 告警阈值至 2.70；
            # 粘连规则与 GBDT 对齐同一步长时钟，保证同位置同时命中时高优先级 intercept 规则不被 monitor 规则抢占
            cand = np.flatnonzero((feat["fs"] < 0.01) & feat["caseless_after_latin"])
            fsw_ok = np.zeros(n, bool)
            if len(cand):
                fsw_ok[cand] = _window_share(feat["cls"], feat["letter"], cand) < 0.05
            ans_caseless = (feat["fs"] < 0.01) & feat["answer"] & feat["caseless_after_latin"]
            checks += [
                (fsw_ok, "fs<0.01 & fsw<0.05 & caseless_after_latin", 1),
                (ans_caseless, "fs<0.01 & answer & caseless_after_latin", 1),
                ((feat["gb"] > 2.7) & feat["letter"], "gb>2.70 & letter", 1),
            ]
        best, best_at = (-1, ""), None
        for mask, name, lookahead in checks:
            h = np.flatnonzero(mask)
            if len(h):
                fired_at = min(int(h[0]) + 1 + lookahead, n)
                if best_at is None or fired_at < best_at:
                    best_at = fired_at
                    best = fired_at, name
        return best
