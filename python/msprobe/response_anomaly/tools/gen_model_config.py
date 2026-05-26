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

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from transformers import AutoTokenizer
from msprobe.core.common.file_utils import check_path_exists, save_json


def parse_args():
    parser=argparse.ArgumentParser(description="")
    parser.add_argument("--model-path", required=True,help="Path of the model for starting the service.")  # 目标模型路径
    parser.add_argument("--model-name",default=None,type=str) # 保存的文件名以及mtype_config.json文件里对应的key
    return parser.parse_args()


@dataclass
class TokenInfo:
    token_id:int
    token_raw:str
    token_decoded:str
    category:str
    printable_fraction:float
    entropy:float


SCRIPT_LABELS = {
    "cjk":"chinese_cjk",
    "hiragana":"japanese_hiragana",
    "katakana":"japanese_katakana",
    "hangul":"korean_hangul",
    "thai":"thai",
    "greek":"greek",
    "variation_selector":"variation_selector",
    "latin":"english_latin",
    "latin_space":"english_latin_space",
    "digit":"numbers",
    "emoji":"emoji",
    "whitespace":"whitespace",
    "punct":"punctuation",
    "symbol":"symbol",
    "control":"control",
    "arabic":"arabic",
    "cyrillic":"cyrillic",
    "devanagari":"devanagari",
    "math_letter":"mathematics",
    "modifier_letter":"mathematics",
    "fraction":"mathematics"
}

PUNCT_CHARS=set("'`\".;:!?-–—()[]{}<>/\\@#*$%&+|~^=_")
WHITESPACE_CHARS=set(" \t\n\r\f\v▁ĠĊ█")


@lru_cache(maxsize=4096)
def _classify_char(ch):
    if ch in WHITESPACE_CHARS:
        return "whitespace"
    codepoint=ord(ch)
    if 0x1F300 <= codepoint <= 0x1FAFF:
        return "emoji"
    if ch.isdigit():
        return "digit"
    name=unicodedata.name(ch,"")
    if not name:
        category=unicodedata.category(ch)
        if category.startswith("C"):
            return "control"
        if category.startswith("P"):
            return "punct"
        if category.startswith("S"):
            return "symbol"
        return "other"
    name_upper=name.upper()
    if "PLANCK CONSTANT" in name_upper or "NATHEMATICAL" in name_upper or "DUBLE-STRUCK CAPITAL" in name_upper:
        return "math_letter"
    if "MODIFIER LETTER" in name_upper:
        return "modifier_letter"
    if "SPACE" in name_upper or unicodedata.category(ch) in {"Zs","Zl","Zp"}:
        return "whitespace"
    if "CJK UNIFIED IDEOGRAPH" in name_upper or "CJK COMPATIBILITY" in name_upper:
        return "cjk"
    if "HIRAGANA" in name_upper:
        return "hiragana"
    if "HANGUL" in name_upper:
        return "hangul"
    if "THAI" in name_upper:
        return "thai"
    if "ARABIC" in name_upper:
        return "arabic"
    if "CYRILLIC" in name_upper:
        return "cyrillic"
    if "DEVANAGARI" in name_upper:
        return "devanagari"
    if "LATIN" in name_upper:
        return "latin"
    if "VARIATION SELECTOR" in name_upper:
        return "variation_selector"
    category=unicodedata.category(ch)
    if category.startswith("P"):
        return "punct"
    if category.startswith("S"):
        return "symbol"
    if category.startswith("C"):
        return "control"
    return "other"
    
def categorize_token(token_id,token_raw,decoded):
    char_counts=Counter()
    printable=0
    for char in decoded:
        char_class=_classify_char(char)
        char_counts[char_class]+=1
        if char_class not in {"control"}:
            printable+=1

    total_chars=sum(char_counts.values()) or 1
    printable_fraction=printable/total_chars
    dominant,dom_count=(char_counts.most_common(1)[0] if char_counts else ("other",0))
    dominant_ratio=dom_count/total_chars

    label=SCRIPT_LABELS.get(dominant,"other")
    if dominant == "latin" and printable_fraction >0.8:
        if "whitespace" in char_counts:
            label="english_latin_space"
        else:
            label="english_latin"
    elif dominant=="digit" and dominant_ratio>0.6:
        label="numbers"
    elif dominant=="punct" and dominant_ratio>0.7:
        label="punctuation"
    elif dominant=="symbol" and dominant_ratio>0.6:
        label="symbol_cluster"
    elif dominant=="control":
        label="control_bytes" 
    elif dominant in {"cjk","hiragana","katakana","hangul","thai","greek","variation_selector"}:
        label=SCRIPT_LABELS[dominant]
    elif dominant=="whitespace" and printable<0.4:
        label="whitespace"
    elif dominant_ratio<0.5 and printable_fraction<0.7:
        label="mixed_noise" 
    
    if label not in {"punctuation","symbol_cluster","numbers"} and dominant not in {"latin","cjk","hiragana","katakana","hangul","thai"}:
        dense_symbol_ratio=(char_counts.get("symbol",0)+char_counts.get("punct",0))/total_chars
        if dense_symbol_ratio>0.6 and total_chars>=3:
            label="gibberish_symbols"
    entropy=1.0  # 时间问题，没有写该函数
    return TokenInfo(
        token_id=token_id,
        token_raw=token_raw,
        token_decoded=decoded,
        category=label,
        printable_fraction=round(printable_fraction,4),
        entropy=round(entropy,4),
    )

def invert_vocab(vocab):
    size=max(vocab.values())+1
    tokens=["" for _ in range(size)]
    for token,idx in vocab.items():
        if idx<size:
            tokens[idx]=token
    return tokens

def _normalize_name(name):
    return "-".join(re.split(r"\.|-|_", name.lower()))

def read_tokenid(path):
    eos_token_id,bos_token_id = None, None
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = json.load(f)
            eos_token_id = data.get("eos_token_id")
            bos_token_id = data.get("bos_token_id")
    return eos_token_id, bos_token_id

def parase_eos_token(tokenizer,model_name,model_path):
    generation_config_path = os.path.join(model_path,"generation_config.json")
    eos_token_id,bos_token_id = read_tokenid(generation_config_path)
    if eos_token_id is None:
        config_path = os.path.join(model_path,"config.json")
        eos_token_id, bos_token_id = read_tokenid(config_path)
    
    if eos_token_id is None:
        eos_token_id = [tokenizer.eos_token_id]
    
    result = {
        model_name:{
            "bos":bos_token_id,
            "eos":eos_token_id
        }
    }

    path = Path(__file__).resolve().parent.parent
    save_path = os.path.join(path,"configs/")
    check_path_exists(save_path)
    save_json(os.path.join(save_path, "mtype_config.json"), result)


def main():
    args=parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    # 处理model_name
    model_name = args.model_name
    if model_name is None:
        model_name = model_path.rstrip(os.sep).split(os.sep)[-1]
    model_name = _normalize_name(model_name)

    # 获取eos、bos token_id，并将其保存到mtype_config.json文件里
    parase_eos_token(tokenizer, model_name, model_path)

    vocab_size = tokenizer.vocab_size  # 获取到vocab_size
    # 生成 token 到 category 的映射文件, 命名规则为 (model_name+vocab_size).json
    vocab=tokenizer.get_vocab()
    tokens=invert_vocab(vocab)
    category_counts=Counter()

    decode=tokenizer.backend_tokenizer.decoder.decode
    tokens_info=[]
    for idx,token in enumerate(tokens):
        decoded=decode([token])
        info=categorize_token(idx,token,decoded)
        tokens_info.append(info)
        category_counts[info.category]+=1

    path = Path(__file__).resolve().parent.parent
    save_path = os.path.join(path,"token2category/")
    check_path_exists(save_path)

    name = model_name + f"_{vocab_size}"  # 保存的文件名
    id_to_category = {info.token_id:info.category for info in tokens_info}
    save_json(os.path.join(save_path,f"{name}.json"), id_to_category)


if __name__=="__main__":
    main()