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

import os
import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
from msprobe.core.common.file_utils import load_yaml, load_json, check_path_exists, listdir_path, check_file_exist


# 单个请求的检测结果
@dataclass
class DetectionResult:
    is_ill: bool = False
    ill_type: int = int

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
    '''从 model_config 中解析出模型名, 与 know_models 做模糊匹配
    
    know_models 来自 mtype_config.json 的所有 key
    匹配规则: 忽略大小写、去掉所有分隔符后比较是否包含
    '''
    if model_config is None:
        return None
    if isinstance(model_config, str):
        raw_name = model_config.lower()
    elif isinstance(model_config, dict):
        raw_name = model_config.get("model_name","").lower()
    else:
        return None
    
    clean_raw = re.sub(r"[-_.\s]", "", raw_name)

    for known in know_models:
        clean_known = re.sub(r"[-_.\s]", "", known.lower())
        if clean_known in clean_raw or clean_raw in clean_known:
            return known  # 返回 mtype_config.json 里的原始 key 名
    return None
    

class ILLDetector():
    """推理异常检测器"""

    def __init__(self, 
                 config_path: str = "./configs/config.yaml", 
                 mtype_path: str = "./configs/mtype_config.json",
                 tk2cat_path: str = "./token2category/") -> None:
        
        # 加载配置文件
        config_data = load_yaml(config_path)
        self.mtype2token = load_json(mtype_path)
    
        # token2category 
        check_path_exists(tk2cat_path)  # 检查路径是否存在, 如果不存在, 报错!
        self._init_tk2cat(tk2cat_path)

        # 检测算法配置参数 
        self.window_size: int = config_data["window_size"]
        self.stride: int = config_data["stride"]
        self.topk: Optional[int] = None

        _rare = config_data["rare_character"]
        self.rare_explogp_sum_thresh: int = _rare["explogp_sum_thresh"]
        self.rare_cat_thresh: int = _rare["category_thresh"]

        _garbled = config_data["garbled"]
        self.garbled_top1_logp_thresh : float =  _garbled["top1_logp_thresh"]
        self.garbled_window_ratio : float =  _garbled["window_ratio"]
        self.garbled_window_thresh : int =  _garbled["window_thresh"]

        _repet = config_data["repetition"]
        _traj = _repet["trajectory"]
        self.repet_n: int =  _traj["n"]
        self.repet_distinct_n_thresh: float =  _traj["distinct_n_thresh"]
        self.repet_logp_thresh: float =  _traj["logp_thresh"]

        _acf = _repet["acf"]
        self.w_std_threshold: float = 1e-12 # w_std_threshold: 1e-12  
        self.acf_threshold: float =  _acf["acf_threshold"]
        self.acf_harmonic_threshold: float =  self.acf_threshold // 2  # 2倍/3倍处的谐波阈值，acf_threshold//2
        self.acf_logp_thresh: float =  _acf["logp_thresh"]
        self.acf_min_period: int =  3
        self.acf_max_period: int =  self.window_size // 3
        self.linalg_logp_thresh: float =  0.9

        self.single_window_thresh: int = _repet["single_window_thresh"]
        self.multi_window_thresh: int = _repet["multi_window_thresh"]
        
        # 符合乱码条件的窗口计数
        self._garbled_count: int = 0  


    def _init_tk2cat(self, tk2cat_path: str) -> None:
        file_names = listdir_path(tk2cat_path)

        # 模型类型与vocab_size的对应
        self.mtype2vocab: Dict[str, int] = {
            val.split('_')[0]:int(val.split('_')[1].split('.')[0]) 
            for val in file_names
        }  
        self.tk2cat_path: str = tk2cat_path

    # 滑窗
    def sliding_window(self, seq: List) -> Generator[Tuple[int, List], None, None]:
        '''
        生成滑窗
        return:
            i: 窗口序列的起始索引
            window_seq: 窗口序列
        '''
        for i in range(0, len(seq), self.stride):
            yield i, seq[i: min(i + self.window_size, len(seq))]

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
        
        eos_candidates = token_map.get('eos', [])
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
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

    def get_distinct_n(self, tokens: List[int]) -> float:
        all_grams = self.get_ngrams(tokens)
        if not all_grams:
            return 1.0
        return len(set(all_grams)) / len(all_grams)

    # 乱码检测
    def _detect_garbled(self,
                        window_topk_logprobs: List[Dict[int, float]],
                        tk2cat: Dict[str, int],
                        vocab_size: int,
                        ) -> bool:
        
        seq_len = len(window_topk_logprobs)  # 当前序列长度

        # 1) 如果有词表信息,使用生僻字的检测方法
        if tk2cat is not None:
            flag, rare_character_count = self._detect_rare_character(window_topk_logprobs, tk2cat, vocab_size)
            if flag and rare_character_count / seq_len > self.garbled_window_ratio:
                return True
            return False
        
        # 2) 如果没有词表信息
        window_logprobs = np.array([
                list(item.values())[:self.topk] 
                for item in window_topk_logprobs
            ]) # top1 logprob

        dims = np.where(np.exp(window_logprobs).sum(-1) < self.rare_explogp_sum_thresh)[0]  # 找出当前window里topk概率总和小于阈值的dims
        
        if dims.size == 0:
            return False
        
        if len(dims) / seq_len <= self.garbled_window_ratio:
            return False
        
        log_dims = np.where(window_logprobs.max(-1) < self.garbled_top1_logp_thresh)[0]
        return len(log_dims) / seq_len > self.garbled_window_ratio

    # 维护_garbled_count 状态, 返回是否触发乱码警告
    def _update_garbled_state(self, garbled_flag: bool) -> bool:
        if garbled_flag:
            self._garbled_count += 1
        return self._garbled_count > self.garbled_window_thresh
    
    # 生僻字检测
    def _detect_rare_character(self,
                        window_topk_logprobs: List[Dict[int, float]],
                        tk2cat: Dict[str, int],
                        vocab_size: int,
                        ) -> Tuple[bool, int]:
        
        # 当tk2cat为None时，不处理生僻字
        if tk2cat is None:
            return False, 0 
        
        window_logprobs = np.array([list(item.values())[:self.topk] for item in window_topk_logprobs])

        dims = np.where(np.exp(window_logprobs).sum(-1) < self.rare_explogp_sum_thresh)[0]  # 找出当前window里topk概率总和小于阈值的dims
        
        if dims.size == 0:
            return False, 0 
        
        cat_hit_count = 0
        for dim in dims:
            window_topk_logprob = {key:val for key,val in sorted(window_topk_logprobs[dim].items(), key = lambda x: x[1], reverse = True)[:self.topk]}
            categories = set(tk2cat[str(item)] for item in list(window_topk_logprob.keys()) if item <= vocab_size)  # 统计topk的token-id对应的类别
            
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

        w_norm = (window_logprobs- window_logprobs.mean()) / w_std
        f = np.fft.rfft(w_norm, n = 2*self.window_size)[:self.window_size]
        acf = np.fft.irfft(f * np.conj(f)[:self.window_size])
        acf /= acf[0]

        acf_segment = acf[self.acf_min_period : self.acf_max_period + 1]
        if len(acf_segment) == 0:
            return False
        
        peak_lag = np.argmax(acf_segment) + self.acf_min_period
        peak_val = acf_segment[peak_lag - self.acf_min_period]
        if peak_val <= self.acf_threshold:
            return False
        
        harmonic_confirmed = any(
            h * peak_lag < self.window_size and acf[h * peak_lag] > self.acf_harmonic_threshold
            for h in [2,3]
        )
        if not harmonic_confirmed or np.linalg.norm(acf[1:] - acf[:-1]) <= self.linalg_logp_thresh:
            return False
        
        return np.min(window_logprobs) > self.acf_logp_thresh

    # 【单窗口-总体判定】acf+轨迹检测
    def _detect_repetitions(self, 
                            window_logprobs: np.ndarray,
                            window_tokens: List[int],
                            count_res: RepetitionCounters) -> Tuple[bool, RepetitionCounters]:
        # acf
        res_acf = self._acf_detector(window_logprobs)
        # 轨迹检测
        res_traj =  self._trajectory_detector(window_logprobs, window_tokens)

        # 总体判定
        if res_acf and res_traj:
            count_res.both += 1
        elif res_acf:
            count_res.acf_only += 1
        elif res_traj:
            count_res.traj_only += 1
        
        if (count_res.both > self.multi_window_thresh 
            or count_res.acf_only > self.single_window_thresh 
            or count_res.traj_only > self.single_window_thresh
        ):
            return True, count_res
        return False, count_res

    # 主流程
    def detector(self, 
                 topk_logprobs: List[Dict[int, float]],
                 tokens: List[int],
                 model_config: Any = None) -> DetectionResult:
        '''
        单个请求的检测入口

        Return:
            DetectionResult 包含 is_ill 和 ill_type
        '''

        # 当 topk_logprobs 或 tokens 为空时，直接返回
        if not (topk_logprobs and tokens):
            return DetectionResult()
        
        tk2cat, vocab_size = self.get_tk2cat(tokens[-1], model_config) # 获取token ids to cagetory

        self.topk = min([len(logp) for logp in topk_logprobs]) if self.topk is None else self.topk
        logprobs = np.array([max(item.values()) for item in topk_logprobs])

        if len(tokens) < self.stride and tk2cat is not None:  # 只检测生僻字
            rare_flag, _ = self._detect_rare_character(topk_logprobs, tk2cat, vocab_size)
            if rare_flag:
                return DetectionResult(is_ill=True, ill_type=1)
            return DetectionResult()

        # 初始化状态
        self._garbled_count = 0
        rare_flag = 0
        repet_counters = RepetitionCounters()

        # 滑窗检测
        for start, window_logprobs in self.sliding_window(logprobs):
            end = start + len(window_logprobs)
            window_tokens = tokens[start:end]
            window_topk_logprobs = topk_logprobs[start:end]

            # 1) 生僻字
            rare_in_window, _ = self._detect_rare_character(window_topk_logprobs, tk2cat, vocab_size)
            rare_flag = rare_in_window or rare_flag # 即使检出生僻字，也不停止
            
            # 2） 乱码
            garbled = self._detect_garbled(window_topk_logprobs, tk2cat, vocab_size)
            if self._update_garbled_state(garbled):
                return DetectionResult(is_ill=True, ill_type=2)
            
            # 最后一个不完整的窗口小于阈值时跳过重复检测
            if len(window_logprobs) < self.stride:
                continue

            # 3) 重复检测
            flag_repet, repet_counters = self._detect_repetitions(window_logprobs, window_tokens, repet_counters)
            if flag_repet:
                return DetectionResult(is_ill=True, ill_type=3)
        
        # 当检查到最后，除生僻字外无其他异常，则返回生僻字异常
        if rare_flag:
            return DetectionResult(is_ill=True, ill_type=1)
        return DetectionResult()


    # 批量处理多个请求
    def run(self,
            topk_logprobs: List[List[Dict[int, float]]],
            tokens: List[List[int]],
            model_configs: Any = None) -> List[List]:
        '''
        return:
            二维列表, 多请求下输出结果, 如:[[False,0],[True,1]] 表示第1个请求推理正常, 第2个请求推理异常,存在生僻字
        '''
        # assert len(topk_logprobs) == len(tokens) , (
        #     "topk_logprobs and tokens must have the same length!"
        #     )
        nums = len(tokens)
        if model_configs is None:
            model_configs = [None]*nums
        results = [
            self.detector(topk_logprobs[i], tokens[i], model_configs[i]) 
            for i in range(nums)
        ]
        return [[res.is_ill, res.ill_type] for res in results]
    

