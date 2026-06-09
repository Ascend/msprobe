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
from msprobe.response_anomaly.detector import ILLDetector


def analyze_output_anomaly(topk_logprobs, tokens, model_configs):
    path = os.path.dirname(os.path.realpath(__file__))
    # 检测算法各阈值配置, 用户可修改配置文件
    config_path = os.path.join(path, "configs/config.yaml")

    # model对应的model_name和eos、bos的token_id, 如果当前文件里没有用户提供的模型，用户可手动添加上去，以达到更优检测效果
    mtype_path = os.path.join(path, "configs/mtype_config.json")

    # 提前存储的token to category文件，用于生僻字和乱码的检测, 如果当前文件里没有用户提供的模型，用户可手动添加上去，以达到更优检测效果
    tk2cat_path = os.path.join(path, "token2category/")

    # 初始化检测类
    detector = ILLDetector(config_path, mtype_path, tk2cat_path)
    return detector.run(topk_logprobs, tokens, model_configs)  # 执行检测算法，返回结果
