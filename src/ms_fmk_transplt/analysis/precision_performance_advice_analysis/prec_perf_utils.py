#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
from collections import namedtuple, defaultdict
from typing import Optional, Dict

from utils import transplant_logger as translog


# precision & performance advice dict union
AdviceInfo = namedtuple(
    "AdviceInfo",
    [
        "api_prec_dict",
        "api_perf_dict",
        "api_params_perf_dict",
        "perf_api_suggest"
    ]
)


class PerfApiSuggest:
    """
    Process the information of suggested api. If the suggested
    api is not used, related suggestions will be proposed.

    Args:
        perf_suggest (Dict): The dict parsed from 'precision_performance_advice' json file.
    """
    def __init__(self, perf_suggest: Optional[Dict[str, Dict[str, str]]]):
        self.dependency : Dict[str, bool] = {}
        self.suggest_apis : Dict[str, bool] = {}
        self.suggest_apis_info : Dict[str, Dict[str, str]] = perf_suggest
        self.__set_dependency()

    def __set_dependency(self):
        if not self.suggest_apis_info or not isinstance(self.suggest_apis_info, dict):
            return

        for api_name, val in self.suggest_apis_info.items():
            if not isinstance(val, dict):
                warn_msg = "The data format in inner json file is not correct!"
                translog.warning(warn_msg)
                continue
            dep_api = val.get("dependency", [])
            for api in dep_api:
                self.dependency[api] = False
            self.suggest_apis[api_name] = False
