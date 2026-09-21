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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

from msprobe.core.common.output_postprocess.processor import get_valid_len_from_group_key, clean_outputs


def postprocess_by_group_index(api_name: str, output, _args, kwargs: Dict[str, Any]):
    valid_len = get_valid_len_from_group_key(api_name, "group_index", kwargs)
    if valid_len is None:
        return output
    return clean_outputs(output, valid_len)


def postprocess_by_group_list(api_name: str, output, _args, kwargs: Dict[str, Any]):
    valid_len = get_valid_len_from_group_key(api_name, "group_list", kwargs)
    if valid_len is None:
        return output
    return clean_outputs(output, valid_len)


def extract_valid_len_by_group_index(api_name: str, _args, kwargs: Dict[str, Any]):
    return get_valid_len_from_group_key(api_name, "group_index", kwargs)


def extract_valid_len_by_group_list(api_name: str, _args, kwargs: Dict[str, Any]):
    return get_valid_len_from_group_key(api_name, "group_list", kwargs)
