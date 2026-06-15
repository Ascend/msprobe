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
