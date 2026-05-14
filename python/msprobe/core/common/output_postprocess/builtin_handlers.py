#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

from msprobe.core.common.output_postprocess import processor


def postprocess_by_group_index(api_name: str, output, _args, kwargs: Dict[str, Any]):
    return processor._clean_by_group_key(api_name, "group_index", output, kwargs)


def postprocess_by_group_list(api_name: str, output, _args, kwargs: Dict[str, Any]):
    return processor._clean_by_group_key(api_name, "group_list", output, kwargs)


def extract_valid_len_by_group_index(api_name: str, _args, kwargs: Dict[str, Any]):
    return processor._extract_valid_len_by_group_key(api_name, "group_index", kwargs)


def extract_valid_len_by_group_list(api_name: str, _args, kwargs: Dict[str, Any]):
    return processor._extract_valid_len_by_group_key(api_name, "group_list", kwargs)