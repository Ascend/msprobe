# coding=utf-8
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

import numpy as np


def get_dot_value(dot_field):
    if dot_field & 0x60 == 0x60: 
        return 4
    if dot_field & 0x60 == 0x40: 
        return 3
    if dot_field & 0x60 == 0x20: 
        return 2
    if dot_field & 0x70 == 0x10: 
        return 1
    if dot_field & 0x78 == 0x8: 
        return 0
    if dot_field & 0x78 == 0x00: 
        return -1
    return None


def hifloat8_to_float32(bits):
    sign = (bits >> 7) & 0x1
    dot_field = bits & 0x7F

    dot_value = get_dot_value(dot_field)
    if dot_value is None: 
        return np.nan
    
    if bits == 0x00:  # 零（不区分正零和负零）
        return 0.0
    elif bits == 0x6F or bits == 0xEF:  # Inf（正 Inf 和负 Inf）
        return np.inf
    elif bits == 0x80:  # NaN（10000000₂）
        return np.nan

    if dot_value == -1:
        mantissa = int(bits & 0x7)
        value = (-1.0 if sign else 1.0) * 2**(mantissa - 23) * 1.0
    else: 
        exp_bits = dot_value
        mant_bits = 5 - dot_value  
        if exp_bits == 0:  
            exp = 0
        else:
            exp_mask = (1 << exp_bits) - 1
            exp_raw = (dot_field >> mant_bits) & exp_mask
            exp_sign = (exp_raw >> (exp_bits - 1)) & 0x1
            exp_mag = exp_raw & ((1 << (exp_bits - 1)) - 1)
            exp = (-1.0 if exp_sign else 1.0) * ((1 << (exp_bits - 1)) + exp_mag)  

        mant_mask = (1 << mant_bits) - 1
        mant = (dot_field & mant_mask) / (1 << mant_bits) + 1.0 
        value = (-1.0 if sign else 1.0) * 2**exp * mant  

    return value


def float8e4m3fn_to_float32(bits):
    sign = (bits >> 7) & 0x1
    exp = int((bits >> 3) & 0xF)
    mantissa = bits & 0x7
    if exp == 0:
        if mantissa == 0:
            return -0.0 if sign else 0.0
        value = (-1.0 if sign else 1.0) * (mantissa / 8.0) * 2**(-6)
    else:
        value = (-1.0 if sign else 1.0) * (1.0 + mantissa / 8.0) * 2**(exp - 7)
    return value


def float8e5m2_to_float32(bits):
    sign = (bits >> 7) & 0x1
    exp = int((bits >> 2) & 0x1F)
    mantissa = bits & 0x3
    if exp == 0x1F and mantissa != 0:
        return np.nan
    elif exp == 0x1F and mantissa == 0:
        return np.inf
    elif exp == 0:
        value = (-1.0 if sign else 1.0) * (mantissa / 4.0) * 2**(-14)
    else:
        value = (-1.0 if sign else 1.0) * (1.0 + mantissa / 4.0) * 2**(exp - 15)
    return value