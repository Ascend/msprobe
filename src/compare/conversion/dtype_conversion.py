# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import numpy as np


def hifloat8_to_float32(bits):
    sign = (bits >> 7) & 0x1
    dot_field = bits & 0x7F
    
    if dot_field & 0x60 == 0x60:  # 2 位：11₂ (4)
        dot_value = 4
    elif dot_field & 0x60 == 0x40:  # 2 位：10₂ (3)
        dot_value = 3
    elif dot_field & 0x60 == 0x20:  # 2 位：01₂ (2)
        dot_value = 2
    elif dot_field & 0x70 == 0x10:  # 3 位：001₂ (1)
        dot_value = 1
    elif dot_field & 0x78 == 0x8:  # 4 位：0001₂ (0)
        dot_value = 0
    elif dot_field & 0x78 == 0x00:  # 4 位：0000₂ (DML)
        dot_value = -1
    else: 
        return np.nan

    if bits == 0x00:  # 零（不区分正零和负零）
        return 0.0
    elif bits == 0x6F or bits == 0xEF:  # Inf（正 Inf 和负 Inf）
        return np.inf
    elif bits == 0x80:  # NaN（10000000₂）
        return np.nan

    if dot_value == -1:
        mantissa = (bits & 0x7)  
        value = (-1)**sign * 2**(mantissa - 23) * 1.0
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
            exp = (-1)**exp_sign * ((1 << (exp_bits - 1)) + exp_mag)  

        mant_mask = (1 << mant_bits) - 1
        mant = (dot_field & mant_mask) / (1 << mant_bits) + 1.0 
        value = (-1)**sign * 2**exp * mant  

    return value


def float8e4m3fn_to_float32(bits):
    sign = (bits >> 7) & 0x1
    exp = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    if exp == 0:
        if mantissa == 0:
            return -0.0 if sign else 0.0
        value = (-1)**sign * (mantissa / 8.0) * 2**(-6)
    else:
        value = (-1)**sign * (1.0 + mantissa / 8.0) * 2**(exp - 7)
    return value


def float8e5m2_to_float32(bits):
    sign = (bits >> 7) & 0x1
    exp = (bits >> 2) & 0x1F
    mantissa = bits & 0x3
    if exp == 0x1F and mantissa != 0:
        return np.nan
    elif exp == 0x1F and mantissa == 0:
        return np.inf
    elif exp == 0:
        value = (-1)**sign * (mantissa / 4.0) * 2**(-14)
    else:
        value = (-1)**sign * (1.0 + mantissa / 4.0) * 2**(exp - 15)
    return value