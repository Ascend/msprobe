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


class CodedException(Exception):
    err_strs = {}

    def __init__(self, code, error_info=""):
        super().__init__()
        self.code = code
        self.error_info = self.err_strs.get(code) + error_info

    def __str__(self):
        return self.error_info


class MsprobeException(CodedException):
    INVALID_PARAM_ERROR = 0
    OVERFLOW_NUMS_ERROR = 1
    RECURSION_LIMIT_ERROR = 2
    INTERFACE_USAGE_ERROR = 3
    UNSUPPORTED_TYPE_ERROR = 4
    INVALID_CHAR_ERROR = 5

    err_strs = {
        INVALID_PARAM_ERROR: "[msprobe] 无效参数：",
        OVERFLOW_NUMS_ERROR: "[msprobe] 超过预设溢出次数 当前溢出次数：",
        RECURSION_LIMIT_ERROR: "[msprobe] 递归调用超过限制：",
        INTERFACE_USAGE_ERROR: "[msprobe] Invalid interface usage: ",
        UNSUPPORTED_TYPE_ERROR: "[msprobe] Unsupported type: ",
        INVALID_CHAR_ERROR: "[msprobe] 存在非法字符：",
    }


class FileCheckException(CodedException):
    INVALID_FILE_ERROR = 0
    FILE_PERMISSION_ERROR = 1
    SOFT_LINK_ERROR = 2
    ILLEGAL_PATH_ERROR = 3
    ILLEGAL_PARAM_ERROR = 4
    FILE_TOO_LARGE_ERROR = 5

    err_strs = {
        SOFT_LINK_ERROR: "[msprobe] 检测到软链接： ",
        FILE_PERMISSION_ERROR: "[msprobe] 文件权限错误： ",
        INVALID_FILE_ERROR: "[msprobe] 无效文件： ",
        ILLEGAL_PATH_ERROR: "[msprobe] 非法文件路径： ",
        ILLEGAL_PARAM_ERROR: "[msprobe] 非法打开方式： ",
        FILE_TOO_LARGE_ERROR: "[msprobe] 文件过大： ",
    }


class ParseJsonException(CodedException):
    UnexpectedNameStruct = 0
    InvalidDumpJson = 1
    err_strs = {
        UnexpectedNameStruct: "[msprobe] Unexpected name in json: ",
        InvalidDumpJson: "[msprobe] Invalid dump.json format: ",
    }


class ScopeException(CodedException):
    InvalidApiStr = 0
    InvalidScope = 1
    ArgConflict = 2
    err_strs = {
        InvalidApiStr: "[msprobe] Invalid api_list: ",
        InvalidScope: "[msprobe] Invalid scope: ",
        ArgConflict: "[msprobe] Scope and api_list conflict: ",
    }


class RepairException(CodedException):
    InvalidRepairType = 0
    err_strs = {InvalidRepairType: "[msprobe] Invalid repair_type: "}


class StepException(CodedException):
    InvalidPostProcess = 0
    err_strs = {
        InvalidPostProcess: "[msprobe] 错误的step后处理配置: ",
    }


class FreeBenchmarkException(CodedException):
    UnsupportedType = 0
    InvalidGrad = 1
    InvalidPerturbedOutput = 2
    OutputIndexError = 3
    err_strs = {
        UnsupportedType: "[msprobe] Free benchmark get unsupported type: ",
        InvalidGrad: "[msprobe] Free benchmark gradient invalid: ",
        InvalidPerturbedOutput: "[msprobe] Free benchmark invalid perturbed output: ",
        OutputIndexError: "[msprobe] Free benchmark output index out of bounds: ",
    }


class DistributedNotInitializedError(Exception):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg


class ApiAccuracyCheckerException(CodedException):
    ParseJsonFailed = 0
    UnsupportType = 1
    WrongValue = 2
    ApiWrong = 3
    err_strs = {
        ParseJsonFailed: "[msprobe] Api Accuracy Checker parse json failed: ",
        UnsupportType: "[msprobe] Api Accuracy Checker get unsupported type: ",
        WrongValue: "[msprobe] Api Accuracy Checker get wrong value: ",
        ApiWrong: "[msprobe] Api Accuracy Checker something wrong with api: ",
    }
