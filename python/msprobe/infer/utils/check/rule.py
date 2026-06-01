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

from typing import Any

from msprobe.infer.utils.check import NumberChecker, ObjectChecker, StringChecker, DictChecker, PathChecker, ListChecker
from msprobe.infer.utils.check.checker import Checker
from msprobe.infer.utils.constants import INPUT_FILE_MAX_SIZE


class NumConverter:
    def __init__(self, convert_type=float):
        self.convert_type = convert_type

    def __call__(self, value) -> Any:
        return self.convert(value)

    def convert(self, value: str):
        try:
            if self.convert_type == int:  # noqa: E721
                return int(value), True, ""
            elif self.convert_type == float:  # noqa: E721
                return float(value), True, ""
            else:
                return value, True, ""
        except ValueError as er:
            return value, False, str(er)


class Rule:
    @staticmethod
    def none() -> Checker:
        return Checker().is_none()

    @staticmethod
    def num() -> NumberChecker:
        return NumberChecker().is_number()

    @staticmethod
    def str() -> StringChecker:
        return StringChecker().is_str()

    @staticmethod
    def dict() -> DictChecker:
        return DictChecker().is_dict()

    @staticmethod
    def obj(obj_type) -> ObjectChecker:
        return ObjectChecker().is_type(obj_type)

    @staticmethod
    def path() -> PathChecker:
        return PathChecker()

    @staticmethod
    def list() -> ListChecker:
        return ListChecker().is_list()

    @staticmethod
    def config_file() -> PathChecker:
        # pylint: disable=no-member
        return (
            PathChecker().exists().is_file().is_readable().is_safe_parent_dir().max_size(10 * 1000 * 1000).as_default()
        )

    @staticmethod
    def input_file() -> PathChecker:
        # pylint: disable=no-member
        return (
            PathChecker()
            .exists()
            .is_file()
            .is_readable()
            .is_safe_parent_dir()
            .max_size(INPUT_FILE_MAX_SIZE)
            .as_default()
        )

    @staticmethod
    def input_dir() -> PathChecker:
        # pylint: disable=no-member
        return PathChecker().exists().is_dir().is_readable().as_default()

    @staticmethod
    def output_dir() -> PathChecker:
        # pylint: disable=no-member
        return Rule.path().any(Rule.anti(PathChecker().exists()), PathChecker().is_dir().is_writeable()).as_default()

    @staticmethod
    def any(*rules: Checker) -> Checker:
        return Checker().any(*rules)

    @staticmethod
    def anti(rule: Checker) -> Checker:
        return Checker().anti(rule)

    @staticmethod
    def to_int() -> NumberChecker:
        return NumberChecker(converter=NumConverter(int))

    @staticmethod
    def to_float() -> NumberChecker:
        return NumberChecker(converter=NumConverter(float))
