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
from enum import Enum, auto, unique
from typing import Union

from msprobe.infer.utils.check.checker import Checker, CheckResult, rule, EnumInstance


@unique
class FileType(Enum):
    DIRECTORY = auto()
    CHARACTER = auto()
    BLOCK = auto()
    FILE = auto()
    FIFO = auto()
    SYMLINK = auto()
    SOCKET = auto()


class FileStatus:
    def __init__(self, file_name: str) -> None:
        file_status = os.lstat(file_name)
        self._file_name = file_name
        self.status_mode = file_status.st_mode
        self._file_uid = file_status.st_uid
        self._file_gid = file_status.st_gid
        self._file_size = file_status.st_size
        self._file_extension = os.path.splitext(file_name)[1]

    @property
    def file_name(self) -> int:
        return self._file_name

    @property
    def size(self) -> int:
        return self._file_size

    @property
    def perm_bits(self) -> int:
        return os.st.S_IMODE(self.status_mode)

    @property
    def uid(self) -> int:
        return self._file_uid

    @property
    def gid(self) -> int:
        return self._file_gid

    @property
    def extension(self) -> str:
        return self._file_extension

    @property
    def ftype(self):
        file_type_map = {
            os.st.S_IFDIR: FileType.DIRECTORY,
            os.st.S_IFCHR: FileType.CHARACTER,
            os.st.S_IFBLK: FileType.BLOCK,
            os.st.S_IFREG: FileType.FILE,
            os.st.S_IFIFO: FileType.FIFO,
            os.st.S_IFLNK: FileType.SYMLINK,
            os.st.S_IFSOCK: FileType.SOCKET,
        }

        file_mode = os.st.S_IFMT(self.status_mode)
        return file_type_map.get(file_mode, None)


class PathChecker(Checker):
    def __init__(self, instance=EnumInstance.NO_INSTANCE, converter=None):
        super().__init__(instance, converter)
        self.f_status = None
        self.f_state = False
        self.converter = converter or self.path_converter
        self.status_err_msg = None

    def path_converter(self, ori_path):
        ori_path = os.path.realpath(ori_path)
        try:
            self.f_status = FileStatus(ori_path)
        except OSError as e:
            self.status_err_msg = e.strerror + ': ' + e.filename
        except TypeError:
            self.status_err_msg = f'TypeError: {ori_path}'
        except Exception as e:
            self.status_err_msg = str(e)
        else:
            self.f_state = True

        return ori_path, True, self.status_err_msg

    @rule()
    def exists(self) -> Union["PathChecker", CheckResult]:
        err_msg = f"No such file or directory: {self.instance}. {self.status_err_msg}"
        return self.f_state, err_msg

    @rule()
    def is_file(self) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return self.f_status.ftype is FileType.FILE, f"Not a file: {self.instance}"

    @rule()
    def is_dir(self) -> Union["PathChecker", CheckResult]:
        if not self.f_state:
            return False, self.status_err_msg
        else:
            return self.f_status.ftype is FileType.DIRECTORY, f"Not a directory: {self.instance}"

    @rule()
    def is_readable(self) -> Union["PathChecker", CheckResult]:
        return os.access(self.instance, os.R_OK), self.instance + " is not readable"

    @rule()
    def is_writeable(self) -> Union["PathChecker", CheckResult]:
        return os.access(self.instance, os.W_OK), self.instance + " is not writable"

    @rule()
    def is_executable(self) -> Union["PathChecker", CheckResult]:
        return os.access(self.instance, os.X_OK), self.instance + " is not executable"

    @rule()
    def max_perm(self, perm_bits: int) -> Union["PathChecker", CheckResult]:
        if 0o777 < perm_bits or perm_bits < 0:
            msg = "Permission bits should be in range from 0 to 0o777"
            raise ValueError(f"{msg}")

        part_mapping = ["others", "groups", "users"]
        perm_mapping = ["executing", "writing", "reading"]

        for count in range(9):
            mask = 1 << count

            if (self.f_status.perm_bits & mask) and not (perm_bits & mask):
                err_msg = (
                    f"{part_mapping[count // 3]} should not have {perm_mapping[count % 3]} permissions: {self.instance}"
                )

                return CheckResult(False, err_msg)

        return CheckResult(True)

    @rule("file size larger than expected")
    def max_size(self, expected_size: int) -> Union["PathChecker", CheckResult]:
        err_msg = f"File size larger than expected: {self.instance}"
        return CheckResult(self.f_status.size < expected_size, err_msg)

    @rule("Wrong file suffix")
    def check_extensions(self, extensions) -> Union["PathChecker", CheckResult]:
        return self.f_status.extension in (extensions, '.' + extensions)

    @rule()
    def is_safe_parent_dir(self) -> Union["PathChecker", CheckResult]:
        path = os.path.realpath(self.instance)
        dirpath = os.path.dirname(path)
        if os.getuid() == 0:
            return True

        dir_checker = PathChecker().any(
            PathChecker().anti(PathChecker().exists()),
            PathChecker().is_dir(),
        )
        return dir_checker.check(dirpath)
