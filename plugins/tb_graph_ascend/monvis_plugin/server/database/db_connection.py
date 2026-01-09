# Copyright (c) 2025, Huawei Technologies.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import stat
import sqlite3
import json
from pathlib import Path
from tensorboard.util import tb_logging
logger = tb_logging.get_logger()

FILE_PATH_MAX_LENGTH = 4096
# 权限码
PERM_GROUP_WRITE = 0o020
PERM_OTHER_WRITE = 0o002
MAX_FILE_SIZE = 3 * 1024 * 1024 * 1024  # 最大文件大小限制


class DBConnection:
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = self._initialize_db_connection()

    @classmethod
    def _bytes_to_human_readable(cls, size_bytes, decimal_places=2):
        """
        将字节大小转换为更易读的格式（如 KB、MB、GB 等）。
        
        :param size_bytes: int 或 float，表示字节大小
        :param decimal_places: 保留的小数位数，默认为 2
        :return: str，人类可读的大小表示
        """
        if size_bytes == 0:
            return "0 B"

        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0

        while size_bytes >= 1024 and unit_index < len(units) - 1:
            size_bytes /= 1024.0
            unit_index += 1

        return f"{size_bytes:.{decimal_places}f} {units[unit_index]}"

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.conn is not None
    
    def _initialize_db_connection(self) -> None:
        """Initialize database connection."""
        try:
            # 目录安全校验
            directory = str(os.path.dirname(self.db_path))
            success, error = self._safe_check_load_file_path(directory, True)
            if not success:
                raise PermissionError(error)
            # 文件安全校验
            success, error = self._safe_check_load_file_path(self.db_path)
            if not success:
                raise PermissionError(error)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            return None

    def _safe_check_load_file_path(self, file_path, is_dir=False):
        # 权限常量定义
        file_path = os.path.normpath(file_path)  # 标准化路径
        real_path = os.path.realpath(file_path)
        st = os.stat(real_path)
        try:
            # 安全验证：路径长度检查
            if len(real_path) > FILE_PATH_MAX_LENGTH:
                raise PermissionError(f"Path length exceeds limit")
            # 安全检查：文件存在性验证
            if not os.path.exists(real_path):
                raise FileNotFoundError(f"File does not exist")
            # 安全验证：禁止符号链接文件
            if os.path.islink(file_path):
                raise PermissionError(f"Detected symbolic link file")
            # 安全验证：文件类型检查（防御TOCTOU攻击）
            # 文件类型
            if not is_dir and not os.path.isfile(real_path):
                raise PermissionError(f"Path is not a regular file")
            # 目录类型
            if is_dir and not Path(real_path).is_dir():
                raise PermissionError(f"Directory does not exist")
            # 可读性检查
            if not st.st_mode & stat.S_IRUSR:
                raise PermissionError(
                    f"Directory lacks read permission for others, there may be a risk of data tampering.")
            # 文件大小校验
            if not is_dir and os.path.getsize(file_path) > MAX_FILE_SIZE:
                file_size = self._bytes_to_human_readable(os.path.getsize(file_path))
                max_size = self._bytes_to_human_readable(MAX_FILE_SIZE)
                raise PermissionError(
                    f"File size exceeds limit ({file_size} > {max_size})")
            # 非windows系统下，属主检查
            if os.name != 'nt':
                current_uid = os.getuid() 
                # 如果是root用户，跳过后续权限检查
                if current_uid == 0:
                    return True, None
                # 属主检查
                if st.st_uid != current_uid:
                    raise PermissionError(f"Directory is not owned by the current user")
                # group和其他用户不可写检查
                if st.st_mode & PERM_GROUP_WRITE or st.st_mode & PERM_OTHER_WRITE:
                    raise PermissionError(f"Directory has group or other write permission")
            return True, None
        except Exception as e:
            logger.error(e)
            return False, e
