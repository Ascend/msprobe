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
import json
import os
import stat
from pathlib import Path
from tensorboard.util import tb_logging
logger = tb_logging.get_logger()
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  
FILE_PATH_MAX_LENGTH = 4096
PERM_GROUP_WRITE = 0o020
PERM_OTHER_WRITE = 0o002


class Utils:  
    @staticmethod   
    def safe_json_loads(json_str, default_value=None):
        """
        安全地解析 JSON 字符串，带长度限制和异常处理。
        :param json_str: 要解析的 JSON 字符串
        :param default_value: 如果解析失败返回的默认值
        :return: 解析后的 Python 对象 或 default_value
        """
        # 类型检查
        if not isinstance(json_str, str):
            return default_value
        # 长度限制
        if len(json_str) > MAX_FILE_SIZE:
            return default_value
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return default_value
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return default_value
    
    @staticmethod
    def is_relative_to(path, base):
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(base)
        return os.path.commonpath([abs_path, abs_base]) == str(abs_base)
    
    
    @staticmethod
    def bytes_to_human_readable(size_bytes, decimal_places=2):
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
    

    @staticmethod
    def safe_check_load_file_path(file_path, is_dir=False):
        # 权限常量定义
        
        file_path = os.path.normpath(file_path)  # 标准化路径
        real_path = os.path.realpath(file_path)
        st = os.stat(real_path)
        try:
            # 安全验证：路径长度检查
            if len(real_path) > FILE_PATH_MAX_LENGTH:
                raise PermissionError(
                    f"Path is too long (max {FILE_PATH_MAX_LENGTH} characters). Please use a shorter path."
                )
            # 安全检查：文件存在性验证
            if not os.path.exists(real_path):
                raise FileNotFoundError(f"File or directory does not exist,please check the path and ensure it exists.")
            # 安全验证：禁止符号链接文件
            if os.path.islink(file_path):
                raise PermissionError(f"Symbolic links are not allowed,Use a real file path instead.")
            # 安全验证：文件类型检查（防御TOCTOU攻击）
            # 文件类型
            if not is_dir and not os.path.isfile(real_path):
                raise PermissionError(
                    f"Path is not a regular file."
                     "make sure the path points to a valid file (not a directory or device)."
                )
            # 目录类型
            if is_dir and not Path(real_path).is_dir():
                raise PermissionError(
                    f"Expected a directory, but it does not exist or is not a directory."
                     "Please check the path and ensure it is a valid directory."
                )
            # 可读性检查
            if not st.st_mode & stat.S_IRUSR:
                raise PermissionError(
                    f"Current user lacks read permission on file or directory"
                     "Run 'chmod u+r \"<path>\"' to grant read access"
                )
            # 文件大小校验
            if not is_dir and os.path.getsize(file_path) > MAX_FILE_SIZE:
                file_size = Utils.bytes_to_human_readable(os.path.getsize(file_path))
                max_size = Utils.bytes_to_human_readable(MAX_FILE_SIZE)
                raise PermissionError(
                    f"File size exceeds limit ({file_size} > {max_size})."
                     "reduce file size or adjust MAX_FILE_SIZE if needed"
                )
            # 非windows系统下，属主检查
            if os.name != 'nt':
                current_uid = os.getuid() 
                # 如果是root用户，跳过后续权限检查
                if current_uid == 0:
                    logger.warning("""Security Warning: Do not run this tool as root. 
                                   Running with elevated privileges may compromise system security. 
                                   Use a regular user account.""")
                    return True, None 
                # 属主检查
                if st.st_uid != current_uid:
                    raise PermissionError(
                        f"File or directory is not owned by current user,"
                         "Run 'chown <user> \"<path>\"' to fix ownership."
                    )
                # group和其他用户不可写检查
                if st.st_mode & PERM_GROUP_WRITE or st.st_mode & PERM_OTHER_WRITE:
                    raise PermissionError(
                        f"File has insecure permissions: group or others have write access. "
                         "Run 'chmod go-w \"<path>\"' to remove write permissions for group and others."
                    )
            return True, None
        except Exception as e:
            logger.error(e)
            return False, e