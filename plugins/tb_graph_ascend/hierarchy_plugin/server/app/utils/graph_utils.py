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

import math
import os
import json
import re
import stat
import threading
from functools import cmp_to_key
from pathlib import Path
from tensorboard.util import tb_logging
from .global_state import GraphState
from .constant import DataType, FILE_NAME_REGEX, MAX_FILE_SIZE, PERM_GROUP_WRITE, PERM_OTHER_WRITE, COLOR_PATTERN
from .i18n import language, ZH_CN
# 创建一个全局锁
_thread_local_lock = threading.Lock()
logger = tb_logging.get_logger()
FILE_PATH_MAX_LENGTH = 4096


class GraphUtils:

    @staticmethod
    def t(key):
        lang = GraphState.get_global_value('lang', ZH_CN)
        return language.get(lang).get(key, '')

    @staticmethod
    def get_graph_data(meta_data):
        if not meta_data:
            return None, 'Error: no query parameters provided'
        try:
            run_name = meta_data.get('run')
            runs = GraphState.get_global_value('runs', {})
            run = runs.get(run_name) or run_name
            tag = meta_data.get('tag')
            current_tag = GraphState.get_global_value('current_tag')
            current_run = GraphState.get_global_value('current_run')
            if current_tag == tag and current_run == run:
                return GraphState.get_global_value('current_file_data'), None  # 直接返回获取结果
            else:
                json_data, error_message = GraphUtils.safe_load_data(run, f"{tag}.vis", False)
                if error_message:
                    return None, 'Error: fail to get graph data'
                GraphState.set_global_value('current_file_data', json_data)
                GraphState.set_global_value('current_tag', tag)
                GraphState.set_global_value('current_run', run)
                return json_data, error_message
        except Exception as e:
            logger.error(f'Error: fail to get graph data by {meta_data}, error: {e}')
            return None, 'Error: fail to get graph data'

    @staticmethod
    def get_opposite_node_name(node_name):
        opposite_node_name = ''
        # 如果npu_node_name包含forward，则opposite_npu_node_name为npu_node_name替换forward为backward
        if 'forward' in node_name:
            opposite_node_name = node_name.replace('forward', 'backward')
        else:
            opposite_node_name = node_name.replace('backward', 'forward')
        return opposite_node_name

    @staticmethod
    def get_parent_node_list(graph_data, node_name):
        """获取父节点列表"""
        # 如果 graph_data 为空或 node_name 不存在，直接返回空列表
        if not graph_data or not node_name:
            return []

        node_list = []  # 存储结果的列表
        current_node = node_name  # 当前节点名称
        nodes = graph_data['node']  # 提取节点数据，避免重复访问

        while current_node:
            node_list.append(current_node)  # 将当前节点加入列表
            # 获取当前节点的 upnode
            current_node = nodes.get(current_node, {}).get('upnode')

            # 检测循环引用（防止死循环）
            if current_node in node_list:
                raise ValueError(
                    f"{GraphUtils.t('circularReferenceError1')}"
                    f"{current_node}{GraphUtils.t('circularReferenceError2')}"
                )

        return list(reversed(node_list))  # 返回结果列表

    @staticmethod
    def split_graph_data_by_microstep(graph_data, micro_step):
        graph_nodes = graph_data.get('node', {})
        if str(micro_step) == str(-1):
            return graph_nodes
        splited_graph_data = {}
        node_list = []  # 存储已遍历的列表

        def traverse_npu(graph_nodes, subnodes):
            for node_name in subnodes:
                node_data = graph_nodes.get(node_name, {})
                micro_step_id = node_data.get('micro_step_id')
                if str(micro_step_id) == str(micro_step) or micro_step_id is None:
                    splited_graph_data[node_name] = (node_data)
                    if node_name in node_list:
                        raise ValueError(
                            f"{GraphUtils.t('circularReferenceError1')}"
                            f"{node_name}{GraphUtils.t('circularReferenceError2')}"
                        )
                    node_list.append(node_name)
                    traverse_npu(graph_nodes, node_data.get('subnodes', []))

        root = graph_data.get('root')
        root_subnodes = graph_nodes.get(root, {}).get('subnodes', [])
        node_list.append(root)
        traverse_npu(graph_nodes, root_subnodes)
        return splited_graph_data

    @staticmethod
    def walk_with_max_depth(logdir, max_depth):
        for root, dirs, files in os.walk(logdir):
            # 计算当前 root 相对于 top 的深度
            depth = root[len(logdir):].count(os.sep) + 1
            if depth >= max_depth:
                del dirs[:]
            yield root, dirs, files

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
    def safe_get_node_info(data, default_value=None):
        node_info = data.get('nodeInfo')
        try:
            # 长度限制 - 检查字典转为字符串后的长度
            node_info_str = str(node_info)
            if len(node_info_str) > MAX_FILE_SIZE:
                logger.error(f"Input length exceeds {MAX_FILE_SIZE} characters.")
                return default_value
            # 验证必要字段是否存在
            required_fields = ["nodeName", "nodeType"]
            for field in required_fields:
                if field not in node_info:
                    logger.error(f"Field {field} is missing in metadata.")
                    return default_value
            return node_info
        except json.JSONDecodeError:
            logger.error("NodeInfo parameter is not in valid JSON format.")
            return default_value
        except Exception as e:
            logger.error(f"An error occurred while parsing the nodeInfo parameter: {str(e)}")
            return default_value

    @staticmethod
    def safe_get_meta_data(data, default_value=None):
        meta_data = data.get('metaData')
        try:
            # 长度限制
            meta_data_str = str(meta_data)
            if len(meta_data_str) > MAX_FILE_SIZE:
                logger.error(f"Input length exceeds {MAX_FILE_SIZE} characters.")
                return default_value
            # 验证必要字段是否存在
            required_fields = ["tag", "microStep", "run", "type", 'lang']
            for field in required_fields:
                if field not in meta_data:
                    logger.error(f"Field {field} is missing in metadata.")
                    return default_value
            config_info = GraphState.get_global_value('config_info', {})
            micro_step = meta_data.get('microStep')
            # 验证非必要字段如果存在,进行范围限制
            if config_info.get('microSteps'):
                if int(micro_step) != -1 and int(micro_step) not in list(range(config_info.get('microSteps'))):
                    logger.error(f"Field microStep {micro_step} is not in config_info.")
                    return default_value
            if meta_data.get('type') == DataType.DB.value and config_info:
                rank = meta_data.get('rank')
                step = meta_data.get('step')
                if config_info.get('ranks') and rank:
                    if rank not in config_info.get('ranks'):
                        logger.error(f"Field rank {rank} is not in config_info.")
                        return default_value
                if config_info.get('steps') and step:
                    if step not in config_info.get('steps'):
                        logger.error(f"Field step {step} is not in config_info.")
                        return default_value
            return meta_data
            
        except json.JSONDecodeError:
            logger.error("MetaData parameter is not in valid JSON format.")
            return default_value
        except Exception as e:
            logger.error(f"An error occurred while parsing the metatdata parameter: {str(e)}")
            return default_value

    @staticmethod
    def remove_prefix(node_data, prefix):
        if node_data is None:
            return {}
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in node_data.items()}

    @staticmethod
    def convert_to_float(value):
        try:
            if isinstance(value, str):
                # 处理'0.0%, 由于Mean小于1e-06, 建议不参考此相对误差，请参考绝对误差'和'0.0%'的情况
                value = value.split(',')[0]
                if value.endswith('%'):
                    value = value.replace('%', '').strip()
                    return float(value) / 100.0
            return float(value)
        except ValueError:
            return float('nan')

    @staticmethod
    def format_relative_err(value):
        """格式化相对误差为百分比，保留四位小数"""
        if value is None or math.isnan(value):
            return "NaN"
        else:
            return "{:.4%}".format(value)

    @staticmethod
    def nan_to_str(value):
        """将 NaN 转换为 'NaN' 字符串"""
        return "NaN" if math.isnan(value) else value

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
    def safe_save_data(data, run_name, tag):
        runs = GraphState.get_global_value('runs', {})
        run = runs.get(run_name) or run_name
        if not run or not tag:
            error_message = 'The query parameters "run" and "tag" are required'
            return None, error_message
        try:
            # 检查 tag 是否为合法文件名
            if not re.match(FILE_NAME_REGEX, tag):
                raise ValueError(f"{GraphUtils.t('invalidTag')}{tag}.")
            # 构建文件路径并标准化
            file_path = os.path.join(run, tag)
            # 目录安全校验
            success, error = GraphUtils.safe_check_save_file_path(run, True)
            if not success:
                raise PermissionError(error)
            # 文件安全校验
            success, error = GraphUtils.safe_check_save_file_path(file_path)
            if not success:
                raise PermissionError(error)
            # 尝试写入文件
            with _thread_local_lock:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    f.flush()  # 强制将缓冲区内容写入操作系统
                    os.fsync(f.fileno())  # 强制将缓冲区内容写入磁盘
                os.chmod(file_path, 0o640)
            # 最终校验（防御TOCTOU攻击）
            if os.path.islink(file_path):
                raise RuntimeError("The file has been replaced with a symbolic link")
            return True, None
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data: {e}")
            return None, 'Invalid data'
        except OSError as e:
            logger.error(f"Failed to create directory: {run}. Error: {e}\n")
            return None, 'failed to create directory '
        except Exception as e:
            logger.error(f'Error: File "{file_path}" is not accessible. Error: {e}')
            return None, 'failed to save file'

    @staticmethod
    def safe_load_data(run_name, tag, only_check=False):
        runs = GraphState.get_global_value('runs', {})
        run_dir = runs.get(str(run_name)) or run_name
        safe_base_dir = GraphState.get_global_value('logdir')
        """Load a single .vis file from a given directory based on the tag."""
        if run_dir is None or tag is None:
            error_message = 'The query parameters "run" and "tag" are required'
            return None, error_message
        try:
            file_path = os.path.join(run_dir, tag)
            # 安全验证：基础路径校验
            if not GraphUtils.is_relative_to(file_path, safe_base_dir):
                raise ValueError(GraphUtils.t('pathMayNotInSecureDirectory'))
            # 目录安全校验
            success, error = GraphUtils.safe_check_load_file_path(run_dir, True)
            if not success:
                raise PermissionError(error)
            # 文件安全校验
            success, error = GraphUtils.safe_check_load_file_path(file_path)
            if not success:
                raise PermissionError(error)
            # 读取文件比较耗时，支持onlyCheck参数，仅进行安全校验
            if only_check:
                return True, None
            # 尝试解析 JSON 文件,校验文件内容是否合理
            with _thread_local_lock:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f), None
        except json.JSONDecodeError:
            logger.error(f'Error: File "{file_path}" is not a valid JSON file!')
            return None, "File is not a valid JSON file!"
        except Exception as e:
            logger.error(f'Error: File "{file_path}" is not accessible. Error: {e}')
            return None, e

    @staticmethod
    def safe_check_save_file_path(file_path, is_dir=False):
        file_path = os.path.normpath(file_path)  # 标准化路径
        real_path = os.path.realpath(file_path)
        safe_base_dir = GraphState.get_global_value('logdir')
        try:
            # 安全验证：路径长度检查
            if len(file_path) > FILE_PATH_MAX_LENGTH:
                raise PermissionError(f"Path length exceeds limit")
            if not is_dir and not os.path.exists(file_path):
                return True, None
            st = os.stat(file_path)
            # 安全验证：禁止符号链接文件
            if os.path.islink(file_path):
                raise PermissionError("The target file is a symbolic link")            
            # 安全验证：检查目录是否存在，如果不存在则创建
            if is_dir and not os.path.exists(real_path):
                os.makedirs(real_path, exist_ok=True)
                os.chmod(file_path, 0o640)
            # 权限校验：检查是否有写权限
            if not os.stat(file_path).st_mode & stat.S_IWUSR:
                raise PermissionError(f"No write permission for directory\n")
            # 安全验证： 非windows系统下，属主检查
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
                    raise PermissionError(f"Directory is not owned by the current user")
                # group和其他用户不可写检查
                if st.st_mode & PERM_GROUP_WRITE or st.st_mode & PERM_OTHER_WRITE:
                    raise PermissionError(
                        f"Directory has group or other write permission, there may be a risk of data tampering.")
            return True, None
        except Exception as e:
            logger.error(e)
            return False, e
    
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
                file_size = GraphUtils.bytes_to_human_readable(os.path.getsize(file_path))
                max_size = GraphUtils.bytes_to_human_readable(MAX_FILE_SIZE)
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
       
    @staticmethod
    def find_config_files(run_name):
        """
        在指定目录下查找所有以 .vis.config 结尾的文件（不包括子目录）
        :param directory: 要搜索的目标目录路径
        :return: 包含所有匹配文件路径的列表
        """
        runs = GraphState.get_global_value('runs', {})
        run = runs.get(run_name)
        dir_path = Path(run)
        try:
            if GraphUtils.safe_check_load_file_path(run, True):
                return [
                    file.name for file in dir_path.iterdir()
                    if file.is_file() and file.name.endswith('.vis.config')
                ]
            else:
                return []
        except Exception as e:
            logger.error(e)
            return []

    @staticmethod
    def compare_tag_names(a: str, b: str) -> int:
        """自然排序比较函数，支持路径格式和数字顺序"""

        # 辅助函数：标准化路径并分割组件
        def split_components(s: str) -> list:
            s = s.replace('\\', '/')  # 统一路径分隔符
            return [c for c in re.split(r'[/_]+', s) if c]  # 按 /_ 分割非空组件

        # 辅助函数：将组件拆分为数字/字符串混合的 tokens
        def tokenize(component: str) -> list:
            tokens = []
            buffer = []
            is_num = False

            for char in component:
                if char.isdigit() == is_num and buffer:
                    buffer.append(char)
                else:
                    if buffer:
                        tokens.append(int(''.join(buffer)) if is_num else ''.join(buffer))
                    buffer = [char]
                    is_num = char.isdigit()

            if buffer:
                tokens.append(int(''.join(buffer)) if is_num else ''.join(buffer))
            return tokens

        # 逐级比较组件
        a_comps = split_components(a)
        b_comps = split_components(b)

        for a_part, b_part in zip(a_comps, b_comps):
            a_tokens = tokenize(a_part)
            b_tokens = tokenize(b_part)

            # 比较 token 序列
            for a_tok, b_tok in zip(a_tokens, b_tokens):
                if isinstance(a_tok, int) and isinstance(b_tok, int):
                    if a_tok != b_tok:
                        return a_tok - b_tok
                elif isinstance(a_tok, int):
                    return -1  # 数字优先于字母
                elif isinstance(b_tok, int):
                    return 1
                elif a_tok != b_tok:
                    return -1 if a_tok < b_tok else 1

            # 处理子序列长度差异
            if len(a_tokens) != len(b_tokens):
                return len(a_tokens) - len(b_tokens)

        # 处理组件数量差异
        return len(a_comps) - len(b_comps)

    @staticmethod
    def sort_data(data: dict) -> dict:
        """自然排序比较函数，支持路径格式和数字顺序"""
        sorted_data = {}
        sorted_keys = sorted(data.keys(), key=cmp_to_key(GraphUtils.compare_tag_names))
        for k in sorted_keys:
            # 对每个键对应的值列表进行排序
            sorted_values = sorted(data.get(k, {}).get('tags'), key=cmp_to_key(GraphUtils.compare_tag_names))
            sorted_data[k] = {'type': data.get(k, {}).get('type'), 'tags': sorted_values}

        return sorted_data
            
    @staticmethod
    def is_safe_string(s: str) -> bool:
        """
        安全检查字符串是否包含恶意内容（XSS 防护）
        """
        if not isinstance(s, str):
            return False

        # 1. 长度限制（字符数，非字节）
        if len(s) > FILE_PATH_MAX_LENGTH:
            return False

        # 2. 转换为小写用于检测
        s_lower = s.lower().strip()

        # 3. 黑名单关键词（常见 XSS 向量）
        dangerous_patterns = [
            '<script', '<img', '<svg', '<iframe', '<video', '<audio',
            'javascript:', 'vbscript:', 'data:text/html',
            'onload=', 'onerror=', 'onmouseover=', 'onclick=',
            'eval(', 'alert(', 'document.cookie', 'document.location',
            'window.location', 'innerHTML', 'outerHTML', 'document.write'
        ]

        if any(pattern in s_lower for pattern in dangerous_patterns):
            return False

        return True
    
    @staticmethod
    def escape_html(input_str):
        """
        将字符串中的特殊 HTML 字符转义为 HTML 实体。
        """
        html_escape_map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
            '/': '&#x2F;',
        }
        # 使用 str.translate 配合 str.maketrans 提高性能
        translate_table = str.maketrans(html_escape_map)
        return input_str.translate(translate_table)

    @staticmethod
    def validate_colors_param(colors_json: str):
        # 合法颜色正则：#FFFFFF 格式，不区分大小写
    
        """
        校验 colors 参数
        返回: (是否合法, 错误信息, 解析后的数据)
        """

        if not isinstance(colors_json, dict):
            return False, GraphUtils.t('colorsNotObject'), {}

        if len(colors_json) == 0:
            return False, GraphUtils.t('colotsEmpty'), {}

        for key, value in colors_json.items():
            # 2. 校验颜色键
            if not re.match(COLOR_PATTERN, key):
                return False, f"{GraphUtils.t('illegalColorKey')}{key}", {}

            if not isinstance(value, dict):
                return False, f"{GraphUtils.t('colorValuesNotObject')}{key}", {}

            if 'value' not in value:
                return False, f"{GraphUtils.t('missingValueField')}{key}", {}

            # 3. 校验 value 字段
            val = value['value']
            if isinstance(val, list):
                if len(val) != 2:
                    return False, f"{GraphUtils.t('notArrayOfLength2')}{key}", {}
                if not all(isinstance(x, (int, float)) for x in val):
                    return False, f"{GraphUtils.t('notArrayConsistNumbers')}{key}", {}
                if val[0] >= val[1]:
                    return False, f"{GraphUtils.t('invalidValueRange')}{key}", {}
            elif isinstance(val, str):
                if val not in ["无匹配节点", "N/A", "No matching nodes"]:
                    return False, f"{GraphUtils.t('unsupportedValue')}{val}", {}
            else:
                return False, f"{GraphUtils.t('valueTypeError')}{key}", {}

            # 4. 校验 description
            desc = value.get('description', '')
            if not GraphUtils.is_safe_string(desc):
                return False, f"{GraphUtils.t('descriptionError')}{key}", {}
            value['description'] = GraphUtils.escape_html(desc)

        return True, None, colors_json
