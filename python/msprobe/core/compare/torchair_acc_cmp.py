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
import re
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import numpy as np

from msprobe.core.common.const import FileCheckConst
from msprobe.core.common.file_utils import FileChecker, FileOpen, check_file_or_directory_path
from msprobe.core.common.log import logger
from msprobe.core.common.utils import CompareException
from msprobe.core.compare.torchair_cmp_utils import BasicDataInfo, fill_row_data, save_compare_result_to_csv
from msprobe.infer.utils.acc_cmp import parse_torchair_dump_data, set_msaccucmp_path_from_cann

GE_GRAPH_FILE_PREFIX = 'dynamo_original_graph_'
GE_DUMP_TIME_PATTERN = 'YYYYMMDDHHMMSS'
FUSION_OP_TYPE = 'AutomaticBufferFusionOp'
DUMP_FILE_FILTER_SUFIX = ['.txt', '.npy', '.bin']
MAX_TOKEN_LEN = 12


def get_rank_id_from_torchair_data(dir_name: str):
    rank_id = -1
    rank_index = dir_name.rfind('rank')
    if dir_name.startswith('worldsize') and rank_index != -1 and str.isdigit(dir_name[rank_index + 4 :]):
        rank_id = int(dir_name[rank_index + 4 :])
    return rank_id


def get_torchair_ge_graph_path(my_path, rank=-1):
    if not os.path.isdir(my_path):
        return None

    ge_graph_files = []
    my_path_depth = len(my_path.split(os.sep))
    timestamp_pattern = re.compile(r"(\d+)")
    for cur_path, _, file_names in os.walk(my_path):
        for file_name in file_names:
            if rank > -1 and f'rank_{rank}_' not in file_name:
                continue
            if file_name.startswith(GE_GRAPH_FILE_PREFIX) and file_name.endswith(".txt"):
                match = timestamp_pattern.search(file_name)
                if match:
                    full_path = os.path.join(cur_path, file_name)
                    timestamp = int(match.group(1))
                    ge_graph_files.append((full_path, timestamp))

            cur_depth = len(cur_path.split(os.sep)) - my_path_depth
            if cur_depth > 5:  # Avoid going too deep
                break

    if ge_graph_files:
        sorted_ge_graph_files = [file for file, timestamp in sorted(ge_graph_files, key=lambda x: x[1])]
        return sorted_ge_graph_files
    return None


def get_unique_key(cur_dict, cur_key):
    split_sign, original_cur_key, cur_key_id = "#", cur_key, 0
    while cur_key in cur_dict:
        cur_key_id += 1
        cur_key = f"{original_cur_key}{split_sign}{cur_key_id}"
    return cur_key


def parse_pbtxt_to_dict(pbtxt_path):
    check_file_or_directory_path(pbtxt_path)
    with FileOpen(pbtxt_path, "r") as ff:
        contents = ff.read()

    result, cur_dict, superior_dicts, brackets_depth = [], {}, [], 0
    for cur_line in contents.split("\n"):
        cur_line = cur_line.strip()
        if len(cur_line) == 0:
            continue

        if " {" in cur_line:
            if brackets_depth == 0:
                cur_dict = {}
                superior_dicts = []
                result.append(cur_dict)
            cur_key = cur_line.split(" {")[0]
            cur_key = get_unique_key(cur_dict, cur_key)
            cur_dict[cur_key] = {}
            if len(superior_dicts) > brackets_depth:
                superior_dicts[brackets_depth] = cur_dict
            else:
                superior_dicts.append(cur_dict)
            cur_dict = cur_dict[cur_key]
            brackets_depth += 1
        elif ": " in cur_line:
            cur_key, cur_value = cur_line.split(": ")
            cur_key = get_unique_key(cur_dict, cur_key)
            cur_value = cur_value[1:-1] if cur_value.startswith('"') and cur_value.endswith('"') else cur_value
            cur_dict[cur_key] = cur_value
        elif "}" in cur_line:
            brackets_depth -= 1
            cur_dict = superior_dicts[brackets_depth]
    return result


def judge_single_or_multi_device(path):
    def is_time_directory(dir_name):
        if not os.path.isdir(os.path.join(path, dir_name)):
            return False
        return len(dir_name) == len(GE_DUMP_TIME_PATTERN) and str.isdigit(dir_name)

    # 获取指定目录下所有文件和文件夹
    entries = os.listdir(path)
    # 过滤出文件夹
    subdirs = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]
    if len(subdirs) > 1:
        return True

    time_dirs = [os.path.join(path, entry) for entry in entries if is_time_directory(entry)]
    if time_dirs:
        entries = os.listdir(time_dirs[0])
        subdirs = [entry for entry in entries if os.path.isdir(os.path.join(time_dirs[0], entry))]
        return len(subdirs) > 1

    return False


def _has_rank_directory(dump_dir):
    if not os.path.isdir(dump_dir):
        return False
    for entry in os.listdir(dump_dir):
        if os.path.isdir(os.path.join(dump_dir, entry)) and get_rank_id_from_torchair_data(entry) != -1:
            return True
    return False


def _validate_read_path(path):
    path_type = FileCheckConst.DIR if os.path.isdir(path) else FileCheckConst.FILE
    FileChecker(path, path_type, ability=FileCheckConst.READ_ABLE).common_check()


def gather_data_with_token_id_fx(data_path, token_dirs, rank_info_existed=False):
    token_dir_groups = []
    if token_dirs:
        token_dir_groups.append(token_dirs)
    else:
        # Sort os.walk by cur_path so the resulting token_dir_groups list is
        # in a stable order regardless of the underlying filesystem's
        # os.listdir() ordering (which differs between e.g. ext4 / overlayfs /
        # some CI sandboxes and would otherwise swap groups for callers that
        # rely on positional ordering).
        for cur_path, dirs, _ in sorted(os.walk(data_path), key=lambda x: x[0]):
            if len(dirs) == 0:
                continue
            if all(len(ii) < MAX_TOKEN_LEN and str.isdigit(ii) for ii in dirs):
                dirs.sort(key=int)
                token_dir_groups.append([os.path.join(cur_path, dir_name) for dir_name in dirs])

    if len(token_dir_groups) == 0:
        token_dir_groups.append([data_path])  # Just use data_path if found no token like dirs

    gathered_files_list = []

    for token_group in token_dir_groups:
        if rank_info_existed:
            gathered_files = {}
            for token_dir in token_group:
                cur_token_id = os.path.basename(token_dir)
                cur_token_id = int(cur_token_id) + 1 if cur_token_id.isdigit() else 0
                file_names = [os.path.join(token_dir, f) for f in os.listdir(token_dir) if f.endswith(".npy")]
                if not file_names:
                    # Skip empty token directories rather than silently include them;
                    # downstream stages assume gathered_files[token_id] is non-empty.
                    continue
                gathered_files[cur_token_id] = file_names
            if gathered_files:
                gathered_files_list.append(gathered_files)
            continue

        dump_dirs = {}
        for token_dir in token_group:
            cur_token_id = os.path.basename(token_dir)
            cur_token_id = int(cur_token_id) if cur_token_id.isdigit() else 0
            dump_dirs[cur_token_id] = sorted(
                [
                    os.path.join(token_dir, d)
                    for d in os.listdir(token_dir)
                    if os.path.isdir(os.path.join(token_dir, d))
                ],
                key=os.path.basename,
            )
        num_dumps_per_token = {tid: len(dumps) for tid, dumps in dump_dirs.items()}
        distinct_counts = set(num_dumps_per_token.values())
        if len(distinct_counts) > 1:
            logger.warning(
                f"gather_data_with_token_id_fx found inconsistent dump counts across tokens: {num_dumps_per_token}. "
                "The smallest count will be used to avoid mixing unmatched dump indices."
            )
        num_dumps = min(num_dumps_per_token.values()) if num_dumps_per_token else 0
        for i in range(num_dumps):
            gathered_files = {}
            for cur_token_id, dumps in dump_dirs.items():
                dump_path = dumps[i]
                file_names = [os.path.join(dump_path, f) for f in os.listdir(dump_path) if f.endswith(".npy")]
                if not file_names:
                    continue
                gathered_files[cur_token_id] = file_names
            if gathered_files:
                gathered_files_list.append(gathered_files)
    return gathered_files_list


def gather_data_with_token_id(data_path, fx=False, rank_info_existed=False):
    token_dirs = []
    # Detect the deepest dir level where sub dirs are all digits, and regard as tokens level.
    if fx:
        return gather_data_with_token_id_fx(data_path, token_dirs, rank_info_existed)

    is_multi_device = False if rank_info_existed else judge_single_or_multi_device(data_path)

    for cur_path, dirs, _ in sorted(os.walk(data_path), key=lambda x: x[0]):
        if not dirs:
            token_dirs.append(cur_path)

    if len(token_dirs) == 0:
        token_dirs.append(data_path)  # Just use data_path if found no token like dirs

    gathered_files_list = []
    parent_dir_dict = {}
    for token_dir in token_dirs:
        if is_multi_device:
            parts = token_dir
            # token_dir格式如下：
            # /home/dump/dump_20241114_113410/0/graph_1_0/1/4/
            # dump路径+时间戳+device_id+子图名称+子图ID号+token_id
            # 对于多卡场景，应取device_id下相同的子图进行比较，
            # 此处parts=/home/dump/dump_20241114_113410/0
            for _ in range(3):
                parts = os.path.dirname(parts)
            parent_dir = os.path.basename(parts)
        else:
            parent_dir = os.path.basename(os.path.dirname(token_dir))
        subdir = os.path.basename(token_dir)
        parent_id = int(parent_dir) if parent_dir.isdigit() else 0
        subdir_id = int(subdir) if subdir.isdigit() else 0
        if parent_id not in parent_dir_dict:
            parent_dir_dict[parent_id] = {}
        if subdir_id not in parent_dir_dict[parent_id]:
            parent_dir_dict[parent_id][subdir_id] = []
        for cur_path, _, file_names in os.walk(token_dir):
            file_names = [os.path.join(cur_path, file_name) for file_name in file_names]
            parent_dir_dict[parent_id][subdir_id].extend(file_names)
    parent_dir_dict = dict(sorted(parent_dir_dict.items()))
    for _, subdirs in parent_dir_dict.items():
        gathered_files_list.append(subdirs)
    return gathered_files_list


def init_ge_dump_data_from_bin_path(ge_dump_path):
    """
    For data like:
      1/Add.Add_2.44.6.1706596912161941,
      1/Cast.Cast_9.19.6.1706596911887829,
      1/ConcatV2D.ConcatV2.42.6.1706596912161117,

    Return dict:
      {1: {
            'Add_2': '1/Add.Add_2.44.6.1706596912161941',
            'Cast_9': '1/Cast.Cast_9.19.6.1706596911887829',
            'ConcatV2': '1/ConcatV2D.ConcatV2.42.6.1706596912161117',
      }}
    """
    gathered_files_list = gather_data_with_token_id(ge_dump_path)
    if not gathered_files_list:
        raise CompareException(
            CompareException.NO_DUMP_FILE_ERROR, "Cannot get ge dump data, because the gathered_files_list is empty."
        )

    dump_data_with_token_id_list = []
    for gathered_files in gathered_files_list:
        dump_data_with_token_id = {}
        for token_id, file_list in gathered_files.items():
            cur_dump_data = {}
            for file_name in sorted(file_list):
                if os.path.splitext(file_name)[-1] in DUMP_FILE_FILTER_SUFIX:
                    continue
                split_name = os.path.basename(file_name).split(".")
                if len(split_name) < 5:
                    logger.warning(f"invalid file name: {file_name}, should contain at least 4 '.'")
                    continue

                cur_op_name = ".".join(split_name[1:-3])
                if cur_op_name in cur_dump_data:
                    exists_file = cur_dump_data[cur_op_name]
                    exists_file_size = os.path.getsize(exists_file)
                    cur_file_size = os.path.getsize(file_name)
                    keep_one = file_name if cur_file_size > exists_file_size else exists_file
                    cur_dump_data[cur_op_name] = keep_one
                    logger.warning(
                        f"duplicated op name: {cur_op_name}."
                        f" [{os.path.basename(file_name)}, {os.path.basename(exists_file)}]."
                        f" Will keep the larger one {os.path.basename(keep_one)}."
                    )
                else:
                    cur_dump_data[cur_op_name] = file_name
            dump_data_with_token_id[token_id] = cur_dump_data
        dump_data_with_token_id_list.append(dump_data_with_token_id)
    return dump_data_with_token_id_list


def init_fx_dump_data_from_path(fx_dump_path, rank_info_existed=False):
    """
    For data like:
      1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy,
      1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy,
      1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy,

    Return dict:
      {1: {'mm-aten.mm.default': {
        'input': [
          '1/mm-aten.mm.default.INPUT.0.20240125031118787351.npy',
          '1/mm-aten.mm.default.INPUT.1.20240125031118787351.npy',
        ],
        'output': ['1/mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy']
      }}}
    """
    gathered_files_list = gather_data_with_token_id(fx_dump_path, fx=True, rank_info_existed=rank_info_existed)
    if not gathered_files_list:
        raise CompareException(
            CompareException.NO_DUMP_FILE_ERROR, "Cannot get fx dump data, because the gathered_files_list is empty."
        )

    dump_data_with_token_id_list = []
    for gathered_files in gathered_files_list:
        dump_data_with_token_id = {}
        for token_id, file_list in gathered_files.items():
            cur_dump_data = {}
            for file_path in sorted(file_list):
                if not file_path.endswith("npy"):
                    continue
                file_name = os.path.basename(file_path)
                is_input = ".INPUT." in file_name
                cur_op_name = file_name.split(".INPUT." if is_input else ".OUTPUT.")[0]
                cur_op_map = cur_dump_data.get(cur_op_name, {})
                cur_op_map.setdefault("input" if is_input else "output", []).append(file_path)
                cur_dump_data[cur_op_name] = cur_op_map
            if len(cur_dump_data) > 0:
                dump_data_with_token_id[token_id - 1] = cur_dump_data  # For FX data, token starts from 1, while GE is 0
        dump_data_with_token_id_list.append(dump_data_with_token_id)
    return dump_data_with_token_id_list


def compare_single_data(golden_path, my_path, token_id=0, golden_data=None, my_data=None):
    data_info = BasicDataInfo(golden_path, my_path, token_id)
    return fill_row_data(data_info, loaded_my_data=my_data, loaded_golden_data=golden_data)


# Comparing GE with FX
def filter_valid_fx_desc_tensor_info(desc_key, desc_value):
    """Valid one like: 'attr': {'key': '_fx_tensor_name', 'value': {'s': 'add_1-aten.add.Tensor.OUTPUT.0'}}"""
    if not (desc_key == "attr" or desc_key.startswith("attr#")) or not isinstance(desc_value, dict):
        return False
    if desc_value.get("key", None) != "_fx_tensor_name" or not isinstance(desc_value.get("value", None), dict):
        return False
    if not isinstance(desc_value.get("value", {}).get("s", None), str):
        return False
    return True


def get_all_ops_from_fusion_op(op_name, graph_map_dict, ge_dump_data):
    all_ops = []
    while len(op_name) > 0:
        cur_op_name = find_longest_name(op_name, graph_map_dict, ge_dump_data, ge_dump_data)
        if cur_op_name is None or cur_op_name not in graph_map_dict:
            logger.debug(f"Failed parsing ge op name: {cur_op_name}.Compare manually if required.")
            break
        all_ops.append(cur_op_name)
        op_name = op_name[len(cur_op_name) :]
    return all_ops


def compare_ge_with_fx(graph_map, ge_dump_data, fx_dump_data, token_id=0):
    gathered_row_data = []
    graph_map_dict = {
        graph["op"]["name"]: graph["op"] for graph in graph_map if "op" in graph and "name" in graph["op"]
    }
    ge_dump_data = sort_ge_dump_data(ge_dump_data, graph_map)
    for op_name, my_path in ge_dump_data.items():
        all_ops = get_all_ops_from_fusion_op(op_name, graph_map_dict, ge_dump_data)
        if len(all_ops) == 1:
            op_info = graph_map_dict.get(all_ops[0])
            gathered_row_data.extend(compare_ge_with_fx_single_op(op_info, fx_dump_data, op_name, my_path, token_id))
        elif len(all_ops) > 1:
            first_op_info = graph_map_dict.get(all_ops[0])
            last_op_info = graph_map_dict.get(all_ops[-1])
            __args = [first_op_info, last_op_info, fx_dump_data, op_name, my_path, token_id]
            gathered_row_data.extend(compare_ge_with_fx_multiple_ops(*__args))
        else:
            op_type = os.path.basename(my_path).split(".")[0]
            if "Cast" in op_type or "TransData" in op_type:
                ge_inputs, ge_outputs = parse_torchair_dump_data(my_path)
                logger.debug(f"ge_inputs length: {len(ge_inputs)}")
                logger.debug(f"ge_outputs length:, {len(ge_outputs)}")
                gathered_row_data.extend(compare_specials_private_ops(ge_inputs, ge_outputs, token_id, my_path))

    return gathered_row_data


def compare_ge_with_fx_single_op(op_info, fx_dump_data, op_name, my_path, token_id=0):
    gathered_row_data = []
    for op_key, op_value in op_info.items():
        if not (op_key == "output_desc" or op_key.startswith("output_desc#")) or not isinstance(op_value, dict):
            continue
        for out_key, out_value in op_value.items():
            if not filter_valid_fx_desc_tensor_info(out_key, out_value):
                continue
            fx_tensor_name = out_value.get("value", {}).get("s", None)
            if fx_tensor_name.split(".")[-2] == "OUTPUT":
                fx_tensor_name = ".".join(fx_tensor_name.split(".")[:-2])
            if fx_tensor_name not in fx_dump_data:
                logger.warning(f"FX data missing, GE tensor name: {op_name}, FX tensor name: {fx_tensor_name}")
                continue

            ge_inputs, ge_outputs = parse_torchair_dump_data(my_path)
            fx_inputs = fx_dump_data.get(fx_tensor_name, {}).get("input", [])
            fx_outputs = fx_dump_data.get(fx_tensor_name, {}).get("output", [])
            logger.debug(f"ge_inputs length: {len(ge_inputs)}, fx_inputs length:, {len(fx_inputs)}")
            logger.debug(f"ge_outputs length: {len(ge_outputs)}, fx_outputs length:, {len(fx_outputs)}")
            gathered_row_data = compare_ops((fx_inputs, fx_outputs), (ge_inputs, ge_outputs), token_id, my_path)

    return gathered_row_data


def compare_ge_with_fx_multiple_ops(first_op_info, last_op_info, fx_dump_data, op_name, my_path, token_id):
    gathered_row_data = []
    gathered_row_data.extend(
        compare_ge_with_fx_multiple_ops_details(first_op_info, fx_dump_data, op_name, my_path, "input", token_id)
    )
    gathered_row_data.extend(
        compare_ge_with_fx_multiple_ops_details(last_op_info, fx_dump_data, op_name, my_path, "output", token_id)
    )
    return gathered_row_data


def compare_ge_with_fx_multiple_ops_details(op_info: dict, *args):
    fx_dump_data, op_name, my_path, input_or_output, token_id = args
    gathered_row_data = []
    for op_key, op_value in op_info.items():
        if not (op_key == "output_desc" or op_key.startswith("output_desc#")) or not isinstance(op_value, dict):
            continue
        for out_key, out_value in op_value.items():
            if not filter_valid_fx_desc_tensor_info(out_key, out_value):
                continue
            fx_tensor_name = out_value.get("value", {}).get("s", None)
            if fx_tensor_name.split(".")[-2] == "OUTPUT":
                fx_tensor_name = ".".join(fx_tensor_name.split(".")[:-2])
            if fx_tensor_name not in fx_dump_data:
                logger.warning(f"FX data missing, GE tensor name: {op_name}, FX tensor name: {fx_tensor_name}")
                continue
            ge_inputs, ge_outputs = parse_torchair_dump_data(my_path)
            fx_inputs_or_outputs = fx_dump_data.get(fx_tensor_name, {}).get(input_or_output, [])
            ge_input_or_output_path = ""
            ge_inputs_or_outputs = []
            if input_or_output == "input":
                logger.debug(f"ge_inputs length: {len(ge_inputs)}, fx_inputs length:, {len(fx_inputs_or_outputs)}")
                ge_input_or_output_path = "inputs"
                ge_inputs_or_outputs = ge_inputs
            elif input_or_output == "output":
                logger.debug(f"ge_outputs length: {len(ge_outputs)}, fx_outputs length:, {len(fx_inputs_or_outputs)}")
                ge_input_or_output_path = "outputs"
                ge_inputs_or_outputs = ge_outputs
            for cur_id, (fx_input_or_output, ge_input_or_output) in enumerate(
                zip(fx_inputs_or_outputs, ge_inputs_or_outputs)
            ):
                cur_ge_data = "{},{},{}".format(my_path, ge_input_or_output_path, cur_id)
                row_data = compare_single_data(fx_input_or_output, cur_ge_data, token_id, my_data=ge_input_or_output)
                gathered_row_data.append(row_data)

    return gathered_row_data


def compare_specials_private_ops(ge_inputs, ge_outputs, token_id, my_path):
    gathered_row_data = []
    for cur_id, (ge_input, ge_output) in enumerate(zip(ge_inputs, ge_outputs)):
        cur_ge_input_data = f"{my_path},inputs,{cur_id}"
        cur_ge_output_data = f"{my_path},outputs,{cur_id}"
        row_data = compare_single_data(cur_ge_input_data, cur_ge_output_data, token_id, ge_input, ge_output)
        gathered_row_data.append(row_data)

    return gathered_row_data


def compare_ops(fx_tuple, ge_tuple, token_id, my_path):
    gathered_row_data = []
    for cur_id, (fx_input, ge_input) in enumerate(zip(fx_tuple[0], ge_tuple[0])):
        cur_ge_data = f"{my_path},inputs,{cur_id}"
        row_data = compare_single_data(fx_input, cur_ge_data, token_id, my_data=ge_input)
        gathered_row_data.append(row_data)
    for cur_id, (fx_output, ge_output) in enumerate(zip(fx_tuple[1], ge_tuple[1])):
        cur_ge_data = f"{my_path},outputs,{cur_id}"
        row_data = compare_single_data(fx_output, cur_ge_data, token_id, my_data=ge_output)
        gathered_row_data.append(row_data)

    return gathered_row_data


# Comparing fused GE with GE
def get_all_op_input_names(op_info):
    inputs = [vv for kk, vv in op_info.items() if kk == "input" or kk.startswith("input#")]
    return [":".join(ii.split(":")[:-1]) for ii in inputs]


def find_longest_name(op_name, op_map, fused_ge_dump_data, ge_dump_data):
    if op_name in op_map:
        return op_name
    op_name_len = len(op_name)
    for idx in range(1, op_name_len):
        cur_op_name = op_name[:-idx]
        if cur_op_name in op_map:
            return cur_op_name
        if cur_op_name in fused_ge_dump_data or cur_op_name in ge_dump_data:
            return None  # op_name in dump data but not op_map, abandon
    return None


def gather_fused_op_data(fused_op_name, op_map, fused_ge_dump_data, ge_dump_data):
    gathered_input_names, gathered_inputs, gatherd_input_pathes, gathered_ops = [], [], [], []
    output_path, op_outputs = None, []
    while len(fused_op_name) > 0:
        cur_op_name = find_longest_name(fused_op_name, op_map, fused_ge_dump_data, ge_dump_data)
        if cur_op_name is None or cur_op_name not in op_map:
            logger.warning(f"Failed parsing fused op name: {fused_op_name}. Compare manually if required.")
            break
        cur_input_names = get_all_op_input_names(op_map[cur_op_name])

        if cur_op_name in ge_dump_data:
            cur_path = ge_dump_data[cur_op_name]
            op_inputs, op_outputs = parse_torchair_dump_data(cur_path)
            min_inputs_len = min(len(cur_input_names), len(op_inputs))
            cur_input_names, op_inputs = cur_input_names[:min_inputs_len], op_inputs[:min_inputs_len]
            input_pathes = [",".join([cur_path, "inputs", str(idx)]) for idx in range(min_inputs_len)]
            output_path = cur_path  # Till get the last op path
        else:
            logger.warning(
                f"No dump data for op: {cur_op_name}. Seldom should this happen. Input data matching may be incorrect."
            )
            empty_data = np.array([], dtype="float32")
            op_inputs = [empty_data] * len(cur_input_names)
            input_pathes = [""] * len(cur_input_names)

        gathered_input_names.extend(cur_input_names)
        gathered_ops.append(cur_op_name)
        gathered_inputs.extend(op_inputs)
        gatherd_input_pathes.extend(input_pathes)
        fused_op_name = fused_op_name[len(cur_op_name) :]

    filtered_input_names, filtered_inputs, filtered_input_pathes = [], [], []
    for input_name, inputs, input_path in zip(gathered_input_names, gathered_inputs, gatherd_input_pathes):
        if input_name not in gathered_ops:
            filtered_input_names.append(input_name)
            filtered_input_pathes.append(input_path)
            filtered_inputs.append(inputs)
    return (filtered_inputs, filtered_input_pathes), (op_outputs, output_path)  # op_outputs is just the last op output


def compare_ge_with_ge(graph_map, fused_ge_dump_data, ge_dump_data, token_id=0):
    graph_map_dict = {ii["op"]["name"]: ii["op"] for ii in graph_map if "op" in ii and "name" in ii["op"]}
    fused_ge_dump_data = sort_ge_dump_data(fused_ge_dump_data, graph_map)
    gathered_row_data = []
    for op_name, my_path in fused_ge_dump_data.items():
        is_fused_op = os.path.basename(my_path).startswith(FUSION_OP_TYPE)
        if is_fused_op:
            (golden_inputs, golden_input_pathes), (golden_outputs, golden_output_path) = gather_fused_op_data(
                op_name, graph_map_dict, fused_ge_dump_data, ge_dump_data
            )
        elif op_name in ge_dump_data:
            golden_path = ge_dump_data[op_name]
            golden_inputs, golden_outputs = parse_torchair_dump_data(golden_path)
            golden_input_pathes = [golden_path] * len(golden_inputs)
            golden_output_path = golden_path
        else:
            logger.warning(f"Golden data missing, My tensor name: {op_name}")
            continue

        my_inputs, my_outputs = parse_torchair_dump_data(my_path)
        logger.debug(f"golden_inputs length: {len(golden_inputs)}, my_inputs length:, {len(my_inputs)}")
        logger.debug(f"golden_outputs length: {len(golden_outputs)}, my_outputs length:, {len(my_outputs)}")

        for cur_id, (golden_input, my_input, golden_input_path) in enumerate(
            zip(golden_inputs, my_inputs, golden_input_pathes)
        ):
            cur_ge_data = f"{my_path},inputs,{cur_id}"
            if ",inputs," not in golden_output_path:
                golden_output_path = f"{golden_output_path},inputs,{cur_id}"
            row_data = compare_single_data(
                golden_input_path, cur_ge_data, token_id, golden_data=golden_input, my_data=my_input
            )
            gathered_row_data.append(row_data)
        for cur_id, (golden_output, my_output) in enumerate(zip(golden_outputs, my_outputs)):
            cur_ge_data = f"{my_path},outputs,{cur_id}"
            golden_output_path = f"{golden_output_path},outputs,{cur_id}"
            row_data = compare_single_data(
                golden_output_path, cur_ge_data, token_id, golden_data=golden_output, my_data=my_output
            )
            gathered_row_data.append(row_data)
    return gathered_row_data


def sort_ge_dump_data(dump_data, graph_map):
    graph_map_sort = {graph["op"]["name"]: id for id, graph in enumerate(graph_map)}
    sort_ops_list = sorted(dump_data, key=lambda x: graph_map_sort.get(x, -1))
    ge_dump_data = OrderedDict((op_name, dump_data[op_name]) for op_name in sort_ops_list)
    return ge_dump_data


def sort_by_timestamp(gathered_row_data):
    """
    gathered_row_data为保存比对结果的列表，列表里每个元素都是一个字典
    1. 取每个字典的'my_data_path'字段x['my_data_path']，该字段的值如：OpType.OpName.12.7.1734070081497686,inputs,0
    2. 按'.'分隔字符串x['my_data_path'].split('.') —> ['OpType', 'OpName', '12', '7', '1734070081497686,inputs,0']
    3. x['my_data_path'].split('.')[-1]取最后一个元素为'1734070081497686,inputs,0'，该值包含时间戳（1734070081497686）和输入输出信息（inputs,0）
    4. 按'1734070081497686,inputs,0'对比对结果重新排序得到sorted_gathered_row_data列表
    5. 返回排序后的结果
    """
    sorted_gathered_row_data = sorted(gathered_row_data, key=lambda x: x['my_data_path'].split('.')[-1])
    return sorted_gathered_row_data


# Main entrance
def acc_compare(golden_path, my_path, output_path='./', rank_id=None, rank_info_existed=False):
    set_msaccucmp_path_from_cann()

    if not get_torchair_ge_graph_path(my_path):
        raise CompareException(
            CompareException.INVALID_PATH_ERROR,
            "Can not get ge graph, Please check whether the input path contains graph.",
        )

    if rank_info_existed:
        if rank_id is None:
            golden_data_ranks = set()
            my_data_ranks = set()

            for subdir in os.listdir(golden_path):
                rank_id = get_rank_id_from_torchair_data(subdir)
                if os.path.isdir(os.path.join(golden_path, subdir)) and rank_id != -1:
                    golden_data_ranks.add(rank_id)

            for subdir in os.listdir(my_path):
                rank_id = get_rank_id_from_torchair_data(subdir)
                if os.path.isdir(os.path.join(my_path, subdir)) and rank_id != -1:
                    my_data_ranks.add(rank_id)

            compared_ranks = list(golden_data_ranks & my_data_ranks)
            if not compared_ranks:
                raise CompareException(
                    CompareException.INVALID_PATH_ERROR, "No common rank data in golden_path and my_path."
                )
        else:
            compared_ranks = [rank_id]
    else:
        compared_ranks = [-1]

    args = [(golden_path, my_path, output_path, rid) for rid in compared_ranks]
    # If only a single rank needs comparison, run synchronously so that any exceptions
    # raised inside `acc_compare_once` are propagated directly.
    if len(args) == 1:
        acc_compare_once(args[0])
    else:
        # Use multiprocessing for multiple ranks but make sure to propagate exceptions
        with Pool(min(len(args), int(cpu_count() * 1.3))) as processes_pool:
            async_results = [processes_pool.apply_async(save_compare_once, (arg,)) for arg in args]
            processes_pool.close()
            # Ensure that exceptions in worker processes are not silenced. Calling `get()` will
            # re-raise any exception occurred in the worker process in the main process.
            for res in async_results:
                res.get()
            processes_pool.join()

    return output_path


def save_compare_once(args):
    try:
        return acc_compare_once(args)
    except Exception as e:
        error_msg = f"Error in acc_compare_once: {str(e)}"
        raise ValueError(error_msg) from e


def compare_torchair_mode(args):
    """
    Entry point used by the CLI to trigger torchair accuracy compare.
    """
    my_path = os.path.realpath(args.target_path)
    golden_path = os.path.realpath(args.golden_path)
    _validate_read_path(my_path)
    _validate_read_path(golden_path)

    rank_arg = getattr(args, "rank", None)
    rank_id = None
    if rank_arg is not None:
        rank_str = str(rank_arg).strip()
        if not rank_str.isdigit():
            logger.error("Argument --rank only supports a single integer when mode=='torchair'.")
            raise CompareException(CompareException.INVALID_PARAM_ERROR, "Invalid rank parameter for torchair mode.")
        rank_id = int(rank_str)

    rank_info_existed = _has_rank_directory(my_path)
    if not rank_info_existed and rank_id is not None:
        logger.warning('The directory structure of torchair data is old, the rank parameter will not take effect.')
    logger.info(f"[compare_torchair] start comparing, golden_path: {golden_path}, target_path: {my_path}")
    return acc_compare(golden_path, my_path, args.output_path, rank_id, rank_info_existed)


def acc_compare_once(*args):
    dir_of_golden_path, dir_of_my_path, output_path, rank_id = args[0]
    if rank_id != -1:
        subdirs = []
        for subdir in os.listdir(dir_of_golden_path):
            is_dir = os.path.isdir(os.path.join(dir_of_golden_path, subdir))
            if is_dir and subdir.startswith('worldsize') and subdir.endswith(f'rank{rank_id}'):
                subdirs.append(subdir)
        if not subdirs:
            raise CompareException(CompareException.INVALID_PATH_ERROR, f'Can not get golden data in rank {rank_id}')
        golden_path = os.path.join(dir_of_golden_path, subdirs[-1])

        subdirs = []
        for subdir in os.listdir(dir_of_my_path):
            is_dir = os.path.isdir(os.path.join(dir_of_my_path, subdir))
            if is_dir and subdir.startswith('worldsize') and subdir.endswith(f'rank{rank_id}'):
                subdirs.append(subdir)
        if not subdirs:
            raise CompareException(CompareException.INVALID_PATH_ERROR, f'Can not get my data in rank {rank_id}')
        my_path = os.path.join(dir_of_my_path, subdirs[-1])
    else:
        golden_path = dir_of_golden_path
        my_path = dir_of_my_path

    ge_graph_path = get_torchair_ge_graph_path(dir_of_my_path, rank_id)

    if not ge_graph_path:
        raise CompareException(
            CompareException.INVALID_PATH_ERROR,
            "Can not get ge graph, Please check whether the input path contains graph.",
        )

    logger.info(f"[compare_torchair], golden_path: {golden_path}, my_path: {my_path}, ge_graph_path: {ge_graph_path}")

    graph_map_list = []
    for path in ge_graph_path:
        graph_map_list.append(parse_pbtxt_to_dict(path))

    my_dump_data_list = init_ge_dump_data_from_bin_path(my_path)

    is_golden_fx = get_torchair_ge_graph_path(dir_of_golden_path) is None
    if is_golden_fx:
        logger.info("Comparing GE with FX")
        golden_dump_data_list = init_fx_dump_data_from_path(golden_path, rank_id != -1)
    else:
        logger.info("Comparing GE with GE")
        golden_dump_data_list = init_ge_dump_data_from_bin_path(golden_path)

    graph_map_list_len = len(graph_map_list)
    for i in range(graph_map_list_len):
        logger.info(f"All token ids in my_dump_data: {my_dump_data_list[i].keys()}")
        logger.info(f"All token ids in golden_dump_data: {golden_dump_data_list[i].keys()}")
        graph_map = graph_map_list[i]
        my_dump_data = my_dump_data_list[i]
        golden_dump_data = golden_dump_data_list[i]

        gathered_row_data = []
        for token_id in my_dump_data:
            if token_id not in golden_dump_data:
                logger.warning(f"My token_id {token_id} not found in golden dump data")
                continue
            logger.info(f"Comparing token_id: {token_id}")
            if is_golden_fx:
                row_data = compare_ge_with_fx(graph_map, my_dump_data[token_id], golden_dump_data[token_id], token_id)
            else:
                row_data = compare_ge_with_ge(graph_map, my_dump_data[token_id], golden_dump_data[token_id], token_id)
            gathered_row_data.extend(row_data)
        sorted_gathered_row_data = sort_by_timestamp(gathered_row_data)
        save_compare_result_to_csv(sorted_gathered_row_data, output_path, rank_id=rank_id)
