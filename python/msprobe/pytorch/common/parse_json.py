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

from msprobe.core.common.exceptions import ParseJsonException
from msprobe.core.common.file_utils import load_json
from msprobe.core.common.log import logger


def parse_json_info_forward_backward(json_path):
    dump_json = load_json(json_path)

    real_data_path = dump_json.get("dump_data_dir")
    dump_data = dump_json.get("data")
    if dump_data is None:
        raise ParseJsonException(
            ParseJsonException.InvalidDumpJson, "something wrong with dump, no data found in dump.json"
        )
    if not dump_data:
        logger.warning("data field is empty, no overflow data found.")

    forward_data = {}
    backward_data = {}
    for data_name, data_item in dump_data.items():
        if "Module" in data_name:
            continue
        name_struct = data_name.split('.')
        if name_struct[-1] == "forward":
            forward_data.update({'.'.join(name_struct[:-1]): data_item})
        elif name_struct[-1] == "backward":
            backward_data.update({'.'.join(name_struct[:-1]): data_item})
        else:
            raise ParseJsonException(ParseJsonException.UnexpectedNameStruct, f"{data_name} in file {json_path}.")

    return forward_data, backward_data, real_data_path
