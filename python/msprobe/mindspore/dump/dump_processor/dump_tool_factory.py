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

from msprobe.mindspore.common.const import Const
from msprobe.core.common.log import logger
from msprobe.mindspore.common.utils import is_graph_mode_cell_dump_allowed
from msprobe.mindspore.dump.debugger.debugger_config import DebuggerConfig
from msprobe.mindspore.dump.dump_processor.kernel_kbyk_dump import KernelKbykDump
from msprobe.mindspore.dump.dump_processor.graph_mode_cell_dump import GraphModeCellDump


class DumpToolFactory:
    tools = {
        Const.CELL: {Const.GRAPH_KBYK_MODE: GraphModeCellDump, Const.GRAPH_GE_MODE: None, Const.PYNATIVE_MODE: None},
        Const.API: {Const.GRAPH_KBYK_MODE: None, Const.GRAPH_GE_MODE: None, Const.PYNATIVE_MODE: None},
        Const.KERNEL: {
            Const.GRAPH_KBYK_MODE: KernelKbykDump,
            Const.GRAPH_GE_MODE: None,
            Const.PYNATIVE_MODE: KernelKbykDump,
        },
    }

    @staticmethod
    def create(config: DebuggerConfig, model=None):
        if config.level == Const.CELL:
            if not is_graph_mode_cell_dump_allowed(config):
                raise Exception("Cell dump is not supported in graph mode.")  # pylint: disable=W0719
            if len(config.data_mode) != 1 or config.data_mode[0] not in Const.GRAPH_CELL_DUMP_DATA_MODE_LIST:
                raise Exception("data_mode must be one of all, forward, backward.")  # pylint: disable=W0719
        else:
            if len(config.data_mode) != 1 or config.data_mode[0] not in Const.GRAPH_DATA_MODE_LIST:
                raise Exception("data_mode must be one of all, input, output.")  # pylint: disable=W0719
        if config.level == Const.KERNEL:
            return (KernelKbykDump(config),)
        tool = DumpToolFactory.tools.get(config.level)
        if not tool:
            raise Exception("Valid level is needed.")  # pylint: disable=W0719
        tool = tool.get(config.execution_mode)
        if not tool:
            logger.error(
                f"Data dump is not supported in {config.execution_mode} mode when dump level is {config.level}."
            )
            raise ValueError
        return tool(config, model) if tool == GraphModeCellDump else tool(config)
