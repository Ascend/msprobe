# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

import sys
from pathlib import Path

import pytest


def _ensure_repo_root_on_syspath():
    """
    让测试可以稳定 import 顶层目录（如 plugins/）。
    通过向上查找 setup.py 来定位仓库根目录。
    """
    cur = Path(__file__).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "setup.py").exists():
            sys.path.insert(0, str(parent))
            return


_ensure_repo_root_on_syspath()


from plugins.tb_graph_ascend.hierarchy_plugin.server.app.utils.global_state import GraphState


@pytest.fixture(autouse=True)
def _reset_graph_state():
    """
    确保每个用例互相隔离，避免 GraphState 的全局缓存导致测试串扰。
    """
    GraphState.reset_global_state()
    yield
    GraphState.reset_global_state()
