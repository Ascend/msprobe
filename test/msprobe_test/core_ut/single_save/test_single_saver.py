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
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from msprobe.core.single_save.single_saver import SingleSave


def mock_tensor(shape):
    class T:
        def __init__(self):
            self._data = np.random.rand(*shape)

        def numpy(self):
            return self._data

        @property
        def shape(self):
            return self._data.shape
    return T()


@pytest.fixture()
def dump_env(tmp_path):
    return str(tmp_path / "mock_dump")


def setup_fmk(Mock, rank=0):
    Mock.is_tensor.side_effect = lambda x: hasattr(x, "shape")
    Mock.tensor_max.side_effect = lambda x: float(np.max(x.numpy()))
    Mock.tensor_min.side_effect = lambda x: float(np.min(x.numpy()))
    Mock.tensor_mean.side_effect = lambda x: float(np.mean(x.numpy()))
    Mock.tensor_norm.side_effect = lambda x: float(np.linalg.norm(x.numpy()))
    Mock.get_rank_id.return_value = rank
    Mock.save_tensor = MagicMock()
    Mock.set_fmk = MagicMock()


#
# ----- TESTS -----
#

@patch("msprobe.core.single_save.single_saver.save_json")
@patch("msprobe.core.single_save.single_saver.create_directory")
@patch("msprobe.core.single_save.single_saver.FmkAdp")
def test_nested_all(MockFmkAdp, mock_mkdir, mock_save_json, dump_env):
    setup_fmk(MockFmkAdp)
    ss = SingleSave(dump_env)

    data = {
        "nested": [
            mock_tensor((2, 2)),
            ]
    }

    ss.save(data)

    # save_json 调用 1 次，save_tensor 调用 4 次
    assert mock_save_json.call_count == 1
    assert MockFmkAdp.save_tensor.call_count == 1


@patch("msprobe.core.single_save.single_saver.save_json")
@patch("msprobe.core.single_save.single_saver.create_directory")
@patch("msprobe.core.single_save.single_saver.FmkAdp")
def test_micro_step_flow(MockFmkAdp, mock_mkdir, mock_save_json, dump_env):
    setup_fmk(MockFmkAdp)
    ss = SingleSave(dump_env)

    ss.save({"k": mock_tensor((1,))})
    ss.save({"k": mock_tensor((1,))})

    # step 后 tag_count 重置
    ss.step()
    ss.save({"k": mock_tensor((1,))})

    # save_json 三次
    assert mock_save_json.call_count == 3

    # 验证路径包含 micro_step0 → micro_step1 → micro_step0
    args_list = [args for args, kw in mock_save_json.call_args_list]
    paths = [a[0] for a in args_list]
    assert "micro_step0" in paths[0]
    assert "micro_step1" in paths[1]
    assert "step1" in paths[2] and "micro_step0" in paths[2]


@patch("msprobe.core.single_save.single_saver.save_json")
@patch("msprobe.core.single_save.single_saver.create_directory")
@patch("msprobe.core.single_save.single_saver.FmkAdp")
def test_save_ex_with_manual_micro(MockFmkAdp, mock_mkdir, mock_save_json, dump_env):
    setup_fmk(MockFmkAdp)
    ss = SingleSave(dump_env)

    ss.save_ex({"abc": mock_tensor((2,))}, micro_batch=5)
    path = mock_save_json.call_args[0][0]

    assert "step1" in path
    assert "rank0" in path
    assert "micro_step5" in path


@patch("msprobe.core.single_save.single_saver.save_json")
@patch("msprobe.core.single_save.single_saver.create_directory")
@patch("msprobe.core.single_save.single_saver.FmkAdp")
def test_rank_change(MockFmkAdp, mock_mkdir, mock_save_json, dump_env):
    setup_fmk(MockFmkAdp, rank=0)
    ss = SingleSave(dump_env)

    ss.save({"rk": mock_tensor((1,))})
    path = mock_save_json.call_args[0][0]
    assert "rank0" in path


@patch("msprobe.core.single_save.single_saver.logger.warning")
@patch("msprobe.core.single_save.single_saver.create_directory")
@patch("msprobe.core.single_save.single_saver.FmkAdp")
def test_save_ex_type_error(MockFmkAdp, mock_mkdir, mock_log_warn, dump_env):
    setup_fmk(MockFmkAdp)
    ss = SingleSave(dump_env)

    ss.save_ex("not dict")
    mock_log_warn.assert_called()


@patch("msprobe.core.single_save.single_saver.save_json")
@patch("msprobe.core.single_save.single_saver.create_directory")
@patch("msprobe.core.single_save.single_saver.FmkAdp")
def test_numeric_stats(MockFmkAdp, mock_mkdir, mock_save_json, dump_env):
    setup_fmk(MockFmkAdp)
    ss = SingleSave(dump_env)

    t = mock_tensor((2, 2))
    arr = t.numpy()

    ss.save({"stat": t})

    # 捕获 save_json 内容
    (path, result_dict), _ = mock_save_json.call_args
    v = result_dict["data"]

    assert v["shape"] == [2, 2]
    assert v["max"] == np.max(arr)
    assert v["min"] == np.min(arr)
    assert v["mean"] == np.mean(arr)
    assert v["norm"] == np.linalg.norm(arr)


def test_singleton(dump_env):
    ss1 = SingleSave(dump_env)
    ss2 = SingleSave(dump_env)
    assert ss1 is ss2