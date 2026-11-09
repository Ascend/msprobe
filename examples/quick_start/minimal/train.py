# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
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

"""One CPU training step for mindstudio-probe 26.2.0 and PyTorch 2.7.1."""

import argparse
from pathlib import Path

import torch
from torch import nn

from msprobe.pytorch import PrecisionDebugger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-path", required=True)
    parser.add_argument("--perturb", action="store_true", help="change the second Linear layer before collection")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    inputs = torch.randn(2, 4)
    targets = torch.randn(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if args.perturb:
        with torch.no_grad():
            model[2].weight.add_(0.1)

    debugger = PrecisionDebugger(config_path=str(Path(__file__).with_name("config.json")), dump_path=args.dump_path)
    debugger.start(model)
    optimizer.zero_grad()
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    debugger.stop()
    debugger.step()
    print(f"loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
