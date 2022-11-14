#!/bin/bash
#Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
# ================================================================================

# get script path
script_path=$(readlink -f "$0")
route=$(dirname "$script_path")

# run analyse
PYTHONPATH="$route":$PYTHONPATH python3 "$route"/pytorch_analyse.py "$@"
