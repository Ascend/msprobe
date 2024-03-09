#!/bin/bash
#Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
# ================================================================================

# get script path
script_path=$(readlink -f "$0")
route=$(dirname "$script_path")

# run transfer
PYTHONPATH="$route":$PYTHONPATH python3 "$route"/ms_fmk_transplt.py "$@"
