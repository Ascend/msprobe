#!/bin/bash

# get script path
script_path=$(readlink -f "$0")
route=$(dirname "$script_path")

# run pytorch_gpu2npu
PYTHONPATH="$route":$PYTHONPATH python3 "$route"/ms_fmk_transplt.py "$@"
