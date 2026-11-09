# msProbe PyTorch Minimal Quick Start

<br>

## 1. Overview

Run one training step of a two-layer fully connected network on Linux CPU to install, instrument, dump, compare, and inspect results. No CANN, `torch_npu`, GPU, model weights, or dataset downloads are required. For cross-device debugging, see [msProbe PyTorch Quick Start](pytorch_quick_start.md).

## 2. Procedure

### 2.1 Environment Preparation

This example pins `mindstudio-probe==26.2.0` and PyTorch `2.7.1`. Use an isolated Python 3.11/3.12 environment. Run the following commands in Linux Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install mindstudio-probe==26.2.0
git clone --depth 1 https://gitcode.com/Ascend/msprobe.git
cd msprobe/examples/quick_start/minimal
python -m pip show mindstudio-probe torch
msprobe --help
```

If you already have a checkout, go to `examples/quick_start/minimal`. The package name is `mindstudio-probe`; both the Python import name and command are `msprobe`. For NPU environment compatibility, see [msProbe Installation Guide](../install_guide/msprobe_install_guide.md).

### 2.2 Configuration and Instrumentation

[config.json](../../../examples/quick_start/minimal/config.json) uses `task=tensor` to save complete Tensors, `level=L1` to collect APIs, `step=[0]` to collect only the first iteration, and `async_dump=false` for synchronous writes. The small network allows you to try comparison of actual Tensor values directly.

The key instrumentation in [train.py](../../../examples/quick_start/minimal/train.py) is:

```python
debugger = PrecisionDebugger(config_path="config.json", dump_path="./dump")
debugger.start(model)
# Forward pass, loss.backward(), and optimizer.step()
debugger.stop()
debugger.step()  # Call once at the end of each training iteration.
```

`train.py` loads the configuration next to the script. The `--dump-path` argument overrides the output directory in the configuration. Both runs use the same random seed, model initialization, and inputs. Before collection starts, `--perturb` adds `0.1` to the second Linear layer's weights to introduce a difference deliberately.

### 2.3 Run and Inspect Dump Data

Use a new directory each time to keep the results separate from previous dumps:

```bash
RUN_DIR=$(mktemp -d "$PWD/run.XXXXXX")
python train.py --dump-path "$RUN_DIR/golden"
python train.py --dump-path "$RUN_DIR/target" --perturb
GOLDEN_JSON=$(find "$RUN_DIR/golden/step0" -name dump.json -print -quit)
TARGET_JSON=$(find "$RUN_DIR/target/step0" -name dump.json -print -quit)
test -n "$GOLDEN_JSON" && test -n "$TARGET_JSON"
find "$RUN_DIR" -name dump.json -o -name stack.json -o -name '*.pt'
```

Single-process output is stored in `step0/proc{pid}/`: `dump.json` records API and Tensor metadata, `stack.json` records call stacks, and `dump_tensor_data/` contains `.pt` files. The two processes can have different PIDs; do not hardcode `proc{pid}`. Multi-device runs typically use `rank{id}` directories.

### 2.4 Compare Precision

Continue in the same terminal:

```bash
msprobe compare -tp "$TARGET_JSON" -gp "$GOLDEN_JSON" -o "$RUN_DIR/compare"
find "$RUN_DIR/compare" -name '*.csv'
```

`-tp` specifies the data to check, and `-gp` specifies the reference data. The default output is `compare_result_{timestamp}.csv`. Keep both complete `step0/proc{pid}` directories: comparison of actual Tensor values also requires their `dump_tensor_data` files.

### 2.5 Analyze Comparison Results

1. Check `Err_message` first for unmatched APIs, inconsistent shapes or data types, and missing Tensor files.
2. Inspect `Result`, `Cosine`, and `MaxAbsErr`. In this example, the second Linear layer's activation inputs should match. Its weight parameters and forward output start to differ, affecting the loss and backward data that follow.
3. Use the API name, `NPU_Stack_Info`, or the target's `stack.json` to locate the call in `train.py` and distinguish where a difference starts from where it propagates. CPU results retain the column names `NPU Name` and `Bench Name`, representing the target and reference sides respectively.

In this example, `MaxAbsErr` is `0` for `Functional.linear.1.forward.input.0`, approximately `0.1` for the weights in `input.1`, and approximately `0.115` for `output.0`. `Result` applies screening rules: a row marked `pass` can still have nonzero error, so inspect the numerical indicators as well.

You can run the target again without `--perturb` and compare using a new output directory to check identical models and inputs. The difference here is injected deliberately and does not establish a CPU/NPU/GPU operator defect. Matching statistical summaries also does not establish elementwise Tensor equality.

For indicator definitions and thresholds, see [PyTorch Precision Comparison](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md). For other single-device, multi-device, and dump scenarios, see [msProbe Examples](../../../examples/README_EN.md).
