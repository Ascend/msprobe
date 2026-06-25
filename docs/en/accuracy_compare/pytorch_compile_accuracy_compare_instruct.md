# Compile Accuracy Comparison in PyTorch

## Overview

`PrecisionChecker` is a tool for **per-module accuracy comparison between the eager mode and the compile mode** of a PyTorch model. After enabling `torch.compile` for a model, if you encounter issues such as loss fluctuation, degraded convergence, or inconsistent inference outputs, this tool can help you locate the source of the difference module by module.

The tool works as follows: modules to be checked are marked with `wrap`, executed in both eager and compile modes, and their forward outputs and input gradients are compared one by one. This identifies the module where the difference occurs, the type of difference (forward or backward), and the magnitude of the error.

The tool interface follows the usage style of FSDP2 `fully_shard`, supporting check scopes specified by module type, module level, or individual module.

## Mode Selection

The tool provides two comparison modes: single-pass and two-pass. Selecting the correct mode is a prerequisite for using the tool. First-time users are advised to choose according to the following table:

| Scenario | Recommended Mode | Entry API |
| --- | --- | --- |
| Training (with `loss.backward()`) | single-pass (default) | `install()` + `collect()` |
| Inference / eval (forward only, no backward) | two-pass | `compare()` |
| Multi-card distributed training such as FSDP2 | single-pass (required) | `install()` + `collect()` |
| Need real whole-network `loss_eager` vs `loss_compiled` comparison | two-pass | `compare()` |
| Model does not support `deepcopy` (e.g., after FSDP2 sharding) | single-pass (required) | `install()` + `collect()` |

### single-pass Recommended for Training

single-pass is the default mode. It does not copy the model, nor does it run a separate eager whole-network pass. Instead, it integrates directly into the existing training flow: within the forward hook of a compiled submodule, it re-executes the uncompiled `_orig_mod` with the same input and compares the eager submodule output against the compile output.

single-pass is suitable for training scenarios for the following reasons:

- Low integration cost. There is no need to modify the existing `forward → loss → backward` flow. You only need to call `install()` before the training step and `collect(loss)` after it.
- It does not alter the optimizer or model state, avoiding state divergence caused by copying the model and then training.
- It is compatible with distributed wrappers such as FSDP2. Such models prohibit `deepcopy`, and single-pass does not perform deepcopy, making it the only available mode in multi-card training scenarios.

Note: single-pass does not run the eager whole model separately, so it does not compute the whole-network eager loss. The `loss_eager` in the report is `NaN`, which is expected behavior (see [About loss_eager=NaN in single-pass](#about-loss_eagernan-in-single-pass)). Accuracy is determined by per-module forward and backward comparison results, not by the whole-network loss.

### two-pass Recommended for Inference

Inference scenarios have no backward computation or optimizer state, and there is no concern about copying the model polluting the training state. Therefore, two-pass is more suitable:

- two-pass runs the eager model and the compiled model completely once each, providing real `loss_eager`, `loss_compiled`, and `loss_diff`, making it easy to determine whether the whole network is affected by compile.
- The inference pipeline is typically an "input-output" pattern. The "run each of the two pipelines once and then compare" semantics of two-pass is intuitive, and the result is complete (forward input, forward output, and gradient can all be compared).

Constraint: two-pass internally performs `deepcopy` on the model to construct the eager and compiled copies, so it does not support models that prohibit deepcopy, such as FSDP2. Regular single-card inference is not affected by this constraint.

> Summary: Use single-pass (`install` + `collect`) for training scenarios, and two-pass (`compare`) for inference scenarios.

## Preparations

**Installation**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

- Only PyTorch scenarios are supported.
- The tool depends on the `torch.compile` capability. Ensure that the current PyTorch version supports `torch.compile`.
- `install()` supports only the `single_pass=True` scenario.
- two-pass (`single_pass=False`) performs `deepcopy` on the model and does not support models that prohibit deepcopy, such as FSDP2.
- `dump_graphs` depends on the PyTorch Dynamo internal interface. Changes in the PyTorch version may affect the graph dump behavior.

## Quick Start

The following two examples correspond to the two most common scenarios in the mode selection table, and each can run independently.

### Training Scenario: single-pass Integration

A training flow usually already contains a `forward → loss → backward` training step. The single-pass integration approach is to call `install()` and `collect()` around the training step, without modifying the training step itself:

```python
import torch
import torch.nn as nn

from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.layers(x)


model = ToyModel()
x = torch.randn(2, 8)
target = torch.randn(2, 4)
loss_fn = nn.MSELoss()


def run_step(model):
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    return loss


checker = PrecisionChecker()                 # single_pass=True is the default
checker.wrap_by_policy(model, (nn.Linear,))  # mark modules to compare
checker.install(model)                       # replace marked modules with compiled versions in place

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path="precision_report.csv")
```

Notes:

- `install()` must be called before `run_step()`. It replaces the modules marked by `wrap` with compiled versions in place.
- `collect(loss)` must be called after `run_step()`. It organizes the per-module comparison results collected in this step.
- The `loss_eager` in the report is `NaN`, which is expected behavior. Accuracy is determined by the per-module `FORWARD OUTPUT` and `BACKWARD`.

### Inference Scenario: two-pass Integration

Inference scenarios have forward computation only. Using two-pass, the tool runs the eager and compiled models completely once each and directly compares the whole-network outputs:

```python
import torch
import torch.nn as nn

from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker

model = build_model()          # inference model
input_ids = make_input()       # inference input


def infer_step(model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        # two-pass requires a scalar as the comparison anchor. Inference has no loss,
        # so a statistic of the output (such as mean) can be used; it is not involved in backward.
        return logits.mean()


checker = PrecisionChecker(single_pass=False)   # use two-pass for inference
checker.wrap_by_policy(model, (nn.Linear,))

result = checker.compare(infer_step, model)     # runs both eager and compiled pipelines internally
checker.report(result, csv_path="precision_report.csv")
```

Notes:

- Inference scenarios use `compare()`, without calling `install()`.
- two-pass provides real `loss_eager` and `loss_compiled` (here `logits.mean()`), allowing you to directly determine whether the whole network is consistent.
- If the inference model uses a wrapper that prohibits deepcopy, such as FSDP2, use the single-pass `install()` + `collect()` path instead.

## Feature Description

The following table lists the complete features of `PrecisionChecker`:

| Feature | Description | Supported Scenario |
| --- | --- | --- |
| [Constructor Configuration](#constructor-configuration) | Configure backend, threshold, graph dump, autocast, input capture, and check mode | PyTorch |
| [Specifying Modules to Check](#specifying-modules-to-check) | Use `wrap`, `wrap_by_policy`, and `wrap_all_children` to specify modules participating in compile and accuracy comparison | PyTorch |
| [Skipping Modules](#skipping-modules) | Use `ignore` and `ignore_by_policy` to skip modules of no interest | PyTorch |
| [single-pass Mode](#single-pass-mode) | Compile wrapped submodules in place and complete a fast comparison via hooks | PyTorch |
| [two-pass Mode](#two-pass-mode) | Run the eager model and the compiled model separately, comparing loss, inputs, outputs, and gradients | PyTorch |
| [autocast Accuracy Check](#autocast-accuracy-check) | Enable `torch.autocast` for wrapped modules to check mixed-precision scenarios | PyTorch |
| [graph dump](#graph-dump) | Save the graph code captured by PyTorch Dynamo | PyTorch |
| [Report Description](#report-description) | Print the stdout report and optionally generate a CSV report (`csv_path` parameter) | PyTorch |
| [Multi-card FSDP2 Accuracy Check](#multi-card-fsdp2-accuracy-check) | Use single-pass for per-module comparison under FSDP2 sharding, with support for multi-card CSV reports | PyTorch |
| [Inference Mode Support](#inference-mode-support) | Run forward only to check the consistency of the inference pipeline before and after compile | PyTorch |

### Constructor Configuration

A tool initialization example is as follows:

```python
checker = PrecisionChecker(
    backend="aot_eager",
    threshold=1e-4,
    dump_graphs=False,
    graph_dir="./graph_dump",
    cast_dtype=None,
    capture_input=True,
    single_pass=True,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | `str` | `"aot_eager"` | The backend used by `torch.compile`. |
| `threshold` | `float` | `1e-4` | Reserved accuracy threshold parameter. Currently the determination is actually made via `torch.allclose(a, b, atol=1e-4, rtol=1e-3)`. |
| `dump_graphs` | `bool` | `False` | Whether to write the graphs captured at compile time to `graph_dir`. |
| `graph_dir` | `str` | `"./graph_dump"` | The graph dump output directory. |
| `cast_dtype` | `torch.dtype` | `None` | Enable autocast for wrapped modules, similar to a mixed precision policy. |
| `capture_input` | `bool` | `True` | Whether to capture module forward input, used to distinguish "the input already has a difference" from "the module computation produces a difference". |
| `single_pass` | `bool` | `True` | Whether to use the single-pass comparison logic. Set to `False` to use two-pass comparison. |

### Specifying Modules to Check

`wrap` is used to mark modules participating in compile and accuracy comparison. Modules not wrapped by `wrap` are not handled as independent compile units.

```python
# Manually mark a single module
checker.wrap(model.layers[3].self_attn)

# Mark modules by type in batch, similar to ModuleWrapPolicy
checker.wrap_by_policy(model, (Qwen2Block,))
checker.wrap_by_policy(model, (Qwen2Attention, Qwen2MLP))

# Automatically wrap submodules to a specified depth
checker.wrap_all_children(model)          # depth=1, default
checker.wrap_all_children(model, depth=0) # wrap direct submodules
checker.wrap_all_children(model, depth=2) # wrap deeper levels

# Wrap the entire model
checker.wrap(model)
```

Notes:

- `install()` replaces the marked submodules in place.
- If the root model itself is marked, the return value of `torch.compile(model)` cannot be written back to the variable held by the caller. It is recommended to use `compare()` directly, or to mark submodules under the root model.
- `wrap_all_children` recursively traverses model submodules and automatically skips container modules such as `ModuleList`, `ModuleDict`, and `Sequential`.
- For large models, it is recommended to start checking at a coarser granularity and then gradually split into smaller modules.
- If a module is already wrapped, its internal submodules are not handled again as top-level compile targets.

### Skipping Modules

`ignore` is used to skip the accuracy comparison of specified modules. A module that is ignored still participates in model execution but is excluded from report determination.

```python
checker.wrap_by_policy(model, (Qwen2Block,))

# Skip a specified module
checker.ignore(model.layers[0])

# Skip modules by type in batch
checker.ignore_by_policy(model, (torch.nn.Dropout,))
```

Skipped modules are displayed as `IGNORED` in the report.

### single-pass Mode

`single_pass=True` is the default mode. It is recommended to use `install()` and `collect()` to integrate into an existing training flow:

```python
checker = PrecisionChecker()
checker.wrap_all_children(model)
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result)
```

single-pass can also be used via `compare()`:

```python
checker = PrecisionChecker()
checker.wrap_by_policy(model, (Qwen2Block,))

result = checker.compare(run_step, model)
checker.report(result)
```

For finer-grained comparison, split the wrap into finer modules:

```python
checker.wrap_by_policy(model, (Qwen2Attention, Qwen2MLP))
```

#### About loss_eager=NaN in single-pass

In single-pass mode, the tool does not run the complete eager model separately and therefore cannot provide a real `loss_eager`. The `loss_eager=NaN` shown in the report is expected behavior, not a tool defect.

In single-pass, accuracy is determined by per-module comparison: within the forward hook of each compiled submodule, the tool re-executes `_orig_mod` (the uncompiled version) with the same input and compares the eager submodule output against the compile submodule output. The advantage of this approach is that it does not need to copy the model or start a separate training pipeline; the cost is that the whole-network eager loss cannot be obtained.

If you need a real comparison of `loss_eager` and `loss_compiled`, use two-pass mode (`single_pass=False`), which runs the complete eager and compiled pipelines separately.

### two-pass Mode

two-pass is enabled when `single_pass=False`. The tool constructs an eager model and a compiled model, runs the same `run_step` for each, and then compares the inputs, outputs, and gradients collected by the hooks on both sides.

```python
checker = PrecisionChecker(single_pass=False)
checker.wrap_by_policy(model, (Qwen2Block,))

result = checker.compare(run_step, model)
checker.report(result)
```

two-pass mode compares `loss_eager` and `loss_compiled`, which can be used to determine whether the whole-network compile pipeline introduces a loss difference.

Constraint: two-pass internally performs `deepcopy` on the model to construct the eager and compiled copies, so it does not support models that prohibit deepcopy, such as FSDP2. If the model already uses FSDP2 wrapping, use single-pass mode instead.

### autocast Accuracy Check

After `cast_dtype` is set, the tool adds a `_CastWrapper` to the wrapped modules and uses `torch.autocast` in the forward pass. This is used to verify whether wrapped modules have accuracy differences in automatic mixed-precision scenarios.

```python
checker = PrecisionChecker(
    backend="aot_eager",
    cast_dtype=torch.bfloat16,
)
checker.wrap_by_policy(model, (torch.nn.Linear,))

result = checker.compare(run_step, model)
checker.report(result)
```

### graph dump

After `dump_graphs` is enabled, the tool intercepts the graph code formatting process of PyTorch Dynamo and saves the captured graphs to `graph_dir`.

```python
checker = PrecisionChecker(
    backend="aot_eager",
    dump_graphs=True,
    graph_dir="./graph_dump",
)
checker.wrap_all_children(model)

result = checker.compare(run_step, model)
checker.report(result)
```

Example output file names:

```text
__compiled_fn_0.Captured_Graph.xxxxxxxx.py
__compiled_fn_1.Forward_Graph.xxxxxxxx.py
__compiled_fn_2.Backward_Graph.xxxxxxxx.py
```

Note: This capability depends on PyTorch internal interfaces. Changes in the PyTorch version may affect the dump behavior.

### Report Description

The tool supports printing the stdout report:

```python
checker.report(result)
```

It also supports generating a CSV report while printing the stdout report. The `csv_path` parameter of `report()` defaults to `None`; a CSV file is generated only when a path is explicitly passed in. After writing is complete, `CSV report saved to: <path>` is printed:

```python
checker.report(result, csv_path="precision_report.csv")
```

For the column definitions of the CSV report, see [CSV Report Format](#csv-report-format) below.

Example stdout report:

```text
========================================================================
  Loss  eager=6.912340  compiled=6.912341  diff=1.200e-06
========================================================================

  FORWARD INPUT
  --------------------------------------------------------------------
  pass  layers.0                                             [OK] max_abs=0.000e+00  mean_abs=0.000e+00  max_rel=0.000e+00  shape=(2, 128, 4096)

  FORWARD OUTPUT
  --------------------------------------------------------------------
  pass  layers.0                                             [OK] max_abs=2.300e-07  mean_abs=1.200e-08  max_rel=5.100e-06  shape=(2, 128, 4096)

  BACKWARD
  --------------------------------------------------------------------
  pass  layers.0.grad_input                                  [OK] max_abs=1.100e-07  mean_abs=9.000e-09  max_rel=4.200e-06  shape=(2, 128, 4096)
  pass  layers.0.grad_output                                 [OK] max_abs=0.000e+00  mean_abs=0.000e+00  max_rel=0.000e+00  shape=(2, 128, 4096)

========================================================================
  RESULT: ALL PASS  (atol=1e-4 rtol=1e-3)
========================================================================
```

> Note: In single-pass mode, the `eager` value in the `Loss` line is shown as `nan`, which is expected. Accuracy is determined by the per-module `FORWARD OUTPUT` and `BACKWARD`.

Report region description:

| Region | Description |
| --- | --- |
| `FORWARD INPUT` | Forward input comparison. Not shown when `capture_input=False`. |
| `FORWARD OUTPUT` | Forward output comparison. |
| `BACKWARD` | Backward gradient comparison. two-pass compares `grad_input` and `grad_output`; single-pass currently only attempts to reconstruct and compare `grad_input`, and `grad_output` is usually `None`. |
| `RESULT` | Overall result. `ALL PASS` means all check items that are not skipped or ignored have passed. |

Common marker descriptions:

| Marker | Meaning |
| --- | --- |
| `pass` | Accuracy passed. |
| `FAIL` | Accuracy is outside the `torch.allclose(a, b, atol=1e-4, rtol=1e-3)` determination range. |
| `skip IGNORED` | The module was skipped by `ignore()` or `ignore_by_policy()`. |
| `skip SKIP_compiled_wrapper` | The module is a compiled wrapper, and some backward hooks are not directly aligned. This is expected. |
| `skip SKIP_inside_compiled` | The module is inside a compiled module, and the hook is not triggered after graph fusion. This is expected. |
| `WARN MISSING_fwd_in_compiled` | The eager side has collected results, but the compiled side has not. Check the wrap granularity or hook triggering. |
| `WARN MISSING_fwd_in_eager` | The compiled side has collected results, but the eager side has not. |
| `WARN MISSING_bwd_in_compiled` | The eager side has backward information, but the compiled side has not collected it. |

#### CSV Report Format

After the `csv_path` parameter is passed in, the tool generates a detailed report in CSV format, containing the following 9 columns:

| Field | Description |
| --- | --- |
| `module_name` | Module name. The whole-network loss is shown as `LOSS`. |
| `check_type` | Check type, including `loss`, `fwd_input`, `fwd_output`, `grad_input`, `grad_output`, and `note`. |
| `tensor_index` | Tensor index, starting from 0. Used to distinguish tensors when there are multiple inputs or outputs. |
| `status` | `PASS`, `FAIL`, `SKIP`, or `WARN`. |
| `max_abs_diff` | Maximum absolute error, in scientific notation (such as `1.192093e-07`). |
| `mean_abs_diff` | Mean absolute error, in scientific notation. |
| `max_rel_diff` | Maximum relative error, in scientific notation. |
| `shape` | Tensor shape (such as `(2, 128, 4096)`) or loss information (such as `eager=6.912340 compiled=6.912341`). |
| `note` | Skip, ignore, or warning information. Empty when passing normally. |

CSV example (partial rows):

```csv
module_name,check_type,tensor_index,status,max_abs_diff,mean_abs_diff,max_rel_diff,shape,note
LOSS,loss,0,PASS,1.200000e-06,N/A,N/A,eager=6.912340 compiled=6.912341,
layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 128, 4096)",
layers.0,fwd_output,0,PASS,2.300000e-07,1.200000e-08,5.100000e-06,"(2, 128, 4096)",
layers.0,grad_input,0,PASS,1.100000e-07,9.000000e-09,4.200000e-06,"(2, 128, 4096)",
layers.1,note,0,SKIP,,,,,IGNORED
```

In single-pass mode, the `shape` field of the `LOSS` row shows only the `compiled` value, and the `note` field is `single_pass mode`:

```csv
LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
```

### Multi-card FSDP2 Accuracy Check

FSDP2 (`torch.distributed.fsdp.fully_shard`) shards parameters into `DTensor` distributed across multiple cards and prohibits `deepcopy`, so only single-pass mode is supported. A typical usage in the FSDP2 scenario is as follows:

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

dist.init_process_group(backend="hccl")  # or "nccl", etc.
rank = dist.get_rank()

model = build_model()
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

checker = PrecisionChecker()  # single_pass=True, default
checker.wrap_by_policy(model, (Qwen2Block,))
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path=f"precision_rank{rank}.csv")
```

Notes:

- Each card runs `checker.report()` independently and generates its own CSV report.
- The tool itself does not have a built-in multi-card result aggregation capability. To merge multi-card results into a single CSV, refer to the multi-card CSV report format below and implement the aggregation logic yourself.

#### Multi-card CSV Report Format

When merging multi-card, multi-scenario results into a single CSV, you can use the following format, prepending the `rank` and `scenario` columns to the original 9 columns:

```csv
rank,scenario,module_name,check_type,tensor_index,status,max_abs_diff,mean_abs_diff,max_rel_diff,shape,note
0,core,LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
0,core,layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 16, 128)",
0,core,layers.0,fwd_output,0,PASS,1.192093e-07,3.421843e-08,2.980232e-07,"(2, 16, 128)",
0,core,layers.0,grad_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 16, 128)",
1,core,LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
1,core,layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 16, 128)",
0,bf16,layers.1,fwd_output,0,PASS,3.051758e-05,7.510185e-06,6.556511e-05,"(2, 16, 128)",
1,ignore,layers.0,-,0,SKIP,N/A,N/A,N/A,,IGNORED
```

Field description:

| Field | Description |
| --- | --- |
| `rank` | Card number, corresponding to `dist.get_rank()`. |
| `scenario` | Test scenario name (such as `core`, `bf16`, `ignore`), used to distinguish different configuration combinations, for example whether `cast_dtype` is enabled or whether a specified module is ignored. |
| Remaining 9 columns | Same as the single-card CSV. |

A reference approach for multi-card result aggregation is as follows: use `torch.multiprocessing.spawn` to start multi-card processes, have each card write its per-module details to a temporary JSON file, and then have the main process read all JSON files and aggregate them into a single CSV.

### Inference Mode Support

Inference scenarios have no backward computation. You can use the single-pass `install()` / `collect()` path, or the two-pass `compare()`. The single-pass integration is as follows:

```python
checker = PrecisionChecker()
checker.wrap_all_children(model)
checker.install(model)

model.eval()
with torch.no_grad():
    logits = model(input_ids)
    pseudo_loss = logits.mean()

result = checker.collect(pseudo_loss)
checker.report(result)
```

In inference mode:

- `FORWARD INPUT` and `FORWARD OUTPUT` are displayed normally.
- `BACKWARD` items are usually `None`.
- It is suitable for verifying the output consistency of the inference pipeline before and after compile.

If you need a complete comparison of `loss_eager` and `loss_compiled` (here a comparison of output statistics), use two-pass instead:

```python
checker = PrecisionChecker(single_pass=False)
checker.wrap_all_children(model)


def infer_step(model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        return logits.mean()


result = checker.compare(infer_step, model)
checker.report(result)
```

## Accuracy Metrics

Accuracy metrics are computed from the comparison tensors collected by the tool. In two-pass mode, the tool runs the eager model and the compiled model separately and compares the inputs, outputs, and gradients collected on both sides; in single-pass mode, the tool re-executes `_orig_mod` with the same input when the compiled module runs, and then compares the results on both sides. Before comparison, the tool converts the tensors involved in the comparison to CPU float32.

| Metric | Description |
| --- | --- |
| `max_abs` | `abs(a - b).max()`, maximum absolute error. |
| `mean_abs` | `abs(a - b).mean()`, mean absolute error. |
| `max_rel` | `(abs(a - b) / (abs(a) + 1e-8)).max()`, maximum relative error. |
| `allclose` | `torch.allclose(a, b, atol=1e-4, rtol=1e-3)`. |
| `shape` | Tensor shape. A shape mismatch is directly determined as a failure. |

Accuracy metrics are not a separate output file but are the core fields in the stdout report and the CSV report. In the CSV, `max_abs_diff`, `mean_abs_diff`, `max_rel_diff`, and `shape` correspond to `max_abs`, `mean_abs`, `max_rel`, and `shape` in the table above respectively, and `status` is converted from the results of checks such as `allclose`. Based on these metrics, you can determine which module the difference occurs in, the type of difference (forward or backward), the magnitude of the error, and whether it exceeds the tool's determination threshold.

## FAQ

**Why is `loss_eager` `NaN` in single-pass?**

single-pass mode does not run the complete eager model separately, so it cannot compute the whole-network eager loss, and `loss_eager=NaN` is expected behavior. Accuracy is determined by the per-module `FORWARD OUTPUT` and `BACKWARD`. For details, see [About loss_eager=NaN in single-pass](#about-loss_eagernan-in-single-pass).

**Can an FSDP2 model use two-pass?**

No. two-pass internally performs `deepcopy` on the model, while FSDP2 prohibits deepcopy. Use single-pass mode for FSDP2 scenarios.

**Which mode should I choose for inference?**

two-pass (`single_pass=False`) is recommended. Inference has no backward computation or optimizer state, and two-pass can provide a complete whole-network eager vs compiled comparison. If the inference model uses a wrapper that prohibits deepcopy, such as FSDP2, use single-pass instead.

**Which mode should I choose for training?**

single-pass (default) is recommended. This mode has a low integration cost, does not alter the training state, and is compatible with FSDP2. If you need a real comparison of `loss_eager` and `loss_compiled`, use two-pass instead, but two-pass does not support FSDP2.
