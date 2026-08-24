# Compilation Accuracy Comparison in PyTorch

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:44:02.908Z pushedAt=2026-08-11T02:51:07.170Z -->

## Overview

`PrecisionChecker` is a tool for **module-by-module precision comparison between eager mode and compile mode** in PyTorch models. When enabling `torch.compile` for a model leads to issues such as loss fluctuation, convergence degradation, or inconsistent inference outputs, this tool helps pinpoint the source of discrepancies module by module.

The tool works as follows: it marks the modules to be inspected via `wrap`, executes these modules in both eager and compile modes, and compares their forward outputs and input gradients one by one, thereby identifying which module the discrepancy originates from, the type of discrepancy (forward or backward), and the error magnitude.

The tool's interface follows the usage pattern of FSDP2's `fully_shard`, supporting inspection scope specification at the granularity of module type, module hierarchy, or individual module.

## Mode Selection

The tool provides two comparison modes: single-pass and two-pass. Selecting the correct mode is a prerequisite for using this tool. For first-time use, refer to the following table to make your choice:

| Scenario | Recommended Mode | Entry API |
| --- | --- | --- |
| Training (including `loss.backward()`) | single-pass (default) | `install()` + `collect()` |
| Inference/eval (forward only, no backward) | two-pass | `compare()` |
| Multi-rank distributed training such as FSDP2 | single-pass (required) | `install()` + `collect()` |
| Need real full-network comparison of `loss_eager` vs. `loss_compiled`  | two-pass | `compare()` |
| Model not supporting `deepcopy` (e.g., after FSDP2 sharding) | single-pass (required) | `install()` + `collect()` |

### Training Scenarios: single-pass Recommended

single-pass is the default mode. This mode does not copy the model, nor does it execute a separate eager full-graph run. Instead, it directly hooks into the existing training pipeline: inside the forward hook of a compiled submodule, it re-executes the uncompiled `_orig_mod` with the same inputs and compares the eager output against the compile output.

single-pass is suitable for training scenarios for the following reasons:

- Low integration cost. There is no need to modify the existing `forward → loss → backward` flow. You only need to call `install()` before the training step and `collect(loss)` after the training step.
- It does not alter the optimizer or model state, thereby avoiding state divergence caused by copying the model and retraining.
- Compatible with distributed wrappers such as FSDP2. Such models prohibit `deepcopy`, and single-pass does not perform `deepcopy`, making it the only available mode in multi-rank training scenarios.

Note that single-pass does not run the full eager model separately, so it does not compute the overall eager loss. The `loss_eager` in the report is `NaN`, which is expected behavior (for details, see [loss_eager=NaN in single-pass Mode](#loss_eagernan-in-single-pass-mode)). Accuracy judgment is based on per-module forward and backward comparison results, not on the overall network loss.

### Inference Scenarios: two-pass Recommended

Inference scenarios involve no backward computation or optimizer states, nor do they have the issue of copying models and contaminating training states. Therefore, two-pass is more suitable:

- Two-pass executes one complete eager model run and one complete compiled model run separately, providing real `loss_eager`, `loss_compiled`, and `loss_diff` values, which makes it easy to determine whether the entire network produces differences due to compilation.
- Inference pipelines typically follow an "input-output" pattern. The two-pass semantics of "executing two pipelines separately and then comparing" are intuitive, and the results are complete (forward inputs, forward outputs, and gradients can all be compared).

Constraint: Two-pass internally performs `deepcopy` on the model to create eager and compiled copies, so it does not support models that prohibit `deepcopy`, such as FSDP2. Ordinary single-rank inference is not affected by this constraint.

> Summary: Use single-pass (`install` + `collect`) for training scenarios, and two-pass (`compare`) for inference scenarios.

## Preparation

**Installation**

Install msProbe by referring to [*msProbe Installation Guide*](../../install_guide/msprobe_install_guide.md).

**Constraints**

- Only PyTorch scenarios are supported.
- The `torch.compile` capability is required. Ensure that the current PyTorch version supports `torch.compile`.
- `install()` only supports the `single_pass=True` scenario.
- The two-pass mode (`single_pass=False`) performs `deepcopy` on the model and does not support models that prohibit `deepcopy`, such as FSDP2.
- `dump_graphs` depends on PyTorch Dynamo internal interfaces. Changes in PyTorch versions may affect the graph dump behavior.

## Quick Start

The following two examples correspond to the two most common scenarios in the mode selection table, and both can be run independently.

### Training Scenario: single-pass

A training pipeline typically already includes the training step of `forward → loss → backward`. The single-pass integration approach is to call `install()` and `collect()` outside the training step, without modifying the training step itself:

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


checker = PrecisionChecker()                 # single_pass=True is the default value.
checker.wrap_by_policy(model, (nn.Linear,))  # Mark the modules that participate in comparison.
checker.install(model)                       # Replace the marked modules in-place with their compiled versions.

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path="precision_report.csv")
```

**Note**:

- `install()` must be called before `run_step()`. It replaces the submodules that have been wrapped with their compiled versions in place.
- `collect(loss)` must be called after `run_step()`. It organizes the per-module comparison results collected in the current step.
- The `loss_eager` field in the report is `NaN`, which is expected behavior. Accuracy judgment is based on the per-module `FORWARD OUTPUT` and `BACKWARD` results.

### Inference Scenario: two-pass

The inference scenario involves only forward computation. Using two-pass, you can execute the eager and compiled models separately in full, and directly compare the overall network output:

```python
import torch
import torch.nn as nn

from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker

model = build_model()          # Inference model
input_ids = make_input()       # Inference input


def infer_step(model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        # Two-pass requires a scalar as the comparison anchor. Inference has no loss,
        # You can use a statistic (such as mean) of the output, which does not participate in backpropagation.
        return logits.mean()


checker = PrecisionChecker(single_pass=False)   # Inference uses two-pass
checker.wrap_by_policy(model, (nn.Linear,))

result = checker.compare(infer_step, model)     # Internally execute both eager and compiled paths
checker.report(result, csv_path="precision_report.csv")
```

**Note**:

- For inference scenarios, use `compare()` without calling `install()`.
- two-pass provides the actual `loss_eager` and `loss_compiled` (here `logits.mean()`), allowing direct judgment of whether the entire network is consistent.
- If the inference model uses wrappers such as FSDP2 that prohibit `deepcopy`, switch to the single-pass `install()` + `collect()` path instead.

## Feature Overview

The following table lists the complete feature set of `PrecisionChecker`:

| Feature | Description | Supported Scenario |
| --- | --- | --- |
| [Constructor Parameter Configuration](#constructor-parameter-configuration) | Configure backend, threshold, graph dump, autocast, input collection, and check mode | PyTorch |
| [Specify Modules for Inspection](#specify-modules-for-inspection) | Specify modules to participate in compile and precision comparison via `wrap`, `wrap_by_policy`, and `wrap_all_children` | PyTorch |
| [Skip Check Modules](#skip-check-modules) | Skip modules of no interest via `ignore` and `ignore_by_policy` | PyTorch |
| [single-pass Mode](#single-pass-mode) | Compile wrapped submodules in-place and perform fast comparison via hooks | PyTorch |
| [two-pass Mode](#two-pass-mode) | Execute the eager model and the compiled model separately, comparing loss, inputs/outputs, and gradients | PyTorch |
| [autocast Precision Check](#autocast-precision-check) | Enable `torch.autocast` for wrapped modules to check mixed-precision scenarios | PyTorch |
| [Graph Dump](#graph-dump) | Save the graph code captured by PyTorch Dynamo | PyTorch |
| [Report Description](#report-description) | Print stdout report; optionally generate a CSV report (via the `csv_path` parameter) | PyTorch |
| [Multi-Rank FSDP2 Accuracy Check](#multi-rank-fsdp2-accuracy-check) | Perform per-module comparison under FSDP2 sharding using single-pass, with support for multi-rank CSV reports | PyTorch |
| [Inference Mode Support](#inference-mode-support) | Execute forward only, used to check consistency before and after compile in inference pipelines | PyTorch |

### Constructor Parameter Configuration

The tool initialization example is as follows:

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

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `backend` | `str` | `"aot_eager"` | Backend used by `torch.compile`. |
| `threshold` | `float` | `1e-4` | Reserved precision threshold. Currently determined via `torch.allclose(a, b, atol=1e-4, rtol=1e-3)`. |
| `dump_graphs` | `bool` | `False` | Whether to write the graphs captured during compilation to `graph_dir`. |
| `graph_dir` | `str` | `"./graph_dump"` | Output directory for graph dumps. |
| `cast_dtype` | `torch.dtype` | `None` | Enables autocast for wrapped modules, similar to a mixed precision policy. |
| `capture_input` | `bool` | `True` | Whether to capture the module's forward inputs, used to distinguish "pre-existing input differences" from "differences introduced by module computation". |
| `single_pass` | `bool` | `True` | Whether to use the single-pass comparison logic. When set to `False`, two-pass comparison is used. |

### Specify Modules for Inspection

`wrap` is used to mark modules that participate in compilation and accuracy comparison. Modules that are not wrapped will not be treated as independent compilation units.

```python
# Manually mark a module.
checker.wrap(model.layers[3].self_attn)

# Batch mark modules by type, similar to ModuleWrapPolicy.
checker.wrap_by_policy(model, (Qwen2Block,))
checker.wrap_by_policy(model, (Qwen2Attention, Qwen2MLP))

# Automatically wrap child modules to the specified depth.
checker.wrap_all_children(model)          # depth=1, default value
checker.wrap_all_children(model, depth=0) # Wrap direct submodules
checker.wrap_all_children(model, depth=2) # Wrap deeper levels

# Wrap the entire model
checker.wrap(model)
```

**Note**:

- `install()` replaces the marked submodule in place.
- If the root model itself is marked, the return value of `torch.compile(model)` cannot be written back to the variable held by the caller. It is recommended to use `compare()` directly, or mark a submodule under the root model instead.
- `wrap_all_children` recursively traverses the model's submodules and automatically skips container modules such as `ModuleList`, `ModuleDict`, and `Sequential`.
- For large models, it is recommended to start inspection at a coarser granularity and then progressively break down into smaller modules.
- If a module has already been `wrap`-ped, its internal submodules will not be processed again as top-level compilation targets.

### Skip Check Modules

`ignore` is used to skip accuracy comparison for specified modules. Modules that are ignored still participate in model execution, but are excluded from report judgment.

```python
checker.wrap_by_policy(model, (Qwen2Block,))

# Skip the specified module.
checker.ignore(model.layers[0])

# Skip modules in batch by type.
checker.ignore_by_policy(model, (torch.nn.Dropout,))
```

Skipped modules are displayed as `IGNORED` in the report.

### single-pass Mode

`single_pass=True` is the default mode. It is recommended to use `install()` and `collect()` to integrate into an existing training pipeline:

```python
checker = PrecisionChecker()
checker.wrap_all_children(model)
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result)
```

You can also use `compare()` for single-pass:

```python
checker = PrecisionChecker()
checker.wrap_by_policy(model, (Qwen2Block,))

result = checker.compare(run_step, model)
checker.report(result)
```

If finer-grained comparison is needed, split the wrap into finer modules:

```python
checker.wrap_by_policy(model, (Qwen2Attention, Qwen2MLP))
```

#### loss_eager=NaN in single-pass Mode

In single-pass mode, the tool does not run a complete eager model separately, so it cannot provide a real `loss_eager`. `loss_eager=NaN` shown in the report is expected behavior, not a tool defect.

The accuracy judgment in single-pass mode is based on per-module comparison: within the forward hook of each compiled submodule, `_orig_mod` (the uncompiled version) is re-executed with the same input, and the eager output is compared against the compiled output. The advantage of this approach is that it requires neither model copying nor a separate training pipeline; the trade-off is that the end-to-end eager loss cannot be obtained.

If you need a real comparison between `loss_eager` and `loss_compiled`, switch to two-pass mode (`single_pass=False`), which executes the complete eager and compiled pipelines separately.

### two-pass Mode

When `single_pass=False`, the two-pass mode is enabled. The tool constructs an eager model and a compiled model, executes the same `run_step` on each, and then compares the inputs, outputs, and gradients collected by the hooks on both sides.

```python
checker = PrecisionChecker(single_pass=False)
checker.wrap_by_policy(model, (Qwen2Block,))

result = checker.compare(run_step, model)
checker.report(result)
```

The two-pass mode compares `loss_eager` with `loss_compiled`, and can be used to determine whether the full-graph compilation pipeline introduces loss discrepancies.

Constraint: Internally, two-pass performs a `deepcopy` of the model to create separate eager and compiled copies. Therefore, it does not support models that prohibit deepcopy, such as FSDP2. If the model is already wrapped with FSDP2, use the single-pass mode instead.

### autocast Precision Check

When `cast_dtype` is set, the tool adds a `_CastWrapper` to the wrapped modules and uses `torch.autocast` in the forward pass. This is used to verify whether precision differences exist in the wrapped modules under automatic mixed precision scenarios.

```python
checker = PrecisionChecker(
    backend="aot_eager",
    cast_dtype=torch.bfloat16,
)
checker.wrap_by_policy(model, (torch.nn.Linear,))

result = checker.compare(run_step, model)
checker.report(result)
```

### Graph Dump

When `dump_graphs` is enabled, the tool intercepts the graph code formatting process of PyTorch Dynamo and saves the captured graphs to `graph_dir`.

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

Output file name examples:

```text
__compiled_fn_0.Captured_Graph.xxxxxxxx.py
__compiled_fn_1.Forward_Graph.xxxxxxxx.py
__compiled_fn_2.Backward_Graph.xxxxxxxx.py
```

**Note**: This capability depends on PyTorch's internal interfaces. Changes in PyTorch versions may affect the dump behavior.

### Report Description

The tool supports printing stdout reports:

```python
checker.report(result)
```

It also supports generating a CSV report while printing the stdout report. The `csv_path` parameter of `report()` defaults to `None`, and a CSV file is generated only when a path is explicitly passed in. After writing is complete, `CSV report saved to: <path>` is printed:

```python
checker.report(result, csv_path="precision_report.csv")
```

For details on the CSV report, see [CSV Report Format](#csv-report-format) below.

Example of a stdout report:

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

> Tip: In single-pass mode, the `eager` value in the `Loss` row is displayed as `nan`, which is expected behavior. The accuracy judgment is based on per-module `FORWARD OUTPUT` and `BACKWARD` comparisons.

Report description:

| Area | Description |
| --- | --- |
| `FORWARD INPUT` | Forward input comparison. Not displayed when `capture_input=False`. |
| `FORWARD OUTPUT` | Forward output comparison. |
| `BACKWARD` | Backward gradient comparison. In two-pass mode, both `grad_input` and `grad_output` are compared; in single-pass mode, only `grad_input` is currently reconstructed and compared, while `grad_output` is typically `None`. |
| `RESULT` | Overall result. `ALL PASS` indicates that all non-skipped and non-ignored checks have passed. |

Common markers:

| Marker | Meaning |
| --- | --- |
| `pass` | Precision check passed. |
| `FAIL` | Precision exceeds the `torch.allclose(a, b, atol=1e-4, rtol=1e-3)` threshold. |
| `skip IGNORED` | The module was skipped by `ignore()` or `ignore_by_policy()`. |
| `skip SKIP_compiled_wrapper` | The module is a compiled wrapper; some backward hooks do not align directly, which is expected behavior. |
| `skip SKIP_inside_compiled` | The module resides inside a compiled module; hooks do not fire after graph fusion, which is expected behavior. |
| `WARN MISSING_fwd_in_compiled` | Data was collected on the eager side but not on the compiled side. Check the wrap granularity or hook triggering conditions. |
| `WARN MISSING_fwd_in_eager` | Data was collected on the compiled side but not on the eager side. |
| `WARN MISSING_bwd_in_compiled` | Backward information exists on the eager side but was not collected on the compiled side. |

#### CSV Report Format

When the `csv_path` parameter is passed, the tool generates a detailed report in CSV format with the following 9 columns:

| Column | Description |
| --- | --- |
| `module_name` | Module name. The overall network loss is displayed as `LOSS`. |
| `check_type` | Check type, including `loss`, `fwd_input`, `fwd_output`, `grad_input`, `grad_output`, and `note`. |
| `tensor_index` | Tensor index, starting from 0. Used to distinguish tensors in the case of multiple inputs or outputs. |
| `status` | `PASS`, `FAIL`, `SKIP`, or `WARN`. |
| `max_abs_diff` | Maximum absolute difference, in scientific notation (e.g., `1.192093e-07`). |
| `mean_abs_diff` | Mean absolute difference, in scientific notation. |
| `max_rel_diff` | Maximum relative difference, in scientific notation. |
| `shape` | Tensor shape (e.g., `(2, 128, 4096)`) or loss information (e.g., `eager=6.912340 compiled=6.912341`). |
| `note` | Skip, ignore, or warning information. Empty when the check passes normally. |

CSV example (partial rows):

```csv
module_name,check_type,tensor_index,status,max_abs_diff,mean_abs_diff,max_rel_diff,shape,note
LOSS,loss,0,PASS,1.200000e-06,N/A,N/A,eager=6.912340 compiled=6.912341,
layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 128, 4096)",
layers.0,fwd_output,0,PASS,2.300000e-07,1.200000e-08,5.100000e-06,"(2, 128, 4096)",
layers.0,grad_input,0,PASS,1.100000e-07,9.000000e-09,4.200000e-06,"(2, 128, 4096)",
layers.1,note,0,SKIP,,,,,IGNORED
```

In single-pass mode, the `shape` field of the `LOSS` row displays only the `compiled` value, and the `note` field shows `single_pass mode`:

```csv
LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
```

### Multi-Rank FSDP2 Accuracy Check

FSDP2 (`torch.distributed.fsdp.fully_shard`) shards parameters into `DTensor` and distributes them across multiple ranks, and it prohibits `deepcopy`, so only single-pass mode is supported. The typical usage in FSDP2 scenarios is as follows:

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

dist.init_process_group(backend="hccl")  # // Or "nccl", etc.
rank = dist.get_rank()

model = build_model()
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

checker = PrecisionChecker()  # // single_pass=True, default
checker.wrap_by_policy(model, (Qwen2Block,))
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path=f"precision_rank{rank}.csv")
```

Note

- Each rank independently invokes `checker.report()` and generates its own CSV report.
- The tool itself does not have built-in multi-rank result aggregation capabilities. If you need to merge multi-rank results into a single CSV, you can implement the aggregation logic yourself by referring to the multi-rank CSV report format described below.

#### Multi-Rank CSV Report Format

When merging multi-rank and multi-scenario results into a single CSV report, the following format can be used, which prepends the `rank` and `scenario` columns before the original 9 columns:

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
| `rank` | Device ID, corresponding to `dist.get_rank()`. |
| `scenario` | Test scenario name (e.g., `core`, `bf16`, `ignore`), used to distinguish different configuration combinations, such as whether `cast_dtype` is enabled or whether specific modules are ignored. |
| Remaining 9 columns | Same as the single-rank CSV. |

Multi-rank result aggregation can be implemented as follows: use `torch.multiprocessing.spawn` to launch multi-rank processes, where each rank writes its per-module details to a temporary JSON file, and then the main process reads all JSON files and aggregates them into a single CSV report.

### Inference Mode Support

In inference scenarios, there is no backward computation. You can use either the single-pass `install()`/`collect()` path or the two-pass `compare()` method. The single-pass integration approach is as follows:

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
- `BACKWARD`-related items are typically `None`.
- Suitable for verifying the output consistency before and after compile in the inference pipeline.

If a complete comparison between `loss_eager` and `loss_compiled` (here referring to the comparison of output statistics) is required, you can switch to two-pass:

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

Accuracy metrics are computed from the comparison tensors collected by the tool. In two-pass mode, the tool executes the eager model and the compiled model separately, and compares the inputs, outputs, and gradients collected from both sides. In single-pass mode, the tool re-executes `_orig_mod` using the same inputs during the compiled module execution, and then compares the results from both sides. Before comparison, the tool converts all tensors involved in the comparison to CPU float32.

| Metric | Description |
| --- | --- |
| `max_abs` | `abs(a - b).max()`, the maximum absolute error. |
| `mean_abs` | `abs(a - b).mean()`, the mean absolute error. |
| `max_rel` | `(abs(a - b) / (abs(a) + 1e-8)).max()`, the maximum relative error. |
| `allclose` | `torch.allclose(a, b, atol=1e-4, rtol=1e-3)`. |
| `shape` | Tensor shape. A shape mismatch is directly treated as a failure. |

Accuracy metrics are not output as a separate file; rather, they are the core fields in the stdout report and the CSV report. In the CSV report, `max_abs_diff`, `mean_abs_diff`, `max_rel_diff`, and `shape` correspond to `max_abs`, `mean_abs`, `max_rel`, and `shape` in the table above, respectively, while `status` is derived from the `allclose` check result and other checks. Users can use these metrics to identify the module where the discrepancy occurs, the type of discrepancy (forward or backward), the error magnitude, and whether the tool's judgment threshold has been exceeded.

## FAQs

**Why is `loss_eager` in single-pass `NaN`?**

The single-pass mode does not independently execute the complete eager model, so the overall eager loss cannot be computed. `loss_eager=NaN` is expected behavior. Accuracy judgment is based on per-module `FORWARD OUTPUT` and `BACKWARD`. For details, see [loss_eager=NaN in single-pass Mode](#loss_eagernan-in-single-pass-mode).

**Can FSDP2 models use two-pass?**

No. Two-pass internally performs `deepcopy` on the model, while FSDP2 prohibits `deepcopy`. For FSDP2 scenarios, use the single-pass mode.

**Which mode should be used for inference scenarios?**

Two-pass (`single_pass=False`) is recommended. Inference involves no backward computation or optimizer state, so two-pass can provide a complete comparison of the entire network between eager and compiled modes. If the inference model uses wrappers that prohibit `deepcopy`, such as FSDP2, single-pass should be used instead.

**Which mode should be used for training scenarios?**

Single-pass (default) is recommended. This mode has low integration cost, does not alter the training state, and is compatible with FSDP2. If a real comparison between `loss_eager` and `loss_compiled` is required, two-pass can be used instead, but two-pass does not support FSDP2.
