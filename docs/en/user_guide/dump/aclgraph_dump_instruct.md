# aclgraph_dump User Guide

## Introduction

When performing precision alignment in **PyTorch ACLGraph** mode, the overall strategy is as follows: **first network-wide screening, then single-point deep dive**. Use **network-wide collection** to quickly narrow down the anomaly scope, and then perform **single-point collection** on suspected problematic operators and save their tensor data for fine-grained investigation. `aclgraph_dump` provides the following collection capabilities:

- Single-point collection: `acl_save`
- Network-wide collection: `AclGraphDumper`, which supports statistical value collection, real tensor data collection, custom API collection, and dynamic enable/disable.

## Preparation

**Environment Preparation**

1. Install and correctly configure TorchNPU.
2. Install the msProbe tool. For details, see *[msProbe Installation Guide](../../install_guide/msprobe_install_guide.md)*.

>[!NOTE]
>
>When building from source, the `aclgraph_dump` module must be included. Install it using the following command:
>
>`python3 build.py -e include-mod=aclgraph_dump -e no-check=true`

**Constraints**

- Only the PyTorch framework is supported.
- Building `aclgraph_dump` requires TorchNPU to participate in compilation; if this module is not included, the feature will not function properly.

## Network-wide Collection

### Quick Start

1. Before using the network-wide collection feature, you need to configure the file (`config.json`):

    ```json
    {
    "task": "statistics",
    "dump_path": "./L0_dump",
    "rank": [],
    "level": "L0",
    "statistics": {
        "list": ["linear", "attention"],
        "slice": [{"end": 1024}]
    }
    }
    ```

    **Reference**

    The currently supported configuration items for network-wide `aclgraph dump` are as follows:

    | Configuration Item | Optional/Mandatory | Description |
    | --- | --- | --- |
    | `task` | Optional | Collection task type (`str`). Values:<br/>&#8226; `statistics`: collects statistics; see the example in this section.<br/>&#8226; `tensor`: collects tensor data; see [Tensor Network-wide Collection](#tensor-network-wide-collection).<br/>&#8226; `nan_check`: detects NaN and saves all inputs and outputs of a call if any output contains NaN; see [Automatic NaN Detection](#automatic-nan-detection).<br/>Defaults to `statistics`. |
    | `dump_path` | Mandatory | Dump result output directory, string. The tool checks and creates this directory. |
    | `rank` | Optional | Specifies the rank(s) to collect, list[int \| str]. The default value is empty, meaning all ranks are collected. Strings only support the `start-end` range format. Full-network collection is not enabled for non-target ranks. |
    | `dump_enable` | Optional | Collection switch, bool. The default value is `true`. Modifying this field at runtime dynamically enables or disables collection. For details, see [Dynamic Enable/Disable](#dynamic-enabledisable). |
    | `level` | Optional | Collection level, string. Supports `L0`, `L1`, and `mix`. The default value is `L0`.<br>`L0` collects module input/output; `L1` collects API input/output; and `mix` collects both module and API. |
    | `list` | Optional | Keyword filter list for module or API names, list[str]. The default value is empty, meaning all data covered by the current `level` is collected. |
    | `custom_api` | Optional | List of full import paths of custom Python APIs, list[str]. Supported only for the `tensor` task. For details, see [Custom API Collection](#custom-api-collection). |
    | `slice` | Optional | Slices a tensor, list[dict]. Takes effect only for the `statistics` task. For details, see [slice Configuration](./config_json_introduct.md#slice-configuration). |

2. After completing the file configuration (`config.json`), the following example shows how to use the network-wide collection function:

    ```diff
      import torch
      import torch_npu
    + from msprobe.pytorch import AclGraphDumper

      N,D_in, H, D_out = 640, 4096, 2048, 1024
      # Initialize the model.
      model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
      ).npu()
    + # Set the default device to ensure the collection switch is created on the NPU before graph compilation.
    + torch.set_default_device("npu:0")
    + # Initialized the configuration.
    + dumper = AclGraphDumper('./config.json')
    + # Configure the collection task before graph compilation.
    + dumper.start(model)
      static_input = torch.randn(N, D_in).npu()
      static_target = torch.randn(N, D_out).npu()

      g = torch.npu.NPUGraph()
      # Graph compilation
      with torch.npu.graph(g):
        static_target = model(static_input)

      real_inputs = [torch.rand_like(static_input) for _ in range(10)]
      real_targets = [torch.rand_like(static_target) for _ in range(10)]

      for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        # Graph replay.
        g.replay()
    +   # Dump data.
    +   dumper.step()
    ```

### Network-wide Collection Overview

#### Overview

`AclGraphDumper` is used for network-wide collection of intermediate data. It supports collection at the module level, API level, and module + API level. The `statistics` task outputs tensor shapes, data types, and statistical values; the `tensor` task outputs real tensor data.
The initialization and `start` call of `AclGraphDumper` must be completed before model graph compilation (such as `torch.npu.graph` or `torch.compile`).

> [!NOTE]
>
> Using network-wide collection makes the graph capture phase slower. This is because the tool inserts collection nodes into the ACLGraph during capture (for statistics computation or tensor data preparation). As the collection scope grows, more nodes are added to the graph, and the capture phase takes longer accordingly; the more modules/APIs are collected, the more noticeable the impact. After capture, these collection nodes become part of the graph and execute on every replay to collect data.

#### Interfaces

**Prototype**

```python
AclGraphDumper(config_path: str | None = None)
```

**Parameter Description**

| Name | Optional/Mandatory | Description |
| --- | --- | --- |
| `config_path` | Optional | Configuration file path, string. If not passed, the built-in `config.json` in the msProbe package is read by default. The collection task and its parameters are all read from this file. |

**Prototype**

```python
AclGraphDumper.start(model: torch.nn.Module) -> None
```

**Parameter Description**

| Name | Optional/Mandatory | Description |
| --- | --- | --- |
| `model` | Mandatory | Model to be collected, of type `torch.nn.Module`. |

**Prototype**

```python
AclGraphDumper.step(dump: bool = True) -> None
```

**Parameter Description**

| Name | Type | Description | Mandatory |
| --- | --- | --- | --- |
| `dump` | bool | Whether to write the current statistical results to `dump.json`; takes effect for the `statistics` and `nan_check` tasks. `True`: clear statistics and write to disk, and `step_id` increments by 1; `False`: only clear statistics without writing to disk, and `step_id` does not increment (can be used during the `dummy_run` warm-up phase). For the `tensor` task, the `.pt` files generated in the current step are archived and `step_id` advances. | No |

If collection has not been started, it returns directly.

### Automatic NaN Detection

Graph mode uses `AclGraphDumper` and supports only `check_mode: "isnan"`. The detection scope is NaN in tensors. Call `start(model)` before graph capture and `step()` after each replay to save the results.

Complete configuration example:

```json
{
  "task": "nan_check",
  "dump_path": "./nan_dump",
  "rank": [],
  "dump_enable": true,
  "level": "L1",
  "nan_check": {
    "check_mode": "isnan",
    "list": [],
    "detail": true
  }
}
```

**Parameters**

The supported configuration for graph-mode `nan_check` consists of the following eight parameters:

| Parameter | Required/Optional | Description |
| ---- | --------- | ---- |
| task | Required | Set to `"nan_check"`. |
| dump_path | Required | Result directory, of type `str`. |
| rank | Optional | Ranks to collect, of type `list[int \| str]`, default `[]` (all ranks). Strings support `"start-end"` ranges. Before distributed initialization, data is collected from the current process. |
| dump_enable | Optional | A `bool`, default `true`. Controls result saving and supports [dynamic enable/disable](#dynamic-enabledisable). |
| level | Optional | A `str`, default `"L0"`. `L0` checks modules, `L1` checks APIs, and `mix` checks both. |
| check_mode | Optional | A `str`, default `"isnan"`. This is the only value supported in graph mode. |
| list | Optional | Module/API name keywords, of type `list[str]`, default `[]` (all data covered by the current `level`). |
| detail | Optional | A `bool`, default `true`. Controls whether `dump.json` is generated; see [nan_check Task Output](#nan_check-task). |

If any output tensor contains NaN, all input/output tensors of the current module/API call are saved. Inplace and `out=` operations are supported, preserving inputs before mutation and results afterward. API collection filters out entries with `empty` in their names.

For tensor `is_nan` values and special data types, see [is_nan Field in the Usage Example](./pytorch_nan_check_quickstart.md#is_nan-field). Graph-mode collection filters out FakeTensors and meta tensors. For saved files, see [nan_check Task Output](#nan_check-task).

### Tensor Network-wide Collection

After configuring `task` as `tensor`, `AclGraphDumper` saves the real data of the input, keyword arguments, and output tensors of the modules or APIs within the collection scope. An example configuration is as follows:

```json
{
    "task": "tensor",
    "dump_path": "./tensor_dump",
    "rank": [],
    "dump_enable": true,
    "level": "mix",
    "tensor": {
        "list": ["linear", "attention"]
    }
}
```

After each ACLGraph replay, call `dumper.step()`, and the tool archives the `.pt` files generated by this replay into the corresponding `step` directory. The input, keyword arguments, and output of the same API call share a call index, and the file name format is as follows:

```text
{api_name}.{call_index}.{input|input_kwargs|output}.{index}.pt
```

A single Tensor output uses `output.0.pt`. Tuple/list outputs use `output.0.pt`, `output.1.pt`, and so on. Nested containers append their positions or dictionary keys.

The process identifier in the directory is distinguished by running scenario: in the multi-rank scenario, `rank{rank_id}` is used; in the single-rank scenario, `rank0` is used; if the distributed environment is not initialized in the single-rank scenario, `pid{pid}` is used.

During collection, the `dump_tensor_data` directory is divided into the following two types (`{process_dir}` below represents the above process identifier):

- `dump_path/dump_tensor_data/{process_dir}`: A temporary working directory used to store the `.pt` files generated by the current replay that have not yet been archived. After `dumper.step()` is called, the files in it are moved to the corresponding `step` directory. The temporary working directory is retained for subsequent replays, so it is usually empty after `step()` is executed. Do not delete or rename this directory during collection.
- `dump_path/step{step_id}/{process_dir}/dump_tensor_data`: The archive directory of the current step, used to store the tensor data finally written to disk in this replay. When viewing or parsing the collection results, use the `.pt` files in this directory.

> [!NOTE]
>
> Tensor network-wide collection incurs significant disk and transfer overhead. It is recommended to narrow the collection scope in combination with `level` and `list`.

### Custom API Collection

For Python APIs that cannot be automatically identified through module hooks or PyTorch dispatch, you can configure the full import path in `tensor.custom_api`. This capability is supported only for the `tensor` task.

```json
{
    "task": "tensor",
    "dump_path": "./tensor_dump",
    "level": "L1",
    "tensor": {
        "custom_api": [
            "my_package.ops.reshape_and_cache"
        ]
    }
}
```

`AclGraphDumper.start()` parses and wraps the target API before graph compilation, and recursively collects the tensors in its positional arguments, keyword arguments, and return values. The following requirements must be met during configuration:

- The configured value must be the full path of an importable, callable object.
- During model graph compilation, the API must be called through the object corresponding to this path; function references that were copied to other variables before `start()` will not be wrapped.
- `custom_api` takes effect at `start()`; modifying this configuration at runtime does not re-wrap the API.
- The input, keyword arguments, and output of a custom API share the same replay call index.

### Dynamic Enable/Disable

Setting `dump_enable` at the root level of the configuration file controls whether the `statistics`, `tensor`, and `nan_check` tasks save collection results:

```json
{
    "task": "tensor",
    "dump_path": "./tensor_dump",
    "dump_enable": false,
    "level": "L1",
    "tensor": {
        "list": []
    }
}
```

Even if the initial value is `false`, `AclGraphDumper.start()` must be called before graph compilation so that the collection nodes are added to the ACLGraph. You can directly modify `dump_enable` in the same configuration file at runtime without recompiling the graph:

- `dumper.step()` detects configuration file changes and refreshes the switch.
- The new switch value takes effect from the subsequent ACLGraph replay.
- Dynamic refresh updates only `dump_enable`; modifying `task`, `level`, `list`, `custom_api`, or `slice` at runtime does not change the captured graph.

`nan_check` also refreshes the switch in `dumper.step()`, taking effect on subsequent replays. `dump_enable: true` enables tensor and detection JSON saving. Captured NaN checks and original-input copies execute during replay, with their computation and memory overhead remaining in both switch states.

The `statistics` and `tensor` tasks share the Host dynamic switch. Adjacent collection points with no interspersed model tasks are merged into one Host callback; during replay, each group reads the switch once, and if it is disabled, returns directly without copying tensors, computing statistical values, or saving data.

The `tensor` task retains the Python `contiguous()`, and the capture process manages the device memory required for contiguity; when collection is disabled, the existing contiguity operations in the graph still execute. The `statistics` task retains the original shape/stride, and only when collection is enabled does it copy the storage range and compute statistics on the CPU.

Collection callbacks are submitted in order through the TaskQueue. When the switch changes, the tool first waits for the callbacks on the current stream to complete, and then updates the Host switch; subsequent replays use the new state. When collection remains disabled, `statistics` adds no collection-specific synchronization, while `tensor` retains the original step synchronization and file archiving checks.

Rebuilding and reinstalling the `aclgraph_dump` C++ extension is required. The capture task query and destroy callback interfaces have been verified on CANN 9.1.0 and torch_npu 2.10.0.post4; older CANN versions and multi-stream concurrent switching have not yet been verified.

> [!NOTE]
>
> For the `statistics` task, calling `dumper.step()` with `dump_enable=false` does not create a dump directory or `dump.json`, but the step number still increments. When collection is re-enabled afterward, the results are written to the corresponding step directory. `dumper.step(dump=False)` also does not write to disk and does not increment the step number.

### Output Description

#### statistics Task

**Single-Rank Scenario**

The output path of `AclGraphDumper` is usually: `dump_path/step{step_id}/rank0/dump.json`. If the distributed environment is not initialized, the output path is: `dump_path/step{step_id}/pid{pid}/dump.json`.

Example of the generated directory:

```text
L0_dump
├── step0
│   └── rank0
│       └── dump.json
├── step1
│   └── rank0
│       └── dump.json
├── step2
|   └── rank0
|       └── dump.json
```

**Multi-Rank Scenario**

The output path of `AclGraphDumper` is `dump_path/step{step_id}/rank{rank_id}/dump.json`.

Example of the generated directory:

```text
L0_dump
├── step0
│   └── rank0
│   |    └── dump.json
│   └── rank1
│   |    └── dump.json
│   └── rank2
│         └── dump.json
```

#### tensor Task

The output path in the single-rank scenario is: `dump_path/step{step_id}/rank0/dump_tensor_data/*.pt`; if the distributed environment is not initialized, the output path is: `dump_path/step{step_id}/pid{pid}/dump_tensor_data/*.pt`. The output path in the multi-rank scenario is: `dump_path/step{step_id}/rank{rank_id}/dump_tensor_data/*.pt`.

```text
tensor_dump
├── dump_tensor_data
│   └── rank0                         # Temporary working directory of the current replay
├── step0
│   └── rank0
│       └── dump_tensor_data          # Archived data of step0
│           ├── linear.0.input.0.pt
│           └── linear.0.output.0.pt
└── step1
    └── rank0
        └── dump_tensor_data          # Archived data of step1
            ├── linear.1.input.0.pt
            └── linear.1.output.0.pt
```

`.pt` files can be read using `torch.load`.

The `tensor` task does not generate `dump.json`.

#### nan_check Task

Results are stored under `dump_path/step{step_id}/{process_dir}/`. After distributed initialization, `{process_dir}` is `rank{rank_id}` (`rank0` for a single rank); before initialization, it is `pid{pid}`.

The example below uses `detail: true`. With `detail: false`, only `.pt` files for calls with NaN outputs remain. Graph-mode output file types are `dump.json` and `.pt`.

- `detail: true` (default): `dump.json` records each input and output tensor's `dtype`, `shape`, and `is_nan`. The tensor's `is_nan` value is `1` for NaN, `0` for a passed NaN check, or `null` for an unknown detection result.
- Positional inputs are under `input_args`, keyword inputs under `input_kwargs`, and outputs under `output`.
- If any output tensor of the current module/API call contains NaN, all input/output tensors of that call are saved under `dump_tensor_data/`. `dump.json` links to the saved tensor files through `data_name`. `data_name` is generated only for calls with NaN outputs.
- `detail: false`: the output consists only of `.pt` files for calls with NaN outputs.

In this `detail: true` example, `input_args[1].is_nan` and `output[0].is_nan` of `Aten.add` are both `1`. The second positional tensor input and the first tensor output contain NaN, so both input tensors and the output tensor of this call are saved. Filenames use the format `{api_name}.{call_index}.{input|input_kwargs|output}.{index}.pt`; use `data_name` below to locate each file:

```text
nan_dump
└── step0
    └── rank0
        ├── dump.json
        └── dump_tensor_data
            ├── Aten.add.0.input.0.pt
            ├── Aten.add.0.input.1.pt
            └── Aten.add.0.output.0.pt
```

Example `dump.json` content (`dump_data_dir` is the actual save path):

```json
{
  "task": "nan_check",
  "level": "L1",
  "framework": "pytorch",
  "dump_data_dir": "/path/to/nan_dump/step0/rank0/dump_tensor_data",
  "data": {
    "Aten.add.0.forward": {
      "input_args": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [2],
          "is_nan": 0,
          "data_name": "Aten.add.0.input.0.pt"
        },
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [2],
          "is_nan": 1,
          "data_name": "Aten.add.0.input.1.pt"
        }
      ],
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [2],
          "is_nan": 1,
          "data_name": "Aten.add.0.output.0.pt"
        }
      ]
    }
  }
}
```

### Comparison Description

The `statistics` task can directly use `msprobe compare` to compare the network-wide collection results.
After the comparison is complete, a CSV report file is generated, for example: `compare_result_{rank_id}_{timestamp}.csv`.

In a distributed multi-process scenario, compare result files are usually generated by rank. Please view the results in conjunction with the rank dimension.

## Single-Point Collection

### Quick Start

The following example shows how to save a tensor during the forward pass:

```diff
  import torch
  import torch_npu

+ from msprobe.pytorch import acl_save


  class ToyModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = torch.nn.Linear(8, 4)

      def forward(self, x):
          y = self.linear(x)
+         # Save the intermediate tensor
+         acl_save(y, "./dump/linear_out.pt")
          return y


  if __name__ == "__main__":
      model = ToyModel().to("npu:0")
      x = torch.randn(2, 8, device="npu:0")
      out = model(x)
```

### Single-Point Collection Overview

#### Overview

`acl_save` is used to save tensor data. After being called, it generates a `.pt` file.

#### Interfaces

**Prototype**

```python
acl_save(x: torch.Tensor, path: str) -> torch.Tensor
```

**Parameter Description**

| Name | Optional/Mandatory | Description |
| --- | --- | --- |
| `x` | Mandatory | The tensor to be saved, of type `torch.Tensor`. |
| `path` | Mandatory | The save path (relative or absolute paths are supported), of type string. The actual file name written to disk will append a sequence number to the file name in this path, in the format `{base}_{seq}.pt`. For example, if `./dump/act.pt` is passed, the actual files written to disk will be `./dump/act_0.pt`, `./dump/act_1.pt`, and so on. |

**Return Value**

Return a tensor with the same shape as the input, used only to trigger the save operation.

#### Usage Example

1. Single-point saving during inference

    ```python
    from msprobe.pytorch import acl_save

    logits = model(x)
    acl_save(logits, "./dump/logits.pt")
    ```

2. Single-point collection in multi-rank scenarios

   ```python
   # In multi-rank scenarios, differentiate ranks as shown below.
   # Ensure that the "./dump/rank{torch.distributed.get_rank()}" directory has been created. Otherwise, a directory-not-found error will occur.
   acl_save(tensor, f'./dump/rank{torch.distributed.get_rank()}/tensor.pt')
   ```

### Output Description

After calling `acl_save`, `.pt` files are generated in the directory specified by path (the file names are automatically appended with sequence numbers), for example: `./dump/act_0.pt`, `./dump/act_1.pt`, `./dump/act_2.pt`.

### Data Parsing

The `.pt` file is in PyTorch serialization format and can be read using `torch.load`:

```python
import torch

tensor = torch.load("./dump/act_0.pt")
```

## Appendix

### FAQs

**1. Import error: Failed to import msprobe.lib.aclgraph_dump_ext**

Please verify:

- The compilation and installation are completed using `python3 build.py -e include-mod=aclgraph_dump -e no-check=true`;
- TorchNPU is installed and the environment variables are correctly configured.
- The current system is Linux.

**2. `Allocate SQ failed`**

On CANN versions earlier than 8.5.0 (excluding 8.5.0), `Allocate SQ failed` may occur due to SQ not being reused in older versions. You can work around this by changing `CurrentNPUStream` to `DefaultNPUStream` in `ccsrc/aclgraph_dump/aclgraph_dump.cpp`, or upgrade to CANN 8.5.0 or later.

**3. Runtime message: `[WARNING]: Invalid statistics detected. Please use tensor mode to collect the affected data.`**

When the statistical results contain statistics with a null value (that is, some statistics cannot be computed, such as min, max, mean, and norm recorded as null), the system triggers this message. The cause of uncomputable statistics is not limited to low-precision data types; some pre-allocated tensors (such as tensors created via torch.empty) can also cause this problem. If you need to further analyze the affected data, it is recommended to switch to real tensor data mode for collection. For details, see [Tensor Network-wide Collection](#tensor-network-wide-collection).
