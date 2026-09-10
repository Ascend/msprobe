# aclgraph_dump User Guide

<!-- md-trans-meta sourceCommit=4d1f90c4053f5cf6d083f7bc3060921d6a055bb8 translatedAt=2026-08-12T08:31:03.566Z pushedAt=2026-08-12T08:42:11.631Z -->

## Introduction

When performing precision alignment in **PyTorch ACLGraph** mode, the overall strategy is as follows: **first network-wide screening, then single-point deep dive**. Use **network-wide collection** to quickly narrow down the anomaly scope, and then perform **single-point collection** on suspected problematic operators and save their tensor data for fine-grained investigation. `aclgraph_dump` provides the following collection capabilities:

- Single-point collection: `acl_save`
- Network-wide collection: `AclGraphDumper`

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
- Data collection and result analysis in low-precision scenarios (`fp8`/`fp4`) are currently not supported. It is recommended to use conventional precisions such as `fp16`/`bf16`/`fp32` for ACLGraph troubleshooting.

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
        "seq_len": 1024
    }
    }
    ```

    **Reference**

    The currently supported configuration items for network-wide `aclgraph dump` are as follows:

    | Configuration Item | Optional/Mandatory | Description |
    | --- | --- | --- |
    | `task` | Optional | Collection task type, string. The default value is `statistics`. Full-network `aclgraph dump` currently only supports `statistics`. |
    | `dump_path` | Mandatory | Dump result output directory, string. The tool checks and creates this directory. |
    | `rank` | Optional | Specifies the rank(s) to collect, list[int \| str]. The default value is empty, meaning all ranks are collected. Strings only support the `start-end` range format. Full-network collection is not enabled for non-target ranks. |
    | `level` | Optional | Root-level collection level, string. Supports `L0`, `L1`, and `mix`. The default value is `L0`.<br>`L0` collects module input/output statistics; `L1` collects API input/output statistics; and `mix` collects both module and API statistics. |
    | `list` | Optional | Module name keyword filter list, list[str]. The default value is empty, meaning all modules are collected. |
    | `seq_len` | Optional | Takes effect only for `statistics`, integer. The default value is `0`, meaning statistics are computed on the full tensor. When `seq_len > 0` and the tensor is large enough, statistics are computed only on the leading `seq_len` slice, which can be used to skip padding tail data in graph mode. |

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

### Network-wide Collection

#### Overview

`AclGraphDumper` is used for network-wide collection of intermediate data. It currently supports statistical value collection at the module level, API level, and module + API level. The results include information such as tensor shapes, data types, and statistical values.
The initialization and `start` call of `AclGraphDumper` must be completed before model graph compilation (such as `torch.npu.graph` or `torch.compile`).

#### Interfaces

**Prototype**

```python
AclGraphDumper(config_path: str | None = None)
```

**Parameter Description**

| Name | Optional/Mandatory | Description |
| --- | --- | --- |
| `config_path` | Optional | Configuration file path, string. If not passed, the built-in `config.json` in the msProbe package is read by default. `dump_path`, `task`, `rank`, `level`, `list`, and `seq_len` are read from this configuration file. |

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
| `dump` | bool | Whether to write the current statistics results to `dump.json`. `True`: clear statistics and write to disk; `step_id` increments by 1; `False`: only clear statistics without writing to disk; `step_id` does not increment (can be used during the `dummy_run` warm-up phase). | No |

If collection has not been started, it returns directly.

### Output Description

**Single-Rank Scenario**

In the single-rank scenario, the output path of `AclGraphDumper` is: `dump_path/step{step_id}/pid{pid}/dump.json`.

Example of the generated directory:

```text
L0_dump
├── step0
│   └── pid9527
│       └── dump.json
├── step1
│   └── pid9527
│       └── dump.json
├── step2
|   └── pid9527
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

### Comparison Description

You can directly use `msprobe compare` to compare the network-wide collection results.
After the comparison is complete, a CSV report file will be generated, for example: `compare_result_{rank_id}_{timestamp}.csv`.

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

### Single-Point Collection

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
   acl_save(tensor,f'./dump/rank{torch.distributed.get_rank()}/tensor.pt')
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

- The `--include-mod=aclgraph_dump` option is included during compilation and installation.
- TorchNPU is installed and the environment variables are correctly configured.
- The current system is Linux.

**2. `Allocate SQ failed`**

On CANN versions earlier than 8.5 (excluding 8.5), `Allocate SQ failed` may occur due to SQ not being reused in older versions. You can work around this by changing `CurrentNPUStream` to `DefaultNPUStream` in `ccsrc/aclgraph_dump/aclgraph_dump.cpp`, or upgrade to CANN 8.5.0 or later.
