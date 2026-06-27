# Inference in Torch Graph Mode (TorchAir)

## Overview

In the inference scenario of Torch graph mode (TorchAir), precision data collection gathers input and output data of intermediate operators during model inference by setting **CompilerConfig**.

## Basic Concepts

- [TorchAir](https://gitcode.com/Ascend/torchair/blob/26.0.0/docs/zh/ascend_ir/api/torchair/compiler_config.md): backend of the Torch graph mode, which is used to compile the Torch model into an executable program on the Ascend AI Processor, including computational graph conversion and optimization.
- Graph Engine (GE): core component of the Ascend AI computing platform. As the control center for computational graph build and execution, GE provides the following functions:
    - Graph optimization: optimizes the neural network computational graph to improve execution efficiency.
    - Graph build management: converts models of different frameworks into a unified internal representation, providing a basis for graph optimization and execution.
    - Graph execution control: efficiently executes the optimized graph.
- PyTorch FX/TorchFX/PyTorch Functional eXecution (FX): function execution and graph conversion framework of PyTorch. It provides the following functions:
    - Converts PyTorch code into an IR graph that can be optimized for graph optimization and analysis.
    - Captures and converts programs to support model reconstruction and optimization.
- Fusion: fuses multiple operators in a neural network into one operator to reduce compute workload and memory usage.
- Collection: collects the input and output data of intermediate operators during model inference and saves the data as files.

## Preparations

### Environment Setup

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md#installation-description).

### Constraints

- Only the inference in Torch graph mode (TorchAir) is supported.

## Dumping Data in GE Fusion Mode

**Description**

- Obtains the configured `config` instance by calling `set_ge_dump_config` or adds dump configuration to the existing `config` instance, configures model compilation, and performs inference.

**Prototype**

```Python
set_ge_dump_config(
        dump_path='',
        dump_mode='all',
        fusion_switch_file=None,
        dump_token=None,
        dump_layer=None,
        compiler_config=None
)
```

**Parameters**

| Parameter            | Mandatory (Yes/No)                                                                         | Description                              |
| ------------------ | --------------------------------------------------------------------------------- | -------------------------------------- |
| dump_path          | No| Path for storing dump data. The default value is **"./"**.                                                               |
| dump_mode          | No| Data dump mode, specifying whether the operator input or output is dumped. The value can be **input**, **output**, or **all**. The default value is **all**, indicating that both the input and output data are dumped.|
| fusion_switch_file | No| Whether to disable dump in fusion mode. The default value is **None**, indicating that fusion is enabled.                                                             |
| dump_token         | No| Token for dump. The format **[1,2,5]** indicates that the data of the first, second, and fifth tokens is dumped. The default value is **None**, indicating that all data is dumped.          |
| dump_layer         | No| Layer for dump. The format **["Add", "Conv_1"]** indicates that the data of the Add and Conv_1 layers is dumped. The default value is **None**, indicating that all data is dumped.          |
| compiler_config    | No| Graph build configuration (CompilerConfig object). The default value is **None**, indicating that the newly created graph build configuration is returned.                    |

**Example (CompilerConfig Object Created)**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config # Import config.
  ...
  model = ...
  config = torchair.CompilerConfig()
  # Add the dump configuration to the existing config.
  set_ge_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**Example (CompilerConfig Object Not Created)**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config # Import config.
  ...
  model = ...
  # Add config.
  config = set_ge_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

### Dump Result File

The dump data is stored in `{dump_path}/msprobe_ge_dump`. `{dump_path}` is the path specified by **dump_path** of the `set_ge_dump_config` API, and `msprobe_ge_dump` is the directory automatically created by msProbe.

When Ascend Extension for PyTorch 7.1.0 or later is used, the directory structure of the result file is as follows:

```ColdFusion
├── ${dump_path}
│   ├── msprobe_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${time}
|   |   |   |   ├── ${device_id}
|   |   |   |   |   ├── ${model_name}
|   |   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── └── # Data in binary format
```

When a version earlier than Ascend Extension for PyTorch 7.1.0 is used, the result file directory structure is as follows:

```ColdFusion
├── ${dump_path}
│   ├── msprobe_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── ${time}
|   |   |   ├── ${device_id}
|   |   |   |   ├── ${model_name}
|   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── # Data in binary format
```

## Dumping Data in FX Mode

**Description**

- Obtains the configured `config` instance by calling `set_fx_dump_config` or adds dump configuration to the existing `config` instance, configures model compilation, and performs inference.

**Prototype**

```Python
set_fx_dump_config(dump_path='', op_list=None, compiler_config=None)
```

**Parameters**

| Parameter         | Mandatory (Yes/No)                                            |  Description                     |
| --------------- | ---------------------------------------------------- | -------------------------------------- |
| dump_path       | No| Path for storing dump data. This parameter is valid only when Ascend Extension for PyTorch 7.0.0 or later is used. The default value is **"./"**.|
| op_list         | No| Operator for dump. The format **["Add", "Conv_1"]** indicates that the data of the Add and Conv_1 layers is dumped. The default value is **None**, indicating that all data is dumped.|
| compiler_config | No| Graph build configuration (CompilerConfig object). The default value is **None**, indicating that the newly created graph build configuration is returned.                    |

**Example (CompilerConfig Object Created)**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config # Import config.
  ...
  model = ...
  config = torchair.CompilerConfig()
  # Add the dump configuration to the existing config.
  set_fx_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**Example (CompilerConfig Object Not Created)**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config # Import config.
  ...
  model = ...
  # Add config.
  config = set_fx_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**Example (Operator Specified for Dump)**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config # Import config.
  ...
  model = ...
  # Add config.
  config = set_fx_dump_config(dump_path="dump", op_list=['Add', 'Conv_1'])
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

### Dump Result File

The dump data is stored in `{dump_path}/msprobe_fx_dump`. `{dump_path}` is the path specified by **dump_path** of the `set_fx_dump_config` API, and `msprobe_fx_dump` is the directory automatically created by msProbe.

When a version later than Ascend Extension for PyTorch 7.1.0, the result file directory structure is as follows:

```ColdFusion
├── ${dump_path}
│   ├── msprobe_fx_dump
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${model_name}
|   |   |   |   ├── ${token_id}
└── └── └── └── └── └── # Data in npy format
```

When Ascend Extension for PyTorch 7.1.0 is used, the result file directory structure is as follows:

```ColdFusion
├── ${dump_path}
│   ├── msprobe_fx_dump
│   |   ├── data_dump
|   |   |   ├── ${token_id+1}
|   |   |   |   ├── gm_${time}_dump
└── └── └── └── └── └── # Data in npy format
```

When a version earlier than Ascend Extension for PyTorch 7.1.0 is used, the result file directory structure is as follows:

```ColdFusion
├── . # Current working directory
│   ├── data_dump
|   |   ├── ${token_id+1}
|   |   |   ├── gm_${time}_dump
└── └── └── └── └── # Data in npy format
```

## Dumping Data in GE Mode with Fusion Disabled

In addition to operations in [Dumping Data in GE Fusion Mode](#dumping-data-in-ge-fusion-mode), pass the configuration file for disabling operator fusion through the `fusion_switch_file` parameter of the `set_ge_dump_config` API. The following is an example of tool usage:

- Create the `fusion_switch.json` file for disabling operator fusion. For details about operator fusion patterns, see [Operator Fusion Pattern Settings (fusion_switch_file)](https://gitcode.com/Ascend/torchair/blob/26.0.0/docs/zh/ascend_ir/features/advanced/fusion_switch_file.md).

  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```

- Call the `set_ge_dump_config` API in the inference script.

**Example (CompilerConfig Object Created)**

  ```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config # Import config.
  ...
  model = ...
  config = torchair.CompilerConfig()
  # Add the dump configuration to the existing config.
  set_ge_dump_config(dump_path="dump_fusion_off",
                     fusion_switch_file="fusion_switch.json", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

**Example (CompilerConfig Object Not Created)**

  ```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config # Import config.
  ...
  model = ...
  # Add config.
  config = set_ge_dump_config(dump_path="dump_fusion_off", fusion_switch_file="fusion_switch.json")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

- For details about the directory structure and description of the dump result files, see [Dump Result File](#dump-result-file).

For details about the complete comparison cases and usage methods, see [Network-wide Operator Accuracy Comparison in Torch Graph Mode](../accuracy_compare/torchair_compare_instruct.md#overview).
