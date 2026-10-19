# Overriding and Loading Precision Data in the PyTorch Scenario

## Introduction

The msProbe tool supports loading the module-level (L0) input tensor data collected previously to override the actual input of the target module during model forward execution. This function is used to isolate the accumulated error of preceding modules and precisely locate the accuracy problem of the current module and downstream modules.

**Features**

- **Configuration-driven**: Specify the load source and target modules through the `load` configuration section of config.json, without modifying the training script.
- **Exact matching**: The module entry name is exactly consistent with the `.pt` file name prefix produced by dump, avoiding mismatching.
- **Decoupling of overriding and dump**: Supports overriding without dumping (`dump_after_load=false`) or continuing to dump after overriding.
- **Automatic alignment**: Step and rank are automatically aligned with the current training progress, and manual specification is also supported.

## Preparations

**Environment Preparation**

Install the msProbe tool. For details, see [msProbe Installation Guide](../../install_guide/msprobe_install_guide.md).

**Prerequisites**

This function depends on existing module-level tensor data as the load source. The user must first perform a dump collection with `level=L0` or `level=mix` and `task=tensor`, producing the `.pt` files under the `dump_tensor_data/` directory, before configuring the `load` section for load overriding. For detailed dump collection operations, see [Precision Data Collection in the PyTorch Scenario](./pytorch_data_dump_instruct.md).

**Constraints**

- Only the PyTorch framework is supported.
- The source dump and the target model have the **same module structure and consistent module naming** (the dotted paths returned by `named_modules()` correspond exactly).
- If the model structure or naming is different (such as naming differences in train-infer frameworks, differences in model structure, differences in precision/sharding), the user needs to manually transform the source dump data (rename `.pt` files, align shapes, adjust sharding); this function does not involve such manual transformation.
- When the shape or dtype of the loaded tensor is inconsistent with the actual input of the current model, the tool prints a warning log but still performs the replacement, which may cause a forward error.
- load overriding replaces the input tensor of the module, and the backward data (backward input/output, parameter gradients) of the overridden module and its directly preceding module may be missing during dump collection. This is because the overridden tensor comes from `torch.load`, and the BackwardHook of msProbe is attached to the original tensor during forward, so the replaced tensor is not within the tracking scope of the hook. This limitation only affects dump data collection and does not affect the actual backward computation result of the model.

## Quick Start

The following uses a simple example to show how to use the load function to override module inputs and verify the overriding effect through accuracy comparison.

This example uses the same model and the same training script `train.py`, and is executed twice:

1. First execution: Normally collect precision data as the data source for overriding.
2. Second execution: Run with different input data, and enable the load function to override the module input collected in the first execution onto the corresponding module of this run.

The input data of the two runs is different in order to verify that the load overriding actually takes effect—if the inputs are the same, the results with and without overriding are exactly the same, and it is impossible to tell whether overriding really works.

1. Write the training script `train.py`, specifying the `config.json` path and the input offset through command-line arguments:

    ```diff
     import sys
     import torch
     import torch.nn as nn

    +# Import the data collection interface of the tool
    +from msprobe.pytorch import PrecisionDebugger, seed_all

    +# Fix randomness before model training begins
    +seed_all()
    +# Instantiate PrecisionDebugger before model training begins
    +debugger = PrecisionDebugger(config_path=sys.argv[1])

     class ModuleOP(nn.Module):
         def __init__(self):
             super().__init__()
             self.linear = nn.Linear(8, 4)
             self.relu = nn.ReLU()

         def forward(self, x):
             x1 = self.linear(x)
             r1 = self.relu(x1)
             return r1

     if __name__ == "__main__":
         model = ModuleOP()

    +    debugger.start(model=model) # Enable data dump

         x = torch.randn(10, 8)
         # In the second run, change the input through an offset so that the data of the two runs differs
         offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
         if offset:
             x = x + offset
         out = model(x)
         loss = out.sum()
         loss.backward()

    +    debugger.stop() # Disable data dump; data dump can be enabled again, and the collected data is recorded in the same step
    +    debugger.step() # End data dump; if data dump is enabled again, the collected data is recorded in the next step
    ```

2. First run: Normally collect precision data as the data source for overriding. Create `config_a.json`:

    ```json
    {
        "task": "tensor",
        "dump_path": "/home/data_dump_a",
        "rank": [],
        "step": [],
        "level": "L0",
        "async_dump": false,
        "extra_info": true
    }
    ```

    Command to execute (offset is 0):

    ```shell
    python train.py config_a.json 0
    ```

    After collection is complete, the following directory structure is generated under the directory specified by `dump_path`:

    ```text
    /home/data_dump_a/
    ├── step0/
    │   └── proc{pid}/
    │       ├── dump_tensor_data/
    │       │   ├── Module.linear.Linear.forward.0.input.0.pt
    │       │   ├── Module.linear.Linear.forward.0.output.0.pt
    │       │   ├── Module.relu.ReLU.forward.0.input.0.pt
    │       │   ├── Module.relu.ReLU.forward.0.output.0.pt
    │       │   └── ...
    │       ├── construct.json
    │       ├── dump.json
    │       └── stack.json
    └── ...
    ```

    Among them, the `.pt` files under the `dump_tensor_data/` directory are the source data to be loaded by the subsequent load function. Record the module entry name to be overridden, for example `Module.relu.ReLU.forward.0` (take the part before `.input.{i}.pt`).

3. Second run: Create `config_b.json`, add the `load` configuration section, and specify the data source path to override and the target module:

    ```json
    {
        "task": "tensor",
        "dump_path": "/home/data_dump_b",
        "rank": [],
        "step": [],
        "level": "L0",
        "load": {
            "path": "/home/data_dump_a",
            "modules": ["Module.relu.ReLU.forward.0"],
            "dump_after_load": true
        }
    }
    ```

    - `load.path` points to the source dump directory collected in step 2 (`/home/data_dump_a`), that is, the parent directory of `step{N}/`.
    - `load.modules` specifies the module entry name to be overridden (here the `relu` module is overridden, and the `linear` module is not). For how to obtain the entry name, see [modules Parameter Configuration Description](#modules-parameter-configuration-description).
    - `dump_after_load` is set to `true`, indicating that dump collection continues after overriding, and the overridden data is saved in the directory specified by `dump_path` (`/home/data_dump_b`).

4. Execute the second run (offset set to 1, so the input differs from the first run). The tool automatically loads the `Module.relu.ReLU` module input data collected in the first run and overrides the corresponding module input of this run:

    ```shell
    python train.py config_b.json 1
    ```

5. View the results. The overridden dump data is saved in the `/home/data_dump_b` directory, and the comparison function of the msProbe tool can be used to verify the overriding effect. Compare the dump data of the second run with that of the first run; if the Cosine similarity of the input and output of `Module.relu.ReLU.forward.0` is 1.0, the overriding takes effect:

    ```shell
    msprobe compare -tp /home/data_dump_b/step0 -gp /home/data_dump_a/step0 -o ./compare_result
    ```

    The comparison result is saved in a CSV file under the `./compare_result` directory. The key results are as follows:

    | NPU Name | Cosine |
    |----------|--------|
    | Module.linear.Linear.forward.0.input.0 | 0.69 |
    | Module.linear.Linear.forward.0.output.0 | 0.60 |
    | Module.relu.ReLU.forward.0.input.0 | 1.0 |
    | Module.relu.ReLU.forward.0.output.0 | 1.0 |

    Conclusion:

    - The `linear` module is not overridden, and the Cosine of its input and output is less than 1.0, indicating that the inputs of the two runs are indeed different (the offset takes effect).
    - The `relu` module is overridden, and the Cosine of its input and output is 1.0 (completely consistent), indicating that the `relu` module input of the second run is successfully overridden with the data collected in the first run.

    For more comparison operations, see [Precision Comparison in the PyTorch Scenario](../accuracy_compare/pytorch_accuracy_compare_instruct.md).

For detailed usage examples, see [Usage Examples](#usage-examples).

## Function Description

### Function Description

This function implements overriding of the input of preceding modules by adding the `load` parameter in the `config.json` file.

The tool performs input overriding in the `forward_pre_hook` of the module. The process is as follows:

1. When the module forward is called, the hook constructs `full_forward_name` (the format is consistent with the dump side).
2. It exactly matches `load.modules`, and if the match succeeds, the corresponding `.pt` file in the source dump is loaded.
3. It traverses all positional arguments (args) and keyword arguments (kwargs) of the module forward, and replaces tensor-type arguments with the source dump data.
4. Non-tensor arguments (such as int, str, bool) are automatically skipped (no `.pt` file is stored on the dump side).
5. The overridden input is passed to the module forward for execution.

### Notes

#### File Naming Consistency

The tensor file path loaded on the load side is strictly consistent with the file naming produced on the dump side, and the directory structure is the same:

| Parameter Type | File Name on the dump Side | Load Path on the load Side |
|----------|-------------|---------------|
| Positional argument args | `{full_forward_name}.input.{i}.pt` | `{load.path}/step{N}/rank{M}/dump_tensor_data/{full_forward_name}.input.{i}.pt` |
| Keyword argument kwargs | `{full_forward_name}.kwargs.{key}.pt` | `{load.path}/step{N}/rank{M}/dump_tensor_data/{full_forward_name}.kwargs.{key}.pt` |

For the directory structure and file naming rules on the dump side, see the [Dump Result File](./pytorch_data_dump_instruct.md#dump-result-file) in [Precision Data Collection in the PyTorch Scenario](./pytorch_data_dump_instruct.md#dump-result-file).

#### Handling of shape/dtype Mismatch

When the shape or dtype of the tensor in the source dump is inconsistent with the actual input of the target model:

- **Loading stage**: A warning log is printed (including the module name, parameter position, source shape/dtype, and current shape/dtype), and the replacement is still performed without skipping.
- **forward stage**: The PyTorch operator may report a `RuntimeError` (a shape mismatch usually causes an operator error; a dtype mismatch may result in an implicit conversion or an error).

Example warning log:

```text
[load] tensor mismatch for Module.blocks.0.attn.A.forward.0.input.0: source shape=[4, 32, 128] dtype=torch.float32 vs current shape=[8, 16, 128] dtype=torch.float16, override may cause forward error
```

**Note**: If the `forward` parameter order of model A and model B is different (for example, A is `forward(x, mask)` and B is `forward(mask, x)`), the tool loads by positional index—A's x is loaded into B's mask position. In this case the shapes may happen to match but the semantics are wrong, and the tool cannot automatically detect this; the user needs to ensure that the forward signatures of the two models are consistent.

#### Single-Card Scenario

In a single-card (non-distributed) scenario, the dump output is in the `proc{pid}` directory rather than the `rank{N}` directory. When loading, the tool automatically searches for the `proc*` directory under the `step{N}/` directory of the source dump, without the user specifying the pid.

#### Relationship Between load and dump level

The load function is decoupled from the `level` configuration of dump:

- load reads module-level (L0) tensor data, requiring the source dump to be collected with `level=L0` or `mix`.
- When load is enabled, module hooks are forcibly attached, so overriding works normally even if `level=L1` in `config.json`.
- When `dump_after_load=true`, the level of dump can be any value (L0/L1/mix, etc.), and the dump data is produced normally.

### Configuration Format

The `load` configuration section is a top-level field of `config.json`, at the same level as `task`, `dump_path`, `level`, etc. For the description of other parameters of `config.json`, see [config.json Introduction](./config_json_introduct.md).

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump_b",
    "level": "L0",
    "rank": [],
    "step": [],
    "load": {
        "path": "/home/data_dump_a",
        "modules": ["Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"],
        "step": [],
        "rank": [],
        "dump_after_load": false
    }
}
```

### Parameter Description

#### load Parameter Description

| Parameter | Optional/Required | Description |
|------|----------|------|
| path | Required | Path of the source dump directory, of type str, that is, the parent directory of `step{N}/`. For example, if the source dump directory is `/home/data_dump_a`, `load.path` is configured as `/home/data_dump_a`, and the tool automatically searches for the corresponding `.pt` file under `/home/data_dump_a/step{N}/rank{M}/dump_tensor_data/`. |
| modules | Required | List of module entry names to be overridden, of type list[str]. For a detailed introduction to the entry name, see [modules Parameter Configuration Description](#modules-parameter-configuration-description). |
| step | Optional | Specifies which steps to load, of type list, in the same format as the `step` parameter of dump (such as `[0]`, `["0-2"]`). The default is empty, indicating that loading is performed for every step. During loading, it is automatically aligned from the corresponding step of the source dump. For details, see [step/rank Parameter Configuration Description](#steprank-parameter-configuration-description). |
| rank | Optional | Specifies which ranks to load, of type list, in the same format as the `rank` parameter of dump. The default is empty, indicating that loading is performed for every rank. During loading, it is automatically aligned from the corresponding rank of the source dump. For details, see [step/rank Parameter Configuration Description](#steprank-parameter-configuration-description). |
| dump_after_load | Optional | Whether to continue dump collection after overriding, of type bool. `false` indicates only overriding without dumping; `true` indicates continuing to dump after overriding. The default is `false`. |

#### modules Parameter Configuration Description

Each entry in `load.modules` takes the part before `.input.{i}.pt` or `.kwargs.{key}.pt` in the `.pt` file name produced by dump. `.backward.{N}` is not supported, and duplicates are not allowed. The format is:

```text
Module.{dotted_path}.{ClassName}.forward.{N}
```

Here, `N` in `forward.{N}` is the sequence number of the module forward call (starting from 0) and **cannot be omitted**—in scenarios such as gradient accumulation, the same module is called multiple times, and `forward.0` and `forward.1` correspond to different calls, so it must be explicitly specified.

The entry name is obtained by viewing the `.pt` file names under the dump_tensor_data/ directory of the source dump. For example:

```text
File name: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.input.0.pt
Entry name: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0
Module name: Module.blocks.0.attn.MultiHeadSelfAttention

File name: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.kwargs.mask.pt
Entry name: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0
Module name: Module.blocks.0.attn.MultiHeadSelfAttention
```

**Entry format requirements:**

- It must start with `Module.`.
- It must end with `.forward.{N}`, where `N` is a non-negative integer; `.backward.{N}` is not supported.
- Duplicate entries are not allowed.
- The module path in the entry (the part after removing the `Module.` prefix and the `.forward.{N}` suffix) must exist in `named_modules()` of the current model; otherwise a warning log is printed at startup.

**Validation and feedback:**

The tool automatically validates the format and module path of `load.modules` at startup and outputs a validation result summary log, for example:

```text
[load] module validation: 2/5 modules valid in model
```

For entries whose module path does not exist or whose class name does not match, a warning is printed one by one. During training, if some entries are never hit because the call sequence number of `forward.{N}` does not match, the tool prints a warning at `debugger.stop()` indicating the entries that are not hit.

#### step/rank Parameter Configuration Description

`load.step` and `load.rank` are of type list, in the same format as the `step`/`rank` parameters of dump, supporting integers, single numeric strings, and range strings. They are used to control which steps/ranks are loaded; during loading, they are always automatically aligned from the corresponding step/rank of the source dump.

- Empty list `[]` (default): loading is performed for every step/rank, loaded from the corresponding step/rank of the source dump.
- Non-empty list such as `[0]`: only step/rank 0 is loaded, and the other steps/ranks are not loaded (no effect on the tool).
- Range string such as `["0-2"]`: only steps/ranks 0, 1, and 2 are loaded, each loaded from the corresponding step/rank of the source dump.

### Usage Examples

The quick start has demonstrated the complete process of dumping after overriding (corresponding to scenario 1). The following describes the configuration of other scenarios.

#### Scenario 1: Dump After Override (Localizing Accuracy Problems of the Current Module)

The example in the quick start is this scenario. For details, see [Quick Start](#quick-start).

Core configuration: `dump_after_load` is set to `true`, dump continues after overriding, and the overriding effect is verified through accuracy comparison.

#### Scenario 2: Override Only Without Dumping (Quickly Verifying the loss)

The user only wants to confirm whether the loss returns to normal after overriding the input of a certain module, and does not need dump data.

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump_b",
    "rank": [],
    "step": [],
    "level": "L0",
    "load": {
        "path": "/home/data_dump_a",
        "modules": ["Module.relu.ReLU.forward.0"],
        "rank": [],
        "step": [],
        "dump_after_load": false
    }
}
```

`load.path` points to the source dump directory `/home/data_dump_a`. When `dump_after_load` is `false`, no dump data is produced, and the user can judge the overriding effect by observing the loss.

#### Scenario 3: Overriding Multiple Modules

When the inputs of multiple modules need to be isolated at the same time, list multiple entry names in `modules`:

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump_b",
    "level": "L0",
    "rank": [],
    "step": [],
    "load": {
        "path": "/home/data_dump_a",
        "modules": [
            "Module.linear.Linear.forward.0",
            "Module.relu.ReLU.forward.0"
        ],
        "rank": [],
        "step": [],
        "dump_after_load": true
    }
}
```

After configuration, the inputs of all modules listed in `modules` are overridden by the source dump data, and the filling order is irrelevant to the loading order.

## Appendix

### Exception Handling

| Exception Scenario | Handling Method |
|----------|----------|
| `load.path` does not exist or is not a directory | Startup error (MsprobeException). |
| `load.modules` is an empty list | Startup error (to prevent the user from mistakenly thinking it is a full override). |
| `load.modules` is not of type list | Startup error. |
| `load.modules` entry format is invalid (does not start with `Module.`, does not end with `.forward.{N}`, contains `.backward.`, or is duplicated) | Startup error. |
| `load.modules` is configured, but `load.path` is not specified | Startup error. |
| The module name does not exist in the model | A warning log is printed at startup, and training continues normally. |
| The entry name is configured but is never hit at runtime (for example, the sequence number of `forward.{N}` does not match) | A warning log is printed at `debugger.stop()` indicating the entries that are not hit, and training continues normally. |
| The source `.pt` file is missing | A warning log is printed, the parameter keeps its original value, and the training program continues to run. |
| shape/dtype mismatch | A warning log is printed, the replacement is still performed, and forward may report an error. |
