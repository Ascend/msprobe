# Configuration File Introduction

- When the `PrecisionDebugger` interface is called to perform dump or other operations, the [config.json](../../../python/msprobe/config.json) file is required. If `config.json` is not specified, the default configuration is used.
- After msProbe is successfully installed, you can determine the path of the `config.json` file based on the msProbe installation path. Run the following command to check the msProbe installation path:

  ```shell
  pip show mindstudio-probe
  ```
  
  For example, if the msProbe installation path is `/usr/local/lib/python3.11/site-packages`, the `config.json` file is located in `/usr/local/lib/python3.11/site-packages/msprobe`.

## Parameters

### General Configuration

The table below describes the common configuration parameters.

| Parameter               | Required/Optional| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|-------------------| -------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task              | Optional    | Dump task type, which is of the string type. Value:<br>&#8226; "statistics": collects only statistics.<br>&#8226; "tensor": collects statistics and real data of the entire network.<br>&#8226; "acc_check": accuracy pre-check, which is supported only in PyTorch scenarios. Do not select this option during data collection.<br>&#8226; "overflow_check": overflow/underflow detection.<br>&#8226; "structure": collects only the model structure and call stack information.<br>The default value is "statistics".<br>You can configure scenario parameters based on the value of **task**.<br>&#8226; [task = statistics](#task = statistics)<br>&#8226; [task = tensor](#task = tensor)<br>&#8226; [task = acc_check](#task = acc_check)<br>&#8226; [task = overflow_check](#task = overflow_check)<br>&#8226; [task = structure](#task = structure)<br>&#8226; [task = exception_dump](#task = exception_dump)<br>Example: "task": "tensor"                                           |
| dump_path         | Required    | Dump data directory, which is of the string type.<br>Example: "dump_path": "./dump_path"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| rank              | Optional    | Data collection on a specified rank, which is of the list[Union[int, str]] type. By default, this parameter is not set, indicating that data of all ranks is collected. The value must be an integer greater than or equal to 0, a single numeric string such as "0", or a range string similar to "4-6", and the actual available rank ID must be configured.<br>&#8226; PyTorch: The rank ID starts from 0 and the maximum value is the total number of available ranks on all nodes minus 1. If the configured value is greater than the rank ID used for training, the dump data is empty. For example, if the rank IDs in the current environment range from 0 to 7 and the training uses ranks 0 to 3, configuring the rank ID to 4 or 10 (which does not exist) results in empty dump data.<br>&#8226; MindSpore: The rank ID starts from 0 and the maximum value is the total number of available ranks on each node minus 1. The rank configuration in the **config.json** file takes effect on all nodes at a time. Static graph L0 dump does not support rank configuration.<br>For single-rank training, **rank** must be an empty list, that is, **[]**.<br>Example: "rank": [1, "0", "4-6"]                                                                                                                                                                                                                                                    |
| step              | Optional    | Data of a specific step to be collected, which is of the list[Union[int, str]] type. By default, this parameter is not set, indicating that data of all steps is collected. To collect data of a specific step, specify a step existing in the training script. You can configure steps one by one, or use a single numeric string such as "0" or a range string such as "4-6".<br>Example: "step": [0, 1, "2", "4-6"]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| level             | Optional| Dump level, which is of the string type. Data is collected at different levels. Value:<br>&#8226; "L0": dumps module-level precision data. For details, see [Module-Level Precision Data Dump](#module-level-precision-data-dump).<br>&#8226; "L1": dumps API-level precision data. This is the default value and is supported only in PyTorch, MSAdapter, and MindSpore dynamic graph scenarios.<br>&#8226; "L2": dumps kernel-level precision data in different scenarios. For details, see [Kernel Precision Data Collection in PyTorch](./pytorch_kernel_dump_instruct.md), [Kernel Precision Data Collection in MindSpore Dynamic Graph Scenario](./mindspore_kernel_dump_instruct.md), and [Precision Data Collection in Static Graph Scenario](./mindspore_data_dump_instruct.md#precision-data-collection-in-static-graph-mode).<br>&#8226; "mix": dumps module-level and API-level precision data, that is, "L0" + "L1". This value is supported only in PyTorch, MSAdapter, and MindSpore dynamic graph scenarios.<br>&#8226; "debug": single-point saving. For details, see [Single-Point Saving Tool](./debugger_save_instruct.md).<br>Example: "level": "L1".|
| async_dump        | Optional    | Asynchronous dump switch, which is of the bool type. **task** can be in tensor or statistics mode, and **level** can be **L0**, **L1**, **mix**, or **debug**. The value can be **true** (enabled) or **false** (disabled). The default value is **false**.<br>If this parameter is set to **true**, asynchronous dump is enabled. That is, the collected precision data is flushed to drive after the current step training is complete. During training, the tool does not trigger the synchronization operation.<br>This mode may cause OOM risk. When **task** is set to **tensor**, that is, the asynchronous dump mode of real data, [list](#task = tensor) must be configured to specify the tensor to be dumped.<br>In this mode, **summary_mode** does not support MD5 values or statistical calculation of complex tensors.                                                                                                                                                                                                                                                                                                                                                                        |
| dump_enable       | Optional    | Dump switch, which is used to start or stop the **PrecisionDebugger** dump. The value is of the bool type. Value:<br>&#8226; **true**: enabled.<br>&#8226; **false**: disabled.<br>This parameter supports dynamic dump start or stop.<br>By default, this parameter is not set, indicating that data is not controlled and is dumped based on the static configuration.<br>For details, see [dump_enable Configuration Description](#dump_enable-configuration).<br>Example: "dump_enable": true                                                                                                                                                                                                                                                                                                                                                                                                            |
| extra_info        | Optional    | Whether to collect extra information and output related files (`stack.json` and `construct.json`). The value is of the bool type. Value:<br>&#8226; **true**: collect extra information and output `stack.json` and `construct.json`.<br>&#8226; **false**: do not collect extra information or generate `stack.json` and `construct.json`.<br>The default value is **true**.<br>Example: "extra_info": false                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| precision         | Optional    | Precision used for calculating statistics. The value is of the string type. The values are **high** and **low**. The default value is low. If this parameter is set to **high**, float32 is used for calculation, which increases device memory usage and improves precision. However, when large values are processed, OOM may occur. If this parameter is set to **low**, the same type as the original data is used for calculation, which occupies less device memory.<br>PyTorch, MindSpore dynamic graph, and MindSpore static graph O0/O1 scenarios are supported.<br>**task** can be set to **statistics** or **tensor**, and **level** can be set to **L0**, **L1**, **mix**, or **debug**.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| risk_level        | Optional    | API risk level filtering, which is of the string type and takes effect only in the PyTorch scenario where **level** is set to **L1** or **mix**. Value:<br>&#8226; "ALL": dumps data of all APIs.<br>&#8226; "CORE": dumps data of core APIs (high risk, prone to precision issues), including converged computing, communication, and precision computation.<br>&#8226; "FOCUS": dumps data of core APIs and focused APIs, excluding low-risk APIs (such as shape transformation APIs like reshape, transpose, permute, to, and view). This is the **default value**.<br>Example: "risk_level": "CORE".                                                                                                                                                                                                                                                                                                                                                                                                             |

#### Module-Level Precision Data Dump

For foundation models, training scripts are not simply ported from GPUs to NPUs by using the automatic porting capability. Instead, a series of targeted adaptations are performed on the NPU network. Therefore, some molecular structures of the NPU model cannot completely correspond to the original GPU model. Inconsistent model structure may result in inconsistent API calling types and quantity. If precision data is dumped and compared based on the API granularity, it is impossible to compare all APIs.

This section describes how to dump data of large-granularity modules in a model so that the modules that cannot be compared by API can be directly compared by module.

A module refers to a subclass that inherits the **nn.Module** class (in the PyTorch and MSAdapter scenarios) or the **nn.Cell** class (in the MindSpore scenario). Generally, such a module is a small model and can be regarded as a whole. Data is dumped by module.

In the PyTorch scenario, to avoid the framework restriction that prevents in-place operations on the output of the **BackwardHook** function, the tool uses `torch._C._autograd._set_creation_meta` to reset the attributes of the output tensor of **BackwardHook**. Consequently, the dumped data may be missing reverse data for in-place operations—**such as nn.ReLU(inplace=True)**—and their preceding modules.

### task = statistics

**Examples**

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,
    "extra_info": true,

    "statistics": {
        "scope": [], 
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

Supported scenarios:

 - PyTorch
 - MindSpore static graph
 - MindSpore dynamic graph

**Parameters**

| Parameter        | Required/Optional| Description                                                        |
| ------------ | --------- | ------------------------------------------------------------ |
| scope        | Optional     | Dump range in the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios. The value is of the list[str] type. By default, this parameter is not configured. (If neither **list** nor **scope** is configured, data of all APIs is dumped.) For details, see [scope Configuration](#scope-configuration).|
| list         | Optional     | List of operators whose data is to be collected. The value is of the list[str] type. By default, this parameter is not configured. (If neither **list** nor **scope** is configured, data of all APIs is dumped.) For details, see [list Configuration](#list-configuration).|
| tensor_list  | Optional     | List of operators whose real data is to be collected. The value is of the list[str] type. By default, this parameter is not configured. For details, see [tensor_list Configuration](#tensor_list-configuration).<br>L0, L1, and mix levels are supported in the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios.<br>This parameter is not supported in the MindSpore static graph scenario.|
| device       | Optional     | Device used for calculating statistics. The value can be **device** or **host**. The default value is **host**. Using device for calculation has higher performance than using host. Only min, max, avg, and l2norm statistics are supported.<br>This parameter is supported only in the MindSpore static graph O0/O1 scenario.|
| data_mode    | Optional     | Dump data filtering. The value is of the list[str] type. For details, see [data_mode Configuration](#data_mode-configuration).|
| summary_mode | Optional     | Dump file output mode. PyTorch, MSAdapter, and MindSpore dynamic/static graphs, jit_level=O2 (L2), and jit_level=O0/O1 (L0) are supported. For details, see [summary_mode Configuration](#summary_mode-configuration).|

### task = tensor

**Examples**

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "async_dump": false,

    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"],
        "bench_path": "/home/bench_data_dump",
        "summary_mode": "md5",
        "diff_nums": 5        
    }
}
```

Supported scenarios:

 - PyTorch
 - MindSpore static graph
 - MindSpore dynamic graph

**Parameters**

| Parameter        | Required/Optional| Description                                                                                                                                                                                                                                                                     |
| -------------- | -------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scope          | Optional    | Dump range in the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios. The value is of the list[str] type. By default, this parameter is not configured. (If neither **list** nor **scope** is configured, data of all APIs is dumped.) For details, see [scope Configuration](#scope-configuration).                                                                                   |
| list           | Optional    | List of operators whose data is to be collected. The value is of the list[str] type. By default, this parameter is not configured. (If neither **list** nor **scope** is configured, data of all APIs is dumped.) For details, see [list Configuration](#list-configuration).                                                                                                                 |
| data_mode      | Optional    | Dump data filtering. The value is of the list[str] type. For details, see [data_mode Configuration](#data_mode-configuration).                                                                                                                                    |
| file_format    | Optional    | Format of the saved tensor data. The value is of the string type. This parameter can be configured only for L2 in the MindSpore static graph scenario. Value:<br>&#8226; "bin": The dumped tensor file is in binary format.<br>&#8226; "npy": The suffix of the dumped tensor file is .npy.<br>The default value is **npy**.                                                                                    |
| summary_mode  | Optional| Dump file output mode. PyTorch, MSAdapter, and MindSpore dynamic graphs are supported. Value:<br>&#8226; "md5": The dump output contains the **dump.json** file that contains the CRC-32 value and API statistics, which is used to verify data integrity.<br>&#8226; "statistics": The dump output contains only the **dump.json** file that contains API statistics.<br>&#8226; "xor": supported only in the PyTorch scenario. The dump output contains only the XOR binary check value (labeled **md5**), and does not contain the max, min, mean, and L2norm statistics.<br>The default value is **statistics**.                            |
| bench_path      | Optional    | Automatically controls the real-time MD5 difference analysis during PyTorch deterministic problem locating. That is, the MD5 data with differences is dumped. The value is of the string type. By default, this parameter is not configured.<br>**bench_path** must be set to the preset MD5 data path (that is, **summary_mode** is set to **md5** during the last dump operation), and **summary_mode** must also be set to **md5** during the current dump operation.<br>After this parameter is configured, the difference between each tensor in the current task and the preset MD5 data is checked. Once a discrepancy is identified, the actual data is dumped.<br>Example: "bench_path": "./bench_dump_path"|
| diff_nums      | Optional    | Maximum number of differences. The value is of the integer type. The default value is **1**. This parameter is supported only in the PyTorch MD5 real-time difference analysis scenario (that is, **bench_path** is configured). After *N* differences occur, difference analysis is not performed. Input-output data associated with detected differences will be dumped during the process.<br>Setting it to **-1** enables continuous overflow/underflow monitoring till training completion.<br>Example: "diff_nums": 3|

### task = acc_check

**Examples**

```json
{
    "task": "acc_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",

    "acc_check": {
        "white_list": [],
        "black_list": [],
        "error_data_path": "./"
    }
}
```

Supported scenarios:

 - PyTorch

**Parameters**

| Parameter       | Required/Optional| Description                  |
| --------------- | ------------ | ------------------------ |
| white_list      | Optional    | API dump trustlist, which is used to dump only specified APIs. By default, no trustlist is configured, that is, all API data is dumped.<br>Example: "white_list": ["conv1d", "conv2d"]|
| black_list      | Optional    | API dump blocklist, which is used to block dump of specified APIs. By default, no blocklist is configured, that is, all API data is dumped.<br>Example: "black_list": ["conv1d", "conv2d"]|
| error_data_path | Optional    | Path for storing the input and output data of APIs whose precision does not meet the requirements. The default value is the current path.<br>Example: "error_data_path": "./"|

If both **white_list** and **black_list** are configured and the API lists configured by them do not overlap, the trustlist takes effect. If the API lists overlap, the APIs excluded by the trustlist and the overlapped APIs are not dumped.

### task = overflow_check

**Examples**

```json
{
    "task": "overflow_check",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2",

    "overflow_check": {
        "check_mode": "all"
    }
}
```

Supported scenarios:

 - MindSpore static graph

**Parameters**

In the MindSpore static graph scenario, **level** must be set to **L2** and the model compilation optimization level (**jit_level**) must be set to **O2**.

| Parameter       | Required/Optional| Description                |
| ------------- | -------- | ---------------------- |
| check_mode    | Optional    | Overflow/Underflow type, which is of the string type. This parameter is supported only in the static graph scenario of MindSpore earlier than v2.3.0. The options are as follows:<br>&#8226; "aicore": AI Core<br>&#8226; "atomic": Atomic<br>&#8226; "all": operator<br>The default value is **all**.<br>Example: "check_mode": "all"|

### task = structure

Only the model structure is collected. No other special configuration is required.

**Examples**

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

Supported scenarios:

 - PyTorch
 - MindSpore dynamic graph

### task = exception_dump

In the MindSpore dynamic graph scenario, **level** must be set to **L2**. In the MindSpore static graph scenario, **level** must be set to **L2** and **jit_level** must be set to **O0** or **O1**.

During the running, the intermediate file **kernel_graph_exception_dump.json** is generated in the specified directory. This file contains the settings related to exception dump.

For details about other dump result files except the intermediate file, see [Dump in Ascend O0/O1 Mode](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/dump.html).

**Examples**

```json
{
    "task": "exception_dump",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2"
}
```

Supported scenarios:

 - MindSpore dynamic graph
 - MindSpore static graph

## Appendixes

### dump_enable Configuration

- `dump_enable` is used to control the dynamic dump start and stop capabilities of `PrecisionDebugger`. If the value is `true`, dump collection is enabled. If the value is `false`, dump collection is disabled. It is recommended to set this parameter only when dynamic collection control is required. The initial value is `false`.
- In the PyTorch scenario, if this field is configured during `PrecisionDebugger` initialization, the tool automatically reads `config_path` and updates the configuration during execution.
- Recommended process: Disable this function in the common training or inference phase. Enable this function when you need to locate a fault. After the fault is located, disable this function to reduce interference to the service process.
- In the `vllm` scenario, if `level` needs to be changed, it is recommended to set the initial value of `level` to `L0`. This ensures that the subsequent `level` can be switched randomly. If the initial value of `level` is not `L0`, `level` may fail to be switched.

**Examples**

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L1",
    "dump_enable": false,
    "statistics": {
        "summary_mode": "statistics"
    }
}
```

> Note: `dump_enable` is configured only when dump needs to be dynamically enabled or disabled. During execution, you can change the value of `dump_enable` from `false` to `true` (or `true` to `false`) to dynamically enable or disable dump. Modifications to other fields in the JSON file also take effect.

Supported scenarios:

 - PyTorch

### list Configuration

- In the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios, a full API name must be configured for data dump. In the PyTorch scenario, if `level` is set to `L2`, `list` is mandatory.

  Example: "list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.backward"]

- In the PyTorch and MindSpore dynamic graph scenarios, if `level` is set to `mix`, you can configure a module name to dump all data from the start to the end of its execution.

  Example: "list": ["Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"] or "list": ["Cell.network_with_loss.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]

- In the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios, you can specify a type of APIs to dump their input and output data.

  Example: "list": ["relu"]

  In the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios, if `level` is set to `mix`, data is dumped for both APIs and modules whose names contain any string from the configured `list`. For modules, the dump includes all data from the start to the end of execution.

- In the MindSpore static graph scenario, `kernel_name` can be set to an operator name list, an operator type (not supported when **jit_level** is set to **O2**), or a regular expression of the operator name (when the string is in the "name-regex(xxx)" format).

  Example: list: ["name-regex(Default/.+)"]

  All operators whose names start with **Default/** can be matched.

### scope Configuration

You can configure two module or API names within square brackets ([]). The list must contain exactly two entries, each specified using the complete tool naming format to precisely lock the range and dump data within it.

Example: "scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"], "scope": ["Cell.conv1.Conv2d.forward.0", "Cell.fc2.Dense.backward.0"], or "scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]<br>The value depends on the value of `level`. When `level` is set to `L0`, the module name can be configured. When `level` is set to `L1`, the API name can be configured. When `level` is set to `mix`, the module name or API name can be configured.

### tensor_list Configuration

In the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios, you can specify a type of APIs or modules to dump their input and output statistics and complete tensor data.<br>Example: "tensor_list": ["relu"]<br>

### data_mode Configuration

- In the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios, the options are **all**, **forward**, **backward**, **input**, and **output**. Except **all**, other options can be combined freely. The default value is **all**, indicating that all dump data is saved.

  Example: "data_mode": ["backward"] (only backward data is saved) or "data_mode": ["forward", "input"] (only forward input data is saved).

- MindSpore static graph scenario: The L0 dump supports only **all**, **forward**, and **backward**. The L2 dump supports only **all**, **input**, and **output**. In addition, the options can be configured only separately and cannot be combined freely.

  Example: "data_mode": ["all"]

### summary_mode Configuration

- PyTorch, MSAdapter, and MindSpore dynamic graph

  The value is of the string type.

  The options are as follows:

  - **md5**: The dump output contains the **dump.json** file that contains the CRC-32 value and API statistics, which is used to verify data integrity.
  - **statistics**: The dump output contains only the **dump.json** file that contains API statistics. The default value is **statistics**.
  - **xor**: supported only in the PyTorch scenario. The dump output contains only the XOR check value (labeled as **md5**), and does not contain the max, min, mean, and L2norm statistics.

  Example: "summary_mode": "md5"

- MindSpore Static Graph

  The value is of the string or list[str] type.
  
  - L2 (jit_level=O2): In addition to **md5** and **statistics**, the statistical item list can be configured. The optional statistical items are **max**, **min**, **mean**, and **l2norm**. You can select any combination of them. The results of mean and l2norm are in float format.
  - L2 (jit_level=O0/O1): In addition to **md5** and **statistics**, the statistical item list can be configured. The optional statistical items are **max**, **min**, **mean**, **l2norm**, **count**, **negative zero count**, **zero count**, **positive zero count**, **nan count**, **negative inf count**, **positive inf count**, and **hash**. You can select any combination of them. **hash** calculates the MD5 value in MindSpore 2.7.0 and earlier versions, and calculates the SHA1 value in later versions.
  - L0 (jit_level=O0/O1): Only **statistics** and any combination of **max**, **min**, **mean**, and **l2norm** are supported.
  
  Example: "summary_mode": ["max", "min"]

> [!NOTE]NOTE
>
> In the PyTorch, MSAdapter, and MindSpore dynamic graph scenarios, when **summary_mode** is set to **md5**, the CRC-32 algorithm is used. In the MindSpore static graph scenario, when **summary_mode** is set to **md5**, the MD5 algorithm is used.
