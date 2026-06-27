# Precision Data Collection in MSAdapter

## Overview

msProbe collects the precision data during model running by adding the `PrecisionDebugger` API to the MSAdapter model training script and starting training. The tool can collect precision data of different levels in MSAdapter scenarios.

For details about performance changes in dump statistics mode and data volume collected in dump tensor mode, see [Dump Baseline] (../baseline/mindspore_data_dump_perf_baseline.md).

**Note**:

* To correctly identify the MSAdapter scenario, import the torch module before importing msProbe.

* Due to the limitation of the automatic differentiation mechanism of the MSAdapter model, the dumped data may be missing reverse data for in-place operation module/API and its previous module/API.

* Loss or gnorm may change after msProbe is used. For details, see [Cause Analysis of Model Calculation Result Changes](../faq.md#model-computation-result-change).

**Concepts**

* **MSAdapter**: a MindSpore ecosystem adaptation tool that can efficiently migrate PyTorch training scripts to MindSpore for execution. In this way, PyTorch code can achieve high performance in the Ascend environment without changing development habits of PyTorch users.

* **dump**: a process of collecting precision data and completing data persistence.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

Only the MSAdapter framework is supported.

## Quick Start

1. Create a configuration file.

    Create a `config.json` file in the current directory to configure dump parameters. The file content is as follows:

    ```json
    {
        "task": "statistics",
        "dump_path": "./output",
        "rank": [],
        "step": [0, 1],
        "level": "L0",
        "statistics": {
            "scope": [],
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics"
        }
    }
    ```

    For details about parameter configuration, see [Configuration File Introduction](./config_json_introduct.md).

2. Compile a model training script.

    Create a Python script file, for example, `net.py` in the current directory. The script content is as follows:

    ```python
    import mindspore as ms
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Import the data collection API of the tool.
    from msprobe.mindspore import PrecisionDebugger

    # Instantiate PrecisionDebugger before model training.
    debugger = PrecisionDebugger(config_path='./config.json')


    # Define a network.
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features=8, out_features=4)
            self.linear2 = nn.Linear(in_features=4, out_features=2)

        def forward(self, x):
            x1 = self.linear1(x)
            x2 = self.linear2(x1)
            logits = F.relu(x2)
            return logits


    net = Net()


    def train_step(inputs):
        return net(inputs)


    if __name__ == "__main__":
        data = (torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 8))
        grad_fn = ms.value_and_grad(train_step, grad_position=0)

        for inputs in data:
            # Enable data dump.
            debugger.start(model=net)

            out, grad = grad_fn(inputs)

            # Disable data dump.
            debugger.stop()
            # Update step information.
            debugger.step()
    ```

3. Train your model.

    Run the following command to train the model:

    ```bash
    python net.py
    ```

    The tool collects precision data during model training.

4. Check data collection results.

    If the following information is displayed, data collection is completed. You can manually stop model training and view the collected data.

    ```markdown
    ****************************************************************************
    *                        msprobe ends successfully.                        *
    ****************************************************************************
    ```

## MSAdapter Dump Functions

### Function Description

In the MSAdapter scenario, msProbe can collect precision data of L0, L1, and mix levels.

* **L0 level**: collects the input and output data of a module object `(nn.Module)`. This level is applicable to the scenario where a specific network module needs to be analyzed.

* **L1 level**: collects the input and output data of APIs. This level is applicable to the scenario where API-level precision problems need to be located.

* **mix level**: collects module-level and API-level data. This level is applicable to the scenario where module-level and API-level precision problems need to be analyzed.

The MSAdapter model uses MindSpore as its underlying framework. Therefore, the msProbe tool interfaces for collecting precision data are the same as the dump interfaces used in MindSpore dynamic graph scenarios.

* **msprobe.mindspore.seed_all**: Fixes randomness in the network and enables deterministic computing.

* **msprobe.mindspore.PrecisionDebugger**: Loads the dump configuration file to determine the detailed dump configuration.

* **msprobe.mindspore.PrecisionDebugger.start**: Starts precision data collection.

* **msprobe.mindspore.PrecisionDebugger.stop**: Stops precision data collection.

* **msprobe.mindspore.PrecisionDebugger.step**: Ends data collection of a step, flushes all data, and updates dump parameters.

For details about the dump APIs, see [API Description](#api-description).

**Precautions**

None

### Usage Example

1. Create a `config.json` configuration file to configure dump parameters. The file content is as follows:

    ```json
    {
        "task": "statistics",
        "dump_path": "./output",
        "rank": [],
        "step": [],
        "level": "L0",
        "statistics": {
            "scope": [],
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics"
        }
    }
    ```

    For details about dump parameters, see [Configuration File Introduction](./config_json_introduct.md).

2. Compile the model training script `net.py` and add the dump APIs of msProbe to the script. The script content is as follows:

    ```python
    import mindspore as ms
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Import the data collection API of the tool.
    from msprobe.mindspore import PrecisionDebugger, seed_all

    # Fix randomness prior to initiating model training.
    seed_all()
    # Instantiate PrecisionDebugger before model training.
    debugger = PrecisionDebugger(config_path='./config.json')


    # Define a network.
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features=8, out_features=4)
            self.linear2 = nn.Linear(in_features=4, out_features=2)

        def forward(self, x):
            x1 = self.linear1(x)
            x2 = self.linear2(x1)
            logits = F.relu(x2)
            return logits


    net = Net()


    def train_step(inputs):
        return net(inputs)


    if __name__ == "__main__":
        data = (torch.randn(10, 8), torch.randn(10, 8), torch.randn(10, 8))
        grad_fn = ms.value_and_grad(train_step, grad_position=0)

        for inputs in data:
            # Enable data dump.
            debugger.start(model=net)

            out, grad = grad_fn(inputs)

            # Disable data dump.
            debugger.stop()
            # Update step information.
            debugger.step()
    ```

3. Execute the model training script to start model training. Run the following command:

    ```bash
    python net.py
    ```

    The tool collects precision data during model training.

### Output Description

After precision data is collected, the dump data file is displayed.

```ColdFusion
dump.json is at <dump_path>/step<N>
```

* `<dump_path>`: dump data output path specified by the **dump_path** parameter in the dump configuration file.

* `step<N>`: (*N*+1)th step. For example, **step0** indicates that all dump data of the first step is stored in this directory.

For details about the dump data, see [Dump Output Description](#dump-output-description).

## Dump Output Description

### Output Directory Structure

In the MSAdapter scenario, the dump output directory structure is as follows:

```lua
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── Tensor.permute.1.forward.npy
|   |   |   |    ├── Functional.linear.5.backward.output.npy    # Naming format: "{api_type}.{api_name}.{Number of API calls}.{forward/backward}.{input/output}.{Parameter No.}", where {Parameter No.} indicates the nth API input or output. For example, the value 1 indicates the first parameter. If the parameters are in list format, they are ordered based on list. For example, the value 1.1 indicates the first element of the first parameter.
|   |   |   |    ...
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.input.0.npy          # "{Module}.{module_name}.{class_name}.{forward/backward}.{Number of calls}.{input/output}.{Parameter No.}", where {Parameter No.} indicates the nth module parameter. For example, the value 1 indicates the first parameter. If the parameters are in list format, they are ordered based on list. For example, the value 1.1 indicates the first element of the first parameter.
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.parameters.bias.npy  # Module parameter, in the format of "{Module}.{module_name}.{class_name}.forward.{Number of calls}.parameters.{parameter_name}".
|   |   |   |    └── Module.conv1.Conv2d.parameters_grad.0.weight.npy     # Module parameter gradient, in the format of "{Module}.{module_name}.{class_name}.parameters_grad.{parameter_name}". Notice that the number of parameter gradient calculations does not mean the number of module calls.
|   |   |   |                                                          # When the model parameter passed during dump is List[torch.nn.Module] or Tuple[torch.nn.Module], the module-level data name contains the index of the module in the list. The naming format is {Module}.{index}.*, where * indicates the preceding three module-level data naming formats, for example, Module.0.conv1.Conv2d.forward.0.input.0.npy.
│   |   |   ├── dump.json
│   |   |   ├── stack.json
│   |   |   └── construct.json
│   |   ├── rank1
|   |   |   ├── dump_tensor_data
|   |   |   |   └── ...
│   |   |   ├── dump.json
│   |   |   ├── stack.json
|   |   |   └── construct.json
│   |   ├── ...
│   |   |
|   |   └── rank7
│   ├── step1
│   |   ├── ...
│   ├── step2
```

* `rank`: device ID. Data for each rank is stored in its respective `rank{ID}` directory. In non-distributed scenarios, there is no rank ID, and the directory is named `proc{pid}`, where *pid* indicates the process ID.

* `dump_tensor_data`: stores collected tensor data.

* `dump.json`: contains statistical information about the input and output data of APIs or modules. The information includes the API name or module name associated with the dump data, as well as statistical details for each data item, such as dtype, shape, max, min, mean, L2norm (L2 norm, square root), and CRC-32 data when **summary_mode** is set to **md5**.

* `dump_error_info.log`: dump errors. This log is generated only when the dump tool reports an error.

* `stack.json`: call stack information of APIs or modules.

* `construct.json`: hierarchical structure. When the **level** is set to **L1**, the **construct.json** file remains empty.

When **task** is set to **tensor** in the dump configuration file, during dump, the .npy file is flushed to drive after the corresponding operator or module is executed. The .json file writes complete data only after **PrecisionDebugger.stop()** is executed. Therefore, when the program is terminated abnormally, the .npy file corresponding to the executed operator has been saved, but the data in the .json file may be lost.

The table below describes prefixes of the .npy file name.

| Prefix       | Description                         |
| ----------- | ---------------------------- |
| Tensor      | torch.Tensor API data         |
| Torch       | torch API data                |
| Functional  | torch.nn.functional API data  |
| NPU         | NPU affinity API data               |
| Distributed | torch.distributed API data    |
| Module      | torch.nn.Module class (module) data|
| Jit         | Module or function data decorated by "jit"  |
| Primitive   | mindspore.ops.Primitive API data|

### dump.json Description

#### L0 Level

The **dump.json** file at the L0 level contains the inputs, outputs, parameters, and parameter gradients of a module. The following uses the Conv2d module as an example. The module invocation code is:
`output = self.conv2(input) # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`  

The **dump.json** file contains the following data: 

* `Module.conv2.Conv2d.forward.0`: module forward data. **input_args** is the input data (position parameter), **input_kwargs** is the input data (keyword parameter), **output** is the output data of the module, and **parameters** is module parameters, including weight and bias.

* `Module.conv2.Conv2d.parameters_grad.0`: parameter gradient, including the gradients of weight and bias.

* `Module.conv2.Conv2d.backward.0`: module backward data. **input** indicates the backward input gradient (corresponding to the forward output gradient), and **output** indicates the backward output gradient (corresponding to the forward input gradient).

**Note**: When the model parameter passed during dump is **List[torch.nn.Module]** or **Tuple[torch.nn.Module]**, the module-level data name contains the index of the module in the list. The naming format is `{Module}.{index}.*`, where ***** indicates the naming format of the preceding three types of module-level data, for example, `Module.0.conv1.Conv2d.forward.0`.    

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "mindtorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 1.638758659362793,
     "Min": 0.0,
     "Mean": 0.2544615864753723,
     "Norm": 70.50277709960938,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.npy"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 1.6815717220306396,
     "Min": -1.5120246410369873,
     "Mean": -0.025344856083393097,
     "Norm": 149.65576171875,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.npy"
    }
   ],
   "parameters": {
    "weight": {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.05992485210299492,
     "Min": -0.05999220535159111,
     "Mean": -0.0006165213999338448,
     "Norm": 3.421217441558838,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.npy"
    },
    "bias": {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.04894155263900757,
     "Mean": 0.006410328671336174,
     "Norm": 0.17263513803482056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.npy"
    }
   }
  },
  "Module.conv2.Conv2d.parameters_grad.0": {
   "weight": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.018550323322415352,
     "Min": -0.008627401664853096,
     "Mean": 0.0006675920449197292,
     "Norm": 0.26084786653518677,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.0.weight.npy"
    }
   ],
   "bias": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32
     ],
     "Max": 0.014914230443537235,
     "Min": -0.006656786892563105,
     "Mean": 0.002657240955159068,
     "Norm": 0.029451673850417137,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.0.bias.npy"
    }
   ]
  },
  "Module.conv2.Conv2d.backward.0": {
   "input": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 0.0015069986693561077,
     "Min": -0.001139344065450132,
     "Mean": 3.3215508210560074e-06,
     "Norm": 0.020567523315548897,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.backward.0.input.0.npy"
    }
   ],
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 0.0007466732058674097,
     "Min": -0.00044813455315306783,
     "Mean": 6.814070275140693e-06,
     "Norm": 0.01474067009985447,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.backward.0.output.0.npy"
    }
   ]
  }
 }
}
```

#### L1 Level

The **dump.json** file at the L1 level contains the input and output data of an API. The following uses the **relu** API as an example. The API invocation code is:
`output = torch.nn.functional.relu(input)`  

The **dump.json** file contains the following data: 

* `Functional.relu.0.forward`: API forward data. **input_args** is the input data (position parameter), **input_kwargs** is the input data (keyword parameter), and **output** is the output data of the API.

* `Functional.relu.0.backward`: API backward data. **input** is the backward input gradient (corresponding to the forward output gradient), and **output** is the backward output gradient (corresponding to the forward input gradient).

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "mindtorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": -1.3364859819412231,
     "Mean": 0.03711778670549393,
     "Norm": 236.20692443847656,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.input.0.npy"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": 0.0,
     "Mean": 0.16849493980407715,
     "Norm": 175.23345947265625,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.output.0.npy"
    }
   ]
  },
  "Functional.relu.0.backward": {
   "input": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 0.0001815402356442064,
     "Min": -0.00013352684618439525,
     "Mean": 0.00011915402356442064,
     "Norm": 0.007598237134516239,
     "requires_grad": false,
     "data_name": "Functional.relu.0.backward.input.0.npy"
    }
   ],
   "output": [
    {
     "type": "mindspore.Tensor",
     "dtype": "Float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 0.0001815402356442064,
     "Min": -0.00012117840378778055,
     "Mean": 2.0098118724831693e-08,
     "Norm": 0.006532244384288788,
     "requires_grad": false,
     "data_name": "Functional.relu.0.backward.output.0.npy"
    }
   ]
  }
 }
}  
```

#### mix Level

The **dump.json** file at the mix level contains both L0 and L1 dump data. The file format is the same as that in the preceding examples. 

## Appendixes

### API Description

#### msprobe.mindspore.seed_all

**Function**

Fixes randomness in the network and enables deterministic computing.

**Prototype**

```python
msprobe.mindspore.seed_all(seed=1234, mode=False, rm_dropout=False)
```

**Parameters**

* **seed** (int): (optional) random seed; defaults to **1234**; example: **seed=1000**. This parameter is used to generate random numbers for **random**, **numpy.random**, **mindspore.common.Initializer**, and **mindspore.nn.probability.distribution**. as well as the hash algorithm for the **str**, **bytes**, and **datetime** objects in Python.

* **mode** (bool): (optional) enables deterministic computing. The value can be **True** or **False** (default), for example, **mode=True**. If this parameter is set to **True**, the deterministic running mode of operators and deterministic computing of reduction communication operators (AllReduce, ReduceScatter, and Reduce) are enabled.

  Note: Deterministic computing deteriorates API execution performance. You are advised to enable it when multiple execution results of your model are found to be different.

* **rm_dropout** (bool): (optional) dropout invalidation switch. The value can be **True** or **False** (default), for example, **rm_dropout=True**. If this parameter is set to **True**, **mindspore.ops.Dropout**, **mindspore.ops.Dropout2D**, **mindspore.ops.Dropout3D**, **mindspore.mint.nn.Dropout**, and **mindspore.mint.nn.functional.dropout** will become invalid to avoid network randomness caused by random dropout. You are advised to enable this function before collecting MindSpore data.

  Note: **rm_dropout** can be called to control dropout invalidation or validation only before the dropout instance is initialized.

**Returns**

None

**Example**

For details, see "Examples" in [MSAdapter Dump Functions](#msadapter-dump-functions).

#### msprobe.mindspore.PrecisionDebugger

**Function**

Loads the dump configuration file to determine the detailed dump configuration.

**Prototype**

```Python
class msprobe.mindspore.PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**Parameters**

* **config_path** (str): (optional) path of the dump configuration file, for example, **./config.json**. If this path is not configured, the default configuration in [config.json](../../../python/msprobe/config.json) is used. For details about configuration items, see [Configuration File Introduction](./config_json_introduct.md).

* Other parameters are optional and can be configured in [config.json](../../../python/msprobe/config.json). For details, see [Configuration File Introduction](./config_json_introduct.md). If the parameter value is not **None**, its priority is higher than that of the configuration with the same name in [config.json](../../../python/msprobe/config.json).

**Returns**

None

**Example**

For details, see "Examples" in [MSAdapter Dump Functions](#msadapter-dump-functions).

#### msprobe.mindspore.PrecisionDebugger.start

**Function**

Starts precision data collection. This API must be added to a training "for" loop together with **stop**.

**Prototype**

```Python
PrecisionDebugger.start(model=None, token_range=None)
```

**Parameters**

* **model**: (optional) instantiated model whose data needs to be collected. The **torch.nn.Module**, **list[torch.nn.Module]**, or **Tuple[torch.nn.Module]** type can be passed. By default, this parameter is not configured. For L0 and mix level dump, **model** must be passed to collect all module data. If there is a module object (for example, a module decorated by `mindspore.jit`) that will be graph-compiled, `start` must be called before the first training step starts. For L1 level dump, if **model** is passed, all API data (including primitive op objects) can be collected. If **model** is not passed, only API data of non-primitive ops is collected. If **token_range** is not **None**, **model** must be passed.

  For a complex model, if only part of the model (for example, **model.A**, **model.A extends torch.nn.Module**) needs to be monitored, you only need to pass the part to be monitored.

  Note: The input layer is not dumped. The tool dumps only the sublayers of the input layer. For example, if **model.A** is passed, **A** is not dumped, but **A.***x* and **A.***x.xx* are dumped.

- **token_range** (Tuple[int, int]): (optional) token range during inference model data collection. The type is **[int, int]**, indicating **[start, end]** of a range. By default, this parameter is not configured.

**Returns**

None

**Example**

For details, see "Examples" in [MSAdapter Dump Functions](#msadapter-dump-functions).

#### msprobe.mindspore.PrecisionDebugger.stop

**Function**

Stops precision data collection. This API can be added anywhere following **start**. If **stop** is added following the backward propagation code, the forward and backward data between **start** and **start** is collected.
If **stop** is added before the backward propagation code, **step** must be added following the backward propagation code to collect the forward and backward data between **start** and **stop**.

**stop** must be called; otherwise, the flushed precision data may be incomplete.

**Prototype**

```Python
PrecisionDebugger.stop()
```

**Parameters**

None

**Returns**

None

**Example**

For details, see "Examples" in [MSAdapter Dump Functions](#msadapter-dump-functions).

#### msprobe.mindspore.PrecisionDebugger.step

**Function**

Increases the number of training steps, flushes all data of the current step, and updates dump parameters. This API is added to a location where a step ends, and it must be called after **stop**. It must be used together with **start** and **stop**. It is recommended that it be added after the backward propagation code. Otherwise, backward data may be lost.

**Prototype**

```Python
PrecisionDebugger.step()
```

**Parameters**

None

**Returns**

None

**Example**

For details, see "Examples" in [MSAdapter Dump Functions](#msadapter-dump-functions).

### API List

During API-level dump, the tool provides a fixed API list. Only APIs in the list can be used to collect precision data. Generally, you do not need to modify the list. Instead, you can specify the dump API by using the **scope**/**list** field in the **config.json** file. To change the API list, manually modify [support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml). The following is an example:

```yaml
functional:  # functional indicates the operator type. Find the corresponding type and delete or add APIs in the following format:
  - conv1d
  - conv2d
  - conv3d
```
