# Precision Data Collection in PyTorch

## Overview

msProbe collects the precision data of a model during running by adding the `PrecisionDebugger` API to the training script and starting training.

**Features**

- **Multi-granularity data collection**: Support data collection at different granularities, including L0 (module-level), L1 (API-level), and mix (L0+L1).
- **Multiple dump modes**: Provide multiple collection modes, including statistics, tensor, acc_check, structure, and overflow_check.
- **Flexible configurations**: The collection scope can be precisely controlled using the **config.json** file.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

- Only the PyTorch framework is supported. The dynamo scenario of PyTorch 2.7 and later versions is not supported.
- Due to the limitation of the automatic differentiation mechanism of the PyTorch framework, the dumped data may be missing reverse data for in-place operation module/API and its previous module/API.
- Loss or gnorm may change after msProbe is used. This can be caused by synchronization introduced by item operations in the tool or by the hook mechanisms of the PyTorch or MindSpore framework. For details, see [Cause Analysis of Model Calculation Result Changes](../faq.md#model-computation-result-change).

## Quick Start

The following uses a simple example to describe how to use msProbe to collect precision data in PyTorch. This example defines a simple nn.Module network and uses the **PrecisionDebugger** prototype to collect data. Pay attention to the highlighted code.

```diff
 import torch
 import torch.nn as nn
 import torch.nn.functional as F

+# Import the data collection API of the tool.
+from msprobe.pytorch import PrecisionDebugger, seed_all
+
+# Fix randomness prior to initiating model training.
+seed_all()
+# Instantiate PrecisionDebugger before model training.
+debugger = PrecisionDebugger(config_path="./config.json")

 # Define a network.
 class ModuleOP(nn.Module):
     def __init__(self) -> None:
         super().__init__()
         self.linear = nn.Linear(in_features=8, out_features=4)

     def forward(self, x):
         x1 = self.linear(x)
         r1 = F.relu(x1)
         return r1

 if __name__ == "__main__":
     model = ModuleOP()

+    debugger.start(model=model)  # Enable data dump.

     x = torch.randn(10, 8)
     out = model(x)
     loss = out.sum()
     loss.backward()

+    debugger.stop()  # Disable data dump. You can enable data dump again. The collected data is recorded in the same step.
+    debugger.step()  # End data dump. If data dump is enabled again, the collected data is recorded in the next step.
```

The **config.json** file is the configuration file for precision data collection and needs to be created by yourself. For details, see [Configuration File Introduction](./config_json_introduct.md). The following is a simple example of the **config.json** file:

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

## Data Collection

### Function Description

Select a proper task type according to your requirements. For details about task configuration items, see [Configuration File Introduction](./config_json_introduct.md#parameters).

| Debugging Requirement  | Recommended Configuration       | Description             |
| ------ | --------------- | --------------- |
| Preliminary precision analysis| task="statistics"      | Low resource usage and fast statistics collection. |
| In-depth precision analysis| task="tensor"          | Complete data collection and detailed analysis.  |
| Deterministic issue analysis| task="statistics"<br>summary_mode="md5"          | Statistics and tensor CRC-32 values collected for quick analysis of deterministic issues.  |

### Usage Example

#### Basic Data Dump

The following is a basic example of data dump. The highlighted code in the example needs to be added to your script.

<details>
<summary><b>Click to see the code.</b></summary>

```diff
+# Import the tool's data collection API. Try to execute **seed_all** and instantiate **PrecisionDebugger** at a location following the import statements in the training script.
+from msprobe.pytorch import seed_all, PrecisionDebugger
+# Fix randomness prior to initiating model training.
+seed_all()
+# Instantiate PrecisionDebugger and load the dump configuration file.
+debugger = PrecisionDebugger(config_path="./config.json")

 # Define and initialize the model and loss function.
 # ...

 # Dataset iteration typically marks the start of model training.
 for data, label in data_loader:
+    debugger.start(model)  # Enable data dump.

     # Logical flow executed at each step of the model
     output = model(data)
     # ...
     # Calculate the gradient.
     loss.backward()
     # Update the parameter.
     optimizer.step()

+    debugger.stop()  # Disable data dump. You can enable data dump again. The collected data is recorded in the same step.
+    debugger.step()  # End data dump. If data dump is enabled again, the collected data is recorded in the next step.
```

</details><br>

API description:

- [seed_all](#seed_all): fixes randomness in the network and enables deterministic computing.
- [[PrecisionDebugger](#precisiondebugger): loads the dump configuration file to determine the detailed dump configuration.
- [start](#start): starts precision data collection.
- [stop](#stop): stops precision data collection.
- [step](#step): ends data collection of a step, flushes all data, and updates dump parameters.

For details about more APIs, see [API Description](#api-description).

**Precautions**

- `PrecisionDebugger` provides the dynamic dump start and stop capabilities, which are controlled by the `dump_enable` field in the configuration file. Dump can be dynamically enabled or disabled as required in the same training or inference job. For details, see the description of `dump_enable` in [Configuration File Introduction](./config_json_introduct.md).

- The tool provides a fixed list of supported APIs. To delete or add dump APIs, manually modify [support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml). The following is an example:

  ```yaml
  functional:  # functional indicates the operator type. Find the corresponding type and delete or add APIs in the following format:
    - conv1d
    - conv2d
    - conv3d
  ```

  Scenario where an API is deleted: The code logic of some models may involve native API type verification. When the tool performs the dump operation, the API type encapsulated by the model may be different from the native API type. In this case, the verification may fail. For details, see [Exceptions](../faq.md#exception-handling) in *FAQs*.

- To collect the precision data of a custom API, register the API with the tool first. For details about the registration method, see [register_custom_api](#register_custom_api).

#### Module Data Dump

The following is an example of collecting module data. The highlighted code in the example needs to be added to your script.

<details>
<summary><b>Click to see the code.</b></summary>

```diff
 # Import packages as required.
 import torch
 import torch.nn as nn
 import torch.nn.functional as F

+# Import the data collection API of the tool.
+from msprobe.pytorch import PrecisionDebugger, module_dump, module_dump_end

+# Instantiate PrecisionDebugger before model training.
+debugger = PrecisionDebugger(config_path='./config.json')

 # Define a network.
 class ModuleOP(nn.Module):
     def __init__(self) -> None:
         super().__init__()
         self.linear_1 = nn.Linear(in_features=8, out_features=4)
         self.linear_2 = nn.Linear(in_features=4, out_features=2)

     def forward(self, x):
         x1 = self.linear_1(x)
         x2 = self.linear_2(x1)
         r1 = F.relu(x2)
         return r1

 if __name__ == "__main__":
     module = ModuleOP()

+    debugger.start()  # Enable data dump.

     x = torch.randn(10, 8)
     # ...                              # Data between start and module_dump is dumped normally.
+    module_dump(module, "MyModuleOP")  # Enable module-level precision data dump.
     out = module(x) # Child modules or APIs in the module will not be dumped.
+    module_dump_end() # Disable module-level precision data dump.
     loss = out.sum() # Data between module_dump_end and stop is dumped normally.
     loss.backward()

+    debugger.stop()  # Disable data dump. You can enable data dump again. The collected data is recorded in the same step.
+    debugger.step()  # End data dump. If data dump is enabled again, the collected data is recorded in the next step.
```

</details><br>

#### Cross-File Data Dump

The following is an example of cross-file data dump. The highlighted code in the example needs to be added to your script.

<details>
<summary><b>Click to see the code.</b></summary>

To ensure that all APIs are encapsulated by the tool, **PrecisionDebugger** is usually instantiated at the entry of a training project. However, model definition may locate in another file.

Assume that there are two files: `train.py` (training project entry) and `module.py` (model definition file). To collect the forward and backward data of some submodules or APIs in the ModuleOP module defined in `module.py`, you need to import **PrecisionDebugger** to `train.py` and `module.py` and set the following configurations:

train.py:

```diff
 # Import packages as required.
 import torch
 from module import ModuleOP

+# Import the data collection API of the tool.
+from msprobe.pytorch import PrecisionDebugger

+# Instantiate PrecisionDebugger at the beginning of the file, immediately following package imports, to guarantee that all APIs are properly encapsulated.
+debugger = PrecisionDebugger(config_path='./config.json')

 if __name__ == "__main__":
     module = ModuleOP()

     x = torch.randn(10, 8)
     out = module(x)
     loss = out.sum()
     loss.backward()
```

module.py:

```diff
 import torch
 import torch.nn as nn
 import torch.nn.functional as F

+from msprobe.pytorch import PrecisionDebugger

 # Define a network.
 class ModuleOP(nn.Module):
     def __init__(self) -> None:
         super().__init__()
         self.linear_1 = nn.Linear(in_features=8, out_features=4)
         self.linear_2 = nn.Linear(in_features=4, out_features=2)

     def forward(self, x):
+        PrecisionDebugger.start()
         x1 = self.linear_1(x)
+        PrecisionDebugger.stop()
         x2 = self.linear_2(x1)
         r1 = F.relu(x2)
         return r1

```

</details><br>

#### token_range Specified During Model Inference Data Collection

The following is an example of collecting data for a specified token range during model inference. The highlighted code in the example needs to be added to your script.

<details>
<summary><b>Click to see the code.</b></summary>

```diff
 from vllm import LLM, SamplingParams
+from msprobe.pytorch import PrecisionDebugger, seed_all

+# Fix randomness prior to initiating model training.
+seed_all()
+# Do not insert the PrecisionDebugger initialization process into the looped code.
+debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")
 # Define and initialize the model.
 prompts = ["Hello, my name is"]
 sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
 llm = LLM(model='...')
 model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
+# Enable data dumping and collect data during the first to third cycles of character-by-character cyclic inference for the inference model.
+debugger.start(model=model, token_range=[1, 3])
 # Logic for generating the inference model
 output = llm.generate(prompts, sampling_params=sampling_params)
+# Disable data dump and flush data.
+debugger.stop()
+debugger.step()
```

</details><br>

### Output Description

After precision data is collected, **dump_path** that stores dump data is displayed, as shown in the following:

```ColdFusion
dump.json is at ./dump_path/step*
```

***** indicates the step number. When the printed step number is the last step, dumping ends. Dump data files are stored in each step directory.

## Dump Result File

### Description of the dump Directory

After training is complete, the tool saves the dump data in the directory specified by **dump_path**. 
The directory structure is as follows:
<details>
<summary><b>Click to see the directory structure.</b></summary>

```txt
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── Tensor.permute.1.forward.pt
|   |   |   |    ├── Functional.linear.5.backward.output.pt            # Naming format: "{api_type}.{api_name}.{Number of API calls}.{forward/backward}.{input/output}.{Parameter No.}", where {Parameter No.} indicates the nth API input or output. For example, the value 1 indicates the first parameter. If the parameters are in list format, they are ordered based on list. For example, the value 1.1 indicates the first element of the first parameter.
|   |   |   |    ...
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # "{Module}.{module_name}.{class_name}.{forward/backward}.{Number of calls}.{input/output}.{Parameter No.}", where {Parameter No.} indicates the nth module parameter. For example, the value 1 indicates the first parameter. If the parameters are in list format, they are ordered based on list. For example, the value 1.1 indicates the first element of the first parameter.
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # Module parameter, in the format of "{Module}.{module_name}.{class_name}.forward.{Number of calls}.parameters.{parameter_name}".
|   |   |   |    └── Module.conv1.Conv2d.parameters_grad.0.weight.pt     # Module parameter gradient, in the format of "{Module}.{module_name}.{class_name}.parameters_grad.{Number of times of grad_fn triggering}.{parameter_name}". Notice that the number of times of grad_fn triggering is calculated, instead of the number of module calls.
|   |   |   |                                                          # When the model parameter passed during dump is List[torch.nn.Module] or Tuple[torch.nn.Module], the module-level data name contains the index of the module in the list. The naming format is {Module}.{index}.*, where * indicates the preceding three module-level data naming formats, for example, Module.0.conv1.Conv2d.forward.0.input.0.pt.
│   |   |   ├── dump.json
│   |   |   ├── stack.json
│   |   |   ├── dump_error_info.log
│   |   |   └── construct.json
│   |   ├── rank1
|   |   |   ├── dump_tensor_data
|   |   |   |   └── ...
│   |   |   ├── dump.json
│   |   |   ├── stack.json
│   |   |   ├── dump_error_info.log
|   |   |   └── construct.json
│   |   ├── ...
│   |   |
|   |   └── rank7
│   ├── step1
│   |   ├── ...
│   ├── step2
```

</details><br>

* `rank`: device ID. Data for each rank is stored in its respective `rank{ID}` directory. If the training process cannot obtain the `rank` information, the data of the current process is stored in `proc{pid}`. `pid` indicates the process ID. `proc` is described as follows:
  * In non-distributed scenarios, such as single-process training or single-rank training, the training process does not have `rank` information. In this case, data is stored in `proc{pid}`. The comparison, hierarchical visualization, and overflow/underflow detection functions support data parsing in this directory.
  * During foundation model training, both the `rank` and `proc` directories may exist. The reason is that some processes may only complete some data preprocessing operations on CPU and do not have `rank` information. In this case, the directory name is `proc{pid}`. Generally, the data does not have precision problems. The comparison, hierarchical visualization, and overflow/underflow detection functions do not support data parsing in this directory.
* `dump_tensor_data`: stores collected tensor data.
* `dump.json`: contains statistical information about the forward and backward data of APIs or modules. This includes the names of APIs or modules containing dump data, along with statistics such as dtype,
  shape, max, min, mean, and L2norm (L2 norm, square root) statistics, as well as the verification value output based on **summary_mode**
  (`md5` corresponds to `md5` of the CRC-32 field, and `xor` corresponds to `md5` of the XOR verification field). For details, see [dump.json Description](#dumpjson-description).
* `dump_error_info.log`: dump errors. This log is generated only when the dump tool reports an error.
* `stack.json`: call stack information of APIs or modules.
* `construct.json`: hierarchical structure. When the **level** is set to **L1**, the **construct.json** file remains empty.

During dump, the .pt file is flushed to drive after the corresponding operator or module is executed. The .json file writes complete data only after **PrecisionDebugger.stop()** is executed.
Therefore, when the program is terminated abnormally, the .pt file corresponding to the executed operator has been saved, but the data in the .json file may be lost.

The table below describes the mapping between the prefix of the .pt file and PyTorch.

| Prefix         | Torch Module            |
|-------------|---------------------|
| Tensor      | torch.Tensor        |
| Torch       | torch               |
| Functional  | torch.nn.functional |
| NPU         | NPU affinity operator            |
| VF          | torch._VF           |
| Aten        | torch.ops.aten      |
| Distributed | torch.distributed   |
| MindSpeed   | mindspeed.ops       |
| Triton   | triton       |

### dump.json Description

#### L0 Level

The **dump.json** file at the L0 level contains the forward and backward inputs and outputs of a module, as well as parameters and parameter gradients. Take Conv2d module of PyTorch as an example. The module invocation code is: 
`output = self.conv2(input) # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`

The **dump.json** file contains the following data:

- `Module.conv2.Conv2d.forward.0`:
  module forward data. **input_args** is the input data (position parameter), **input_kwargs** is the input data (keyword parameter), **output** is the output data of the module, and **parameters** is module parameters, including weight and bias.
- `Module.conv2.Conv2d.parameters_grad.0`: parameter gradient, including the gradients of weight and bias.
- `Module.conv2.Conv2d.backward.0`: module backward data. **input** indicates the backward input gradient (corresponding to the forward output gradient), and **output** indicates the backward output gradient (corresponding to the forward input gradient).

**Note**: When **model** passed during dump is **List[torch.nn.Module]** or **Tuple[torch.nn.Module]**,
the module-level data name contains the index of the module in the list. The naming format is `{Module}.{index}.*`, where *****
indicates the naming format of the preceding three types of module-level data, for example, `Module.0.conv1.Conv2d.forward.0`.

The **dump.json** file is as follows:
<details>
<summary><b>Click to see the **dump.json** file.</b></summary>

```json
{
  "task": "tensor",
  "level": "L0",
  "framework": "pytorch",
  "dump_data_dir": "/dump/path",
  "data": {
    "Module.conv2.Conv2d.forward.0": {
      "input_args": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
        }
      ],
      "input_kwargs": {},
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
        }
      ],
      "parameters": {
        "weight": {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
        },
        "bias": {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32
          ],
          "Max": 0.05744686722755432,
          "Min": -0.04894155263900757,
          "Mean": 0.006410328671336174,
          "Norm": 0.17263513803482056,
          "requires_grad": true,
          "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
        }
      }
    },
    "Module.conv2.Conv2d.parameters_grad.0": {
      "weight": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Module.conv2.Conv2d.parameters_grad.0.weight.pt"
        }
      ],
      "bias": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [
            32
          ],
          "Max": 0.014914230443537235,
          "Min": -0.006656786892563105,
          "Mean": 0.002657240955159068,
          "Norm": 0.029451673850417137,
          "requires_grad": false,
          "data_name": "Module.conv2.Conv2d.parameters_grad.0.bias.pt"
        }
      ]
    },
    "Module.conv2.Conv2d.backward.0": {
      "input": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Module.conv2.Conv2d.backward.0.input.0.pt"
        }
      ],
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Module.conv2.Conv2d.backward.0.output.0.pt"
        }
      ]
    }
  }
}
```

</details><br>

#### L1 Level

The **dump.json** file at the L1 level contains the forward and backward inputs and outputs of an API. Take the **relu** API of PyTorch as an example. The API invocation code is:
`output = torch.nn.functional.relu(input)`

The **dump.json** file contains the following data:

- `Functional.relu.0.forward`: API forward data. **input_args** is the input data (position parameter), **input_kwargs** is the input data (keyword parameter), and **output** is the output data of the API.
- `Functional.relu.0.backward`: API backward data. **input** is the backward input gradient (corresponding to the forward output gradient), and **output** is the backward output gradient (corresponding to the forward input gradient).

The **dump.json** file is as follows:
<details>
<summary><b>Click to see the **dump.json** file.</b></summary>

```json
{
  "task": "tensor",
  "level": "L1",
  "framework": "pytorch",
  "dump_data_dir": "/dump/path",
  "data": {
    "Functional.relu.0.forward": {
      "input_args": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Functional.relu.0.forward.input.0.pt"
        }
      ],
      "input_kwargs": {},
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Functional.relu.0.forward.output.0.pt"
        }
      ]
    },
    "Functional.relu.0.backward": {
      "input": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Functional.relu.0.backward.input.0.pt"
        }
      ],
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
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
          "data_name": "Functional.relu.0.backward.output.0.pt"
        }
      ]
    }
  }
}  
```

</details><br>

#### mix Level

The **dump.json** file at the mix level contains both L0 and L1 dump data. The file format is the same as that in the preceding examples.

## Extended Functions

### Collecting Randomness APIs

**Function**

The [random_save](#random_save) API is used to find the call positions of all randomness APIs in the model script and output the result to a .csv file, helping you quickly locate the random number generation points in the model.

**Example**

```diff
 import torch
 import torch.nn as nn
 import torch.nn.functional as F

+# Import the tool to search for randomness APIs. These APIs in the network are saved in the output path.
+from msprobe.pytorch import random_save
+random_save(output_path="./output")

 # Define a network.
 class ModuleOP(nn.Module):
     def __init__(self) -> None:
         super().__init__()
         self.linear = nn.Linear(in_features=8, out_features=4)

     def forward(self, x):
         x1 = self.linear(x)
         r1 = F.relu(x1)
         return r1

 if __name__ == "__main__":
     model = ModuleOP()

     x = torch.randn(10, 8)
     out = model(x)
     loss = out.sum()
     loss.backward()
```

**Output Description**

After the model script is executed, a .csv file is generated in the specified **output_path**. If the executed process can obtain the rank information, the result file name is `random_rank{id}_{timestamp}.csv`; otherwise, the result file name is `random_proc{pid}_{timestamp}.csv`.

The figure below shows the content of the .csv file.

![random_save](../figures/random_save.png)

| Parameter    | Description                       |
| -------- | --------------------------- |
| api_name | Name of a randomness API in the model script.|
| stack    | Stack information corresponding to a randomness API.  |

## API Description

### PrecisionDebugger

**Function**

Loads the dump configuration file to determine the detailed dump configuration. The initialization of this API must be in the same process as the collection target. Otherwise, the target data cannot be collected.

**Prototype**

```Python
debugger = PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**Parameters**

- **config_path**: (optional) path of the dump configuration file, in the string type, for example, **./config.json**. If this path is not configured, the default configuration in [config.json](../../../python/msprobe/config.json) is used.
  For details about the default configuration of the file, see [Configuration File Introduction](./config_json_introduct.md).
- Other parameters can be configured in the **config.json** file. For details, see [Configuration File Introduction](./config_json_introduct.md).
- The parameters of this API are optional. If none of the parameters is configured, the L1 statistics of all ranks and steps are collected by default. The priority of this API is higher than the configuration in **config.json**.
  However, the number of configurable parameters of this API is less than that in **config.json**.

**Returns**

**debugger** is an instance object of the **PrecisionDebugger** class. This instance object provides functions such as **start**, **stop**, and **step** to start and stop data collection.

**Example**

See [Basic Data Dump](#basic-data-dump).

### start

**Function**

Starts precision data collection. This API is added to the position following model initialization. It must be used together with **stop** and **step**.

**Prototype**

```Python
debugger.start(model=None, token_range=None, rank_id=None)
```

**Parameters**

- **model**: model whose module-level data needs to be collected. The input can be of the **torch.nn.Module**, **list[torch.nn.Module]**, or **Tuple[torch.nn.Module]** type. By default, this parameter is not configured.
  This parameter must be set in this API when **level** is set to **L0** or **mix**, or **token_range** is not **None**. For a complex model, if only part of the model needs to be monitored
  (for example, **model.A** and **model.A extends torch.nn.Module**),
  pass the part to be monitored. Note: The input layer is not dumped. The tool dumps only the sub-layers of the input layer. For example, if **model.A** is passed, **A** is not dumped, but
  **A**.*x* or **A**.*x.xx* is dumped.
- **token_range**: token range during inference model data collection. The type is **[int, int]**, indicating **[start, end]** of a range. By default, this parameter is not configured.
- **rank_id**: user-defined rank ID. The value is an integer greater than or equal to 0. By default, this parameter is not configured. The tool obtains the rank ID via the **torch.distributed.get_rank** API.
  After this parameter is configured, *{ID}* in the name of the rank folder of the dump results is the value of this parameter.

  Note: By default, the tool automatically obtains the unique rank ID using the **torch.distributed.get_rank** API (**get_rank**) in the multi-rank multi-process scenario.
  However, in some special scenarios, **get_rank** may fail to obtain the unique rank ID. For example, in the SGLang DP inference scenario, DP workers are independent, distributed clusters. As a result, **get_rank** returns duplicate rank IDs, causing the rank folders of the dump results to be overwritten and dump data to be lost.
  
  To solve this problem, you can set **rank_id** to name the rank folder, but ensure that rank ID is unique in each process. The value can be obtained from the model script or training/inference framework. For example, **self.gpu_id** in the SGLang inference framework is unique in each process.
  
  Example: `debugger.start(rank_id=self.gpu_id)`

**Returns**

None

**Example**

See [Basic Data Dump](#basic-data-dump).

### stop

**Function**

Stops precision data collection. This API is added to any position following **start**.

If **stop** is added following backward propagation code (e.g., loss.backward), the forward and backward data between **start** and **stop** is collected.

If **stop** is added before backward propagation code, **step** must be added following backward propagation code to collect the forward and backward data
between **start** and **stop**.

**Precautions**

**stop** must be called; otherwise, the flushed precision data may be incomplete.

**Prototype**

```Python
debugger.stop()
```

**Returns**

None

**Example**

See [Basic Data Dump](#basic-data-dump).

### step

**Function**

Ends data collection of a step, flushes all data, and updates dump parameters. This API is added to a location where a step ends, and it must be called after **stop**.

**Precautions**

It must be used together with **start** and **stop**. It is recommended that it be added after the backward propagation code (e.g., loss.backward). Otherwise, backward data may be lost.

**Prototype**

```Python
debugger.step()
```

**Returns**

None

**Example**

See [Basic Data Dump](#basic-data-dump).

### module_dump

**Function**
Enables module-level precision data dump. This API is module‑level, meaning it dumps only the input data of the module itself, not the data of its submodules or internal APIs.

**Precautions**

This function must be used in conjunction with **start**, **stop**, and **step**.

**Prototype**

```Python
module_dump(module, module_name)
```

**Parameters**

- **module**: (required) an instantiated **nn.Module** class object. The value is of the **torch.nn.Module** type.
- **module_name**: (required) user-defined module name, which is used to name the dump data. The value is a string.

**Returns**

None

**Example**

For details, see [Module Data Dump](#module-data-dump).

### module_dump_end

**Function**

Ends module-level precision data dump. For APIs or modules between **module_dump** and **module_dump_end**, only the input module data is dumped.
After **module_dump_end** is executed, the dump mode is restored to the normal mode.

**Prototype**

```Python
module_dump_end()
```

**Returns**

None

**Example**

For details, see [Module Data Dump](#module-data-dump).

### save

**Function**

Saves the forward and backward values at a single point during network execution and flushes the statistics or tensor files.

**Prototype**

```python
save(variable, name, save_backward=True)
```

**Parameters**

| Parameter         | Description    | Support Data Type                                          | Required (Yes/No)|
|---------------|----------|--------------------------------------------------|------|
| variable      | Variable to be saved. | dict, list, tuple, torch.tensor, int, float, str | Yes   |
| name          | Specified name.   | str                                              | Yes   |
| save_backward | Whether to save backward data.| boolean                                          | No   |

**Returns**

None

**Example**

For details, see [Single-Point Saving Tool](./debugger_save_instruct.md).

### set_init_step

**Function**

Sets the start step. By default, steps begin at 0. After this API is called, the step starts from the specified value. It must be placed before a training loop and cannot be placed inside the loop.

**Prototype**

```Python
debugger.set_init_step(step)
```

**Parameters**

**step**: start step.

**Returns**

None

**Example**

None

### register_custom_api

**Function**

Registers a user-defined API with the tool for API-level data dump.

**Prototype**

```Python
debugger.register_custom_api(module, api, api_prefix)
```

**Parameters**

**torch.matmul** is used as an example.

- **module**: (required) package to which the API belongs, that is, **torch**.
- **api**: (required) API name, which is of the str type, that is, **matmul**.
- **api_prefix**: (optional) prefix of the API name in [dump.json](#dumpjson-description). The default value is the string format of the package name, that is, **torch**.

**Returns**

None

**Example**

```diff
+# Import the custom operator module.
+import module_a
+from msprobe.pytorch import seed_all, PrecisionDebugger
+seed_all()
+debugger = PrecisionDebugger(config_path="./config.json")
+# Assume that the custom operator is called using module_a.compute(args). The registration method is as follows:
+debugger.register_custom_api(module = module_a, api = "compute")

 # Define and initialize the model and loss function.
 # ...

 # Dataset iteration typically marks the start of model training.
 for data, label in data_loader:
+    debugger.start(model)  # Enable data dump.

     # Logical flow executed at each step of the model
     output = model(data)
     # ...
     # Calculate the gradient.
     loss.backward()
     # Update the parameter.
     optimizer.step()

+    debugger.stop()  # Disable data dump. You can enable data dump again. The collected data is recorded in the same step.
+    debugger.step()  # End data dump. If data dump is enabled again, the collected data is recorded in the next step.

+# Cancel the dump of the API. Use it as required.
+debugger.restore_custom_api(module_a, "compute")
```

### restore_custom_api

**Function**

Restores the original user-defined API and cancels the dump of the API.

**Prototype**

```Python
debugger.restore_custom_api(module, api)
```

**Parameters**

**torch.matmul** is used as an example.

- **module**: (required) package to which the API belongs, that is, **torch**.
- **api**: (required) API name, which is of the str type, that is, **matmul**.

**Returns**

None

**Example**

For details, see [register_custom_api](#register_custom_api).

### seed_all

**Function**

Controls randomness in the model network. It supports fixed random seeds and enables deterministic computing to ensure experiment reproducibility.

**Prototype**

```python
seed_all(seed=1234, mode=False, rm_dropout=False, is_enhanced=False)
```

**Parameters**

- **seed**: (optional) random seed, for example, **seed=1000**. The default value is 1**234**.
- **mode**: (optional) deterministic computing mode. The value can be **True** or **False**, for example, **mode=True**. The default value is **False**. Note: Deterministic computing deteriorates API execution performance. You are advised to enable it when multiple execution results of your model are found to be different.
- **rm_dropout**: (optional) dropout invalidation switch. The value can be **True** or **False** (default), for example, **rm_dropout=True**.
  If this parameter is set to **True**, the tool automatically sets the **p** parameter of `torch.nn.functional.dropout`, `torch.nn.functional.dropout2d`, `torch.nn.functional.dropout3d`, `torch.nn.Dropout`, `torch.nn.Dropout2d`, and `torch.nn.Dropout3d` to **0**.
  This avoid network randomness caused by random dropout. Note: **rm_dropout** can be called to control dropout invalidation only before the dropout instance is initialized.
- **is_enhanced**: (optional) switch for enhanced randomness fixation. The value can be **True** or **False** (default), for example, **is_enhanced=True**. After it is enabled, the status of the built-in random number generators of PyTorch, NumPy, and Python is further fixed. When the same randomness API is executed multiple times in the same process or different processes, the random values generated each time can be the same. This helps achieve strict reproducibility in more complex random scenarios.

**Returns**

None

**Example**

```diff
 import torch
+from msprobe.pytorch import seed_all

 # seed_all only fixes random seeds and enables deterministic computing.
+seed_all(seed=1234, mode=True)
 num1 = torch.mean(torch.randn(2,2))
 print(num1)  # tensor(-0.0866)
 num2 = torch.mean(torch.randn(2,2))
 print(num2)  # tensor(0.2038)
```

If the preceding script is executed for multiple times, the execution results **num1** and **num2** are the same. However, if the same randomness API is executed twice in the script, the results are different, that is, **num1** is not equal to **num2**.

```diff
 import torch
+from msprobe.pytorch import seed_all

 # seed_all fixes random seeds, enables deterministic computing, and enhances random seed fixation.
+seed_all(seed=1234, mode=True, is_enhanced=True)
 num1 = torch.mean(torch.randn(2,2))
 print(num1)  # tensor(-0.0866)
 num2 = torch.mean(torch.randn(2,2))
 print(num2)  # tensor(-0.0866)
```

In the preceding script, the same randomness API is executed twice, and the generated results are the same. Even if the script is executed for multiple times, the two values are the same.

**Remarks**

By default, the dump function of the tool does not fix randomness. To ensure that the data collected each time is consistent, it is advised to call the **seed_all** API before dumping data.

The following table lists the range of random numbers that can be fixed by **seed_all**.

| API                                      | Fixed Random Number                                     |
|------------------------------------------|--------------------------------------------|
| os.environ['PYTHONHASHSEED'] = str(seed) | Disables hash randomization in Python.                         |
| os.environ['HCCL_DETERMINISTIC'] = True  | Fixes deterministic computing of communication operators.                              |
| np.random.seed(seed)                     | Sets the seed of the random generator in NumPy.                          |
| torch.manual_seed(seed)                  | Sets the random seed of the current CPU.                              |
| torch.cuda.manual_seed(seed)             | Sets the random seed of the current GPU.                              |
| torch.cuda.manual_seed_all(seed)         | Sets the random seed of all GPUs.                              |
| torch_npu.npu.manual_seed(seed)          | Sets the random seed of the current NPU.                              |
| torch_npu.npu.manual_seed_all(seed)      | Sets the random seed of all NPUs.                              |
| torch.use_deterministic_algorithms(True) | Enables deterministic computing for CUDA/CANN. (Note that this method can be called only when **mode** is set to **True**.)|
| torch.backends.cudnn.enable=False        | Disables cuDNN.                                   |
| torch.backends.cudnn.benchmark=False     | Deterministically selects algorithms for cuDNN.                             |
| torch.backends.cudnn.deterministic=True  | Uses only deterministic convolution algorithms for cuDNN.                          |
| torch.nn.functional.dropout              | Sets **p** of the **dropout** API to **0**.                         |
| torch.nn.functional.dropout2d            | Sets **p** of the **dropout2d** API to **0**.                       |
| torch.nn.functional.dropout3d            | Sets **p** of the **dropout3d** API to **0**.                       |
| torch.nn.Dropout                         | Sets **p** of the **Dropout** API to **0**.                         |
| torch.nn.Dropout2d                       | Sets **p** of the **Dropout2d** API to **0**.                       |
| torch.nn.Dropout3d                       | Sets **p** of the **Dropout3d** API to **0**.                       |

The comparison of dump data is meaningful only when the model inputs of the CPU, GPU, and NPU are the same. **seed_all** cannot ensure that the model inputs are the same. The following table lists a scenario where input consistency needs to be ensured.

| Scenario         | Fixing Method      |
|-------------|------------|
| Dataset shuffle| Disable shuffle.|

Example of disabling shuffle:

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)
```

### random_save

**Function**

Finds the call positions of all randomness APIs in the model script and outputs the result to a .csv file, helping you quickly locate the random number generation points in the model.

**Prototype**

```python
random_save(output_path="./output")
```

**Parameters**

**output_path**: (optional) path for storing randomness APIs. The value is of the string type. The default value is **./output**, indicating that randomness APIs are saved in the **output** directory of the current path. Example: output_path="./output".

**Returns**

None

**Example**

For details, see [Collecting Randomness APIs](#collecting-randomness-apis).

## Supplementary Notes

For details about performance changes in dump statistics mode and data volume collected in dump tensor mode, see [Dump Baseline] (../baseline/pytorch_data_dump_perf_baseline.md).
