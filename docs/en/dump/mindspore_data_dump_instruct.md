
# Precision Data Collection in MindSpore

## Overview

msProbe collects the precision data (dump) of a model during running by adding the `PrecisionDebugger` API to the training script and starting training. This tool can collect precision data of different levels in MindSpore static and dynamic graph scenarios.

For details about performance changes in dump statistics mode and data volume collected in dump tensor mode, see [Dump Baseline] (../baseline/mindspore_data_dump_perf_baseline.md).

**Note**:

* Due to the limitation of the automatic differentiation mechanism of the MindSpore framework, the dumped data may be missing reverse data for in-place operation module/API and its previous module/API.

* Loss or gnorm may change after msProbe is used. This can be caused by synchronization introduced by item operations in the tool or by the hook mechanisms of the PyTorch or MindSpore framework. For details, see [Cause Analysis of Model Calculation Result Changes](../faq.md#model-computation-result-change).

**Concepts**

* **Static graph**: The network structure is determined during compilation. The static graph mode has high training performance but is difficult to debug.
* **Dynamic graph**: The network is dynamically constructed at runtime. Compared with static graph mode, dynamic graph mode is easier to debug but more difficult to execute efficiently
* **High-level API** refer to APIs that encapsulate training processes, such as `mindspore.train.Model`.
* **Just-In-Time (JIT) compilation**: MindSpore provides the JIT technology to further optimize performance. In JIT mode, code is parsed into an intermediate representation (IR) through abstract syntax tree (AST) parsing or Python bytecode parsing. The IR serves as the unique representation of the code, which the compiler optimizes to improve runtime performance. JIT compilation mode, also known as static graph mode, is the counterpart to dynamic graph mode.
* **Primitive op**: foundational operator in MindSpore, which is usually defined by `mindspore.ops.Primitive` and provides bottom-layer operator operation APIs.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

The MindSpore framework is supported.

## Quick Start

The following uses a simple example to describe how to use msProbe to collect precision data in MindSpore.

For details, see [Dynamic Graph Quick Start](mindspore_dump_quick_start.md).

## Precision Data Collection in Static Graph Mode

### Overview

In static graph mode, msProbe supports data collection at L0 and L2 levels. If the MindSpore version is later than 2.5.0 and L2 data needs to be collected, you must use the **mindstudio-probe .whl** package with the `--include-mod=adump` option added during compilation to install msProbe.

- **L0 level (cell level)**: collects data of `cell` objects. This level is applicable to the scenario where a specific network module needs to be analyzed. Only MindSpore 2.7.0 and later versions support this feature.

- **L2 level (kernel level)**: collects the input and output data of bottom-layer operators. This level is applicable to in-depth analysis of operator-level precision problems.

For details, see "level" in [Configuration File Introduction](./config_json_introduct.md#general-configuration).

Common APIs:

- [seed_all](#seed_all): fixes randomness in the network and enables deterministic computing.
- [msprobe.mindspore.PrecisionDebugger](#msprobemindsporeprecisiondebugger): loads the dump configuration file to determine the detailed dump configuration.
- [start](#start): starts precision data collection.
- [stop](#stop): stops precision data collection.
- [step](#step): ends data collection of a step, flushes all data, and updates dump parameters.

For details about more APIs, see [API Description](#api-description).

**Precautions**

None

### Examples (L0)

**Note**: The L0 dump function of the static graph is implemented based on the **mindspore.ops.TensorDump** operator. In graph mode on the Ascend platform, you can set the environment variables [MS_DUMP_SLICE_SIZE/MS_DUMP_WAIT_TIME](https://www.mindspore.cn/docs/en/r2.5.0/api_python/env_var_list.html) to prevent operator execution failures when output tensors are large or dense.

#### High-Level Model APIs Not Used

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")

# Define and initialize the model and loss function.
# ...
model = Network()
# Dataset iteration typically marks the start of model training.
for data, label in data_loader:
    debugger.start(model) # Collect data of cell objects at the L0 level.
    # Logical flow executed at each step of the model
    grad_net = ms.grad(model)(data)
    # ...
    debugger.step()        # Update the number of iterations.
```

#### High-Level Model APIs Used

```python
import mindspore as ms
from mindspore.train import Model
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
from msprobe.mindspore.common.utils import MsprobeStep
debugger = PrecisionDebugger(config_path="./config.json")

# Define and initialize the model and loss function.
# ...

model = Network()
# Collect data of cell objects at the L0 level.
debugger.start(model)
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
trainer.train(1, train_dataset, callbacks=[MsprobeStep(debugger)])
```

### Examples (L2)

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.start()
# Do not place the preceding initialization process after model instantiation or mindspore.communication.init calling.
# Model definition and training code
# ...
debugger.stop()
debugger.step()
```

## Precision Data Collection in Dynamic Graph Mode

### Overview

In dynamic graph mode, msProbe supports data collection at L0, L1, mix, L2, and debug levels.

- **High-level APIs used**
  - The `MsprobeStep` callback class is required to control the start and stop of data collection. This method applies to L0, L1, mix, and L2 data collection.

- **High-level APIs not used**
  - Manually call APIs such as `start`, `stop`, and `step` in training epochs. This method applies to L0, L1, mix, and L2 data collection.

> **Note**: In dynamic graph mode, the part decorated by `mindspore.jit` is actually executed in static graph mode. In this case, the kernel-level (L2) data collection method is the same as that used in the static graph mode.

- **L0 level (cell level)**: collects data of cell objects. This level is applicable to the scenario where a specific network module needs to be analyzed.

- **L1 level (API level)**: collects the input and output data of MindSpore APIs. This level is applicable to locating API-level precision problems.

- **mix (module level + API level)**: collects module-level and API-level data based on L0 and L1 levels. This level is applicable to the scenario where module-level and API-level precision problems need to be analyzed.

- **debug level (single-point saving)**: saves the forward and backward data of variables on the network at a single point. This level is applicable to the scenario where you are familiar with the network structure.

For details, see "level" in [Configuration File Introduction](./config_json_introduct.md#general-configuration).

Common APIs:

- [seed_all](#seed_all): fixes randomness in the network and enables deterministic computing.
- [msprobe.mindspore.PrecisionDebugger](#msprobemindsporeprecisiondebugger): loads the dump configuration file to determine the detailed dump configuration.
- [start](#start): starts precision data collection.
- [stop](#stop): stops precision data collection.
- [step](#step): ends data collection of a step, flushes all data, and updates dump parameters.

For details about more APIs, see [API Description](#api-description).

**Precautions**

None

### Examples (L0, L1, and mix)

#### High-Level Model APIs Not Used

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")

# Define and initialize the model and loss function.
# ...
model = Network()
# Dataset iteration typically marks the start of model training.
for data, label in data_loader:
    debugger.start()        # Collect non-primitive op data at the L1 level.
    # debugger.start(model) # Collect primitive op data at the L0, mix, or L1 level.
    # Logical flow executed at each step of the model
    grad_net = ms.grad(model)(data)
    # ...
    debugger.stop()        # Disable data dumping.
    debugger.step()        # Update the number of iterations.
```

#### High-Level Model APIs Used

```python
import mindspore as ms
from mindspore.train import Model
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
from msprobe.mindspore.common.utils import MsprobeStep
debugger = PrecisionDebugger(config_path="./config.json")

# Define and initialize the model and loss function.
# ...

model = Network()
# Called only when data is collected for cell objects at the L0, mix levels, as well as for primitive op at the L1 level.
# debugger.start(model)
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
trainer.train(1, train_dataset, callbacks=[MsprobeStep(debugger)])
```

#### Forward and Backward Data Collection of a Specified Code Block

```python
import mindspore as ms
from mindspore import set_device
from mindspore.train import Model
ms.set_context(mode=ms.PYNATIVE_MODE)

set_device("Ascend", 0)

from msprobe.mindspore import PrecisionDebugger
from msprobe.mindspore.common.utils import MsprobeStep
debugger = PrecisionDebugger(config_path="./config.json")

# Define and initialize the model and loss function.
# ...
# Dataset iteration typically marks the start of model training.
for data, label in data_loader:
    debugger.start()  # Enable data dumping.
    # Logical flow executed at each step of the model
    output = model(data)

    debugger.stop()  # Insert this function after the start function. Only the forward and backward data between the start function and this function is dumped. Segment-based collection using start-stop-start-stop-step sequences is supported.
    # ...
    loss.backward()
    debugger.step ()  # Stop the dump of a step.
```

### Examples (L2)

#### High-Level Model APIs Not Used

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.start()
# Do not place the preceding initialization process after model instantiation or mindspore.communication.init calling.

# Define and initialize the model and loss function.
# ...
model = Network()
# Dataset iteration typically marks the start of model training.
for data, label in data_loader:
    # Logical flow executed at each step of the model
    grad_net = ms.grad(model)(data)
    # ...
```

#### High-Level Model APIs Used

```python
import mindspore as ms
from mindspore.train import Model
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.start()
# Do not place the preceding initialization process after model instantiation or mindspore.communication.init calling.

# Define and initialize the model and loss function.
# ...

model = Network()
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
trainer.train(1, train_dataset)
```

#### token_range Specified During Model Inference Data Collection

The MindTorch suite is required to reconstruct the original inference code. After packaging, the usage is the same as PyTorch, with the only difference being the import of **PrecisionDebugger** from **msprobe.mindspore**.

```Python
from vllm import LLM, SamplingParams
from msprobe.mindspore import PrecisionDebugger, seed_all
# Fix randomness prior to initiating model training.
seed_all()
# Do not insert the PrecisionDebugger initialization process into the looped code.
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")
# Define and initialize the model.
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model='...')
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
# Enable data dumping and collect data during the first to third cycles of character-by-character cyclic inference for the inference model.
debugger.start(model=model, token_range=[1,3])
# Logic for generating the inference model
output = llm.generate(prompts, sampling_params=sampling_params)
# Disable data dumping and flush data.
debugger.stop()
debugger.step()
```

### Output Description

After precision data is collected, **./dump_path** of the dump data file is displayed.

```ColdFusion
dump.json is at ./dump_path/step*
```

***** indicates the step number. When the printed step number is the last step, dumping ends. The **dump.json** file of a step is stored in the corresponding step directory.

## Dump Result File

### Static Graph

After training is complete, the data is saved in the directory specified by `dump_path`.

- The directory structure of L0 dump is the same as that in the dynamic graph scenario.

- The directory structure of L2 dump is as follows:

  If **jit_level** is set to **O2**, the MindSpore version is 2.5.0 or later, and the `--include-mod=adump` option is added when the msProbe release package or source code is compiled, the directory structure is as follows:

  ```ColdFusion
  ├── dump_path
  │   ├── acl_dump_{device_id}.json
  │   ├── rank_0
  │   |   ├── {timestamp}
  │   |   │   ├── step_0
  |   |   |   |    ├── AssignAdd.Default_network-TrainOneStepCell_optimzer-Gsd_AssignAdd-op0.0.10.1735011096403740.input.0.ND.INT32.npy
  |   |   |   |    ├── Cast.Default_network-TrainOneStepCell_network-WithLossCell__backbone-Net_Cast-op0.9.10.1735011096426349.input.0.ND.FLOAT.npy
  |   |   |   |    ├── GetNext.Default_GetNext-op0.0.11.17350110964032987.output.0.ND.FLOAT.npy
  |   |   |   |    ...
  |   |   |   |    ├── RefDAata.accum_bias1.6.10.1735011096424907.output.0.ND.FLOAT.npy
  |   |   |   |    ├── Sub.Default_network-TrainOneStepCell_network-WithLossCell__backbone-Net_Sub-op0.10.10.1735011096427368.input.0.ND.BF16
  |   |   |   |    └── mapping.csv
  │   |   │   ├── step_1
  |   |   |   |    ├── ...
  |   |   |   ├── ...
  |   |   ├── ...
  |   |
  │   ├── ...
  |   |
  │   └── rank_7
  │       ├── ...
  ```

  **Note**

- - If the configuration file specifies .npy format but the actual data format is unsupported by .npy (e.g., bf16 or int4), the tensor is dumped in its original raw format rather than being converted to .npy.
  - If the full name of the original file exceeds 255 characters, the base file name is converted into a random numeric string of 32 characters. `mapping.csv` shows the mapping between the original file name and the converted file name.
  - **acl_dump_{device_id}.json** is an intermediate file generated during dump API call. Generally, you do not need to pay attention to it.

- In other scenarios, for details about dump result files except intermediate files such as **kernel_kbyk_dump.json (jit_level=O0/O1)** and **kernel_graph_dump.json (jit_level=O2)**, see [Dump in Ascend O0/O1 Mode >Introduction to Data Object Directory and Data File](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/dump.html).

### Dynamic Graph

The following is an example of the dump result directory structure:

```lua
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── MintFunctional.relu.0.backward.input.0.npy
|   |   |   |    ├── Mint.abs.0.forward.input.0.npy
|   |   |   |    ├── Functional.split.0.forward.input.0.npy       # Naming format: "{api_type}.{api_name}.{Number of API calls}.{forward/backward}.{input/output}.{Parameter No.}", where {Parameter No.} indicates the nth API input or output. For example, the value 1 indicates the first parameter. If the parameters are in list format, they are ordered based on list. For example, the value 1.1 indicates the first element of the first parameter.
|   |   |   |    ├── Tensor.__add__.0.forward.output.0.npy
|   |   |   |    ...
|   |   |   |    ├── Jit.AlexNet.0.forward.input.0.npy
|   |   |   |    ├── Primitive.conv2d.Conv2d.0.forward.input.0.npy
|   |   |   |    ├── Cell.conv1.Conv2d.forward.0.parameters.weight.npy # Module parameter, in the format of "{Cell}.{cell_name}.{class_name}.forward.{Number of calls}.parameters.{parameter_name}".
|   |   |   |    ├── Cell.conv1.Conv2d.parameters_grad.0.weight.npy      # Module parameter gradient, in the format of "{Cell}.{cell_name}.{class_name}.parameters_grad.{Number of parameter gradient calculations}.{parameter_name}". Notice that {Number of parameter gradient calculations} is not the number of module calls.
|   |   |   |    └── Cell.relu.ReLU.forward.0.input.0.npy              # Naming format: "{Cell}.{cell_name}.{class_name}.{forward/backward}.{Number of calls}.{input/output}.{Parameter No.}", where {Parameter No.} indicates the nth cell parameter. For example, the value 1 indicates the first parameter. If the parameters are in list format, they are ordered based on list. For example, the value 1.1 indicates the first element of the first parameter.
|   |   |   |                                                          # When the model parameter passed during dump is List[mindspore.nn.Cell] or Tuple[mindspore.nn.Cell], the module-level data name contains the index of the module in the list. The naming format is {Cell}.{index}.*, where * indicates the preceding three module-level data naming formats, for example, Cell.0.relu.ReLU.forward.0.input.0.npy.
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

* `rank`: device ID. Data for each rank is stored in its respective `rank{ID}` directory. If the training process cannot obtain the `rank` information, the data of the current process is stored in `proc{pid}`. `pid` indicates the process ID. `proc` is described as follows:
  * In non-distributed scenarios, such as single-process training or single-rank training, the training process does not have rank information. In this case, data is stored in `proc{pid}`. The comparison and hierarchical visualization functions support data parsing in this directory.
  * During foundation model training, both the `rank` and `proc` directories may exist. The reason is that some processes may only complete some data preprocessing operations on CPU and do not have `rank` information. In this case, the directory name is `proc{pid}`. Generally, the data does not have precision problems. The comparison and hierarchical visualization functions do not support data parsing in this directory.
* `dump_tensor_data`: stores collected tensor data.
* `dump.json`: contains statistical information about the forward and backward data of APIs or cells. The information includes the API or cell name associated with the dump data, as well as statistical details for each data item, such as dtype, shape, max, min, mean, L2norm (L2 norm, square root), and CRC-32 data when **summary_mode** is set to **md5**. For details, see [dump.json Description](#dumpjson-description).
* `dump_error_info.log`: dump errors. This log is generated only when the dump tool reports an error.
* `stack.json`: call stack information of APIs or cells.
* `construct.json`: hierarchical structure based on the model level. When **level** is set to **L1**, the **construct.json** file is empty.

During dump, the .npy file is flushed to drive after the corresponding API or module is executed. The .json file writes complete data only after **PrecisionDebugger.stop()** is executed. Therefore, when the program is terminated abnormally, the .npy file corresponding to the executed API has been saved, but the data in the .json file may be lost.

In the dynamic graph scenario, when `mindspore.jit` is used to decorate a specific cell or function, the decorated part is compiled into a static graph for execution.

If **level** is set to **L0** or **mix** in **config.json**, the MindSpore version is 2.7.0 or later, and a cell object of the **construct** method decorated by `mindspore.jit`, the `graph` and `pynative` directories are generated in **dump_path** to store precision data of that cell object and precision data of other cell or API objects, respectively. Example:

```lua
├── dump_path
│   ├── graph
│   |   ├── step0
│   |   |   ├── rank0
│   |   │   |   ├── dump_tensor_data
|   |   |   |   |   ├── ...
│   |   |   |   ├── dump.json
│   |   |   |   ├── stack.json
│   |   |   |   └── construct.json
│   |   |   ├── ...
│   ├── pynative
│   |   ├── step0
│   |   |   ├── rank0
│   |   │   |   ├── dump_tensor_data
|   |   |   |   |   ├── ...
│   |   |   |   ├── dump.json
│   |   |   |   ├── stack.json
│   |   |   |   └── construct.json
│   |   |   ├── ...
```

**Note**: The dump operators inserted before and after the **construct** method decorated by `mindspore.jit` are in both dynamic and static graph modes. Therefore, the precision data of the outermost decorated cell object is collected repeatedly.

- When **level** is set to **L1** in the **config.json** file, if `capture_mode` of `mindspore.jit` is set to **ast** (original PSJit scenario), the decorated part is also dumped to the corresponding directory as an API. If `capture_mode` is set to **bytecode** (original PIJit scenario), the decorated part is restored to a dynamic graph and dumped at API-level.

- When **level** is set to **L2** in the **config.json** file, only the kernel precision data decorated by `mindspore.jit` is dumped. The result directory is the same as that generated for static graph dumping when **jit_level** is set to **O0** or **O1**.

The table below describes prefixes of the .npy file name.

| Prefix          | Description                          |
| -------------- |------------------------------|
| Tensor         | mindspore.Tensor API data           |
| Functional     | mindspore.ops API data              |
| Primitive      | mindspore.ops.Primitive API data    |
| Mint           | mindspore.mint API data            |
| MintFunctional | mindspore.mint.nn.functional API data|
| MintDistributed | mindspore.mint.distributed API data|
| Distributed    | mindspore.communication.comm_func API data   |
| Jit            | Module or function data decorated by "jit"              |
| Cell           | mindspore.nn.Cell class (module) data         |

#### dump.json Description

##### L0 Level

The **dump.json** file at the L0 level contains the forward and backward inputs and outputs of a module, as well as parameters and parameter gradients.
Take the Conv2d module of MindSpore as an example. The module invocation code used in **dump.json** is `output = self.conv2(input) # self.conv2 = mindspore.nn.Conv2d(64, 128, 5, pad_mode='same', has_bias=True)`.

The **dump.json** file contains the following data:

- `Cell.conv2.Conv2d.forward.0`: module forward data. **input_args** is the input data (position parameter), **input_kwargs** is the input data (keyword parameter), **output** is the output data of the module, and **parameters** is module parameters, including weight and bias.
- `Cell.conv2.Conv2d.parameters_grad.0`: parameter gradient, including the gradients of weight and bias.
- `Cell.conv2.Conv2d.backward.0`: module backward data. **input** indicates the backward input gradient (corresponding to the forward output gradient), and **output** indicates the backward output gradient (corresponding to the forward input gradient).

**Note**: When the model parameter passed during dump is **List[mindspore.nn.Cell]** or **Tuple[mindspore.nn.Cell]**, the module-level data name contains the index of the module in the list. The naming format is `{Cell}.{index}.*`, where ***** indicates the naming format of the preceding three types of module-level data, for example, **Cell.0.conv2.Conv2d.forward.0**.

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "mindspore",
 "dump_data_dir": "/dump/path",
 "data": {
  "Cell.conv2.Conv2d.forward.0": {
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
     "data_name": "Cell.conv2.Conv2d.forward.0.input.0.npy"
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
     "data_name": "Cell.conv2.Conv2d.forward.0.output.0.npy"
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
     "data_name": "Cell.conv2.Conv2d.forward.0.parameters.weight.npy"
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
     "data_name": "Cell.conv2.Conv2d.forward.0.parameters.bias.npy"
    }
   }
  },
  "Cell.conv2.Conv2d.parameters_grad.0": {
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
     "data_name": "Cell.conv2.Conv2d.parameters_grad.0.weight.npy"
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
     "data_name": "Cell.conv2.Conv2d.parameters_grad.0.bias.npy"
    }
   ]
  },
  "Cell.conv2.Conv2d.backward.0": {
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
     "data_name": "Cell.conv2.Conv2d.backward.0.input.0.npy"
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
     "data_name": "Cell.conv2.Conv2d.backward.0.output.0.npy"
    }
   ]
  }
 }
}
```

##### L1 Level

The **dump.json** file at the L1 level contains the forward and backward inputs and outputs of an API. Take the **relu** function of MindSpore as an example. The API invocation is `output = mindspore.ops.relu(input)`.

The **dump.json** file contains the following data:

- `Functional.relu.0.forward`: API forward data. **input_args** is the input data (position parameter), **input_kwargs** is the input data (keyword parameter), and **output** is the output data of the API.
- `Functional.relu.0.backward`: API backward data. **input** is the backward input gradient (corresponding to the forward output gradient), and **output** is the backward output gradient (corresponding to the forward input gradient).

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "mindspore",
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
     "data_name": "Functional.relu.0.backward.output.0.npy"
    }
   ]
  }
 }
}  
```

##### mix Level

The **dump.json** file at the mix level contains both L0 and L1 dump data. The file format is the same as that in the preceding examples.

## Appendixes

### API Description

#### seed_all

**Function**

Fixes randomness in the network and enables deterministic computing.

**Prototype**

```python
seed_all(seed=1234, mode=False, rm_dropout=False)
```

**Parameters**

- **seed** (int): (optional) random seed; defaults to **1234**; example: **seed=1000**. This parameter is used to generate random numbers for **random**, **numpy.random**, **mindspore.common.Initializer**, and **mindspore.nn.probability.distribution**. as well as the hash algorithm for the **str**, **bytes**, and **datetime** objects in Python.

- **mode** (bool): (optional) enables deterministic computing. The value can be **True** or **False** (default), for example, **mode=True**. If this parameter is set to **True**, the deterministic running mode of operators and deterministic computing of reduction communication operators (AllReduce, ReduceScatter, and Reduce) are enabled.

  Note: Deterministic computing deteriorates API execution performance. You are advised to enable it when multiple execution results of your model are found to be different.

- **rm_dropout** (bool): (optional) dropout invalidation switch. The value can be **True** or **False** (default), for example, **rm_dropout=True**. If this parameter is set to **True**, **mindspore.ops.Dropout**, **mindspore.ops.Dropout2D**, **mindspore.ops.Dropout3D**, **mindspore.mint.nn.Dropout**, and **mindspore.mint.nn.functional.dropout** will become invalid to avoid network randomness caused by random dropout. You are advised to enable this function before collecting MindSpore data.

  Note: **rm_dropout** can be called to control dropout invalidation or validation only before the dropout instance is initialized.

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

#### msprobe.mindspore.PrecisionDebugger<a name="msprobemindsporeprecisiondebugger"></a>

**Function**

Loads the dump configuration file to determine the detailed dump configuration.

**Prototype**

```Python
PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**Parameters**

- **config_path** (str): (optional) path of the dump configuration file, for example, **./config.json**. This path must be configured in the static graph scenario. In the dynamic graph scenario, this path does not need to be configured. By default, the default configuration of [config.json](../../../python/msprobe/config.json) is used. For details about the configuration items, see [Configuration File Introduction](./config_json_introduct.md).
- Other parameters can be configured in [config.json](../../../python/msprobe/config.json). For details, see [Configuration File Introduction](./config_json_introduct.md).

In the dynamic graph scenario, parameters of this API are not necessary but have a higher priority than the configuration in [config.json](../../../python/msprobe/config.json). However, the number of configurable parameters is less than that in **config.json**. In the static graph scenario, `config_path` must be passed.

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### start

**Function**

Starts precision data collection. In the static graph scenario, this API must be called before **mindspore.communication.init**. If high-level [Model](https://www.mindspore.cn/tutorials/en/r2.3.1/advanced/model.html) APIs are not used for training, this API must be added to the "for" loop together with the **stop** function. Otherwise, this API is used only when **model** needs to be passed.

**Prototype**

```Python
start(model=None, token_range=None, rank_id=None)
```

**Parameters**

- **model**: (optional) instantiated model whose data is to be collected. The input can be of the **mindspore.nn.Cell**, **List[mindspore.nn.Cell]**, or **Tuple[mindspore.nn.Cell]** type. By default, this parameter is not configured. For L0 and mix level dump, **model** must be passed to collect data of all cell objects. If there is a cell object (for example, a cell decorated by `mindspore.jit`) that will be graph-compiled, `start` must be called before the first training step starts. For L1 level dump, if **model** is passed, all API data (including primitive op objects) can be collected. If **model** is not passed, only API data of non-primitive ops is collected. If **token_range** is not **None**, **model** must be passed.

  For a complex model, if only part of the model (for example, **model.A**, **model.A extends mindspore.nn.Cell**) needs to be monitored, you only need to pass the part to be monitored.

  Note: The input layer is not dumped. The tool dumps only the sublayers of the input layer. For example, if **model.A** is passed, **A** is not dumped, but **A.***x* and **A.***x.xx* are dumped.

- **token_range** (list[int, int]): (optional) token range during inference model data collection. The type is **[int, int]**, indicating **[start, end]** of a range. By default, this parameter is not configured.

- **rank_id**: (optional) user-defined rank ID. The value is an integer greater than or equal to 0. By default, this parameter is not configured. The tool obtains the rank ID via the **mindspore.communication.get_rank** API. After this parameter is configured, *{ID}* in the rank folder name in the dump result is the value of this parameter.

  Note: By default, the tool automatically obtains the unique rank ID using the **mindspore.communication.get_rank** API (**get_rank**) in the multi-rank multi-process scenario.
  However, in some special scenarios, **get_rank** may fail to obtain the unique rank ID. For example, in the SGLang DP inference scenario, DP workers are independent, distributed clusters. As a result, **get_rank** returns duplicate rank IDs, causing the rank folders of the dump results to be overwritten and dump data to be lost.
  
  To solve this problem, you can set **rank_id** to name the rank folder, but ensure that rank ID is unique in each process. The value can be obtained from the model script or training/inference framework. For example, **self.gpu_id** in the SGLang inference framework is unique in each process.
  
  Example: `debugger.start(rank_id=self.gpu_id)`

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### stop

**Function**

Stops precision data collection. This API can be added anywhere following the **start** API. If **stop** is added following backward propagation code, the forward and backward data between **start** and **stop** is collected.

If **stop** is added before backward propagation code, [**step**](#step) must be added following backward propagation code to collect the forward and backward data between **start** and **stop**. For details, see [Forward and Backward Data Collection of a Specified Code Block](#forward-and-backward-data-collection-of-a-specified-code-block).

This API is supported only in dynamic graph scenarios where high-level model APIs are not used.

**stop** must be called; otherwise, the flushed precision data may be incomplete.

**Prototype**

```Python
stop()
```

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### step

**Function**

Ends data collection of a step, flushes all data, and updates dump parameters. This API is added to a location where a step ends, and it must be called after **stop**.

It must be used together with **start** and **stop**. It is recommended that it be added after the backward propagation code. Otherwise, backward data may be lost.

This API is supported only in dynamic and static graph scenarios where high-level model APIs are not used.

**Prototype**

```Python
step()
```

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### save

**Function**

Saves the forward and backward values at a single point during network execution and flushes the statistics or tensor files.

**Prototype**

```python
save(variable, name, save_backward=True)
```

**Parameters**

- **variable** (dict, list, tuple, mindspore.Tensor, int, float, str): (required) variable to be saved.
- **name** (str): (required) specified name.
- **save_backward** (bool): (optional) whether to save backward data. The value can be **True** (default) or **False**.

For details, see [Single-Point Saving Tool](./debugger_save_instruct.md).

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### set_init_step

**Function**

Sets the start step. By default, steps begin at 0. After this API is called, the step starts from the specified value. It must be placed before a training loop and cannot be placed inside the loop.

**Prototype**

```Python
set_init_step(step)
```

**Parameters**

**step** (int): (required) start step.

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### register_custom_api

**Function**

Registers a user-defined API with the tool for L1 dump.

**Prototype**

```Python
debugger.register_custom_api(module, api, api_prefix)
```

**Parameters**

The **mindspore.ops.matmul** API is used as an example.

- **module** (class): (required) package to which the API belongs, that is, **mindspore**.
- **api** (str): (required) API name, that is, **matmul**.
- **api_prefix** (str): (required) prefix of the API name in [dump.json](#dumpjson-description).

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

##### restore_custom_api

**Function**

Restores the original user-defined API and cancels dump.

**Prototype**

```Python
debugger.restore_custom_api(module, api)
```

**Parameters**

The **mindspore.ops.matmul** API is used as an example.

- **module** (class): (required) package to which the API belongs, that is, **mindspore**.
- **api** (str): (required) API name, that is, **matmul**.

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

#### msprobe.mindspore.MsprobeStep

**Function**

MindSpore callback class, which automatically calls the **start()** API at the beginning of each step and calls the **stop()** and **step()** APIs at the end of each step. This API collects and manages precision data at L0, L1, and mix levels in the dynamic graph scenario and at L0 level in the static graph scenario where high-level model APIs are used. Its control granularity is a single step, whereas **PrecisionDebugger.start()** and **PrecisionDebugger.stop()** control any training code segment.

**Prototype**

```Python
MsprobeStep(debugger)
```

**Parameters**

**debugger** (class): (required) PrecisionDebugger object.

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

#### msprobe.mindspore.MsprobeInitStep

**Function**

MindSpore callback class, which automatically obtains and sets the initial step value. This API applies only to the resumable training scenario of static graphs in O0/O1 mode.

**Prototype**

```Python
MsprobeInitStep()
```

**Returns**

None

**Example**

For details, see "Examples" in [Precision Data Collection in Static Graph Mode](#precision-data-collection-in-static-graph-mode) or [Precision Data Collection in Dynamic Graph Mode](#precision-data-collection-in-dynamic-graph-mode).

### Modifying the API List

During API-level dump of a dynamic graph, the tool provides a fixed API list. Only the APIs in the list can be used to collect precision data. Generally, you do not need to modify the list. Instead, you can specify the dump API by using the **scope**/**list** field in the **config.json** file. To change the API list, manually modify [support_wrap_ops.yaml](../../../python/msprobe/mindspore/dump/dump_processor/hook_cell/support_wrap_ops.yaml). The following is an example:

```yaml
ops:
  - adaptive_avg_pool1d
  - adaptive_avg_pool2d
  - adaptive_avg_pool3d
```
