# Single-Point Saving Tool

## Overview

The dump capabilities at the L0, L1, and mix levels have blind spots. The input and output of non-API or non-module on the network cannot be dumped in batches. The single-point saving tool provides functions similar to **np.save** and **print**, to save specified variables. In addition, the tool is enhanced for foundation model scenarios and has the following features:

- The backward gradient result of a variable can be saved.
- The nested structure data (such as list and dict) can be directly saved without manual traversal.
- Data can be automatically saved by rank.
- Data can be saved by step.
- The number of calls can be automatically counted.
- Statistics or tensors can be saved (not supported in MindSpore static graph mode).
- Asynchronous saving is supported.

The single-point saving tool may be used across files. For details about how to enable the tool, see [Cross-File Data Collection](./pytorch_data_dump_instruct.md#cross-file-data-dump).

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

The PyTorch and MindSpore frameworks are supported.

## Application Scenarios

## Dynamic Graph (PyTorch & MindSpore)

### Enablement Method

#### Configuration File Description

For more details, see [General Configuration](./config_json_introduct.md#general-configuration).

| Parameter      | Description                                                                                                   | Mandatory (Yes/No)|
| ---------- | ------------------------------------------------------------------------------------------------------- | -------- |
| task       | Dump task type, which is of the string type. In the single-point saving scenario, the value can only be **statistics** or **tensor**.                             | Yes      |
| level      | Dump level, which is of the string type. Data is collected at different levels. In the single-point saving scenario, the value is **debug**.                               | Yes      |
| dump_path  | Dump data directory, which is of the string type.                                                                     | Yes      |
| rank       | Rank whose data is to be collected, which is of the list[Union[int, str]] type.                                             | No      |
| step       | Step whose data is to be collected, which is of the list[Union[int, str]] type.                                                 | No      |
| async_dump | Asynchronous dump switch, which is of the bool type. In this mode, **summary_mode** does not support MD5 values or statistical calculation of complex tensors.| No      |

The table below lists a sub-configuration item for a **statistics** task.

| Parameter        | Description                                                        | Mandatory (Yes/No)|
| ------------ | ------------------------------------------------------------ | -------- |
| summary_mode | Dump file output mode, which is of the string type. The value can be **statistics** or **md5**. For details, see [task = statistics](./config_json_introduct.md#task = statistics).| No      |

There is no sub-configuration items for a **tensor** task.

#### API Call Description

Call **PrecisionDebugger.save**, pass the variables to be saved, specify variable names, and specify whether to save the backward data. For details about the input parameters of the API, see [PyTorch Single-Point Saving API](./pytorch_data_dump_instruct.md#save) or [MindSpore Single-Point Saving API](./mindspore_data_dump_instruct.md#save).

#### Example

Configuration file:
(The PyTorch scenario is used as an example. In the MindSpore scenario, you only need to import the package from the msprobe.mindspore module.)

```json
{
    "task": "statistics",
    "dump_path": "./dump_path",
    "rank": [],
    "step": [],
    "level": "debug",
    "async_dump": false,
    "statistics": {
        "summary_mode": "statistics"
    }
}
```

Initialization:

```python
# Training startup script
from msprobe.pytorch import PrecisionDebugger
debugger = PrecisionDebugger("./config.json")
for data, label in data_loader:
    # Perform model training.
    train(data, label)

```

Initialization (without a configuration file):

```python
# Training startup script
from msprobe.pytorch import PrecisionDebugger
debugger = PrecisionDebugger(dump_path="dump_path", level="debug")
for data, label in data_loader:
    # Perform model training.
    train(data, label)

```

Example of calling the save API (using PyTorch code as an example; the usage is the same for MindSpore):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from msprobe.pytorch import PrecisionDebugger, seed_all
# Instantiate PrecisionDebugger before model training.
debugger = PrecisionDebugger(dump_path="dump_path", level="debug")

# Define a network.
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        debugger.save(x2, "x2", save_backward=True)  # Call the save API.
        r1 = F.relu(x2)
        return r1

if __name__ == "__main__":
    module = ModuleOP()

    x = torch.randn(10, 8)
    out = module(x)
    loss = out.sum()
    loss.backward()
```

Save data by step (using PyTorch code as an example; the usage is the same for MindSpore):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from msprobe.pytorch import PrecisionDebugger
# Instantiate PrecisionDebugger before model training.
debugger = PrecisionDebugger(dump_path="dump_path", level="debug")

# Define a network.
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        debugger.save(x2, "x2", save_backward=True)  # Call the save API.
        r1 = F.relu(x2)
        return r1

if __name__ == "__main__":
    module = ModuleOP()
    train_iter = 10
    for i in range(train_iter):
        x = torch.randn(10, 8)
        out = module(x)
        loss = out.sum()
        loss.backward()
        debugger.step()  # Call debugger.step to distinguish steps for data saving.

```

## Static Graph (MindSpore)

### Enablement Method

### API Description

The tool provides three external APIs for saving data during training.

| API | Description                | Supported Device      | MindSpore Version| Application Scenario                                  |
| --------- | ------------------------ | -------------- | ------------- | ---------------------------------------------- |
| save      | Saves tensor data of forward propagation.| Ascend/GPU/CPU | >= 2.6.0      | Ascend is supported only in graph mode. All platforms are supported in PyNative mode.|
| save_grad | Saves gradient data of backpropagation.  | Ascend/GPU/CPU | >= 2.6.0      | Ascend is supported only in graph mode. All platforms are supported in PyNative mode.|
| step      | Updates the number of training steps.            | Ascend/GPU/CPU | >= 2.6.0      | Managing step directory for storing data                        |

### API Details

#### 1. save

```python
save(save_dir: str, name: str, data: Union[Tensor, List, Tuple, Dict])
```

**Parameters**:

- `save_dir`: data storage directory
- `name`: data identifier (used as the prefix of the file name)
- `data`: multiple data types are supported.
  - `mindspore.Tensor`: single tensor
  - `List/Tuple/Dict`: nested structure (automatically expanded and saved)

**Example**:

```python
from msprobe.mindspore import save

class Net(nn.Cell):
    def construct(self, x):
        save("./dump_data", 'input', x)  # Save the input data.
        return x * 2
```

#### 2. save_grad

```python
save_grad(save_dir: str, name: str, data: Tensor) -> Tensor
```

**Parameters**:

- `save_dir`: gradient storage directory
- `name`: gradient identifier (used as the prefix of the file name)
- `data`: Must be of type `mindspore.Tensor`.

**Precautions**:

- The return value must be received and passed back to the original computational graph.
- This operation does not affect computing precision.

**Example**:

```python
from msprobe.mindspore import save_grad

class Net(nn.Cell):
    def construct(self, x):
        x = save_grad("./dump_data", 'grad', x)  # Save the gradient data.
        return x * 2
```

#### 3. step

```python
step()
```

**Description**:

- Counts the number of incremental training steps.
- Manage step directories (such as **step0/** and **step1/**) for storing data.
- If this API is not called, all data is stored in the same step directory.

**Example**:

```python
from msprobe.mindspore import save, step

# Training epoch
for epoch in range(epochs):
    train_one_epoch()
    step()  # Update the step after each epoch.
```

## Return Value

### Dynamic Graph (PyTorch & MindSpore)

* **task = statistics**: `debug.json` that contains variable statistics is generated in the dump directory.
  The key of the statistics in `debug.json` is named in the format of {`{variable_name}{grad_flag}.{count}.debug`.
* **task = tensor**: In addition to `debug.json` that contains variable statistics in the dump directory, a tensor binary file is stored in `dump_tensor_data`. The file name is in the format of `{variable_name}{grad_flag}.{count}.debug.{indexes}.{file_suffix}`.

  - **variable_name**: name of the variable passed to the save API.
  - **grad_flag**: backward flag. **_grad** indicates backward data, and **""** indicates forward data.
  - **count**: number of calls with the same variable name.
  - **indexes**: index of the nested structure data to be saved. For example, if the nested structure is `{"key1": "value1", "key2": ["value2", "value3"]}`, the index of **value2** is **key2.0**.
  - **file_suffix**: file name extension. The value is **pt** in the PyTorch scenario and **npy** in the MindSpore scenario.

### Static Graph (MindSpore)

The `{step}/{rank}` directory is generated in the specified directory `save_dir`, and a.npy file with the specified `{name}` is generated in the directory. If the `save_grad` API is called, a .npy file with `{name}_grad` is generated.

For example, `save("./test_dump", 'x', x)` -> `./test_dump/step0/rank0/x_float32_0.npy`

or `z = save_grad("./test_dump", 'z', z)` -> `./test_dump/step0/rank0/z_grad_float32_0.npy`.

The structure is as follows:

```ColdFusion
./save_dir/
    ├── step0/
    │   ├── rank0/
    │   │   ├── x_float32_0.npy       # Forward data saved by save
    │   │   └── z_grad_float32_0.npy  # Gradient data saved by save_grad
    ├── step1/
    │   ├── rank0/
    │   │   ├── ...
```
