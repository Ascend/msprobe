# 单点保存工具

## 简介

L0、L1、mix级别的dump能力存在盲区，网络中的非API或module的输入输出不会被批量dump下来。单点保存提供类似np.save和print的功能和使用体验，可以保存指定的变量。同时针对大模型场景进行了增强，具备以下特性：

- 可保存变量的反向梯度结果。
- 能直接保存嵌套结构数据（如 list、dict），无需手动遍历。
- 自动分 Rank 保存。
- 可分 Step 保存数据。
- 多次调用时会自动计数。
- 可配置保存统计值（MindSpore静态图暂不支持）或者张量。
- 支持异步保存。

单点保存工具的使用过程中可能会涉及到工具跨文件使用的场景，具体使能方式见[跨文件采集数据](./pytorch_data_dump_instruct.md#跨文件采集数据)。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

支持PyTorch和MindSpore框架。

## 支持场景

## 动态图场景（Pytorch&MindSpore）

### 使能方式

#### 配置文件说明

通用配置 （细节详见[通用配置说明](./config_json_introduct.md#通用配置) ）：

| 参数       | 解释                                                                                                    | 是否必选 |
| ---------- | ------------------------------------------------------------------------------------------------------- | -------- |
| task       | dump 的任务类型，str 类型。 单点保存场景仅支持传入"statistics", "tensor"。                              | 是       |
| level      | dump 级别，str 类型，根据不同级别采集不同数据。单点保存场景传入"debug"。                                | 是       |
| dump_path  | 设置 dump 数据目录路径，str 类型。                                                                      | 是       |
| rank       | 指定对某张卡上的数据进行采集，list[Union[int, str]] 类型。                                              | 否       |
| step       | 指定采集某个 Step 的数据，list[Union[int, str]] 类型。                                                  | 否       |
| async_dump | 异步 dump 开关，bool 类型。该模式下，summary_mode 不支持 md5 值，也不支持复数类型 tensor 的统计量计算。 | 否       |

"statistics" 任务子配置项：

| 参数         | 解释                                                         | 是否必选 |
| ------------ | ------------------------------------------------------------ | -------- |
| summary_mode | 控制 dump 文件输出的模式，str 类型。支持传入"statistics", "md5"。 详细介绍请参见[statistics任务子配置项说明](./config_json_introduct.md#task配置为statistics)。 | 否       |

"tensor" 任务无子配置项。

#### 接口调用说明

调用PrecisionDebugger.save，传入需要保存的变量，指定变量名称以及是否需要保存反向数据。接口入参说明详见[PyTorch单点保存接口](./pytorch_data_dump_instruct.md#save)或[MindSpore单点保存接口](./mindspore_data_dump_instruct.md#save)。

#### 实例

（以PyTorch场景为例，MindSpore场景只需要从msprobe.mindspore模块导包即可）
配置文件

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

初始化

```python
# 训练启动py脚本
from msprobe.pytorch import PrecisionDebugger
debugger = PrecisionDebugger("./config.json")
for data, label in data_loader:
    # 执行模型训练
    train(data, label)

```

初始化（无配置文件）

```python
# 训练启动py脚本
from msprobe.pytorch import PrecisionDebugger
debugger = PrecisionDebugger(dump_path="dump_path", level="debug")
for data, label in data_loader:
    # 执行模型训练
    train(data, label)

```

调用保存接口示例（以PyTorch代码为例，MindSpore使用方法相同）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from msprobe.pytorch import PrecisionDebugger
# 在模型训练开始前实例化PrecisionDebugger
debugger = PrecisionDebugger(dump_path="dump_path", level="debug")

# 定义网络
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        debugger.save(x2, "x2", save_backward=True)  # 调用save接口
        r1 = F.relu(x2)
        return r1

if __name__ == "__main__":
    module = ModuleOP()

    x = torch.randn(10, 8)
    out = module(x)
    loss = out.sum()
    loss.backward()
```

分step保存数据（以PyTorch代码为例，MindSpore使用方法相同）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from msprobe.pytorch import PrecisionDebugger
# 在模型训练开始前实例化PrecisionDebugger
debugger = PrecisionDebugger(dump_path="dump_path", level="debug")

# 定义网络
class ModuleOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=8, out_features=4)
        self.linear_2 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x1)
        debugger.save(x2, "x2", save_backward=True)  # 调用save接口
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
        debugger.step()  # 调用debugger.step用于区分step保存

```

## 静态图场景（MindSpore）

### 使能方式

### 接口说明

工具提供三个对外接口用于保存训练过程中的数据：

| 接口名称  | 功能描述                 | 支持设备       | MindSpore版本 | 使用场景说明                                   |
| --------- | ------------------------ | -------------- | ------------- | ---------------------------------------------- |
| save      | 保存正向传播的tensor数据 | Ascend/GPU/CPU | >= 2.6.0      | 图模式下仅支持Ascend，PyNative模式下支持全平台 |
| save_grad | 保存反向传播的梯度数据   | Ascend/GPU/CPU | >= 2.6.0      | 图模式下仅支持Ascend，PyNative模式下支持全平台 |
| step      | 更新训练步数             | Ascend/GPU/CPU | >= 2.6.0      | 控制数据保存的step目录                         |

### 详细接口定义

#### 1. save 接口

```python
save(save_dir: str, name: str, data: Union[Tensor, List, Tuple, Dict])
```

**参数说明**:

- `save_dir`: 数据保存目录路径
- `name`: 数据标识名称（将作为文件名前缀）
- `data`: 支持多种数据类型：
  - `mindspore.Tensor` 单个张量
  - `List/Tuple/Dict` 嵌套结构（会自动展开保存）

**使用示例**:

```python
from msprobe.mindspore import save

class Net(nn.Cell):
    def construct(self, x):
        save("./dump_data", 'input', x)  # 保存输入数据
        return x * 2
```

#### 2. save_grad 接口

```python
save_grad(save_dir: str, name: str, data: Tensor) -> Tensor
```

**参数说明**:

- `save_dir`: 梯度保存目录路径
- `name`: 梯度标识名称（将作为文件名前缀）
- `data`: 必须是 `mindspore.Tensor`类型

**特别注意**:

- 必须接收返回值并传回原计算图
- 此操作不会影响计算精度

**使用示例**:

```python
from msprobe.mindspore import save_grad

class Net(nn.Cell):
    def construct(self, x):
        x = save_grad("./dump_data", 'grad', x)  # 保存梯度数据
        return x * 2
```

#### 3. step 接口

```python
step()
```

**功能说明**:

- 递增训练步数计数器
- 控制数据保存到不同的step目录（如step0/, step1/等）
- 如果不调用，所有数据会保存到同一个step目录

**使用示例**:

```python
from msprobe.mindspore import save, step

# 训练循环中
for epoch in range(epochs):
    train_one_epoch()
    step()  # 每个epoch后更新step
```

## 输出结果

### 动态图场景（Pytorch&MindSpore）

* **"task" 配置为 "statistics" 场景** ：在 dump 目录下会生成包含变量统计值信息的 `debug.json` 文件。
  `debug.json` 中统计值的key命名格式为 `{variable_name}{grad_flag}.{count}.debug`。
* **"task" 配置为 "tensor" 场景** ：除了在 dump 目录下生成包含变量统计值信息的 `debug.json` 文件外，还会在 dump 子目录 `dump_tensor_data` 中保存张量二进制文件，文件名称格式为 `{variable_name}{grad_flag}.{count}.debug.{indexes}.{file_suffix}`。

  - variable_name： 传入save接口的变量名称。
  - grad_flag： 反向数据标识，反向数据为"_grad"，正向数据为""。
  - count： 调用计数，多次以相同变量名称调用时的计数。
  - indexes： 索引，在保存嵌套结构数据时的索引。例如：嵌套结构为 `{"key1": "value1", "key2": ["value2", "value3"]}`，"value2"的索引为"key2.0"。
  - file_suffix：文件后缀，PyTorch场景为"pt"，MindSpore场景为"npy"。

### 静态图场景（MindSpore）

在指定目录 `save_dir`下生成 `{step}/{rank}`目录，目录下生成指定 `{name}`的npy文件，如果是save_grad接口调用，则会生成 `{name}_grad`的npy文件。

如 `save("./test_dump", 'x', x)` -> `./test_dump/step0/rank0/x_float32_0.npy`。

或如 `z = save_grad("./test_dump", 'z', z)` -> `./test_dump/step0/rank0/z_grad_float32_0.npy`。

结构如下：

```ColdFusion
./save_dir/
    ├── step0/
    │   ├── rank0/
    │   │   ├── x_float32_0.npy       # save保存的正向数据
    │   │   └── z_grad_float32_0.npy  # save_grad保存的梯度数据
    ├── step1/
    │   ├── rank0/
    │   │   ├── ...
```
