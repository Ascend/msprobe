# MSAdapter场景精度数据采集

## 简介

msProbe工具通过在MSAdapter模型训练脚本中添加`PrecisionDebugger`接口并启动训练的方式，采集模型运行过程中的精度数据。该工具支持MSAdapter场景不同level等级的精度数据采集。

采集精度数据时，"statistics"模式的性能开销与"tensor"模式采集的数据量大小与MindSpore场景类似，可以参考[MindSpore场景的精度数据采集基线](../baseline/mindspore_data_dump_perf_baseline.md)。

**注意**：

* 为了正确识别MSAdapter场景，在导入msProbe工具前，需完成torch模块的导入。

* 因MSAdapter模型底层自动微分机制的限制，dump数据中可能会缺少原地操作模块/API及其上一个模块/API的反向数据。

* 使用msProbe工具后模型训练的loss或gnorm值可能发生变化，详见《[模型计算结果改变原因分析](../faq.md#模型计算结果改变原因分析)》。

**基本概念**

* **MSAdapter**：一款MindSpore生态适配工具，可以将PyTorch训练脚本高效迁移至MindSpore框架执行，以实现在不改变原有PyTorch用户开发习惯的情况下，使得PyTorch代码能在昇腾环境上获得高效性能。

* **dump**：采集精度数据，并完成数据持久化的过程。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

仅支持MSAdapter框架。

## 快速入门

1. 配置文件创建

    在当前目录下创建`config.json`文件，用于配置dump参数。文件内容如下：

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

    配置参数详细介绍请参见《[config.json配置文件介绍](./config_json_introduct.md)》。

2. 模型训练脚本编写

    在当前目录下创建一个Python脚本文件，例如`net.py`。脚本内容如下：

    ```python
    import mindspore as ms
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 导入工具的数据采集接口
    from msprobe.mindspore import PrecisionDebugger
    
    # 在模型训练开始前实例化PrecisionDebugger
    debugger = PrecisionDebugger(config_path='./config.json')


    # 定义网络
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
            # 开启数据 dump
            debugger.start(model=net)
    
            out, grad = grad_fn(inputs)
    
            # 停止数据 dump
            debugger.stop()
            # 更新 step 信息
            debugger.step()
    ```

3. 模型训练

    在命令行中执行以下命令：

    ```bash
    python net.py
    ```

    工具将会采集模型训练过程中的精度数据。

4. 查看采集结果

    出现以下回显信息后，说明数据采集完成，即可手动停止模型训练查看采集数据。

    ```markdown
    ****************************************************************************
    *                        msprobe ends successfully.                        *
    ****************************************************************************
    ```

## MSAdapter dump功能介绍

### 功能说明

在MSAdapter场景下，msProbe工具支持**L0**、**L1**、**mix**三个level等级的精度数据采集。

* **L0 level**：采集`模块(nn.Module)`对象的输入输出数据，适用于需要分析特定网络模块的场景。

* **L1 level**：采集`API`的输入输出数据，适用于需要定位API层面精度问题的场景。

* **mix level**：同时采集`模块`级和`API`级数据，适用于需要分析模块和API层面精度问题的场景。

因MSAdapter模型底层使用MindSpore框架，所以采集精度数据时使用的msProbe工具接口与MindSpore动态图场景下使用的dump接口一致，分别为：

* **msprobe.mindspore.seed_all**：用于固定网络中的随机性和开启确定性计算。

* **msprobe.mindspore.PrecisionDebugger**：通过加载dump配置文件的方式来确定dump操作的详细配置。

* **msprobe.mindspore.PrecisionDebugger.start**：启动精度数据采集。

* **msprobe.mindspore.PrecisionDebugger.stop**：停止精度数据采集。

* **msprobe.mindspore.PrecisionDebugger.step**：结束一个step的数据采集，完成所有数据落盘并更新dump参数。

dump接口详细介绍请参见[接口介绍](#接口介绍)章节。

**注意事项**

无

### 使用示例

1. 创建`config.json` dump配置文件，用于配置dump参数。文件内容如下：

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

    dump参数详细介绍请参见《[config.json配置文件介绍](./config_json_introduct.md)》。

2. 编写模型训练脚本`net.py`，并在脚本中插入msprobe工具dump接口。脚本内容如下：

    ```python
    import mindspore as ms
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 导入工具的数据采集接口
    from msprobe.mindspore import PrecisionDebugger, seed_all
    
    # 在模型训练开始前固定随机性
    seed_all()
    # 在模型训练开始前实例化PrecisionDebugger
    debugger = PrecisionDebugger(config_path='./config.json')


    # 定义网络
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
            # 开启数据 dump
            debugger.start(model=net)
    
            out, grad = grad_fn(inputs)
    
            # 停止数据 dump
            debugger.stop()
            # 更新 step 信息
            debugger.step()
    ```

3. 执行模型训练脚本，开始模型训练。执行命令如下：

    ```bash
    python net.py
    ```

    工具将会采集模型训练过程中的精度数据。

### 输出说明

完成精度数据采集后，将打印dump数据文件生成路径。

```ColdFusion
dump.json is at <dump_path>/step<N>
```

* `<dump_path>`：dump配置文件中“dump_path”参数设置的dump数据输出路径。

* `step<N>`：第N+1个step，例如step0，表示该目录存放第1个step的所有dump数据。

dump数据的具体介绍请参见[dump输出件说明](#dump输出件说明)章节。

## dump输出件说明

### 输出件目录结构

MSAdapter场景下，dump输出件的目录结构示例如下：

```lua
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── Tensor.permute.1.forward.npy
|   |   |   |    ├── Functional.linear.5.backward.output.npy    # 命名格式为{api_type}.{api_name}.{API调用次数}.{forward/backward}.{input/output}.{参数序号}, 其中，“参数序号”表示该API的第n个输入或输出，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该API的第1个参数的第1个元素。
|   |   |   |    ...
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.input.0.npy          # 命名格式为{Module}.{module_name}.{class_name}.{forward/backward}.{调用次数}.{input/output}.{参数序号}, 其中，“参数序号”表示该Module的第n个参数，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该Module的第1个参数的第1个元素。
|   |   |   |    ├── Module.conv1.Conv2d.forward.0.parameters.bias.npy  # 模块参数数据：命名格式为{Module}.{module_name}.{class_name}.forward.{调用次数}.parameters.{parameter_name}。
|   |   |   |    └── Module.conv1.Conv2d.parameters_grad.0.weight.npy     # 模块参数梯度数据：命名格式为{Module}.{module_name}.{class_name}.parameters_grad.{parameter_name}, 其中，参数梯度中的计数是参数梯度的计算次数，不是模块的调用次数。
|   |   |   |                                                          # 当dump时传入的model参数为List[torch.nn.Module]或Tuple[torch.nn.Module]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为{Module}.{index}.*，*表示以上三种模块级数据的命名格式，例如：Module.0.conv1.Conv2d.forward.0.input.0.npy。
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

* `rank`：设备ID，每张卡的数据保存在对应的`rank{ID}`目录下。非分布式场景下没有rank ID，目录名称为`proc{pid}`，pid为进程ID。

* `dump_tensor_data`：保存采集到的张量数据。

* `dump.json`： 保存API或Module输入输出数据的统计量信息。包含dump数据的API名称或Module名称，各数据的dtype、 shape、max、min、mean、L2norm（L2范数，平方根）统计信息以及当配置 summary_mode="md5" 时的CRC-32 数据。

* `dump_error_info.log`：仅在dump工具报错时拥有此记录日志，用于记录dump错误日志。

* `stack.json`：API/Module的调用栈信息。

* `construct.json`：分层分级结构，level为L1时，construct.json内容为空。

当 dump配置文件中的“task”参数为“tensor”时，dump过程中，npy文件在对应算子或者模块被执行后就会落盘，而json文件则需要在正常执行PrecisionDebugger.stop()后才会写入完整数据。因此如果程序异常终止，终止前被执行算子的相关npy文件得以保存，但json文件中的数据可能丢失。

npy 文件名的前缀含义如下：

| 前缀        | 含义                          |
| ----------- | ---------------------------- |
| Tensor      | torch.Tensor API数据。          |
| Torch       | torch API数据。                 |
| Functional  | torch.nn.functional API数据。   |
| NPU         | NPU亲和API数据。                |
| Distributed | torch.distributed API数据。     |
| Module      | torch.nn.Module 类（模块）数据。 |
| Jit         | 被 "jit" 装饰的模块或函数数据。   |
| Primitive   | mindspore.ops.Primitive API数据。|

### dump.json文件说明

#### L0级别

L0级别的dump.json文件中包含了模块的输入输出、参数以及参数梯度数据。以 Conv2d 模块为例，网络中模块调用代码为：
`output = self.conv2(input) # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`  

dump.json文件中包含以下数据名称：  

* `Module.conv2.Conv2d.forward.0`：模块的前向数据，其中input_args为模块的输入数据（位置参数），input_kwargs为模块的输入数据（关键字参数），output为模块的输出数据，parameters为模块的参数数据，包括权重（weight）和偏置（bias）。

* `Module.conv2.Conv2d.parameters_grad.0`：模块的参数梯度数据，包括权重（weight）和偏置（bias）的梯度。

* `Module.conv2.Conv2d.backward.0`：模块的反向数据，其中input为模块反向的输入梯度（对应前向输出的梯度），output为模块的反向输出梯度（对应前向输入的梯度）。

**说明**：当dump时传入的model参数为List[torch.nn.Module]或Tuple[torch.nn.Module]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为`{Module}.{index}.*`，*表示以上三种模块级数据的命名格式，例如：`Module.0.conv1.Conv2d.forward.0`。     

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

#### L1级别

L1级别的dump.json文件中包含了API的输入输出数据。以relu API为例，网络中API调用代码为：
`output = torch.nn.functional.relu(input)`  

dump.json文件中包含以下数据名称：  

* `Functional.relu.0.forward`：API的前向数据，其中input_args为API的输入数据（位置参数），input_kwargs为API的输入数据（关键字参数），output为API的输出数据。

* `Functional.relu.0.backward`：API的反向数据，其中input为API的反向输入梯度（对应前向输出的梯度），output为API的反向输出梯度（对应前向输入的梯度）。

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

#### mix级别

mix级别的dump.json文件同时包括L0和L1级别的dump数据，文件格式与上述示例相同。  

## 附录

### 接口介绍

#### msprobe.mindspore.seed_all

**功能说明**

用于固定网络中的随机性和开启确定性计算。

**函数原型**

```python
msprobe.mindspore.seed_all(seed=1234, mode=False, rm_dropout=False)
```

**参数说明**

* **seed** (int)：可选参数，随机性种子，默认值：1234。参数示例：seed=1000。该参数用于random、numpy.random、mindspore.common.Initializer、mindspore.nn.probability.distribution的随机数生成以及Python中str、bytes、datetime对象的hash算法。

* **mode** (bool)：可选参数，确定性计算使能，可配置True或False，默认值：False。参数示例：mode=True。该参数设置为True后，将会开启算子确定性运行模式与归约类通信算子（AllReduce、ReduceScatter、Reduce）的确定性计算。

  注意：确定性计算会导致API执行性能降低，建议在发现模型多次执行结果不同的情况下开启。

* **rm_dropout** (bool)：可选参数，控制dropout失效的开关。可配置True或False，默认值：False。参数示例：rm_dropout=True。该参数设置为True后，将会使mindspore.ops.Dropout，mindspore.ops.Dropout2D，mindspore.ops.Dropout3D，mindspore.mint.nn.Dropout和mindspore.mint.nn.functional.dropout失效，以避免因随机dropout造成的网络随机性。建议在采集mindspore数据前开启。

  注意：通过rm_dropout控制dropout失效或生效需要在初始化Dropout实例前调用才能生效。

**返回值说明**

无

**调用示例**

请参见[MSAdapter dump功能介绍](#msadapter-dump功能介绍)中的“使用示例”。

#### msprobe.mindspore.PrecisionDebugger

**功能说明**

通过加载dump配置文件的方式来确定dump操作的详细配置。

**函数原型**

```Python
class msprobe.mindspore.PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**参数说明**

* **config_path** (str)：可选参数，指定dump配置文件路径。参数示例："./config.json"。未配置该路径时，默认使用[config.json](../../../python/msprobe/config.json)文件的默认配置，配置选项含义可见[config.json介绍](./config_json_introduct.md)。

* 其他参数均在[config.json](../../../python/msprobe/config.json)文件中可配，非必选。详细配置可见[config.json介绍](./config_json_introduct.md)。当参数值非None时，优先级高于[config.json](../../../python/msprobe/config.json)文件中的同名配置。

**返回值说明**

无

**调用示例**

请参见[MSAdapter dump功能介绍](#msadapter-dump功能介绍)中的“使用示例”。

#### msprobe.mindspore.PrecisionDebugger.start

**功能说明**

启动精度数据采集。需要与**stop**接口一起添加在训练迭代的for循环内。

**函数原型**

```Python
PrecisionDebugger.start(model=None, token_range=None)
```

**参数说明**

* **model**：可选参数，指定需要采集数据的实例化模型，支持传入torch.nn.Module、list[torch.nn.Module]或Tuple[torch.nn.Module]类型，默认未配置。模块级别（"L0" level）dump与"mix" level dump时，必须传入model才可以采集model内的所有Module对象数据，且若存在会进行图编译的Module对象（例如被`mindspore.jit`装饰的Module），则必须在第一个step训练开始前调用`start`接口。API级别（"L1" level）dump时，传入model可以采集model内包含primitive op对象在内的所有API数据，若不传入model参数，则只采集非primitive op的API数据。token_range不为None时，必须传入model参数。

  对于复杂模型，如果仅需要监测一部分（如model.A，model.A extends torch.nn.Module），传入需要监测的部分（如model.A）即可。

  注意：传入的当前层不会被dump，工具只会dump传入层的子层级。如传入了model.A，A本身不会被dump，而是会dump A.x, A.x.xx等。

- **token_range** (Tuple[int, int])：可选参数，指定推理模型采集时的token循环始末范围，支持传入[int, int]类型，代表[start, end]，范围包含边界，默认未配置。

**返回值说明**

无

**调用示例**

请参见[MSAdapter dump功能介绍](#msadapter-dump功能介绍)中的“使用示例”。

#### msprobe.mindspore.PrecisionDebugger.stop

**功能说明**

停止精度数据采集。在**start**接口调用之后的任意位置添加。若**stop**接口添加在反向计算代码之后，则会采集**start**和该接口之间的前向和反向数据。
若**stop**接口添加在反向计算代码之前，则需要将**step**添加到反向计算代码之后，才能采集**start**和该接口之间的前向和反向数据。

**stop**接口必须调用，否则可能导致精度数据落盘不全。

**函数原型**

```Python
PrecisionDebugger.stop()
```

**参数说明**

无

**返回值说明**

无

**调用示例**

请参见[MSAdapter dump功能介绍](#msadapter-dump功能介绍)中的“使用示例”。

#### msprobe.mindspore.PrecisionDebugger.step

**功能说明**

进行训练step数的自增，完成当前step所有数据的落盘并更新dump参数。在一个step训练结束的位置添加，且必须在**stop**接口之后的位置调用。该接口需要配合**start**和**stop**函数使用，尽量添加在反向计算代码之后，否则可能会导致反向数据丢失。

**函数原型**

```Python
PrecisionDebugger.step()
```

**参数说明**

无

**返回值说明**

无

**调用示例**

请参见[MSAdapter dump功能介绍](#msadapter-dump功能介绍)中的“使用示例”。

### API支持列表

dump API级精度数据时，本工具提供固定的API支持列表，仅支持对列表中的API进行精度数据采集。一般情况下，无需修改该列表，而是通过config.json中的scope/list字段进行dump API指定。若需要改变API支持列表，可以在[support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml)文件内手动修改，如下示例：

```yaml
functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
  - conv1d
  - conv2d
  - conv3d
```
