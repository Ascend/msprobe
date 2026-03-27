
# MindSpore场景精度数据采集

## 简介

msProbe工具通过在训练脚本中添加`PrecisionDebugger`接口并启动训练的方式，采集模型在运行过程中的精度数据（dump）。该工具支持对MindSpore的静态图和动态图场景进行不同level等级的精度数据采集。

dump "statistics"模式的性能膨胀大小与"tensor"模式采集的数据量大小，可以参考[dump基线](../baseline/mindspore_data_dump_perf_baseline.md)。

**注意**：

* 因MindSpore框架自动微分机制的限制，dump数据中可能会缺少原地操作模块/API及其上一个模块/API的反向数据。

* 使用msProbe工具后loss/gnorm发生变化：可能是工具中的item操作引入同步，PyTorch或MindSpore框架的hook机制等原因导致的，详见《[模型计算结果改变原因分析](../faq.md#模型计算结果改变原因分析)》。

**基本概念**

* **静态图**：在编译时就确定网络结构，静态图模式拥有较高的训练性能，但难以调试。
* **动态图**：运行时动态构建网络，相较于静态图模式虽然易于调试，但难以高效执行。
* **高阶API**：如`mindspore.train.Model`，封装了训练过程的高级接口。
* **JIT（Just-In-Time编译）**：MindSpore提供JIT（just-in-time）技术进一步进行性能优化。JIT模式会通过AST树解析的方式或者Python字节码解析的方式，将代码解析为一张中间表示图（IR，intermediate representation）。IR图作为该代码的唯一表示，编译器通过对该IR图的优化，来达到对代码的优化，提高运行性能。与动态图模式相对应，这种JIT的编译模式被称为静态图模式。
* **Primitive op**：MindSpore中的基本算子，通常由`mindspore.ops.Primitive`定义，提供底层的算子操作接口。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

支持MindSpore框架。

## 快速入门

以下通过一个简单的示例，展示如何在MindSpore中使用msProbe工具进行精度数据采集。

您可以参考《[动态图快速入门示例](mindspore_dump_quick_start.md)》了解详细步骤。

## 静态图场景精度数据采集功能介绍

### 功能说明

在静态图场景下，msProbe支持**L0 level**和**L2 level**的数据采集。且当MindSpore版本高于2.5.0时，若需采集**L2 level**数据，必须使用编包时添加了`--include-mod=adump`选项的mindstudio-probe whl包进行msProbe工具安装。

- **L0 level（Cell级）**：采集`Cell`对象的数据，适用于需要分析特定网络模块的情况。仅支持2.7.0及以上版本的MindSpore框架。

- **L2 level（Kernel级）**：采集底层算子的输入输出数据，适用于深入分析算子级别的精度问题。

详细介绍请参见《[config.json配置文件介绍](./config_json_introduct.md#通用配置)》中的“level参数”。

常用接口介绍：

- [seed_all](#seed_all)：用于固定网络中的随机性和开启确定性计算。
- [msprobe.mindspore.PrecisionDebugger](#msprobemindsporeprecisiondebugger)：通过加载dump配置文件的方式来确定dump操作的详细配置。
- [start](#start)：启动精度数据采集。
- [stop](#stop)：停止精度数据采集。
- [step](#step)：结束一个step的数据采集，完成所有数据落盘并更新dump参数。

更多接口介绍请参见[接口介绍](#接口介绍)章节。

**注意事项**

无

### 使用示例（L0级别）

**说明**：静态图L0级别的Dump功能是基于mindspore.ops.TensorDump算子实现。在Ascend平台上的Graph模式下，可以通过设置环境变量[MS_DUMP_SLICE_SIZE和MS_DUMP_WAIT_TIME](https://www.mindspore.cn/docs/zh-CN/r2.5.0/api_python/env_var_list.html)解决在输出大Tensor或输出Tensor比较密集场景下算子执行失败的问题。

#### 未使用Model高阶API

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数的定义以及初始化等操作
# ...
model = Network()
# 数据集迭代的地方往往是模型开始训练的地方
for data, label in data_loader:
    debugger.start(model) # 进行L0级别下Cell对象的数据采集时调用
    # 如下是模型每个 step 执行的逻辑
    grad_net = ms.grad(model)(data)
    # ...
    debugger.step()         # 更新迭代数
```

#### 使用Model高阶API

```python
import mindspore as ms
from mindspore.train import Model
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
from msprobe.mindspore.common.utils import MsprobeStep
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数的定义以及初始化等操作
# ...

model = Network()
# 进行L0级别下Cell对象的数据采集时调用
debugger.start(model)
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
trainer.train(1, train_dataset, callbacks=[MsprobeStep(debugger)])
```

### 使用示例（L2级别）

```python
import mindspore as ms
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.start()
# 请勿将以上初始化流程置于模型实例化或mindspore.communication.init调用后
# 模型定义和训练代码
# ...
debugger.stop()
debugger.step()
```

## 动态图场景精度数据采集功能介绍

### 功能说明

在动态图场景下，msProbe支持**L0**、**L1**、**mix**、**L2**、**debug**的数据采集，具体分为以下几种情况：

- **使用高阶API（如Model高阶API）**：
  - 需要使用`MsprobeStep`回调类来控制数据采集的启停，适用于**L0**、**L1**、**mix**、**L2**数据采集。

- **未使用高阶API**：
  - 手动在训练循环中调用`start`、`stop`、`step`等接口，适用于**L0**、**L1**、**mix**、**L2**数据采集。

> **注意**：动态图模式下，使用`mindspore.jit`装饰的部分实际以静态图模式执行，此时的**Kernel级（L2 level）**数据采集方式与静态图场景相同。

- **L0 level（Cell级）**：采集`Cell`对象的数据，适用于需要分析特定网络模块的情况。

- **L1 level（API级）**：采集MindSpore API的输入输出数据，适用于定位API层面的精度问题。

- **mix（模块级+API级）**：在`L0`和`L1`级别的基础上同时采集模块级和API级数据，适用于需要分析模块和API层面精度问题的场景。

- **debug level（单点保存）**：单点保存网络中变量的正反向数据，适用于用户熟悉网络结构的场景。

详细介绍请参见《[config.json配置文件介绍](./config_json_introduct.md#通用配置)》中的“level参数”。

常用接口介绍：

- [seed_all](#seed_all)：用于固定网络中的随机性和开启确定性计算。
- [msprobe.mindspore.PrecisionDebugger](#msprobemindsporeprecisiondebugger)：通过加载dump配置文件的方式来确定dump操作的详细配置。
- [start](#start)：启动精度数据采集。
- [stop](#stop)：停止精度数据采集。
- [step](#step)：结束一个step的数据采集，完成所有数据落盘并更新dump参数。

更多接口介绍请参见[接口介绍](#接口介绍)章节。

**注意事项**

无

### 使用示例（L0、L1、mix级别）

#### 未使用Model高阶API

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数的定义以及初始化等操作
# ...
model = Network()
# 数据集迭代的地方往往是模型开始训练的地方
for data, label in data_loader:
    debugger.start()        # 进行L1级别下非primitive op采集时调用
    # debugger.start(model) # 进行L0, mix级别或L1级别下primitive op的数据采集时调用
    # 如下是模型每个 step 执行的逻辑
    grad_net = ms.grad(model)(data)
    # ...
    debugger.stop()         # 关闭数据dump
    debugger.step()         # 更新迭代数
```

#### 使用Model高阶API

```python
import mindspore as ms
from mindspore.train import Model
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
from msprobe.mindspore.common.utils import MsprobeStep
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数的定义以及初始化等操作
# ...

model = Network()
# 只有进行L0级别下Cell对象，mix级别，L1级别下primitive op的数据采集时才需要调用
# debugger.start(model)
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
trainer.train(1, train_dataset, callbacks=[MsprobeStep(debugger)])
```

#### 采集指定代码块的前反向数据

```python
import mindspore as ms
from mindspore import set_device
from mindspore.train import Model
ms.set_context(mode=ms.PYNATIVE_MODE)

set_device("Ascend", 0)

from msprobe.mindspore import PrecisionDebugger
from msprobe.mindspore.common.utils import MsprobeStep
debugger = PrecisionDebugger(config_path="./config.json")

# 模型、损失函数的定义及初始化等操作
# ...
# 数据集迭代的位置一般为模型训练开始的位置
for data, label in data_loader:
    debugger.start()  # 开启数据dump
    # 如下是模型每个step执行的逻辑
    output = model(data)

    debugger.stop()  # 插入该函数到start函数之后，只dump start函数到该函数之间的前反向数据，可以支持start-stop-start-stop-step分段采集。
    # ...
    loss.backward()
    debugger.step()  # 结束一个step的dump
```

### 使用示例（L2级别）

#### 未使用Model高阶API

```python
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.start()
# 请勿将以上初始化流程置于模型实例化或mindspore.communication.init调用后

# 模型、损失函数的定义以及初始化等操作
# ...
model = Network()
# 数据集迭代的地方往往是模型开始训练的地方
for data, label in data_loader:
    # 如下是模型每个step执行的逻辑
    grad_net = ms.grad(model)(data)
    # ...
```

#### 使用Model高阶API

```python
import mindspore as ms
from mindspore.train import Model
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

from msprobe.mindspore import PrecisionDebugger
debugger = PrecisionDebugger(config_path="./config.json")
debugger.start()
# 请勿将以上初始化流程置于模型实例化或mindspore.communication.init调用后

# 模型、损失函数的定义以及初始化等操作
# ...

model = Network()
trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
trainer.train(1, train_dataset)
```

#### 推理模型采集指定token_range

需要配合mindtorch套件改造原推理代码，套件包装后使用方式与torch一致，唯一区别为import的是msprobe.mindspore下的PrecisionDebugger。

```Python
from vllm import LLM, SamplingParams
from msprobe.mindspore import PrecisionDebugger, seed_all
# 在模型训练开始前固定随机性
seed_all()
# 请勿将PrecisionDebugger的初始化流程插入到循环代码中
debugger = PrecisionDebugger(config_path="./config.json", dump_path="./dump_path")
# 模型定义及初始化等操作
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model='...')
model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
# 开启数据dump, 指定采集推理模型逐字符循环推理中的第1~3次
debugger.start(model=model, token_range=[1,3])
# 推理模型生成的逻辑
output = llm.generate(prompts, sampling_params=sampling_params)
# 关闭数据dump并落盘
debugger.stop()
debugger.step()
```

### 输出说明

完成精度数据采集后，将打印dump数据文件生成路径./dump_path。

```ColdFusion
dump.json is at ./dump_path/step*
```

*表示step的编号，当打印的step编号为最后一个step时，表示dump结束。一个step目录下保存该step的dump数据文件dump.json。

## dump结果文件说明

### 静态图场景

训练结束后，数据将保存在`dump_path`指定的目录下。

- L0级别dump的目录结构与动态图场景下目录结构一致。

- L2级别dump的目录结构如下所示：

  若jit_level=O2，MindSpore版本不低于2.5.0，且使用mindstudio-probe发布包或源码编包时添加了`--include-mod=adump`选项，目录结构示例如下：

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

  **说明**

- - 若配置文件中指定落盘npy格式，但是实际数据格式不在npy支持范围内(如bf16、int4等)，则该tensor会以原始码流落盘，并不会转换为npy格式。
  - 若原始文件全名长度超过255个字符，则文件基础名会被转换为长度为32位的随机数字字符串，原始文件名与转换后文件名的对应关系会保存在同目录下的`mapping.csv`文件中。
  - acl_dump_{device_id}.json为在Dump接口调用过程中生成的中间文件，一般情况下无需关注。

- 其他场景下，除kernel_kbyk_dump.json（jit_level=O0/O1）、kernel_graph_dump.json（jit_level=O2）等无需关注的中间文件外的其他dump结果文件请参见MindSpore官方文档中的[Ascend下O0/O1模式Dump > 数据对象目录和数据文件介绍](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/debug/dump.html#%E6%95%B0%E6%8D%AE%E5%AF%B9%E8%B1%A1%E7%9B%AE%E5%BD%95%E5%92%8C%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6%E4%BB%8B%E7%BB%8D)。

### 动态图场景

dump 结果目录结构示例如下：

```lua
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data
|   |   |   |    ├── MintFunctional.relu.0.backward.input.0.npy
|   |   |   |    ├── Mint.abs.0.forward.input.0.npy
|   |   |   |    ├── Functional.split.0.forward.input.0.npy       # 命名格式为{api_type}.{api_name}.{API调用次数}.{forward/backward}.{input/output}.{参数序号}, 其中，“参数序号”表示该API的第n个输入或输出，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该API的第1个参数的第1个元素。
|   |   |   |    ├── Tensor.__add__.0.forward.output.0.npy
|   |   |   |    ...
|   |   |   |    ├── Jit.AlexNet.0.forward.input.0.npy
|   |   |   |    ├── Primitive.conv2d.Conv2d.0.forward.input.0.npy
|   |   |   |    ├── Cell.conv1.Conv2d.forward.0.parameters.weight.npy # 模块参数数据：命名格式为{Cell}.{cell_name}.{class_name}.forward.{调用次数}.parameters.{parameter_name}。
|   |   |   |    ├── Cell.conv1.Conv2d.parameters_grad.0.weight.npy      # 模块参数梯度数据：命名格式为{Cell}.{cell_name}.{class_name}.parameters_grad.{参数梯度的计算次数}.{parameter_name}，其中，参数梯度中的计数是参数梯度的计算次数，不是模块的调用次数。
|   |   |   |    └── Cell.relu.ReLU.forward.0.input.0.npy              # 命名格式为{Cell}.{cell_name}.{class_name}.{forward/backward}.{调用次数}.{input/output}.{参数序号}, 其中，“参数序号”表示该Cell的第n个参数，例如1，则为第一个参数，若该参数为list格式，则根据list继续排序，例如1.1，表示该Cell的第1个参数的第1个元素。
|   |   |   |                                                          # 当dump时传入的model参数为List[mindspore.nn.Cell]或Tuple[mindspore.nn.Cell]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为{Cell}.{index}.*，*表示以上三种模块级数据的命名格式，例如：Cell.0.relu.ReLU.forward.0.input.0.npy。
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

* `rank`：设备ID，每张卡的数据保存在对应的`rank{ID}`目录下。当训练进程无法获取到`rank`信息时，当前进程的数据保存在`proc{pid}`，pid为进程ID，`proc`的详细介绍如下：
  * ①在非分布式场景下，如单进程训练或单卡训练中，训练进程没有`rank`信息，此时数据保存在`proc{pid}`，比对和分级可视化功能支持该目录下的数据解析。
  * ②在大模型训练过程中，可能既存在`rank`目录又存在`proc`目录，原因是一些进程可能仅在CPU上完成一些数据预处理操作，没有`rank`信息，此时目录名称为`proc{pid}`，这部分数据一般不存在精度问题，比对和分级可视化等功能将不会支持该目录下的数据解析。
* `dump_tensor_data`：保存采集到的张量数据。
* `dump.json`：保存API或Cell前反向数据的统计量信息。包含dump数据的API名称或Cell名称，各数据的dtype、shape、max、min、mean、L2norm（L2范数，平方根）统计信息以及当配置summary_mode="md5"时的CRC-32数据。具体介绍可参考[dump.json文件说明](#dumpjson文件说明)。
* `dump_error_info.log`：仅在dump工具报错时拥有此记录日志，用于记录dump错误日志。
* `stack.json`：API/Cell的调用栈信息。
* `construct.json`：根据model层级展示分层分级结构，level为L1时，construct.json内容为空。

dump过程中，npy文件在对应API或者模块被执行后就会落盘，而json文件则需要在正常执行PrecisionDebugger.stop()后才会写入完整数据，因此，程序异常终止时，被执行API对应的npy文件已被保存，但json文件中的数据可能丢失。

动态图场景下使用`mindspore.jit`装饰特定Cell或function时，被装饰的部分会被编译成**静态图**执行。

config.json文件配置level为L0或mix，且MindSpore版本不低于2.7.0时，若存在construct方法被`mindspore.jit`装饰的Cell对象，则dump_path下将生成`graph`与`pynative`目录，分别存放construct方法被`mindspore.jit`装饰的Cell对象的精度数据、其它Cell或API对象的精度数据。示例如下：

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

**注意**：因为在被`mindspore.jit`装饰的construct方法前后插入的Dump算子既处于动态图模式，也处于静态图模式，所以最外层被装饰的Cell对象的精度数据将被重复采集。

- config.json文件配置level为L1时，若`mindspore.jit`的`capture_mode`参数设置为ast（原PSJit场景），则被装饰的部分也作为API被dump到对应目录；若`mindspore.jit`的`capture_mode`参数设置为bytecode（原PIJit场景），则被装饰的部分会被还原为动态图，按API粒度进行dump。

- config.json文件配置level为L2时，仅会dump被`mindspore.jit`装饰部分的kernel精度数据，其结果目录同jit_level为O0/O1时的静态图dump结果相同。

npy文件名的前缀含义如下：

| 前缀           | 含义                           |
| -------------- |------------------------------|
| Tensor         | mindspore.Tensor API数据。            |
| Functional     | mindspore.ops API数据。               |
| Primitive      | mindspore.ops.Primitive API数据。     |
| Mint           | mindspore.mint API数据。             |
| MintFunctional | mindspore.mint.nn.functional API数据。 |
| MintDistributed | mindspore.mint.distributed API数据。 |
| Distributed    | mindspore.communication.comm_func API数据。    |
| Jit            | 被"jit"装饰的模块或函数数据。               |
| Cell           | mindspore.nn.Cell类（模块）数据。          |

#### dump.json文件说明

##### L0级别

L0级别的dump.json文件包括模块的前反向的输入输出，以及模块的参数和参数梯度。
以MindSpore的Conv2d模块为例，dump.json文件中使用的模块调用代码为：`output = self.conv2(input) # self.conv2 = mindspore.nn.Conv2d(64, 128, 5, pad_mode='same', has_bias=True)`。

dump.json文件中包含以下数据名称：

- `Cell.conv2.Conv2d.forward.0`：模块的前向数据，其中input_args为模块的输入数据（位置参数），input_kwargs为模块的输入数据（关键字参数），output为模块的输出数据，parameters为模块的参数数据，包括权重（weight）和偏置（bias）。
- `Cell.conv2.Conv2d.parameters_grad.0`：模块的参数梯度数据，包括权重（weight）和偏置（bias）的梯度。
- `Cell.conv2.Conv2d.backward.0`：模块的反向数据，其中input为模块反向的输入梯度（对应前向输出的梯度），output为模块的反向输出梯度（对应前向输入的梯度）。

**说明**：当dump时传入的model参数为List[mindspore.nn.Cell]或Tuple[mindspore.nn.Cell]时，模块级数据的命名中包含该模块在列表中的索引index，命名格式为`{Cell}.{index}.*`，*表示以上三种模块级数据的命名格式，例如：`Cell.0.conv2.Conv2d.forward.0`。

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

##### L1级别

L1级别的dump.json文件包括API的前反向的输入输出，以MindSpore的relu函数为例，网络中API调用代码为：`output = mindspore.ops.relu(input)`。

dump.json文件中包含以下数据名称：

- `Functional.relu.0.forward`：API的前向数据，其中input_args为API的输入数据（位置参数），input_kwargs为API的输入数据（关键字参数），output为API的输出数据。
- `Functional.relu.0.backward`：API的反向数据，其中input为API的反向输入梯度（对应前向输出的梯度），output为API的反向输出梯度（对应前向输入的梯度）。

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

##### mix级别

mix级别的dump.json文件同时包括L0和L1级别的dump数据，文件格式与上述示例相同。

## 附录

### 接口介绍

#### seed_all

**功能说明**

用于固定网络中的随机性和开启确定性计算。

**函数原型**

```python
seed_all(seed=1234, mode=False, rm_dropout=False)
```

**参数说明**

- **seed** (int)：可选参数，随机性种子，默认值：1234。参数示例：seed=1000。该参数用于random、numpy.random、mindspore.common.Initializer、mindspore.nn.probability.distribution的随机数生成以及Python中str、bytes、datetime对象的hash算法。

- **mode** (bool)：可选参数，确定性计算使能，可配置True或False，默认值：False。参数示例：mode=True。该参数设置为True后，将会开启算子确定性运行模式与归约类通信算子（AllReduce、ReduceScatter、Reduce）的确定性计算。

  注意：确定性计算会导致API执行性能降低，建议在发现模型多次执行结果不同的情况下开启。

- **rm_dropout** (bool)：可选参数，控制dropout失效的开关。可配置True或False，默认值：False。参数示例：rm_dropout=True。该参数设置为True后，将会使mindspore.ops.Dropout，mindspore.ops.Dropout2D，mindspore.ops.Dropout3D，mindspore.mint.nn.Dropout和mindspore.mint.nn.functional.dropout失效，以避免因随机dropout造成的网络随机性。建议在采集mindspore数据前开启。

  注意：通过rm_dropout控制dropout失效或生效需要在初始化Dropout实例前调用才能生效。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

#### msprobe.mindspore.PrecisionDebugger<a name="msprobemindsporeprecisiondebugger"></a>

**功能说明**

通过加载dump配置文件的方式来确定dump操作的详细配置。

**函数原型**

```Python
PrecisionDebugger(config_path=None, task=None, dump_path=None, level=None, step=None)
```

**参数说明**

- **config_path** (str)：可选参数，指定dump配置文件路径。参数示例："./config.json"。静态图场景必须配置该路径；动态图场景可不配置该路径，默认使用[config.json](../../../python/msprobe/config.json)文件的默认配置，配置选项含义可见[config.json介绍](./config_json_introduct.md)。
- 其他参数均在[config.json](../../../python/msprobe/config.json)文件中可配，详细配置可见[config.json介绍](./config_json_introduct.md)。

动态图场景下此接口参数均不是必要，且优先级高于[config.json](../../../python/msprobe/config.json)文件中的配置，但可配置的参数相比config.json较少。静态图场景下必须传入 `config_path`。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### start

**功能说明**

启动精度数据采集。静态图场景下，必须在mindspore.communication.init调用前添加。如果没有使用[Model](https://www.mindspore.cn/tutorials/zh-CN/r2.3.1/advanced/model.html)高阶API进行训练，则需要与stop函数一起添加在for循环内，否则只有需要传入model参数时，才使用该接口。

**函数原型**

```Python
start(model=None, token_range=None, rank_id=None)
```

**参数说明**

- **model**：可选参数，指定需要采集数据的实例化模型，支持传入mindspore.nn.Cell、List[mindspore.nn.Cell]或Tuple[mindspore.nn.Cell]类型，默认未配置。Cell级别（"L0" level）dump与"mix" level dump时，必须传入model才可以采集model内的所有Cell对象数据，且若存在会进行图编译的Cell对象（例如被`mindspore.jit`装饰的Cell），则必须在第一个step训练开始前调用`start`接口。API级别（"L1" level）dump时，传入model可以采集model内包含primitive op对象在内的所有API数据，若不传入model参数，则只采集非primitive op的API数据。token_range不为None时，必须传入model参数。

  对于复杂模型，如果仅需要监测一部分(如model.A，model.A extends mindspore.nn.Cell)，传入需要监测的部分(如model.A)即可。

  注意：传入的当前层不会被dump，工具只会dump传入层的子层级。如传入了model.A，A本身不会被dump，而是会dump A.x, A.x.xx等。

- **token_range** (list[int, int])：可选参数，指定推理模型采集时的token循环始末范围，支持传入[int, int]类型，代表[start, end]，范围包含边界，默认未配置。

- **rank_id**：可选参数，指定自定义的rank ID，支持传入大于等于0的整数。默认未配置，则工具基于mindspore.communication.get_rank接口获取rank ID；配置此参数后，dump的结果中，rank文件夹名称中的{ID}将使用该参数所配置的值。

  注意：通常情况下，用户无需手动配置rank_id参数，工具默认通过mindspore.communication.get_rank接口（下面简称get_rank接口）可自动获取多卡多进程的唯一rank ID；
  然而，在某些特殊场景下，get_rank接口可能无法正确获取唯一的rank ID。例如，在推理框架sglang的DP推理场景中，各DP worker之间是独立的分布式集群，导致get_rank接口返回重复的rank ID，进而引发dump结果中rank文件夹同名覆盖的问题，造成dump数据丢失。
  
  针对此类特殊场景，可通过配置rank_id参数为rank文件夹命名，但需要保证rank_id在各个进程中唯一。该值通常可在模型脚本或训练推理框架中获取，例如推理框架sglang中的self.gpu_id，其在每个进程中均保持唯一性。
  
  配置示例：`debugger.start(rank_id=self.gpu_id)`

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### stop

**功能说明**

停止精度数据采集。在**start**函数之后的任意位置添加。若**stop**函数添加在反向计算代码之后，则会采集**start**和该函数之间的前反向数据。

若**stop**函数添加在反向计算代码之前，则需要将[**step**](#step)函数添加到反向计算代码之后，才能采集**start**和该函数之间的前反向数据，参考[采集指定代码块的前反向数据](#采集指定代码块的前反向数据)。

仅未使用Model高阶API的动态图场景支持。

**stop**函数必须调用，否则可能导致精度数据落盘不全。

**函数原型**

```Python
stop()
```

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### step

**功能说明**

结束一个step的数据采集，完成所有数据落盘并更新dump参数。在一个step结束的位置添加，且必须在**stop**函数之后的位置调用。

该函数需要配合**start**和**stop**函数使用，尽量添加在反向计算代码之后，否则可能会导致反向数据丢失。

仅未使用Model高阶API的动态图和静态图场景支持。

**函数原型**

```Python
step()
```

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### save

**功能说明**

单点保存网络执行过程中正反向数值，并以统计值/张量文件落盘。

**函数原型**

```python
save(variable, name, save_backward=True)
```

**参数说明**

- **variable**(dict, list, tuple, mindspore.Tensor, int, float, str)：必选参数，需要保存的变量。
- **name** (str)：必选参数，指定的名称。
- **save_backward** (bool)：可选参数，是否保存反向数据。可配置True或False，默认值：True。

具体使用样例可参考：[单点保存工具](./debugger_save_instruct.md)。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### set_init_step

**功能说明**

设置起始step数，step数默认从0开始计数，使用该接口后step从指定值开始计数。该函数需要写在训练迭代的循环开始前，不能写在循环内。

**函数原型**

```Python
set_init_step(step)
```

**参数说明**

**step** (int)：必选参数，指定的起始step数。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### register_custom_api

**功能说明**

注册用户自定义的API到工具，用于L1 dump。

**函数原型**

```Python
debugger.register_custom_api(module, api_name, api_prefix)
```

**参数说明**

以mindspore.ops.matmul API为例。

- **module** (class)：必选参数，API所属的包，即传入mindspore。
- **api_name** (str)：必选参数，API名称，即传入"matmul"。
- **api_prefix** (str)：必选参数，[dump.json](#dumpjson文件说明)中API名称的前缀。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

##### restore_custom_api

**功能说明**

恢复用户原有的自定义的API，取消dump。

**函数原型**

```Python
debugger.restore_custom_api(module, api_name)
```

**参数说明**

以mindspore.ops.matmul API为例。

- **module** (class)：必选参数，API所属的包，即传入mindspore。
- **api_name** (str)：必选参数，API名称，即传入"matmul"。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

#### msprobe.mindspore.MsprobeStep

**功能说明**

MindSpore Callback类，自动在每个step开始时调用start()接口，在每个step结束时调用stop()、step()接口。实现使用Model高阶API的动态图场景下L0、L1、mix级别，和静态图场景下L0级别的精度数据采集控制，控制粒度为单个**Step**，而PrecisionDebugger.start、PrecisionDebugger.stop接口的控制粒度为任意训练代码段。

**函数原型**

```Python
MsprobeStep(debugger)
```

**参数说明**

**debugger** (class)：必选参数，PrecisionDebugger对象。

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

#### msprobe.mindspore.MsprobeInitStep

**功能说明**

MindSpore Callback类，自动获取并设置初始step值。仅适用于静态图O0/O1模式的断点续训场景。

**函数原型**

```Python
MsprobeInitStep()
```

**返回值说明**

无

**调用示例**

请参见[静态图场景精度数据采集功能介绍](#静态图场景精度数据采集功能介绍)或[动态图场景精度数据采集功能介绍](#动态图场景精度数据采集功能介绍)中的“使用示例”。

### 修改API支持列表

动态图API级dump时，本工具提供固定的API支持列表，仅支持对列表中的API进行精度数据采集。一般情况下，无需修改该列表，而是通过config.json中的scope/list字段进行dump API指定。若需要改变API支持列表，可以在[support_wrap_ops.yaml](../../../python/msprobe/mindspore/dump/dump_processor/hook_cell/support_wrap_ops.yaml)文件内手动修改，如下示例：

```yaml
ops:
  - adaptive_avg_pool1d
  - adaptive_avg_pool2d
  - adaptive_avg_pool3d
```
