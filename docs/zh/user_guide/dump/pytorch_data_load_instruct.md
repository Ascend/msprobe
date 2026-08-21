# PyTorch场景精度数据覆盖加载

## 简介

msProbe工具支持加载先前采集的模块级（L0）输入tensor数据，在模型前向执行时覆盖目标模块的实际输入。该功能用于隔离前序模块累计误差，精准定位本模块及下游的精度问题。

**功能特点**

- **配置驱动**：通过config.json的`load`配置段指定加载源和目标模块，无需修改训练脚本。
- **精确匹配**：模块条目名与dump产出的`.pt`文件名前缀完全一致，避免误匹配。
- **覆盖与dump解耦**：支持只覆盖不dump（`dump_after_load=false`）或覆盖后继续dump。
- **自动对齐**：step/rank自动对齐当前训练进度，也支持手动指定。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../../install_guide/msprobe_install_guide.md)》。

**前提条件**

本功能依赖已有的模块级tensor数据作为加载源。用户必须先执行一次`level=L0`或`level=mix`、`task=tensor`的dump采集，产出`dump_tensor_data/`目录下的`.pt`文件后，才能配置`load`段进行加载覆盖。dump采集的详细操作请参见[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md)。

**约束**

- 仅支持PyTorch框架。
- 源dump与目标模型的**模块结构相同、模块命名一致**（`named_modules()`返回点分路径完全对应）。
- 若模型结构或命名不同（如训推框架命名差异、模型结构差异、精度/切分差异），需用户手动改造源dump数据（重命名`.pt`文件、对齐shape、调整切分），本功能不涉及此类手动改造。
- 加载的tensor与当前模型实际输入的shape或dtype不一致时，工具会打印warning日志但仍执行替换，可能导致forward报错。

## 快速入门

以下通过一个简单的示例，展示如何加载模型A的模块输入数据，覆盖到模型B的对应模块。

1. 使用模型A执行一次精度数据dump，采集模块级（L0）tensor数据。dump采集的详细操作请参见[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md)，示例代码如下：

    ```diff
     import torch
     import torch.nn as nn
    +from msprobe.pytorch import PrecisionDebugger, seed_all

    +seed_all()
    +debugger = PrecisionDebugger(config_path="./config.json")

     # 模型定义与训练循环
     model = MyModel()
     for data, label in data_loader:
    +    debugger.start(model=model)

         output = model(data)
         loss = criterion(output, label)
         loss.backward()
         optimizer.step()

    +    debugger.stop()
    +    debugger.step()
    ```

    其中config.json文件为精度数据采集的配置文件，需使用`level=L0`或`mix`、`task=tensor`采集，详细说明请参见[config.json介绍](./config_json_introduct.md)。

2. 在config.json文件中新增`load`配置段，指定加载源路径和目标模块。

    ```json
    {
        "task": "tensor",
        "dump_path": "/home/data_dump_b",
        "level": "L0",
        "load": {
            "path": "/home/data_dump_a",
            "modules": ["Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"],
            "dump_after_load": false
        }
    }
    ```

3. 使用相同脚本加载模型B，添加dump接口，执行训练。训练过程中工具会自动加载模型A的模块输入数据，覆盖模型B的对应模块输入。

4. 查看结果。若`dump_after_load`为`true`，则覆盖后的dump数据保存在`dump_path`指定的目录中，可用于精度比对。若`dump_after_load`为`false`，则不产生dump数据，用户观察loss即可判断覆盖效果。

详细使用示例请参见[使用示例](#使用示例)。

## 功能介绍

### 功能说明

该功能通过在config.json文件中添加`load`参数，实现对前序模块输入的覆盖。

工具在模块的`forward_pre_hook`中执行输入覆盖，流程如下：

1. 模块前向调用时，hook构造`full_forward_name`（格式与dump侧一致）。
2. 与`load.modules`精确匹配，匹配成功则加载源dump中对应的`.pt`文件。
3. 遍历模块forward的所有位置参数（args）和关键字参数（kwargs），将tensor类型的参数替换为源dump数据。
4. 非tensor参数（如int、str、bool）自动跳过（dump侧未存储`.pt`文件）。
5. 覆盖后的输入传递给模块forward执行。

### 注意事项

#### 文件命名一致性

load侧加载的tensor文件路径与dump侧产出的文件命名严格一致，目录结构相同：

| 参数类型 | dump侧文件名 | load侧加载路径 |
|----------|-------------|---------------|
| 位置参数 args | `{full_forward_name}.input.{i}.pt` | `{load.path}/step{N}/rank{M}/dump_tensor_data/{full_forward_name}.input.{i}.pt` |
| 关键字参数 kwargs | `{full_forward_name}.kwargs.{key}.pt` | `{load.path}/step{N}/rank{M}/dump_tensor_data/{full_forward_name}.kwargs.{key}.pt` |

dump侧的目录结构、文件命名规则请参见[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md#dump结果文件介绍)中的[dump结果文件介绍](./pytorch_data_dump_instruct.md#dump结果文件介绍)。

#### shape/dtype不匹配处理

当源dump的tensor与目标模型实际输入的shape或dtype不一致时：

- **加载阶段**：打印warning日志（包含模块名、参数位置、源shape/dtype和当前shape/dtype），仍执行替换，不跳过。
- **forward阶段**：PyTorch算子可能报`RuntimeError`（shape不匹配通常导致算子报错；dtype不匹配可能隐式转换或报错）。

warning日志示例：

```text
[load] tensor mismatch for Module.blocks.0.attn.A.forward.0.input.0: source shape=[4, 32, 128] dtype=torch.float32 vs current shape=[8, 16, 128] dtype=torch.float16, override may cause forward error
```

**注意**：如果模型A和模型B的`forward`参数顺序不同（如A是`forward(x, mask)`，B是`forward(mask, x)`），工具会按位置index加载——A的x加载到B的mask位置。这种情况shape可能碰巧能对上但语义错误，工具无法自动检测，需用户确保两个模型的forward签名一致。

#### 单卡场景

单卡（非分布式）场景下，dump产出在`proc{pid}`目录而非`rank{N}`目录。load时工具会自动在源dump的`step{N}/`目录下查找`proc*`目录，无需用户指定pid。

#### load与dump level的关系

load功能与dump的`level`配置解耦：

- load读取的是模块级（L0）tensor数据，要求源dump使用`level=L0`或`mix`采集。
- load启用时会强制挂载模块hook，即使config.json中`level=L1`也能正常覆盖。
- `dump_after_load=true`时，dump的level可以是任意值（L0/L1/mix等），dump数据正常产出。

### 配置格式

`load`配置段是config.json的顶层字段，与`task`、`dump_path`、`level`等同级。config.json的其他参数说明请参见[config.json介绍](./config_json_introduct.md)。

```json
{
    "task": "tensor",
    "dump_path": "/home/data_dump_b",
    "level": "L0",
    "load": {
        "path": "/home/data_dump_a",
        "modules": ["Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"],
        "step": [],
        "rank": [],
        "dump_after_load": false
    }
}
```

### 参数说明

#### load参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| path | 必选 | 源dump目录路径，str类型，需包含`step{N}/rank{M}/dump_tensor_data/`目录结构。 |
| modules | 必选 | 要覆盖的模块条目名列表，list[str]类型。条目名与dump产出的`.pt`文件名前缀一致，格式为`Module.{path}.{ClassName}.forward.{N}`。详细介绍请参见[modules参数配置说明](#modules参数配置说明)。 |
| step | 可选 | 指定哪些step进行load，list类型，格式与dump的`step`参数一致（如`[0]`、`["0-2"]`）。默认为空，表示每个step都进行load。加载时从源dump的对应step自动对齐。详细介绍请参见[step/rank参数配置说明](#steprank参数配置说明)。 |
| rank | 可选 | 指定哪些rank进行load，list类型，格式与dump的`rank`参数一致。默认为空，表示每个rank都进行load。加载时从源dump的对应rank自动对齐。详细介绍请参见[step/rank参数配置说明](#steprank参数配置说明)。 |
| dump_after_load | 可选 | 是否在覆盖后继续执行dump采集，bool类型。`false`表示只覆盖不dump，`true`表示覆盖后继续dump，默认为`false`。 |

#### modules参数配置说明

`load.modules`中的每个条目是一个**完整的dump条目名**，与dump产出的`.pt`文件名前缀完全一致。格式为：

```text
Module.{dotted_path}.{ClassName}.forward.{N}
```

其中`forward.{N}`的`N`是模块前向调用的序号（从0开始），**不可省略**——在梯度累积等场景下同一模块会被多次调用，`forward.0`、`forward.1`对应不同的调用，必须明确指定。

**如何获取条目名**：在源dump的`dump_tensor_data/`目录下查看`.pt`文件名，取`.input.{i}.pt`之前的部分即可。例如：

```text
文件名: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.input.0.pt
条目名: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0
```

#### step/rank参数配置说明

`load.step`和`load.rank`为list类型，格式与dump的`step`/`rank`参数一致，支持整数、单个数字字符串和范围字符串。用于控制哪些step/rank进行load，加载时始终从源dump的对应step/rank自动对齐。

- 空列表`[]`（默认）：每个step/rank都进行load，从源dump的对应step/rank加载。
- 非空列表如`[0]`：仅step/rank 0进行load，其余step/rank不进行load（工具无影响）。
- 范围字符串如`["0-2"]`：仅step/rank 0、1、2进行load，各自从源dump的对应step/rank加载。

### 使用示例

#### 场景一：覆盖后dump（定位本模块精度问题）

模型A和模型B结构相同，比对发现`blocks.0.attn`输出有差异，需确认是本模块问题还是前序误差。

1. 先对模型A采集L0 tensor dump（`dump_source/`）。
2. 配置`load`段指向`dump_source`，`dump_after_load`设为`true`。config.json示例如下：

    ```json
    {
        "task": "tensor",
        "dump_path": "/home/dump_override",
        "level": "L0",
        "load": {
            "path": "/home/dump_source",
            "modules": ["Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"],
            "dump_after_load": true
        }
    }
    ```

3. 对模型B运行采集（`dump_override/`）。
4. 比对`dump_override`中该模块的input，应与`dump_source`一致。
5. 比对`dump_override`中该模块的output，若仍有差异则本模块有问题。

#### 场景二：只覆盖不dump（快速验证loss）

用户只想确认覆盖某模块输入后loss是否恢复正常，不需要dump数据。

```json
{
    "task": "tensor",
    "level": "L1",
    "load": {
        "path": "/home/dump_source",
        "modules": ["Module.blocks.0.attn.MultiHeadSelfAttention.forward.0"],
        "dump_after_load": false
    }
}
```

`dump_after_load`为`false`时，`dump_path`可不配置（工具自动使用`load.path`作为fallback）。训练过程中不产生dump数据，用户观察loss即可判断覆盖效果。

#### 场景三：覆盖多个模块

需同时隔离多个模块的输入时，在`modules`中列出多个条目名：

```json
{
    "load": {
        "path": "/home/dump_source",
        "modules": [
            "Module.blocks.0.attn.MultiHeadSelfAttention.forward.0",
            "Module.blocks.1.mlp.MLP.forward.0"
        ],
        "dump_after_load": true
    }
}
```

`modules`中的填写顺序与加载顺序无关。工具内部将`modules`转为set进行查找，模块的覆盖时机由模型前向执行顺序决定——PyTorch逐层调用`forward`时，每个模块触发`forward_pre_hook`，hook中用`full_forward_name`在set中匹配，匹配到则覆盖。因此覆盖顺序取决于模型的前向计算图，与`modules`列表中的排列顺序无关。

## 附录

### 异常处理

| 异常场景 | 处理方式 |
|----------|----------|
| `load.path`不存在或非目录 | 启动报错（MsprobeException）。 |
| `load.modules`为空列表 | 启动报错（避免用户误认为全覆盖）。 |
| `load.modules`非list类型 | 启动报错。 |
| 已配置`load.modules`，但未指定`load.path` | 启动报错。 |
| 模块条目名在模型中不存在 | 不报错，该模块覆盖未生效，训练正常继续。 |
| 源`.pt`文件缺失 | 打印warning日志，该参数保持原值，训练程序继续运行。 |
| shape/dtype不匹配 | 打印warning日志，仍执行替换，forward可能报错。 |

### 补充说明

- `load`配置段不存在时，工具行为与未开发此功能前完全一致，不影响现有dump流程。
- `load.dump_after_load=false`时，`dump_path`可不配置，工具自动使用`load.path`作为fallback。
- tensor加载有缓存机制，同一step内同模块同位置只加载一次IO，每个step开始时清空缓存。
- 加载的tensor会自动映射到当前参数所在设备（如`npu:0`、`npu:1`），无需用户处理device。
