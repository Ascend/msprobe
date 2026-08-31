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
- load覆盖会替换模块的输入tensor，被覆盖的模块及其直接上游模块的backward数据（反向输入输出、参数梯度）在dump采集时可能缺失。这是因为覆盖后的tensor来自`torch.load`加载，msProbe的BackwardHook在forward时挂载到原始tensor上，替换后的tensor不在hook的追踪范围内。此限制仅影响dump数据采集，不影响模型实际的反向计算结果。

## 快速入门

以下通过一个简单的示例，展示如何使用load功能覆盖模块输入，并通过精度比对验证覆盖效果。

本示例使用同一个模型和同一份训练脚本`train.py`，分两次执行：

1. 第一次执行：正常采集精度数据，作为覆盖的数据源。
2. 第二次执行：使用不同的输入数据运行，同时启用load功能将第一次采集的模块输入覆盖到本次运行的对应模块。

两次运行的输入数据不同，是为了验证load覆盖确实生效——如果输入相同，覆盖与不覆盖的结果完全一致，无法判断覆盖是否真正起作用。

1. 编写训练脚本`train.py`，通过命令行参数指定config.json路径和输入偏移量：

    ```diff
     import sys
     import torch
     import torch.nn as nn

    +# 导入工具的数据采集接口
    +from msprobe.pytorch import PrecisionDebugger, seed_all

    +# 在模型训练开始前固定随机性
    +seed_all()
    +# 在模型训练开始前实例化PrecisionDebugger
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

    +    debugger.start(model=model) # 开启数据dump

         x = torch.randn(10, 8)
         # 第二次运行时通过偏移量改变输入，使两次运行的数据不同
         offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
         if offset:
             x = x + offset
         out = model(x)
         loss = out.sum()
         loss.backward()

    +    debugger.stop() # 关闭数据dump，可继续开启数据dump，采集数据会记录在同一个step中
    +    debugger.step() # 结束数据dump，若继续开启数据dump，采集数据将记录在下一个step中
    ```

2. 第一次运行：正常采集精度数据，作为覆盖的数据源。创建`config_a.json`：

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

    执行命令（偏移量为0）：

    ```shell
    python train.py config_a.json 0
    ```

    采集完成后，在`dump_path`指定的目录下会生成如下目录结构：

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

    其中`dump_tensor_data/`目录下的`.pt`文件即为后续load功能要加载的源数据。记录下需要覆盖的模块条目名，例如`Module.relu.ReLU.forward.0`（取`.input.{i}.pt`之前的部分）。

3. 第二次运行：创建`config_b.json`，新增`load`配置段，指定覆盖的数据源路径和目标模块：

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

    - `load.path`指向步骤2采集的源dump目录（`/home/data_dump_a`），即`step{N}/`的上一层目录。
    - `load.modules`指定要覆盖的模块条目名（此处覆盖`relu`模块，不覆盖`linear`模块），如何获取条目名请参见[modules参数配置说明](#modules参数配置说明)。
    - `dump_after_load`设为`true`，表示覆盖后继续执行dump采集，覆盖后的数据保存在`dump_path`指定的目录（`/home/data_dump_b`）中。

4. 执行第二次运行（偏移量设为1，使输入与第一次不同），工具会自动加载第一次采集的`Module.relu.ReLU`模块输入数据，覆盖本次运行的对应模块输入：

    ```shell
    python train.py config_b.json 1
    ```

5. 查看结果。覆盖后的dump数据保存在`/home/data_dump_b`目录中，可使用msProbe工具的比对功能验证覆盖效果。将第二次运行的dump数据与第一次的dump数据进行比对，若`Module.relu.ReLU.forward.0`的input和output的Cosine相似度为1.0，说明覆盖生效：

    ```shell
    msprobe compare -tp /home/data_dump_b/step0 -gp /home/data_dump_a/step0 -o ./compare_result
    ```

    比对结果保存在`./compare_result`目录下的CSV文件中，关键结果如下：

    | NPU Name | Cosine |
    |----------|--------|
    | Module.linear.Linear.forward.0.input.0 | 0.69 |
    | Module.linear.Linear.forward.0.output.0 | 0.60 |
    | Module.relu.ReLU.forward.0.input.0 | 1.0 |
    | Module.relu.ReLU.forward.0.output.0 | 1.0 |

    结论：

    - `linear`模块未被覆盖，其input和output的Cosine小于1.0，说明两次运行的输入确实不同（偏移生效）。
    - `relu`模块被覆盖，其input和output的Cosine为1.0（完全一致），说明第二次运行的`relu`模块输入被成功覆盖为第一次采集的数据。

    更多比对操作请参见[PyTorch场景精度比对](../accuracy_compare/pytorch_accuracy_compare_instruct.md)。

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

### 参数说明

#### load参数说明

| 参数 | 可选/必选 | 说明 |
|------|----------|------|
| path | 必选 | 源dump目录路径，str类型，即`step{N}/`的上一层目录。例如源dump目录为`/home/data_dump_a`，则`load.path`配置为`/home/data_dump_a`，工具会自动在`/home/data_dump_a/step{N}/rank{M}/dump_tensor_data/`下查找对应的`.pt`文件。 |
| modules | 必选 | 要覆盖的模块条目名列表，list[str]类型。条目名详细介绍请参见[modules参数配置说明](#modules参数配置说明)。 |
| step | 可选 | 指定哪些step进行load，list类型，格式与dump的`step`参数一致（如`[0]`、`["0-2"]`）。默认为空，表示每个step都进行load。加载时从源dump的对应step自动对齐。详细介绍请参见[step/rank参数配置说明](#steprank参数配置说明)。 |
| rank | 可选 | 指定哪些rank进行load，list类型，格式与dump的`rank`参数一致。默认为空，表示每个rank都进行load。加载时从源dump的对应rank自动对齐。详细介绍请参见[step/rank参数配置说明](#steprank参数配置说明)。 |
| dump_after_load | 可选 | 是否在覆盖后继续执行dump采集，bool类型。`false`表示只覆盖不dump，`true`表示覆盖后继续dump，默认为`false`。 |

#### modules参数配置说明

`load.modules`中的每个条目取dump产出的`.pt`文件名中`.input.{i}.pt`或`.kwargs.{key}.pt`之前的部分，不支持.backward.{N}，不允许重复。格式为：

```text
Module.{dotted_path}.{ClassName}.forward.{N}
```

其中`forward.{N}`的`N`是模块前向调用的序号（从0开始），**不可省略**——在梯度累积等场景下同一模块会被多次调用，`forward.0`、`forward.1`对应不同的调用，必须明确指定。

条目名从源dump的dump_tensor_data/目录下查看.pt文件名获取。例如：

```text
文件名: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.input.0.pt
条目名: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0
模块名: Module.blocks.0.attn.MultiHeadSelfAttention

文件名: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0.kwargs.mask.pt
条目名: Module.blocks.0.attn.MultiHeadSelfAttention.forward.0
模块名: Module.blocks.0.attn.MultiHeadSelfAttention
```

**条目格式要求：**

- 必须以`Module.`开头。
- 必须以`.forward.{N}`结尾，其中`N`为非负整数，不支持`.backward.{N}`。
- 不允许重复条目。
- 条目中的模块路径（去掉`Module.`前缀和`.forward.{N}`后缀后的部分）必须在当前模型的`named_modules()`中存在，否则启动时会打印warning日志提示。

**校验与反馈：**

工具在启动时会自动校验`load.modules`的格式和模块路径，输出校验结果汇总日志，例如：

```text
[load] module validation: 2/5 modules valid in model
```

对于模块路径不存在或类名不匹配的条目，会逐条打印warning。训练过程中，如果某些条目因`forward.{N}`的调用序号不匹配而从未命中，工具会在`debugger.stop()`时打印warning提示未命中的条目。

#### step/rank参数配置说明

`load.step`和`load.rank`为list类型，格式与dump的`step`/`rank`参数一致，支持整数、单个数字字符串和范围字符串。用于控制哪些step/rank进行load，加载时始终从源dump的对应step/rank自动对齐。

- 空列表`[]`（默认）：每个step/rank都进行load，从源dump的对应step/rank加载。
- 非空列表如`[0]`：仅step/rank 0进行load，其余step/rank不进行load（工具无影响）。
- 范围字符串如`["0-2"]`：仅step/rank 0、1、2进行load，各自从源dump的对应step/rank加载。

### 使用示例

快速入门已演示了覆盖后dump的完整流程（对应场景一），以下为其他场景的配置说明。

#### 场景一：覆盖后dump（定位本模块精度问题）

快速入门中的示例即此场景，详见[快速入门](#快速入门)。

核心配置：`dump_after_load`设为`true`，覆盖后继续dump，通过精度比对验证覆盖效果。

#### 场景二：只覆盖不dump（快速验证loss）

用户只想确认覆盖某模块输入后loss是否恢复正常，不需要dump数据。

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

`load.path`指向源dump目录`/home/data_dump_a`。`dump_after_load`为`false`时，不产生dump数据，用户观察loss即可判断覆盖效果。

#### 场景三：覆盖多个模块

需同时隔离多个模块的输入时，在`modules`中列出多个条目名：

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

配置后，`modules`中列出的所有模块的输入都会被源dump数据覆盖，且填写顺序与加载顺序无关。

## 附录

### 异常处理

| 异常场景 | 处理方式 |
|----------|----------|
| `load.path`不存在或非目录 | 启动报错（MsprobeException）。 |
| `load.modules`为空列表 | 启动报错（避免用户误认为全覆盖）。 |
| `load.modules`非list类型 | 启动报错。 |
| `load.modules`条目格式不合法（不以`Module.`开头、不以`.forward.{N}`结尾、含`.backward.`、重复） | 启动报错。 |
| 已配置`load.modules`，但未指定`load.path` | 启动报错。 |
| 模块名在模型中不存在 | 启动时打印warning日志提示，训练正常继续。 |
| 条目名配置了但运行时从未命中（如`forward.{N}`序号不匹配） | `debugger.stop()`时打印warning日志提示未命中条目，训练正常继续。 |
| 源`.pt`文件缺失 | 打印warning日志，该参数保持原值，训练程序继续运行。 |
| shape/dtype不匹配 | 打印warning日志，仍执行替换，forward可能报错。 |
