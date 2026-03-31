# 配置文件介绍

- 当调用**PrecisionDebugger**接口执行dump或其他操作时，需要使用[config.json](../../../python/msprobe/config.json)文件；当未指定config.json时，将使用默认配置。
- msProbe成功安装后，config.json一般位于如下目录：

  ```ColdFusion
  /home/xxx/miniconda3/envs/xxx/lib/python3.xx/site-packages/msprobe/
  ```

## 参数介绍

### 通用配置

#### 通用配置参数说明

| 参数                | 可选/必选 | 解释                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|-------------------| -------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| task              | 可选     | dump的任务类型，str类型。可选参数：<br/>&#8226; "statistics"：仅采集统计信息。<br/>&#8226; "tensor"：采集统计信息和完全复刻整网的真实数据。<br/>&#8226; "acc_check"：精度预检，仅PyTorch场景支持，采集数据时勿选。<br/>&#8226; "overflow_check"：溢出检测。<br/>&#8226; "structure"：仅采集模型结构以及调用栈信息，不采集具体数据。<br/>默认值为"statistics"。<br/>根据task参数取值的不同，可以配置不同场景参数，详细介绍请参见：<br/>&#8226; [task配置为statistics](#task配置为statistics)<br/>&#8226; [task配置为tensor](#task配置为tensor)<br/>&#8226; [task配置为acc_check](#task配置为acc_check)<br/>&#8226; [task配置为overflow_check](#task配置为overflow_check)<br/>&#8226; [task配置为structure](#task配置为structure)<br/>&#8226; [task配置为exception_dump](#task配置为exception_dump)<br/>配置示例："task": "tensor"。                                            |
| dump_path         | 必选     | 设置dump数据目录路径，str类型。<br/>配置示例："dump_path": "./dump_path"。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| rank              | 可选     | 指定对某张卡上的数据进行采集，list[Union[int, str]]类型，默认未配置（表示采集所有卡的数据），应配置元素为≥ 0的整数或类似"4-6"的字符串，且须配置实际可用的Rank ID。<br/>&#8226; PyTorch场景：Rank ID从0开始计数，最大取值为所有节点可用卡总数-1，若所配置的值大于实际训练所运行的卡的Rank ID，则dump数据为空，比如当前环境Rank ID为0到7，实际训练运行0到3卡，此时若配置Rank ID为4或不存在的10等其他值，dump数据为空。<br/>&#8226; MindSpore场景：所有节点的Rank ID均从0开始计数，最大取值为每个节点可用卡总数-1，config.json配置一次rank参数对所有节点同时生效。静态图L0级别dump暂不支持指定rank。<br/>单卡训练时，rank必须为[]，即空列表，不能指定rank。<br/>配置示例："rank": [1, "4-6"]。                                                                                                                                                                                                                                                     |
| step              | 可选     | 指定采集某个step的数据，list[Union[int, str]]类型。默认未配置，表示采集所有step数据。采集特定step时，须指定为训练脚本中存在的step，可逐个配置，也可以指定范围。<br/>配置示例："step": [0, 1 , 2, "4-6"]。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| level             | 可选 | dump级别，str类型，根据不同级别采集不同数据。可选参数：<br/>&#8226; "L0"：dump模块级精度数据，使用背景详细介绍请参见[模块级精度数据dump说明](#模块级精度数据dump说明)。<br/>&#8226; "L1"：dump API级精度数据，默认值，仅PyTorch、MSAdapter以及MindSpore动态图场景支持。<br/>&#8226; "L2"：dump kernel级精度数据，PyTorch场景详细介绍见[PyTorch场景kernel精度数据采集](./pytorch_kernel_dump_instruct.md)；MindSpore动态图场景详细介绍请参见[MindSpore动态图场景kernel精度数据采集](./mindspore_kernel_dump_instruct.md)；MindSpore静态图场景详细介绍请参见《MindSpore场景精度数据采集》中的 ["静态图场景精度数据采集功能介绍"](./mindspore_data_dump_instruct.md#静态图场景精度数据采集功能介绍)小节。<br/>&#8226; "mix"：dump module模块级和API级精度数据，即"L0"+"L1"，仅PyTorch、MSAdapter以及MindSpore动态图场景支持。<br/>&#8226; "debug"：单点保存功能，详细介绍请参见[单点保存工具](./debugger_save_instruct.md)。<br/>配置示例："level": "L1"。 |
| async_dump        | 可选     | 异步dump开关，bool类型，支持task为tensor或statistics模式，level支持L0、L1、mix、debug模式。可选参数true（开启）或false（关闭），默认为false。<br/>配置为true后开启异步dump，即采集的精度数据会在当前step训练结束后统一落盘，训练过程中工具不触发同步操作。<br/>由于使用该模式有**显存溢出**的风险，当task配置为tensor时，即真实数据的异步dump模式，必须配置[list](#task配置为tensor)参数，指定需要dump的tensor 。<br/>该模式下，summary_mode不支持md5值，也不支持复数类型tensor的统计量计算。                                                                                                                                                                                                                                                                                                                                                                         |
| dump_enable       | 可选     | dump功能开关，用于控制PrecisionDebugger dump的启动和停止，bool类型。可选参数：<br/>&#8226; true：允许执行dump采集。<br/>&#8226; false：关闭dump采集。<br/>该参数支持动态启停，即在dump任务运行过程中，可以随时启动和停止dump进程。<br/>默认未配置，表示不对dump数据进行控制，按照静态配置dump数据。<br/>更多配置说明请参见：[dump_enable参数配置说明](#dump_enable参数配置说明)。<br/>配置示例："dump_enable": true。                                                                                                                                                                                                                                                                                                                                                                                                             |
| extra_info        | 可选     | 控制是否采集并输出额外信息文件（`stack.json`和`construct.json`），bool类型。可选参数：<br/>&#8226; true：采集并输出`stack.json`和`construct.json`。<br/>&#8226; false：不采集额外信息，且不生成`stack.json`和`construct.json`。<br/>默认值为true。<br/>配置示例："extra_info": false。                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| precision         | 可选     | 控制统计值计算所用精度，str类型，可选值["high", "low"]，默认值为"low"。选择"high"时，统计量使用float32进行计算，会增加device内存占用，精度更高，但在处理较大数值时可能会导致**显存溢出**；为"low"时使用与原始数据相同的类型进行计算，device内存占用较少。<br/>支持Pytorch、MindSpore动态图和MindSpore静态图O0/O1场景。<br/>支持task配置为statistics或tensor，level配置为L0，L1，mix，debug。                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| risk_level        | 可选     | API风险级别过滤，str类型，仅PyTorch场景且level配置为L1以及mix时生效。可选参数：<br/>&#8226; "ALL"：dump所有API的数据。<br/>&#8226; "CORE"：仅dump核心（高风险，易出现精度问题）API的数据，包括融合计算、通信、精密计算等。<br/>&#8226; "FOCUS"：dump核心API和关注API的数据，排除低风险API（如reshape、transpose、permute、to、view等形状变换类API），**默认值**。<br/>配置示例："risk_level": "CORE"。                                                                                                                                                                                                                                                                                                                                                                                                              |

#### 模块级精度数据dump说明

大模型场景下，通常不是简单的利用自动迁移能力实现从GPU到NPU的训练脚本迁移，而是会对NPU网络进行一系列针对性的适配，因此，常常会造成迁移后的NPU模型存在部分子结构不能与GPU原始模型完全对应。模型结构不一致导致API调用类型及数量不一致，若直接按照API粒度进行精度数据dump和比对，则无法完全比对所有的API。

本小节介绍的功能是对模型中的大粒度模块进行数据dump，使其比对时，对于无法以API粒度比对的模块可以直接以模块粒度进行比对。

模块指的是继承nn.Module类（PyTorch与MSAdapter场景）或nn.Cell类（MindSpore场景）的子类，通常情况下这类模块就是一个小模型，可以被视为一个整体，dump数据时以模块为粒度进行dump。

特别地，在PyTorch场景中，为了规避BackwardHook函数的输出不能进行原地操作的框架限制，工具使用了`torch._C._autograd._set_creation_meta`接口对BackwardHook函数的输出张量进行属性重置，这可能会造成dump数据中缺少原地操作模块nn.ReLU(inplace=True)及其上一个模块的反向数据。

### task配置为statistics

**配置样例**

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

支持场景：

 - PyTorch场景
 - MindSpore静态图场景
 - MindSpore动态图场景

**参数说明**

| 参数         | 可选/必选 | 解释                                                         |
| ------------ | --------- | ------------------------------------------------------------ |
| scope        | 可选      | PyTorch、MSAdapter以及MindSpore动态图场景dump范围，list[str]类型，默认未配置（list也未配置时表示dump所有API的数据）。详细配置方法请参见[scope参数配置说明](#scope参数配置说明)。 |
| list         | 可选      | 自定义采集的算子列表，list[str]类型，默认未配置（scope也未配置时表示dump所有API的数据）。详细配置方法请参见[list参数配置说明](#list参数配置说明)。 |
| tensor_list  | 可选      | 自定义采集真实数据的算子列表，list[str]类型，默认未配置。详细配置方法请参见[tensor_list参数配置说明](#tensor_list参数配置说明)。<br/>PyTorch、MSAdapter以及MindSpore动态图场景目前只支持level配置为L0、L1和mix级别。<br/>MindSpore静态图场景不支持。 |
| device       | 可选      | 控制统计值计算所用的设备，可选值["device", "host"]，默认值为"host"。使用device计算会比host有性能加速，只支持min/max/avg/l2norm统计量。<br/>仅MindSpore静态图O0/O1场景支持。 |
| data_mode    | 可选      | dump数据过滤，list[str]类型。详细配置方法请参见[data_mode参数配置说明](#data_mode参数配置说明)。 |
| summary_mode | 可选      | 控制dump文件输出的模式，支持PyTorch、MSAdapter、MindSpore动态图以及MindSpore静态图L2级别jit_level=O2场景和L0级别jit_level=O0/O1场景。详细配置方法请参见[summary_mode参数配置说明](#summary_mode参数配置说明)。 |

### task配置为tensor

**配置样例**

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

支持场景：

 - PyTorch场景
 - MindSpore静态图场景
 - MindSpore动态图场景

**参数说明**

| 参数         | 可选/必选 | 解释                                                                                                                                                                                                                                                                      |
| -------------- | -------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| scope          | 可选     | PyTorch、MSAdapter以及MindSpore动态图场景dump范围，list[str]类型，默认未配置（list也未配置时表示dump所有API的数据）。详细配置方法请参见[scope参数配置说明](#scope参数配置说明)。                                                                                    |
| list           | 可选     | 自定义采集的算子列表，list[str]类型，默认未配置（scope也未配置时表示dump所有API的数据）。详细配置方法请参见[list参数配置说明](#list参数配置说明)。                                                                                                                  |
| data_mode      | 可选     | dump数据过滤，list[str]类型。详细配置方法请参见[data_mode参数配置说明](#data_mode参数配置说明)。                                                                                                                                     |
| file_format    | 可选     | tensor数据的保存格式，str类型，仅支持MindSpore静态图场景的L2级别配置该字段，其他场景不生效。可选参数：<br/>&#8226; "bin"：dump的tensor文件为二进制格式。<br/>&#8226; "npy"：dump的tensor文件后缀为.npy。<br/>默认值为"npy"。                                                                                     |
| summary_mode  | 可选 | 控制dump文件输出的模式，支持PyTorch、MSAdapter、MindSpore动态图。可选参数：<br/>&#8226; md5：dump输出包含CRC-32值以及API统计信息的dump.json文件，用于验证数据的完整性。<br/>&#8226; statistics：dump仅输出包含API统计信息的dump.json文件。<br/>&#8226; xor：仅PyTorch场景支持。dump输出仅包含XOR二进制校验值（字段名为md5），不输出max、min、mean、L2norm统计信息。<br/>默认值为statistics。                             |
| bench_path      | 可选     | 自动控制在PyTorch确定性问题定位时进行md5实时差异分析，即dump存在差异的md5数据，str类型，默认未配置本参数。<br/>需要在bench_path参数传入提前预置的md5数据路径（即在上一次dump操作时，summary_mode参数配置为md5），并且本次dump时同样配置summary_mode为md5。<br/>配置本参数后，dump会判断本次任务中每个tensor与预置的md5数据的差异，识别到差异节点后，进行真实数据dump。<br/>配置示例："bench_path": "./bench_dump_path"。 |
| diff_nums      | 可选     | 最大差异次数，int类型，默认为1，仅PyTorch md5实时差异分析场景支持（即配置bench_path）。 表示第N次差异出现后，不再进行差异分析。过程中检测到差异对应的输入输出数据均dump。<br/>配置为-1时，表示持续检测溢出直到训练结束。<br/>配置示例："diff_nums": 3。 |

### task配置为acc_check

**配置样例**

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

支持场景：

 - PyTorch场景

**参数说明**

| 参数        | 可选/必选 | 解释                   |
| --------------- | ------------ | ------------------------ |
| white_list      | 可选     | API dump白名单，仅对指定的API进行dump。默认未配置白名单，即dump全量API数据。<br/>配置示例："white_list": ["conv1d", "conv2d"]。 |
| black_list      | 可选     | API dump黑名单，被指定的API不进行 dump。默认未配置黑名单，即dump全量API数据。<br/>配置示例："black_list": ["conv1d", "conv2d"]。 |
| error_data_path | 可选     | 配置保存精度未达标的API输入输出数据路径，默认为当前路径。<br/>配置示例："error_data_path": "./"。 |

white_list和black_list同时配置时，二者配置的API名单若无交集，则白名单生效，若API名单存在交集，则白名单排除的部分以及交集的API不进行dump。

### task配置为overflow_check

**配置样例**

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

支持场景：

 - MindSpore静态图场景

**参数说明**

MindSpore静态图场景下，"level"须为"L2"，且模型编译优化等级（jit_level）须为"O2"。

| 参数        | 可选/必选 | 解释                 |
| ------------- | -------- | ---------------------- |
| check_mode    | 可选     | 溢出类型，str类型，仅MindSpore v2.3.0以下版本的静态图场景支持，可选参数：<br/>&#8226; "aicore"：开启AI Core的溢出检测。<br/>&#8226; "atomic"：开启Atomic的溢出检测。<br/>&#8226; "all"：开启算子的溢出检测。<br/>默认值为all。<br/>配置示例："check_mode": "all"。 |

### task配置为structure

structure模式仅采集模型结构，无其他特殊配置。

**配置样例**

```json
{
    "task": "structure",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "mix"
}
```

支持场景：

 - PyTorch场景
 - MindSpore动态图场景

### task配置为exception_dump

MindSpore动态图场景下，"level"须为"L2"; MindSpore静态图场景下，"level"须为"L2"，且模型编译优化等级（jit_level）须为"O0"或"O1"。

在运行过程中会在指定目录下生成kernel_graph_exception_dump.json的中间文件，该文件包含异常dump的相关设置。

除中间文件外的其他dump结果文件请参见MindSpore官方文档中的[Ascend下O0/O1模式Dump数据对象目录和数据文件介绍](https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/debug/dump.html#%E6%95%B0%E6%8D%AE%E5%AF%B9%E8%B1%A1%E7%9B%AE%E5%BD%95%E5%92%8C%E6%95%B0%E6%8D%AE%E6%96%87%E4%BB%B6%E4%BB%8B%E7%BB%8D)。

**配置样例**：

```json
{
    "task": "exception_dump",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [],
    "level": "L2"
}
```

支持场景：

 - MindSpore动态图场景
 - MindSpore静态图场景

## 附录

### dump_enable参数配置说明

- `dump_enable`用于控制`PrecisionDebugger`的dump动态启停能力：取值为`true`时执行dump采集，取值为`false`时关闭dump采集。建议仅在需要动态控制采集时配置该参数，初始值设为`false`。
- PyTorch场景下，当`PrecisionDebugger`初始化时配置了该字段，工具会在执行过程中自动读取`config_path`并刷新配置。
- 推荐流程：常规训练/推理阶段保持关闭；需要定位问题时改为开启采集；完成定位后再关闭，以减少对业务流程的干扰。
- `vllm`场景下，如果有`level`切换的需要，建议先设置`level`的初始值为`L0`，这样能保证后续的`level`可以任意切换；如果`level`的初始值不是`L0`可能会导致切换`level`失败。

**配置样例**：

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

> 说明：`dump_enable`仅在需要动态开关dump时配置。运行中可将`dump_enable`从`false`改为`true`（或反向修改）实现动态开关，json中其他字段修改也能生效。

支持场景：

 - PyTorch场景

### list参数配置说明

- PyTorch、MSAdapter以及MindSpore动态图场景配置具体的API全称，dump该API数据。在PyTorch场景，如果level配置成L2，list参数为必填项。

  配置示例："list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.backward"]。

- PyTorch和MindSpore动态图场景在level为mix级别时可以配置模块名称，dump该模块展开数据（dump该模块从执行开始到执行结束期间的所有数据）。

  配置示例："list": ["Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]或"list": ["Cell.network_with_loss.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]。

- PyTorch、MSAdapter以及MindSpore动态图场景指定某一类API，dump某一类的API级别输入输出数据。

  配置示例："list": ["relu"]。

  PyTorch、MSAdapter以及MindSpore动态图场景在level为mix级别时，会dump名称中包含list中配置的字符串的API数据，还会将名称中包含list中配置的字符串的模块进行展开dump（dump该模块从执行开始到执行结束期间的所有数据）。

- MindSpore静态图场景配置kernel_name，可以是算子的名称列表，也可以指定算子类型（jit_level=O2时不支持），还可以配置算子名称的正则表达式（当字符串符合“name-regex(xxx)”格式时），后台则会将其作为正则表达式。

  配置示例：list: ["name-regex(Default/.+)"]。

  可匹配算子名称以“Default/”开头的所有算子。

### scope参数配置说明

该参数可以在[ ]内配置两个模块名或API名，要求列表长度必须为2，需要配置按照工具命名格式的完整模块名或API名称，用于锁定区间，dump该范围内的数据。

配置示例："scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]或"scope": ["Cell.conv1.Conv2d.forward.0", "Cell.fc2.Dense.backward.0"]或"scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]。<br/>与level参数取值相关，level为L0级别时，可配置模块名；level为L1级别时，可配置API名，level为mix级别时，可配置为模块名或API名。

### tensor_list参数配置说明

PyTorch、MSAdapter以及MindSpore动态图场景指定某一类API或模块，即会dump这一类API或模块输入输出的统计量信息和完整的tensor数据。<br/>配置示例："tensor_list": ["relu"]。<br/>

### data_mode参数配置说明

- PyTorch、MSAdapter以及MindSpore动态图场景：支持"all"、"forward"、"backward"、"input"和"output"，除"all"外，其余参数可以自由组合。默认为["all"]，即保存所有dump的数据。

  配置示例："data_mode": ["backward"]（仅保存反向数据）或"data_mode": ["forward", "input"]（仅保存前向的输入数据）。

- MindSpore静态图场景：L0级别dump仅支持"all"、"forward"和"backward"参数；L2级别dump仅支持"all"、"input"和"output"参数。且各参数只能单独配置，不支持自由组合。

  配置示例："data_mode": ["all"]。

### summary_mode参数配置说明

- PyTorch、MSAdapter以及MindSpore动态图场景

  str类型。

  可选参数为：

  - md5：dump输出包含CRC-32值以及API统计信息的dump.json文件，用于验证数据的完整性。
  - statistics：dump仅输出包含API统计信息的dump.json文件。默认值为statistics。
  - xor：仅PyTorch场景支持。dump输出仅包含XOR校验值（字段名为md5），不输出max、min、mean、L2norm统计信息。

  配置示例："summary_mode": "md5"。

- MindSpore静态图场景

  str或list[str]类型。
  
  - L2级别jit_level=O2：支持上述"md5"和"statistics"参数的同时额外支持配置统计项列表，可选统计项为max、min、mean、l2norm，可从中任意选取组合搭配。其中mean、l2norm的结果为float数据格式。
  - L2级别jit_level=O0/O1：支持上述"md5"和"statistics"参数的同时额外支持配置统计项列表，可选统计项为max、min、mean、l2norm、count、negative zero count、zero count、positive zero count、nan count、negative inf count、positive inf count、hash、md5，可从中任意选取组合搭配。其中，hash统计项在MindSpore 2.7.0及以前版本计算MD5值，在以后版本计算SHA1值。
  - L0级别jit_level=O0/O1：仅支持上述"statistics"参数和max、min、mean、l2norm中任意组合搭配的统计项列表。
  
  配置示例："summary_mode": ["max", "min"]。

> [!NOTE] 说明
>
> PyTorch、MSAdapter以及MindSpore动态图场景，"summary_mode"配置为"md5"时，所使用的校验算法为CRC-32算法；MindSpore静态图场景，"summary_mode"配置为"md5"时，所使用的校验算法为MD5算法。
