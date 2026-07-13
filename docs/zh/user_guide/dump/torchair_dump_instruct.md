# 基于torch图模式（torchair）推理场景

## 简介

torch图模式（torchair）推理场景的精度数据采集是对torchair图编译模型推理过程中中间算子的输入、输出数据进行采集，通过设置编译配置(CompilerConfig)来实现。

## 基本概念

- [torchair](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/torchair/compiler_config.md)：torch图模式的后端，用于将torch模型编译为昇腾AI处理器上的可执行程序，包括计算图转换与优化。
- GE: GE指Graph Engine（图引擎），是昇腾AI计算平台的核心组件。GE作为计算图编译和运行的控制中心，主要提供：
    - 图优化：对神经网络计算图进行优化，提高执行效率。
    - 图编译管理：将不同框架的模型转换为统一的内部表示，为图优化和执行提供基础。
    - 图执行控制：负责优化后图的高效执行。
- FX: PyTorch FX（或称为 TorchFX，PyTorch Functional eXecution）是PyTorch的函数式执行和图转换框架，主要功能包括：
    - 将PyTorch代码转换为可优化的中间表示 (IR) 图，用于图优化和分析。
    - 提供程序捕获和转换功能，支持模型重构和优化。
- 融合：图融合是指将神经网络中的多个算子合并为一个算子，以减少计算量和内存占用，一般按照规则进行融合。
- 采集：是指在模型推理过程中，将模型中间算子的输入、输出数据采集下来，保存为文件。

## 使用前准备

### 环境准备

安装msProbe工具，详情请参见《[msProbe安装指南](../../install_guide/msprobe_install_guide.md#)》。

### 约束

- 仅支持torch图模式（torchair）推理场景。

## GE融合模式dump数据

**功能说明**

- 调用 `set_ge_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加dump配置，配置模型compile，并执行推理。

**接口原型**

```Python
set_ge_dump_config(
        dump_path='',
        dump_mode='all',
        fusion_switch_file=None,
        dump_token=None,
        dump_layer=None,
        compiler_config=None
)
```

**参数说明**

| 参数名             | 可选/必选 | 参数描述                                                     |
| ------------------ | --------- | ------------------------------------------------------------ |
| dump_path          | 可选      | dump数据的存放路径。默认为"./"。                             |
| dump_mode          | 可选      | data dump模式，用于指定dump算子输入还是输出数据。可选值有"input", "output", "all"。默认为"all"，dump输入与输出数据。 |
| fusion_switch_file | 可选      | 是否关闭融合dump功能。默认为None，开启融合。                 |
| dump_token         | 可选      | 指定token进行dump。格式：[1,2,5] 代表dump第1、2、5个token数据。默认为None，dump全量数据。 |
| dump_layer         | 可选      | 指定layer进行dump。格式：["Add", "Conv_1"] 代表dump Add和Conv_1两层数据。默认为None，dump全量数据。 |
| compiler_config    | 可选      | 图编译配置（CompilerConfig对象）。默认为None，返回新创建的图编译配置。 |

**使用示例（已创建CompilerConfig对象）**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有config上增加dump配置
  set_ge_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**使用示例（未创建CompilerConfig对象）**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_ge_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

### dump结果文件介绍

dump数据保存路径为 `{dump_path}/msprobe_ge_dump`。其中 `{dump_path}` 为用户通过 `set_ge_dump_config` 接口的'dump_path' 参数传入的路径，`msprobe_ge_dump`为msProbe工具自动创建的目录。

使用TorchNPU 7.1.0及以上版本时，结果文件目录结构为

```ColdFusion
├── ${dump_path}
│   ├── msprobe_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${time}
|   |   |   |   ├── ${device_id}
|   |   |   |   |   ├── ${model_name}
|   |   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── └── # bin格式数据
```

使用TorchNPU 7.1.0以下版本时，结果文件目录结构为

```ColdFusion
├── ${dump_path}
│   ├── msprobe_ge_dump
│   |   ├── dynamo_optimized_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── dynamo_original_${graph_name}_rank_${rank_id}_pid_${pid}_ts_${time}.txt
│   |   ├── ${time}
|   |   |   ├── ${device_id}
|   |   |   |   ├── ${model_name}
|   |   |   |   |   ├── ${model_id}
|   |   |   |   |   |   ├── ${token_id}
└── └── └── └── └── └── └── └── # bin格式数据
```

## FX模式dump数据

**功能说明**

- 调用 `set_fx_dump_config` 接口，获取配置后的 `config` 实例，或在已有 `config` 实例上增加dump配置，配置模型compile，并执行推理。

**接口原型**

```Python
set_fx_dump_config(dump_path='', op_list=None, compiler_config=None)
```

**参数说明**

| 参数名          | 可选/必选                                        |  参数描述                      |
| --------------- | ---------------------------------------------------- | -------------------------------------- |
| dump_path       | 可选 | dump数据的存放路径。仅在使用TorchNPU 7.0.0及以上版本时生效。默认为"./"。 |
| op_list         | 可选 | 指定算子进行dump。格式：["Add", "Conv_1"] 代表dump Add和Conv_1两层数据。默认为None，dump全量数据。 |
| compiler_config | 可选 | 图编译配置（CompilerConfig对象）。默认为None，返回新创建的图编译配置。                     |

**使用示例（已创建CompilerConfig对象）**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有config上增加dump配置
  set_fx_dump_config(dump_path="dump", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**使用示例（未创建CompilerConfig对象）**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_fx_dump_config(dump_path="dump")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

**使用示例（指定算子进行dump）**

```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_fx_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_fx_dump_config(dump_path="dump", op_list=['Add', 'Conv_1'])
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
```

### dump结果文件介绍

dump数据保存路径为 `{dump_path}/msprobe_fx_dump`。其中 `{dump_path}` 为用户通过 `set_fx_dump_config` 接口的'dump_path' 参数传入的路径，`msprobe_fx_dump`为msProbe工具自动创建的目录。

使用TorchNPU 7.1.0及以上版本时，结果文件目录结构为

```ColdFusion
├── ${dump_path}
│   ├── msprobe_fx_dump
│   |   ├── worldsize${rank_size}_global_rank${rank_id}
│   |   │   ├── ${model_name}
|   |   |   |   ├── ${token_id}
└── └── └── └── └── └── # npy格式数据
```

使用TorchNPU 7.1.0版本时，结果文件目录结构为

```ColdFusion
├── ${dump_path}
│   ├── msprobe_fx_dump
│   |   ├── data_dump
|   |   |   ├── ${token_id+1}
|   |   |   |   ├── gm_${time}_dump
└── └── └── └── └── └── # npy格式数据
```

使用TorchNPU 7.1.0以下版本时，结果文件目录结构为

```ColdFusion
├── . # 当前工作目录
│   ├── data_dump
|   |   ├── ${token_id+1}
|   |   |   ├── gm_${time}_dump
└── └── └── └── └── # npy格式数据
```

## GE模式关闭融合dump数据

在[GE融合模式dump数据](#ge融合模式dump数据)的基础上，通过 `set_ge_dump_config` 接口的 `fusion_switch_file` 参数传入设置关闭算子融合的配置文件。工具使用示例如下：

- 创建设置关闭算子融合的配置文件 `fusion_switch.json`。算子融合规则的详解介绍可参见《PyTorch图模式使用(torchair)》中的“[算子融合规则配置功能](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/fusion_switch_file.md)”章节。

  ```json
  {
    "Switch": {
      "GraphFusion": {
        "ALL": "off"
      },
      "UBFusion": {
        "ALL": "off"
      }
    }
  }
  ```

- 在推理脚本中调用 `set_ge_dump_config` 接口。

**使用示例（已创建CompilerConfig对象）**

  ```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  config = torchair.CompilerConfig()
  # 在已有config上增加dump配置
  set_ge_dump_config(dump_path="dump_fusion_off",
                     fusion_switch_file="fusion_switch.json", compiler_config=config)
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

**使用示例（未创建CompilerConfig对象）**

  ```py
  import torch, torch_npu, torchair
  from msprobe.pytorch import set_ge_dump_config  # 添加导入
  ...
  model = ...
  # 添加获取config
  config = set_ge_dump_config(dump_path="dump_fusion_off", fusion_switch_file="fusion_switch.json")
  ...
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend)
  ...
  ```

- dump结果文件的目录结构与含义可见“[GE融合模式dump结果文件介绍](#dump结果文件介绍)”小节。

完整精度比对案例和使用方法请参考《[基于torch图模式（torchair）整网算子精度比对](../accuracy_compare/torchair_compare_instruct.md#简介)》。
