# aclgraph_dump 使用指南

## 简介

在 **PyTorch ACLGraph** 图模式下执行精度对齐时，整体策略如下：**先整网筛查，后单点深挖**。通过**整网采集**快速收敛异常范围，针对疑似问题算子进行**单点采集**并保存其 Tensor 数据，以便于细粒度排查。`aclgraph_dump` 提供如下采集能力：

- 单点采集：`acl_save`
- 整网采集：`AclGraphDumper`

## 使用前准备

**环境准备**

1. 安装并正确配置 TorchNPU。
2. 安装 msProbe 工具，详见《[msProbe 安装指南](../../install_guide/msprobe_install_guide.md)》。

>[!NOTE]
>
>源码编译安装时需包含 `aclgraph_dump` 模块，通过如下命令安装：
>
>`python3 build.py -e include-mod=aclgraph_dump -e no-check=true`

**约束**

- 仅支持 PyTorch 框架。
- 构建 `aclgraph_dump` 需要 TorchNPU 参与编译；若未包含该模块，将无法正常使用该功能。
- 当前不支持低精场景（`fp8`/`fp4`）的数据采集与结果分析，建议使用 `fp16`/`bf16`/`fp32` 等常规精度进行 ACLGraph 排查。

## 整网采集

### 快速入门

1. 在使用整网采集功能前，需要配置文件（`config.json`）：

    ```json
    {
    "task": "statistics",
    "dump_path": "./L0_dump",
    "rank": [],
    "level": "L0",
    "statistics": {
        "list": ["linear", "attention"],
        "seq_len": 1024
    }
    }
    ```

    **参考说明**

    整网 aclgraph dump 当前支持的配置项如下：

    | 配置项 | 可选/必选 | 说明 |
    | --- | --- | --- |
    | `task` | 可选 | 采集任务类型，str类型。默认值为`statistics`，整网 aclgraph dump 当前仅支持 `statistics`。 |
    | `dump_path` | 必选 | dump 结果输出目录，str类型。工具会检查并创建该目录。 |
    | `rank` | 可选 | 指定采集的 rank，list[int \|str]类型。默认值为空，表示采集所有 rank；字符串仅支持 `"start-end"` 范围格式。非目标 rank 不开启整网采集。|
    | `level` | 可选 | 根级采集级别，str类型。支持 `L0`、`L1`、`mix`，默认值为`L0`。<br>`L0` 采集 module 输入/输出统计值；`L1` 采集 API 输入/输出统计值；`mix` 同时采集 module 和 API 统计值。 |
    | `list` | 可选 | 模块名关键词过滤列表，list[str]类型。默认值为空，表示采集所有模块。 |
    | `seq_len` | 可选 | 仅对 `statistics` 生效，int类型。默认值为 `0`，表示按完整 Tensor 统计；当 `seq_len > 0` 且 Tensor 足够大时，仅对前导 `seq_len` 切片做统计，可用于跳过图模式下 pad 尾部数据。 |

2. 完成文件（`config.json`）配置后，下面示例展示如何使用整网采集功能：

    ```diff
      import torch
      import torch_npu
    + from msprobe.pytorch import AclGraphDumper

      N,D_in, H, D_out = 640, 4096, 2048, 1024
      # 模型初始化
      model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
      ).npu()
    + # 初始化配置
    + dumper = AclGraphDumper('./config.json')
    + # 在编图前配置采集任务
    + dumper.start(model)
      static_input = torch.randn(N, D_in).npu()
      static_target = torch.randn(N, D_out).npu()
    
      g = torch.npu.NPUGraph()
      # 编图
      with torch.npu.graph(g):
        static_target = model(static_input)

      real_inputs = [torch.rand_like(static_input) for _ in range(10)]
      real_targets = [torch.rand_like(static_target) for _ in range(10)]

      for data, target in zip(real_inputs, real_targets):
        static_input.copy_(data)
        static_target.copy_(target)
        # 图replay
        g.replay()
    +   # 数据落盘
    +   dumper.step()
    ```

### 整网采集功能介绍

#### 功能说明

`AclGraphDumper` 用于采集整网中间数据，当前支持 module 级别、API 级别以及 module+API 混合级别的统计值采集，结果包括张量形状、数据类型、统计值等信息。  
`AclGraphDumper` 的初始化与 `start` 调用需在模型编图（ 如`torch.npu.graph`或`torch.compile` ）之前完成。

#### 接口说明

**函数原型**

```python
AclGraphDumper(config_path: str | None = None)
```

**参数说明**

| 参数名 | 可选/必选 | 说明 |
| --- | --- | --- |
| config_path | 可选 | 配置文件路径，str类型。若不传，默认读取 msprobe 包内置 `config.json`。`dump_path`、`task`、`rank`、`level`、`list` 与 `seq_len` 从该配置文件中读取。 |

**函数原型**

```python
AclGraphDumper.start(model: torch.nn.Module) -> None
```

**参数说明**

| 参数名 | 可选/必选 | 说明 |
| --- | --- | --- |
| model | 必选 | 待采集模型，torch.nn.Module类型。 |

**函数原型**

```python
AclGraphDumper.step(dump: bool = True) -> None
```

**参数说明**

| 参数名 | 类型 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| dump | bool | 是否将当前统计结果落盘到 `dump.json`。`True`：清理统计并落盘，`step_id` 增加 1；`False`：仅清理统计不落盘，`step_id` 不增加（可用于 `dummy_run` 预热阶段）。 | 否 |

若未启动采集则直接返回。

### 输出说明

**单卡场景**

`AclGraphDumper` 单卡场景输出路径为：`dump_path/step{step_id}/pid{pid}/dump.json`。

生成目录示例：

```text
L0_dump
├── step0
│   └── pid9527
│       └── dump.json
├── step1
│   └── pid9527
│       └── dump.json
├── step2
|   └── pid9527
|       └── dump.json
```

**多卡场景**

`AclGraphDumper` 输出路径为：`dump_path/step{step_id}/rank{rank_id}/dump.json`。

生成目录示例：

```text
L0_dump
├── step0
│   └── rank0
│   |    └── dump.json
│   └── rank1
│   |    └── dump.json
│   └── rank2
│         └── dump.json
```

### 比对说明

可直接通过 `msprobe compare` 对整网采集结果进行比对。  
比对完成后会生成 csv 报告文件，例如：`compare_result_{rank_id}_{timestamp}.csv`。

在分布式多进程场景中，通常会按 rank 生成对应的 compare 结果文件，请结合 rank 维度查看结果。

## 单点采集

### 快速入门

下面示例展示如何在前向过程中保存某个张量：

```diff
  import torch
  import torch_npu
 
+ from msprobe.pytorch import acl_save
 
 
  class ToyModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.linear = torch.nn.Linear(8, 4)
 
      def forward(self, x):
          y = self.linear(x)
+         # 保存中间张量
+         acl_save(y, "./dump/linear_out.pt")
          return y
 
 
  if __name__ == "__main__":
      model = ToyModel().to("npu:0")
      x = torch.randn(2, 8, device="npu:0")
      out = model(x)
```

### 单点采集功能介绍

#### 功能说明

`acl_save` 用于保存张量数据，调用后会生成 `.pt` 文件。

#### 接口说明

**函数原型**

```python
acl_save(x: torch.Tensor, path: str) -> torch.Tensor
```

**参数说明**

| 参数名 | 可选/必选 | 说明 |
| --- | --- | --- |
| x | 必选 | 待保存张量，torch.Tensor类型。 |
| path | 必选 | 保存路径（支持相对/绝对路径），str类型。实际落盘文件名会在该路径文件名基础上追加序号，格式为 `{base}_{seq}.pt`。例如传入 `./dump/act.pt`，实际落盘为 `./dump/act_0.pt`、`./dump/act_1.pt`。 |

**返回值**

返回一个与输入形状一致的张量，仅用于触发保存操作。

#### 使用示例

1. 推理过程中的单点保存

    ```python
    from msprobe.pytorch import acl_save
    
    logits = model(x)
    acl_save(logits, "./dump/logits.pt")
    ```

2. 多卡场景下的单点采集

   ```python
   # 多卡场景需要区分rank，使用参考如下。
   # 需要保证“./dump/rank{torch.distributed.get_rank()}”目录已创建，否则会出现目录不存在问题。
   acl_save(tensor,f'./dump/rank{torch.distributed.get_rank()}/tensor.pt')
   ```

### 输出说明

调用 acl_save 后，会在 path 指定目录下生成 .pt 文件（文件名自动追加序号），例如生成：./dump/act_0.pt、./dump/act_1.pt、./dump/act_2.pt。

### 数据解析

`.pt` 文件为 PyTorch 序列化格式，可通过 `torch.load` 读取：

```python
import torch

tensor = torch.load("./dump/act_0.pt")
```

## 附录

### 常见问题

**1. 导入报错：Failed to import msprobe.lib.aclgraph_dump_ext**

请确认：

- 编译安装时已包含 `--include-mod=aclgraph_dump`；
- 已安装 TorchNPU 且环境变量配置正确；
- 当前系统为 Linux。

**2. `Allocate SQ failed` 问题**

CANN 8.5 以下（不含 8.5）可能出现 `Allocate SQ failed`，这是老版本 SQ 不复用导致。可将 `ccsrc/aclgraph_dump/aclgraph_dump.cpp` 中 `CurrentNPUStream` 改为 `DefaultNPUStream` 规避，或升级至 CANN 8.5.0及以上版本。
