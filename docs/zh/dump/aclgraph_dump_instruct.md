# aclgraph_dump 使用指南

## 简介

针对 aclgraph 场景，aclgraph_dump 提供如下采集能力:

- 整网采集：`AclGraphDumper`
- 单点采集：`acl_save`

## 使用前准备

**环境准备**

1. 安装 msProbe 工具，详见《[msProbe 安装指南](../msprobe_install_guide.md)》。
2. 源码编译安装时需包含 `aclgraph_dump` 模块：

   ```bash
   python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
   ```

3. 安装并正确配置 Ascend Extension for PyTorch（torch_npu）和 CANN（同 msProbe 安装要求）环境。

**约束**

- 仅支持 PyTorch 框架。
- 构建 `aclgraph_dump` 需要 torch_npu 参与编译；若未包含该模块，将无法导入 `msprobe.lib.aclgraph_dump_ext`。

## 快速入门

### 1. 整网采集（`AclGraphDumper`）

```diff
+ from msprobe.pytorch import AclGraphDumper

+ dumper = AclGraphDumper(config_path="./config.json")
+ dumper.start(model)

# 正常执行前向
_ = model(*inputs, **kwargs)

+ # 落盘
+ dumper.step()
```

### 2. 单点采集（`acl_save`）

下面示例展示如何在前向过程中保存某个张量：

```diff
 import torch
 import torch_npu
 
+from msprobe.pytorch import acl_save
 
 
 class ToyModel(torch.nn.Module):
     def __init__(self):
         super().__init__()
         self.linear = torch.nn.Linear(8, 4)
 
     def forward(self, x):
         y = self.linear(x)
+        # 保存中间张量
+        acl_save(y, "./dump/linear_out.pt")
         return y
 
 
 if __name__ == "__main__":
     model = ToyModel().to("npu:0")
     x = torch.randn(2, 8, device="npu:0")
     out = model(x)
```

## 数据采集功能介绍

### 功能说明

`AclGraphDumper` 用于采集整网中间数据，当前仅支持 module 级别统计值采集，结果包括张量形状、数据类型、统计值等信息。  
`AclGraphDumper` 的初始化与 `start` 调用需在 `torch.compile` 之前完成。

`acl_save` 用于保存张量数据，调用后会生成 `.pt` 文件。

### 接口说明

#### AclGraphDumper

**函数原型**

```python
AclGraphDumper(config_path: str | None = None)
```

**参数说明**

| 参数名 | 类型 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| config_path | str | 配置文件路径。若不传，默认读取 msprobe 包内置 `config.json`。`dump_path` 与 `list` 从该配置文件中读取。 | 否 |

**函数原型**

```python
AclGraphDumper.start(model: torch.nn.Module) -> None
```

**参数说明**

| 参数名 | 类型 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| model | torch.nn.Module | 待采集模型 | 必选 |

**函数原型**

```python
AclGraphDumper.step() -> None
```

**参数说明**

落盘当前 step 的 `dump.json`；若未启动采集则直接返回。

#### acl_save

**函数原型**

```python
acl_save(x: torch.Tensor, path: str) -> torch.Tensor
```

**参数说明**

| 参数名 | 类型 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| x | torch.Tensor | 待保存张量 | 必选 |
| path | str | 保存路径（支持相对/绝对路径）。实际落盘文件名会在该路径文件名基础上追加序号，格式为 `{base}_{seq}.pt`。例如传入 `./dump/act.pt`，实际落盘为 `./dump/act_0.pt`、`./dump/act_1.pt`。 | 必选 |

**返回值**

返回一个与输入形状一致的张量，仅用于触发保存操作。

### 使用示例

#### 1. 整网采集

配置文件示例（`config.json`）：

```json
{
  "task": "statistics",
  "dump_path": "./L0_dump",
  "statistics": {
    "list": ["linear", "attention"]
  }
}
```

说明：

- `dump_path`：采集结果输出目录。
- `list`：模块名关键词列表；仅采集包含任一关键词的模块（不区分大小写）。为空时采集整网模块。

```python
from msprobe.pytorch import AclGraphDumper

dumper = AclGraphDumper(config_path="./config.json")
dumper.start(model)

for _ in range(3):
    _ = model(*inputs, **kwargs)
    dumper.step()
```

#### 2. 推理过程中的单点保存

```python
from msprobe.pytorch import acl_save

logits = model(x)
acl_save(logits, "./dump/logits.pt")
```

#### 3. 保存多次调用的序号文件

```python
for _ in range(3):
    y = model(x)
    acl_save(y, "./dump/act.pt")
```

## 输出说明

### dump 结果文件

#### 整网采集结果

`AclGraphDumper` 输出路径为：`dump_path/step{step_id}/rank{rank_id}/dump.json`。

生成目录示例：

```text
L0_dump
├── step0
│   └── rank0
│       └── dump.json
├── step1
│   └── rank0
│       └── dump.json
└── step2
    └── rank0
        └── dump.json
```

#### 单点采集结果

调用 acl_save 后，会在 path 指定目录下生成 .pt 文件（文件名自动追加序号），例如生成：./dump/act_0.pt、./dump/act_1.pt、./dump/act_2.pt。

### 比对说明

可直接通过 `msprobe compare` 对整网采集结果进行比对。  
比对完成后会生成 xlsx 报告文件，例如：`compare_result_{rank_id}_{timestamp}.xlsx`。

在分布式多进程场景中，通常会按 rank 生成对应的 compare 结果文件，请结合 rank 维度查看结果。

新增说明（`+ python` 高亮）：

```python
+# 对整网采集结果执行比对
+msprobe compare ...
+
+# 结果示例
+compare_result_{rank_id}_{timestamp}.xlsx
```

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
- 已安装 torch_npu 且环境变量配置正确；
- 当前系统为 Linux。

**2. `Allocate SQ failed` 问题**

CANN 8.5 以下（不含 8.5）可能出现 `Allocate SQ failed`，这是老版本 SQ 不复用导致。可将 `ccsrc/aclgraph_dump/aclgraph_dump.cpp` 中 `CurrentNPUStream` 改为 `DefaultNPUStream` 规避，或升级至 CANN 8.5.0+。
