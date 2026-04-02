# aclgraph_dump 使用指南

## 简介

aclgraph_dump 提供 `acl_save` 接口，适用于aclgraph场景，可以将张量保存为 `.pt` 文件。

## 使用前准备

**环境准备**

1. 安装 msProbe 工具，详见《[msProbe 安装指南](../msprobe_install_guide.md)》。
2. 源码编译安装时需包含 `aclgraph_dump` 模块：

   ```bash
   python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
   ```

3. 安装并正确配置 Ascend Extension for PyTorch（torch_npu）和 CANN （同msProbe安装要求）环境。

**约束**

- 仅支持 PyTorch 框架。
- 构建 `aclgraph_dump` 需要 torch_npu 参与编译；若未包含该模块，将无法导入 `msprobe.lib.aclgraph_dump_ext`。

## 快速入门

下面示例展示如何在前向过程中保存某个张量：

```python
import torch
import torch_npu

from msprobe.pytorch import acl_save

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        y = self.linear(x)
        # 保存中间张量
        acl_save(x, "./dump/linear_out.pt")
        return y

if __name__ == "__main__":
    model = ToyModel()
    x = torch.randn(2, 8)
    x.to("npu:4")
    out = model(x)
```

## 数据采集功能介绍

### 功能说明

`acl_save` 用于保存张量数据，调用后会保存`.pt`文件。

### 接口说明

**函数原型**

```python
acl_save(x: torch.Tensor, path: str) -> torch.Tensor
```

**参数说明**

| 参数名 | 类型 | 说明 | 是否必选 |
| --- | --- | --- | --- |
| x | torch.Tensor | 待保存的张量 | 必选 |
| path | str | 保存路径（支持相对/绝对路径），str类型。实际落盘文件名会在该路径的文件名基础上追加序号，格式为 `{base}_{seq}.pt`。例如传入 `./dump/act.pt`，实际落盘为 `./dump/act_0.pt`、`./dump/act_1.pt`（其中父目录`./dump/`须为已存在的目录文件夹）。| 必选 |

**返回值**

返回一个与输入形状一致的张量。无实际意义，仅用于触发保存操作。

### 使用示例

#### 1. 推理过程中的单点保存

```python
from msprobe.pytorch import acl_save

logits = model(x)
acl_save(logits, "./dump/logits.pt")
```

#### 2. 保存多次调用的序号文件

```python
for step in range(3):
    y = model(x)
    acl_save(y, "./dump/act.pt")
```

会生成：`./dump/act_0.pt`、`./dump/act_1.pt`、`./dump/act_2.pt`。

## 输出说明

### dump 结果文件介绍

调用 `acl_save` 后，会在 `path` 指定目录下生成 `.pt` 文件。

### 数据解析

保存格式为 PyTorch `.pt` 文件（pickle 序列化），可通过 `torch.load` 读取：

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

**2. `Allocate SQ failed`问题**

CANN 8.5 以下（不含8.5版本）可能会出现`Allocate SQ failed`问题，这是由于老版本存在SQ不复用的问题，建议可以将ccsrc/aclgraph_dump/aclgraph_dump.cpp中的`CurrentNPUStream`改为`DefaultNPUStream`规避，或者升级至 CANN 8.5.0+ 版本。
