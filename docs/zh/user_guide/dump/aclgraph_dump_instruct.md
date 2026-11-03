# aclgraph_dump 使用指南

## 简介

在 **PyTorch ACLGraph** 图模式下执行精度对齐时，整体策略如下：**先整网筛查，后单点深挖**。通过**整网采集**快速收敛异常范围，针对疑似问题算子进行**单点采集**并保存其 Tensor 数据，以便于细粒度排查。`aclgraph_dump` 提供如下采集能力：

- 单点采集：`acl_save`
- 整网采集：`AclGraphDumper`，支持统计值采集、Tensor 真实数据采集、自定义 API 采集以及动态启停。

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
        "slice": [{"end": 1024}]
    }
    }
    ```

    **参考说明**

    整网 aclgraph dump 当前支持的配置项如下：

    | 配置项         | 可选/必选 | 说明                                                                                                                     |
    |-------------| -- |------------------------------------------------------------------------------------------------------------------------|
    | `task` | 可选 | 采集任务类型，str类型。可选值：<br/>&#8226; `statistics`：采集统计信息，配置示例见本节。<br/>&#8226; `tensor`：采集 Tensor 真实数据，配置示例见[Tensor 整网采集](#tensor-整网采集)。<br/>&#8226; `nan_check`：检测 NaN，输出中任一 Tensor 含 NaN 时保存该调用的全部输入和输出，配置示例见[NaN 自动检测](#nan-自动检测)。<br/>默认值为 `statistics`。 |
    | `dump_path` | 必选 | dump 结果输出目录，str类型。工具会检查并创建该目录。                                                                                         |
    | `rank`      | 可选 | 指定采集的 rank，list[int \|str]类型。默认值为空，表示采集所有 rank；字符串仅支持 `"start-end"` 范围格式。非目标 rank 不开启整网采集。                             |
    | `dump_enable` | 可选 | 采集开关，bool类型，默认值为 `true`。运行过程中修改该字段可动态启停采集，详细说明请参见[动态启停](#动态启停)。                                              |
    | `level`     | 可选 | 采集级别，str类型。支持 `L0`、`L1`、`mix`，默认值为 `L0`。<br>`L0` 采集 module 输入/输出；`L1` 采集 API 输入/输出；`mix` 同时采集 module 和 API。      |
    | `list`      | 可选 | module 或 API 名称的关键词过滤列表，list[str]类型。默认值为空，表示采集当前 `level` 覆盖的全部数据。                                                        |
    | `custom_api` | 可选 | 自定义 Python API 的完整导入路径列表，list[str]类型，仅 `tensor` 任务支持。详细说明请参见[自定义 API 采集](#自定义-api-采集)。                              |
    | `slice`     | 可选 | 对 Tensor 进行切片，list[dict]类型，仅 `statistics` 任务生效。详细配置方法请参见[slice参数配置说明](./config_json_introduct.md#slice参数配置说明)。             |

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
    + # 设置默认设备，确保采集开关在编图前创建在 NPU 上
    + torch.set_default_device("npu:0")
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

`AclGraphDumper` 用于采集整网中间数据，支持 module 级别、API 级别以及 module+API 混合级别采集。`statistics` 任务输出Tensor形状、数据类型和统计值；`tensor` 任务输出 Tensor 真实数据。
`AclGraphDumper` 的初始化与 `start` 调用需在模型编图（如`torch.npu.graph`或`torch.compile`）之前完成。

> [!NOTE]
>
> 使用整网采集会使图 capture（编图）阶段变慢。这是因为工具会在编图过程中向 ACLGraph 中插入采集节点（用于统计值计算或 Tensor 数据准备等），图中节点数量随采集范围增大而增多，capture 阶段耗时相应增加，采集的 module/API 数量越多，影响越明显。编图完成后，这些采集节点会作为图的一部分在每次 replay 时执行，以完成数据采集。

#### 接口说明

**函数原型**

```python
AclGraphDumper(config_path: str | None = None)
```

**参数说明**

| 参数名 | 可选/必选 | 说明                                                                                                            |
| --- | --- |---------------------------------------------------------------------------------------------------------------|
| config_path | 可选 | 配置文件路径，str类型。若不传，默认读取 msProbe 包内置 `config.json`。采集任务及其参数均从该文件读取。 |

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
| dump | bool | 是否将当前统计结果落盘到 `dump.json`，`statistics` 和 `nan_check` 任务生效。`True`：清理统计并落盘，`step_id` 增加 1；`False`：仅清理统计不落盘，`step_id` 不增加（可用于 `dummy_run` 预热阶段）。`tensor` 任务会归档当前步骤生成的 `.pt` 文件并推进 `step_id`。 | 否 |

若未启动采集则直接返回。

### NaN 自动检测

将 `task` 配置为 `nan_check`，使用相同的 `start(model)` / `step()` 接口：

```json
{
  "task": "nan_check",
  "dump_path": "./nan_dump",
  "level": "L1",
  "nan_check": {
    "list": [],
    "detail": true
  }
}
```

**参数说明**

| 参数 | 可选/必选 | 解释 |
| ---- | --------- | ---- |
| list | 可选 | 待检测 module/API 的名称关键字，`list[str]` 类型，默认为 `[]`，表示不限制范围。 |
| check_mode | 可选 | `str` 类型，仅支持 `"isnan"`。省略时与显式设置 `"isnan"` 等效，均只检测 NaN，不检测 Inf 或过程溢出；其他取值会报错。 |
| detail | 可选 | `bool` 类型，默认为 `true`，控制是否额外生成 `dump.json`。两档均在输出含 NaN 时保存该调用全部输入和输出，详见[nan_check 任务输出](#nan_check-任务)。 |

- ACLGraph 捕获时记录检测与采集节点，后续由 replay 执行。预热结果使用 `step(dump=False)` 清理，`step()` 无采集数据时不生成空 `dump.json`。
- 检测、元信息记录和保存直接组合已有算子，经 `torch.compile` 后仍可进行 ACLGraph 捕获和重放，无需新增自定义算子。
- 支持 `L0`（module）、`L1`（API）和 `mix`，可用 `list` 和 `rank` 缩小范围。该次调用的输出中任一 Tensor 含 NaN 时，保存全部输入和输出 Tensor；输入的检测结果不控制保存。
- 非空 FP8/FP4 Tensor 保留原始 dtype 和 shape，`is_nan` 记录为 `null`，不执行不支持的 NaN 归约；未知标志不触发保存。同次调用的其他输出含 NaN 时，仍保存全部原始输入、输出，包括 FP8/FP4 Tensor。整数、bool 和空 Tensor 的 `is_nan` 为 `0`。
- L1 将返回值和 schema 标注为可写的 Tensor 入参作为执行后输出，同一个 Tensor 不重复计入输出。L0 输入和 L1 可写调用的输入会在执行前保留快照，以记录 inplace 修改前的值。直接跳过 API 名称包含 `empty` 的调用。
- 每次 replay 后调用 `step()`，结果位于 `dump_path/step{step_id}/{rank或pid}/`，真实数据在该目录的 `dump_tensor_data/`；仅 `detail: true` 时生成 `dump.json`。
- 预热后调用 `step(dump=False)` 会清理本轮结果及已生成的 NaN 文件，不推进 step。检测覆盖完整 Tensor，`slice` 仅适用于 `statistics`。

支持检测的浮点 Tensor 先在设备上归约为标量，再做 `isnan`。输出检测标志合并后控制该调用全部输入、输出的保存。`detail: false` 不调用 `acl_stat` 记录结果；`detail: true` 分别回传输入和输出的检测标志，通过零元素视图记录 dtype 和 shape。命中调用的输入和输出由 `acl_tensor_save` 完整回传并保存，复用现有 C++ 原语及接口。输入快照和非连续 Tensor 的连续化会产生设备侧拷贝。动态关闭 `dump_enable` 会跳过回调采集，图内检测和拷贝节点仍会执行。

### Tensor 整网采集

将 `task` 配置为 `tensor` 后，`AclGraphDumper` 会保存采集范围内 module 或 API 的输入、关键字参数和输出 Tensor 真实数据。配置示例如下：

```json
{
    "task": "tensor",
    "dump_path": "./tensor_dump",
    "rank": [],
    "dump_enable": true,
    "level": "mix",
    "tensor": {
        "list": ["linear", "attention"]
    }
}
```

每次 ACLGraph replay 后调用 `dumper.step()`，工具会将本次 replay 生成的 `.pt` 文件归档到对应的 `step` 目录。同一 API 调用的输入、关键字参数和输出共享调用序号，文件名格式如下：

```text
{api_name}.{call_index}.{input|input_kwargs|output}[.{index}].pt
```

其中 `{api_name}` 为采集对象名称：module/API 采集使用采集点名称（如 `linear`）；自定义 API 采集使用 `custom_api` 中配置的完整导入路径，详见[自定义 API 采集](#自定义-api-采集)。

目录中的进程标识按运行场景区分：多卡场景使用 `rank{rank_id}`，单卡场景使用 `rank0`；若单卡场景未初始化分布式环境，则使用 `pid{pid}`。

采集过程中，`dump_tensor_data` 目录分为以下两类（以下 `{process_dir}` 表示上述进程标识）：

- `dump_path/dump_tensor_data/{process_dir}`：临时工作目录，用于保存当前 replay 生成、尚未归档的 `.pt` 文件。调用 `dumper.step()` 后，其中的文件会被移动到对应的 `step` 目录；临时工作目录会保留，供后续 replay 继续使用，因此在 `step()` 执行完成后通常为空。采集运行期间请勿删除或重命名该目录。
- `dump_path/step{step_id}/{process_dir}/dump_tensor_data`：当前 step 的归档目录，用于保存该次 replay 最终落盘的 Tensor 数据。查看或解析采集结果时，请使用该目录下的 `.pt` 文件。

> [!NOTE]
>
> Tensor 整网采集会产生较大的磁盘和传输开销，建议结合 `level` 和 `list` 缩小采集范围。

### 自定义 API 采集

对于无法通过 module hook 或 PyTorch dispatch 自动识别的 Python API，可在 `tensor.custom_api` 中配置完整导入路径。该能力仅支持 `tensor` 任务。

```json
{
    "task": "tensor",
    "dump_path": "./tensor_dump",
    "level": "L1",
    "tensor": {
        "custom_api": [
            "my_package.ops.reshape_and_cache"
        ]
    }
}
```

`AclGraphDumper.start()` 会在编图前解析并包装目标 API，递归采集其位置参数、关键字参数和返回值中的 Tensor。配置时需满足以下要求：

- 配置值必须是可导入、可调用对象的完整导入路径。
- 模型编图时必须通过该路径对应的对象调用 API；在 `start()` 前已复制到其他变量的函数引用不会被包装。
- `custom_api` 在 `start()` 时生效，运行过程中修改该配置不会重新包装 API。
- 自定义 API 的输入、关键字参数和输出共享同一个 replay 调用序号。

自定义 API 采集的落盘文件名以配置的完整导入路径为前缀。例如配置 `my_package.ops.reshape_and_cache` 时，落盘文件形如：

```text
my_package.ops.reshape_and_cache.0.input.0.pt
my_package.ops.reshape_and_cache.0.input_kwargs.value_cache.pt
my_package.ops.reshape_and_cache.0.output.pt
```

完整导入路径前缀可区分不同模块或不同类下的同名方法，避免同名方法的数据混入同一组文件名、按文件名取数时数据错位；调用序号按完整导入路径独立计数。

### 动态启停

在配置文件根级设置 `dump_enable` 可控制 `statistics` 和 `tensor` 任务是否执行采集：

```json
{
    "task": "tensor",
    "dump_path": "./tensor_dump",
    "dump_enable": false,
    "level": "L1",
    "tensor": {
        "list": []
    }
}
```

即使初始值为 `false`，也必须在编图前调用 `AclGraphDumper.start()`，以便将采集节点加入 ACLGraph。运行过程中可直接修改同一配置文件中的 `dump_enable`，无需重新编图：

- `dumper.step()` 检测配置文件变化并刷新开关。
- 新开关值从后续 ACLGraph replay 开始生效。
- 动态刷新仅更新 `dump_enable`；运行过程中修改 `task`、`level`、`list`、`custom_api` 或 `slice` 不会改变已捕获的图。

`statistics` 和 `tensor` 共用 Host 动态开关。相邻且没有模型任务穿插的采集点合并为一个 Host 回调；回放时每组读取一次开关，关闭则直接返回，不复制 Tensor、不计算统计值、不保存数据。

`tensor` 保留 Python 中的 `contiguous()`，由捕图流程管理连续化所需的设备内存；关闭采集时，图中已有的连续化操作仍会执行。`statistics` 保留原始 shape/stride，开启后才复制存储范围并在 CPU 上计算统计。

采集回调通过 TaskQueue 按序提交。开关变化时先等待当前 stream 上的回调完成，再更新 Host 开关；后续 replay 使用新状态。持续关闭时，`statistics` 不增加采集专用同步，`tensor` 保留原有的 step 同步和文件归档检查。

需要重新编译安装 `aclgraph_dump` C++ 扩展。捕获任务查询与销毁回调接口已在 CANN 9.1.0、torch_npu 2.10.0.post4 验证；更旧 CANN 及多 stream 并发切换尚未验证。

> [!NOTE]
>
> 对于 `statistics` 任务，`dump_enable=false` 时调用 `dumper.step()` 不创建 dump 目录或 `dump.json`，但 step 编号仍会递增。之后重新开启采集时，结果写入对应的 step 目录。`dumper.step(dump=False)` 也不落盘，且不递增 step 编号。

### 输出说明

#### statistics 任务

**单卡场景**

`AclGraphDumper` 输出路径通常为：`dump_path/step{step_id}/rank0/dump.json`。若未初始化分布式环境，输出路径为：`dump_path/step{step_id}/pid{pid}/dump.json`。

生成目录示例：

```text
L0_dump
├── step0
│   └── rank0
│       └── dump.json
├── step1
│   └── rank0
│       └── dump.json
├── step2
|   └── rank0
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

#### tensor 任务

单卡场景输出路径为：`dump_path/step{step_id}/rank0/dump_tensor_data/*.pt`；若未初始化分布式环境，输出路径为：`dump_path/step{step_id}/pid{pid}/dump_tensor_data/*.pt`。多卡场景输出路径为：`dump_path/step{step_id}/rank{rank_id}/dump_tensor_data/*.pt`。

```text
tensor_dump
├── dump_tensor_data
│   └── rank0                         # 当前 replay 的临时工作目录
├── step0
│   └── rank0
│       └── dump_tensor_data          # step0 的归档数据
│           ├── linear.0.input.0.pt
│           └── linear.0.output.0.pt
└── step1
    └── rank0
        └── dump_tensor_data          # step1 的归档数据
            ├── linear.1.input.0.pt
            └── linear.1.output.0.pt
```

`.pt` 文件可通过 `torch.load` 读取。

`tensor` 任务不生成 `dump.json`。

#### nan_check 任务

结果位于 `dump_path/step{step_id}/{process_dir}/`。分布式场景使用 `rank{rank_id}`，未初始化分布式环境时使用 `pid{pid}`。

- `detail: true`（默认）：`dump.json` 分别记录每个输入和输出的 `dtype`、`shape`、`is_nan`；`1` 表示含 NaN，`0` 表示不含 NaN，`null` 表示无法检测。
- 位置参数位于 `input_args`，关键字参数位于 `input_kwargs`，输出位于 `output`。
- 输出中任一 Tensor 含 NaN 时，全部输入和输出保存到 `dump_tensor_data/`，JSON 通过 `data_name` 关联文件；未命中的调用不包含 `data_name`。
- `detail: false`：仅保存上述命中调用的 `.pt` 文件，不生成 `dump.json`。

以下示例中，`Aten.add` 的第二个输入及输出含 NaN，两个输入和输出均保存：

```text
nan_dump
└── step0
    └── rank0
        ├── dump.json
        └── dump_tensor_data
            ├── Aten.add.input.0.0.pt
            ├── Aten.add.input.1.0.pt
            └── Aten.add.output.0.pt
```

对应的 `dump.json` 内容示例（`dump_data_dir` 为实际保存路径）：

```json
{
  "task": "nan_check",
  "level": "L1",
  "framework": "pytorch",
  "dump_data_dir": "/path/to/nan_dump/step0/rank0/dump_tensor_data",
  "data": {
    "Aten.add.0.forward": {
      "input_args": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [2],
          "is_nan": 0,
          "data_name": "Aten.add.input.0.0.pt"
        },
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [2],
          "is_nan": 1,
          "data_name": "Aten.add.input.1.0.pt"
        }
      ],
      "output": [
        {
          "type": "torch.Tensor",
          "dtype": "torch.float32",
          "shape": [2],
          "is_nan": 1,
          "data_name": "Aten.add.output.0.pt"
        }
      ]
    }
  }
}
```

### 比对说明

`statistics` 任务可直接通过 `msprobe compare` 对整网采集结果进行比对。
比对完成后会生成 csv 报告文件，例如：`compare_result_{rank_id}_{timestamp}.csv`。

在分布式多进程场景中，通常会按 rank 生成对应的 compare 结果文件，请结合 rank 维度查看结果。

## 单点采集

### 快速入门

下面示例展示如何在前向过程中保存某个Tensor：

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
+         # 保存中间Tensor
+         acl_save(y, "./dump/linear_out.pt")
          return y


  if __name__ == "__main__":
      model = ToyModel().to("npu:0")
      x = torch.randn(2, 8, device="npu:0")
      out = model(x)
```

### 单点采集功能介绍

#### 功能说明

`acl_save` 用于保存Tensor数据，调用后会生成 `.pt` 文件。

#### 接口说明

**函数原型**

```python
acl_save(x: torch.Tensor, path: str) -> torch.Tensor
```

**参数说明**

| 参数名 | 可选/必选 | 说明 |
| --- | --- | --- |
| x | 必选 | 待保存Tensor，torch.Tensor类型。 |
| path | 必选 | 保存路径（支持相对/绝对路径），str类型。实际落盘文件名会在该路径文件名基础上追加序号，格式为 `{base}_{seq}.pt`。例如传入 `./dump/act.pt`，实际落盘为 `./dump/act_0.pt`、`./dump/act_1.pt`。 |

**返回值**

返回一个与输入形状一致的Tensor，仅用于触发保存操作。

#### 使用示例

1. 推理过程中的单点保存

    ```python
    from msprobe.pytorch import acl_save

    logits = model(x)
    acl_save(logits, "./dump/logits.pt")
    ```

2. 多卡场景下的单点采集

   ```python
   # 多卡场景需要区分rank，使用方法参考如下。
   # 需要保证“./dump/rank{torch.distributed.get_rank()}”目录已创建，否则会出现目录不存在问题。
   acl_save(tensor, f'./dump/rank{torch.distributed.get_rank()}/tensor.pt')
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

- 编译安装时通过 `python3 build.py -e include-mod=aclgraph_dump -e no-check=true` 完成编译安装；
- 已安装 TorchNPU 且环境变量配置正确；
- 当前系统为 Linux。

**2. `Allocate SQ failed` 问题**

CANN 8.5.0 以下（不含 8.5.0）可能出现 `Allocate SQ failed`，这是老版本 SQ 不复用导致。可将 `ccsrc/aclgraph_dump/aclgraph_dump.cpp` 中 `CurrentNPUStream` 改为 `DefaultNPUStream` 规避，或升级至 CANN 8.5.0及以上版本。

**3. 运行时出现提示信息：`[WARNING]: Invalid statistics detected. Please use tensor mode to collect the affected data.`**

统计结果中存在取值为 null 的统计量（即部分统计量无法计算，如 min、max、mean、norm 记录为 null）时，系统会触发此提示。导致统计量无法计算的原因不限于低精度数据类型，一些预分配 Tensor（如通过 torch.empty 创建的 Tensor）也会导致该问题。若需要进一步分析受影响的数据，建议切换至真实数据模式进行数据采集，具体操作可参见[Tensor 整网采集](#tensor-整网采集)。
