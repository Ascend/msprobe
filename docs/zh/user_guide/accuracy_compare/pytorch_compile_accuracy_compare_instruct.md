# PyTorch场景编译精度比对

## 简介

`PrecisionChecker` 是一个用于 PyTorch 模型 **eager 模式与 compile 模式逐模块精度对比** 的工具。在为模型启用 `torch.compile` 后，若出现 loss 抖动、收敛变差或推理输出不一致等问题，可借助本工具逐模块定位差异来源。

工具的工作方式为：通过 `wrap` 标记需要检查的模块，分别以 eager 和 compile 两种方式执行这些模块，并逐个对比其前向输出与输入梯度，从而确定差异所在的模块、差异类型（前向或反向）及误差量级。

工具接口参考 FSDP2 `fully_shard` 的使用方式，支持按模块类型、模块层级或单个模块粒度指定检查范围。

## 模式选择

工具提供 single-pass 与 two-pass 两种对比模式，正确选择模式是使用本工具的前提。建议首次使用时参照下表选择：

| 场景 | 推荐模式 | 入口 API |
| --- | --- | --- |
| 训练（包含 `loss.backward()`） | single-pass（默认） | `install()` + `collect()` |
| 推理 / eval（仅 forward，无反向） | two-pass | `compare()` |
| FSDP2 等多卡分布式训练 | single-pass（必须） | `install()` + `collect()` |
| 需要真实的整网 `loss_eager` 与 `loss_compiled` 对比 | two-pass | `compare()` |
| 模型不支持 `deepcopy`（如 FSDP2 分片后） | single-pass（必须） | `install()` + `collect()` |

### 训练场景推荐 single-pass

single-pass 为默认模式。该模式不复制模型，也不单独执行 eager 全网链路，而是直接接入已有的训练流程：在被 compile 的子模块 forward hook 中，使用相同输入重新执行未编译的 `_orig_mod`，并将 eager 子模块输出与 compile 输出进行对比。

single-pass 适用于训练场景的原因如下：

- 接入成本低。无需修改原有的 `forward → loss → backward` 流程，仅需在训练步前调用 `install()`，训练步结束后调用 `collect(loss)`。
- 不改动优化器与模型状态，避免因复制模型再训练导致的状态分叉问题。
- 兼容 FSDP2 等分布式封装。此类模型禁止 `deepcopy`，而 single-pass 不执行 deepcopy，因此是多卡训练场景下唯一可用的模式。

需要注意：single-pass 不单独执行 eager 全模型，因此不计算整网 eager loss，报告中的 `loss_eager` 为 `NaN`，属于预期行为（详见[关于 single-pass 的 loss_eager=NaN](#关于-single-pass-的-loss_eagernan)）。精度判定依据为逐模块的前向与反向对比结果，而非整网 loss。

### 推理场景推荐 two-pass

推理场景无反向计算与优化器状态，也不存在复制模型污染训练状态的问题，因此更适合使用 two-pass：

- two-pass 会分别完整执行一遍 eager 模型与一遍 compiled 模型，可给出真实的 `loss_eager`、`loss_compiled` 及 `loss_diff`，便于判断整网是否因 compile 产生差异。
- 推理链路通常为“输入—输出”模式，two-pass 的“两条链路各执行一遍再对比”语义直观，且结果完整（可对比前向输入、前向输出与梯度）。

约束：two-pass 内部会对模型执行 `deepcopy` 以构造 eager 与 compiled 两份副本，因此不支持 FSDP2 等禁止 deepcopy 的模型。普通单卡推理不受此约束影响。

> 小结：训练场景使用 single-pass（`install` + `collect`），推理场景使用 two-pass（`compare`）。

## 使用前准备

**安装**

安装 msProbe 工具，详情请参见《[msProbe 安装指南](../../install_guide/msprobe_install_guide.md)》。

**约束**

- 仅支持 PyTorch 场景。
- 依赖 `torch.compile` 能力，请确保当前 PyTorch 版本支持 `torch.compile`。
- `install()` 仅支持 `single_pass=True` 场景。
- two-pass（`single_pass=False`）会 `deepcopy` 模型，不支持 FSDP2 等禁止 deepcopy 的模型。
- `dump_graphs` 依赖 PyTorch Dynamo 内部接口，PyTorch 版本变化可能影响 graph dump 行为。

## 快速入门

以下两个示例分别对应模式选择表中最常见的两类场景，均可独立运行。

### 训练场景：single-pass 接入

训练流程通常已包含 `forward → loss → backward` 的训练步。single-pass 的接入方式为在训练步外层调用 `install()` 与 `collect()`，无需修改训练步本身：

```python
import torch
import torch.nn as nn

from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.layers(x)


model = ToyModel()
x = torch.randn(2, 8)
target = torch.randn(2, 4)
loss_fn = nn.MSELoss()


def run_step(model):
    model.zero_grad()
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    return loss


checker = PrecisionChecker()                 # single_pass=True 为默认值
checker.wrap_by_policy(model, (nn.Linear,))  # 标记参与对比的模块
checker.install(model)                       # 原地替换被标记模块为 compiled 版本

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path="precision_report.csv")
```

说明：

- `install()` 须在 `run_step()` 之前调用，用于原地将被 `wrap` 的子模块替换为编译版本。
- `collect(loss)` 须在 `run_step()` 之后调用，用于整理本步采集到的逐模块对比结果。
- 报告中的 `loss_eager` 为 `NaN`，属于预期行为，精度判定依据为逐模块的 `FORWARD OUTPUT` 与 `BACKWARD`。

### 推理场景：two-pass 接入

推理场景仅有前向计算。使用 two-pass 可分别完整执行 eager 与 compiled 模型，直接对比整网输出：

```python
import torch
import torch.nn as nn

from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker

model = build_model()          # 推理模型
input_ids = make_input()       # 推理输入


def infer_step(model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        # two-pass 需要一个标量作为对比锚点。推理无 loss，
        # 可使用输出的某个统计量（如 mean），其不参与反向。
        return logits.mean()


checker = PrecisionChecker(single_pass=False)   # 推理使用 two-pass
checker.wrap_by_policy(model, (nn.Linear,))

result = checker.compare(infer_step, model)     # 内部执行 eager 与 compiled 两条链路
checker.report(result, csv_path="precision_report.csv")
```

说明：

- 推理场景使用 `compare()`，无需调用 `install()`。
- two-pass 会给出真实的 `loss_eager` 与 `loss_compiled`（此处为 `logits.mean()`），可直接判断整网是否一致。
- 若推理模型使用了 FSDP2 等禁止 deepcopy 的封装，应改用 single-pass 的 `install()` + `collect()` 路径。

## 功能介绍

下表为 `PrecisionChecker` 的完整功能点：

| 功能 | 说明 | 支持场景 |
| --- | --- | --- |
| [构造参数配置](#构造参数配置) | 配置后端、阈值、graph dump、autocast、输入采集和检查模式 | PyTorch |
| [指定检查模块](#指定检查模块) | 通过 `wrap`、`wrap_by_policy`、`wrap_all_children` 指定参与 compile 和精度对比的模块 | PyTorch |
| [跳过检查模块](#跳过检查模块) | 通过 `ignore`、`ignore_by_policy` 跳过不关注模块 | PyTorch |
| [single-pass 模式](#single-pass-模式) | 原地编译被 wrap 的子模块，并通过 hook 完成快速对比 | PyTorch |
| [two-pass 模式](#two-pass-模式) | 分别执行 eager 模型和 compiled 模型，比较 loss、输入输出和梯度 | PyTorch |
| [autocast 精度检查](#autocast-精度检查) | 对被 wrap 模块启用 `torch.autocast`，检查混合精度场景 | PyTorch |
| [graph dump](#graph-dump) | 保存 PyTorch Dynamo 捕获到的 graph 代码 | PyTorch |
| [报告说明](#报告说明) | 打印 stdout 报告，可选生成 CSV 报告（`csv_path` 参数） | PyTorch |
| [多卡 FSDP2 精度检查](#多卡-fsdp2-精度检查) | 在 FSDP2 分片下使用 single-pass 完成逐模块对比，并支持多卡 CSV 报告 | PyTorch |
| [推理模式支持](#推理模式支持) | 仅执行 forward，用于检查推理链路 compile 前后一致性 | PyTorch |

### 构造参数配置

工具初始化示例如下：

```python
checker = PrecisionChecker(
    backend="aot_eager",
    threshold=1e-4,
    dump_graphs=False,
    graph_dir="./graph_dump",
    cast_dtype=None,
    capture_input=True,
    single_pass=True,
)
```

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `backend` | `str` | `"aot_eager"` | `torch.compile` 使用的后端。 |
| `threshold` | `float` | `1e-4` | 预留精度阈值参数。当前实际通过 `torch.allclose(a, b, atol=1e-4, rtol=1e-3)` 判定。 |
| `dump_graphs` | `bool` | `False` | 是否将编译期捕获到的 graph 写入 `graph_dir`。 |
| `graph_dir` | `str` | `"./graph_dump"` | graph dump 输出目录。 |
| `cast_dtype` | `torch.dtype` | `None` | 对被 wrap 的模块启用 autocast，类似 mixed precision policy。 |
| `capture_input` | `bool` | `True` | 是否采集模块前向输入，用于区分“输入已有差异”与“模块计算产生差异”。 |
| `single_pass` | `bool` | `True` | 是否使用 single-pass 对比逻辑。设为 `False` 时使用 two-pass 对比。 |

### 指定检查模块

`wrap` 用于标记参与编译和精度对比的模块。未被 `wrap` 的模块不会作为独立编译单元处理。

```python
# 手动标记单个模块
checker.wrap(model.layers[3].self_attn)

# 按类型批量标记模块，类似 ModuleWrapPolicy
checker.wrap_by_policy(model, (Qwen2Block,))
checker.wrap_by_policy(model, (Qwen2Attention, Qwen2MLP))

# 自动 wrap 子模块到指定深度
checker.wrap_all_children(model)          # depth=1，默认值
checker.wrap_all_children(model, depth=0) # wrap 直接子模块
checker.wrap_all_children(model, depth=2) # wrap 更深层级

# wrap 整个模型
checker.wrap(model)
```

注意事项：

- `install()` 会原地替换被标记的子模块。
- 若标记的是根模型本身，`torch.compile(model)` 的返回值无法写回调用者持有的变量，建议直接使用 `compare()`，或标记根模型下的子模块。
- `wrap_all_children` 会递归遍历模型子模块，并自动跳过 `ModuleList`、`ModuleDict`、`Sequential` 等容器模块。
- 对于大型模型，建议先从较粗粒度开始检查，再逐步拆分至更小模块。
- 若某个模块已被 `wrap`，其内部子模块不会重复作为顶层编译目标处理。

### 跳过检查模块

`ignore` 用于跳过指定模块的精度对比。被 `ignore` 的模块仍参与模型执行，仅不参与报告判定。

```python
checker.wrap_by_policy(model, (Qwen2Block,))

# 跳过指定模块
checker.ignore(model.layers[0])

# 按类型批量跳过模块
checker.ignore_by_policy(model, (torch.nn.Dropout,))
```

报告中被跳过的模块显示为 `IGNORED`。

### single-pass 模式

`single_pass=True` 为默认模式，推荐使用 `install()` 与 `collect()` 接入已有训练流程：

```python
checker = PrecisionChecker()
checker.wrap_all_children(model)
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result)
```

也可通过 `compare()` 使用 single-pass：

```python
checker = PrecisionChecker()
checker.wrap_by_policy(model, (Qwen2Block,))

result = checker.compare(run_step, model)
checker.report(result)
```

若需要更细粒度的对比，应将 wrap 拆分至更细的模块：

```python
checker.wrap_by_policy(model, (Qwen2Attention, Qwen2MLP))
```

#### 关于 single-pass 的 loss_eager=NaN

single-pass 模式下，工具不单独运行完整的 eager 模型，因此无法给出真实的 `loss_eager`。报告中显示的 `loss_eager=NaN` 为预期行为，非工具缺陷。

single-pass 的精度判定依据为逐模块对比：在每个被 compile 的子模块 forward hook 中，使用相同输入重新执行 `_orig_mod`（未编译版本），并将 eager 子模块输出与 compile 子模块输出进行对比。此方式的优势在于无需复制模型、无需另起一条训练链路，代价为无法得到整网的 eager loss。

若需要真实的 `loss_eager` 与 `loss_compiled` 对比，请改用 two-pass 模式（`single_pass=False`），该模式会分别执行完整的 eager 与 compiled 两条链路。

### two-pass 模式

`single_pass=False` 时启用 two-pass。工具会构造 eager 模型与 compiled 模型，分别执行同一个 `run_step`，再对两侧 hook 采集到的输入、输出和梯度进行对比。

```python
checker = PrecisionChecker(single_pass=False)
checker.wrap_by_policy(model, (Qwen2Block,))

result = checker.compare(run_step, model)
checker.report(result)
```

two-pass 模式会比较 `loss_eager` 与 `loss_compiled`，可用于判断整网 compile 链路是否引入 loss 差异。

约束：two-pass 内部会 `deepcopy` 模型以构造 eager 与 compiled 两份副本，因此不支持 FSDP2 等禁止 deepcopy 的模型。若模型已使用 FSDP2 封装，请改用 single-pass 模式。

### autocast 精度检查

设置 `cast_dtype` 后，工具会对被 wrap 的模块增加 `_CastWrapper`，并在 forward 中使用 `torch.autocast`。用于验证自动混合精度场景下被 wrap 的模块是否存在精度差异。

```python
checker = PrecisionChecker(
    backend="aot_eager",
    cast_dtype=torch.bfloat16,
)
checker.wrap_by_policy(model, (torch.nn.Linear,))

result = checker.compare(run_step, model)
checker.report(result)
```

### graph dump

开启 `dump_graphs` 后，工具会拦截 PyTorch Dynamo 的图代码格式化流程，将捕获到的 graph 保存至 `graph_dir`。

```python
checker = PrecisionChecker(
    backend="aot_eager",
    dump_graphs=True,
    graph_dir="./graph_dump",
)
checker.wrap_all_children(model)

result = checker.compare(run_step, model)
checker.report(result)
```

输出文件名示例：

```text
__compiled_fn_0.Captured_Graph.xxxxxxxx.py
__compiled_fn_1.Forward_Graph.xxxxxxxx.py
__compiled_fn_2.Backward_Graph.xxxxxxxx.py
```

注意：该能力依赖 PyTorch 内部接口，PyTorch 版本变化可能影响 dump 行为。

### 报告说明

工具支持打印 stdout 报告：

```python
checker.report(result)
```

也支持在打印 stdout 报告的同时生成 CSV 报告。`report()` 的 `csv_path` 参数默认为 `None`，仅在显式传入路径时生成 CSV 文件，写入完成后会打印 `CSV report saved to: <path>`：

```python
checker.report(result, csv_path="precision_report.csv")
```

CSV 报告的列定义详见下文 [CSV 报告格式](#csv-报告格式)。

stdout 报告示例：

```text
========================================================================
  Loss  eager=6.912340  compiled=6.912341  diff=1.200e-06
========================================================================

  FORWARD INPUT
  --------------------------------------------------------------------
  pass  layers.0                                             [OK] max_abs=0.000e+00  mean_abs=0.000e+00  max_rel=0.000e+00  shape=(2, 128, 4096)

  FORWARD OUTPUT
  --------------------------------------------------------------------
  pass  layers.0                                             [OK] max_abs=2.300e-07  mean_abs=1.200e-08  max_rel=5.100e-06  shape=(2, 128, 4096)

  BACKWARD
  --------------------------------------------------------------------
  pass  layers.0.grad_input                                  [OK] max_abs=1.100e-07  mean_abs=9.000e-09  max_rel=4.200e-06  shape=(2, 128, 4096)
  pass  layers.0.grad_output                                 [OK] max_abs=0.000e+00  mean_abs=0.000e+00  max_rel=0.000e+00  shape=(2, 128, 4096)

========================================================================
  RESULT: ALL PASS  (atol=1e-4 rtol=1e-3)
========================================================================
```

> 提示：single-pass 模式下 `Loss` 行的 `eager` 显示为 `nan`，属于预期现象，精度判定依据为逐模块的 `FORWARD OUTPUT` 与 `BACKWARD`。

报告区域说明：

| 区域 | 说明 |
| --- | --- |
| `FORWARD INPUT` | 前向输入对比。`capture_input=False` 时不展示。 |
| `FORWARD OUTPUT` | 前向输出对比。 |
| `BACKWARD` | 反向梯度对比。two-pass 会对比 `grad_input` 与 `grad_output`；single-pass 当前仅尝试重建并对比 `grad_input`，`grad_output` 通常为 `None`。 |
| `RESULT` | 整体结果，`ALL PASS` 表示所有未跳过、未忽略的检查项通过。 |

常见标记说明：

| 标记 | 含义 |
| --- | --- |
| `pass` | 精度通过。 |
| `FAIL` | 精度超出 `torch.allclose(a, b, atol=1e-4, rtol=1e-3)` 判定范围。 |
| `skip IGNORED` | 模块被 `ignore()` 或 `ignore_by_policy()` 跳过。 |
| `skip SKIP_compiled_wrapper` | 该模块为 compiled wrapper，部分 backward hook 不直接对齐，属于预期现象。 |
| `skip SKIP_inside_compiled` | 该模块位于已编译模块内部，graph 融合后 hook 不触发，属于预期现象。 |
| `WARN MISSING_fwd_in_compiled` | eager 侧有采集结果，但 compiled 侧未采集到，需排查 wrap 粒度或 hook 触发情况。 |
| `WARN MISSING_fwd_in_eager` | compiled 侧有采集结果，但 eager 侧未采集到。 |
| `WARN MISSING_bwd_in_compiled` | eager 侧有反向信息，但 compiled 侧未采集到。 |

#### CSV 报告格式

传入 `csv_path` 参数后，工具会生成 CSV 格式的详细报告，包含以下 9 列：

| 字段 | 说明 |
| --- | --- |
| `module_name` | 模块名称，整网 loss 显示为 `LOSS`。 |
| `check_type` | 检查类型，包括 `loss`、`fwd_input`、`fwd_output`、`grad_input`、`grad_output`、`note`。 |
| `tensor_index` | 张量索引，从 0 开始。多输入或多输出时用于区分张量。 |
| `status` | `PASS`、`FAIL`、`SKIP` 或 `WARN`。 |
| `max_abs_diff` | 最大绝对误差，格式为科学计数法（如 `1.192093e-07`）。 |
| `mean_abs_diff` | 平均绝对误差，格式为科学计数法。 |
| `max_rel_diff` | 最大相对误差，格式为科学计数法。 |
| `shape` | 张量形状（如 `(2, 128, 4096)`）或 loss 信息（如 `eager=6.912340 compiled=6.912341`）。 |
| `note` | 跳过、忽略或告警信息。正常通过时为空。 |

CSV 示例（部分行）：

```csv
module_name,check_type,tensor_index,status,max_abs_diff,mean_abs_diff,max_rel_diff,shape,note
LOSS,loss,0,PASS,1.200000e-06,N/A,N/A,eager=6.912340 compiled=6.912341,
layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 128, 4096)",
layers.0,fwd_output,0,PASS,2.300000e-07,1.200000e-08,5.100000e-06,"(2, 128, 4096)",
layers.0,grad_input,0,PASS,1.100000e-07,9.000000e-09,4.200000e-06,"(2, 128, 4096)",
layers.1,note,0,SKIP,,,,,IGNORED
```

single-pass 模式下，`LOSS` 行的 `shape` 字段仅显示 `compiled` 值，`note` 字段为 `single_pass mode`：

```csv
LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
```

### 多卡 FSDP2 精度检查

FSDP2（`torch.distributed.fsdp.fully_shard`）会将参数切分为 `DTensor` 并分布至多张卡，且禁止 `deepcopy`，因此仅支持 single-pass 模式。FSDP2 场景下的典型用法如下：

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

dist.init_process_group(backend="hccl")  # 或 "nccl" 等
rank = dist.get_rank()

model = build_model()
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

checker = PrecisionChecker()  # single_pass=True，默认
checker.wrap_by_policy(model, (Qwen2Block,))
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path=f"precision_rank{rank}.csv")
```

说明：

- 每张卡独立执行 `checker.report()`，生成各自的 CSV 报告。
- 工具本身不内置多卡结果汇总能力。若需将多卡结果合并至单个 CSV，可参照下文的多卡 CSV 报告格式自行实现汇总逻辑。

#### 多卡 CSV 报告格式

将多卡、多场景结果合并至单个 CSV 时，可采用以下格式，即在原 9 列基础上前置 `rank` 与 `scenario` 两列：

```csv
rank,scenario,module_name,check_type,tensor_index,status,max_abs_diff,mean_abs_diff,max_rel_diff,shape,note
0,core,LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
0,core,layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 16, 128)",
0,core,layers.0,fwd_output,0,PASS,1.192093e-07,3.421843e-08,2.980232e-07,"(2, 16, 128)",
0,core,layers.0,grad_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 16, 128)",
1,core,LOSS,loss,0,PASS,N/A,N/A,N/A,compiled=0.333408,single_pass mode
1,core,layers.0,fwd_input,0,PASS,0.000000e+00,0.000000e+00,0.000000e+00,"(2, 16, 128)",
0,bf16,layers.1,fwd_output,0,PASS,3.051758e-05,7.510185e-06,6.556511e-05,"(2, 16, 128)",
1,ignore,layers.0,-,0,SKIP,N/A,N/A,N/A,,IGNORED
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `rank` | 卡号，对应 `dist.get_rank()`。 |
| `scenario` | 测试场景名称（如 `core`、`bf16`、`ignore`），用于区分不同配置组合，例如是否开启 `cast_dtype`、是否 `ignore` 指定模块。 |
| 其余 9 列 | 与单卡 CSV 相同。 |

多卡结果汇总可采用如下实现思路：使用 `torch.multiprocessing.spawn` 启动多卡进程，各卡将本卡的逐模块明细写入临时 JSON 文件，再由主进程读取所有 JSON 并汇总为单个 CSV。

### 推理模式支持

推理场景无反向计算，可使用 single-pass 的 `install()` / `collect()` 路径，也可使用 two-pass 的 `compare()`。single-pass 接入方式如下：

```python
checker = PrecisionChecker()
checker.wrap_all_children(model)
checker.install(model)

model.eval()
with torch.no_grad():
    logits = model(input_ids)
    pseudo_loss = logits.mean()

result = checker.collect(pseudo_loss)
checker.report(result)
```

推理模式下：

- `FORWARD INPUT` 与 `FORWARD OUTPUT` 正常展示。
- `BACKWARD` 相关项通常为 `None`。
- 适合验证推理链路 compile 前后的输出一致性。

若需完整的 `loss_eager` 与 `loss_compiled` 对比（此处为输出统计量的对比），可改用 two-pass：

```python
checker = PrecisionChecker(single_pass=False)
checker.wrap_all_children(model)


def infer_step(model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        return logits.mean()


result = checker.compare(infer_step, model)
checker.report(result)
```

## 精度指标

精度指标由工具采集到的对比张量计算得到。two-pass 模式下，工具分别执行 eager 模型和 compiled 模型，对两侧采集到的输入、输出和梯度进行比较；single-pass 模式下，工具在 compiled 模块执行时使用相同输入重新执行 `_orig_mod`，再比较两侧结果。比较前，工具会将参与比较的张量转换为 CPU float32。

| 指标 | 说明 |
| --- | --- |
| `max_abs` | `abs(a - b).max()`，最大绝对误差。 |
| `mean_abs` | `abs(a - b).mean()`，平均绝对误差。 |
| `max_rel` | `(abs(a - b) / (abs(a) + 1e-8)).max()`，最大相对误差。 |
| `allclose` | `torch.allclose(a, b, atol=1e-4, rtol=1e-3)`。 |
| `shape` | 张量形状。shape 不一致时直接判定失败。 |

精度指标不是单独的输出文件，而是 stdout 报告和 CSV 报告中的核心字段。CSV 中的 `max_abs_diff`、`mean_abs_diff`、`max_rel_diff`、`shape` 分别对应上表中的 `max_abs`、`mean_abs`、`max_rel`、`shape`，`status` 由 `allclose` 等检查结果转换得到。用户可根据这些指标判断差异发生的模块、差异类型（前向或反向）、误差量级，以及是否超过工具判定阈值。

## 常见问题

**single-pass 的 `loss_eager` 为何为 `NaN`？**

single-pass 模式不单独执行完整的 eager 模型，因此无法计算整网 eager loss，`loss_eager=NaN` 为预期行为。精度判定依据为逐模块的 `FORWARD OUTPUT` 与 `BACKWARD`。详见[关于 single-pass 的 loss_eager=NaN](#关于-single-pass-的-loss_eagernan)。

**FSDP2 模型是否可使用 two-pass？**

不可。two-pass 内部会 `deepcopy` 模型，而 FSDP2 禁止 deepcopy。FSDP2 场景请使用 single-pass 模式。

**推理场景应选择哪种模式？**

推荐 two-pass（`single_pass=False`）。推理无反向计算与优化器状态，two-pass 可给出完整的整网 eager 与 compiled 对比。若推理模型使用了 FSDP2 等禁止 deepcopy 的封装，应改用 single-pass。

**训练场景应选择哪种模式？**

推荐 single-pass（默认）。该模式接入成本低、不改动训练状态，且兼容 FSDP2。若需要真实的 `loss_eager` 与 `loss_compiled` 对比，可改用 two-pass，但 two-pass 不支持 FSDP2。
