# 训练状态轻量化监测工具

## 简介

`Monitor V2` 是 msProbe 的训练状态轻量化监测工具。用户可以在训练过程中按需采集关键中间量（如模块输入/输出、权重梯度、优化器动量、通信算子统计等），并以 CSV 形式落盘，用于训练稳定性评估与异常定位。

**工具使用流程**

1) 配置需要采集的监测项；
2) 在训练代码中初始化并按 step 调用一次 `mon.step()`；
3) 在输出目录查看对应 CSV 结果。

**工具适用场景**

- 当 loss 上扬或出现尖刺时，通过 `module` 观察模块前向输入/输出与反向梯度的数值分布，判断异常是否来自特定层/特定阶段。
- 当 grad norm 异常时，通过 `weight_grad` 定位异常参数与异常阶段（`unreduced`：梯度累积期；`reduced`：梯度聚合前）。
- 当收敛变差或出现震荡时，通过 `optimizer` 观察动量（`exp_avg/exp_avg_sq`）分布是否发生突变。
- 当分布式训练出现同步/通信异常时，通过 `cc` 聚合通信算子统计，并结合代码行过滤缩小排查范围。

**推荐启用策略**

- 首先长期启用 `weight_grad` 作为底座监测；随后在问题发生时按需启用 `module` / `cc` ，并通过目标筛选降低开销。

## 使用前准备

**安装**

请先安装 msProbe 工具，参考：[`msprobe_install_guide.md`](./msprobe_install_guide.md)。

**约束**

- PyTorch 场景：`torch >= 2.1`
- MindSpore 场景：`mindspore >= 2.4.10`（通常要求动态图环境；若项目中使用了 MSAdapter/mindtorch，请以工程实际为准）
- `monitor_v2` 当前仅支持 `format=csv`

**版本范围与限制**

为避免与 `monitor_instruct.md`（v1 Monitor）混淆，这里明确 `monitor_v2` 当前实现范围：

- 输出：仅支持 `format=csv`（不支持 `tensorboard` / `api`）
- 配置：以 `monitors.<name>` 组织功能，不使用 v1 的 `xy_distribution/wg_distribution/...` 等开关命名
- 配套能力：当前版本暂不提供 v1 中的 `print_struct`、`stack_info`、异常告警（alert）、csv2tensorboard/csv2db 等工具链

## 快速入门<a id="quickstart-config"></a>

### 准备配置文件

在训练脚本同目录创建 `monitor_v2_config.json`，示例（只采集 `weight_grad`，建议作为长期低开销监测项）：

更多配置字段与含义请参考：[详细配置](#config-details)。

```json
{
  "framework": "pytorch",
  "output_dir": "./monitor_v2_out",
  "rank": [0],
  "start_step": 0,
  "step_interval": 1,
  "step_count_per_record": 1,
  "collect_times": 100,
  "format": "csv",
  "monitors": {
    "weight_grad": {
      "enabled": true,
      "ops": ["min", "max", "mean", "norm", "nans"],
      "eps": 1e-8,
      "monitor_mbs_grad": false
    }
  }
}
```

### PyTorch场景示例<a id="quickstart-pytorch"></a>

首先在「模型与优化器准备完成后」初始化监测器；随后在「每个训练 step 的结束处」调用一次 `mon.step()`；最后在训练结束时调用 `mon.stop()` 释放资源。关键在于保证每个 step 调用一次（通常放在 `optimizer.step()` 之后、`optimizer.zero_grad()` 之前或之后）。

> [!note]
>
> 若配置 `patch_optimizer_step=true`（或传入 optimizer 且未显式配置该项），会自动包装 `optimizer.step()` 触发采集，此时不要再手动调用 `mon.step()`；如需手动调用，请显式设置 `patch_optimizer_step=false`。

```python
from msprobe.core.monitor_v2.trainer import TrainerMonitorV2

mon = TrainerMonitorV2("./monitor_v2_config.json", fr="pytorch")  # fr 可省略，默认读 config.framework
mon.start(model=model, optimizer=optimizer, grad_acc_steps=grad_acc_steps)

for _ in range(num_steps):
    loss = forward(...)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mon.step()

mon.stop()
```

### MindSpore场景示例<a id="quickstart-mindspore"></a>

```python
from msprobe.core.monitor_v2.trainer import TrainerMonitorV2

mon = TrainerMonitorV2("./monitor_v2_config.json", fr="mindspore")
mon.start(model=model, optimizer=optimizer, grad_acc_steps=grad_acc_steps)
for _ in range(num_steps):
    ...
    mon.step()
mon.stop()
```

### 常见用法建议

为提高定位效率，建议按以下顺序收敛问题空间：

- 若 loss 异常但 grad norm 正常：优先启用 `module`，并通过 `targets` 聚焦可疑模块，观察前向与反向的输入输出分布。
- 若 grad norm 异常：优先启用 `weight_grad`，以 `unreduced/reduced` 将异常划分到反向累积期或 step 前。
- 若怀疑通信相关：尽量在训练早期实例化并 `start()` `TrainerMonitorV2`，以降低加速库缓存原始通信 API 导致拦截不生效的风险；同时配合 `cc_codeline` 进行过滤。

## 功能介绍

`monitor_v2` 通过配置文件的 `monitors` 字段按需开启子功能。每个子功能独立启停、独立输出，便于按问题场景组合使用。

### module（模块输入/输出与反向梯度）

**功能说明**

采集指定模块的前向输入/输出，以及反向的 `grad_input/grad_output` 统计指标，用于定位“哪一层、哪一类张量”出现尖刺、NaN/Inf、尺度突变等问题。

**注意事项**

- 开销通常高于 `weight_grad`，建议优先缩小监测范围（通过 `targets` 只监测可疑模块）。
- 输出中的 `module_name` 会包含 `input/output/grad_input/grad_output`，用于区分采集位置。

**使用示例**

以下字段配置仅展示 `monitors.module`；需合并到完整 `monitor_v2_config.json`，见[快速入门 > 准备配置文件](#quickstart-config)。

```json
{
  "monitors": {
    "module": {
      "enabled": true,
      "targets": ["encoder.layers.0", "mlp"],
      "ops": ["min", "max", "mean", "norm", "nans"],
      "eps": 1e-8
    }
  }
}
```

配置字段说明：见[详细配置 > module](#config-module)。

代码使能示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

**输出说明**

见[module.csv](#output-module-csv)。

### weight_grad（权重梯度）

**功能说明**

采集权重梯度统计指标，用于定位“哪个参数的梯度先异常、异常发生在反向累积期还是 step 前。

- `unreduced`：反向传播阶段采集（更贴近梯度生成过程）。
- `reduced`：在调用 `optimizer.step()` 前采集（更贴近 step 前最终梯度形态）。

**注意事项**

若存在梯度累积/微步（micro-batch），建议通过 `grad_acc_steps` 或 `micro_batch_number` 告知微步数；需要微步粒度时开启 `monitor_mbs_grad`。

**使用示例**

以下字段配置仅展示 `monitors.weight_grad`；需合并到完整配置，见[快速入门 > 准备配置文件](#quickstart-config)；微步展开。

```json
{
  "monitors": {
    "weight_grad": {
      "enabled": true,
      "monitor_mbs_grad": true,
      "grad_acc_steps": 8
    }
  }
}
```

配置字段说明：见[详细配置 > weight_grad](#config-weight-grad)。

代码使能示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

**输出说明**

见[weight_grad.csv](#output-weight-grad-csv)。

### optimizer（优化器动量m/v）

**功能说明**

采集优化器状态的统计指标，当前以 Adam 类动量为主（`exp_avg` / `exp_avg_sq`），用于定位“优化器状态是否突变/是否出现异常分布”。

**注意事项**

- 仅在提供 `optimizer` 的前提下生效（`mon.start(model=..., optimizer=...)`）。
- 当前主要覆盖动量类信息（`mv_distribution`），其余扩展能力以版本为准。

**使用示例**

以下字段配置仅展示 `monitors.optimizer`；需合并到完整配置，见[快速入门 > 准备配置文件](#quickstart-config)。

```json
{
  "monitors": {
    "optimizer": {
      "enabled": true,
      "mv_distribution": true,
      "ops": ["min", "max", "mean", "norm", "nans"]
    }
  }
}
```

配置字段说明：见[详细配置 > optimizer](#config-optimizer)。

代码使能示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

**输出说明**

见[optimizer.csv](#output-optimizer-csv)。

### param（参数分布）

**功能说明**

采集参数在优化器 step 前后的分布统计，用于定位参数异常或更新异常。

**注意事项**

- 仅在提供 `optimizer` 的前提下生效（`mon.start(model=..., optimizer=...)`）。
- 默认开启 `param_distribution`，可按需关闭。

**使用示例**

以下字段配置仅展示 `monitors.param`；需合并到完整配置，见[快速入门 > 准备配置文件](#quickstart-config)。

```json
{
  "monitors": {
    "param": {
      "enabled": true,
      "param_distribution": true,
      "ops": ["min", "max", "mean", "norm", "nans"]
    }
  }
}
```

配置字段说明：见[详细配置 > monitors.param](#config-param)。

代码使用示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

**输出说明**

见[param.csv](#output-param-csv)。

### cc（通信算子）

**功能说明**

在分布式训练中采集通信算子的统计信息与日志，用于定位“异常通信调用/异常输入输出/与训练无关的通信”等问题，并可通过代码行过滤缩小排查范围。

**注意事项**

- 仅在分布式环境初始化后生效（例如 PyTorch `torch.distributed.is_initialized()` 为 true）。
- 建议尽量在训练早期实例化并 `start()`，避免部分加速库缓存原始通信 API 后导致拦截不生效。
- `cc_log_only=true` 适合用于“先打日志再收敛过滤规则”的场景，可能会中断训练，请谨慎使用。

**使用示例**

以下字段配置仅展示 `monitors.cc`；需合并到完整配置，见[快速入门 > 准备配置文件](#quickstart-config)；采集通信统计。

```json
{
  "monitors": {
    "cc": {
      "enabled": true,
      "ops": ["min", "max", "mean", "norm", "nans"],
      "cc_codeline": [],
      "cc_pre_hook": false,
      "cc_log_only": false
    }
  }
}
```

以下字段配置仅展示 `monitors.cc`；需合并到完整配置，见[快速入门 > 准备配置文件](#quickstart-config)；仅打印日志用于筛选 `cc_codeline`。

```json
{
  "monitors": {
    "cc": {
      "enabled": true,
      "cc_log_only": true
    }
  }
}
```

配置字段说明：见[详细配置 > cc](#config-cc)。

代码使能示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

**输出说明**

见[cc.csv](#output-cc-csv)。

## 输出结果文件说明

### 输出目录说明

输出目录由配置项 `output_dir` 指定。为便于多卡分析，每个 rank 独立输出到 `rank_<rank_id>/`。

当前仅支持 `format=csv`。每个 rank 单独输出到 `rank_<rank_id>/`，每个监测模块一个 CSV 文件。

目录结构如下：

```Python
<output_dir>/
  rank_<rank_id>/
    module_step0-0.csv
    weight_grad_step0-0.csv
    optimizer_step0-0.csv
    param_step0-0.csv
    cc_step0-0.csv
```

> [!note]
>
> 只有在 `monitors` 中启用的模块才会生成对应的 CSV 文件。

### CSV表头与字段说明

为便于用户直接对比不同监测项的结果，CSV 输出采用统一规则：

- 每行至少包含：`vpp_stage`、`step`
- 监测模块写入的通用字段通常包含：`module_name`、`scope`（以及部分场景的 `micro_step`）
- 统计指标按 `ops` 展开为列：`min/max/mean/norm/nans`（只会写入用户配置/启用的指标）

因此一个常见的 CSV 表头形态为：

`vpp_stage | step | module_name | scope | micro_step(可选) | min | max | mean | norm | nans`

字段含义：

- `step`：训练 step（由 `TrainerMonitorV2.step()` 递增）
- `vpp_stage`：多模型/多 stage 场景的 stage 标识（从 name 前缀 `<idx><NAME_SEP>` 推导；无前缀时默认为 0）
- `module_name`：监测对象 tag（不同模块的 tag 规则不同，详细介绍请参见[各功能特有字段说明](#output-module-csv)）
- `scope`：监测范围/阶段（不同模块语义不同，详细介绍请参见[各功能特有字段说明](#output-module-csv)）
- `micro_step`：仅在启用微步相关能力时出现（例如 `weight_grad.monitor_mbs_grad=true`）

### 各功能特有字段说明

#### module.csv<a id="output-module-csv"></a>

- `scope`：`forward` / `backward`
- `module_name`：形如 `<module_name>.<io_kind>.<idx>`，其中 `io_kind` 可能为 `input/output/grad_input/grad_output`
- 使用解读：优先关注异常 step 的同一模块在 `forward` 与 `backward` 是否同时异常，以判断异常是“前向激活异常传导”还是“梯度链路异常”。

#### weight_grad.csv<a id="output-weight-grad-csv"></a>

- `scope`：`unreduced` / `reduced`
- `module_name`：参数名（不带 scope 后缀）；`micro_step` 字段用于区分微步
- `micro_step`：开启微步监测时记录当前微步序号；未开启时记录梯度累积总微步数（来自 `micro_batch_number/grad_acc_steps`）
- 使用解读：若 `unreduced` 正常而 `reduced` 异常，通常更偏向“梯度在 step 前形态发生变化/被修改”；若 `unreduced` 已异常则更偏向“反向链路内产生异常”。

#### optimizer.csv<a id="output-optimizer-csv"></a>

- `scope`：`exp_avg` / `exp_avg_sq`（与 `module_name` 的后缀一致）
- `module_name`：形如 `<name>.exp_avg` / `<name>.exp_avg_sq`
- 使用解读：当训练震荡/不收敛时，优先对比异常前后的 `exp_avg/exp_avg_sq` 指标变化，判断是否为优化器状态突变导致。

#### param.csv<a id="output-param-csv"></a>

- `scope`：`param_origin` / `param_updated`
- `module_name`：参数名
- 使用解读：对比 step 前后参数分布变化，定位更新异常或数值突变。

#### cc.csv<a id="output-cc-csv"></a>

- `scope`：`comm`
- `module_name`：由通信监测生成的通信 tag（通常包含通信算子/序号/代码位置信息）
- 使用解读：先用 `cc_log_only=true` 获取通信日志，再设置 `cc_codeline` 过滤掉与训练无关的通信，最后开启统计采集定位异常通信的输入输出分布。

## 公开接口

本节列出 `monitor_v2` 面向用户的主要可调用接口。
说明：以下“函数原型”为文档表达，其中 `Any/Optional/Dict/Type` 为 Python `typing` 类型名。

### TrainerMonitorV2

- 功能说明：训练监测编排器；负责加载配置、初始化监测模块，并在每个 step 收集与写出监测结果。
- 函数原型：
  
  ```Python
  TrainerMonitorV2(config_path: str, fr: Optional[str] = None) -> TrainerMonitorV2
  ```

- 参数说明：
  - `config_path`：配置文件路径（JSON）。
  - `fr`：框架类型，可选（`pytorch` / `mindspore`，也支持 `pt/torch/ms`）；不传则读取配置文件的 `framework` 字段。
- 返回值说明：返回 `TrainerMonitorV2` 实例。
- 调用示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

### TrainerMonitorV2.start

- 功能说明：启动监测。根据配置创建并启动 `monitors` 中启用的模块，并建立写出上下文。
- 函数原型：
  
  ```Python
  TrainerMonitorV2.start(model: Any = None, optimizer: Any = None, **context: Any) -> None
  ```

- 参数说明：
  - `model`：待监测的模型对象（PyTorch `nn.Module` / MindSpore `nn.Cell`；也支持部分场景的模型列表）。
  - `optimizer`：待监测的优化器对象；开启 `weight_grad/optimizer` 时必须提供。
  - `context`：可选上下文信息，用于补充监测模块所需的运行参数，例如：
    - `grad_acc_steps` / `micro_batch_number`：梯度累积/微步数（影响 `weight_grad` 的 `micro_step` 语义）。
    - 其他自定义字段：会透传到各监测模块。
- 返回值说明：无。
- 调用示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

### TrainerMonitorV2.step

- 功能说明：推进一步训练 step，并触发本 step 的采集与写出（受 `start_step/stop_step/step_interval/collect_times` 控制）。
- 函数原型：
  
  ```Python
  TrainerMonitorV2.step() -> None
  ```

- 参数说明：无。
- 返回值说明：无。
- 调用示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

### TrainerMonitorV2.stop

- 功能说明：停止监测并释放资源（移除监测模块内部注册/拦截，关闭 writer）。
- 函数原型：
  
  ```Python
  TrainerMonitorV2.stop() -> Non
  ```

- 参数说明：无。
- 返回值说明：无。
- 调用示例：见[PyTorch场景示例](#quickstart-pytorch)或[MindSpore场景示例](#quickstart-mindspore)。

## 详细配置<a id="config-details"></a>

### monitor_v2_config.json

| 字段 | 可选/必选 | 类型 | 说明 |
| --- | --- | --- | --- |
| `framework` | 可选 | string | 框架类型：`pytorch` / `mindspore`（也支持 `pt/torch/ms` 别名）。 |
| `output_dir` | 可选 | string | 输出目录，默认为 `./`。 |
| `format` | 可选 | string | 输出格式，当前仅支持 `csv`。 |
| `async_write` | 可选 | bool | 暂不生效（当前 CSV 写入为同步）。 |
| `rank` | 可选 | int / list[int] | 指定需要监测的 rank；为空/不配置表示所有 rank 均监测。 |
| `start_step` | 可选 | int | 开始写出 step（包含），默认 0。 |
| `stop_step` | 可选 | int | 结束写出 step（不包含）；未配置时由 `collect_times` 推导。 |
| `step_interval` | 可选 | int | 写出频率：每隔 N 个 step 写一次（默认 1）。 |
| `step_count_per_record` | 可选 | int | 每多少个 step 合并到一个 CSV 文件（默认 1）。 |
| `patch_optimizer_step` | 可选 | bool | 是否自动包装 `optimizer.step()` 触发采集；未显式配置且传入 optimizer 时默认开启。 |
| `collect_times` | 可选 | int | 最大写出次数；达到后停止写出（默认值很大，表示几乎一直采集）。 |
| `monitors` | 可选 | dict | 监测模块配置集合，key 为模块名（见下表）。 |

### monitors配置总览

`monitors` 的每个子项格式为：

```json
{
  "enabled": true,
  "...": "各模块自定义字段"
}
```

公共字段（适用于大多数模块）：

| 字段 | 可选/必选 | 类型 | 说明 |
| --- | --- | --- | --- |
| `enabled` | 可选 | bool | 是否启用该模块；未配置时 `module` 默认启用，其余默认关闭。 |
| `ops` | 可选 | list[string] | 统计指标，支持：`min/max/mean/norm/nans`；若无有效项则使用默认值。 |
| `eps` | 可选 | number | 数值稳定项，默认 `1e-8`。 |

### module（模块输入/输出/反向梯度）<a id="config-module"></a>

| 字段 | 可选/必选 | 类型 | 说明 |
| --- | --- | --- | --- |
| `targets` | 可选 | list[string] | 目标模块筛选：为空表示全量；否则按「模块名包含关键字」命中。 |
| `ops` / `eps` | 可选 | - | 同公共字段。 |

### weight_grad（权重梯度）<a id="config-weight-grad"></a>

| 字段 | 可选/必选 | 类型 | 说明 |
| --- | --- | --- | --- |
| `monitor_mbs_grad` | 可选 | bool | 是否记录微步梯度（micro-batch），默认`false`。<br>PyTorch FSDP场景：`weight_grad` 会自动检测并在reduce前采集（`scope=unreduced`），无需单独的 `fsdp_grad`模块。 |
| `micro_batch_number` | 可选 | int | 微步数（优先级高于`grad_acc_steps`）。 |
| `grad_acc_steps` | 可选 | int | 梯度累积步数，可通过`TrainerMonitorV2.start(..., grad_acc_steps=...)`传入。 |
| `ops` / `eps` | 可选 | - | 同公共字段。 |

> [!note]
>
> `weight_grad` 会在反向阶段记录 `unreduced`，并在调用 `optimizer.step()` 前抓取并记录 `reduced`。

### optimizer（优化器动量m/v）<a id="config-optimizer"></a>

| 字段 | 可选/必选 | 类型 | 说明 |
| --- | --- | --- | --- |
| `mv_distribution` | 可选 | bool | 是否采集动量（m/v，典型为 Adam 的 `exp_avg/exp_avg_sq`），默认`true`。 |
| `ops` / `eps` | 可选 | - | 同公共字段。 |

### param（参数分布）<a id="config-param"></a>

| 字段                 | 可选/必选 | 类型 | 说明                           |
| -------------------- | --------- | ---- | ------------------------------ |
| `param_distribution` | 可选      | bool | 是否采集参数分布，默认`true`。 |
| `ops` / `eps`        | 可选      | -    | 同公共字段。                   |

> [!note]
>
> `param` 会在 `optimizer.step()` 前后采集参数分布，输出 `scope=param_origin/param_updated`。

### cc（通信算子）<a id="config-cc"></a>

仅当分布式环境已初始化时生效（例如 PyTorch `torch.distributed.is_initialized()` 为 true）。

| 字段 | 可选/必选 | 类型 | 说明 |
| --- | --- | --- | --- |
| `cc_codeline` | 可选 | list[string] | 仅监测指定代码行（形如`train.py[23]`）；为空表示不过滤。 |
| `cc_log_only` | 可选 | bool | 是否仅打印日志不采集（部分实现会在打印后中断训练，请谨慎使用）。 |
| `cc_pre_hook` | 可选 | bool | 是否监测通信输入（前置采集）。 |
| `module_ranks` | 可选 | list[int] | 仅在指定ranks上生效（未配置时默认空列表）。 |
| `ops` / `eps` | 可选 | - | 同公共字段。 |

> [!note]
>
> `monitors.cc` 同时兼容两种写法：直接配置上述字段或将字段嵌套在 `cc_distribution` 内（旧结构兼容）。
