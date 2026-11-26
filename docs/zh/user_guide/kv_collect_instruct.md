# vLLM KV Cache统计值采集

## 简介

KV Cache用于保存模型生成过程中各token的Key/Value数据。vLLM按缓存块（block）管理这些数据，一个缓存块包含多个token的存储槽位（slot）。调度器为每个请求安排使用的缓存块，模型执行时通过`block_table`指定读取的缓存块，通过`slot_mapping`指定各token的写入槽位。

KV Collect按步采集每个token的KV Cache统计值（均值和方差）及其序列位置、物理槽位，按rank、请求和步号写入`step{N}.json`，并在账本`ledger.jsonl`中记录写入与PD分离下KV Cache传输事件，用于比对P/D两侧或不同步的KV Cache数据，辅助定位精度问题。

## 使用前准备

**环境准备**

在启动vLLM的Python环境中执行以下命令，从源码安装msprobe：

```shell
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
pip install --no-deps .
```

**约束**

- 本功能适用于vLLM Ascend，支持Eager和图模式。
- 当前仅支持单一注意力机制模型，会排除mamba层。

## 功能介绍

**功能说明**

在推理服务运行过程中，按步采集每个包含KV Cache的层在当前步写入的token，计算每个token的KV Cache统计值：

- 均值（`mean`）和方差（`var`），用于比对P/D两侧或不同步的KV Cache数据。
- token在请求序列中的全局位置（`pos`）和对应的物理槽位（`slot`），用于将统计值与缓存块、槽位对应起来。

采集结果按rank、请求和步号写入`step{N}.json`；同时在账本`ledger.jsonl`中记录KV Cache写入事件（`prefill`、`decode`、`dummy`）和PD分离下的传输事件（`transfer_kv_cache`），用于还原每个step的写入来源。

**命令格式**

```shell
vllm serve <model_path> --enforce-eager --additional-config '<additional_config>'
```

**参数说明**

| 参数名 | 可选/必选 | 参数说明 |
| --- | --- | --- |
| `<model_path>` | 必选 | 模型路径 |
| `--additional-config` | 必选 | 启用本功能的JSON配置，包含`kv_trace`对象。配置示例：`--additional-config '{"kv_trace":{"enabled":false,"kv_stats_enable":true,"kv_stats_dir":"/path/to/kv_stats","kv_stats_max_layers":2}}'`。各字段含义见[kv_trace对象参数](#kv_trace_parameters) |

**kv_trace对象参数**<a name="kv_trace_parameters"></a>

| 参数名 | 可选/必选 | 参数说明                                                    |
| --- | --- |---------------------------------------------------------|
| `enabled` | 可选 | 布尔值，`true` 开启 KV Trace（详细介绍请参见[KV Trace](./kv_trace_instruct.md)），`false` 关闭。服务默认关闭 KV Trace；提供 `kv_trace` 配置后，此字段默认值为 `true`。当使用Model Runner V1时，需要配置为`false` |
| `kv_stats_enable` | 可选 | 布尔值，是否开启采集，`true`开启采集，`false`关闭采集，默认是`false`           |
| `kv_stats_dir` | 可选 | 采集落盘的路径；`kv_stats_enable`为`true`时必填，否则服务启动失败 |
| `kv_stats_max_layers` | 可选 | 非负整数，采集包含KV Cache的最大层数，设为`0`不采集任何层，默认采集所有层          |

**使用示例**

以下示例使用Eager模式开启采集，仅采集前2层：

```shell
vllm serve /path/to/model \
  --enforce-eager \
  --additional-config '{"kv_trace":{"enabled":false,"kv_stats_enable":true,"kv_stats_dir":"/path/to/kv_stats","kv_stats_max_layers":2}}' \
  > server.log 2>&1
```

将`/path/to/model`替换为模型路径，`/path/to/kv_stats`替换为实际落盘目录。使用图模式时，删除上述命令中的`--enforce-eager`参数，并沿用服务现有的编译配置。

需要采集所有层时，删除`kv_stats_max_layers`，仅保留`enabled`（设为`false`）、`kv_stats_enable`和`kv_stats_dir`。采集会带来端到端时延开销，建议先用少量层验证配置和输出，再按需扩大采集范围。

**输出说明**

采集结果按rank分目录存放，目录结构与各文件字段的含义请参见[输出结果说明](#输出结果说明)。

配置缺失或路径无效时，服务启动阶段会报错并停止启动；单个step写入失败时输出警告日志，不影响推理继续执行。

## 输出结果说明

采集结果按rank分目录存放，结构如下：

```text
{kv_stats_dir}/
└── rank{N}/
    ├── ledger.jsonl
    └── {req_id}/
        ├── step0.json
        ├── step1.json
        └── ...
```

- `rank{N}`：分布式rank
- `{req_id}`：请求id
- `step{M}`：该请求的采集步号，从0开始

**step文件结构**

`meta`字段：

| 字段 | 类型 | 说明 |
|-----|-----|------|
| role | `str` | `P`=prefill步，`D`=decode步 |
| req_id | `str` | 请求id |
| time | `float` | 采集时间戳 |
| block_size | `int` | KV Cache缓存块大小 |
| prompt_len | `int` | 该请求的prompt长度 |
| dtype | `str` | KV Cache缓存张量的dtype |
| step | `int` | 采集步号，从0开始 |
| seq_len | `int` | 该请求的序列长度 |

`per_token_stats`结构：

第一层：键为模块完整名，例如`model.layers.0.self_attn.attn`；具体命名取决于模型结构。

第二层：键为`0`、`1`，其语义由注意力类型决定：使用MLA时，分别为压缩KV Cache潜在向量、RoPE键；非MLA时分别为K、V。

第三层：统计数组，每个元素对应一个token：

| 字段 | 类型 | 说明 |
|-----|-----|------|
| pos | `int` | 该token在序列中的全局位置，从0开始 |
| slot | `int` | 该token对应的KV Cache物理槽位 |
| mean | `float` | 该token对应KV值的均值 |
| var | `float` | 该token对应KV值的方差 |

示例：

```json
{
  "meta": {
    "role": "D",
    "req_id": "chatcmpl-abc",
    "time": 1794233530.6088,
    "block_size": 128,
    "prompt_len": 11,
    "dtype": "torch.bfloat16",
    "step": 0,
    "seq_len": 12
  },
  "per_token_stats": {
    "model.layers.0.self_attn.attn": {
      "0": [
        {
          "pos": 0,
          "slot": 128,
          "mean": 0.008538579568266869,
          "var": 0.02398228459060192
        }
      ],
      "1": [
        {
          "pos": 0,
          "slot": 128,
          "mean": -0.004213,
          "var": 0.019884
        }
      ]
    }
  }
}
```

**账本`ledger.jsonl`**

每行一条JSON，记录一次KV Cache写入或者传输事件：

| 字段 | 类型 | 说明 |
|-----|-----|------|
| `write` | `str` | 写入类型：`prefill` = 当前步以prefill方式写入；`decode` = 当前步以decode方式写入；`dummy` = 空闲步或抓图阶段的dummy批写入；`transfer_kv_cache` = PD分离下收到对端传输 |
| `req_id` | `str` | 请求id |
| `time` | `float` | 事件时间戳（Unix秒） |
| `tid` | `int` | 采集线程id |
| `block_size` | `int` | KV Cache块大小 |
| `blocks` | `int[]` | 涉及的块号列表 |
| `slots` | `int[]` | 涉及的物理槽位列表 |

示例：

```json
{"write": "transfer_kv_cache", "req_id": "chatcmpl-abc", "blocks": [1], "block_size": 128, "tid": 281461865115904, "time": 1794233530.2128}
{"write": "prefill", "req_id": "chatcmpl-abc", "slots": [138], "tid": 281473053683840, "time": 1794233530.4455}
{"write": "decode", "req_id": "chatcmpl-abc", "slots": [139], "tid": 281473053683840, "time": 1794233530.6424}
```

## 性能基线

vLLM KV Cache统计值采集性能膨胀大小，请参见[vLLM KV Cache统计值采集性能基线报告](../baseline/kv_collect_perf_baseline.md)
