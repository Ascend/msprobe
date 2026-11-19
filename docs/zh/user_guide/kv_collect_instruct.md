# vLLM KV Cache 统计值采集

## 简介

KV Cache 用于保存模型生成过程中各 token 的 Key/Value 数据。vLLM 按缓存块（block）管理这些数据，一个缓存块包含多个 token 的存储槽位（slot）。调度器为每个请求安排使用的缓存块，模型执行时通过 `block_table` 指定读取的缓存块，通过 `slot_mapping` 指定各 token 的写入槽位。

KV Collect 按步采集每个 token 的 KV Cache 统计值（均值和方差）及其序列位置、物理槽位，按 rank、请求和步号写入 `step{N}.json`，并在账本 `ledger.jsonl` 中记录写入与 PD 传输事件，用于比对 P/D 两侧或不同步的 KV 数据，辅助定位精度问题。

## 使用前准备

**环境准备**

在启动 vLLM 的 Python 环境中执行以下命令，从源码安装 msprobe：

```shell
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
pip install --no-deps .
```

**约束**

- 本功能适用于 vLLM Ascend，支持 Eager 和图模式。
- 模型执行组件只支持 Model Runner V1（MRV1），通过 `VLLM_USE_V1=1` 启用。

## 功能介绍

**功能说明**

按步采集每个 token 的 KV Cache 统计值（均值和方差）及其序列位置、物理槽位，按 rank、请求和步号写入 `step{N}.json`，并在账本 `ledger.jsonl` 中记录写入与 PD 传输事件，用于比对 P/D 两侧或不同步的 KV 数据，辅助定位精度问题。

**使用方式**

通过 `--additional-config` 的 `kv_trace` 对象开启采集。以下示例使用 Eager 模式，仅采集前 2 层：

```shell
export VLLM_USE_V1=1
vllm serve /path/to/model \
  --enforce-eager \
  --additional-config '{"kv_trace":{"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats","kv_stats_max_layers":2}}'
```

将 `/path/to/model` 替换为模型路径，`/data/kv_stats` 替换为实际落盘目录。使用图模式时，删除上述命令中的 `--enforce-eager` 参数，并沿用服务现有的编译配置。

**参数说明**

| 参数名 | 可选/必选 | 参数说明 |
| --- | --- | --- |
| `kv_trace.kv_stats_enable` | 可选 | 布尔值，`true` 开启，`false` 关闭，默认关闭 |
| `kv_trace.kv_stats_dir` | 可选 | 采集落盘的绝对路径；`kv_stats_enable` 为 `true` 时必填 |
| `kv_trace.kv_stats_max_layers` | 可选 | 采集包含 KV Cache 的最大层数，默认采集所有层 |

## 输出说明
 
采集结果按 rank 分目录存放，结构如下：
 
```
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
| block_size | `int` | kvcache缓存块大小 |
| prompt_len | `int` | 该请求的prompt长度 |
| dtype | `str` | kvcache缓存张量的dtype |
| step | `int` | 采集步号，从0开始 |
| seq_len | `int` | 该请求的序列长度 |
 
`per_token_stats`结构：

第一层：键为模块完整名，例如`model.layers.0.self_attn.attn`；具体命名取决于模型结构。

第二层：键为分量名`0`或者`1`，含义随架构而定：非MLA为K/V，MLA为压缩KV潜在向量/RoPE键。

第三层：统计数组，每个元素对应一个 token：
 
| 字段 | 类型 | 说明 |
|-----|-----|------|
| pos | `int` | 该token在序列中的全局位置，从0开始 |
| slot | `int` | 该token对应的kvcache物理槽位 |
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
 
**账本 `ledger.jsonl`**
 
每行一条 JSON，记录一次KV写入或者传输事件：
 
| 字段 | 类型 | 说明 |
|-----|-----|------|
| `write` | `str` | 写入类型：`prefill` = 该请求本步以 prefill 方式写入；`decode` = 该请求本步以 decode 方式写入；`dummy` = 空闲步或抓图阶段的 dummy 批写入；`transfer_kv_cache` = PD 分离下收到对端传输 |
| `req_id` | `str` | 请求 id |
| `time` | `float` | 事件时间戳（Unix 秒） |
| `tid` | `int` | 采集线程 id |
| `block_size` | `int` | kvcache 块大小 |
| `blocks` | `int[]` | 涉及的块号列表 |
| `slots` | `int[]` | 涉及的物理槽位列表 |
 
示例：
 
```json
{"write": "transfer_kv_cache", "req_id": "chatcmpl-abc", "blocks": [1], "block_size": 128, "tid": 281461865115904, "time": 1794233530.2128}
{"write": "prefill", "req_id": "chatcmpl-abc", "slots": [138], "tid": 281473053683840, "time": 1794233530.4455}
{"write": "decode", "req_id": "chatcmpl-abc", "slots": [139], "tid": 281473053683840, "time": 1794233530.6424}
```

## 性能基线

vLLM KV Cache 统计值采集性能膨胀大小，请参见 [vLLM KV Cache 统计值采集性能基线报告](../baseline/kv_collect_perf_baseline.md)
