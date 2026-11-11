# vLLM KV数据统计值采集

KV collect 通过 vLLM 的 general plugin entry point 加载，本特性仅支持 vLLM Ascend 的 V1 Engine，配置字段名为 `kv_trace`。

## 启用

在启动 vLLM 的 Python 环境中正常安装 msprobe，使 entry point 可被发现；单独设置 `PYTHONPATH` 不会注册入口。
 
```shell
export VLLM_USE_V1=1
vllm serve /path/to/model \
  --additional-config '{"kv_trace":{"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats"}}'
```
 
Eager 模式可增加 `--enforce-eager`；图模式沿用原生 compilation 配置。
 
**参数说明**
 
| 参数 | 类型 |解释 |
|-----|--------|-------|
| kv_stats_enable | `bool` | `true`开启KV统计值采集, `false`关闭KV统计值采集，默认关闭 |
| kv_stats_dir | `str` | 采集落盘的路径，必须是绝对路径，采集开关开启后必须填采集落盘的路径 |
| kv_stats_max_layers | `int` | 采集包含kvcache的最大层数，默认采集所有层 |

## 产物介绍 
 
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

## 模型信息和性能基线

**环境信息**

NPU：Atlas A2 训练服务器系列产品（Ascend 910B4，单卡 32 GB，本次使用 2 张）

软件：vLLM 0.22.1 + vllm-ascend，Torch 2.10.0

**模型信息**

主要数据类型：BFLOAT16

模型层数：27

模型结构：DeepSeek-V2-Lite（MLA），每层对应一个采集模块，`per_token_stats` 的分量 `0`/`1` 分别为压缩 KV 潜在向量与 RoPE 键。

并行配置：TP=2

启动命令参数：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve /data/weights/DeepSeek-V2-Lite \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --seed 1024 \
  --max-model-len 512 --max-num-batched-tokens 512 \
  --trust-remote-code --gpu-memory-utilization 0.9 --enforce-eager \
  --additional-config '{"kv_trace":{"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats"}}'
```

**测试场景**

| 项 | 值 |
|-----|-----|
| 部署形态 | 单实例，TP=2 |
| 请求方式 | 20 次串行请求，前 3 次预热，取后 17 次的中位数 |
| 输入长度 | prompt 184 token |
| 输出长度 | `max_tokens=32`，`temperature=0` |
| 采集配置 | 关闭 / 全层 / `kv_stats_max_layers=2` 三档 |

**性能基线**

以下数据为多次测试后取平均值，实际运行时性能会随环境状态稍有浮动（本次轮间波动约 ±3%）。

| 采集配置 | 端到端时延 |
|-----|-----|
| 关闭采集 | 3661 ms |
| 全层采集 | 5852 ms（+59.8%） |
| 限 2 层采集 | 3958 ms（+8.1%） |

**复现方式**

- 按上文启动命令启动服务，等待 `/v1/models` 返回 200。

- 用 curl 串行发送 20 次请求，记录每次端到端耗时：

```shell
PROMPT=$(for i in $(seq 16); do printf 'Explain the difference between latency and throughput in one paragraph. '; done)

for i in $(seq 20); do
  start=$(date +%s%N)
  curl -s -o /dev/null http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"dsv2\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":32,\"temperature\":0,\"stream\":true}"
  end=$(date +%s%N)
  echo "req${i}: $(( (end - start) / 1000000 )) ms"
done
```

- 丢弃前 3 次预热结果，对后 17 次耗时取平均值，即为上表中的端到端时延。
