# vLLM KV Cache 统计值采集性能基线

本文档介绍vLLM KV Cache统计值采集性能基线，给出关闭采集、全层采集、限2层采集三档配置的端到端时延数据与复现命令，适用于需要评估KV Cache统计值采集开销的用户。

**环境信息**

NPU：Atlas A2 训练服务器系列产品（单卡 32 GB，本次使用 2 张）

软件：vLLM 0.22.1 + vllm-ascend、Torch 2.10.0

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
| 请求方式 | 20 次串行请求，前 3 次预热，取后 17 次的平均数 |
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
