# KV Cache Statistics Collection Performance Baseline

**Environment**

NPU: Atlas A2 training server products (32 GB per card, 2 cards in this test)

Software: vLLM 0.22.1 + vllm-ascend, Torch 2.10.0

**Model Information**

Main data type: BFLOAT16

Number of model layers: 27

Model architecture: DeepSeek-V2-Lite (MLA). Each layer corresponds to one collection module, and components `0` and `1` of `per_token_stats` are the compressed KV latent vector and the RoPE key, respectively.

Parallel configuration: TP=2

Launch command parameters:

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve /data/weights/DeepSeek-V2-Lite \
  --host 0.0.0.0 --port 8000 --served-model-name dsv2 \
  --tensor-parallel-size 2 --seed 1024 \
  --max-model-len 512 --max-num-batched-tokens 512 \
  --trust-remote-code --gpu-memory-utilization 0.9 --enforce-eager \
  --additional-config '{"kv_trace":{"enabled":false,"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats"}}'
```

**Test Scenario**

| Item | Value |
|-----|-----|
| Deployment | Single instance, TP=2 |
| Request pattern | 20 serial requests; the first 3 are warm-up, and the average of the last 17 is taken |
| Input length | 184-token prompt |
| Output length | `max_tokens=32`, `temperature=0` |
| Collection configuration | Collection disabled: `{"kv_trace":{"enabled":false,"kv_stats_enable":false}}`<br>All layers collected: `{"kv_trace":{"enabled":false,"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats"}}`<br>Two layers collected: `{"kv_trace":{"enabled":false,"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats","kv_stats_max_layers":2}}` |

**Request Method**

- Send 20 serial requests with curl and record the end-to-end time of each request:

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

- Discard the first 3 warm-up results and average the remaining 17 results to obtain the end-to-end latency in the table below.

**Performance Baseline**

The following data is averaged over multiple test runs. Actual performance varies slightly with the environment state (the fluctuation between runs in this test was about ±3%).

| Collection Configuration | End-to-End Latency |
|-----|-----|
| Collection disabled | 3661 ms |
| All layers collected | 5852 ms (+59.8%) |
| Two layers collected | 3958 ms (+8.1%) |
