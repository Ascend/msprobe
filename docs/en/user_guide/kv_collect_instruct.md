# KV Statistics Collection for vLLM

KV Collect is loaded through vLLM's general plugin entry point. This feature is supported only on the V1 Engine of vLLM Ascend, and its configuration key is `kv_trace`.

## Enable collection

Install msProbe in the Python environment that starts vLLM so that the entry point can be discovered. Setting `PYTHONPATH` alone does not register the entry point.

```shell
export VLLM_USE_V1=1
vllm serve /path/to/model \
  --additional-config '{"kv_trace":{"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats"}}'
```

For eager mode, add `--enforce-eager`. For graph mode, retain the native compilation configuration.

**Parameters**

| Parameter | Type | Description |
|-----|--------|-------|
| kv_stats_enable | `bool` | `true` enables KV statistics collection and `false` disables it. The feature is disabled by default. |
| kv_stats_dir | `str` | Path where the collected data is written to disk. It must be an absolute path, and it is mandatory once collection is enabled. |
| kv_stats_max_layers | `int` | Maximum number of layers that contain the KV cache to collect. By default, all layers are collected. |

## Output Description

Collection results are stored in a separate directory for each rank as follows:

```text
{kv_stats_dir}/
└── rank{N}/
    ├── ledger.jsonl
    └── {req_id}/
        ├── step0.json
        ├── step1.json
        └── ...
```

- `rank{N}`: Distributed rank.
- `{req_id}`: Request ID.
- `step{M}`: Collection step number of the request, starting from 0.

**Step File Structure**

`meta` field:

| Field | Type | Description |
|-----|-----|------|
| role | `str` | `P` for prefill step, `D` for decode step |
| req_id | `str` | Request ID |
| time | `float` | Collection timestamp |
| block_size | `int` | KV cache block size |
| prompt_len | `int` | Prompt length of the request |
| dtype | `str` | Data type of the KV cache tensor |
| step | `int` | Collection step number, starting from 0 |
| seq_len | `int` | Sequence length of the request |

`per_token_stats` structure:

Level 1: The key is the full module name, for example, `model.layers.0.self_attn.attn`. The exact naming depends on the model architecture.

Level 2: The key is the component name, `0` or `1`, whose meaning depends on the architecture. For non-MLA models, the components are K and V, and for MLA models, they are the compressed KV latent vector and the RoPE key.

Level 3: An array of statistics, where each element corresponds to one token:

| Field | Type | Description |
|-----|-----|------|
| pos | `int` | Global position of the token in the sequence, starting from 0 |
| slot | `int` | Physical KV cache slot corresponding to the token |
| mean | `float` | Mean of the KV values of the token |
| var | `float` | Variance of the KV values of the token |

Example:

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

**`ledger.jsonl`**

Each line is a JSON object that records one KV write or transfer event.

| Field | Type | Description |
|-----|-----|------|
| `write` | `str` | Write type. `prefill` means that the request is written in prefill mode in this step; `decode`means that the request is written in decode mode in this step; `dummy` indicates a dummy batch write in an idle step or during graph capture; `transfer_kv_cache` indicates a transfer received from the peer under PD disaggregation. |
| `req_id` | `str` | Request ID |
| `time` | `float` | Event timestamp (Unix seconds) |
| `tid` | `int` | Collection thread ID |
| `block_size` | `int` | KV cache block size |
| `blocks` | `int[]` | List of block numbers involved |
| `slots` | `int[]` | List of physical slots involved |

Example:

```json
{"write": "transfer_kv_cache", "req_id": "chatcmpl-abc", "blocks": [1], "block_size": 128, "tid": 281461865115904, "time": 1794233530.2128}
{"write": "prefill", "req_id": "chatcmpl-abc", "slots": [138], "tid": 281473053683840, "time": 1794233530.4455}
{"write": "decode", "req_id": "chatcmpl-abc", "slots": [139], "tid": 281473053683840, "time": 1794233530.6424}
```

## Model Information and Performance Baseline

**Environment**

NPU: Atlas A2 training server products (Ascend 910B4, 32 GB per card, 2 cards in this test)

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
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --seed 1024 \
  --max-model-len 512 --max-num-batched-tokens 512 \
  --trust-remote-code --gpu-memory-utilization 0.9 --enforce-eager \
  --additional-config '{"kv_trace":{"kv_stats_enable":true,"kv_stats_dir":"/data/kv_stats"}}'
```

**Test Scenario**

| Item | Value |
|-----|-----|
| Deployment | Single instance, TP=2 |
| Request pattern | 20 serial requests; the first 3 are warm-up, and the median of the last 17 is taken. |
| Input length | 184-token prompt |
| Output length | `max_tokens=32`, `temperature=0` |
| Collection configuration | Three settings: disabled, all layers, and `kv_stats_max_layers=2` |

**Performance Baseline**

The following data is averaged over multiple test runs. Actual performance varies slightly with the environment state (the fluctuation between runs in this test was about ±3%).

| Collection Configuration | End-to-End Latency |
|-----|-----|
| Collection disabled | 3661 ms |
| All layers collected | 5852 ms (+59.8%) |
| Two layers collected | 3958 ms (+8.1%) |

**Reproduction**

- Start the service with the launch command above and wait until `/v1/models` returns `200`.

- Send 20 serial requests with curl and record the end-to-end time of each request.

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

- Discard the first 3 warm-up results and average the remaining 17 results to obtain the end-to-end latency in the table above.
