# vLLM KV Cache Recording and Validation

## Introduction

KV Cache stores the Key/Value data for tokens during model generation. vLLM manages this data in cache blocks, each containing storage slots for multiple tokens. The Scheduler assigns cache blocks to each request. Model execution uses `block_table` to specify the blocks to read and `slot_mapping` to specify each token's write slot.

KV Trace records request identifiers, cache block IDs, token positions, and read/write locations. It checks whether the locations specified in model inputs agree with the blocks assigned by the Scheduler. For example, when a token should use slot 80 but the model input specifies slot 96, the log identifies the request, rank, token index, and differing slots to help locate the issue.

## Preparations

**Environment Preparation**

Follow the [msProbe Installation Guide](../install_guide/msprobe_install_guide.md) to install msprobe in the Python environment used to start vLLM.

**Constraints**

- The feature supports vLLM Ascend in eager and graph modes.
- Use Model Runner V2 (MRV2), enabled with `VLLM_USE_V2_MODEL_RUNNER=1`. Model Runner V2 is a public vLLM component that prepares model inputs and executes inference. For details, see the official vLLM [Model Runner V2 Design Document](https://docs.vllm.ai/en/latest/design/model_runner_v2/).

## Function

**Function Description**

Record each scheduling step's requests, cache blocks, read/write locations, and results. Check the consistency of request identifiers, token counts, cache block IDs, and write slots. Inspect the service logs to locate failures by request, rank, and KV Cache group.

**Command Format**

```shell
vllm serve <model_path> --enforce-eager --additional-config '<additional_config>'
```

**Parameter Description**

Configure `enabled` and `action` in the `kv_trace` object passed through `--additional-config`.

| Parameter Name | Optional/Mandatory | Parameter Description |
| --- | --- | --- |
| `model_path` | Mandatory | Model path |
| `--additional-config` | Mandatory | JSON configuration containing the `kv_trace` object to enable this feature |
| `kv_trace.enabled` | Optional | Boolean: `true` enables recording and `false` disables it. KV Trace is disabled by default; when a `kv_trace` configuration is supplied, this field defaults to `true` |
| `kv_trace.action` | Optional | `log` (default): report check failures and continue inference; `raise`: report check failures and raise `KVGuardError`, causing the inference engine to fail |

**Usage Example**

When selecting plugins through `VLLM_PLUGINS`, add `msprobe_kv_guard` to the existing list. The following example selects only this plugin; run it before starting the service:

```shell
export VLLM_PLUGINS=msprobe_kv_guard
```

Separate plugin names with commas when enabling multiple plugins.

The following example enables KV Trace in eager mode and saves the service log as `server.log`:

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/model \
  --enforce-eager \
  --additional-config '{"kv_trace":{"enabled":true,"action":"log"}}' \
  > server.log 2>&1
```

Replace `/path/to/model` with the model path. For graph mode, remove `--enforce-eager` from the command above and use the service's existing compilation configuration.

Use `log` to collect context during investigation. Use `raise` to stop inference after a check fails. Enable tensor, statistics, and NaN dumping through their respective configurations.

**Output Description**

KV Trace writes to process stdout. The launch command above redirects the logs to `server.log`. For container deployments, search the container logs for `[KV_GUARD]`. For log formats and field meanings, see [Log Output Description](#log-output-description).

## Log Output Description

### Log Format and Example

Each record contains a log prefix, `[KV_GUARD]`, an event name, and JSON fields. Regular records use INFO; `kv.violation` and `kv.error` use ERROR.

The following illustrative example selects fields needed for analysis and omits prefixes such as timestamps and process IDs. Request `req-1` processes two tokens in this step using cache block 5:

```text
[KV_GUARD] kv.enabled {"rank":0,"action":"log","boundary":"worker.init_device.return"}
[KV_GUARD] kv.bind {"trace_id":"0123456789abcdef0123456789abcdef","evidence":"scheduler_plan","num_scheduled_tokens":{"req-1":2},"total_num_scheduled_tokens":2,"scheduled_new_reqs":[{"req_id":"req-1","block_ids":[[5]],"num_computed_tokens":0,"replay_start":0}]}
[KV_GUARD] kv.write {"trace_id":"0123456789abcdef0123456789abcdef","rank":0,"req_ids":["req-1"],"evidence":"model_runner_plan","group_id":0,"block_size":16,"kernel_block_size":16,"num_tokens":2,"query_start_loc":[0,2],"positions":[0,1],"slot_mapping_scope":"local_tokens","slot_mapping":[80,81]}
[KV_GUARD] kv.read {"trace_id":"0123456789abcdef0123456789abcdef","rank":0,"req_ids":["req-1"],"evidence":"model_runner_plan","group_id":0,"block_size":16,"kernel_block_size":16,"seq_lens":[2],"block_table":[[5]]}
[KV_GUARD] boundary {"trace_id":"0123456789abcdef0123456789abcdef","rank":0,"boundary":"worker.output","phase":"observed","evidence":"native_result","req_ids":["req-1"],"req_id_to_index":{"req-1":0},"num_output_tokens":{"req-1":1}}
```

### Events

| Event | Information to inspect |
| --- | --- |
| `kv.enabled` | Enablement, process rank, and check failure action |
| `kv.bind` | Scheduled requests, block bindings, token budgets, and scheduling information for completion, preemption, resumption, copying, and zeroing |
| `kv.write` | Write slots and token positions specified in model inputs for each KV Cache group; placeholder inputs, such as dummy execution, use `evidence=dummy_plan` |
| `kv.read` | Read block tables and sequence lengths specified in model inputs for each KV Cache group |
| `boundary` | Inputs and outputs at inference stages; `boundary`, `evidence`, and `phase` identify the context |
| `kv.violation` | A KV Trace check failure, with its type in `reason`, covering request associations, read/write locations, sampling counts, and placeholder inputs |
| `kv.error` | An inference exception, described by `error` and, when available, `error_type` |

### Key fields

| Field | Meaning and usage |
| --- | --- |
| `trace_id` | Correlation ID for a scheduling step, used to collect records from the Scheduler, execution processes (Workers), and read/write locations |
| `req_id` / `req_ids` / `request_id` | Request identifiers in scheduling entries, batch lists, and some failure records, respectively |
| `rank` | Logical ID of a distributed execution process, used to identify the process executing a request |
| `dp_rank` | Data parallel instance ID; each instance processes its own request batches |
| `group_id` | KV Cache group ID, starting from 0, used to distinguish cache groups in the model |
| `num_scheduled_tokens` | Per-request token budget for the current step |
| `query_start_loc` | Request boundaries in the batch token array. For example, `[0,2,5]` assigns `[0,2)` and `[2,5)` to two requests |
| `positions` / `seq_lens` | Token positions within request sequences and sequence lengths for this step |
| `block_ids` / `block_table` | Scheduled cache blocks and the blocks specified for reading in model inputs; inspect the corresponding cache group and request row |
| `slot_mapping` | Per-token write slot specified in model inputs; `-1` denotes a padding (PAD) slot |
| `block_size` | Tokens per cache block managed by the Scheduler; `block_ids` uses these block IDs |
| `kernel_block_size` | Tokens per cache block used for model reads; `block_table` uses these block IDs |
| `evidence` / `phase` | Record context, such as scheduling records (`scheduler_plan`), model execution inputs (`model_runner_plan`), dummy inputs (`dummy_plan`), or result observation (`observed`) |
| `coverage` | Field completeness for KV transfer records. `selected_public_fields` means available public fields were recorded; `not_expanded` provides only the metadata type and available request identifiers |

## Result Analysis

### 1. Confirm enablement and find failures

```shell
grep -F '[KV_GUARD] kv.enabled ' server.log
grep -E '\[KV_GUARD\] kv\.(violation|error) ' server.log
```

`kv.enabled` confirms enablement in the corresponding process. Start with `reason` for `kv.violation`; analyze `kv.error` using `error` and the service exception stack.

### 2. Collect context by step, request, and rank

Take the `trace_id` from the failure record and filter the step's logs:

```shell
grep -F '"trace_id":"0123456789abcdef0123456789abcdef"' server.log
```

Replace the example ID with the actual value, then locate related records using `request_id`, `req_ids`, `rank`, and `group_id`. To follow a request across steps, search for its identifier:

```shell
grep -F '"req-1"' server.log
```

Logs from multiple processes can interleave; organize records by their correlation fields. For a `null` `trace_id`, use the record's process, rank, `boundary`, and timestamp to inspect its context.

### 3. Compare scheduled blocks and read/write locations

Read records for the same step as `kv.bind → kv.write / kv.read → boundary`:

1. In `kv.bind`, identify the request, token budget, and `block_ids` for its KV Cache group. For continuing requests, append `scheduled_cached_reqs.new_block_ids` increments to the previous block list. For requests in `resumed_req_ids`, replace the previous list with the current step's block list.
2. In `kv.write`, use `req_ids` and `query_start_loc` to locate the request's token range, then compare `positions` and `slot_mapping`.
3. In `kv.read`, inspect `block_table` and `seq_lens` for the same request, rank, and group. When `block_size` equals `kernel_block_size`, compare block IDs directly; when they differ, first convert using the respective block sizes.
4. In result `boundary` records, inspect requests and output token counts. `phase=observed` at `worker.output` means the Worker has observed a result; assess the final service status together with check logs and exception stacks.

In the single-card example above, block size is 16 and positions 0 and 1 use block 5. Their slots are `5 × 16 + 0 = 80` and `5 × 16 + 1 = 81`. `seq_lens=[2]` gives a sequence length of 2 for this request in this step.

For a single-card case with different block sizes, such as `block_size=16` and `kernel_block_size=8`, scheduled block 5 corresponds to blocks 10 and 11 in `block_table`, covering slots 80–87 and 88–95. Compare `block_ids=[[5]]` with `block_table=[[10,11]]`; positions 0 and 1 still use write slots 80 and 81.

With a sliding window, which retains only recent context, `block_table` can use 0 for older prefix entries outside the window. Analyze these entries together with the current `positions` and check logs.

### 4. Locate differences using the failure type

If `slot_mapping` in the model input changes the first token's write slot from 80 to 96, the check can produce:

```text
[KV_GUARD] kv.violation {"reason":"write_target_mismatch","rank":0,"evidence":"boundary_plan","trace_id":"0123456789abcdef0123456789abcdef","request_id":"req-1","group_id":0,"token_index":0,"expected_slot":80,"actual_slot":96}
```

This identifies rank 0, group 0, and batch token index 0 belonging to `req-1`: the token should use slot 80, while the model input specifies slot 96. Compare the cache block IDs in `kv.bind` with the token positions and write slots in `kv.write` to locate the difference for this request.

| Example `reason` | Information to inspect |
| --- | --- |
| `write_target_mismatch` | `expected_slot`, `actual_slot`, `token_index`, and the corresponding block bindings and token positions |
| `read_block_mapping_mismatch` / `write_block_mapping_mismatch` | Scheduled cache block IDs and the block table in `kv.read` |
| `batch_request_mismatch` / `result_request_not_scheduled` | Associations between scheduled requests, execution batch requests, and result requests |
| `execution_range_mismatch` / `read_length_mismatch` | Token budgets, `query_start_loc`, `positions`, and `seq_lens` |
| `padded_write_target` / `dummy_write_target` | Write slots in padding or dummy records |

### 5. Inspect scenario-specific records

Data Parallel (DP) deployment uses multiple instances to process separate request batches. Dummy execution lets an instance participate in parallel computation with placeholder inputs. For these concepts, see the official vLLM [Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) guide.

Context parallelism distributes long-sequence computation or cache storage across execution processes. Prefill Context Parallel (PCP) distributes input processing; Decode Context Parallel (DCP) distributes storage of the KV Cache used during generation. For these concepts, see the official vLLM Ascend [Context Parallel Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html).

| Scenario | Analysis |
| --- | --- |
| Preemption, resumption, copying, zeroing | Inspect `preempted_req_ids`, `resumed_req_ids`, `kv_cache_block_copies`, and `new_block_ids_to_zero` in `kv.bind`, together with bindings from surrounding steps |
| DP dummy execution | In `evidence=dummy_plan` records, inspect `slot_count` (total slots) and `non_pad_slots` (`[index,value]` entries with PAD entries omitted). Ordinary dummy execution uses PAD slots; `allow_null_block=true` permits slots in the reserved placeholder block, with `0 <= slot < block_size` |
| Speculative decoding (candidate token generation and verification) | Compare `speculative` budgets in `kv.bind` with `num_sampled` (sampled count), `num_rejected` (rejected count), and `query_start_loc` in `evidence=sample_postprocess` records. `association=current_step` identifies current-step results; `observed_call_only` records just this call, with `trace_id=null` |
| PCP / DCP | Inspect the PCP/DCP fields across execution processes. `slot_mapping_scope=pcp_frame` describes the write slot list for the PCP group; compare it using global requests, token associations, and write flags (masks) in `evidence=parallel_context` records. For DCP, use `dcp_local_seq_lens` to inspect this process's read lengths |
| KV transfer | Inspect requests and local/remote cache block lists in `evidence=connector_plan` records. Under the result's `kv_connector_output`, first check `failed_recving` (requests with failed receives) and `invalid_block_ids` (invalid cache block IDs), then assess `finished_sending` (requests with completed sends) and `finished_recving` (requests with completed receives) |

## Interpreting Results

KV Trace checks the scheduling records, cache blocks and slots specified in model inputs, and inference results visible in the logs. Use it to locate consistency issues between these records. Physical cache block allocation and release, device memory access completion, and KV numerical correctness require additional runtime evidence or numerical comparisons.

`finished_req_ids` describes request completion information; copy, zero, and transfer fields describe scheduling arrangements. Combine scheduling records with execution results and service status to build the context for an investigation.
