# vLLM KV boundary tracing

KV Trace is loaded through vLLM's general plugin entry point. It records boundaries between the Scheduler, Worker, and Ascend Model Runner V2 and checks scheduling declarations against execution address plans. The configuration key is `kv_trace`. It does not reconstruct allocator state or validate KV values.

## Enable tracing

Install msprobe in the Python environment that starts vLLM so its entry point is discoverable. Setting `PYTHONPATH` alone does not register it.

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/model \
  --additional-config '{"kv_trace":{"enabled":true,"action":"log"}}'
```

Use `--enforce-eager` for eager execution, or retain the native compilation configuration for graph mode. No `dump_config`, `PrecisionDebugger`, or `AclGraphDumper` is required. The feature is disabled when `kv_trace` is absent or `enabled:false`. Accepted fields are boolean `enabled` and `action`, either `log` (default) or `raise`. If the native `VLLM_PLUGINS` allowlist is set, include `msprobe_kv_guard`.

`log` records `kv.violation` and continues native execution. `raise` throws `KVGuardError` at the scheduling exit, worker result boundary, or `_dummy_run` result exit and propagates through the native execution path, causing EngineCore to fail. Raw records are preserved before enforcement. Checks are not guaranteed to precede the current kernels and do not roll back device writes.

Tensor, statistics, and NaN dumping retain their own configuration and work independently.

## Stage 1: Core boundaries

```text
Scheduler.schedule -> SchedulerOutput + trace_id -> Worker.execute_model
  -> model_state.prepare_attn input plan -> worker observes output -> scheduler receives output
```

| Event | Boundary and meaning |
| --- | --- |
| `kv.bind` | Native new-request block tables, cached `new_block_ids` deltas, and scheduling fields, not an allocator mirror |
| `kv.write` / `kv.read` | Per-group slots, block tables, lengths, requests, and positions at the `model_state.prepare_attn` input |
| `boundary` | Worker input, observed output, scheduler result input, and later public boundaries; `boundary`, `evidence`, and `phase` identify the source |
| `kv.enabled` | Enabled process, rank, and action |
| `kv.error` | Native boundary exception, recorded and re-raised unchanged |
| `kv.violation` | Boundary-plan inconsistency, logged or used to block output according to `action` |

Only `kv.bind/write/read` describe KV operations. Enablement, native exceptions, and violations are diagnostics, not new lifecycle states. One `trace_id` travels with each scheduling object; it correlates records and is not a device-completion sequence. `kv.bind` includes `num_scheduled_tokens` and `total_num_scheduled_tokens`, without a separate scheduling event. Read/write records use `evidence=model_runner_plan` and include native `execution_mode`, `block_size`, and `kernel_block_size`. An entire block-table row describes input addresses, not proof that every entry was read by a kernel.

Worker result records use `phase=observed`: the native caller has observed the output, but the guard has not necessarily released it. The record may exist even when a subsequent `guard.check()` raises under `action=raise`. Tracing neither calls asynchronous `get_output()` early nor adds another call. Scheduler result-input records likewise do not prove processing or device completion; no duplicate `output_processed` event is emitted.

Device address descriptions are copied asynchronously to host buffers. The previous snapshot is completed before reusing buffers; records and checks run at native output retrieval or the next plan boundary. The scheduler retains only a request-to-block-table map and sends an immutable current-step plan with its native output: new/resumed requests replace bindings, cached deltas append, and finished/preempted requests remove bindings. This compares device plans with their scheduling source rather than merely checking device-table self-consistency.

Checks cover request association, token counts and declared lengths, block routing, and write slots. Scheduled blocks used by current writes must be within native `kv_cache_config.num_blocks`; ordinary requests must not write null block 0. Only the asynchronous snapshot and first failure reason are retained. There is no physical-allocation epoch, separate owners index, valid-prefix state, or per-layer Attention hook.

Native zero holes in retired sliding-window prefixes are allowed. Table columns used by current writes are checked strictly; other nonzero routes are compared with scheduling bindings. This does not detect every zero-hole fault or prove initialized reads. Speculative/adaptive compaction may leave an active query smaller than the `num_tokens` buffer budget; the unused slot tail must be PAD, and stale position-buffer entries are not treated as actual writes.

## Stage 2: Local special-path boundaries

Native `finished_req_ids`, `preempted_req_ids`, cached `resumed_req_ids`, `new_block_ids_to_zero`, `kv_cache_block_copies`, and speculative controls are included in `kv.bind`, with `evidence=scheduler_plan`, without adding KV operation states. Finished/preempted requests do not imply physical block release. Copies retain native source, destination, and other fields; zero/copy plans do not prove device completion. Speculative controls retain `draft_counts`, `num_spec_tokens_to_schedule`, `skip_drafts`, and `num_invalid_spec_tokens`, without candidate token values.

Computed prefixes and `replay_start` retain native values without inferring prefix hits or recomputation. Ordinary `replay_start=0` is not labeled a recomputation. Cached deltas append and resumed requests replace bindings; CoW does not globally replace source blocks with destinations.

Native dummy calls, including DP accompaniment, produce `kv.write` at the `model_state.prepare_attn` input with `evidence=dummy_plan` and `phase=dummy`. Snapshots include both the supplied slot frame and the complete native persistent slot buffer, including padding. `slot_count` and `non_pad_slots` retain the original indices and values; omitted entries are PAD `-1`. Ordinary dummy writes must all be `-1`, otherwise `dummy_write_target` is recorded. The native `redirect_writes_to_null_block` flag is read at each boundary; only an explicitly enabled flag permits slots in the group's null block, `0 <= slot < block_size`. FULL serving-time dummy calls are checked even when `for_capture=True`; CPP profiling is skipped separately.

`runner._dummy_run` receives an independent `trace_id`. Its return boundary completes pending snapshots and enforces `action`, without using the previous real request's scheduling plan. Initialization graph capture outside this function may produce `trace_id=null`; native warmup without a real scheduling trace is not compared with an old plan. These checks concern slot write plans, not Mamba state tables, metadata derived inside backends, actual contents, or performance.

The native `runner.postprocess_sampled` return uses an ordinary `boundary` record for `idx_mapping`, `num_sampled`, `num_rejected`, and query bounds, with `evidence=sample_postprocess`. Token values are not exported, and accepted-state or valid-prefix transitions are not inferred. Ordinary non-speculative generation adds no count copies.

Checks cover count shapes, nonnegative counts, and sampling capacity. When query bounds are available, only the prefix for actual requests is checked, including rejected counts; additional FULL-graph padding bounds are allowed. Raw records retain the complete query-bound array. Only the last PP rank compares current-step query counts with scheduling budgets. Sampled plus rejected counts are not required to equal the budget. Native masked rows with negative indices are skipped.

On non-last PP ranks, postprocessing may belong to an older step. Records retain `trace_id=null`, `observed_trace_id`, and `association=observed_call_only`, without inventing the original step association.

## Stage 3: CP and connector boundaries

`kv.write/read` include native PCP/DCP rank, size, and interleave. DCP read descriptions retain `dcp_local_seq_lens`. PCP writes retain the full native slot frame with `slot_mapping_scope=pcp_frame`; the frame must not be paired elementwise with only the local request list.

The native `pcp_manager.build_attention_context` return uses an ordinary `boundary` record for global requests/positions, global tables/slots, local-to-global request/token indices, and the native write mask. It shares the input plan's `trace_id`, without constructing another context, address projection, or collective. Single-card execution adds no such snapshot.

Checks compare global plans with scheduling bindings, local token/request mappings, and the complete write frame. Native `padded_gather_idx` is retained because this rank's token indices cannot describe the full frame. The returned gather indices and write mask are used to compare global slots without reconstructing PCP partitioning. Repeated request rows, empty ranks, and FULL-graph placeholder queries are allowed. Local DCP lengths are checked using the native CP coordinates at this input boundary.

Scheduler output and worker input use ordinary `boundary` records for the same native connector metadata, with `evidence=connector_plan`. Public `requests` / `requests_to_send` interfaces provide selected local/remote blocks, computed/external token counts, remote engine/request IDs, and sending request IDs. Mooncake exposes these fields. Unknown types use `coverage=not_expanded`; projections use `selected_public_fields`, and requests that cannot be expanded appear in `unexpanded_req_ids`.

Native results retain `finished_sending`, `finished_recving`, `failed_recving`, and `invalid_block_ids` under `kv_connector_output`. A failed receive may also appear as finished; finished alone does not establish success. Native null values are preserved rather than converted into fabricated completions. Tracing never additionally calls the destructive `get_finished/get_transfer_results/get_block_ids_with_load_errors` interfaces.

These records describe plans and native reported results, not transfer-thread submission, valid remote contents, or completion of every stream. No model tokens, KV values, internal thread objects, or custom transfer protocol are captured. Unsupported log field types are omitted without interrupting native inference.

## Logs and scope

Use `trace_id` to identify a scheduling step, then compare `kv.bind -> kv.write/read` and ordinary boundaries by request, rank, and group. Stdout line order alone does not establish device execution order. There are no actual `alloc/release` records; finished requests do not imply physical free.

Checks that require unknown coordinates or unavailable native mappings are skipped without a coverage-gap record or failure. Only `kv.violation` denotes a check failure and triggers the configured `log/raise` action; native exceptions still produce `kv.error` and propagate normally.

The msprobe logger writes to process stdout. A launch script or container logging system can collect it into `server.log`; no separate KV Trace file is created. Each line contains `[KV_GUARD]`, an event name, and JSON fields. Normal records use INFO and native exceptions use ERROR.

The boundary contract references vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58` and vllm-ascend `28e9e97ba254b076a88275e8a275556cb9a11664`. The entry point adapts Ascend tool-key validation and spawn ordering entirely inside msprobe, without changing the original `additional_config`.

Checks concern visible declarations and address plans. They do not prove device accesses, correct KV contents, physical release, or stale references after reuse of the same block ID. Prefix, copy, zero, and remote-load plans do not prove content validity. Internal events without native boundary evidence are not fabricated; physical-allocation epoch/owners/valid_end guarantees are not provided.
