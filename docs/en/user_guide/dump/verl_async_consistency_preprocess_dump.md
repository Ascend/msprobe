# Collecting Data for Asynchronous verl Training-Inference Consistency Comparison

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:43:16.180Z pushedAt=2026-08-11T02:51:07.161Z -->

## Introduction

*[Collecting Data for Verifying Data Consistency Between verl Training and Inference Based on FSDP](./verl_fsdp_consistency_preprocess_dump.md)* and *[Collecting Data for Verifying Data Consistency Between verl Training and Inference Based on Megatron](./verl_megatron_consistency_preprocess_dump.md)* were originally designed for the Single Program Multiple Data (SPMD) rollout architecture in verl versions below v0.7.0. Starting from v0.7, verl has removed SPMD mode from its main repository and fully transitioned to an asynchronous scheduling architecture, requiring corresponding changes to the collection scheme. Currently, verl has two asynchronous rollout modes:

| Mode                         | Resource Allocation                                       | Architecture                                                                                   |
| ---------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Hybrid AgentLoop** (default) | Inference and training share the same NPU group (`hybrid_engine`) | `LLMServerClient` + `GlobalRequestLoadBalancer`, synchronously scheduled via `AgentLoopManager` |
| **Fully Async**              | Rollouter and Trainer each have independent NPU pools, fully decoupled | `FullyAsyncRollouter` + `MessageQueue` + `FullyAsyncTrainer` + `ParameterSynchronizer`         |

This document uses verl v0.8.0 as an example to demonstrate how to collect data for training-inference consistency comparison in an asynchronous way.

## Preparations

### Basic Configuration

First refer to *[Collecting Data for Verifying Data Consistency Between verl Training and Inference Based on FSDP](./verl_fsdp_consistency_preprocess_dump.md)* or *[Collecting Data for Verifying Data Consistency Between verl Training and Inference Based on Megatron](./verl_megatron_consistency_preprocess_dump.md)*, and make preparations based on the actual training backend.

In addition, the following adjustments are required for the current scenario:

- In the current asynchronous rollout mode, to enable the vLLM dump function, you need to add the `dump_config_path` parameter in vLLM's `additional_config`, pointing to the msProbe inference-side configuration file. `/home/config_generate.json` is an example path. In actual deployment, modify it according to the actual configuration file path.
- On the training side, disable `val_before_train` to prevent the pre-training validation from calling the `generate_sequence` interface, which would interfere with the dump results.

```diff
export DUMP_ON=1              # Enable training-side msProbe collection
export PROMPTS_ONLY=1         # Only compute the prompt part (required; consistency only supports prefill)

# The entry point is main_ppo
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
+   actor_rollout_ref.rollout.enforce_eager=True \
+   '+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config={dump_config_path:"/home/config_generate.json"}' \
+   trainer.val_before_train=False \
    trainer.balance_batch=False \
```

### Fully Async

In Fully Async mode, the Rollouter and Trainer each have independent NPU pools and are decoupled through `MessageQueue` and `ParameterSynchronizer`. The pre-configuration for training-inference consistency collection is almost the same as in Hybrid AgentLoop, with the differences being the startup entry point and the need to disable bypass mode:

```diff
export DUMP_ON=1              # Enable training-side msProbe collection
export PROMPTS_ONLY=1         # Only compute the prompt part (required; consistency only supports prefill)
export TORCHDYNAMO_DISABLE=1  # Disable torchdynamo

# The entry point is fully_async_main
python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_batch_size=0 \
    data.shuffle=False \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
+   actor_rollout_ref.rollout.enforce_eager=True \
+   algorithm.rollout_correction.bypass_mode=False \
+   algorithm.rollout_correction=null \
+   '+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config={dump_config_path:"/home/config_generate.json"}' \
+   trainer.val_before_train=False \
```

## msProbe Configuration Files

Both the inference side and the training side need to provide their own msProbe configuration files. Refer to [config_json_introduct.md](./config_json_introduct.md) and specify them as follows:

- **Inference side**: Pass the file to the vLLM worker via `dump_config_path` in `additional_config`.
- **Training side**: Hard-code `config_path` in `_ensure_debugger()` of `transformer_impl.py`.

### Inference-Side Configuration (`config_generate.json`)

```json
{
  "task": "statistics",
  "dump_path": "/dump_data/generate_sequence",
  "rank": [],
  "step": [],
  "level": "L0",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

### Training-Side Configuration (`config_actor.json`)

```json
{
  "task": "statistics",
  "dump_path": "/dump_data/update_actor",
  "rank": [],
  "step": [],
  "level": "L0",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

## Code Changes

### File Modification

| File                                                               | Modification Type | Description                                                              | Corresponding Section                                                |
| ------------------------------------------------------------------ | ----------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| `vllm_ascend/worker/dispatch_logger.py`                | New           | Inference dispatch logging (`DispatchLogger`)                              | [Inference Side: Dispatch Logging](#inference-side-dispatch-logging)             |
| `vllm_ascend/worker/model_runner_v1.py`                | Modified          | Add `DispatchLogger` initialization + 4 `log_step` calls                     | [Inference Side: Collecting During vLLM Model Execution](#inference-side-collecting-during-vllm-model-execution)   |
| `verl/workers/engine/fsdp/transformer_impl.py`                | Modified          | FSDP backend: Add training-side debugger + `micro_batch` `request_id` log    | [Training Side: FSDP Backend](#fsdp)                         |
| `verl/workers/engine/megatron/transformer_impl.py`            | Modified          | Megatron backend: Add training-side debugger + `forward_step` `request_id` log | [Training Side: Megatron Backend](#megatron)                 |
| `verl/workers/rollout/llm_server.py`                          | Modified          | `request_id` injected into `extra_fields` in `LLMServerClient`              | [Request ID](#request-id)             |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | Modified        | `FullyAsyncLLMServerClient` inherits the parent class's `extra_fields` and passes through `request_id`| [Request ID](#request-id)             |
| `verl/trainer/ppo/ray_trainer.py`                             | Modified          | PROMPTS_ONLY mode (Hybrid AgentLoop)                                     | [Training Side: Compute-Only Prompt Part](#training-side-compute-only-prompt-part) |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | Modified          | PROMPTS_ONLY mode (Fully Async)                                          | [Training Side: Compute-Only Prompt Part](#training-side-compute-only-prompt-part) |

### Inference Side: Collecting During vLLM Model Execution

**File**: `vllm_ascend/worker/model_runner_v1.py`

**Note**: The reading of `dump_cfg`, initialization of `PrecisionDebugger`, and calls to `debugger.start/stop/step` are all existing upstream logic in vllm-ascend. On top of this, the current solution only adds **DispatchLogger** initialization and **log_step** calls.

Changes added in `__init__`:
Initialize `DispatchLogger`, point the dump path to a PID subdirectory, and record the distributed rank of the current process.

```diff
class NPUModelRunner(GPUModelRunner):
    def __init__(self, ...):
        ...
        dump_cfg = self.ascend_config.dump_config_path
        self.debugger = None
        if dump_cfg is not None:
            if self.model_config.enforce_eager:
                from msprobe.pytorch import PrecisionDebugger
                self.debugger = PrecisionDebugger(dump_cfg)
+               import os
+               from vllm_ascend.worker.dispatch_logger import DispatchLogger
+               self.debugger.service.config.dump_path = os.path.join(
+                   self.debugger.config.dump_path, f'{os.getpid()}')
+               self._dispatch_logger = DispatchLogger(
+                   dump_path=self.debugger.config.dump_path,
+                   pid=os.getpid(),
+                   rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
+               )
            else:
                raise RuntimeError(
                    "Dumping/debugging only works in eager mode.")
+       if not hasattr(self, "_dispatch_logger") or self._dispatch_logger is None:
+           self._dispatch_logger = None
```

Changes added at each return point in `execute_model`:

The `execute_model()` method contains multiple `self.debugger.stop()` calls (distributed across different return paths). Before each `self.debugger.stop()` call, a `self._dispatch_logger.log_step(...)` call must be inserted to cover all return branches.

After each model forward pass completes, call `DispatchLogger.log_step()` to record the scheduling information for that step (including the requests involved, the number of tokens allocated to each request, and the scheduling status of each request during the prefill and decode phases), then execute msProbe's `stop`/`step` to complete the current round of tensor dump. Globally search for `self.debugger.stop()` in `model_runner_v1.py` and insert the `log_step` call. For example:

```diff
    def execute_model(self, ...):
        ...
        if self.debugger is not None:
+           if self._dispatch_logger is not None and not self.debugger.service.should_stop_service:
+               self._dispatch_logger.log_step(scheduler_output, self.attn_state)
            self.debugger.stop()
            self.debugger.step()
        return output
```

### Inference Side: Dispatch Logging

**File**: `vllm_ascend/worker/dispatch_logger.py` (Create this file in the directory.)

**Function**: Upon each `execute_model` call, record the dispatch metadata of that step (step sequence number, phase, all request_ids dispatched in that step and the number of tokens allocated to each), and write to `dispatch_log.jsonl`. Each JSONL record contains fields such as `pid`, `rank`, `step`, `phase`, and `requests[]`, which are used for subsequent correlation with msProbe's `step_N/dump.json` and the training side's `update_actor_log.jsonl`.

```python
import json
import time
from pathlib import Path


class DispatchLogger:
    """Records which requests are scheduled at each execute_model step.

    One line per ``execute_model()`` call, written alongside the msprobe
    ``generate_sequence`` dump so that dispatch records can be correlated
    with ``generate_sequence/step{N}`` through the shared ``request_id``.

    Output file: ``{dump_path}/{pid}/dispatch_log.jsonl``
    """

    def __init__(self, dump_path: str, pid: int, rank: int = 0):
        log_dir = Path(dump_path) / str(pid)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(log_dir / "dispatch_log.jsonl", "w")
        self._step_counter = 0
        self._pid = pid
        self._rank = rank

    def log_step(self, scheduler_output, attn_state) -> None:
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        is_prefill = attn_state != AscendAttentionState.DecodeOnly

        requests = []
        for req in scheduler_output.scheduled_new_reqs:
            requests.append({
                "request_id": req.req_id,
                "type": "new",
                "tokens": scheduler_output.num_scheduled_tokens.get(req.req_id, 0),
            })
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            requests.append({
                "request_id": req_id,
                "type": "cached",
                "tokens": scheduler_output.num_scheduled_tokens.get(req_id, 0),
            })

        record = {
            "source": "dispatch_logger",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": self._pid,
            "rank": self._rank,
            "step": self._step_counter,
            "phase": "prefill" if is_prefill else "decode",
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "requests": requests,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()
        self._step_counter += 1

    def close(self) -> None:
        if self._fp and not self._fp.closed:
            self._fp.close()
```

Output example:

```json
{"source":"dispatch_logger","timestamp":"2026-05-13T10:00:00","pid":3680237,"rank":0,"step":0,"phase":"prefill","total_num_scheduled_tokens":95,"requests":[{"request_id":"f1f254c04e0c443b85ea1e7359e842dc","type":"new","tokens":95}]}
```

**Note:** For vLLM ≥ v0.14.0, an 8-character random suffix is appended to the externally passed `request_id`, producing the format `{original_request_id}-{8hex}`, for example `f1f254c04e0c443b85ea1e7359e842dc-12345678`. When selecting, the suffix must be stripped, using `f1f254c04e0c443b85ea1e7359e842dc` to match the training-side `request_id`.

### Training Side: Collecting During Model Execution

**Files involved**:

| Backend     | Class Name                         | File                                               |
| -------- | ---------------------------- | -------------------------------------------------- |
| FSDP     | `FSDPEngine`     | `verl/workers/engine/fsdp/transformer_impl.py`     |
| Megatron | `MegatronEngine` and `MegatronEngineWithLMHead` | `verl/workers/engine/megatron/transformer_impl.py` |

The `forward_backward_batch` architectures of FSDP and Megatron differ, so they need to be handled separately.

---

#### FSDP

FSDP's `forward_backward_batch` has an explicit `for i, micro_batch in enumerate(micro_batches)` loop (`fsdp/transformer_impl.py`), allowing debugger calls to be wrapped directly inside the loop body. Make modifications in `verl/workers/engine/fsdp/transformer_impl.py`.

1. `__init__` (`FSDPEngine` class)

    ```diff
    class FSDPEngine(BaseEngine):
        def __init__(self, ...):
            ...
            self.mode = None
    +       self._debugger = None
    +       self._update_actor_logger_fp = None
            self.rank = torch.distributed.get_rank()
            ...
    ```

2. `forward_backward_batch` (`FSDPEngine` class)

    ```diff
    class FSDPEngine(BaseEngine):
        ...
        def forward_backward_batch(self, ...):
            ...
            scaler = getattr(self, "scaler", None)
    +       self._ensure_debugger()
    +       dump_phase = os.environ.get("DUMP_PHASE", "log_prob")  # "all" | "log_prob" | "update_actor"
    +       phase = "log_prob" if forward_only else "update_actor"
    +       should_dump = dump_phase == "all" or dump_phase == phase

            for micro_batch in micro_batches:
    +           if self._debugger is not None and should_dump:
    +               self._debugger.start(model=self.module)
                with ctx:
                    loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)

                    if not forward_only:
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
    +           if self._debugger is not None and should_dump:
    +               if not self._debugger.service.should_stop_service:
    +                   self._log_update_actor_step(micro_batch)
    +               self._debugger.stop()
    +               self._debugger.step()
                output_lst.append(meta_info)
            ...
    ```

#### Megatron

Megatron's `forward_backward_batch` does not have an explicit micro_batch loop — it hands all micro_batches to the Megatron scheduler via `forward_backward_func()` for unified execution. Internally, the scheduler calls `forward_step` once for each micro_batch it processes, so injecting the debugger's `start`/`stop`/`step` directly inside `forward_step` achieves per-micro-batch collection. Make modifications in `verl/workers/engine/megatron/transformer_impl.py`.

1. `__init__` (`MegatronEngine` class)

    ```diff
    class MegatronEngine(BaseEngine):
        def __init__(self, ...):
            ...
            self.mode = None
    +       self._debugger = None
    +       self._update_actor_logger_fp = None
    +       self._should_dump = False
    +       self.rank = torch.distributed.get_rank()
            ...
    ```

2. `forward_backward_batch` (inserted after the `forward_step` declaration in the `MegatronEngine` class)

    `_ensure_debugger()` is responsible for lazy initialization of the debugger, and `self._should_dump` serves as a switch for `forward_step` to determine whether to perform collection.

    ```diff
    class MegatronEngine(BaseEngine):
        ...
        def forward_backward_batch(self, ...):
            ...
            forward_step = partial(
                self.forward_step,
                logits_processor_func=loss_function,
                postprocess_micro_batch_func=postprocess_micro_batch_func,
            )
    +       self._ensure_debugger()
    +       dump_phase = os.environ.get("DUMP_PHASE", "log_prob")  # "all" | "log_prob" | "update_actor"
    +       phase = "log_prob" if forward_only else "update_actor"
    +       self._should_dump = self._debugger is not None and (dump_phase == "all" or dump_phase == phase)
            enable_routing_replay = ...
    ```

3. `forward_step` (`MegatronEngineWithLMHead` class)

    Insert `debugger.start` before the `forward_fn` call; insert `debugger.stop/step/_log_update_actor_step` after the call.

    ```diff
    class MegatronEngineWithLMHead(MegatronEngine):
        ...
        def forward_step(
            self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func
        ):
            ...
            if use_fused_kernels:
                from verl.models.mcore import get_mcore_forward_fused_model_engine_fn
                fused_forward_fn = get_mcore_forward_fused_model_engine_fn(self.model_config.hf_config)
            else:
                ...
                forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
                ...
                logits_processor_args = {
                    "label": label,
                    "temperature": temperature,
                    "loss_mask": loss_mask,
                    "response_attention_mask": response_attention_mask,
                }
    +           if self._should_dump:
    +               self._debugger.start(model=model)
                output = forward_fn(
                    model,
                    input_ids,
                    multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                    pad_token_id=self.model_config.tokenizer.pad_token_id,
                    data_format=data_format,
                    mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                    local_cp_size=local_cp_size,
                )
    +           if self._should_dump:
    +               if not self._debugger.service.should_stop_service:
    +                   self._log_update_actor_step(batch)
    +               self._debugger.stop()
    +               self._debugger.step()
                ...
    ```

#### Auxiliary Methods (FSDP Backend and Megatron Backend)

(Added at the end of the `FSDPEngine`/`MegatronEngine` class; the additions are identical for both backends.)

**Note**: `/home/config_actor.json` is an example path. In actual deployment, use the actual configuration file path.

```python
    def _ensure_debugger(self):
        """Lazy init debugger and logger on first ``forward_backward_batch`` call.
        Only the actor engine creates the debugger; ref engine (forward_only=True) skips.
        """
        if self._debugger is not None:
            return
        if self.engine_config.forward_only:
            return
        dump_flag = int(os.environ.get("DUMP_ON", 0))
        if not dump_flag:
            return
        from pathlib import Path

        from msprobe.pytorch import PrecisionDebugger, seed_all
        seed_all(mode=True)
        self._debugger = PrecisionDebugger(
            config_path="/home/config_actor.json")
        try:
            dump_path = self._debugger.config.dump_path
            log_dir = Path(dump_path) / str(os.getpid())
            log_dir.mkdir(parents=True, exist_ok=True)
            self._update_actor_logger_fp = open(
                log_dir / "update_actor_log.jsonl", "a")
        except Exception as e:
            logger.warning(f"Failed to initialize update_actor_logger: {e}")

    def _log_update_actor_step(self, micro_batch: TensorDict) -> None:
        """Extract request_ids from micro_batch and write one line to update_actor_log.jsonl."""
        if self._update_actor_logger_fp is None:
            return
        try:
            req_data = tu.get(micro_batch, key="request_id", default=None)
            if not req_data:
                request_ids = []
            elif isinstance(req_data, list):
                request_ids = [str(r) for r in req_data]
            else:
                request_ids = [str(req_data)]
        except Exception:
            request_ids = []

        import json
        import time

        record = {
            "source": "update_actor",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": os.getpid(),
            "rank": self.rank,
            "step": self._debugger.service.current_iter,
            "request_ids": request_ids,
            "num_requests": len(request_ids),
        }
        self._update_actor_logger_fp.write(
            json.dumps(record, ensure_ascii=False) + "\n")
        self._update_actor_logger_fp.flush()
```

Output example:

```json
{"source":"update_actor","timestamp":"2026-05-13T10:00:01","pid":3665398,"rank":0,"step":0,"request_ids":["f1f254c04e0c443b85ea1e7359e842dc"],"num_requests":1}
```

### Training Side: Compute Only Prompt Part

**Function**: Trim the response tokens from the training data returned by rollout, so that the forward pass during training includes only the prompt prefill part. Training-inference consistency comparison currently supports only the prefill part. This modification ensures that the tensors collected by msProbe on the training side are equivalent in computation content to the tensors of the prefill step on the inference side. This is controlled by the `PROMPTS_ONLY=1` environment variable.

The modification locations differ between the two asynchronous modes.

#### Hybrid AgentLoop

**File**: `verl/trainer/ppo/ray_trainer.py`

**Method**: `RayPPOTrainer.fit()`

**Insertion position**: Search for `bypass_recomputing_logprobs` and insert before it.

```diff
class RayPPOTrainer:
    ...
    def fit(self):
        ...
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                ...
                with marked_timer("step", timing_raw):
                    ...
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
+                   compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
+                   if compute_prompts_only:
+                       def get_prompts_only_batch(data: DataProto):
+                           responses_len = data.batch["responses"].size(1)
+                           data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
+                           data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
+                           if data.batch["position_ids"].dim() == 3:
+                               data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
+                           else:
+                               data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
+                           data.batch["responses"] = data.batch["responses"][:, :0]
+                           if "rollout_log_probs" in data.batch:
+                               data.batch["rollout_log_probs"] = data.batch["rollout_log_probs"][:, :0]
+                           if "response_mask" in data.batch:
+                               data.batch["response_mask"] = data.batch["response_mask"][:, :0]
+                           return data
+                       batch = get_prompts_only_batch(batch)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        ...
```

#### Fully Async

**File**: `verl/experimental/fully_async_policy/fully_async_trainer.py`

**Method**: `FullyAsyncTrainer._fit_generate()`

**Insertion Position**: After `_get_samples_from_queue()` returns `batch` and before `batch.meta_info["temperature"]` assignment:

```diff
class FullyAsyncTrainer:
    ...
    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
+       compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
+       if compute_prompts_only:
+           if "responses" in batch.batch and batch.batch["responses"] is not None:
+               responses_len = batch.batch["responses"].size(1)
+               batch.batch["input_ids"] = batch.batch["input_ids"][:, :-responses_len]
+               batch.batch["attention_mask"] = batch.batch["attention_mask"][:, :-responses_len]
+               if batch.batch["position_ids"].dim() == 3:
+                   batch.batch["position_ids"] = batch.batch["position_ids"][:, :, :-responses_len]
+               else:
+                   batch.batch["position_ids"] = batch.batch["position_ids"][:, :-responses_len]
+               batch.batch["responses"] = batch.batch["responses"][:, :0]
+               if "rollout_log_probs" in batch.batch:
+                   batch.batch["rollout_log_probs"] = batch.batch["rollout_log_probs"][:, :0]
+               if "response_mask" in batch.batch:
+                   batch.batch["response_mask"] = batch.batch["response_mask"][:, :0]
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        return batch
```

### Request ID

**Files Involved**:

| File | Description |
| ---- | ---- |
| `verl/workers/rollout/llm_server.py` | Inject `request_id` into `extra_fields` in `LLMServerClient` |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | `FullyAsyncLLMServerClient` inherits the parent class's  `extra_fields` and passes through `request_id` |

**Function**: Inject the `request_id` used internally by vLLM into `TokenOutput.extra_fields`, allowing it to propagate automatically through the verl data flow all the way to the training-side micro_batch. This enables precise correlation between inference dispatch records (`dispatch_log.jsonl`) and training micro_batch records (`update_actor_log.jsonl`) via `request_id`.

#### 1. LLMServerClient: Inject request_id

`LLMServerClient.generate()` generates `vllm_request_id` and uses it as both the vLLM `request_id` and the `request_id` field in `extra_fields`:

```diff
class LLMServerClient:
    ...
    @rollout_trace_op
    async def generate(
        self,
        ...
    ) -> TokenOutput:
        ...
        server_id, server = await self._acquire_server(request_id)
        try:
            ...
+           vllm_request_id = uuid4().hex
            output: TokenOutput = await server.generate.remote(
-                request_id=uuid4().hex,  # Use new request_id for each turn
+               request_id=vllm_request_id,  # Use new request_id for each turn
                ...
            )
+           output.extra_fields["request_id"] = vllm_request_id
            return output
```

#### 2. FullyAsyncLLMServerClient: Passthrough extra_fields

In Fully Async mode, `FullyAsyncLLMServerClient` inherits from `LLMServerClient` and calls the parent class via `super().generate()` to obtain the `TokenOutput` with `request_id` already injected. Since `FullyAsyncLLMServerClient` supports **partial rollout** (multi-round resume), a new `final_output` needs to be created to accumulate the results of multiple `super().generate()` calls. The new `final_output` must inherit the complete `extra_fields` returned by the parent class to ensure that fields such as `request_id` continue to be passed downstream:

```diff
class FullyAsyncLLMServerClient(LLMServerClient):
    ...
    @rollout_trace_op
    async def generate(
        self,
        ...
    ) -> TokenOutput:
        ...
+       final_output.extra_fields.update(output.extra_fields)  # Inherit all extra_fields (including request_id) of the parent class
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output
```

`request_id` automatically runs through the following pipeline:

```plain
vLLM Server (request_id)
  → TokenOutput.extra_fields["request_id"]
    → AgentLoopOutput.extra_fields
      → _InternalAgentLoopOutput.extra_fields
        → DataProto.non_tensor_batch["request_id"]
          → XXXEngine micro_batch → update_actor_log.jsonl
```

## Dump Result Files

After training is completed, the following files are generated under the dump path:

```plain
{dump_generate_path}/
└── {pid}/
    ├── step_0/
    │   └── rank_0/dump.json
    ├── step_1/
    │   └── rank_0/dump.json
    └── dispatch_log.jsonl

{dump_actor_path}/
├── step_0/
│   └── rank_0/dump.json
├── step_1/
│   └── rank_0/dump.json
└── {pid}/
    └── update_actor_log.jsonl
```

File description:

| File                                                 | Content                                    | Granularity                              |
| ---------------------------------------------------- | ------------------------------------------ | ---------------------------------------- |
| `{dump_generate_path}/{pid}/step_N/rank_M/dump.json` | vLLM forward tensor statistics            | Per `execute_model` invocation           |
| `{dump_actor_path}/step_N/rank_M/dump.json`          | Training forward + backward tensor statistics | Per `micro_batch`                       |
| `{dump_generate_path}/{pid}/dispatch_log.jsonl`      | vLLM scheduling information               | One line per `execute_model` invocation  |
| `{dump_actor_path}/{pid}/update_actor_log.jsonl`     | Training request_id record                | One line per `micro_batch`                 |

## Data Association Method

Through the `request_id` injected in the pipeline as described in [Request ID](#request-id), associate the [inference-side collecting](#inference-side-collecting-during-vllm-model-execution) (`dispatch_log.jsonl` + `step_N/dump.json`) with the [training-side collecting](#training-side-collecting-during-model-execution) (`update_actor_log.jsonl` + `step_N/dump.json`), thereby supporting training-inference consistency comparison (see *[Precision Comparison in PyTorch](../accuracy_compare/pytorch_accuracy_compare_instruct.md#verl-training-and-inference-consistency-comparison)*). The specific steps are as follows:

### How to Do

1. **Select an inference step**: Find a suitable `step` and `request_id` in `dispatch_log.jsonl` (where `phase` is `prefill` and the number of `requests` is 1). Note that vLLM ≥ v0.14.0 appends an 8-character random suffix to the externally passed `request_id` in the format of `{original_request_id}-{8hex}`. When selecting, you must remove the suffix to match the `request_id` on the training side.
2. **Locate the training step**: Search for the same `request_id` in `update_actor_log.jsonl` to find the target `step` and `rank`.
3. **Read the dump data**: Read the corresponding `dump.json` based on the step sequence number and `rank` sequence number.
4. **Perform training-inference consistency comparison**.

### JSON Field Specification

All JSONL logs share the following top-level fields:

| Field        | Type   | Description                            |
| ----------- | ------ | -------------------------------------- |
| `source`    | string | `"dispatch_logger"`/`"update_actor"` |
| `timestamp` | string | ISO 8601 timestamp                     |
| `pid`       | int    | Process ID                             |

Fields specific to `dispatch_log.jsonl`:

| Field                         | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `step`                       | Step sequence number of `execute_model`           |
| `rank`                       | Distributed rank                               |
| `phase`                      | `"prefill"`/`"decode"`                       |
| `total_num_scheduled_tokens` | Total number of tokens scheduled in this step  |
| `requests[].request_id`      | vLLM's internal `request_id`                      |
| `requests[].type`            | `"new"`/`"cached"`                           |
| `requests[].tokens`          | Number of allocated tokens                     |

Specific to `update_actor_log.jsonl`:

| Field            | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `step`           | Step sequence number of `micro_batch`                     |
| `rank`           | Distributed rank                                     |
| `request_ids[]`  | `request_id` contained in this `micro_batch`             |
| `num_requests`   | Number of requests (data count in `micro_batch`, should be 1) |
