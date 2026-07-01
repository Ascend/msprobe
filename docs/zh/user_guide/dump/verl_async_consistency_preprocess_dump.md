# 异步架构verl训推一致性比对数据采集

## 简介

《[fsdp训练后端verl训推一致性比对数据采集](./verl_fsdp_consistency_preprocess_dump.md)》和 《[megatron训练后端verl训推一致性比对数据采集](./verl_megatron_consistency_preprocess_dump.md)》最初针对 verl v0.7.0 以下版本的 SPMD (Single Program Multiple Data) rollout 架构设计。自 v0.7 起，verl 主仓移除了 SPMD 模式，全面转向异步调度架构，采集方案需要随之变动。当前 verl 有两种异步 rollout 模式：

| 模式                         | 资源分配                                       | 架构                                                                                   |
| ---------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Hybrid AgentLoop**（默认） | 推理和训练共享同一组 NPU（`hybrid_engine`）    | `LLMServerClient` + `GlobalRequestLoadBalancer`，通过 `AgentLoopManager` 同步调度      |
| **Fully Async**              | Rollouter 和 Trainer 各自独立 NPU 池，完全解耦 | `FullyAsyncRollouter` + `MessageQueue` + `FullyAsyncTrainer` + `ParameterSynchronizer` |

本文针对verl的上述异步架构，以verl v0.8.0 为例，介绍训推一致性比对数据采集的适配方案。

## 前置操作

### 基础配置

前置操作首先参照 《[fsdp训练后端verl训推一致性比对数据采集](./verl_fsdp_consistency_preprocess_dump.md#前置操作)》 或 《[megatron训练后端verl训推一致性比对数据采集](./verl_megatron_consistency_preprocess_dump.md#前置操作)》， 根据实际训练后端做选择。

此外，当前场景下还需做以下调整：

- 在当前异步rollout模式下，要使能vllm的dump功能，需要在vllm的`additional_config`中添加`dump_config_path`参数，指向msprobe的推理侧配置文件。**注意**：`/home/config_generate.json` 为示例路径，实际部署时需根据实际配置文件路径修改。
- 训练侧需关闭 `val_before_train` ，避免训练前验证调用`generate_sequence`接口，对 dump 结果造成干扰。

```diff
export DUMP_ON=1              # 启用训练侧 msprobe 采集
export PROMPTS_ONLY=1         # 仅计算 prompt 部分（必要，一致性仅支持 prefill）

# 启动入口为 main_ppo 
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
+   actor_rollout_ref.rollout.enforce_eager=True \
+   '+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config={dump_config_path:"/home/config_generate.json"}' \
+   trainer.val_before_train=False \
    trainer.balance_batch=False \
```

### Fully Async 场景

Fully Async 模式下，Rollouter 和 Trainer 各自独立 NPU 池，通过 `MessageQueue` 和 `ParameterSynchronizer` 解耦。训推一致性采集的前置配置与 Hybrid AgentLoop 基本一致，差异在于启动入口以及需要关闭bypass模式：

```diff
export DUMP_ON=1              # 启用训练侧 msprobe 采集
export PROMPTS_ONLY=1         # 仅计算 prompt 部分（必要，一致性仅支持 prefill）
export TORCHDYNAMO_DISABLE=1  # 关闭torchdynamo

# 启动入口为 fully_async_main
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

## msprobe 配置文件

训推两侧需要分别提供 msprobe 配置文件，参考[config_json_introduct.md](./config_json_introduct.md)，通过以下方式指定：

- **推理侧**：通过 `additional_config` 中的 `dump_config_path` 传递给 vLLM worker。
- **训练侧**：在 `transformer_impl.py` 的 `_ensure_debugger()` 中硬编码 `config_path`。

### 推理侧配置 (`config_generate.json`)

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

### 训练侧配置 (`config_actor.json`)

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

## 代码改动

### 文件改动清单

| 文件                                                               | 修改类型 | 说明                                                              | 对应小节                                                |
| ------------------------------------------------------------------ | -------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| `vllm_ascend/worker/dispatch_logger.py`                | **新增** | 推理调度日志记录（DispatchLogger）                                | [推理侧：调度日志记录](#推理侧调度日志记录)             |
| `vllm_ascend/worker/model_runner_v1.py`                | 修改     | 增加 DispatchLogger 初始化 + 4 处 log_step 调用                   | [推理侧：vLLM 模型执行采集](#推理侧vllm-模型执行采集)   |
| `verl/workers/engine/fsdp/transformer_impl.py`                | 修改     | FSDP 后端：增加训练侧 debugger + micro_batch request_id 日志      | [训练侧：FSDP 后端](#fsdp-后端)                         |
| `verl/workers/engine/megatron/transformer_impl.py`            | 修改     | Megatron 后端：增加训练侧 debugger + forward_step request_id 日志 | [训练侧：Megatron 后端](#megatron-后端)                 |
| `verl/workers/rollout/llm_server.py`                          | 修改     | `LLMServerClient` 中 request_id 注入 extra_fields                 | [Request ID 贯穿链路](#request-id-贯穿链路)             |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | 修改   | `FullyAsyncLLMServerClient` 继承父类 extra_fields 透传 request_id | [Request ID 贯穿链路](#request-id-贯穿链路)             |
| `verl/trainer/ppo/ray_trainer.py`                             | 修改     | PROMPTS_ONLY 模式（Hybrid AgentLoop）                             | [训练侧：仅计算 Prompt 部分](#训练侧仅计算-prompt-部分) |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | 修改     | PROMPTS_ONLY 模式（Fully Async）                                  | [训练侧：仅计算 Prompt 部分](#训练侧仅计算-prompt-部分) |

### 推理侧：vLLM 模型执行采集

**文件**：`vllm_ascend/worker/model_runner_v1.py`

**说明**：`dump_cfg` 读取、`PrecisionDebugger` 初始化、`debugger.start/stop/step` 调用均为 vllm-ascend 上游已有逻辑。本方案在此之上仅增加 **DispatchLogger** 初始化和 **log_step** 调用。

`__init__` 中增加的改动：
初始化 `DispatchLogger`，将 dump 路径指向 PID 子目录，并记录当前进程的分布式 rank。

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

`execute_model` 各 return 点增加的改动：

`execute_model()` 方法内部有多处 `self.debugger.stop()` 调用（分布在不同 return 路径），**每一处** `self.debugger.stop()` 之前都必须插入 `self._dispatch_logger.log_step(...)` 调用，才能覆盖所有return分支。

在每次模型前向完成后，调用 `DispatchLogger.log_step()` 记录该 step 的调度信息（包括涉及的请求 request、各请求分配的 token 数量，以及各请求在 prefill 与 decode 阶段的调度情况），随后执行 msprobe 的 stop/step 完成本轮 tensor dump。 `model_runner_v1.py` 中全局搜索 `self.debugger.stop()`，插入`log_step` 调用，以其中一处为例：

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

### 推理侧：调度日志记录

**文件**： `vllm_ascend/worker/dispatch_logger.py`（请在目录下创建该文件）

**功能**：在每次 `execute_model` 调用时，记录该 step 的调度元数据（step 序号、phase、该 step 调度的所有 request_id 及各分配的 token 数），写入 `dispatch_log.jsonl`。每条 JSONL 记录含 `pid`、`rank`、`step`、`phase`、`requests[]` 等字段，用于后续与 msprobe 的 `step_N/dump.json` 和训练侧的 `update_actor_log.jsonl` 做关联。

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

输出示例：

```json
{"source":"dispatch_logger","timestamp":"2026-05-13T10:00:00","pid":3680237,"rank":0,"step":0,"phase":"prefill","total_num_scheduled_tokens":95,"requests":[{"request_id":"f1f254c04e0c443b85ea1e7359e842dc","type":"new","tokens":95}]}
```

**注意**: vllm ≥ v0.14.0的版本会给外部传入的 request_id 追加 8 字符随机后缀，生成格式 {original_request_id}-{8hex}，例如 `f1f254c04e0c443b85ea1e7359e842dc-12345678`，选取时需要去掉后缀，用`f1f254c04e0c443b85ea1e7359e842dc` 才能与训练侧 `request_id` 匹配。

### 训练侧：模型执行采集

**涉及文件**：

| 后端     | 类名                         | 文件                                               |
| -------- | ---------------------------- | -------------------------------------------------- |
| FSDP     | `FSDPEngine`     | `verl/workers/engine/fsdp/transformer_impl.py`     |
| Megatron | `MegatronEngine`和`MegatronEngineWithLMHead` | `verl/workers/engine/megatron/transformer_impl.py` |

FSDP 和 Megatron 的 `forward_backward_batch` 架构不同，需分别处理。

---

#### FSDP 后端

FSDP 的 `forward_backward_batch` 存在显式的 `for i, micro_batch in enumerate(micro_batches)` 循环（`fsdp/transformer_impl.py`），可直接在循环体内包裹 debugger 调用。修改文件：`verl/workers/engine/fsdp/transformer_impl.py`。

1. `__init__`（`FSDPEngine` 类）

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

2. `forward_backward_batch`（`FSDPEngine` 类）

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

#### Megatron 后端

Megatron 的 `forward_backward_batch` 没有显式的 micro_batch 循环——它将所有 micro_batch 通过 `forward_backward_func()` 交给 Megatron 调度器统一执行。调度器内部每处理一个 micro_batch 就会调用一次 `forward_step`，因此直接在 `forward_step` 内部注入 debugger 的 start/stop/step 即可实现 per-micro-batch 粒度采集。修改文件：`verl/workers/engine/megatron/transformer_impl.py`。

1. `__init__`（`MegatronEngine` 类）

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

2. `forward_backward_batch`（`MegatronEngine` 类 `forward_step` 声明之后插入）

    `_ensure_debugger()` 负责惰性初始化 debugger，`self._should_dump` 作为开关供 `forward_step` 内部判断是否执行采集。

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

3. `forward_step`（`MegatronEngineWithLMHead` 类）

    在 `forward_fn` 调用之前插入 debugger.start；在调用之后插入 debugger.stop/step/_log_update_actor_step。

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

#### 辅助方法（FSDP后端和Megatron后端）

（`FSDPEngine` / `MegatronEngine` 类末尾新增，两个后端新增内容相同）

**注意**：`/home/config_actor.json` 为示例路径，实际部署时需根据实际配置文件所在路径修改。

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

输出示例：

```json
{"source":"update_actor","timestamp":"2026-05-13T10:00:01","pid":3665398,"rank":0,"step":0,"request_ids":["f1f254c04e0c443b85ea1e7359e842dc"],"num_requests":1}
```

### 训练侧：仅计算 Prompt 部分

**功能**：从 rollout 返回的训练数据中裁剪掉 response token，使训练阶段的前向只包含 prompt prefill 部分。训推一致性比对目前仅支持 prefill 部分，此改动确保 msprobe 在训练侧采集的 tensor 与推理侧 prefill step 的 tensor 在计算内容上等价。通过 `PROMPTS_ONLY=1` 环境变量控制。

两种异步模式下的改动位置不同：

#### Hybrid AgentLoop 模式

**文件**：`verl/trainer/ppo/ray_trainer.py`

**方法**：`RayPPOTrainer.fit()`

**插入位置**：搜索 `bypass_recomputing_logprobs`，在其之前插入

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

#### Fully Async 模式

**文件**：`verl/experimental/fully_async_policy/fully_async_trainer.py`

**方法**：`FullyAsyncTrainer._fit_generate()`

**插入位置**：`_get_samples_from_queue()` 返回 `batch` 之后，`batch.meta_info["temperature"]` 赋值之前：

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

### Request ID 贯穿链路

**涉及文件**：

| 文件 | 说明 |
| ---- | ---- |
| `verl/workers/rollout/llm_server.py` | `LLMServerClient` 中注入 `request_id` 到 extra_fields |
| `verl/experimental/fully_async_policy/fully_async_rollouter.py` | `FullyAsyncLLMServerClient` 继承父类 extra_fields，确保 `request_id` 透传 |

**功能**：将 vLLM 内部使用的 `request_id` 注入 `TokenOutput.extra_fields`，使其自动随 verl 数据流贯穿至训练侧 micro_batch，实现推理调度记录（`dispatch_log.jsonl`）与训练 micro_batch 记录（`update_actor_log.jsonl`）通过 `request_id` 精确关联。

#### 1. LLMServerClient：注入 request_id

`LLMServerClient.generate()` 生成 `vllm_request_id`，将其同时作为 vLLM 的 `request_id` 和 `extra_fields` 中的 `request_id` 字段：

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
-                request_id=uuid4().hex,  # use new request_id for each turn
+               request_id=vllm_request_id,  # use new request_id for each turn
                ...
            )
+           output.extra_fields["request_id"] = vllm_request_id
            return output
```

#### 2. FullyAsyncLLMServerClient：透传 extra_fields

Fully Async 模式下，`FullyAsyncLLMServerClient` 继承自 `LLMServerClient`，通过 `super().generate()` 调用父类获取已注入 `request_id` 的 `TokenOutput`。由于 `FullyAsyncLLMServerClient` 支持 **partial rollout**（多轮 resume），需要创建新的 `final_output` 来累积多次 `super().generate()` 的结果。在新的 `final_output` 中需继承父类返回的完整 `extra_fields`，确保 `request_id` 等字段继续向下传递：

```diff
class FullyAsyncLLMServerClient(LLMServerClient):
    ...
    @rollout_trace_op
    async def generate(
        self,
        ...
    ) -> TokenOutput:
        ...
+       final_output.extra_fields.update(output.extra_fields)  # 继承父类全部 extra_fields（含 request_id）
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output
```

`request_id` 自动贯穿以下链路：

```plain
vLLM Server (request_id)
  → TokenOutput.extra_fields["request_id"]
    → AgentLoopOutput.extra_fields
      → _InternalAgentLoopOutput.extra_fields
        → DataProto.non_tensor_batch["request_id"]
          → XXXEngine micro_batch → update_actor_log.jsonl
```

## dump结果文件介绍

训练完成后，dump 路径下生成以下文件：

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

文件说明：

| 文件                                                 | 内容                      | 粒度                      |
| ---------------------------------------------------- | ------------------------- | ------------------------- |
| `{dump_generate_path}/{pid}/step_N/rank_M/dump.json` | vLLM 前向 tensor 统计     | 每次 `execute_model`      |
| `{dump_actor_path}/step_N/rank_M/dump.json`          | 训练前向+反向 tensor 统计 | 每个 micro_batch          |
| `{dump_generate_path}/{pid}/dispatch_log.jsonl`      | vLLM 调度信息             | 每次 `execute_model` 一行 |
| `{dump_actor_path}/{pid}/update_actor_log.jsonl`     | 训练 request_id 记录      | 每个 micro_batch 一行     |

## 数据关联方法

通过 [Request ID 贯穿链路](#request-id-贯穿链路) 中注入的 `request_id`，将 [推理侧：vLLM 模型执行采集](#推理侧vllm-模型执行采集)（`dispatch_log.jsonl` + `step_N/dump.json`）与 [训练侧：模型执行采集](#训练侧模型执行采集)（`update_actor_log.jsonl` + `step_N/dump.json`）进行关联，从而支持训推一致性的比对工作(参见《[PyTorch场景精度比对](../accuracy_compare/pytorch_accuracy_compare_instruct.md#verl训推一致性比对场景)》)。具体步骤如下：

### 关联步骤

1. **选取推理 step**：在 `dispatch_log.jsonl` 中找到合适的 `step`和`request_id`( `phase`为 `prefill`，且`requests`数量为1的)，注意，vllm ≥ v0.14.0的版本会给外部传入的 request_id 追加 8
  字符随机后缀，生成格式 {original_request_id}-{8hex}，选取时需要去掉后缀，才能与训练侧 `request_id` 匹配。
2. **定位训练 step**：在 `update_actor_log.jsonl` 中搜索同一 `request_id`，找到 `step` 和 `rank`
3. **读取 dump 数据**：根据 step 序号和`rank`序号读取对应的 `dump.json`
4. **进行训推一致性比对**

### JSON 字段规范

所有 JSONL 日志共用顶层字段：

| 字段        | 类型   | 说明                                   |
| ----------- | ------ | -------------------------------------- |
| `source`    | string | `"dispatch_logger"` / `"update_actor"` |
| `timestamp` | string | ISO 8601 时间戳                        |
| `pid`       | int    | 进程 ID                                |

`dispatch_log.jsonl` 特有：

| 字段                         | 说明                       |
| ---------------------------- | -------------------------- |
| `step`                       | execute_model 的 step 序号 |
| `rank`                       | 分布式 rank                |
| `phase`                      | `"prefill"` / `"decode"`   |
| `total_num_scheduled_tokens` | 该 step 调度的 token 总数  |
| `requests[].request_id`      | vLLM 内部 request_id       |
| `requests[].type`            | `"new"` / `"cached"`       |
| `requests[].tokens`          | 分配的 token 数            |

`update_actor_log.jsonl` 特有：

| 字段            | 说明                                          |
| --------------- | --------------------------------------------- |
| `step`          | micro_batch step 序号                         |
| `rank`          | 分布式 rank                                   |
| `request_ids[]` | 该 micro_batch 包含的 request_id              |
| `num_requests`  | request 数量 （micro_batch中数据数量，应为1） |
