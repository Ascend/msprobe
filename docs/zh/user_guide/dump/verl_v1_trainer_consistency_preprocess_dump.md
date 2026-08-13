# verl V1 Trainer训推一致性比对数据采集

## 简介

本文以verl v0.9.0.dev为例，介绍V1 Trainer和当前Fully Async架构下训推一致性比对数据的采集适配方法，适用于FSDP或Megatron训练后端。

verl v0.8.0的主入口采用Hybrid AgentLoop，rollout生成的数据由trainer组织并进入训练流程。verl v0.9.0.dev引入V1 Trainer，将AgentLoop生成的数据通过TransferQueue传递给训练侧，并使用jagged TensorDict组织训练batch。数据通路的变化使V1能够通过同一套trainer入口支持同步训练，以及不同资源部署方式下的异步训练。训推一致性采集也需要相应调整request ID的传递位置和prompt-only输入的裁剪方式。

V1 Trainer通过 `trainer.use_v1=True` 启用，并通过 `trainer.v1.trainer_mode` 选择以下三种模式：

| 模式 | 配置值 | 说明 |
| --- | --- | --- |
| Sync | `sync` | rollout与训练按step依次执行，trainer等待当前step的rollout数据后再进行训练，数据通过TransferQueue传递 |
| Colocate Async | `colocate_async` | rollout与训练异步调度，但部署在同一组计算资源上，适用于资源共享场景 |
| Separate Async | `separate_async` | rollout与训练异步调度，并部署在相互独立的计算资源上，通过队列解耦两侧执行 |

verl还在 `verl.experimental.fully_async_policy` 中提供独立的Fully Async实现。
该实现不属于上述三种V1 Trainer模式，仍通过MessageQueue连接Rollouter和Trainer。
其核心采集方案与verl v0.8.0一致，但request ID处理位置和partial rollout处理方式发生变化。

本文以《[异步架构verl训推一致性比对数据采集](./verl_async_consistency_preprocess_dump.md)》为基线，只介绍verl v0.9.0.dev新增或不同的改动。未在本文展开的公共改动均沿用基线文档。

## 前置操作

首先根据实际训练后端，完成基线文档中的[前置操作](./verl_async_consistency_preprocess_dump.md#前置操作)和[msprobe配置](./verl_async_consistency_preprocess_dump.md#msprobe-配置文件)。

### V1

以下代码块汇总了基线文档和V1需要修改的参数，可直接合入`main_ppo` 启动脚本。`/home/config_generate.json` 为示例路径，使用时需要替换为推理侧msprobe配置文件的实际路径。

```diff
export DUMP_ON=1                # 启用训练侧msprobe采集
export PROMPTS_ONLY=1           # 将训练输入裁剪为prompt-only

# 启动入口为main_ppo
python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config={dump_config_path:"/home/config_generate.json"}' \
    trainer.val_before_train=False \
    trainer.balance_batch=False \
    ...
```

**说明**：

- `PROMPTS_ONLY=1` 会改变训练输入，仅用于训推一致性采集。
- 该模式下的loss和梯度不代表正常训练结果。
- 采集结束后，应取消 `DUMP_ON`、`PROMPTS_ONLY` 和推理侧
  `dump_config_path`，恢复正常训练配置。

### Fully Async

Fully Async继续沿用[基线文档](./verl_async_consistency_preprocess_dump.md#前置操作)的前置操作，包括独立的Rollouter和Trainer资源、关闭`rollout_correction.bypass_mode`以及在 `FullyAsyncTrainer` 中裁剪prompt-only输入。

## 代码改动

### 文件改动清单

#### V1 Trainer场景

| 文件 | 修改类型 | 说明 | 对应小节 |
| --- | --- | --- | --- |
| `vllm_ascend/worker/dispatch_logger.py` | 新增 | 记录推理侧每个采集step的request ID | [推理侧：调度日志记录](./verl_async_consistency_preprocess_dump.md#推理侧调度日志记录) |
| `vllm_ascend/worker/model_runner_v1.py` | 修改 | 初始化DispatchLogger，并同步记录调度日志和msprobe step | [推理侧：vLLM模型执行采集](./verl_async_consistency_preprocess_dump.md#推理侧vllm-模型执行采集) |
| `verl/workers/engine/fsdp/transformer_impl.py` | 修改 | FSDP训练侧采集和request ID日志 | [训练侧：FSDP后端](./verl_async_consistency_preprocess_dump.md#fsdp-后端) |
| `verl/workers/engine/megatron/transformer_impl.py` | 修改 | Megatron训练侧采集和request ID日志 | [训练侧：Megatron后端](./verl_async_consistency_preprocess_dump.md#megatron-后端) |
| `verl/workers/rollout/llm_server.py` | 修改 | `LLMServerClient` 注入request ID | [Request ID贯穿链路](./verl_async_consistency_preprocess_dump.md#request-id-贯穿链路) |
| `verl/trainer/ppo/v1/agent_loop_tq.py` | 修改 | 将request ID写入TransferQueue顶层字段 | [AgentLoopWorkerTQ传递request ID](#agentloopworkertq传递request-id) |
| `verl/workers/engine_workers.py` | 修改 | 裁剪V1 jagged batch中的response | [Actor Worker裁剪prompt-only输入](#actor-worker裁剪prompt-only输入) |

#### 独立Fully Async场景

| 文件 | 修改类型 | 说明 | 对应小节 |
| --- | --- | --- | --- |
| `vllm_ascend/worker/dispatch_logger.py` | 新增 | 记录推理侧每个采集step的request ID | [推理侧：调度日志记录](./verl_async_consistency_preprocess_dump.md#推理侧调度日志记录) |
| `vllm_ascend/worker/model_runner_v1.py` | 修改 | 初始化DispatchLogger，并同步记录调度日志和msprobe step | [推理侧：vLLM模型执行采集](./verl_async_consistency_preprocess_dump.md#推理侧vllm-模型执行采集) |
| `verl/workers/engine/fsdp/transformer_impl.py` | 修改 | FSDP训练侧采集和request ID日志 | [训练侧：FSDP后端](./verl_async_consistency_preprocess_dump.md#fsdp-后端) |
| `verl/workers/engine/megatron/transformer_impl.py` | 修改 | Megatron训练侧采集和request ID日志 | [训练侧：Megatron后端](./verl_async_consistency_preprocess_dump.md#megatron-后端) |
| `verl/workers/rollout/llm_server.py` | 修改 | 注入request ID，并保留partial rollout分段ID | [保留partial rollout的request ID](#保留partial-rollout的request-id) |
| `verl/experimental/fully_async_policy/fully_async_trainer.py` | 修改 | 裁剪padded DataProto中的response | [prompt-only输入](#prompt-only输入) |

### V1组件改动

#### AgentLoopWorkerTQ传递request ID

**当前版本涉及文件**：`verl/trainer/ppo/v1/agent_loop_tq.py`

**基线文档相关文件**：`verl/workers/rollout/llm_server.py`，用于注入request ID。

**说明**：将rollout生成的request ID加入TransferQueue数据，使其随训练数据传递到训练侧，用于关联推理侧和训练侧的dump数据。

`LLMServerClient` 注入的request ID位于 `AgentLoopOutput.extra_fields`。V1通过TransferQueue将rollout数据传递给训练侧，需要在`AgentLoopWorkerTQ._agent_loop_postprocess()` 组装 `field` 时，将该ID写入顶层`request_id` 字段：

```diff
class AgentLoopWorkerTQ(AgentLoopWorker):
    ...
    async def _agent_loop_postprocess(
        self,
        ...
    ) -> None:
        ...
        field = output.as_dict()
        field.update(kwargs)
+       extra_fields = field.get("extra_fields")
+       field["request_id"] = (
+           extra_fields.get("request_id")
+           if isinstance(extra_fields, dict)
+           else None
+       )
        ...
```

即使没有取得request ID，也要保留值为 `None` 的 `request_id` 字段，避免同一
TransferQueue中的数据字段不一致。

request ID经过以下链路进入训练micro-batch：

```text
LLMServerClient
  -> TokenOutput.extra_fields["request_id"]
  -> AgentLoopOutput.extra_fields
  -> AgentLoopWorkerTQ顶层field["request_id"]
  -> TransferQueue
  -> actor TensorDict
  -> FSDP或Megatron micro-batch
  -> update_actor_log.jsonl
```

#### Actor Worker裁剪prompt-only输入

**当前版本涉及文件**：`verl/workers/engine_workers.py`

**基线文档对应文件**：`verl/trainer/ppo/ray_trainer.py`，用于Hybrid AgentLoop的
prompt-only裁剪，V1不需要修改该文件。

**说明**：将训练输入裁剪为prompt-only，使训练侧输入与推理侧prefill输入保持一致。

基线文档中的Hybrid AgentLoop在 `RayPPOTrainer.fit()` 中按response固定长度裁剪padded DataProto。V1输入为jagged TensorDict，需要在actor worker中按每条样本的实际prompt长度进行裁剪：

```diff
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    ...
    def update_actor(self, data: TensorDict) -> TensorDict:
+       if int(os.environ.get("PROMPTS_ONLY", "0")):
+           data = data.clone(recurse=False)
+           prompts = data["prompts"]
+           if prompts.is_nested:
+               prompt_lengths = prompts.offsets().diff().tolist()
+               data["input_ids"] = prompts
+               data["position_ids"] = torch.nested.as_nested_tensor(
+                   [
+                       position_ids[..., :prompt_length]
+                       for position_ids, prompt_length in zip(
+                           data["position_ids"].unbind(),
+                           prompt_lengths,
+                           strict=True,
+                       )
+                   ],
+                   layout=torch.jagged,
+               )
+               for key in (
+                   "responses",
+                   "rollout_log_probs",
+                   "response_mask",
+                   "loss_mask",
+                   "old_log_probs",
+                   "advantages",
+                   "rollout_is_weights",
+                   "ref_log_prob",
+               ):
+                   value = data.get(key)
+                   if isinstance(value, torch.Tensor):
+                       data[key] = (
+                           torch.nested.as_nested_tensor(
+                               [sample[:0] for sample in value.unbind()],
+                               layout=torch.jagged,
+                           )
+                           if value.is_nested
+                           else value[:, :0]
+                       )
        output = self.actor.train_mini_batch(data=data)
        return output
```

此处先浅复制 `data`，避免修改TransferQueue中保存的原始数据。V1只应用本节的
jagged TensorDict裁剪，不再应用基线文档中 `RayPPOTrainer.fit()` 的padded
DataProto裁剪。

### Fully Async组件改动

#### 保留partial rollout的request ID

**当前版本涉及文件**：`verl/workers/rollout/llm_server.py`

**基线文档对应文件**：
`verl/experimental/fully_async_policy/fully_async_rollouter.py`。当前版本的客户端已
迁移到公共 `llm_server.py`，不再修改该基线文件。

**说明**：记录partial rollout中首段和全部推理分段的request ID，用于关联训练侧数据并追踪完整的推理过程。

当前 `FullyAsyncLLMServerClient` 已从 `fully_async_rollouter.py` 迁入公共`llm_server.py`。一次partial rollout可能因abort和resume拆分为多个实际vLLM请求，因此需要同时保存第一段request ID和全部分段ID：

```diff
class FullyAsyncLLMServerClient(LLMServerClient):
    ...
    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids,
        sampling_params,
        ...
    ) -> TokenOutput:
        ...
        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None
+       inference_request_ids = []

        while True:
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                ...
            )
+           inference_request_ids.append(output.extra_fields["request_id"])
            ...

        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
+       final_output.extra_fields["request_id"] = inference_request_ids[0]
+       final_output.extra_fields["inference_request_ids"] = inference_request_ids
        return final_output
```

#### prompt-only输入

**当前版本涉及文件**：
`verl/experimental/fully_async_policy/fully_async_trainer.py`

**基线文档对应文件**：
`verl/experimental/fully_async_policy/fully_async_trainer.py`，文件路径和
prompt-only改动均未发生变化。

**说明**：将训练输入裁剪为prompt-only，使训练侧输入与推理侧第一段prefill输入保持一致。

Fully Async仍使用padded DataProto，与基线文档的[Fully Async模式prompt-only改动](./verl_async_consistency_preprocess_dump.md#fully-async-模式)改动保持一致。

## 数据关联

公共日志格式、vLLM request ID随机后缀处理和dump文件定位方法参见基线文档的[数据关联方法](./verl_async_consistency_preprocess_dump.md#数据关联方法)。

## dump结果文件介绍

相较于基线文档无新增改动文件，详细结果介绍请参见
[dump结果文件介绍](./verl_async_consistency_preprocess_dump.md#dump结果文件介绍)。
