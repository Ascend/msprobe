# Training-Inference Consistency Preprocessing and Data Collection in the slime Framework

## 1. Introduction

The slime framework adopts a training-inference decoupled architecture of "Megatron on the training side + SGLang on the inference side".

When locating accuracy differences, since the default input of `old_log_prob` on the training side is prompt + response while the prefill input on the inference side is only prompt, the input tokens on the two sides are inconsistent, and the dumped statistics cannot be directly compared.

This document specifically targets the training-inference consistency accuracy localization scenario and provides a complete preprocessing and data collection process:

1. Unify the input tokens of the prefill stage on both sides through preprocessing, ensuring that the data satisfies engineering constraints such as parallel sharding, batch, and padding. See [Training-Inference Consistency Preprocessing](#3-training-inference-consistency-preprocessing).
2. Collect only the data corresponding to the training-side `old_log_prob` and the inference-side prefill stages, used for per-rank comparison to quickly locate the source of the accuracy difference. See [msProbe Data Collection on Both Sides](#4-msprobe-data-collection-on-both-sides).

For scenarios that do not require training-inference input alignment, the general collection method of the slime framework can be used. For details, see "[Precision Data Collection for Training and Inference in the slime Framework](./slime_train_rollout_dump_instruct.md)".

## 2. Environment Preparation

Install the msProbe tool. For details, see [msProbe Tool Installation Guide](../../install_guide/msprobe_install_guide.md).

```shell
pip install mindstudio-probe
```

Runtime environment and image:

**Dockerfile to build the image**: You can build the image and environment based on the [Slime_ascend Dockerfile](https://gitcode.com/Ascend/slime-ascend/tree/main/docker/npu_docker).

## 3. Training-Inference Consistency Preprocessing

Before the training-inference consistency comparison, it must be ensured that the **input tokens and sequence lengths** during training and inference are consistent, and that constraints such as **parallel sharding and batch configuration** are satisfied, so that the dumped precision data can be matched and compared.

The preprocessing goals (the effect to be achieved on each side) are as follows:

| Target Side | Target Effect | Corresponding Section |
|----|----------|----------|
| **Inference side** | The first EXTEND (prefill) input of SGLang is the prompt token; graph mode is disabled, and constraints such as DP / chunked prefill, parallel sharding, and sequence length alignment are satisfied, ensuring that the formal prefill forward is collected | [3.2](#32-sequence-length-and-padding-constraints), [3.3](#33-sglang-dp-and-chunked-prefill-constraints), [3.4](#34-consistent-training-inference-parallel-sharding), [3.7](#37-disabling-cuda-graph-mode) |
| **Training side** | The Megatron `old_log_prob` input is truncated to **prompt-only** (removing the response) to align with the inference prefill; at the same time, constraints such as padding, a single prompt, and disabled data balancing are satisfied | [3.1](#31-prompt-only-input-alignment), [3.2](#32-sequence-length-and-padding-constraints), [3.5](#35-single-prompt-input-and-dp-configuration), [3.6](#36-disabling-data-balancing-and-dynamic-batch) |

### 3.1 prompt-only Input Alignment

**Preprocessing action**: Truncate the input of `old_log_prob` on the training side from "prompt + response" to prompt only, making it consistent with the inference-side prefill input.

**Reason**: In the normal GRPO pipeline of slime, rollout produces `sample.tokens` (prompt + response), and the training-side `old_log_prob` computes the log prob over the complete sequence by default. The training-inference consistency scenario only needs to compare the **prefill segment**, so the training side must remove the response and keep only the prompt tokens.

**Code implementation**: see [4.2 Data Collection in the Training Stage](#42-data-collection-in-the-training-stage): add `consistent_dump_utils.py`, and when `DUMP_ON=1` and `PROMPTS_ONLY=1`, at the `train_actor` entry make a prompt-only copy of the rollout batch before executing `compute_log_prob`, and **skip actor training** after collection is complete.

### 3.2 Sequence Length and padding Constraints

To ensure that the model weights used during training and inference are exactly the same and to eliminate floating-point precision differences caused by different padding strategies, the training side and the inference side must use the same sequence length alignment strategy.

It is recommended to set the padding coefficient `--data-pad-size-multiplier` on the training side to `1`, ensuring that the sequence length is aligned only to a multiple of the tensor parallel degree (TP), thereby minimizing the number of padding tokens and the potential numerical error.

#### Training-Side Megatron Configuration

On the training side, Megatron computes the target length to which each sequence needs to be padded during the `get_batch` stage according to the following formula:

```python
pad_size = tensor_model_parallel_world_size * data_pad_size_multiplier
```

- **tensor_model_parallel_world_size (TP)**: Tensor parallel degree. For example, if 4 ranks are used for tensor parallel training, TP=4. TP is usually configured as a power of 2, and the specific value depends on the number of ranks.
- **data_pad_size_multiplier (multiplier)**: The coefficient of the padding stride. It ensures that the length of all input sequences is an integer multiple of `TP × multiplier`. The default value of multiplier is 128, and it is recommended to configure it as 1.

#### Inference-Side (SGLang) Data Preprocessing Requirements

SGLang itself supports dynamic input of any length, but to align with the Megatron training side, the sequence length must be aligned in the preprocessing stage with the same `pad_size` as the training side before passing the data to SGLang.

The specific operation depends on the configuration of the training side:

| Configuration | pad_size | Description |
|------|----------|------|
| **Recommended** TP=1, multiplier=1 | 1 | No padding required. The original prompt length can be used directly, and the training side and the inference side are naturally aligned. |
| TP>1, multiplier=1 | TP | Padding is required. Pad the prompt length to an integer multiple of TP. |
| TP>1, multiplier=128 | TP × 128 | Padding is required. Pad the prompt length to an integer multiple of `TP × 128`. Because of excessive padding, this scenario is not recommended for accuracy comparison. |

In dump mode, `--data-pad-size-multiplier 1` must be set; if the training TP>1, the prompt lengths passed to both sides must be ensured to be an integer multiple of `TP × multiplier`.

### 3.3 SGLang DP and chunked prefill Constraints

When `--sglang-dp-size > 1` is enabled, the following must be satisfied:

```text
sglang-chunked-prefill-size / sglang-dp-size  >  actual prefill prompt length
```

### 3.4 Consistent Training-Inference Parallel Sharding

The training-inference consistency comparison requires that the **model parallel sharding method** of training and inference be consistent, ensuring that each rank on both sides undertakes the same parameter sharding and computation path, so that the dump data can correspond one-to-one by rank.

| Dimension | Training Side (Megatron) | Inference Side (SGLang) | Alignment Requirement |
|------|-------------------|------------------|----------|
| TP | `--tensor-model-parallel-size` | `--rollout-num-gpus-per-engine` (the inference TP when there is a single engine) | The TP of both sides must be consistent |
| PP | `--pipeline-model-parallel-size` | Usually 1 | The PP of both sides must be consistent |
| CP | `--context-parallel-size` | Usually 1 | The CP of both sides must be consistent |
| DP | Derived from world_size and TP/PP/CP | `--sglang-dp-size` (when DP-Attention is enabled) | The DP semantics of both sides must be consistent |

### 3.5 Single prompt Input and DP Configuration

To ensure that both the training and inference sides process the **same prompt** and that the training-side Megatron can distribute samples normally by DP, dump mode must fix:

```shell
--rollout-batch-size 1          # Only 1 prompt per rollout round
--n-samples-per-prompt ${DP}    # Must be equal to DP, ensuring normal training operation
--global-batch-size ${DP}         # rollout-batch-size × n-samples-per-prompt
```

The data parallel degree is calculated as follows:

```text
DP = world_size / TP / PP / CP
```

Here, `world_size` is the total number of training participating processes (usually equal to the number of training NPUs), and `TP`, `PP`, and `CP` correspond to the tensor / pipeline / context parallel degrees respectively.

**Example** (4-rank training, `TP=1, PP=1, CP=1`):

```text
DP = 4 / 1 / 1 / 1 = 4
→ --rollout-batch-size 1 --n-samples-per-prompt 4 --global-batch-size 4
```

If `global_batch_size` cannot be divided evenly by DP, Megatron training reports an error.

### 3.6 Disabling Data Balancing and Dynamic Batch

The training-inference consistency scenario must disable optimization items that change the batch sharding or sample distribution, to avoid a mismatch between the data obtained by each rank on the training side and the inference prefill:

| Parameter | Requirement | Reason |
|------|------|------|
| `--balance-data` | **Disabled** (not passed) | Automatic balancing rearranges the batch, breaking the one-to-one correspondence between ranks and samples |
| `--use-dynamic-batch-size` | **Disabled** (not passed) | Dynamic batch changes the micro-batch sharding, affecting the comparability of the dump |

In dump mode, it is also recommended to disable parameters that introduce random sampling, such as `--rollout-shuffle` and `--over-sampling-batch-size`, ensuring that the input of each round is deterministic and reproducible.

### 3.7 Disabling CUDA Graph Mode

Training-inference consistency dump must disable the CUDA Graph (graph compilation cache in the Ascend scenario) optimization of SGLang; otherwise, msProbe cannot correctly hook the operators of each EXTEND forward, and the dump data is incomplete or inconsistent with eager mode.

When slime starts SGLang, the following must be specified:

```shell
--sglang-disable-cuda-graph
```

## 4. msProbe Data Collection on Both Sides

After completing the preprocessing constraints in chapter 3, configure and collect the msProbe data of the inference side (generate / prefill) and the training side (old_log_prob) separately according to this section.

### 4.1 Creating the Configuration File

Each of the training and inference sides needs a `config.json`. When collecting statistics, `task` is configured as `statistics`. An example is as follows:

**Inference side (config_generate.json)**:

```json
{
    "task": "statistics",
    "dump_path": "/example_dump_path/msprobe_dump/generate",
    "rank": [],
    "step": [0],
    "level": "mix",
    "async_dump": false,
    "extra_info": true,
    "statistics": {
        "scope": [],
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

**Training side (config_actor.json)**:

```json
{
    "task": "statistics",
    "dump_path": "/example_dump_path/msprobe_dump",
    "rank": [],
    "step": [0],
    "level": "mix",
    "async_dump": false,
    "extra_info": true,
    "statistics": {
        "scope": [],
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

Description of the main parameters:

- **task**: `statistics` indicates collecting statistics; if real tensors are needed, configure it as `tensor`.
- **dump_path**: The dump save root path; the training-side code dynamically concatenates the `old_log_prob` subdirectory under `dump_path` for writing to disk, and the inference side points directly to the `generate` subdirectory for writing to disk.
- **step**: The number of collection steps. Training-inference consistency dump usually collects only `step0`.
- **level**: `mix` indicates collecting both Module-level and API-level data.

For a detailed introduction to the above configuration parameters, see "[Configuration File Introduction](./config_json_introduct.md)".

### 4.2 Data Collection in the Training Stage

Precision data collection in the training stage is completed on the Megatron training backend of slime. Instantiate `PrecisionDebugger` in `slime/backends/megatron_utils/actor.py`, call the `start` and `stop` interfaces before and after the **old_log_prob forward**, and call `step()` after stop to advance the step count. For more introduction to the `PrecisionDebugger` interface, see "[Precision Data Collection in the PyTorch Scenario](./pytorch_data_dump_instruct.md)".

The following shows the code modification method for training-inference consistency dump (using slime v0.2.2 as an example). The modification steps are as follows:

1. Add `slime/backends/megatron_utils/consistent_dump_utils.py`

    ```python
    """Utilities for training-inference consistency msProbe dump."""

    from copy import deepcopy
    import torch
    from slime.utils.types import RolloutBatch

    def copy_rollout_data_prompts_only(rollout_data: RolloutBatch) -> RolloutBatch:
        """Return a copy with tokens truncated to prompt-only (no response)."""
        dump_data = deepcopy(rollout_data)
        new_tokens, new_total_lengths, new_response_lengths = [], [], []

        for tokens, total_length, response_length in zip(
            dump_data["tokens"],
            dump_data["total_lengths"],
            dump_data["response_lengths"],
            strict=True,
        ):
            prompt_length = total_length - response_length
            new_tokens.append(tokens[:prompt_length])
            new_total_lengths.append(prompt_length)
            new_response_lengths.append(0)

        dump_data["tokens"] = new_tokens
        dump_data["total_lengths"] = new_total_lengths
        dump_data["response_lengths"] = new_response_lengths
        if "loss_masks" in dump_data:
            ref = dump_data["loss_masks"][0]
            if isinstance(ref, torch.Tensor):
                dump_data["loss_masks"] = [
                    torch.tensor([], dtype=ref.dtype, device=ref.device) for _ in new_tokens
                ]
            else:
                dump_data["loss_masks"] = [[] for _ in new_tokens]

        return dump_data
    ```

2. Instantiate PrecisionDebugger in `MegatronTrainRayActor.init` of `slime/backends/megatron_utils/actor.py`

    It must be instantiated **before** `monkey_patch_torch_dist()` to avoid conflict with `bias_swiglu_fusion`.

    ```diff
     class MegatronTrainRayActor(TrainRayActor):
         def init(
             self,
             args: Namespace,
             role: str,
             with_ref: bool = False,
         ) -> int | None:

    +        # [msProbe] Training-side precision collection switch
    +        dump_flag = int(os.environ.get("DUMP_ON", 0))
    +        if dump_flag:
    +            from msprobe.pytorch import PrecisionDebugger, seed_all
    +            seed_all(mode=True)
    +            self.debugger = PrecisionDebugger(config_path=os.environ["MSPROBE_CONFIG_PATH"])
    +            self.dump_path_prefix = self.debugger.config.dump_path
    +        else:
    +            self.debugger = None

             monkey_patch_torch_dist()
             super().init(args, role, with_ref)
             ...
    ```

3. Collect old_log_prob and skip actor training in `train_actor` of `slime/backends/megatron_utils/actor.py`

    ```diff
         def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:

    +        # [msProbe] Training-inference consistency dump: collect only the old_log_prob forward (prompt-only), and skip actor training after completion
    +        if int(os.environ.get("DUMP_ON", 0)):
    +            from .consistent_dump_utils import copy_rollout_data_prompts_only
    +
    +            dump_rollout_data = copy_rollout_data_prompts_only(rollout_data)
    +            dump_data_iterator, dump_num_microbatches = get_data_iterator(
    +                self.args, self.model, dump_rollout_data
    +            )
    +            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
    +            if self.debugger:
    +                self.debugger.service.config.dump_path = os.path.join(
    +                    self.dump_path_prefix, "old_log_prob"
    +                )
    +                self.debugger.start(model=self.model)
    +            self.compute_log_prob(
    +                dump_data_iterator,
    +                dump_num_microbatches,
    +                store_prefix="",
    +            )
    +            if self.debugger:
    +                self.debugger.stop()
    +                self.debugger.step()
    +            train_dump_utils.save_debug_train_data(
    +                self.args, rollout_id=rollout_id, rollout_data=rollout_data
    +            )
    +            return

             # Create data iterator for log_probs and train.
             data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
             ...
    ```

Note:

- The training-inference consistency scenario **collects only the old_log_prob stage**; `debugger.step()` needs to be called only once after `stop()`.

### 4.3 Data Collection in the Inference Stage

The inference side of the slime framework uses the SGLang engine to perform rollout generation.

Since version 0.5.11, SGLang has natively built in the msProbe capability:

- Versions lower than 0.5.11: `PrecisionDebugger` must be inserted intrusively.
- Versions not lower than 0.5.11: Although the dump capability is built in, each forward (including prefill and decode) triggers the built-in `start/stop/step` process and does not skip the dummy forward in the slime startup stage, causing the dummy forward to occupy the earlier steps (`step0`, `step1`, etc.).

Therefore, both versions require a small modification to `ModelRunner.forward()` to skip the dummy forward.

Please choose the operation method according to the SGLang version:

| SGLang Version | Operation Method |
|-------------|------|
| **< 0.5.11** | Intrusively insert `PrecisionDebugger` and skip the dummy forward, see [SGLang Below 0.5.11](#sglang-below-0511-intrusive-instrumentation) below. |
| **≥ 0.5.11** | msProbe is natively built in; specify `--sglang-msprobe-dump-config` in `SGLANG_ARGS`; but `forward()` still needs to be modified to skip the dummy forward, see [SGLang 0.5.11 and Above](#sglang-0511-and-above-built-in-msprobe-and-skipping-dummy-forward) below. |

#### SGLang Below 0.5.11 (Intrusive Instrumentation)

Insert the `PrecisionDebugger` interface in `sglang/srt/model_executor/model_runner.py`. The modification steps are as follows:

1. Instantiate the debugger at the end of `ModelRunner.__init__`

    ```diff
             self.forward_pass_id = 0
             self.draft_model_idx = draft_model_idx

    +        # [msProbe] Inference-side precision collection, enabled by SGLANG_MSPROBE_DUMP=1
    +        if int(os.environ.get("SGLANG_MSPROBE_DUMP", "0")):
    +            from msprobe.pytorch import PrecisionDebugger, seed_all
    +            seed_all(mode=True)
    +            self.debugger = PrecisionDebugger(config_path=os.environ["MSPROBE_GENERATE_CONFIG"])
    ```

2. Collect for the real prefill in `forward()`

    Only dump the EXTEND forward whose token count is ≥ `MSPROBE_MIN_DUMP_TOKENS`, skipping the dummy forward of about 1 token in the startup stage.

    ```diff
         def forward(self, forward_batch, ...):
             self.forward_pass_id += 1

    +        _msprobe_dump = False
    +        if hasattr(self, "debugger"):
    +            _min_tokens = int(os.environ.get("MSPROBE_MIN_DUMP_TOKENS", "2"))
    +            _num_tokens = (
    +                forward_batch.input_ids.numel()
    +                if forward_batch.input_ids is not None else 0
    +            )
    +            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True) \
    +                    and _num_tokens >= _min_tokens:
    +                _msprobe_dump = True
    +                self.debugger.start(model=self.model, rank_id=self.gpu_id)

             ...
             output = ...

    +        if hasattr(self, "debugger") and _msprobe_dump:
    +            self.debugger.stop()
    +            self.debugger.step()

             return output
    ```

#### SGLang 0.5.11 and Above (Built-in msProbe and Skipping dummy forward)

Taking SGLang 0.5.13 as an example, the built-in msProbe implementation of SGLang is located in `sglang/srt/model_executor/model_runner.py`:

- `ModelRunner.__init__` (about lines 443~445): when `server_args.msprobe_dump_config` is not empty, call `init_msprobe()` to instantiate `self.msprobe_debugger`;
- `init_msprobe()` (about lines 603~618): after `seed_all(mode=True)`, create `PrecisionDebugger(config_path=self.server_args.msprobe_dump_config)`;
- `ModelRunner.forward()` (starting from about line 3407): each forward **unconditionally** executes `self.msprobe_debugger.start(...)` → forward → `self.msprobe_debugger.stop()` + `self.msprobe_debugger.step()`.

The built-in implementation advances the msProbe step on every forward. When slime starts SGLang, the health check calls `/health_generate` (`input_ids=[0]`, `max_new_tokens=1`), which produces **two** dummy forwards: one 1-token EXTEND prefill and one 1-token DECODE, which respectively occupy `step0` and `step1` (so the shape lengths of `step0` and `step1` are both 1), and the formal prefill is pushed to `step2` and later. The solution is the same as for < 0.5.11: execute `start/stop/step` only for the EXTEND forward whose token count is ≥ `MSPROBE_MIN_DUMP_TOKENS`, skipping the dummy forward.

Modify `ModelRunner.forward()`: add a check before `start`, and change the trigger condition of `start/stop/step` from `self.msprobe_debugger is not None` to `_msprobe_dump`:

```diff
     def forward(self, forward_batch, ...):
         self.forward_pass_id += 1

+        # [msProbe] Collect only for the EXTEND forward whose token count is >= MSPROBE_MIN_DUMP_TOKENS,
+        # skipping the dummy forward of MSPROBE_MIN_DUMP_TOKENS tokens in the startup stage, avoiding it occupying step0.
+        _msprobe_dump = False
+        if self.msprobe_debugger is not None:
+            _min_tokens = int(os.environ.get("MSPROBE_MIN_DUMP_TOKENS", "2"))
+            _num_tokens = (
+                forward_batch.input_ids.numel()
+                if forward_batch.input_ids is not None else 0
+            )
+            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True) \
+                    and _num_tokens >= _min_tokens:
+                _msprobe_dump = True

         # Try msProbe debugger
-        if self.msprobe_debugger is not None:
+        if _msprobe_dump:
             rank_id = (
                 self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
             )
             self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

         ...
         output = ...

-        if self.msprobe_debugger is not None:
+        if _msprobe_dump:
             self.msprobe_debugger.stop()
             self.msprobe_debugger.step()

         return output
```

## 5. Startup Commands

The training-inference consistency dump mode is isolated from formal training. Before startup, the training/inference-side environment variables need to be delivered first, and then slime is started with the recommended parameters.

### 5.1 Environment Variable Configuration

slime starts the training worker and inference engine based on Ray. Environment variables must be delivered through `env_vars` of the `runtime-env-json` of `ray job submit` so that Ray subprocesses also take effect.

**Training side**:

```json
{
  "env_vars": {
    "DUMP_ON": "1",
    "PROMPTS_ONLY": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "MSPROBE_SEED": "1234",
    "MSPROBE_CONFIG_PATH": "/path/to/config_actor.json"
  }
}
```

| Variable | Description |
|------|------|
| `DUMP_ON=1` | Master switch of training-side msProbe; if 0 or not set, it is consistent with the original logic |
| `PROMPTS_ONLY=1` | Training runs only the old_log_prob prompt segment and skips actor training |
| `MSPROBE_CONFIG_PATH` | Path of the training-side config.json |
| `MSPROBE_SEED` | Fixed random seed, ensuring that results are reproducible across multiple runs |
| `TORCHDYNAMO_DISABLE=1` | Globally disables dynamo when a dynamo error occurs |

**Inference side**:

SGLang **< 0.5.11** (intrusive instrumentation) must deliver:

```json
{
  "env_vars": {
    "SGLANG_MSPROBE_DUMP": "1",
    "MSPROBE_GENERATE_CONFIG": "/path/to/config_generate.json",
    "MSPROBE_MIN_DUMP_TOKENS": "2",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0"
  }
}
```

SGLang **≥ 0.5.11** (built-in msProbe) only needs to deliver `MSPROBE_MIN_DUMP_TOKENS` and `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION`, and the collection configuration is specified through the startup parameter `--sglang-msprobe-dump-config` (see [5.2 Recommended Startup Parameters](#52-recommended-startup-parameters)):

```json
{
  "env_vars": {
    "MSPROBE_MIN_DUMP_TOKENS": "2",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0"
  }
}
```

| Variable | Description |
|------|------|
| `SGLANG_MSPROBE_DUMP=1` | Inference-side msProbe switch (used only for < 0.5.11 intrusive instrumentation) |
| `MSPROBE_GENERATE_CONFIG` | Path of the inference-side config.json (used only for < 0.5.11 intrusive instrumentation) |
| `MSPROBE_MIN_DUMP_TOKENS` | Filters the dummy forward in the startup stage (1 token each for prefill and decode), avoiding them occupying `step0`/`step1` (must be delivered for both versions) |
| `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` | Disables the generation probe of the `/health` endpoint, avoiding msProbe collecting data of the unnecessary /health request stage |

Because when slime starts SGLang, the health check `/health_generate` (`input_ids=[0]`, `max_new_tokens=1`) produces **one 1-token dummy prefill** and **one 1-token dummy decode** forward, `MSPROBE_MIN_DUMP_TOKENS` is used to **filter these dummy forwards in the startup stage**:

| Forward Type | Typical Token Count | When `MSPROBE_MIN_DUMP_TOKENS=2` |
|----------|--------------|-------------------------------|
| Startup dummy prefill (EXTEND) | 1 | **Skipped** (token count < threshold), not dumped |
| Startup dummy decode (DECODE) | 1 | **Skipped** (not EXTEND), not dumped |
| Formal rollout prefill | prompt length | Collected, as generate `step0` |

### 5.2 Recommended Startup Parameters

```shell
# rollout: only 1 time; 1 prompt, n-samples = DP
--num-rollout 1
--rollout-batch-size 1
--n-samples-per-prompt ${DP}    # DP = world_size / TP / PP / CP
--global-batch-size ${DP}

# Training parallelism (TP=1 recommended, no padding constraint)
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 1
--data-pad-size-multiplier 1
--no-gradient-accumulation-fusion
# Do not pass --balance-data, --use-dynamic-batch-size

# SGLang (training-inference TP/DP aligned with §3.4; graph mode see §3.7)
--sglang-disable-cuda-graph
--sglang-chunked-prefill-size 16384   # Must satisfy chunked_prefill_size/dp_size > prompt length
```

**4-rank example** (`TP=1, PP=1, CP=1` → `DP=4`):

```shell
--rollout-batch-size 1
--n-samples-per-prompt 4
--global-batch-size 4

--tensor-model-parallel-size 1

--rollout-num-gpus-per-engine 4   # When DP-Attention is enabled, this value is DP rather than TP; here DP=4, TP=1, consistent with the training-side TP=1
--sglang-dp-size 4
--sglang-enable-dp-attention
```

In dump mode, it is recommended to disable eval and not save checkpoints, and to write logs to `run.log` in the experiment directory.

## 6. dump Result Directory

The dump directory structure of the training-inference consistency scenario is as follows (taking the `dump_path` root directory `/example_dump_path/msprobe_dump` as an example):

```text
/example_dump_path/msprobe_dump/
├── generate/                 # Inference prefill collection data
│   └── step0/
│       └── rank{ID}/
│           ├── dump.json
│           ├── stack.json
│           └── construct.json
└── old_log_prob/             # Training old_log_prob collection data
    └── step0/
        └── rank{ID}/
            ├── dump.json
            ├── stack.json
            └── construct.json
```

Description of directories and files at each level:

- **rank{ID}**: Device ID. The data of each rank is saved in the corresponding rank directory.
- **dump.json**: The statistics of the forward/backward data of the API or Module (Max/Min/Mean/L2 Norm, etc.). For details, see "[Precision Data Collection in the PyTorch Scenario](./pytorch_data_dump_instruct.md)".
- **stack.json**: Call stack information of the API/Module.
- **construct.json**: Hierarchical structure information.

**rank alignment note**: In the training-inference decoupled architecture, training and inference run on different ranks, and the rank IDs of the two sides usually cannot correspond one-to-one (for example, in a 4-rank scenario: the first 2 ranks run inference and the last 2 ranks run training, so the training-side rank directory may start from `rank2` and is not aligned with `rank0` on the generate side). Therefore, visual comparison **can only be performed rank by rank**, and the path must be specified to a specific `rank` directory; multi-rank batch comparison cannot be used.

## 7. msProbe Visual Comparison

```shell
msprobe graph_visualize \
  -tp /example_dump_path/msprobe_dump/generate/step0/rank0 \
  -gp /example_dump_path/msprobe_dump/old_log_prob/step0/rank0 \
  -o /example_dump_path/msprobe_vis

tensorboard --logdir /example_dump_path/msprobe_vis
```

During comparison, `-tp` specifies the inference (generate) data and `-gp` specifies the training (actor) data; it must be compared rank by rank, with the path specified to the corresponding `rank` directory.
