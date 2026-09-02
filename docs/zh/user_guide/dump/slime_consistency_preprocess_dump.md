# slime框架训推一致性预处理与数据采集

## 1. 简介

slime 框架采用"训练侧 Megatron + 推理侧 SGLang"的训推分离架构。

在定位精度差异时，由于训练侧 `old_log_prob` 默认输入为 prompt + response，而推理侧 prefill 输入仅为 prompt，导致两侧输入 token 不一致，无法直接比对 dump 出的统计量。

本文档专门针对训推一致性精度定位场景，提供完整的预处理与数据采集流程：

1. 通过预处理统一两侧 prefill 阶段的输入 token，确保数据满足并行切分、batch 及 padding 等工程约束。参见[训推一致性预处理](#3-训推一致性预处理)。
2. 仅采集训练侧 `old_log_prob` 与推理侧 prefill 对应阶段的数据，用于逐 rank 比对以快速定位精度差异来源。参见[训推两侧 msProbe 数据采集](#4-训推两侧-msprobe-数据采集)。

对于不要求训推输入对齐的场景，可使用 slime 框架通用的采集方式，详细介绍请参见《[slime 框架训推精度数据采集](./slime_train_rollout_dump_instruct.md)》。

## 2. 环境准备

安装 msProbe 工具，详情请参见《[msProbe 工具安装指南](../../install_guide/msprobe_install_guide.md)》。

```shell
pip install mindstudio-probe
```

运行环境与镜像：

**Dockerfile构建镜像**：可基于 [Slime_ascend Dockerfile](https://gitcode.com/Ascend/slime-ascend/tree/main/docker/npu_docker) 构建镜像和环境。

## 3. 训推一致性预处理

训推一致性比对前，须保证训练和推理时的**输入 token 与序列长度**一致，并满足**并行切分、batch 配置**等约束，才能确保 dump 的精度数据可匹配比对。

预处理目标（两侧分别要达到的效果）如下：

| 目标侧 | 目标效果 | 对应小节 |
|----|----------|----------|
| **推理侧** | SGLang 第一次 EXTEND（prefill）输入为 prompt token；关闭图模式，并满足 DP / chunked prefill、并行切分、序列长度对齐等约束，保证采集到正式 prefill 前向 | [3.2](#32-序列长度与-padding-约束)、[3.3](#33-sglang-dp-与-chunked-prefill-约束)、[3.4](#34-训推并行切分一致)、[3.7](#37-关闭-cuda-graph-图模式) |
| **训练侧** | Megatron `old_log_prob` 输入截断为 **prompt-only**（去掉 response），与推理 prefill 对齐；同时满足 padding、单条 prompt、禁用数据均衡等约束 | [3.1](#31-prompt-only-输入对齐)、[3.2](#32-序列长度与-padding-约束)、[3.5](#35-单条-prompt-输入与-dp-配置)、[3.6](#36-禁用数据均衡与动态-batch) |

### 3.1 prompt-only 输入对齐

**预处理动作**：将训练侧 `old_log_prob` 的输入从「prompt + response」截断为仅含 prompt，使其与推理侧 prefill 输入一致。

**原因**：slime 正常 GRPO 链路中，rollout 产出 `sample.tokens`（prompt + response），训练侧 `old_log_prob` 默认对完整序列计算 log prob。训推一致性场景只需比对 **prefill 段**，因此训练侧须去掉 response，仅保留 prompt token。

**代码实现方式**：见 [4.2 训练阶段数据采集](#42-训练阶段数据采集)：新增 `consistent_dump_utils.py`，在 `DUMP_ON=1` 且 `PROMPTS_ONLY=1` 时，于 `train_actor` 入口对 rollout batch 做 prompt-only 副本后再执行 `compute_log_prob`，并在采集完成后**跳过 actor 训练**。

### 3.2 序列长度与 padding 约束

为了保证训练和推理时使用的模型权重完全一致，并消除因 padding 策略不同导致的浮点精度差异，训练侧和推理侧必须使用相同的序列长度对齐策略。

建议将训练侧的填充系数 `--data-pad-size-multiplier` 设置为 `1`，确保序列长度仅对齐到张量并行度（TP）的倍数，从而最小化 Padding Token 的数量和潜在的数值误差。

#### 训练侧 Megatron 配置

训练侧 Megatron 在 `get_batch` 阶段，会根据以下公式计算每个序列需要填充到的目标长度：

```python
pad_size = tensor_model_parallel_world_size * data_pad_size_multiplier
```

- **tensor_model_parallel_world_size（TP）**：张量并行度。例如使用 4 张卡做张量并行训练，则 TP=4。TP 通常配置为 2 的幂次方，具体值取决于卡的数量。
- **data_pad_size_multiplier（multiplier）**：填充步长的系数。保证所有输入序列的长度是 `TP × multiplier` 的整数倍。multiplier 默认值为 128，建议配置为 1。

#### 推理侧（SGLang）数据预处理要求

SGLang 本身支持任意长度的动态输入，但为了与 Megatron 训练侧对齐，必须在将数据传入 SGLang 之前，在预处理阶段按与训练侧相同的 `pad_size` 完成序列长度对齐。

具体操作取决于训练侧的配置：

| 配置 | pad_size | 说明 |
|------|----------|------|
| **推荐** TP=1，multiplier=1 | 1 | 无需填充。直接使用原始 prompt 长度即可，训练侧与推理侧天然对齐。 |
| TP>1，multiplier=1 | TP | 必须填充。将 prompt 长度补齐到 TP 的整数倍。 |
| TP>1，multiplier=128 | TP × 128 | 必须填充。将 prompt 长度补齐到 `TP × 128` 的整数倍。由于填充过多，此场景不推荐用于精度比对。 |

dump 模式须设置 `--data-pad-size-multiplier 1`；若训练 TP>1，须保证传入两侧的 prompt 长度均为 `TP × multiplier` 的整数倍。

### 3.3 SGLang DP 与 chunked prefill 约束

启用 `--sglang-dp-size > 1` 时，须满足：

```text
sglang-chunked-prefill-size / sglang-dp-size  >  实际 prefill prompt 长度
```

### 3.4 训推并行切分一致

训推一致性比对要求训练与推理的**模型并行切分方式一致**，确保两侧各 rank 承担相同的参数分片与计算路径，dump 数据才能按 rank 一一对应。

| 维度 | 训练侧（Megatron） | 推理侧（SGLang） | 对齐要求 |
|------|-------------------|------------------|----------|
| TP | `--tensor-model-parallel-size` | `--rollout-num-gpus-per-engine`（单 engine 时即推理 TP） | 两侧 TP 须一致 |
| PP | `--pipeline-model-parallel-size` | 通常为 1 | 两侧 PP 须一致 |
| CP | `--context-parallel-size` | 通常为 1 | 两侧 CP 须一致 |
| DP | 由 world_size 与 TP/PP/CP 推导 | `--sglang-dp-size`（启用 DP-Attention 时） | 两侧 DP 语义须一致 |

### 3.5 单条 prompt 输入与 DP 配置

为保证训推两侧处理**同一条 prompt**，且训练侧 Megatron 能按 DP 正常分发样本，dump 模式须固定：

```shell
--rollout-batch-size 1          # 每轮 rollout 仅 1 条 prompt
--n-samples-per-prompt ${DP}    # 须等于 DP，保证训练正式运行
--global-batch-size ${DP}         # rollout-batch-size × n-samples-per-prompt
```

数据并行度计算公式：

```text
DP = world_size / TP / PP / CP
```

其中 `world_size` 为训练参与进程总数（通常等于训练 NPU 卡数），`TP`、`PP`、`CP` 分别对应 tensor / pipeline / context 并行度。

**示例**（4 卡训练，`TP=1, PP=1, CP=1`）：

```text
DP = 4 / 1 / 1 / 1 = 4
→ --rollout-batch-size 1 --n-samples-per-prompt 4 --global-batch-size 4
```

若 `global_batch_size` 无法被 DP 整除，Megatron 训练会报错。

### 3.6 禁用数据均衡与动态 batch

训推一致性场景须关闭会改变 batch 切分或样本分布的优化项，避免训练侧各 rank 拿到的数据与推理 prefill 不对齐：

| 参数 | 要求 | 原因 |
|------|------|------|
| `--balance-data` | **禁用**（不传） | 自动均衡会重排 batch，破坏 rank 与样本的一一对应 |
| `--use-dynamic-batch-size` | **禁用**（不传） | 动态 batch 会改变 micro-batch 切分，影响 dump 可比对性 |

dump 模式下同时建议关闭 `--rollout-shuffle`、`--over-sampling-batch-size` 等会引入随机采样的参数，保证每轮输入确定可复现。

### 3.7 关闭 CUDA Graph 图模式

训推一致性 dump 须关闭 SGLang 的 CUDA Graph（昇腾场景下为图编译缓存）优化，否则 msProbe 无法正确钩住每次 EXTEND 前向的算子，dump 数据不完整或与 eager 模式不一致。

slime 启动 SGLang 时须指定：

```shell
--sglang-disable-cuda-graph
```

## 4. 训推两侧 msProbe 数据采集

完成第 3 章预处理约束后，按本节分别配置并采集推理侧（generate / prefill）与训练侧（old_log_prob）的 msProbe 数据。

### 4.1 创建配置文件

训推两侧各需一份 `config.json`。采集统计量时 `task` 配置为 `statistics`，示例如下：

**推理侧（config_generate.json）**：

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

**训练侧（config_actor.json）**：

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

主要参数说明：

- **task**：`statistics` 表示采集统计量；若需真实张量，配置为 `tensor`。
- **dump_path**：dump 保存根路径；训练侧代码会在 `dump_path` 下动态拼接 `old_log_prob` 子目录落盘，推理侧直接指向 `generate` 子目录落盘。
- **step**：采集步数，训推一致性 dump 通常仅采 `step0`。
- **level**：`mix` 表示同时采集 Module 级和 API 级数据。

以上配置参数的详细介绍请参见《[配置文件介绍](./config_json_introduct.md)》。

### 4.2 训练阶段数据采集

训练阶段精度数据采集在 slime 的 Megatron 训练后端完成。在 `slime/backends/megatron_utils/actor.py` 中实例化 `PrecisionDebugger`，在 **old_log_prob 前向**前后调用 `start`、`stop` 接口，并在 stop 之后调用 `step()` 推进步数。`PrecisionDebugger` 接口更多介绍请参见《[PyTorch 场景精度数据采集](./pytorch_data_dump_instruct.md)》。

以下展示训推一致性 dump 的代码修改方式（以 slime v0.2.2 为例）。修改步骤如下。

1. 新增 `slime/backends/megatron_utils/consistent_dump_utils.py`

    ```python
    """Utilities for train-inference consistency msProbe dump."""

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

2. 在`slime/backends/megatron_utils/actor.py` 的`MegatronTrainRayActor.init` 中实例化 PrecisionDebugger

    须在 `monkey_patch_torch_dist()` **之前**实例化，避免与 `bias_swiglu_fusion` 冲突。

    ```diff
     class MegatronTrainRayActor(TrainRayActor):
         def init(
             self,
             args: Namespace,
             role: str,
             with_ref: bool = False,
         ) -> int | None:

    +        # [msProbe] 训练侧精度采集开关
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

3. 在`slime/backends/megatron_utils/actor.py` 的 `train_actor` 中采集 old_log_prob 并跳过 actor 训练

    ```diff
         def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:

    +        # [msProbe] 训推一致性 dump：仅采集 old_log_prob 前向（prompt-only），完成后跳过 actor 训练
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

说明：

- 训推一致性场景**仅采集 old_log_prob 一个阶段**；`debugger.step()` 在 `stop()` 之后调用一次即可。

### 4.3 推理阶段数据采集

slime 框架推理侧使用 SGLang 引擎执行 rollout 生成。

SGLang 从 0.5.11 版本起原生内置 msProbe 能力，因此低于 0.5.11 版本须按照本节操作进行侵入式修改，不低于 0.5.11 版本直接传配置参数。请根据 SGLang 版本选择操作方法：

| SGLang 版本 | 操作方法 |
|-------------|------|
| **< 0.5.11** | 侵入式修改 `ModelRunner`，见下文。 |
| **≥ 0.5.11** | 已原生内置msProbe工具，可直接在 `SGLANG_ARGS` 中指定参数 `--sglang-msprobe-dump-config` 进行精度数据采集。 |



在 `sglang/srt/model_executor/model_runner.py` 中插入 `PrecisionDebugger` 接口。修改步骤如下。

1. 在 `ModelRunner.__init__` 末尾实例化 debugger

    ```diff
             self.forward_pass_id = 0
             self.draft_model_idx = draft_model_idx

    +        # [msProbe] 推理侧精度采集，由 SGLANG_MSPROBE_DUMP=1 开启
    +        if int(os.environ.get("SGLANG_MSPROBE_DUMP", "0")):
    +            from msprobe.pytorch import PrecisionDebugger, seed_all
    +            seed_all(mode=True)
    +            self.debugger = PrecisionDebugger(config_path=os.environ["MSPROBE_GENERATE_CONFIG"])
    ```

2. 在 `forward()` 中对真实 prefill 采集

    仅对 token 数 ≥ `MSPROBE_MIN_DUMP_TOKENS` 的 EXTEND 前向 dump，跳过启动阶段约 1 个 token 的 dummy forward。

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

## 5. 启动命令

训推一致性 dump 模式与正式训练隔离。启动前需先下发训练/推理侧环境变量，再按建议参数启动 slime。

### 5.1 环境变量配置

slime 基于 Ray 启动训练 Worker 与推理引擎，环境变量须通过 `ray job submit` 的 `runtime-env-json` 的 `env_vars` 下发，才能保证 Ray 子进程同样生效。

**训练侧**：

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

| 变量 | 说明 |
|------|------|
| `DUMP_ON=1` | 训练侧 msProbe 总开关；0 或未设则与原始逻辑一致 |
| `PROMPTS_ONLY=1` | 训练仅跑 old_log_prob prompt 段并跳过 actor 训练 |
| `MSPROBE_CONFIG_PATH` | 训练侧 config.json 路径 |
| `MSPROBE_SEED` | 固定随机种子，保证多次运行结果可复现 |
| `TORCHDYNAMO_DISABLE=1` | 遇 dynamo 报错时全局关闭 |

**推理侧**（SGLang 版本 < 0.5.11，需侵入式插桩时）：

```json
{
  "env_vars": {
    "SGLANG_MSPROBE_DUMP": "1",
    "MSPROBE_GENERATE_CONFIG": "/path/to/config_generate.json",
    "MSPROBE_MIN_DUMP_TOKENS": "2"
  }
}
```

| 变量 | 说明 |
|------|------|
| `SGLANG_MSPROBE_DUMP=1` | 推理侧 msProbe 开关 |
| `MSPROBE_GENERATE_CONFIG` | 推理侧 config.json 路径 |
| `MSPROBE_MIN_DUMP_TOKENS` | 过滤启动阶段约 1 个 token 的 dummy EXTEND forward，避免采集 warmup 探针请求（见下表） |

因 SGLang engine 初始化时会执行一次约 **1 个 token** 的 dummy EXTEND forward（用于 warmup / 图编译探测），`MSPROBE_MIN_DUMP_TOKENS` 用于**过滤推理侧极短 EXTEND 前向**：

| 前向类型 | 典型 token 数 | `MSPROBE_MIN_DUMP_TOKENS=2` 时 |
|----------|--------------|-------------------------------|
| 启动 dummy forward | 约 1 | **跳过**，不 dump |
| 正式 rollout prefill | prompt 长度 | 采集，作为 generate `step0` |

### 5.2 启动参数建议

```shell
# rollout：仅 1 次；1 条 prompt，n-samples = DP
--num-rollout 1
--rollout-batch-size 1
--n-samples-per-prompt ${DP}    # DP = world_size / TP / PP / CP
--global-batch-size ${DP}

# 训练并行（推荐 TP=1，无填充约束）
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 1
--data-pad-size-multiplier 1
--no-gradient-accumulation-fusion
# 勿传 --balance-data、--use-dynamic-batch-size

# SGLang（训推 TP/DP 与 §3.4 对齐；图模式见 §3.7）
--sglang-disable-cuda-graph
--sglang-chunked-prefill-size 16384   # 须满足 chunked_prefill_size/dp_size > prompt 长度
```

**4 卡示例**（`TP=1, PP=1, CP=1` → `DP=4`）：

```shell
--rollout-batch-size 1 
--n-samples-per-prompt 4 
--global-batch-size 4

--tensor-model-parallel-size 1

--rollout-num-gpus-per-engine 4   # 启用 DP-Attention 时该值为 DP 而非 TP，此处 DP=4、TP=1，与训练侧 TP=1 一致
--sglang-dp-size 4 
--sglang-enable-dp-attention
```

dump 模式建议关闭 eval、不保存 checkpoint，并将日志写入实验目录 `run.log`。

## 6. dump 结果目录

训推一致性场景的 dump 落盘目录结构如下（以 `dump_path` 根目录 `/example_dump_path/msprobe_dump` 为例）：

```text
/example_dump_path/msprobe_dump/
├── generate/                 # 推理 prefill 采集数据
│   └── step0/
│       └── rank{ID}/
│           ├── dump.json
│           ├── stack.json
│           └── construct.json
└── old_log_prob/             # 训练 old_log_prob 采集数据
    └── step0/
        └── rank{ID}/
            ├── dump.json
            ├── stack.json
            └── construct.json
```

各级目录及文件说明：

- **rank{ID}**：设备 ID，每张卡的数据保存在对应 rank 目录下。
- **dump.json**：API 或 Module 前反向数据的统计量（Max/Min/Mean/L2 Norm 等）。详见《[PyTorch 场景精度数据采集](./pytorch_data_dump_instruct.md)》。
- **stack.json**：API/Module 的调用栈信息。
- **construct.json**：分层分级结构信息。

**rank 对齐说明**：训推分离架构下，训练与推理运行在不同卡上，两侧 rank ID 通常无法一一对应（例如 4 卡场景：前 2 张卡跑推理、后 2 张卡跑训练，generate 侧 rank 目录可能从 `rank2` 起，与训练侧 `rank0` 不对齐）。因此可视化比对**只能按单卡逐 rank 进行**，须将路径指定到具体的 `rank` 目录，不能使用多卡批量比对。

## 7. msProbe 可视化比对

```shell
msprobe graph_visualize \
  -tp /example_dump_path/msprobe_dump/generate/step0/rank0 \
  -gp /example_dump_path/msprobe_dump/old_log_prob/step0/rank0 \
  -o /example_dump_path/msprobe_vis

tensorboard --logdir /example_dump_path/msprobe_vis
```

比对时 `-tp` 指定推理（generate）数据，`-gp` 指定训练（actor）数据；须逐 rank 单卡比对，路径指定到对应 `rank` 目录。
