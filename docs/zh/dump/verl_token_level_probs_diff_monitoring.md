# 训推一致性监控：逐Token级别的probs_diff监控

## 1. 概述

### 1.1 背景

在大规模强化学习训练（GRPO/PPO）中，一条 response token 序列会经过两条路径分别计算对数概率：

| 路径 | 阶段 | 引擎 | 参数来源 |
|------|------|------|----------|
| rollout | vLLM 推理生成 response | vLLM engine | 上一个同步周期的 base model 权重 |
| actor 前向 | 训练时重新 forward | 训练引擎（如 Megatron） | 当前 actor model 权重（decoupled 模式下可能使用历史版本） |

理论上，若两条路径使用的模型权重一致，对相同 token 计算出的概率应当一致。但在实际训练中可能出现偏差：

- **权重同步滞后**：actor 更新后，vLLM engine 的权重尚未同步
- **算子精度差异**：不同硬件（昇腾 NPU / GPU）或不同框架（vLLM / Megatron）的算子实现存在数值精度差异
- **推理/训练模式差异**：dropout、flash attention 等在前向推理和训练时的行为不同

### 1.2 功能定位

本模块是一个 **可开关的调试工具**，仅在开启时产生额外开销，不影响正常训练流程。其核心能力：

1. 逐 token 计算 rollout 与 actor 的概率差异（probability diff）
2. 将前 N 个 sample × 前 M 个 position 的差异值以网格形式输出到训练日志
3. 将完整的 2D diff 矩阵和 mask 矩阵保存到磁盘，供离线分析

### 1.3 适用场景

| 场景 | 说明 |
|------|------|
| 训推一致性排查 | 怀疑 vLLM 权重同步延迟导致训练效果异常时 |
| 精度对比 | 切换硬件平台（如 GPU → 昇腾 NPU）后验证推理/训练一致性 |
| 调试阶段排查 | 确认训练各阶段 prob diff 是否正常 |

## 2. 启动配置

### 2.1 前提条件

必须设置以下参数，否则 vLLM 不会返回 `rollout_log_probs`，diff 无法计算：

```
actor_rollout_ref.rollout.calculate_log_probs=True
```

### 2.2 启动命令示例

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +trainer.enable_token_level_prob_diff=True \
    +trainer.prob_diff_save_dir="/path/to/save" \
    +trainer.prob_diff_token_max_print=10 \
    +trainer.prob_diff_sample_max_print=8 \
    ...
```

### 2.3 自定义配置说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `actor_rollout_ref.rollout.calculate_log_probs` | bool | false | **前提配置**，vLLM 返回 rollout_log_probs |
| `+trainer.enable_token_level_prob_diff` | bool | false | 开启逐 token prob diff 监控 |
| `+trainer.prob_diff_save_dir` | str | None | npy 文件保存目录，None 不保存 |
| `+trainer.prob_diff_token_max_print` | int | 10 | 每个 sample 输出的 position 数 |
| `+trainer.prob_diff_sample_max_print` | int | 8 | 输出的 sample 数 |

> 带 `+` 前缀的参数是 Hydra 未预定义的配置项，需要在命令行参数中加 `+` 号。

### 2.4 注意事项

1. **前提条件**：必须设置 `actor_rollout_ref.rollout.calculate_log_probs=True`。
2. **日志量**：开启后每步增加 `sample_max_print × token_max_print` 个指标，建议仅在调试阶段开启。
3. **文件保存**：npy 文件只在单 driver 进程上保存一次，不存在多副本问题。

## 3. 代码改动

### 3.1 文件清单

| 文件 | 类型 | 变更 |
|------|------|------|
| `verl/utils/debug/metrics.py` | 已有文件 | 新增 `calculate_token_level_prob_diff` 函数 |
| `verl/trainer/ppo/ray_trainer.py` | 已有文件 | 在 `fit()` 中新增条件调用 |

### 3.2 `verl/utils/debug/metrics.py`

在文件末尾新增 `calculate_token_level_prob_diff` 函数：

```diff
+import os
+import numpy as np
+
+def calculate_token_level_prob_diff(
+    data: DataProto,
+    save_dir: str | None = None,
+    step: int | None = None,
+    token_max_print: int = 10,
+    sample_max_print: int = 8,
+) -> dict:
+    rollout_log_probs = data.batch.get("rollout_log_probs")
+    actor_log_probs = data.batch.get("old_log_probs")
+    if rollout_log_probs is None or actor_log_probs is None:
+        return {}
+
+    responses = data.batch.get("responses")
+    if responses is None:
+        return {}
+    response_length = responses.size(1)
+
+    if "response_mask" in data.batch:
+        mask = data.batch["response_mask"]
+    elif "attention_mask" in data.batch:
+        mask = data.batch["attention_mask"][:, -response_length:]
+    else:
+        return {}
+
+    rollout_probs = torch.exp(rollout_log_probs)
+    actor_probs = torch.exp(actor_log_probs)
+    diff = torch.abs(rollout_probs - actor_probs)
+
+    mask_bool = mask.bool()
+    masked_diff = diff * mask_bool
+
+    metrics = {}
+    batch_size = min(masked_diff.size(0), sample_max_print)
+    n_tokens = min(response_length, token_max_print)
+    for sample_id in range(batch_size):
+        for position_id in range(n_tokens):
+            if mask_bool[sample_id, position_id]:
+                metrics[f"training/rollout_probs_diff/s{sample_id}_p{position_id:04d}"] = \
+                    masked_diff[sample_id, position_id].detach().item()
+
+    if save_dir is not None:
+        os.makedirs(save_dir, exist_ok=True)
+        step_str = f"_step_{step}" if step is not None else ""
+        np.save(os.path.join(save_dir, f"prob_diff{step_str}.npy"),
+                masked_diff.detach().cpu().numpy())
+        np.save(os.path.join(save_dir, f"prob_mask{step_str}.npy"),
+                mask_bool.detach().cpu().numpy())
+
+    return metrics
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data` | `DataProto` | 必填 | 包含 `rollout_log_probs`、`old_log_probs`、`responses`、`response_mask` |
| `save_dir` | `str \| None` | `None` | npy 文件保存目录，`None` 时不保存文件 |
| `step` | `int \| None` | `None` | 当前训练步数，用于文件名（如 `prob_diff_step_10.npy`） |
| `token_max_print` | `int` | `10` | 每个 sample 输出前多少个 position |
| `sample_max_print` | `int` | `8` | 输出前多少个 sample |

**返回值：** `dict`，key 格式为 `training/rollout_probs_diff/s{sample_id}_p{position_id:04d}`。

**边界情况：**

| 输入条件 | 行为 |
|----------|------|
| `rollout_log_probs` 或 `old_log_probs` 缺失 | 返回空 dict |
| `responses` 缺失 | 返回空 dict |
| `response_mask` 和 `attention_mask` 均缺失 | 返回空 dict |
| mask 全为 False | 网格不输出任何值，prob_diff.npy 保存全 0 矩阵，prob_mask.npy 保存全 False 的 bool 矩阵|

### 3.3 `verl/trainer/ppo/ray_trainer.py`

在 `RayPPOTrainer.fit()` 的 `calculate_debug_metrics` 调用之后插入：

```diff
                             if "rollout_log_probs" in batch.batch.keys():
                                 from verl.utils.debug.metrics import calculate_debug_metrics

                                 metrics.update(calculate_debug_metrics(batch))

+                                # Token-level prob diff (optional, for debug/analysis)
+                                if self.config.trainer.get("enable_token_level_prob_diff", False):
+                                    from verl.utils.debug.metrics import calculate_token_level_prob_diff
+                                    metrics.update(
+                                        calculate_token_level_prob_diff(
+                                            batch,
+                                            save_dir=self.config.trainer.get("prob_diff_save_dir", None),
+                                            step=self.global_steps,
+                                            token_max_print=self.config.trainer.get("prob_diff_token_max_print", 10),
+                                            sample_max_print=self.config.trainer.get("prob_diff_sample_max_print", 8),
+                                        )
+                                    )
```

## 4. 输出结果说明

### 4.1 训练日志指标

当 `enable_token_level_prob_diff=True` 时，每步训练日志中会输出以下格式的指标：

```
training/rollout_probs_diff/s0_p0000    0.0012
training/rollout_probs_diff/s0_p0001    0.0008
training/rollout_probs_diff/s0_p0002    0.0023
...
training/rollout_probs_diff/s1_p0000    0.0015
training/rollout_probs_diff/s1_p0001    0.0009
...
```

> 指标数量 = `min(batch_size, sample_max_print) × min(response_length, token_max_print)`，
> 仅包含 mask=1 的有效位置。默认最多 8×10=80 个指标。

同时，原有的聚合指标继续输出：

```
training/rollout_probs_diff_max            0.0234
training/rollout_probs_diff_mean           0.0012
training/rollout_probs_diff_std            0.0031
training/rollout_actor_probs_pearson_corr  0.9987
```

### 4.2 与 `calculate_debug_metrics` 的关系

| 函数 | 输出 | 定位 |
|------|------|------|
| `calculate_debug_metrics` | 聚合统计（max/mean/std/pearson corrcoef） | 宏观概览，适合监控和告警 |
| `calculate_token_level_prob_diff` | 逐 token 明细 + npy 文件 | 微观排查，适合调试和离线分析 |

二者互补，共享同一数据源。

### 4.3 磁盘文件

#### 4.3.1 文件格式

完整的 2D diff 矩阵和 mask 矩阵保存到磁盘，供离线分析。当指定 `prob_diff_save_dir` 时，每步训练在该目录下生成两个文件：

| 文件 | 形状 | 内容 |
|------|------|------|
| `prob_diff_step_{step}.npy` | `[batch, response_length]` | float32 diff 矩阵，padding 位置为 0 |
| `prob_mask_step_{step}.npy` | `[batch, response_length]` | bool mask，1=有效位置，0=padding |

#### 4.3.2 离线分析示例

```python
import numpy as np

diff = np.load("prob_diff_step_10.npy")
mask = np.load("prob_mask_step_10.npy")

# 收集所有有效位置的 diff
valid_diffs = diff[mask]

print(f"有效 token 数: {len(valid_diffs)}")
print(f"mean: {valid_diffs.mean():.6f}")
print(f"max:  {valid_diffs.max():.6f}")
print(f"std:  {valid_diffs.std():.6f}")

# 按 position 聚合（跨 batch），观察是否存在特定位置偏差
mask_sum = mask.sum(axis=0)
pos_means = np.where(mask_sum > 0, (diff * mask).sum(axis=0) / mask_sum, 0)
```
