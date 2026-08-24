# Training-Inference Consistency Monitoring: Token-Level probs_diff Monitoring

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:43:13.955Z pushedAt=2026-08-11T02:51:07.158Z -->

## 1. Overview

### 1.1 Background

In large-scale reinforcement learning training (GRPO/PPO), a response token sequence passes through two paths to compute log probabilities:

| Path | Stage | Engine | Parameter Source |
|------|------|------|----------|
| rollout | vLLM inference generates response | vLLM engine | Base model weights from the previous synchronization cycle |
| actor forward | Re-forward during training | Training engine (e.g., Megatron) | Current actor model weights (may use a historical version in decoupled mode) |

In theory, if the model weights used by both paths are identical, the probabilities computed for the same token should be consistent. However, deviations may occur in actual training:

- **Weight synchronization delay**: After the actor is updated, the vLLM engine's weights have not yet been synchronized.
- **Operator precision differences**: Numerical precision differences exist in operator implementations across different hardware (Ascend NPU/GPU) or different frameworks (vLLM/Megatron).
- **Inference/training mode differences**: Dropout, flash attention, and other parts behave differently during forward inference and training.

### 1.2 Positioning

This module is a **switchable debugging tool** that incurs additional overhead only when enabled and does not affect the normal training workflow. Its core capabilities:

1. Compute the probability difference (`prob diff`) between rollout and actor on a per-token basis.
2. Output the `diff` values of the first N samples × first M positions to the training log in a grid format.
3. Save the complete 2D `diff` matrix and mask matrix to disk for offline analysis.

### 1.3 Applicable Scenarios

| Scenario | Description |
|------|------|
| Training-inference consistency troubleshooting | When abnormal training results are suspected to be caused by vLLM weight synchronization delays |
| Precision comparison | Verifying inference/training consistency after switching hardware platforms (e.g., GPU → Ascend NPU) |
| Debugging phase troubleshooting | Confirming whether `prob diff` is normal at each training stage |

## 2. Startup Configuration

### 2.1 Prerequisites

The following parameter must be set; otherwise, vLLM will not return `rollout_log_probs`, and `diff` cannot be calculated:

```bash
actor_rollout_ref.rollout.calculate_log_probs=True
```

### 2.2 Startup Command Example

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

### 2.3 Custom Configuration

| Parameter | Type | Default Value | Description |
|--------|------|--------|------|
| `actor_rollout_ref.rollout.calculate_log_probs` | bool | false | **Prerequisite configuration**. vLLM returns `rollout_log_probs`. |
| `+trainer.enable_token_level_prob_diff` | bool | false | Enable per-token `prob diff` monitoring. |
| `+trainer.prob_diff_save_dir` | str | None | Directory for saving `.npy` files. `None` means no file is saved. |
| `+trainer.prob_diff_token_max_print` | int | 10 | Number of positions to output per sample. |
| `+trainer.prob_diff_sample_max_print` | int | 8 | Number of samples to output. |

> Parameters prefixed with `+` are configuration items not predefined by Hydra and must be specified with a `+` sign in command-line.

### 2.4 Important Notes

1. **Prerequisite**: `actor_rollout_ref.rollout.calculate_log_probs=True` must be set.
2. **Log Volume**: After enabling, `sample_max_print × token_max_print` metrics are added per step. It is recommended to enable this only during the debugging phase.
3. **File Saving**: The `.npy` file is saved only once on a single driver process, so there is no multi-copy issue.

## 3. Code Changes

### 3.1 File List

| File | Type | Change |
|------|------|------|
| `verl/utils/debug/metrics.py` | Existing file | Added `calculate_token_level_prob_diff` function |
| `verl/trainer/ppo/ray_trainer.py` | Existing file | Added conditional invocation in `fit()` |

### 3.2 `verl/utils/debug/metrics.py`

Add the `calculate_token_level_prob_diff` function at the end of the file:

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

**Parameters**

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| `data` | `DataProto` | - | Mandatory. Contains `rollout_log_probs`, `old_log_probs`, `responses`, and `response_mask` |
| `save_dir` | `str \| None` | `None` | Directory for saving `.npy` files; no files are saved when set to `None` |
| `step` | `int \| None` | `None` | Current training step, used in the file name (e.g., `prob_diff_step_10.npy`) |
| `token_max_print` | `int` | `10` | Number of leading positions to output per sample |
| `sample_max_print` | `int` | `8` | Number of leading samples to output |

**Return value:** `dict`, with keys in the format of `training/rollout_probs_diff/s{sample_id}_p{position_id:04d}`.

**Edge Cases:**

| Input Condition | Behavior |
|----------|------|
| `rollout_log_probs` or `old_log_probs` is missing | Return empty dict |
| `responses` is missing | Return empty dict |
| Both `response_mask` and `attention_mask` are missing | Return empty dict |
| All mask values are `False` | The grid outputs no values; `prob_diff.npy` saves an all-zero matrix, and `prob_mask.npy` saves an all-False bool matrix |

### 3.3 `verl/trainer/ppo/ray_trainer.py`

Insert after the `calculate_debug_metrics` call in `RayPPOTrainer.fit()`:

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

## 4. Output Result

### 4.1 Training Log Metrics

When `enable_token_level_prob_diff=True`, the following metrics are output in the training log of each step:

```ColdFusion
training/rollout_probs_diff/s0_p0000    0.0012
training/rollout_probs_diff/s0_p0001    0.0008
training/rollout_probs_diff/s0_p0002    0.0023
...
training/rollout_probs_diff/s1_p0000    0.0015
training/rollout_probs_diff/s1_p0001    0.0009
...
```

> Number of metrics = `min(batch_size, sample_max_print) × min(response_length, token_max_print)`.
> Include only valid positions where `mask=1`. By default, there are at most `8 × 10 = 80` metrics.

Meanwhile, the original aggregated metrics continue to be output:

```ColdFusion
training/rollout_probs_diff_max            0.0234
training/rollout_probs_diff_mean           0.0012
training/rollout_probs_diff_std            0.0031
training/rollout_actor_probs_pearson_corr  0.9987
```

### 4.2 Relationship with `calculate_debug_metrics`

| Function | Output | Positioning |
|------|------|------|
| `calculate_debug_metrics` | Aggregated statistics (`max`/`mean`/`std`/`pearson corrcoef`) | Macro-level overview, suitable for monitoring and alerting |
| `calculate_token_level_prob_diff` | Per-token details + `.npy` files | Micro-level investigation, suitable for debugging and offline analysis |

The two are complementary and share the same data source.

### 4.3 Disk File

#### 4.3.1 File Format

The complete 2D `diff` matrix and mask matrix are saved to disk for offline analysis. When `prob_diff_save_dir` is specified, two files are generated in that directory at each training step:

| File | Shape | Content |
|------|------|------|
| `prob_diff_step_{step}.npy` | `[batch, response_length]` | float32 `diff` matrix, with `0` at padding positions |
| `prob_mask_step_{step}.npy` | `[batch, response_length]` | bool mask, `1` for valid position, `0` for padding |

#### 4.3.2 Offline Analysis Example

```python
import numpy as np

diff = np.load("prob_diff_step_10.npy")
mask = np.load("prob_mask_step_10.npy")

# Collect diff values at all valid positions.
valid_diffs = diff[mask]

print(f"Number of valid tokens: {len(valid_diffs)}")
print(f"mean: {valid_diffs.mean():.6f}")
print(f"max:  {valid_diffs.max():.6f}")
print(f"std:  {valid_diffs.std():.6f}")

# Aggregate by position (across batches) to observe whether position-specific bias exists.
mask_sum = mask.sum(axis=0)
pos_means = np.where(mask_sum > 0, (diff * mask).sum(axis=0) / mask_sum, 0)
```
