# Reinforcement Learning Training-Inference Consistency Troubleshooting

## Abstract

In the development and deployment of reinforcement learning systems, the behavioral deviation between the training and inference stages—namely "training-inference inconsistency"—is often the hidden root cause of training loss oscillation, slow convergence, and even reward collapse. This document first introduces the background and definition of training-inference consistency and points out the essential causes of training-inference inconsistency. Then, it focuses on the core technology of training-inference alignment: introducing the msProbe precision tool and its capabilities in the training-inference consistency alignment scenario. Finally, through a retrospective of a real-world case, it shows how training-inference inconsistency causes violent fluctuations in the training curve and reward collapse, and the whole process of restoring training stability and improving overall training precision through alignment methods. This document aims to provide reinforcement learning practitioners with a practical stability analysis framework and solution ideas, so that the model is not only at ease in the training environment but also consistent in inference deployment.

## 1. Background and Significance of Training-Inference Consistency

### 1.1 What Is Training-Inference Inconsistency?

**Training-inference inconsistency** refers to the **numerical inconsistency** between the rollout (inference) engine and the training engine: even if the two engines use exactly the same model weights, their computed log probabilities for the same token sequence may differ slightly. Mapped to model behavior, this difference can be summarized as the inconsistency between the behavior policy and the reference policy.

Behavior policy: The policy that actually generates the rollout, that is, "under what distribution you sampled this data". In modern LLM-RL systems, it corresponds to the implementation in the inference engine (vLLM/SGLang, etc.), and in asynchronous frameworks it is often a mixed distribution of multiple worker policies.

Reference policy: The policy used in the training objective for importance sampling, clipping, or KL constraints, typically the "old policy" in PPO/GRPO.

Target policy: The policy to be optimized in the training objective, that is, "what you want the model to become". Typically the "new policy" in PPO/GRPO.

In the most classic and idealized setting, we usually expect **behavior policy = reference policy**. However, in real systems, due to factors such as asynchronous updates, different inference/training backends, MoE routing fluctuations, and even hardware numerical differences, the two often deviate to varying degrees.

### 1.2 Why Is Training-Inference Consistency Hard to Guarantee?

#### (1) Framework Differences

As shown in the figure below, megatron is used for training and vLLM is used for inference, and the difference in the overall framework leads to inconsistent training-inference results.

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/framework_diff.png)

Looking further, within the framework, the implementation ideas for model layers with the same semantics also differ, so the results cannot be fully aligned.

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/layer_impl_diff.png)

For the same operation, such as LayerNorm, training may use small operator composition while inference uses a fusion operator, and the two may also have slight differences in precision.

```makeup
# PyTorch standard implementation (training)
y = (x - mean) / sqrt(var + eps) * gamma + beta
# Step-by-step computation: mean → var → normalize → scale → shift

# vLLM fused implementation (inference)
y = fused_rms_norm(x, weight)
# Single kernel: vectorized load + register-level fusion
# Different intermediate result truncation strategies → difference in the last few decimal places

```

#### (2) Precision Type Differences: Numerical Truncation Caused by Training FP32/BF16 and Inference FP16/INT8 Quantization

#### (3) Parallel Strategy: Tensor Parallelism During Training vs Continuous Batching During Inference, Causing Differences in Floating-Point Accumulation Order

```makeup
(0.1 + 1e20) - 1e20
>>> 0
0.1 + (1e20 - 1e20)
>>> 0.1

```

#### (4) Randomness Control: Implementation Deviations of Dropout and Sampling Strategies

### 1.3 Typical Consequences of Training-Inference Inconsistency

In theory, the training-inference differences caused by the above difference points are very small, but these differences accumulate and amplify layer by layer, causing obvious differences between the training and inference (rollout) results, and even producing the phenomenon that the output probabilities of the same token are 0 and 1 in training and inference respectively. **Such training-inference differences cause the theoretical on-policy property not to hold, thereby bringing instability and a low upper bound to RL training.**

The impact of training-inference inconsistency is not uniformly distributed; its destructive power is sharply amplified under two typical configurations:

**MoE architecture**: the "butterfly effect" of expert routing

The expert selection mechanism is extremely sensitive to numerical errors. As long as there is a slight disagreement at both ends when selecting Top-K experts—even just a ranking difference of a single expert—it will cause a complete divergence of the subsequent computation path. This disagreement is not linear accumulation but a **step-like jump**: once the wrong expert is selected, all subsequent activation values, attention patterns, and output generation will go to a completely different branch.

**Long-reasoning models**: the "snowball effect" of error propagation

The longer the model output, the more exponentially the accumulation of floating-point errors expands. In the short-output scenario, the `logprob` error of the last token may only come from the numerical noise of the last layer; while in a long reasoning chain, the small deviation of early tokens propagates layer by layer through the autoregressive mechanism, finally causing the tail distribution to shift by orders of magnitude.

Ultimately, this leads to unstable model training, a gradually soaring `logp_diff_mean`, and even reward collapse.

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/logp_diff_escalation.png)

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/reward_collapse.png)

## 2. Training-Inference Alignment Troubleshooting Ideas and Tools

This chapter takes the mainstream verl framework as an example to introduce the troubleshooting ideas for training-inference alignment in RL training under this framework and the related precision tools of the MindStudio toolchain.
At present, mainstream RL algorithms are developed based on the **On-Policy** premise. The **On-Policy** theory requires that the behavior policy for sampling data remain basically consistent with the target policy for gradient computation, so as to ensure that the gradient estimate is unbiased and make the training process more stable. In RL, sampling is called **rollout** inference, and gradient computation corresponds to **actor** training. When the inference and training policies remain consistent, it is called training-inference consistency. The main metric of training-inference consistency is **logp_diff_mean**. When **logp_diff_mean** is abnormal, it indicates that there is a certain degree of difference between training and inference in RL, and the root cause of the training-inference difference needs to be found. Its calculation formula is:
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/logp_diff_formula.png)
Where, `M` is `response_mask`.

In the verl framework, the corresponding behavior is as follows:

* Configuration hyperparameters for enabling `logp_diff` monitoring:

  ```makeup
  actor_rollout_ref.rollout.calculate_log_probs=True  # Default is False

  ```

* Specific code implementation of `logp_diff`:

  ```makeup
  mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
  mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
  log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
  metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()

  ```

When the metric satisfies the following condition, training-inference consistency troubleshooting needs to be started:

* **The logp_diff of the first step is abnormally large (>0.01) or clearly different from the benchmark.**

### 2.1 kv cache Troubleshooting

In the inference process:

* **prefill**: Clear the **kv cache** of the previous round and write the current context.
* **decode**: Incrementally read and write the **kv cache**.

In actual engineering implementations, the cache is usually managed through **state block** rather than forcibly cleared sentence by sentence. If problems such as untimely cache cleanup, incorrect reuse, or address offset occur, they will cause abnormal reads in the **decode** stage, further causing long-term stability drift in inference and finally making **logp_diff** larger in the later stage.
In this case, since **kvcache** is logic specific to inference and unrelated to training, it will definitely trigger **logp_diff** inconsistency in RL training.

For kv cache troubleshooting, the following two schemes can be used:

1. Compare whether there are logical differences in the **kv cache** code read/write parts between **NPU** and the benchmark.
2. Forcibly clear the **kv cache** sentence by sentence.

### 2.2 Training-Inference Consistency Comparison

At present, training-inference consistency comparison only supports the **prefill** stage, so it is necessary to first determine whether the consistency difference occurs only in the **decode** stage or also exists in the **prefill** stage. The specific verification experiment is as follows:

Set `max_response_len`=`1`, let the inference model run only `prefill`, and observe the `logp_diff` situation:

* Still abnormal → single `prefill` also has a problem, and training-inference consistency comparison can be carried out (trigger the comparison metric).
* Becomes normal → most likely a problem specific to the inference `decode` module; focus on the **kv cache** read and write logic in the previous section for troubleshooting.

#### 2.2.1 Comparison Prerequisites

During RL training under verl, the `batch` dimension is flattened to the `token` dimension, so there is a prerequisite for training-inference comparison: the `token` and `feature` dimensions during model `forward` must be consistent to perform statistics comparison and dual-thousand comparison between training and inference.

In conventional RL training, the `forward<span> token` sequence dimension of inference and training usually has the following differences:

* Take a single `prompt` as an example
  * Inference is divided into `prefill`+`N*decode`
    In the `prefill` stage the `token` dimension of `forward` is `prompt_len`, and in the `decode` stage the `token` dimension of `forward` is `1`; after completion, `prompt`+`response` is obtained
  * Training performs `forward` as a whole
    * If there is no `pad`, the `token` dimension is `prompt_len`+`response_len`
    * If there is `pad`, the `token` dimension is `max_prompt_len`
      ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/forward_token_dim_padding.png)
* Take multiple `prompt`s as an example
  * Inference concatenates multiple `prompt`s in a single received request; if the budget length is not exceeded, they are concatenated into the `token` dimension of `prompt1`+`prompt2` for `prefill` computation; in the `decode` stage the `token` dimension of `forward` is `2`, and after completion `prompt1`+`response1` and `prompt2`+`response2` are obtained
  * Training may also have a `mini<span> batch` loop + gradient accumulation loop that splits the `prompt` data. Once split, it is no longer possible to have the same `shape` as inference, so the premise requires ensuring that there is no `batch` loop splitting in training.

In conventional RL training, the `forward<span> feature` feature dimension of inference and training also differs when the sharding configuration is inconsistent (such as enabling different `TP` strategies).

In summary, to ensure training-inference comparability, the prerequisites are:

1. Ensure that the training `batch` is not split.

   * It is necessary to ensure that the number of `mini<span> batch` used for gradient update in each training round `mini_batch_num`= `1`. The calculation formula is:

   ```makeup
   mini_batch_num = train_batch_size/train_ppo_mini_batch_size

   ```

   * It is necessary to ensure that the number of gradient accumulation steps `gac` = 1. The calculation formula is:

   ```makeup
   gac = train_ppo_mini_batch_size*n_resp_per_prompt/train_ppo_micro_batch_size_per_gpu/DP

   ```

   Here, the `DP` calculation formula for different training backends is:

   * FSDP: data parallelism, `DP=world_size`
   * Megatron: model parallelism, `DP=world_size/TP/PP/CP`

   In the verl framework script, the specific hyperparameters corresponding to all the above values are:

   ```makeup
   data.train_batch_size=${train_batch_size}
   actor_rollout_ref.actor.ppo_mini_batch_size=${train_ppo_mini_batch_size}
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu}
   actor_rollout_ref.actor.ppo_epochs=1  # Default is 1
   actor_rollout_ref.rollout.n=${n_resp_per_prompt}

   ```

2. Disable `pad` and dynamic group `batch` in training. The corresponding specific hyperparameters in the verl framework script are:

   ```makeup
   actor_rollout_ref.model.use_remove_padding=True
   actor_rollout_ref.actor.use_dynamic_bsz=False

   ```

3. The sharding configuration of training and inference is consistent, ensuring that sharding strategies such as `TP, PP, CP` are exactly the same.
4. Inference performs only prefill. The corresponding specific hyperparameter in the verl framework script is:

   ```makeup
   data.max_response_length=1

   ```

5. Training and inference have no response difference
   There are two schemes to achieve alignment:

* **Scheme 1: Turn training into a single `prompt`.**
* **Scheme 2: Make the inference prefill carry the `response`**, specifically, after inference completes `prefill` and `decode` and obtains the complete `prompt`+`response`, set `max_response`=`1` and redo `prefill`(`prompt`+`response`).

Since the latter involves repeated inference and affects performance, this guide prefers scheme 1.

**Specific implementation of scheme 1**
In the verl framework, modify the input data for training and adapt the loss computation to ensure no errors. For different backends, the corresponding modified files are as follows:

* `fsdp`: `verl/workers/actor/dp_actor.py`
* `megatron`: `verl/workers/actor/megatron_actor.py`

Taking FSDP as an example, the specific code modifications are as follows:

```makeup
...
def _forward_micro_batch(
    self, micro_batch, temperature, calculate_entropy=False
) -> tuple[torch.Tensor, torch.Tensor]:
    """..."""
        response_length = micro_batch["responses"].size(-1)
+        if "responses" in micro_batch and micro_batch["responses"] is not None:
+            response_length = micro_batch["responses"].size(-1)
+        else:
+            response_length = 0
multi_modal_inputs = {}
...

@GPUMemoryLogger(role="dp actor", logger=logger)
def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
    """..."""
    # set to eval
    self.actor_module.eval()

+        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
+        if compute_prompts_only:
+            if "responses" in data.batch:
+                responses_len = data.batch["responses"].size(1)
+                data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
+                data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
+                if data.batch["position_ids"].dim() == 3:
+                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
+                else:
+                    data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
+                # remove responses from batch
+                data.batch["responses"] = None
+                if "rollout_log_probs" in data.batch:
+                    data.batch["rollout_log_probs"] = None
+                if "response_mask" in data.batch:
+                    data.batch["response_mask"] = None
+

micro_batch_size = data.meta_info["micro_batch_size"]
...

@GPUMemoryLogger(role="dp actor", logger=logger)
def update_policy(self, data: DataProto):
    # make sure we are in training mode
    self.actor_module.train()

    temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

+        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
+        if compute_prompts_only:
+            if "responses" in data.batch:
+                responses_len = data.batch["responses"].size(1)
+                data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
+                data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
+                if data.batch["position_ids"].dim() == 3:
+                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
+                else:
+                    data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
+                # remove responses from batch
+                data.batch["responses"] = None
+                if "rollout_log_probs" in data.batch:
+                    data.batch["rollout_log_probs"] = None
+                if "response_mask" in data.batch:
+                    data.batch["response_mask"] = None
+
         select_keys = [
             "responses",
             "response_mask",
             "input_ids",
             "attention_mask",
             "position_ids",
             "old_log_probs",
             "advantages",
         ]
         ...
                     ...
                     # Extract pre-computed rollout correction weights if present
                     # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                     rollout_is_weights = model_inputs.get("rollout_is_weights", None)

+                    if response_mask is None:
+                        prompt_mask = torch.ones_like(log_prob, dtype=torch.bool)
+                        response_mask = prompt_mask
+
                     # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                     # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                     policy_loss_fn = get_policy_loss_fn(loss_mode)

                     # Compute policy loss (any function is expected to return 2 values)
                     pg_loss, pg_metrics = policy_loss_fn(
                         old_log_prob=old_log_prob,
                         log_prob=log_prob,
                         advantages=advantages,
                         response_mask=response_mask,
                         loss_agg_mode=loss_agg_mode,
                         config=self.config,
                         rollout_is_weights=rollout_is_weights,
                     )
                     micro_batch_metrics.update(pg_metrics)
                     ...

```

As shown in the above code, the single-prompt training mode is switched on and off through the environment variable `PROMPTS_ONLY`; setting it to 1 means enabling single-prompt mode.

#### 2.2.2 Routing Replay

In a typical RL process, an efficient inference engine (such as SGLang) is used for data sampling, and then the data is fed into the training framework for model optimization. For standard dense models, this framework difference may only bring slight numerical errors; but in MoE models, this problem is sharply amplified. The core lies in the MoE of Routing Mechanism: a slight environmental or implementation difference may cause the model to select a completely different expert combination for the same input token, thus going to a completely different computation path. This inconsistency in routing decisions may lead to unstable RL training of MoE models. It makes the "experience" obtained from the inference stage completely different for the training stage, so the optimization signal is distorted, ultimately leading to disastrous consequences.
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/moe_routing_inconsistency.png)

Therefore, in the training-inference consistency alignment process, routing replay also needs to be enabled for MoE models.

Routing Replay has two variants, R2/R3.

(1) Vanilla Routing Replay (R2): corresponds to the verl switch `actor_rollout_ref.actor.router_replay.mode="R2"`

Mechanism: Replay the expert path computed by the training engine in the sampling stage. The goal of R2 is to mitigate the impact of expert routing on policy staleness; its method is to reproduce, in the gradient update stage, the routing experts selected by the rollout policy in the training engine.

Effect: Mainly reduces the impact of policy staleness on routing.

(2) Rollout Routing Replay (R3): corresponds to the verl switch `actor_rollout_ref.actor.megatron.router_replay.mode="R3"`

Mechanism: Capture the routing distribution of the inference engine during sequence generation and directly replay it into the training engine.

Effect: Reduces both training-inference bias and policy staleness.
In the actual training-inference consistency alignment process, R3 can be enabled for precision alignment.

#### 2.2.3 msProbe Data Collection

For the data collection capability, the msProbe tool provided by the MindStudio team can be used. The msProbe tool collects the precision data of the model during operation by adding the `PrecisionDebugger` interface to the model script and starting training.
**Features**

* **Multi-granularity data collection**: Supports data collection at different granularities: L0 (module level), L1 (API level), and mix (L0+L1).
* **Multiple dump modes**: Provides various collection modes such as statistics, tensor, acc_check, structure, and overflow_check.
* **Flexible configuration options**: Precisely control the collection scope through the `config.json` file.

**Usage Instructions**

1. To use the precision collection tool, you need to configure the `config.json` file first. For collecting statistics, the two most commonly used configurations are as follows:

   * Collect the statistics of specified steps

     ```makeup
     {
         "task": "statistics",
         "dump_path": "/home/data_dump",
         "rank": [],
         "step": [0,1],
         "level": "mix",
         "statistics": {
             "scope": [],
             "list": [],
             "data_mode": ["all"],
             "summary_mode": "statistics"
         }
     }

     ```

   * Collect the tensor values of specified steps

     ```makeup
     {
         "task": "tensor",
         "dump_path": "/home/data_dump",
         "rank": [],
         "step": [0,1],
         "level": "mix",
         "tensor": {
             "scope": [],
             "list":[],
            "data_mode": ["all"]
         }
     }

     ```

   In addition, several common modifications based on these two configurations are: * Collection level: modify level to collect "mix" (API + module level), "L0" (module level), "L1" (API).

   * Statistics + md5: modify "summary_mode" in the statistics to "md5".
   * Specify the collection steps (or rank numbers): modify "step" (or "rank"); [] means collect all, a value inside means collect that step, and multiple steps are separated by English commas.
   * Filter target APIs, specifically in the following two ways:
     * Modify the "list" attribute; when added, all APIs or modules whose names contain the string will be collected, multiple collection objects are separated by English commas, and if the string is a module type, the internal APIs are expanded for synchronous collection.
     * Modify the "scope" attribute and add the start and end APIs or modules to collect the APIs or modules between them.

#### 2.2.4 Training-Side Data Collection

Taking Megatron backend collection as an example.

In the `verl/workers/actor/megatron_actor.py` file, in the `compute_log_prob` method of the `MegatronPPOActor` class, before and after calling `forward_backward_batch`, add tool initialization and the dump switch.

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/megatron_train_dump_code.png)

#### 2.2.5 Inference-Side Data Collection

Taking vLLM inference backend collection as an example.

1. In the `vllm_ascend/worker/model_runner_v1.py` file, in the `__init__` method of `NPUModelRunner`, add tool initialization code and add a determinism switch.

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/vllm_init_dump_code.png)

2. In the `execute_model` method, perform `debugger start` (start dump).

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/vllm_start_dump_code.png)

3. In the `execute_model` method, after `_generate_process_reqs_hidden_states` finishes executing, perform `stop` (end dump).

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/vllm_stop_dump_code.png)

#### 2.2.6 Automated Comparison Capability

For training-inference consistency comparison, in addition to shape, there are also many inconsistencies in module hierarchy names. Taking Qwen2.5-0.5B as an example, if inference uses the vLLM backend and training uses the FSDP backend, the corresponding module names of the two sides are as follows:
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/module_name_mapping.png)

It can be seen that the following differences exist:

1. There are many differences in module names, with different prefixes and different class names.
2. `qkv_proj` has a splitting difference between the two sides: inference merges into `qkv_proj`, while training splits it into `q_proj`, `k_proj`, and `v_proj`.
3. `rotary` position difference: inference places `rotary` inside the `self_attn` of each layer, while training uniformly performs `rotary` before entering `decode`.
4. `silu` activation function implementation difference: inference uses `AscendSiluAndMul` to fuse the computation of `Silu` and `Mul`, while training uses small operators and is not collected at the `L0` level.

##### Introduction to the Comparison Tool

For this kind of training-inference comparison, the original comparison tool produces a large number of unmatched cases. To this end, the comparison tool has been specially adapted for automatic matching in the commonly used training-inference consistency comparison of the Qwen series models, mainly by automatically adding the `mapping` list, `qkv` merging, etc. Adding the hyperparameter `--consistent_check` during comparison can greatly improve the comparison success rate.
**Single-Rank Scenario**

```makeup
msprobe compare -tp /train_dump/dump.json -gp /infer_dump/dump.json --consistent_check --backend fsdp -o ./output
```

**Multi-Rank Scenario**

```makeup
msprobe compare -tp /train_dump/step0 -gp /infer_dump/step0 --consistent_check --backend fsdp -o ./output
```

##### Introduction to the Visual Comparison Tool

In the visual comparison tool, a point-to-point matching function is provided. You can select two gray nodes to be matched with the mouse through the browser interface. Currently, only the statistics data mode is supported.

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/visual_point_matching.png)

## 3. Study Cases

### 3.1 Kimi2.5 Training-Inference Alignment

#### 3.1.1 Phenomenon

In the Kimi2.5 RL training scenario, it was found that the reward curve oscillated without an upward trend. The `logp_diff` first-step difference is 3.5, and this metric should be kept within 0.01. Based on this phenomenon, it is suspected that training-inference inconsistency causes the training precision problem.

#### 3.1.2 Localization Idea

Overall idea:

1. Reproduce at a reduced scale: the overall 512-rank scenario has logp_diff 3.5; when reduced to a two-node 32-rank test, `logp_diff 0.7` is still very large.
2. Strip away the RL training process, execute training and inference independently, and use the same input to execute them separately and collect dump data.
3. First try to reduce layers, for example troubleshoot the dense layer first, reducing both training and inference to 1 layer; after ensuring the dense layer is aligned, add layers and compare the MoE layer to complete the training-inference alignment.

#### 3.1.3 Localization Process

Preparatory operation: training and inference use the same input to perform data dump; collect module-level data first to facilitate matching and alignment and quickly lock onto the problem module.

1. The first difference module is the MLA module.

   As shown in the figure below, the left side is the inference implementation, using the NPU custom operator `npu_ring_mla`; the right side is the training implementation, using small operators encapsulated as an MLA module. There is an obvious difference between the two outputs (four statistics dimensions).

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mla_output_diff.png)

   Looking further at the input, it can be found that although the two modules have the same function, their input definitions are different. Taking the `q` matrix input of attention as an example, the inference module takes `q_nope` and `q_rope` as inputs separately, while the training module directly concatenates `q_nope` and `q_rope` into `q` as the input of the MLA module. Observing the statistics of the two, it is found that the inputs cannot be aligned, so further troubleshooting forward is needed.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mla_input_diff.png)

2. `q_rope` difference

   Troubleshooting further forward, find the training-side implementation that concatenates `q_nope` and `q_rope` into `q`, and further compare it with the inference `q_nope` and `q_rope`, and find that the `q_rope` operation is not aligned.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_rope_diff.png)

   Troubleshooting further forward to observe the computation source of `q_rope`, it can first be confirmed that the implementations of RoPE in training and inference are inconsistent.

   Inference implementation: uses the `npu_interleave_rope` fusion operator.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_fused_rope_impl.png)

   Training implementation: uses a series of small operators composed together.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/train_composed_rope_impl.png)

   Both use the cos and sin encoding matrices. Through comparison of the dump data, it is found that the cos and sin encoding matrices themselves cannot be aligned.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/cos_sin_matrix_diff.png)

   From this, it can basically be inferred that the `freqs` input participating in the computation of cos and sin differ.

3. `freqs` source analysis and troubleshooting

   The computations of cos and sin both come from `freqs`. Tracing the source of `freqs` upward, it can be found that `freqs` is actually the internal `self.rotary_pos_emb`, which is obtained through the instantiated `YarnRotaryEmbedding`, so it is suspected that a parameter problem occurs during the instantiation of `YarnRotaryEmbedding`.

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/freqs_source_code.png)

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/yarn_rotary_embedding_init.png)

    Comparing the input parameters of `YarnRotaryEmbedding` with inference, it is found that the `original_max_position_embeddings=1024` participating in the computation is actually configured as 4096 in the outer layer, that is, the expected value of `original_max_position_embeddings` is 4096, but the actual value passed in is 1024.

    Further confirmation shows that this is because MindSpeed's adaptation of `YarnRotaryEmbedding` differs from Megatron.

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mindspeed_yarn_impl_1.png)

    ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mindspeed_yarn_impl_2.png)

#### 3.1.4 Problem Fix

Fix the mindspeed implementation so that `original_max_position_embeddings=self.config.original_max_position_embeddings,` is used again for training; `logp_diff` drops to about 0.004, and the reward rises steadily for 100 training steps.

### 3.2 DS3.2 Training-Inference Alignment

#### 3.2.1 Background

In the DeepseekV3.2 RL training scenario, the long-running curve is as shown below. It can be seen that it has collapsed at around step 50. Observing the `logp diff` curve, the `logp diff` is too large, with the first step around 0.05 and gradually rising, so the direction is shifted to the training-inference inconsistency problem.

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/ds32_reward_collapse.png)

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/ds32_logp_diff_curve.png)

#### 3.2.2 Localization Idea

1. Reduce the model from 61 layers to 5 layers, keep the same sharding strategy for training and inference, keep the input consistent, and collect dump data.
2. First check the input and output of large modules, find the position that is not aligned, and then gradually narrow the troubleshooting scope according to the call stack until finding the first position where the input is consistent but the output is inconsistent.
3. Check the specific implementation code of training and inference at the position where the input is consistent but the output is inconsistent, modify the code to align the processing flow, and re-dump the data for comparison until the training-inference alignment is completed.

#### 3.2.3 Localization Process

Fix the input prompt, `batchsize=1`, `response=1`, ensure that training and inference adopt the same sharding strategy, collect mix-level data of the reduced layers, and perform detailed comparison.

1. First perform coarse-grained troubleshooting at the large module level, and find that the input is consistent but the output is inconsistent (inference on the left, training on the right) after passing through the entire large MLA, as shown in the figure below.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mla_module_inout_diff.png)

   Then look upward for the specific source of the MLA module output and find that the MLA output comes from an all_reduce operation. Observing the input and output of this all_reduce, the results are as follows. It is observed that its input already has differences, and the inputs of other ranks also have differences, but the differences are small; however, after all_reduce, the maximum difference of the output is obvious.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/all_reduce_diff.png)

   Next, look upward for the source of the all_reduce input. As shown in the figure below, the left side (inference) is a linear operation and the right side (training) is a matmul operation; and it is observed that the input values and weights are numerically consistent in statistics, but the outputs are inconsistent.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_linear_op.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/train_matmul_op.png)

   The linear and matmul operators should not have precision problems. We preliminarily suspect that the inputs of these two operators are already inconsistent (**although they appear consistent in statistics**), so we reproduce with a single operator and find that with consistent inputs, the outputs of these two operators are consistent. We compare the input and weight and find that the weight is completely consistent while the input cannot be matched.

   Then continue to look forward for the source of this input, and find the corresponding API outputs in inference and training. The specific results are as follows, showing that the input is consistent and the output is also consistent.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/upstream_api_output_1.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/upstream_api_output_2.png)

   Then look upward for the input source of these two operators. The following operation is found at present. Although the input statistics are the same, saving the inputs of these two operators and performing element-by-element comparison shows that the inputs of these two operators differ.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/elementwise_diff_1.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/elementwise_diff_2.png)

   Following the call stack all the way forward, the source on the inference side is the custom fusion operator of vllm_ascend on the left, and the source on the training side is the fusion operator in CANN 8.5 on the right. The comparison results are as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fused_op_compare.png)

   Because of a tool issue, the input and output of these two fused operators were not collected, so we can only look forward for the source of the input `q`, `k`, `v`. First troubleshoot the consistency of `q;` the source of `q` is as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_source_code.png)

   Through element-by-element comparison, it is found that `q` is consistent. The results are as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_elementwise_match.png)

   Since the semantics of the key and value of inference and training are inconsistent and the dimensions are also inconsistent, they cannot be directly compared. We directly compare the `topk_indices` output by the indexer module of inference and training. At present, it can be confirmed that the `topk_indices` output by training and inference are already misaligned in the statistical mean.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/topk_indices_diff.png)

   The process of `topk_indices` being called in training and inference is as follows. Since msProbe does not collect the input and output of lightning_indexer, modify the following code to re-collect dump data, with the dump level being tensor+mix.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_dump_code_mod.png)

2. `npu_lightning_indexer` operator input/output troubleshooting

   According to the collection results, we compare the input query, key, weights and the output `topk_indices` element by element, and the comparison results differ greatly.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_input_output_diff.png)

   Then first troubleshoot the source of weights. The calls of training and inference are as follows, both being a `weights_proj` operation, which corresponds to a linear operation, as shown below:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/train_weights_proj_call.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_weights_proj_call.png)

   We perform element-by-element comparison on the input *x*, weight, and output output, and the input *x*, weight, and output output can be fully aligned. The comparison results are as follows.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_proj_align_1.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_proj_align_2.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_proj_align_3.png)

   For the output after linear, inference and training each perform additional operations: inference performs a maybe_all_gather, and training performs two scaling operations. The inputs and outputs of lines 962 and 695 in the figure below can be fully aligned.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_extra_ops_code.png)

   After lines 963 and 696 in the figure above, the difference in the weight value is very obvious. The comparison results are as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_scaled_diff.png)

   Next, locate the `q` input of `lightning_indexer`. The call positions of inference and training are as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_source_code.png)

   The input `qr`, weight, and output `q` can be fully aligned element by element. The comparison results are as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_align_1.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_align_2.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_align_3.png)

   Next, inference and training perform split and RoPE operations on `q` respectively (inference on the left, training on the right).

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/split_rope_ops_compare.png)

   Inference and training go through different rope operations. After the rope operation, the statistics of the output `q_pe` and `x_pe` can be matched, but the element-by-element comparison cannot be matched.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/rope_output_stats_1.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/rope_output_stats_2.png)

   Going inside the `apply_rotary_pos_emb_bshd_in_complex` function, it can be seen that an FP32 to BF16 operation is performed on the output.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fp32_to_bf16_code.png)

   The type conversion operation causes a certain loss of precision.

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/precision_loss_diff.png)

   This conversion operation causes the `x_pe` output by training and inference to be misaligned with the `q_pe` output by inference. Then training performs a `rotate_activation` operation on `q`, while the inference side does not perform this operation. Inside `rotate_activation`, a Hadamard transform is performed. The call comparison is as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/rotate_activation_call.png)

   After this operation, the `q` of training and inference is completely misaligned, as shown in the figure below:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_after_rotate_diff.png)

   `k` is similar. For `q` and `k`, training and inference mainly have two differences: when performing the RoPE operation on `q_pe` and `k_pe`, training first uses FP32 for computation and converts to BF16 at the end, while inference uses BF16 throughout, so training has precision loss.

   For the input weights of the `npu_lightning_indexer` operator, the operations of training and inference are also inconsistent. The operation comparison is as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_ops_compare.png)

   Inference and training can be fully aligned after the `weights_proj` operation. After the `maybe_all_gather_and_maybe_unpad` operation on the inference side, the weights do not change, while training undergoes the two scaling operations on the right, causing the weights to be misaligned. The weights statistics comparison is as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_stats_compare.png)

   Based on the above troubleshooting process, the problem is currently located as follows: before inference and training compute `topk_indices` in the indexer module (calling the `npu_lightning_indexer` fusion operator), `npu_lightning_indexer` has 3 inputs `q`, `k`, and `weights`, and inference and training process `q`, `k`, and `weights` differently, specifically as follows:

   1. For `q` and `k`, the training side uses FP32 precision when performing RoPE, and performs a down-precision operation FP32->BF16 after rope; the training side then performs an additional rotate_activation operation, which internally performs a Hadamard transform.
   2. For `weights`, the training side performs one more scaling operation than the inference side.

   #### 3.2.4 Problem Fix

   Make inference strictly align with the training-side implementation: add the `rotate_activation` operation for `q` and `k`; for `weights`, align the scaling rule with the training side. The specific modifications on the inference side are as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_fix_code_1.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_fix_code_2.png)

   After alignment, run the full-layer experiment. The curve for 22 steps is as follows:

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fixed_reward_curve.png)

   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fixed_logp_diff_curve.png)

   From the curves, the reward rises normally, the `logp diff` has no upward trend, and the `logp diff` remains in the thousandths. There are currently no resources for a long run, so the above indexer alignment modification can only be concluded to solve the training-inference inconsistency problem by observing the first 22 steps.
