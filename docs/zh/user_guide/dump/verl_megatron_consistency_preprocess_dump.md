# megatron训练后端verl训推一致性比对数据采集

## 简介

[verl训推一致性比对场景](../accuracy_compare/pytorch_accuracy_compare_instruct.md#verl训推一致性比对场景)在比对前，需要保证训练和推理时的输入 shape 一致，才能确保比对时训练和推理dump的精度数据匹配。

## verl训练推理输入对齐分析

一般情况下，训练和推理时的输入 shape 不一致，以下从训练推理运行原理分析。

* 推理运行分为 prefill 和 k 个 decode 两个步骤：

  1. 在 prefill 步骤时，推理输入为 prompt。
  2. 在 k 个 decode 步骤时，kv cache 加上上一个 decode 得到的输出 token ，最终输出推理的 response。

* 训练运行时，输入为 prompt 加上推理输出的 response，最终输出 logits。

综合以上信息，推理输入为 prompt；训练输入为 prompt 加上推理输出的 response。

**结论**：需要将训练的输入调整为单 prompt，与推理的输入保持一致。

## 前置操作

要保证训练 forward 和推理 prefill 的 shape 一致，需要去掉训练输入中的 response，首先需要满足如下前提，并修改训练脚本：

1. 保证训练中的batch size维度未被拆分。

   1. 需保证每轮训练中用于梯度更新的mini batch个数mini_batch_num = 1

      计算公式为：mini_batch_num = train_batch_size / train_ppo_mini_batch_size
      - train_batch_size: 训练中总的样本数。
      - train_ppo_mini_batch_size: 每个 mini batch 的样本数量。

   2. 需保证梯度累计步骤数gac (Gradient Accumulation Steps) = 1

      计算公式为：gac = train_ppo_mini_batch_size * n_resp_per_prompt / train_ppo_micro_batch_size_per_gpu / DP
      - train_ppo_mini_batch_size: 每个 mini batch 的样本数量。
      - n_resp_per_prompt: 每个提示（prompt）下的响应数。
      - train_ppo_micro_batch_size_per_gpu: 每个GPU上处理的 micro batch 大小。
      - DP: 数据并行度。DP = world_size / TP / PP / CP

   上述参数在脚本中具体修改为：

   ```shell
   data.train_batch_size=${train_batch_size}
   actor_rollout_ref.actor.ppo_mini_batch_size=${train_ppo_mini_batch_size}
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu}
   actor_rollout_ref.rollout.n=${n_resp_per_prompt}
   ```

2. 保证训练中无pad。
    use_remove_padding: 移除 padding 优化

    ```shell
    actor_rollout_ref.model.use_remove_padding=True
   ```

3. 在训练脚本中修改环境变量。

   ```shell
   export DUMP_ON=1
   export PROMPTS_ONLY=1
   ```

4. 保证训练和推理采集的数据在每张卡上是一一对应的。
    balance_batch: 自动平衡、均分batch数据

   ```shell
   trainer.balance_batch=False
   ```

## verl代码修改

去掉训练输入中的 response，需要修改 verl/workers/actor/megatron_actor.py、verl/utils/debug/metrics.py、verl/trainer/ppo/rollout_corr_helper.py，以 release/v0.6.1 为例，修改处高亮显示如下：

verl/workers/actor/megatron_actor.py

```diff
 ...
     ...
     def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
         """..."""
         ...
+        # 检查是否仅对提示计算 log_probs（不包括响应）
+        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
+        if compute_prompts_only:
+            # 从 input_ids、attention_mask 和 position_ids 中移除响应部分
+            if "responses" in data.batch:
+                response_length = data.batch["responses"].size(1)
+                data.batch["input_ids"] = data.batch["input_ids"][:, :-response_length]
+                data.batch["attention_mask"] = data.batch["attention_mask"][:, :-response_length]
+                if data.batch["position_ids"].dim() == 3:  # qwen2vl mrope
+                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-response_length]
+                else:
+                    data.batch["position_ids"] = data.batch["position_ids"][:, :-response_length]
+                # 从批处理中移除响应
+                data.batch.pop("responses", None)
+                if "rollout_log_probs" in data.batch:
+                    data.batch.pop("rollout_log_probs", None)
+                if "response_mask" in data.batch:
+                    data.batch.pop("response_mask", None)

         def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
-            response = data["responses"]
-            response_length = response.size(1)
-            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
+            if "responses" in data and data["responses"] is not None:
+                response = data["responses"]
+                response_length = response.size(1)
+                log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
+            else:
+                # 仅针对提示，返回所有提示 token 的 log_probs（不包括用于下一个 token 预测的最后一个 token）
+                log_probs = output["log_probs"][:, :-1].contiguous()
             return {"log_probs": log_probs}
             ...
         if recompute_old_log_prob:
-            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
+            # 这里默认使用 recompute_old_log_prob。
+            select_keys = ["input_ids", "attention_mask", "position_ids"]
+            if "responses" in data.batch:
+                select_keys.append("responses")
             batch = data.select(batch_keys=select_keys).batch
             input_ids = batch["input_ids"]
             batch_size = input_ids.size(0)
-            response = batch["responses"]
-            response_length = response.size(1)
+            if "responses" in batch and batch["responses"] is not None:
+                response = batch["responses"]
+                response_length = response.size(1)
+            else:
+                response = None
+                response_length = 0
             with torch.no_grad():
             ...
-                    log_probs = torch.empty(
-                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
-                    )
+                    # 仅用于提示，log_probs 的形状为 [batch_size, prompt_length - 1]（不包括用于下一个 token 预测的最后一个 token）
+                    if response_length > 0:
+                        log_probs_shape = (batch_size, response_length)
+                    else:
+                        prompt_length = input_ids.size(1)
+                        log_probs_shape = (batch_size, prompt_length - 1) if prompt_length > 1 else (batch_size, 0)
+                    log_probs = torch.empty(
+                        size=log_probs_shape, dtype=torch.float32, device=input_ids.device
+                    )
                     ...
-                        entropys = torch.empty(
-                            size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
-                        )
+                        if response_length > 0:
+                            entropy_shape = (batch_size, response_length)
+                        else:
+                            prompt_length = input_ids.size(1)
+                            entropy_shape = (batch_size, 0)
+                        entropys = torch.empty(
+                            size=entropy_shape, dtype=torch.float32, device=input_ids.device
+                        )
                         ...

     def forward_backward_batch(...):
         """..."""
         ...
         def loss_func(output, data, meta_info):
             ...
-            responses = data["responses"]
-            response_length = responses.size(1)
-            response_mask = data["response_mask"].to(bool)
-            loss_agg_mode = self.config.loss_agg_mode
-            # compute policy loss
-            log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
+            # 检查是否有响应或只有提示
+            if "responses" in data and data["responses"] is not None:
+                responses = data["responses"]
+                response_length = responses.size(1)
+                response_mask = data["response_mask"].to(bool)
+                # 计算策略损失
+                log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
+            else:
+                # 仅用于提示：使用除最后一个标记外的所有 log_probs
+                response_length = 0
+                response_mask = None
+                log_prob = log_probs[:, :-1].contiguous() if log_probs.size(1) > 0 else log_probs
+            loss_agg_mode = self.config.loss_agg_mode
             ...
                 rollout_is_weights = data.get("rollout_is_weights", None)
+                # 仅用于提示，为所有提示标记（不包括最后一个）创建掩码
+                if response_mask is None:
+                    # 为除最后一个之外的所有提示标记创建掩码
+                    prompt_mask = torch.ones_like(log_prob, dtype=torch.bool)
+                    response_mask = prompt_mask
                     ...
                     from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
+                    # 仅用于提示，使用与上面相同的掩码
+                    if response_mask is None:
+                        prompt_mask = torch.ones_like(log_prob, dtype=torch.bool)
+                        response_mask = prompt_mask
                         ...
             if calculate_entropy:
-                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
-                if not forward_only:
-                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
+                if response_length > 0:
+                    entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
+                else:
+                    # 仅用于提示：使用除最后一个标记外的所有熵
+                    entropy = output["entropy"][:, :-1].contiguous() if output["entropy"].size(1) > 0 else output["entropy"]
+                if not forward_only:
+                    if response_mask is not None:
+                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
+                    else:
+                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=None, loss_agg_mode=loss_agg_mode)
                         ...
                     kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
-                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
+                    if response_mask is not None:
+                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
+                    else:
+                        kl_loss = agg_loss(loss_mat=kld, loss_mask=None, loss_agg_mode=self.config.loss_agg_mode)
                         ...

         def forward_step(batch_iter, model, return_schedule_plan: bool = False):
             ...
-            responses = batch["responses"]
-            response_length = responses.size(1)
-            label = position_ids.clone()
-            label[:, -response_length - 1 : -1] = responses
-            label_mask = attention_mask.clone()
-            label_mask[:, : -response_length - 1] = False
-            label_mask[:, -1] = False
+            # 检查是否有响应或只有提示
+            if "responses" in batch and batch["responses"] is not None:
+                responses = batch["responses"]
+                response_length = responses.size(1)
+                label = position_ids.clone()
+                label[:, -response_length - 1 : -1] = responses
+                label_mask = attention_mask.clone()
+                label_mask[:, : -response_length - 1] = False
+                label_mask[:, -1] = False
+            else:
+                # 仅针对提示：计算所有提示令牌的 log_probs（下一个令牌预测）
+                # 标签是将 input_ids 向后移动 1 个位置以进行下一个令牌预测
+                response_length = 0
+                label = input_ids.clone()
+                label_mask = attention_mask.clone()
+                # 仅针对提示，计算除最后一个之外的所有 token 的对数概率
+                # （因为最后一个 token 没有下一个 token 可以预测）
+                if label_mask.size(1) > 0:
+                    label_mask[:, -1] = False

```

verl/utils/debug/metrics.py

```diff
 ...
 def calculate_debug_metrics(data: DataProto) -> dict:
     """..."""
+        if "rollout_log_probs" not in data.batch:
+            logger.warning("rollout_log_probs not found in batch, skipping debug metrics calculation")
+            return {
+                "training/rollout_probs_diff_valid": 0,
+                "training/rollout_probs_diff_max": 0.0,
+                "training/rollout_probs_diff_mean": 0.0,
+                "training/rollout_probs_diff_std": 0.0,
+                "training/rollout_actor_probs_pearson_corr": 0.0,
+            }
+
+        if "old_log_probs" not in data.batch:
+            logger.warning("old_log_probs not found in batch, skipping debug metrics calculation")
+            return {
+                "training/rollout_probs_diff_valid": 0,
+                "training/rollout_probs_diff_max": 0.0,
+                "training/rollout_probs_diff_mean": 0.0,
+                "training/rollout_probs_diff_std": 0.0,
+                "training/rollout_actor_probs_pearson_corr": 0.0,
+            }
+
+        if "responses" not in data.batch:
+            logger.warning(
+                "responses not found in batch(possibly compute_prompts_only mode), skipping debug metrics calculation")
+            return {
+                "training/rollout_probs_diff_valid": 0,
+                "training/rollout_probs_diff_max": 0.0,
+                "training/rollout_probs_diff_mean": 0.0,
+                "training/rollout_probs_diff_std": 0.0,
+                "training/rollout_actor_probs_pearson_corr": 0.0,
+            }
+
         rollout_old_log_probs = data.batch["rollout_log_probs"]
```

verl/trainer/ppo/rollout_corr_helper.py

```diff
 ...
 def compute_rollout_correction_and_add_to_batch(
     batch: DataProto, rollout_corr_config: RolloutCorrectionConfig
 ) -> tuple[DataProto, dict]:
     """..."""
+    if int(os.getenv("PROMPTS_ONLY", "0")):
+        return batch, {}
     rollout_is = rollout_corr_config.get("rollout_is", None)
```

## 数据采集

在 verl/workers/megatron_workers.py 中添加msProbe工具的PrecisionDebugger接口进行dump操作。PrecisionDebugger接口更多介绍请参见《[PyTorch场景精度数据采集](../dump/pytorch_data_dump_instruct.md)》。

修改示例代码高亮显示如下：

```diff
 ...
 class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
     """..."""

     def __init__(self, config: DictConfig, role: str, **kwargs):
         ...
             self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)
+        # __init__方法中修改
+        # 实例化PrecisionDebugger
+        # 设置环境变量DUMP_ON用于快速开关dump功能
+        dump_flag = int(os.environ.get("DUMP_ON", 0))
+        if dump_flag:
+            from msprobe.pytorch import PrecisionDebugger, seed_all
+            seed_all(mode=True)
+            self.debugger = PrecisionDebugger(task='tensor', level='L0', step=[0], dump_path='0_dump_path/')
+            self.dump_path_prefix = self.debugger.config.dump_path
+        else:
+            self.debugger = None
             ...

     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
     @GPUMemoryLogger(role="generate_sequences", logger=logger)
     @DistProfiler.annotate(color="red")
     def generate_sequences(self, prompts: DataProto):
         ...
         with simple_timer("generate_sequences", timing_generate):
+            # generate_sequences推理处使能工具dump采集推理前向数据
+            if self.debugger:
+                infer_model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
+                # 推理阶段前向dump数据保存在generate_sequences文件夹
+                self.debugger.service.config.dump_path = os.path.join(self.dump_path_prefix, 'generate_sequences')
+                self.debugger.start(model=infer_model, token_range=[0, 0])
+            output = self.rollout.generate_sequences(prompts=prompts)
+            if self.debugger:
+                self.debugger.stop()
+                self.debugger.service._reset_status()
                 ...

     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
     @GPUMemoryLogger(role="compute_log_prob", logger=logger)
     @DistProfiler.annotate(color="blue")
     def compute_log_prob(self, data: DataProto):
         ...
         data.meta_info["temperature"] = self.config.rollout.temperature
+        # compute_log_prob训练处使能工具dump采集训练module级别输入输出数据
+        if self.debugger:
+            # 训练阶段dump数据保存在update_actor文件夹
+            self.debugger.service.config.dump_path = os.path.join(self.dump_path_prefix, 'update_actor')
+            self.debugger.start(model=self.actor.actor_module)
+        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
+        if self.debugger:
+            self.debugger.stop()
+            self.debugger.step()
             ...
```
