# fsdp训练后端verl训推一致性比对数据采集

## 简介

[verl训推一致性比对场景](../accuracy_compare/pytorch_accuracy_compare_instruct.md#verl训推一致性比对场景)在比对前，需要保证训练和推理时的输入 shape 一致，才能确保比对时训练和推理dump的精度数据匹配。

## verl训练推理输入对齐分析

一般情况下，训练和推理时的输入 shape 不一致，从训练推理运行原理分析。

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
   
      计算公式为：gac = (train_ppo_mini_batch_size * n_resp_per_prompt) / (train_ppo_micro_batch_size_per_gpu / DP)
      - train_ppo_mini_batch_size: 每个 mini batch 的样本数量。
      - n_resp_per_prompt: 每个提示（prompt）下的响应数。
      - train_ppo_micro_batch_size_per_gpu: 每个GPU上处理的 micro batch 大小。
      - DP: 数据并行度。

   上述参数在脚本中具体修改为：

   ```shell
   data.train_batch_size=${train_batch_size}
   actor_rollout_ref.actor.ppo_mini_batch_size=${train_ppo_mini_batch_size}
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu}
   actor_rollout_ref.rollout.n=${n_resp_per_prompt}
   ```

2. 保证训练中无pad。

    ```shell
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=False
   ```

3. 在训练脚本中修改环境变量。

   ```shell
   export DUMP_ON=1
   export PROMPTS_ONLY=1
   export TORCHDYNAMO_DISABLE=1
   ```

4. 保证训练和推理采集的数据在每张卡上是一一对应的。
    balance_batch: 自动平衡、均分batch数据

   ```shell
   trainer.balance_batch=False
   ```

## verl代码修改

去掉训练输入中的 response，需要修改 verl/workers/actor/dp_actor.py，以 release/v0.6.1 为例，修改处高亮显示如下：

```diff
 ...
     ...
     def _forward_micro_batch(
         self, micro_batch, temperature, calculate_entropy=False
     ) -> tuple[torch.Tensor, torch.Tensor]:
         """..."""

+        # _forward_micro_batch方法中修改
-        response_length = micro_batch["responses"].size(-1)
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

+        # compute_log_prob方法中修改
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

+        # update_policy方法中修改
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

+                    # update_policy方法中修改                     
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

## 数据采集

在 verl/workers/fsdp_workers.py 中添加msProbe工具的PrecisionDebugger接口进行dump操作。PrecisionDebugger接口更多介绍请参见《[PyTorch场景精度数据采集](../dump/pytorch_data_dump_instruct.md)》。

修改示例代码高亮显示如下：

```diff
 ...
 class ActorRolloutRefWorker(Worker, DistProfilerExtension):
     """
     This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
     or a hybrid engine based on the config.rollout
     """
 
     def __init__(self, config: DictConfig, role: str, **kwargs):
         ...
         # normalize rollout config
         if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
             self.config.rollout.log_prob_micro_batch_size //= (
                 self.device_mesh.size() // self.ulysses_sequence_parallel_size
             )
             self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
         # normalize ref config
         if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
             self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
             self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

+        # __init__方法中修改
+        # 实例化PrecisionDebugger
+        dump_flag = int(os.environ.get("DUMP_ON", 0))  # 设置环境变量DUMP_ON用于快速开关dump功能
+        if dump_flag:
+            from msprobe.pytorch import PrecisionDebugger, seed_all
+            seed_all(mode=True)
+            self.debugger = PrecisionDebugger(task='tensor', level='L0', dump_path='example_dump_path', step=[0])
+            self.dump_path_prefix = self.debugger.config.dump_path
+        else:
+            self.debugger = None
 
     ...
     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
     @DistProfiler.annotate(color="red", role="actor_update")
     def update_actor(self, data: DataProto):
         ...
         with self.ulysses_sharding_manager:
             data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy
+            # update_actor方法中修改
+            if self.debugger:
+                self.debugger.service.config.dump_path = os.path.join(self.dump_path_prefix, 'update_actor')  # 训练结果保存在update_actor文件夹
+                self.debugger.start(model=self.actor.actor_module)
             # perform training
             with Timer(name="update_policy", logger=None) as timer:
                 metrics = self.actor.update_policy(data=data)
+            if self.debugger:
+                self.debugger.stop()
+                self.debugger.step()
             delta_time = timer.last
             global_num_tokens = data.meta_info["global_token_num"]
             ...
     
     @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
     @DistProfiler.annotate(color="red", role="rollout_generate")
     def generate_sequences(self, prompts: DataProto):
         ...
         with simple_timer("generate_sequences", timing_generate):
+            # generate_sequences方法中修改
+            if self.debugger:
+                self.debugger.service.config.dump_path = os.path.join(self.dump_path_prefix, 'generate_sequences')  # 推理结果保存在generate_sequences文件夹
+                infer_model = self.rollout.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
+                self.debugger.start(model=infer_model, token_range=[0, 0])
             output = self.rollout.generate_sequences(prompts=prompts)
+            if self.debugger:
+                self.debugger.stop()
+                self.debugger.service._reset_status()
 
         if self._is_actor:
             loop.run_until_complete(self.trainer_mode())
             log_gpu_memory_usage("After switch to trainer mode", logger=logger)
         ...
```
