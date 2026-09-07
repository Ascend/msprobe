
## 摘要：

在强化学习系统的开发与部署中，训练与推理阶段的行为偏差——即“训推不一致”——往往是导致训练loss震荡、收敛缓慢甚至reward崩溃的隐性根源。本文首先介绍训推一致性的背景与定义，指出导致训推不一致的本质原因。随后，聚焦训推对齐核心技术：介绍msprobe精度工具以及在训推一致对齐场景下的工具能力。最后，通过一次真实场景的实战案例复盘，展示如何因训推不一致引发训练曲线剧烈波动，以及reward崩溃。通过对齐手段恢复训练稳定性、提升整体训练精度的的全过程。本文旨在为强化学习实践者提供一套可落地的稳定性分析框架与解决思路，让模型不仅在训练环境中“游刃有余”，更在推理部署中“表里如一”。

## 1. 训推一致性的背景与意义

### 1.1 什么是训推不一致？

**训推不一致**指的是 Rollout（推理）引擎与训练引擎之间存在的​**数值不一致性**​。即使两个引擎使用完全相同的模型权重，针对相同的 Token 序列，它们计算出的对数概率也可能存在细微差异。该差异映射到模型行为，可以总结为行为策略（behavior policy）和参考策略（reference policy）不一致。

行为策略：实际负责生成 rollout 的策略，也就是“你在什么分布下采样到了这些数据”。在现代 LLM-RL 系统里，它对应的是推理引擎里的那套实现（vLLM / SGLang 等），在异步框架下往往还是多个 worker 策略的混合分布。

参考策略：训练目标里拿来做重要性采样、clipping 或 KL 约束的策略，典型地就是 PPO / GRPO 里的“旧策略”（old policy）。

目标策略：训练目标里要优化的策略，也就是“你想让模型变成什么样”。典型地就是 PPO / GRPO 里的“新策略”（new policy）。

在最经典、理想化的设定里，我们通常期望​**行为策略**​**​=​**​**参考策略** 。但在现实系统中，受异步更新、不同推理 / 训练后端、MoE 路由波动甚至硬件数值差异等因素影响，二者往往会出现不同程度的偏离。

### 1.2 为什么训推一致性难以保证？

#### (1)训练侧与推理侧的框架差异

如下图所示: 训练侧使用Megatron vs vLLM推理,整体框架的差异，导致训推结果不一致

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/framework_diff.png)

进一步展开来看，框架内部，对于相同语义模型层，实现思路也不同，导致结果不能完全对齐

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/layer_impl_diff.png)

针对同样操作，如layernorm，训练可能使用小算子拼接实现，推理使用融合算子实现，两者在精度上也可能有细微差异

```makeup
# PyTorch 标准实现（训练）
y = (x - mean) / sqrt(var + eps) * gamma + beta
# 分步计算：mean → var → normalize → scale → shift

# vLLM 融合实现（推理）
y = fused_rms_norm(x, weight)  
# 单内核：向量化加载 + 寄存器级融合
# 中间结果截断策略不同 → 最后几位小数差异
​
```

#### (2)​**精度类型差异**​：训练 FP32/BF16 与推理 FP16/INT8 量化带来的数值截断

#### (3)​​**并行策略**​：训练时的张量并行 vs 推理时的连续批处理，导致浮点累加顺序差异

```makeup
(0.1 + 1e20) - 1e20
>>> 0
0.1 + (1e20 - 1e20)
>>> 0.1
​
```

#### (4)​​**随机性控制**​：Dropout、采样策略在训推阶段的实现偏差

### 1.3 训推不一致的典型后果

理论上上述差异点导致的训推差异十分微小，但是这些差异会随着逐层累积、放大，导致训练（training）和推理（rollout）的结果差异明显，甚至会出现同一个token训推输出概率分别为0和1的现象。**这样的训推差异会导致理论上的on-policy并不成立，从而带来RL训练的不稳定和低上限。**

训推不一致的影响并非均匀分布，其破坏力在两种典型配置下会急剧放大：

​**MoE 架构**​：专家路由的"蝴蝶效应"

专家选择机制对数值误差极度敏感。训推两端只要在选择 Top-K 专家时出现微小分歧——哪怕只是单个专家的排序差异——就会导致后续计算路径的彻底分叉。这种分歧不是线性累积，而是​**阶跃式跳变**​：一旦选错专家，后续所有激活值、注意力模式、输出生成都会走向完全不同的分支。

​**长推理模型**​：误差传播的"雪球效应"

模型输出越长，浮点误差的累积就越呈指数级扩张。短输出场景下，末尾 token 的 logprob 误差可能仅源于最后一层的数值噪声；而在长推理链中，早期 token 的微小偏差会通过自回归机制逐层传导，最终使末端分布产生数量级的偏移。

最终便会导致模型训练不稳定，logp_diff_mean逐步飙升，甚至reward崩溃。

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/logp_diff_escalation.png)

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/reward_collapse.png)

## 2. 训推对齐排查思路及工具

本章节以主流VeRL框架举例，介绍基于该框架下的强化学习训练训推对齐排查思路及MindStudio工具链相关的精度工具。
目前主流RL算法是基于**On-Policy**前提展开的，**On-Policy**理论要求采样数据的行为策略与梯度计算的目标策略基本保持一致，才能确保梯度估计是无偏的，从而使训练过程更平稳。在强化学习中采样我们称之为**rollout**推理，梯度计算则对应**actor**训练，当推理与训练策略保持一致，即称为训推一致。 训推一致性的主要指标为​**logp_diff_mean**​，当**logp_diff_mean**异常时表示强化学习中训练和推理存在一定程度的差异，需进行训推差异的根因查找，其计算公式为：
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/logp_diff_formula.png)
其中M为response_mask。

在**verl**框架中，对应行为如下：

* 开启logp_diff监控的配置超参：
  ```makeup
  actor_rollout_ref.rollout.calculate_log_probs=True  # 默认为False
  ​
  ```
* logp_diff的具体代码实现：
  ```makeup
  mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
  mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
  log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
  metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
  ​
  ```

当该指标满足以下条件时，需启动训推一致排查：

* **第一步logp_diff异常偏大（>0.01）或与标杆差异明显。**

### 2.1 kv cache排查

推理流程中：

* ​**prefill**​：清除上一轮​**kvcache**​，写入当前上下文。
* ​**decode**​：增量读写​**kvcache**​。

在实际工程实现中，通常通过**state block**管理缓存，而非逐句强制清除，若出现缓存清理不及时、复用错误、地址偏移等问题，会导致**decode**阶段读取异常，进而引发推理长稳漂移，最终导致后期**logp_diff**变大。
此种情况下，由于**kvcache**为推理时特有的逻辑，与训练无关，因此在强化学习训练中必会触发**logp_diff**不一致。

对于kv cache的排查可通过如下两种方案：

1. 对比**NPU**与标杆的**kvcache**代码读写部分是否存在逻辑差异。
2. 强制逐句清除**kvcache**缓存。

### 2.2 训推一致比对

目前训推一致比对只支持 **prefill**阶段，所以需要先定界一致性差异只在 **decode**阶段发生还是 **prefill**阶段也存在，具体验证实验如下：

设`max_response_len`=`1`，让推理模型只跑`prefill`，同时观察`logp_diff`情况：

* 仍异常 → 单`prefill`也有问题，可进行训推一致性比对（触发比对指标）。
* 变正常 → 大概率为推理`decode`模块特有的问题，可以重点参考排查上一节的**kvcache**读取与写入逻辑。

#### 2.2.1 **比对前置条件**

在`verl`强化学习训练时，`batch`维度被`flat`到`token`维度，因此训推比对时有个前提条件，需保证模型`forward`时的`token`、`feature`维度一致才能进行训练与推理双方统计值、双千比对。

在常规强化学习训练中，推理和训练的模型`forward<span> token`序列维度通常存在如下差异：

* 单`prompt`为例
  * 推理分为`prefill`+`N*decode`
    `prefill`阶段`forward`的`token`维度为`prompt_len`，`decode`阶段`forward`的`token`维度为`1`，结束后得到`prompt`+`response`
  * 训练则为整体`forward`
    * 若不存在`pad`，`token`维度为`prompt_len`+`response_len`
    * 若存在`pad`，`token`维度为`max_prompt_len`
      ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/forward_token_dim_padding.png)
* 多`prompt`为例
  * 推理会对接收到的单次请求中的多条`prompt`进行拼接，在不超过预算长度的情况下，会拼成`prompt1`+`prompt2`的`token`维度进行`prefill`计算，`decode`阶段`forward`的`token`维度为`2`，结束后得到`prompt1`+`response1`、`prompt2`+`response2`
  * 训练还可能存在`mini<span> batch`循环+梯度累积循环对`prompt`数据进行拆分，一旦进行拆分则与推理不再可能`shape`一致，因此前提需要保证训练中无`batch`循环拆分。

在常规强化学习训练中，推理和训练的模型`forward<span> feature`特征维度在切分配置不一致时（如开启不同策略的`TP`等）也会存在差异。

综上所述，要保证训推可比，前提条件为：

1. 保证训练`batch`未被拆分
   
   * 需保证每轮训练中用于梯度更新的`mini<span> batch`个数`mini_batch_num`= `1`，计算公式为:
   
   ```makeup
   mini_batch_num = train_batch_size/train_ppo_mini_batch_size
   ​
   ```
   
   * 需保证梯度累计步骤数gac =1， 计算公式为：
   
   ```makeup
   gac = train_ppo_mini_batch_size*n_resp_per_prompt/train_ppo_micro_batch_size_per_gpu/DP
   ​
   ```
   
   其中，不同训练后端的`DP`计算公式为：
   
   * `fsdp`是数据并行，`DP=world_size`
   * `megatron`有模型并行，`DP=world_size/TP/PP/CP`
   
   在`VERL`框架脚本中以上所有值对应的具体超参为：
   
   ```makeup
   data.train_batch_size=${train_batch_size}
   actor_rollout_ref.actor.ppo_mini_batch_size=${train_ppo_mini_batch_size}
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu}
   actor_rollout_ref.actor.ppo_epochs=1  # 默认为1
   actor_rollout_ref.rollout.n=${n_resp_per_prompt}
   ​
   ```
2. 关闭训练中`pad`与动态组`batch`，在`VERL`框架脚本中对应的具体超参为：
   
   ```makeup
   actor_rollout_ref.model.use_remove_padding=True
   actor_rollout_ref.actor.use_dynamic_bsz=False
   ​
   ```
3. 训练与推理的切分配置一致，保证如`TP、PP、CP`等切分策略完全一致。
4. 推理只执行prefill，在`VERL`框架脚本中对应的具体超参为：
   
   ```makeup
   data.max_response_length=1
   ​
   ```
5. 训练与推理不带response差异
   有如下两种实现对齐的方案：

* **方案1：将训练变单`prompt`。**
* ​**方案2：将推理的prefill带上`response`**​，具体为推理做完`prefill`和`decode`拿到完整的`prompt`+`response`后，设`max_response`=`1`重做一次`prefill`(`prompt`+`reponse`) 。

由于后者存在重复推理影响性能，因此本指南优先以前方案1进行操作。

**方案1具体实现**
在verl框架中针对进行训练的输入数据改动并适配`loss`计算保证不报错，对于不同后端，分别对应的修改文件如下：

* `fsdp`：`verl/workers/actor/dp_actor.py`
* `megatron`：`verl/workers/actor/megatron_actor.py`

以`fsdp`为例，具体代码修改处如下：

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
+               data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
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
​
```

如上方代码所示，通过环境变量PROMPTS_ONLY来对单prompt的训练模式进行开关控制，设为1则表示打开单prompt模式。

#### 2.2.2 routing replay

在典型的RL流程中，使用高效的推理引擎（如SGLang）进行数据采样，再将数据送入训练框架进行模型优化。对于标准稠密模型，这种框架差异或许只会带来微小的数值误差；但在MoE模型中，这个问题被急剧放大。其核心在于MoE的（Routing Mechanism）：微小的环境或实现差异，都可能导致模型为同一个输入token选择完全不同的专家组合，从而走向截然不同的计算路径。这种路由决策的不一致，可能会导致MoE模型RL训练不稳定。它使得从推理阶段获取的“经验”对于训练阶段而言变得完全不同，优化信号因此失真，最终导致灾难性的后果。
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/moe_routing_inconsistency.png)

因此在训推一致对齐过程中，对于MoE模型还需要开启routing replay。

Routing Replay有R2/R3两种变体。

（1）Vanilla Routing Replay (R2):对应verl中开关`actor_rollout_ref.actor.router_replay.mode="R2"`

机制：回放训练引擎在采样阶段计算出的专家路径。R2 的目标是减轻专家路由对策略陈旧的影响，其方法是在梯度更新阶段，复现训练引擎中 rollout 策略所选择的路由专家

作用：主要减少策略陈旧性对路由的影响。

（2）Rollout Routing Replay (R3): :对应verl中开关`actor_rollout_ref.actor.megatron.router_replay.mode="R3"`

机制：在序列生成过程中捕捉推理引擎的路由分布，并将其直接重放到训练引擎中。

作用：同时减少训练-推理偏差和策略陈旧性。
在实际训推一致对齐过程中，可以开启R3来进行精度对齐。

#### 2.2.3 msprobe工具数据采集

数据采集能力可以使用MindStudio团队提供的msprobe工具。msProbe工具通过在模型脚本中添加`PrecisionDebugger`接口并启动训练的方式，采集模型在运行过程中的精度数据。
**功能特点**

* ​**多粒度数据采集**​：支持L0(模块级)、L1(API级)以及mix(L0+L1)的不同粒度的数据采集
* ​**多种dump模式**​：提供statistics、tensor、acc_check、structure、overflow_check等多种采集模式
* ​**灵活的配置选项**​：通过config.json文件可以精确控制采集范围

**使用说明**

1. 使用精度采集工具需要先配置config.json文件 对于采集统计值来说，最常用的两种配置如下：
   
   * 采集指定步的统计量
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
     ​
     ```
   * 采集指定步的tensor值
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
     ​
     ```
   
   此外，在这两种配置基础上常见的几种修改为：* 采集级别：修改level采集"mix"（API+模块级）、“L0”（模块级）、“L1”（API）。
   
   * 统计量+md5：修改统计量中的"summary_mode"为 “md5”。
   * 指定采集步数（或卡号）：修改"step"（或"rank"），[]代表采集所有，内有数值代表采集该步，多步用英文逗号分割。
   * 筛选目标API，具体有如下两种方式：
     * 修改"list"属性，添加时会采集名称包含该字符串的所有API或模块，多个采集对象用英文逗号分隔，若字符串为模块类型则展开内部API进行同步采集。
     * 修改"scope"属性，添加开始和结束的API或模块，可采集两者之间的API或模块。

#### 2.2.4 训练侧数据采集

· 以 Megatron 后端采集为例

在verl/workers/actor/megatron_actor.py文件下

MegatronPPOActor类的compute_log_prob方法中调用forward_backward_batch前后，增加工具初始化及dump开关

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/megatron_train_dump_code.png)

#### 2.2.5 推理侧数据采集

· 以 vLLM 推理后端采集为例

1. 在vllm_ascend/worker/model_runner_v1.py文件下NPUModelRunner的__init__方法中，增加工具初始化代码，同时增加确定性开关

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/vllm_init_dump_code.png)

1. execute_model方法，开始进行debugger start（开始dump）

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/vllm_start_dump_code.png)

1. execute_model方法中_generate_process_reqs_hidden_states执行完后进行stop（结束dump）

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/vllm_stop_dump_code.png)

#### 2.2.6 自动化比对能力

对于训推一致的比对，除了`shape`之外，还存在较多的模块层级名称不一致，以`qwen2.5-0.5b`为例，若推理使用`vllm`后端、训练使用`fsdp`后端，双方对应的模块名称如下：
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/module_name_mapping.png)

可见存在如下差异：

1. 模块名称大量差异，不同前缀、不同类名。
2. `qkv_proj`双方存在拆分差异，推理为`qkv_proj`合并，训练则拆分为`q_proj`、`k_proj`、`v_proj`。
3. `rotary`位置差异，推理`rotary`放在每一层的`self_attn`内部，而训练则在进`decode`之前统一做`rotary`。
4. `silu`激活函数实现差异，推理使用`AscendSiluAndMul`融合计算`Silu`和`Mul`，而训练由于使用小算子计算未被`L0`层级采集到。

##### 比对工具使用介绍

原始比对工具在这类训推比对的情况下，会出现大量无法匹配的现象，为此，比对工具专门针对较常用的`Qwen`系列模型的训推一致比对进行了自动匹配，主要在于自动加入`mapping`列表、`qkv`合并等，在比对时加入超参`--consistent_check`，即可大幅加强比对成功率。
**单卡场景**

```makeup
msprobe compare -tp /train_dump/dump.json -gp /infer_dump/dump.json --consistent_check --backend fsdp -o ./output
​
```

**多卡场景**

```makeup
msprobe compare -tp /train_dump/step0 -gp /infer_dump/step0 --consistent_check --backend fsdp -o ./output
​
```

##### 可视化比对工具介绍

在可视化比对工具中，提供`点点匹配`功能，用户可通过浏览器界面，鼠标选择两个待匹配的灰色节点进行匹配。当前仅支持统计值数据模式。

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/visual_point_matching.png)

## 3. 实战案例复盘

### 3.1 kimi2.5 训推对齐案例

#### 3.1.1 问题现象

在kimi2.5强化学习训练场景中，发现reward曲线震荡，没有上升趋势。logp_diff`首step差异3.5`，该指标应该控制在0.01以内。基于此现象，优化怀疑训推不一致导致的训练精度问题。

#### 3.1.2 定位思路

整体思路：

1. 缩小规模复现，整体512卡场景logp_diff 3.5，缩小至双机32卡测试，发现`logp_diff 0.7`，依旧很大。
2. 剥离强化学习训练流程，独立执行训练和推理，使用相同输入分别执行，采集dump数据
3. 优先尝试减层，比如先排查dense层，训推各减层至1层；确保dense层对齐后，再加层，比较moe层，完成训推对齐。

#### 3.1.3 定位流程

前置操作，训练推理使用相同输入，进行数据dump，优先采集模块级数据，方便匹配对齐，快速锁定问题模块。

1. 首差异模块MLA模块：
   如下图所示，左侧为推理实现，使用NPU自定义算子`npu_ring_mla`；右侧为训练实现，使用小算子封装为MLA模块；两者输出存在明显差异（四个统计量维度）
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mla_output_diff.png)
   再进一步观测输入，可以发现，两个模块虽然功能一致，但是输入定义不同，以attention的q矩阵输入为例，推理模块分别将q_nope和q_rope作为输入，而训练模块直接将q_nope和q_rope拼接为q，作为MLA模块输入。观测二者统计量发现，输入就无法对齐，因此需要进一步向前排查：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mla_input_diff.png)
2. q_rope差异：
   进一步向前排查，找到训练侧对于q_nope和q_rope拼接为q的实现操作，再进一步与推理的q_nope和q_rope比对，发现q_rope操作没有对齐
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_rope_diff.png)
   进一步向前排查观测q_rope的计算来源，首先可以确认训推关于rope的实现方式不一致
   推理实现：使用`npu_interleave_rope`融合算子
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_fused_rope_impl.png)
   训练实现：使用一系列小算子拼接实现
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/train_composed_rope_impl.png)
   其中两者均用到了cos和sin的编码矩阵，通过dump数据比对发现，cos和sin编码矩阵就已经无法对齐
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/cos_sin_matrix_diff.png)
   由此基本可以推断出参与计算cos和sin的输入freqs存在差异.
3. freqs来源分析及排查
   cos和sin的计算均来自于freqs，向上追溯freqs来源，可以发现freqs实际为内部的`self.rotary_pos_emb`，该变量通过实例化的`YarnRotaryEmbedding`而得到，因此怀疑`YarnRotaryEmbedding`的实例化即传参有问题

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/freqs_source_code.png)

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/yarn_rotary_embedding_init.png)
YarnRotaryEmbedding入参，与推理做对比
发现
参与计算的original_max_position_embeddings=1024
实际外层配置为4096，即original_max_position_embeddings的期望值是4096，实际传入的是1024
进一步确认，发现是mindspeed对`YarnRotaryEmbedding`的适配与megatron存在差异

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mindspeed_yarn_impl_1.png)

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mindspeed_yarn_impl_2.png)

#### 3.1.4 问题修复

修复mindspeed实现，使得`original_max_position_embeddings=self.config.original_max_position_embeddings,`再次训练，logp_diff下降为0.004左右，reward训练100步稳定上升。

### 3.2 DS3.2 训推对齐案例

#### 3.2.1 背景

在DeepseekV3.2强化学习训练场景中，长跑曲线如下所示，可以看出在50步左右已经训崩，观察logp diff曲线，logp diff偏大，首个step在0.05左右且保持逐渐上升，将方向转移到训推不一致问题。
![](../figures/cases/ascend_rl_train_infer_consistency_alignment/ds32_reward_collapse.png)

![](../figures/cases/ascend_rl_train_infer_consistency_alignment/ds32_logp_diff_curve.png)

#### 3.2.2 定位思路

1. 将模型从61层减层至5层，训练和推理保持相同切分策略，输入保持一致，采集dump数据。
2. 首先检查大模块的输入输出，找到对不齐的位置，随后根据调用堆栈逐步缩小排查范围，直至找到第一个输入一致输出不一致的位置。
3. 查看输入一致输出不一致的训推具体实现代码，修改代码对齐处理流程，重新dump数据进行比较，直至完成训推对齐。

#### 3.2.3 定位流程

固定输入的prompt，batchsize=1，response=1，保证训练和推理采用相同的切分策略，采集减层的mix级别数据，进行详细比对。

1. 首先粗粒度排查大模块级别，发现经过整个大的MLA模块输入一致，输出不一致（左边推理，右边训练），如下图所示：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/mla_module_inout_diff.png)
   接着往上找MLA模块输出的具体来源，发现该mla的输出是由一个all_reduce操作而来，观察该all_reduce的输入输出，结果如下，观察到其输入已经有差异了，其他卡的输入也有差异，但差异较小，但经过all_reduce之后，输出的最大值差值明显。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/all_reduce_diff.png)
   紧接着往上找all_reduce的输入的来源，如下图所示，左边推理是一个linear操作，右边训练是一个matmul操作，并且观察输入值和权重在统计值上数值一致，但是输出不一致。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_linear_op.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/train_matmul_op.png)
   linear和matmul这两个算子不应该有精度问题，我们初步怀疑这两个算子在输入就已经不一致了（​**虽然在统计值上表现一致**​），于是我们进行单算子复现，发现输入一致，这两个算子输出是一致的，我们比对输入和权重，发现权重完全一致，输入存在对不上的情况。
   于是我们紧接着继续往前找该输入的来源，在推理和训练中找到了相应的api输出，具体结果如下，具体表现为输入一致，输出也一致。紧接着继续往前找该输入的来源，在推理和训练中找到了相应的api输出，具体结果如下，具体表现为输入一致，输出也一致。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/upstream_api_output_1.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/upstream_api_output_2.png)
   接着往上找这两个算子的输入来源，目前找到下面的操作，虽然输入的统计值一样，但是save该两个算子的输入，进行逐元素比对，发现该两个算子的输入有差异。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/elementwise_diff_1.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/elementwise_diff_2.png)
   根据调用堆栈一直往前找，推理侧的来源是经过左边的vllm_ascend的自定义融合算子，训练侧的来源是经过右边的cann8.5中的融合算子，对比结果如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fused_op_compare.png)
   因为工具原因没采到这两个融合算子的输入输出，只能往前找输入的q，k，v的来源。先排差q的一致性，q的来源如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_source_code.png)
   通过逐元素对比，发现q是一致的，结果如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_elementwise_match.png)
   由于推理和训练的key和value的语义不一致，维度也不一致，没办法直接比较，我们直接比较推理和训练的indexer模块输出的topk_indices，目前可以确认训推输出的topk_indices在统计的均值上就已经对不齐了。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/topk_indices_diff.png)
   topk_indices训推调用的过程如下，由于msprobe没采到lightning_indexer的输入输出，修改如下代码重新采集dump数据，dump级别为tensor+mix。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_dump_code_mod.png)
2. npu_lightning_indexer算子输入输出排查
   根据采集的结果，我们对输入的query，key，weights和输出topk_indices通过逐元素进行比对，比对结果差异较大：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_input_output_diff.png)
   紧接着首先排查weights的来源，训练和推理的调用如下所示，都是weights_proj操作，其对应的都是一个linear操作，如下所示：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/train_weights_proj_call.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_weights_proj_call.png)
   我们对其入参x，weight和输出output进行逐元素对比，输入x，weight和输出output能完全对齐，对比结果如下。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_proj_align_1.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_proj_align_2.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_proj_align_3.png)
   对于linear之后的输出，推理和训练分别进行了额外的操作，推理做了一次maybe_all_gather，训练做了两次缩放，如下图中的962和695行的输入输出是完全能对齐的。
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_extra_ops_code.png)
   经过上图中的963和696之后，weight的值差值非常明显，对比结果如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_scaled_diff.png)
   接下来在来定位lighting_indexer的输入的q，推理和训练的调用位置如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_source_code.png)
   输入qr，权重weight和输出q逐元素能够完全对齐，对比结果如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_align_1.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_align_2.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/indexer_q_align_3.png)
   接下来推理和训练分别对q做split和rope操作（左边推理，右边训练）：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/split_rope_ops_compare.png)
   推理和训练分别走了不同的rope操作，经过rope操作之后，输出的q_pe和x_pe统计值能对上，但是逐元素对比对不上：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/rope_output_stats_1.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/rope_output_stats_2.png)
   走进apply_rotary_pos_emb_bshd_in_complex函数内部，可以看到针对输出进行了一个fp32转bf16的操作：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fp32_to_bf16_code.png)
   类型转换操作导致有一定的精度丢失：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/precision_loss_diff.png)
   这个转换操作，导致训推输出的x_pe和推理的输出q_pe对不齐了。紧接着训练对q执行了一次rotate_activation操作，推理侧没有执行该操作，rotate_activation里面执行了哈达玛转换，调用对比如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/rotate_activation_call.png)
   经过此操作，训练和推理的q完全对不齐全了，如下图所示：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/q_after_rotate_diff.png)
   k也是同理，训练和推理对于q和k，主要有两点差异，对q_pe和k_pe执行rope操作时，训练是先使用fp32进行计算，最后转成bfloat16，推理全程用的是bfloat16，训练会有精度丢失。
   对于npu_lighting_indexer算子的输入weights，训练和推理的操作也不一致，操作对比如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_ops_compare.png)
   推理和训练经过weights_proj操作之后能完全对齐，推理侧经过maybe_all_gather_and_maybe_unpad操作之后，weights不会改变，训练经过了右侧的两次缩放，导致weights对不齐，weights统计值对比如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/weights_stats_compare.png)
   基于上述排查过程，目前把问题定位到是推理和训练在indexer模块里算topk_indices（调用npu_lightning_indexer融合算子）之前，npu_lightning_indexer有3个输入q，k和weights，推理和训练对q，k和weights的处理逻辑不一样，具体表现为如下两点：
   1.针对q和k，训练侧进行rope时使用的精度是fp32，rope完之后进行了一次降精操作，fp32->bfloat16，训练侧随后多进行一次rotate_activation操作，里面执行了一次哈达玛转换。
   2.针对weights，训练侧相比于推理侧多进行了一次缩放操作。
   
   #### 3.2.4 问题修复
   
   让推理严格对齐训练侧实现，针对q和k加上rotate_activation操作，针对weights，与训练侧对齐缩放规则，推理侧具体修改如下：
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_fix_code_1.png)
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/infer_fix_code_2.png)
   对齐之后进行全层拉起实验，22个steps曲线图如下所示：
   
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fixed_reward_curve.png)
   
   ![](../figures/cases/ascend_rl_train_infer_consistency_alignment/fixed_logp_diff_curve.png)
   从曲线来看，reward正常上涨，logp diff没有上涨趋势，并且logp diff保持在千分位差异，目前没有资源进行长跑，只能通过观察前22steps得出上述indexer对齐改动能够解决训推不一致问题。

