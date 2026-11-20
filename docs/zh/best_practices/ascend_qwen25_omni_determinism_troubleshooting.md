# 基于昇腾的Qwen2.5-omni模型确定性排查

本文档介绍基于昇腾平台排查Qwen2.5-omni模型训练loss与grad_norm不一致问题的方法，通过dump比对、单算子复现与plog分析逐层定位到业务代码重复设置随机种子导致确定性失效的根因，适用于需要解决训练复现性问题的用户。

## 问题背景

在深度学习训练中，结果的可复现性（确定性）直接关系到问题定位与实验的可信度。本文记录了在昇腾平台上，Qwen2.5-omni 模型重复训练时 loss 与 grad_norm 不一致的排查过程。通过 Dump 比对、单算子复现与 PLOG 分析等手段逐层收敛，最终定位到业务代码中随机种子重复设置导致确定性失效的问题，为类似的复现性问题提供了排查思路。

## 问题现象

在 NPU 上对 Qwen2.5-omni 模型进行重复训练时，得到的 loss 与 grad_norm 不一致。

## 定位过程

### 步骤 1：dump 两次数据进行比对

比对发现 cross_entropy 存在确定性问题。

![两次dump的cross_entropy输出比对截图，可见同一step的cross_entropy数值不一致](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/dump_compare_cross_entropy_1.png)

![第二次dump的cross_entropy输出截图，与首次结果逐step比对可看出数值差异](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/dump_compare_cross_entropy_2.png)

尝试通过 to CPU 后进行对齐。to CPU 后第一步 loss 一致，grad_norm 仍偶现不一致，且经过几步后会累积到 loss 不一致。

![数据to CPU对齐后前几个step的loss与grad_norm对比，loss一致但grad_norm出现波动](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/tocpu_align_loss_gradnorm_1.png)

![to CPU对齐后中段step的grad_norm对比截图，可见grad_norm仍偶现不一致](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/tocpu_align_loss_gradnorm_2.png)

![to CPU对齐后后续step的loss对比截图，可见误差累积后loss再次不一致](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/tocpu_align_loss_gradnorm_3.png)

继续分析发现 grad_norm 差异来源为第 0 步 rank3 上的 embed_token 反向存在偶现的差异，前向完全一致。

![第0步rank3上embed_token反向梯度比对截图，反向存在偶现差异、前向完全一致](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/gradnorm_diff_embed_tokens_backward.png)

### 步骤 2：多次采集 embed_token 的 tensor 数据，直到采集到 2 次不一样的

![多次采集embed_token tensor数据的采样截图，用于筛出两次数值不同的结果](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/embed_tokens_tensor_sampling.png)

### 步骤 3：进行单算子复现

复现脚本：

![单算子复现脚本的代码截图，用于在本地重复运行embedding backward算子并比对结果](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/single_op_repro_script.png)

单算子结果：单算子复现和整网训练的结果不一致。整网跑几遍 embedding backward 结果概率性有变化，本地单算子重复跑很多遍都没有变化。整网采集的 2 次结果中，1 次跟单算子的自动 backward 一致，1 次跟单算子的手动算梯度一致。

### 步骤 4：怀疑存在其他干扰，排查 plog

plog 中底层 aclnn 算子 deterministic 为 0，中途操作部分为1，部分为0，确认为中途 1 改 0。

![plog中aclnn算子deterministic字段日志截图，显示该标识初始为0](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/plog_aclnn_deterministic_1.png)

![plog中aclnn算子deterministic字段日志截图，显示运行中途部分记录为1](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/plog_aclnn_deterministic_2.png)

![plog中aclnn算子deterministic字段日志截图，显示中途该标识由1被改为0](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/plog_aclnn_deterministic_3.png)

### 步骤 5：检查固定的代码

项目中全局搜索 torch.use_deterministic_algorithms，只有这一个地方用到。

![全局搜索torch.use_deterministic_algorithms的结果截图，仅trainer中一处命中](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/search_use_deterministic_algorithms.png)

发现第 40 行有一个 fix_randn。点开看到，除了 trainer 中的 seed_all(mode=True) 外，业务代码中自定义了 fix_randn 对 randn 接口进行了装饰 wrap，该 wrap 内重复进行了 seed_all() 且没带 mode。

![第40行fix_randn装饰器代码截图，wrap内重复调用未带mode的seed_all()](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/fix_randn_wrap_code.png)

修改为 seed_all(mode=True) 后，确定性固定成功。

![改为seed_all(mode=True)后重复训练的确定性验证结果截图，多轮运行loss与grad_norm一致](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/determinism_fixed_result_1.png)

![修复后再次重复训练的确定性验证结果截图，进一步确认多轮结果一致](../figures/cases/ascend_qwen25_omni_determinism_troubleshooting/determinism_fixed_result_2.png)

## 问题根因

业务代码中自定义了 fix_randn，对 randn 接口进行了装饰 wrap，该 wrap 内重复进行了 seed_all() 且没带 mode，覆盖了环境变量，导致确定性无法固定。

## 问题结论

1. 本次 Qwen2.5‑omni 训练 loss、grad_norm 复现失败，**并非昇腾算子本身非确定性缺陷**，而是上层业务代码逻辑破坏确定性开关状态引发的偶现问题。
2. `seed_all()` 不传 `mode=True` 时，不会开启 PyTorch 确定性算法开关 `torch.use_deterministic_algorithms(True)`；业务装饰器 `fix_randn` 在运行过程中反复调用无参 `seed_all()`，会将已经置为 True 的确定性开关重置为 False，传导到底层 CANN，使得 aclnn 算子 deterministic 标识由 1 变为 0，算子切换到非确定路径执行。

## 解决方案

上层业务代码修改 `fix_randn` 装饰器内部调用，将 `seed_all()` 修改为 `seed_all(mode=True)`，保证每次设置随机种子时，同步维持 `torch.use_deterministic_algorithms(True)` 确定性开关处于打开状态，避免开关被意外重置。
