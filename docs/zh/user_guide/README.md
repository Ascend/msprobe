# msProbe 使用指南

MindStudio Probe（msProbe）针对昇腾 AI 处理器提供全场景精度调试能力，覆盖数据采集、精度比对、可视化以及多种进阶调试功能。下文按框架和使用场景汇总各场景下支持的完整功能项，请根据实际使用场景选择对应功能并参考相应文档进行配置和使用。

## PyTorch

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 数据采集 | 通过 `config.json` 配置，完成 msProbe 精度数据采集操作 | [数据采集](dump/pytorch_data_dump_instruct.md) |
| 分级可视化构图比对 | 将 msProbe 工具 dump 的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对 | [分级可视化构图比对](accuracy_compare/pytorch_visualization_instruct.md) |
| 精度比对 | 将 msProbe 工具 dump 的精度数据进行精度比对，进而定位精度问题 | [精度比对](accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| 训练状态监测 | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况 | [训练状态监测](monitor_instruct.md) |

<details>
<summary><b>展开更多功能</b></summary>

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 训练前配置检查 | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异 | [训练前配置检查](config_check_instruct.md) |
| 数据加载 | 加载已采集的模块级输入 tensor，覆盖模型实际输入，隔离前序累计误差 | [数据加载](dump/pytorch_data_load_instruct.md) |
| 精度预检 | 在昇腾 NPU 上扫描训练模型中的所有 API，给出精度情况的诊断和分析 | [精度预检](accuracy_checker/pytorch_accuracy_checker_instruct.md) |
| 编译精度比对 | 对启用 `torch.compile` 的模型进行 eager 与 compile 逐模块精度对比，定位编译引入的前向、反向或 loss 差异 | [编译精度比对](accuracy_compare/pytorch_compile_accuracy_compare_instruct.md) |
| checkpoint 比对 | 训练过程中或结束后，比较两个不同的 checkpoint，评估模型相似度 | [checkpoint 比对](checkpoint_compare_instruct.md) |
| 整网首个溢出节点分析 | 多 rank 场景下通过 dump 数据找到首个出现 Nan 或 Inf 的节点 | [整网首个溢出节点分析](overflow_check/overflow_check_instruct.md) |
| 趋势可视化 | 将 msProbe 工具数据采集或训练状态监测的统计量数据从迭代步数、节点 rank 和张量目标三个维度进行趋势可视化 | [趋势可视化](accuracy_compare/trend_visualization_instruct.md) |

</details><br>

## vLLM 推理场景

| 细分场景 | 功能项 | 功能说明 | 参考文档 |
|---|---|---|---|
| Eager/图模式 | 数据采集 | 完成 msProbe 精度数据采集操作 | [数据采集](dump/vllm_dump_instruct.md) |
| | 数据比对 | 将 msProbe 工具 dump 的精度数据进行精度比对，进而定位精度问题<br/>请参考分级可视化构图比对或精度比对 | [分级可视化构图比对](accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| torchair | 数据采集 | 通过 `set_ge_dump_config` 接口完成精度数据采集操作 | [数据采集](dump/torchair_dump_instruct.md) |
| | 精度比对 | 将 msProbe 工具 dump 的精度数据进行精度比对，进而定位精度问题 | [精度比对](accuracy_compare/torchair_compare_instruct.md) |
| 通用场景 | 推理异常检测 | 获取 vLLM 推理输出，感知异常问题 | [推理异常检测](response_anomaly_instruct.md) |

## SGLang 推理场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 数据采集 | 完成 msProbe 精度数据采集操作 | [数据采集](dump/sglang_eager_dump_instruct_new.md) |
| 数据比对 | 将 msProbe 工具 dump 的精度数据进行精度比对，进而定位精度问题 | [分级可视化构图比对](accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](accuracy_compare/pytorch_accuracy_compare_instruct.md) |

## VeRL 场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| VeRL 超参比对与关键超参校验 | VeRL 训练过程中或结束后，比对两台不同服务器上训练日志中采集到的真实超参配置，或者校验配置是否与关键超参取值相同，辅助用户高效比对真实超参值配置，加速定位因配置差异所引发的训练精度问题 | [VeRL 超参比对与关键超参校验](verl_param_compare_or_verify_instruct.md) |
| 异步架构 VeRL 训推一致性比对数据采集 | VeRL ≥ v0.7.0，保证 VeRL 训推一致性比对时的输入 shape 一致的比对数据采集 | [异步架构 VeRL 训推一致性比对数据采集](dump/verl_async_consistency_preprocess_dump.md) |
| VeRL V1 Trainer 训推一致性比对数据采集 | VeRL 0.9.0.dev，保证 VeRL 训推一致性比对时的输入 shape 一致的比对数据采集 | [VeRL V1 Trainer 训推一致性比对数据采集](dump/verl_v1_trainer_consistency_preprocess_dump.md) |
| FSDP 训练后端 VeRL 训推一致性比对数据采集 | VeRL < v0.7.0，FSDP 训练后端，保证 VeRL 训推一致性比对时的输入 shape 一致的比对数据采集 | [FSDP 训练后端 VeRL 训推一致性比对数据采集](dump/verl_fsdp_consistency_preprocess_dump.md) |
| Megatron 训练后端 VeRL 训推一致性比对数据采集 | VeRL < v0.7.0，Megatron 训练后端，保证 VeRL 训推一致性比对时的输入 shape 一致的比对数据采集 | [Megatron 训练后端 VeRL 训推一致性比对数据采集](dump/verl_megatron_consistency_preprocess_dump.md) |
| 精度比对 | 将 msProbe 工具 dump 的精度数据进行精度比对，进而定位精度问题 | [精度比对](accuracy_compare/pytorch_accuracy_compare_instruct.md#verl训推一致性比对场景) |
| 训推一致性监控：逐 Token 级别的 probs_diff 监控 | 训推一致性监控，逐 Token 级别监控 probs_diff | [训推一致性监控：逐 Token 级别的 probs_diff 监控](dump/verl_token_level_probs_diff_monitoring.md) |
| VeRL 训推交叉打桩 | 在两个阶段结束位置打桩，对阶段输出进行替换，通过替换后的训练效果来判断阶段输出是否存在异常 | [VeRL 训推交叉打桩](verl_cross_validation.md) |

## Slime 场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 数据采集 | 训推阶段完成 msProbe 精度数据采集操作 | [数据采集](dump/slime_train_rollout_dump_instruct.md) |
| Slime 超参比对 | Slime 训练过程中或结束后，比对两台不同服务器上训练日志中采集到的真实超参配置是否取值相同，辅助用户高效比对真实超参值配置，加速定位因配置差异所引发的训练精度问题 | [Slime 超参比对与关键超参校验](slime_param_compare_instruct.md) |
| 训推一致性比对数据采集 | Megatron 训练后端，SGLang 推理引擎，保证训推一致性比对时的输入 shape 一致的比对数据采集操作 | [Slime 框架训推一致性预处理与数据采集](dump/slime_consistency_preprocess_dump.md) |

## ATB 推理场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 数据采集 | 通过在 ATB 模型运行前，加载 ATB dump 模块的方式，实现对 ATB 模型运行过程中的精度数据的采集 | [数据采集](dump/atb_data_dump_instruct.md) |
| 精度比对 | 将 ATB dump 的精度数据进行精度比对，进而定位精度问题 | [精度比对](accuracy_compare/atb_data_compare_instruct.md) |
| 数据转换 | 将 ATB dump 的精度数据转换为 numpy（.npy）或 PyTorch tensor（.pt）格式文件 | [数据转换](dump/data_parse_instruct.md) |

## 离线模型推理场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 数据采集 | 完成 msProbe 精度数据采集操作 | [数据采集](dump/infer_offline_dump_instruct.md) |
| 精度比对 | 提供一键式离线模型比对功能，仅需输入模型即可完成比对，无需提前采集数据，快速输出结果 | [精度比对](accuracy_compare/infer_compare_offline_model_instruct.md) |
| 离线模型数据精度比对 | 提供离线模型数据比对功能，输入离线模型的 dump 数据进行精度比对 | [离线模型数据精度比对](accuracy_compare/offline_data_compare_instruct.md) |
| 数据转换 | 将离线模型的 dump 数据转换为 numpy（.npy）或 PyTorch tensor（.pt）格式文件 | [数据转换](dump/data_parse_instruct.md) |

## MindSpore 训练场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 训练前配置检查 | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异 | [训练前配置检查](config_check_instruct.md) |
| 数据采集 | 通过 `config.json` 配置，完成 msProbe 精度数据采集操作 | [数据采集](dump/mindspore_data_dump_instruct.md) |
| 精度预检 | 在昇腾 NPU 上扫描训练模型中的所有 API，给出精度情况的诊断和分析 | [精度预检](accuracy_checker/mindspore_accuracy_checker_instruct.md) |
| 分级可视化构图比对 | 将 msProbe 工具 dump 的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对 | [分级可视化构图比对](accuracy_compare/mindspore_visualization_instruct.md) |
| 精度比对 | 将 msProbe 工具 dump 的精度数据进行精度比对，进而定位精度问题 | [精度比对](accuracy_compare/mindspore_accuracy_compare_instruct.md) |
| 训练状态监测 | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况 | [训练状态监测](monitor_instruct.md) |
| checkpoint 比对 | 训练过程中或结束后，比较两个不同的 checkpoint，评估模型相似度 | [checkpoint 比对](checkpoint_compare_instruct.md) |
| 趋势可视化 | 将 msProbe 工具数据采集或训练状态监测的统计量数据从迭代步数、节点 rank 和张量目标三个维度进行趋势可视化 | [趋势可视化](accuracy_compare/trend_visualization_instruct.md) |

## MSAdapter 场景

| 功能项 | 功能说明 | 参考文档 |
|---|---|---|
| 数据采集 | 通过 `config.json` 配置，完成 msProbe 精度数据采集操作 | [数据采集](dump/msadapter_data_dump_instruct.md) |
| checkpoint 比对 | 训练过程中或结束后，比较两个不同的 checkpoint，评估模型相似度 | [checkpoint 比对](checkpoint_compare_instruct.md) |

## 📚 补充材料

🔹 [PyTorch 场景的精度数据采集基线报告](../baseline/pytorch_data_dump_perf_baseline.md)

🔹 [MindSpore 场景的精度预检基线报告](../baseline/mindspore_accuracy_checker_perf_baseline.md)

🔹 [MindSpore 场景的精度数据采集基线报告](../baseline/mindspore_data_dump_perf_baseline.md)

🔹 [训练状态监测工具标准性能基线报告](../baseline/monitor_perf_baseline.md)
