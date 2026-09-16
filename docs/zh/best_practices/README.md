# msProbe 典型案例

本章节通过精度调试的典型场景提供对应的精度调试案例，包含如下案例。

下列案例以基础案例和训练、推理两大场景分类，其中训练、推理场景下以问题根因分子类。

**基础案例**

- [大模型训练精度定位指南](./train_debug_guide.md)
- [大模型推理精度定位指南](./infer_debug_guide.md)
- [常见框架dump工具使能](./dump_enable_guide.md)

**训练**

- AI框架
  - [基于昇腾的MOVA模型NaN问题定位分析](./ascend_mova_model_nan_analysis.md)
- CANN算子
  - [基于昇腾的80卡大模型训练确定性排查](./ascend_80_card_large_model_training_determinism_troubleshooting.md)
  - [基于昇腾的多模态理解训练NaN问题定位](./ascend_multimodal_training_nan_analysis.md)
- 模型代码
  - [基于昇腾的Qwen2.5-omni模型确定性排查](./ascend_qwen25_omni_determinism_troubleshooting.md)
- 训练数据
  - [基于昇腾的Gemma模型重复训练Loss差异定位](./ascend_gemma_retrain_loss_diff_analysis.md)
  - [基于昇腾的mmdetection训练NaN问题定位](./ascend_mmdetection_training_nan_localization.md)
- 引擎&加速库
  - [基于昇腾的复杂dump数据分析GradNorm NaN定位实践](./ascend_complex_dump_gradnorm_nan_analysis.md)
  - [基于昇腾的Kimi2.5训推一致性优化](./ascend_kimi2_5_train_infer_consistency_optimization.md)
  - [基于昇腾的GLM5适配与训推一致性优化](./ascend_glm5_adaptation_training_Inference_consistency_optimization.md)
  - [强化学习训推一致性排查](./ascend_rl_train_infer_consistency_alignment.md)
- 非昇腾软件栈问题
  - [基于昇腾的Qwen2.5-omni框架迁移Loss差异定位](./ascend_qwen2_5_omni_loss_diff_localization.md)

**推理**

- CANN算子
  - [CANN版本升级后的推理请求回复乱码问题](./problem_of_garbled_response_after_cann_upgrade.md)
- 模型代码
  - [基于昇腾的tora模型推理差异定位](./ascend_tora_inference_diff_localization.md)
  - [基于昇腾的Qwen-Image-Edit模型TP8花图问题定位](./ascend_qwen_image_edit_tp8_distorted_image_localization.md)
- 引擎&加速库
  - [基于昇腾的VL模型迁移vLLM精度问题](./ascend_vl_migration_vllm_precision.md)
  - [基于昇腾的DeepSeek-V4 GPQA精度优化](./ascend_deepseek_v4_gpqa_accuracy_optimization.md)
  - [Qwen2.5模型推理回复复读问题](./inference_reply_replay_problem_of_Qwen2.5.md)
