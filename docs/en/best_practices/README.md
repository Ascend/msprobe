# msProbe Typical Cases

This chapter provides corresponding precision debugging cases based on typical precision debugging scenarios.

The following cases are classified into basic cases and the two major scenarios of training and inference. Under the training and inference scenarios, they are further categorized by root cause.

**Basic Cases**

- [Large Model Training Precision Localization Guide](./train_debug_guide.md)
- [Large Model Inference Precision Localization Guide](./infer_debug_guide.md)
- [Common Framework Dump Tool Enabling](./dump_enable_guide.md)

**Training**

- AI Framework
  - [NaN Problem Localization for the Ascend-based MOVA Model](./ascend_mova_model_nan_analysis.md)
- CANN Operators
  - [Determinism Troubleshooting for Ascend-based 80-Card Large Model Training](./ascend_80_card_large_model_training_determinism_troubleshooting.md)
  - [NaN Problem Localization for Ascend-based Multimodal Understanding Training](./ascend_multimodal_training_nan_analysis.md)
- Model Code
  - [Determinism Troubleshooting for the Ascend-based Qwen2.5-omni Model](./ascend_qwen25_omni_determinism_troubleshooting.md)
- Training Data
  - [Loss Difference Localization for Ascend-based Gemma Model Retraining](./ascend_gemma_retrain_loss_diff_analysis.md)
  - [NaN Problem Localization for Ascend-based mmdetection Training](./ascend_mmdetection_training_nan_localization.md)
- Engines & Acceleration Libraries
  - [GradNorm NaN Localization Based on Complex Dump Data Analysis on Ascend](./ascend_complex_dump_gradnorm_nan_analysis.md)
  - [Kimi2.5 Training-Inference Consistency Optimization on Ascend](./ascend_kimi2_5_train_infer_consistency_optimization.md)
  - [GLM-5 Adaptation and Training-Inference Consistency Optimization on Ascend](./ascend_glm5_adaptation_training_Inference_consistency_optimization.md)
  - [RL Training-Inference Consistency Troubleshooting](./ascend_rl_train_infer_consistency_alignment.md)
- Non-Ascend Software Stack Issues
  - [Loss Difference Localization for Qwen2.5-omni Migration on Ascend](./ascend_qwen2_5_omni_loss_diff_localization.md)

**Inference**

- CANN Operators
  - [Garbled Inference Response Problem After CANN Upgrade](./problem_of_garbled_response_after_cann_upgrade.md)
- Model Code
  - [Inference Difference Localization for the Ascend-based tora Model](./ascend_tora_inference_diff_localization.md)
  - [TP8 Distorted Image Localization for the Ascend-based Qwen-Image-Edit Model](./ascend_qwen_image_edit_tp8_distorted_image_localization.md)
- Engines & Acceleration Libraries
  - [Precision Problem of Ascend-based VL Model Migration to vLLM](./ascend_vl_migration_vllm_precision.md)
  - [DeepSeek-V4 GPQA Precision Optimization on Ascend](./ascend_deepseek_v4_gpqa_accuracy_optimization.md)
  - [Inference Response Repetition Problem of the Qwen2.5 Model](./inference_reply_replay_problem_of_Qwen2.5.md)
