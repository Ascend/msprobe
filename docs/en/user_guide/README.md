# msProbe User Guide

MindStudio Probe (msProbe) provides full-scenario precision debugging capabilities for Ascend AI processors, covering data collection, precision comparison, visualization, and various advanced debugging functions. The following summarizes the complete function items supported in each scenario by framework and usage scenario. Please select the corresponding function according to your actual usage scenario and refer to the corresponding document for configuration and use.

## PyTorch

| Function Item | Function Description | Reference Document |
|---|---|---|
| Data Collection | Configure through `config.json` to complete msProbe precision data collection. | [Data Collection](dump/pytorch_data_dump_instruct.md) |
| Hierarchical Visualization Graph Comparison | Parse the precision data dumped by the msProbe tool, restore the model graph structure, and implement precision data comparison at each level of the model. | [Hierarchical Visualization Graph Comparison](accuracy_compare/pytorch_visualization_instruct.md) |
| Precision Comparison | Perform precision comparison on the precision data dumped by the msProbe tool, and then localize precision problems. | [Precision Comparison](accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| Training Status Monitoring | Collect and aggregate the intermediate values of the network layer, optimizer, and communication operators during model training, to help diagnose anomalies in the computation, communication, and optimizer parts during model training. | [Training Status Monitoring](monitor_instruct.md) |

<details>
<summary><b>Expand More Functions</b></summary>

| Function Item | Function Description | Reference Document |
|---|---|---|
| Pre-training Configuration Check | Before training or precision comparison, compare the configuration differences between two environments that may affect training precision. | [Pre-training Configuration Check](config_check_instruct.md) |
| Data Loading | Load the collected module-level input tensors to override the actual model input and isolate the accumulated errors of the preceding sequence. | [Data Loading](dump/pytorch_data_load_instruct.md) |
| Precision Pre-check | Scan all APIs of the training model on the Ascend NPU, and provide diagnosis and analysis of the precision situation. | [Precision Pre-check](accuracy_checker/pytorch_accuracy_checker_instruct.md) |
| Compilation Precision Comparison | Perform eager and compile module-by-module precision comparison for models with `torch.compile` enabled, to localize forward, backward, or Loss differences introduced by compilation. | [Compilation Precision Comparison](accuracy_compare/pytorch_compile_accuracy_compare_instruct.md) |
| checkpoint Comparison | During or after training, compare two different checkpoints and evaluate model similarity. | [checkpoint Comparison](checkpoint_compare_instruct.md) |
| Network-wide First Overflow Node Analysis | In multi-rank scenarios, find the node where Nan or Inf first appears through dump data. | [Network-wide First Overflow Node Analysis](overflow_check/overflow_check_instruct.md) |
| Trend Visualization | Visualize the trend of the statistical data from msProbe data collection or training status monitoring from the three dimensions of iteration step, node rank, and Tensor target. | [Trend Visualization](accuracy_compare/trend_visualization_instruct.md) |

</details><br>

## vLLM Inference Scenario

| Sub-scenario | Function Item | Function Description | Reference Document |
|---|---|---|---|
| Eager/Graph Mode | Data Collection | Complete the msProbe precision data collection operation. | [Data Collection](dump/vllm_dump_instruct.md) |
| | Data Comparison | Perform precision comparison on the precision data dumped by the msProbe tool, and then localize precision problems.<br/>Please refer to *Hierarchical Visualization Graph Comparison or Precision Comparison* | [Hierarchical Visualization Graph Comparison](accuracy_compare/pytorch_visualization_instruct.md)<br>[Precision Comparison](accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| torchair | Data Collection | Complete the precision data collection operation through the `set_ge_dump_config` interface. | [Data Collection](dump/torchair_dump_instruct.md) |
| | Precision Comparison | Perform precision comparison on the precision data dumped by the msProbe tool, and then localize precision problems. | [Precision Comparison](accuracy_compare/torchair_compare_instruct.md) |
| General Scenario | Inference Anomaly Detection | Obtain the vLLM inference output and perceive anomaly problems. | [Inference Anomaly Detection](response_anomaly_instruct.md) |

## SGLang Inference Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| Data Collection | Complete the msProbe precision data collection operation. | [Data Collection](dump/sglang_eager_dump_instruct_new.md) |
| Data Comparison | Perform precision comparison on the precision data dumped by the msProbe tool, and then localize precision problems. | [Hierarchical Visualization Graph Comparison](accuracy_compare/pytorch_visualization_instruct.md)<br>[Precision Comparison](accuracy_compare/pytorch_accuracy_compare_instruct.md) |

## verl Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| verl Hyperparameter Comparison and Key Hyperparameter Verification | During or after verl training, compare the real hyperparameter configurations collected in the training logs on two different servers, or verify whether the configuration has the same values as the key hyperparameters, helping users efficiently compare real hyperparameter value configurations and accelerating the localization of training precision problems caused by configuration differences. | [verl Hyperparameter Comparison and Key Hyperparameter Verification](verl_param_compare_or_verify_instruct.md) |
| verl Training-Inference Weight Consistency Verification | In the verl training and inference stages, weights may have precision problems due to factors such as shard conversion, transmission inconsistency, and pointer offset during the transmission and format conversion stages; a full-process consistency verification solution is provided for localization analysis. | [verl Training-Inference Weight Consistency Verification](rl_weight_consistency_verify.md) |
| Asynchronous Architecture verl Training-Inference Consistency Comparison Data Collection | verl ≥ v0.7.0; comparison data collection that ensures consistent input shape during verl training-inference consistency comparison. | [Asynchronous Architecture verl Training-Inference Consistency Comparison Data Collection](dump/verl_async_consistency_preprocess_dump.md) |
| verl V1 Trainer Training-Inference Consistency Comparison Data Collection | verl 0.9.0.dev; comparison data collection that ensures consistent input shape during verl training-inference consistency comparison. | [verl V1 Trainer Training-Inference Consistency Comparison Data Collection](dump/verl_v1_trainer_consistency_preprocess_dump.md) |
| FSDP Training Backend verl Training-Inference Consistency Comparison Data Collection | verl < v0.7.0, FSDP training backend; comparison data collection that ensures consistent input shape during verl training-inference consistency comparison. | [FSDP Training Backend verl Training-Inference Consistency Comparison Data Collection](dump/verl_fsdp_consistency_preprocess_dump.md) |
| Megatron Training Backend verl Training-Inference Consistency Comparison Data Collection | verl < v0.7.0, Megatron training backend; comparison data collection that ensures consistent input shape during verl training-inference consistency comparison. | [Megatron Training Backend verl Training-Inference Consistency Comparison Data Collection](dump/verl_megatron_consistency_preprocess_dump.md) |
| Precision Comparison | Perform precision comparison on the precision data dumped by the msProbe tool, and then localize precision problems. | [Precision Comparison](accuracy_compare/pytorch_accuracy_compare_instruct.md#verl-training-and-inference-consistency-comparison) |
| Training-Inference Consistency Monitoring: Token-by-Token probs_diff Monitoring | Training-inference consistency monitoring, token-by-token monitoring of probs_diff. | [Training-Inference Consistency Monitoring: Token-by-Token probs_diff Monitoring](dump/verl_token_level_probs_diff_monitoring.md) |
| verl Training-Inference Instrumentation | Instrument at the end positions of the two stages to replace the stage outputs, and judge whether the stage outputs are abnormal through the training effect after replacement. | [verl Training-Inference Cross Staking](verl_cross_validation.md) |

## Slime Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| Data Collection | Complete the msProbe precision data collection operation in the training and inference stages. | [Data Collection](dump/slime_train_rollout_dump_instruct.md) |
| Slime Hyperparameter Comparison | During or after Slime training, compare whether the real hyperparameter configurations collected in the training logs on two different servers have the same values, helping users efficiently compare real hyperparameter value configurations and accelerating the localization of training precision problems caused by configuration differences. | [Slime Hyperparameter Comparison and Key Hyperparameter Verification](slime_param_compare_instruct.md) |
| Training-Inference Consistency Comparison Data Collection | Megatron training backend, SGLang inference engine; the comparison data collection operation that ensures consistent input shape during training-inference consistency comparison. | [Slime Framework Training-Inference Consistency Preprocessing and Data Collection](dump/slime_consistency_preprocess_dump.md) |

## ATB Inference Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| Data Collection | By loading the ATB dump module before the ATB model runs, implement the collection of precision data during the ATB model running. | [Data Collection](dump/atb_data_dump_instruct.md) |
| Precision Comparison | Perform precision comparison on the precision data dumped by ATB, and then localize precision problems. | [Precision Comparison](accuracy_compare/atb_data_compare_instruct.md) |
| Data Conversion | Convert the precision data dumped by ATB into numpy (.npy) or PyTorch Tensor (.pt) format files. | [Data Conversion](dump/data_parse_instruct.md) |

## Offline Model Inference Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| Data Collection | Complete the msProbe precision data collection operation. | [Data Collection](dump/infer_offline_dump_instruct.md) |
| Precision Comparison | Provide a one-click offline model comparison function; you only need to input the model to complete the comparison, without collecting data in advance, and output the result quickly. | [Precision Comparison](accuracy_compare/infer_compare_offline_model_instruct.md) |
| Offline Model Data Precision Comparison | Provide an offline model data comparison function; input the dump data of the offline model for precision comparison. | [Offline Model Data Precision Comparison](accuracy_compare/offline_data_compare_instruct.md) |
| Data Conversion | Convert the dump data of the offline model into numpy (.npy) or PyTorch tensor (.pt) format files. | [Data Conversion](dump/data_parse_instruct.md) |

## MindSpore Training Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| Pre-training Configuration Check | Before training or precision comparison, compare the configuration differences between two environments that may affect training precision. | [Pre-training Configuration Check](config_check_instruct.md) |
| Data Collection | Configure through `config.json` to complete the msProbe precision data collection operation. | [Data Collection](dump/mindspore_data_dump_instruct.md) |
| Precision Pre-check | Scan all APIs of the training model on the Ascend NPU, and provide diagnosis and analysis of the precision situation. | [Precision Pre-check](accuracy_checker/mindspore_accuracy_checker_instruct.md) |
| Hierarchical Visualization Graph Comparison | Parse the precision data dumped by the msProbe tool, restore the model graph structure, and implement precision data comparison at each level of the model. | [Hierarchical Visualization Graph Comparison](accuracy_compare/mindspore_visualization_instruct.md) |
| Precision Comparison | Perform precision comparison on the precision data dumped by the msProbe tool, and then localize precision problems. | [Precision Comparison](accuracy_compare/mindspore_accuracy_compare_instruct.md) |
| Training Status Monitoring | Collect and aggregate the intermediate values of the network layer, optimizer, and communication operators during model training, to help diagnose anomalies in the computation, communication, and optimizer parts during model training. | [Training Status Monitoring](monitor_instruct.md) |
| Checkpoint Comparison | During or after training, compare two different checkpoints and evaluate model similarity. | [Checkpoint Comparison](checkpoint_compare_instruct.md) |
| Trend Visualization | Visualize the trend of the statistical data from msProbe data collection or training status monitoring from the three dimensions of iteration step, node rank, and Tensor target. | [Trend Visualization](accuracy_compare/trend_visualization_instruct.md) |

## MSAdapter Scenario

| Function Item | Function Description | Reference Document |
|---|---|---|
| Data Collection | Configure through `config.json` to complete the msProbe precision data collection operation. | [Data Collection](dump/msadapter_data_dump_instruct.md) |
| checkpoint Comparison | During or after training, compare two different checkpoints and evaluate model similarity. | [Checkpoint Comparison](checkpoint_compare_instruct.md) |

## 📚 Supplementary Materials

🔹 [Precision Data Collection Baseline Report for the PyTorch Scenario](../baseline/pytorch_data_dump_perf_baseline.md)

🔹 [Precision Pre-check Baseline Report for the MindSpore Scenario](../baseline/mindspore_accuracy_checker_perf_baseline.md)

🔹 [Precision Data Collection Baseline Report for the MindSpore Scenario](../baseline/mindspore_data_dump_perf_baseline.md)

🔹 [Standard Performance Baseline Report of the Training Status Monitoring Tool](../baseline/monitor_perf_baseline.md)
