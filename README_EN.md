<!-- md-trans-meta sourceCommit=d8769dcb657f88517a1877f5ff6691464e86ec45 translatedAt=2026-08-11T02:43:55.875Z pushedAt=2026-08-11T02:53:36.740Z -->

<h1 align="center">MindStudio Probe</h1>

<div align="center">
<p><b><span style="font-size:24px;">A Powerful Full-Scenario Precision Debugging Tool for Ascend AI</span></b></p>

[![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](docs/en/quick_start/pytorch_quick_start.md)
[![Exact Search](https://badgen.net/badge/Exact%20Search/ReadTheDocs/blue)](https://msprobe.readthedocs.io/zh-cn/latest/)
[![AI Q&A (DeepWiki)](https://badgen.net/badge/AI%20Q&A/DeepWiki/blue)](https://deepwiki.com/mindstudio-docs/26.1.0)
[![AI Q&A (ZRead)](https://badgen.net/badge/AI%20Q&A/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
[![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/)
[![Report Issues](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://gitcode.com/Ascend/msprobe/issues)

</div>

English | [简体中文](./README.md)

## ✨ Latest News

<span style="font-size:14px;">

🔹 **[2026.03.28]**: [End-of-Life Notice for the ADump Module in the msProbe Repository](https://gitcode.com/Ascend/msprobe/discussions/2)  
🔹 **[2026.03.20]**: Released *Foundation Model Training Accuracy Debugging*, *Foundation Model Inference Accuracy Debugging*, and *Enabling Tools for Common Frameworks*  
🔹 **[2025.12.31]**: MindStudio Probe Completely Open-Sourced

</span>

## ℹ️ Introduction

MindStudio Probe ( msProbe) is a full-scenario precision debugging toolchain built for Ascend AI processors, specifically designed for the precision debugging phase of model development. It supports mainstream frameworks such as PyTorch and MindSpore, significantly improving the efficiency of locating model precision issues.

## ⚙️ Features

| Usage Scenario | Sub-mode/Sub-scenario | Feature Item | Feature Description | Reference Document |
|---|---|---|---|---|
| **vLLM inference** | Eager/Graph mode | Data collection | Complete the msProbe precision data collection. | [Data Collection](docs/en/user_guide/dump/vllm_dump_instruct.md) |
| | | Data comparison | Perform precision comparison on the precision data dumped by msProbe to locate precision issues.<br/>Refer to hierarchical visual graph comparison or precision comparison modes. | [Graph Comparison in Hierarchical Visualization](docs/en/user_guide/accuracy_compare/pytorch_visualization_instruct.md)<br/>[Precision Comparison](docs/en/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | torchair | Data collection | Collect precision data via the `set_ge_dump_config` API. | [Data Collection](docs/en/user_guide/dump/torchair_dump_instruct.md) |
| | | Precision comparison | Perform precision comparison on the precision data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/user_guide/accuracy_compare/torchair_compare_instruct.md) |
| | General scenario | Inference anomaly detection | Obtain vLLM inference output and detect anomalies. | [Inference Anomaly Detection](docs/en/user_guide/response_anomaly_instruct.md) |
| **SGLang inference** | Eager mode | Data collection | Complete the msProbe precision data collection operation. | [Data Collection](docs/en/user_guide/dump/sglang_eager_dump_instruct_new.md) |
| | | Data comparison | Perform precision comparison on the precision data dumped by msProbe to locate precision issues. | [Graph Comparison in Hierarchical Visualization](docs/en/user_guide/accuracy_compare/pytorch_visualization_instruct.md)<br/>[Precision Comparison](docs/en/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| **ATB inference** | - | Data collection | Collect precision data during ATB model execution by loading the ATB dump module before the ATB model runs. | [Data Collection](docs/en/user_guide/dump/atb_data_dump_instruct.md) |
| | | Precision comparison | Perform precision comparison on the ATB dump precision data to locate precision issues. | [Precision Comparison](docs/en/user_guide/accuracy_compare/atb_data_compare_instruct.md) |
| | | Data conversion | Convert the ATB dump precision data into numpy (`.npy`) or PyTorch tensor (`.pt`) format files. | [Data Conversion](docs/en/user_guide/dump/data_parse_instruct.md) |
| **Offline model inference** | - | Data collection | Complete the msProbe precision data collection operation. | [Data Collection](docs/en/user_guide/dump/infer_offline_dump_instruct.md) |
| | | Precision comparison | Provide a one-click offline model comparison feature; complete the comparison by simply inputting the model, without the need to collect data in advance, and quickly output results. | [Precision Comparison](docs/en/user_guide/accuracy_compare/infer_compare_offline_model_instruct.md) |
| | | Offline model data precision comparison | Provide offline model data comparison; perform precision comparison using the dump data of the offline model | [Offline Model Data Precision Comparison](docs/en/user_guide/accuracy_compare/offline_data_compare_instruct.md) |
| | | Data conversion | Convert the offline model dump data into numpy (`.npy`) or PyTorch tensor (`.pt`) format files. | [Data Conversion](docs/en/user_guide/dump/data_parse_instruct.md) |
| **PyTorch** | Training scenario | Pre-training configuration check | Before training or precision comparison, compare configuration differences between two environments that may affect training precision. | [Pre-training Configuration Check](docs/en/user_guide/config_check_instruct.md) |
| | | Data collection | Complete the msProbe precision data collection operation via `config.json` configuration. | [Data Collection](docs/en/user_guide/dump/pytorch_data_dump_instruct.md) |
| | | Precision pre-check | Scan all APIs in the training model running on Ascend NPUs and provide diagnosis and analysis of the precision status. | [Precision Pre-check](docs/en/user_guide/accuracy_checker/pytorch_accuracy_checker_instruct.md) |
| | | Graph comparison in hierarchical visualization | Parse the precision data dumped by msProbe, reconstruct the model graph structure, and compare precision data at various model levels. | [Graph Comparison in Hierarchical Visualization](docs/en/user_guide/accuracy_compare/pytorch_visualization_instruct.md) |
| | | Precision comparison | Perform precision comparison on the precision data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | | Compilation precision comparison | Perform module-by-module precision comparison between eager mode and compile mode for models with `torch.compile` enabled, to identify forward, backward, or loss discrepancies introduced by compilation | [Compilation Precision Comparison](docs/en/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md) |
| | | Training status monitoring | Collect and aggregate intermediate values of network layers, optimizers, and communication operators during model training to help diagnose anomalies in computation, communication, and optimizer during training. | [Training Status Monitoring](docs/en/user_guide/monitor_instruct.md) |
| | | Checkpoint comparison | During or after training, compare two different checkpoints to evaluate model similarity. | [Checkpoint Comparison](docs/en/user_guide/checkpoint_compare_instruct.md) |
| | | Cross-network analysis of first overflowed node  | Locate the first node where NaN or Inf appears in multi-rank scenarios using dump data. | [Cross-Network Analysis of First Overflowed Node](docs/en/user_guide/overflow_check/overflow_check_instruct.md) |
| | | Trend visualization | Visualize the statistical data from msProbe data collection or training status monitoring across three dimensions: iteration steps, node rank, and tensor target. | [Trend Visualization](docs/en/user_guide/accuracy_compare/trend_visualization_instruct.md) |
| | verl scenario | verl hyperparameter comparison and key hyperparameter verification | During or after verl training, compare the actual hyperparameter configurations collected from training logs on two different servers, or verify whether the configuration matches the key hyperparameter values, assisting users in efficiently comparing actual hyperparameter configurations and accelerating the identification of training precision issues caused by configuration discrepancies. | [verl Hyperparameter Comparison and Key Hyperparameter Verification](docs/en/user_guide/verl_param_compare_or_verify_instruct.md) |
| | | Asynchronous training-inference consistency comparison data collection | For verl ≥ v0.7.0, collect comparison data that ensures consistent input shapes during verl training-inference consistency comparison. | [Collecting Data for Asynchronous verl Training-Inference Consistency Comparison](docs/en/user_guide/dump/verl_async_consistency_preprocess_dump.md) |
| | | verl training-inference consistency comparison data collection based on FSDP | For verl < v0.7.0 with FSDP training backend, collect comparison data that ensures consistent input shapes during verl training-inference consistency comparison. | [Collecting Data for Verifying Data Consistency Between verl Training and Inference Based on FSDP](docs/en/user_guide/dump/verl_fsdp_consistency_preprocess_dump.md) |
| | | verl Training-Inference Consistency Comparison Data Collection based on Megatron | For verl < v0.7.0 with Megatron training backend, collect comparison data that ensures consistent input shapes during verl training-inference consistency comparison | [Collecting Data for Verifying Data Consistency Between verl Training and Inference Based on Megatron](docs/en/user_guide/dump/verl_megatron_consistency_preprocess_dump.md) |
| | | Precision comparison | Perform precision comparison on the precision data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | | Training-inference consistency monitoring: token-Level probs_diff monitoring | Monitor training-inference consistency via `probs_diff` at the token level. | [Training-Inference Consistency Monitoring: Token-Level probs_diff Monitoring](docs/en/user_guide/dump/verl_token_level_probs_diff_monitoring.md) |
| | | verl Training-Inference Cross-Validation | Insert checkpoints at the end of two stages, replace the stage outputs, and determine whether the stage outputs are abnormal based on the training results after replacement | [verl Training-Inference Cross-Validation](docs/en/user_guide/verl_cross_validation.md) |
| **MindSpore training** | - | Pre-training configuration check | Before training or precision comparison, compare configuration differences between two environments that may affect training precision. | [Pre-training Configuration Check](docs/en/user_guide/config_check_instruct.md) |
| | | Data collection | Complete the msProbe precision data collection operation via `config.json` configuration. | [Data Collection](docs/en/user_guide/dump/mindspore_data_dump_instruct.md) |
| | | Precision pre-check | Scan all APIs in the training model running on Ascend NPUs and provide diagnosis and analysis of the precision status. | [Precision Pre-check](docs/en/user_guide/accuracy_checker/mindspore_accuracy_checker_instruct.md) |
| | | Graph comparison in hierarchical visualization | Parse the precision data dumped by msProbe, reconstruct the model graph structure, and compare precision data at various model levels | [Graph comparison in hierarchical visualization](docs/en/user_guide/accuracy_compare/mindspore_visualization_instruct.md) |
| | | Precision comparison | Perform precision comparison on the precision data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/user_guide/accuracy_compare/mindspore_accuracy_compare_instruct.md) |
| | | Training status monitoring | Collect and aggregate intermediate values of network layers, optimizers, and communication operators during model training to help diagnose anomalies in computation, communication, and optimizer during training. | [Training Status Monitoring](docs/en/user_guide/monitor_instruct.md) |
| | | Checkpoint comparison | During or after training, compare two different checkpoints to evaluate model similarity. | [Checkpoint Comparison](docs/en/user_guide/checkpoint_compare_instruct.md) |
| | | Trend visualization | Visualize the statistical data from msProbe data collection or training status monitoring across three dimensions: iteration steps, node rank, and tensor target. | [Trend Visualization](docs/en/user_guide/accuracy_compare/trend_visualization_instruct.md) |
| **MSAdapter scenario** | - | Data collection | Complete the msProbe precision data collection operation via `config.json` configuration. | [Data Collection](docs/en/user_guide/dump/msadapter_data_dump_instruct.md) |
| | | Checkpoint comparison | During or after training, compare two different checkpoints to evaluate model similarity. | [Checkpoint Comparison](docs/en/user_guide/checkpoint_compare_instruct.md) |

## 🚀 Quick Start

For a quick one-stop experience of the complete workflow of data collection, NPU/GPU precision comparison, and graph comparison in hierarchical visualization mode in 10 minutes, see [*msProbe Quick Start*](docs/en/quick_start/pytorch_quick_start.md).

## 📦 Installation Guide

msProbe supports three installation methods: PyPI, WHL, and compilation from source. For details, see [*msProbe Installation Guide*](docs/en/install_guide/msprobe_install_guide.md).

## 📘 Usage Guide

msProbe's features cover various scenarios such as training and inference. Based on your actual usage scenario, select the corresponding feature item as mentioned above, and refer to the relevant documentation for configuration and usage.

## 💡 Typical Cases

🔹 [Foundation Model Training Accuracy Debugging](docs/en/best_practices/train_debug_guide.md)  
🔹 [Foundation Model Inference Accuracy Debugging](docs/en/best_practices/infer_debug_guide.md)  
🔹 [Enabling Tools for Common Frameworks](docs/en/best_practices/dump_enable_guide.md)  

## 📚 Supplementary Materials

🔹 [Precision Data Collection Baseline in PyTorch](docs/en/baseline/pytorch_data_dump_perf_baseline.md)  
🔹 [Precision Pre-check Baseline in MindSpore](docs/en/baseline/mindspore_accuracy_checker_perf_baseline.md)  
🔹 [Precision Data Collection Baseline in MindSpore](docs/en/baseline/mindspore_data_dump_perf_baseline.md)  
🔹 [Performance Baseline Report of the Training Status Monitoring Tool](docs/en/baseline/monitor_perf_baseline.md)  

## ❓ FAQs

For a summary of frequently asked questions and solutions, see *[FAQs](docs/en/support/faq.md)*.

## 🌌 Intelligent Search

To improve document search efficiency, we provide:
🔹 [Exact Search (ReadTheDocs)](https://msprobe.readthedocs.io/zh-cn/latest/): Full-text keyword search, directly accessing information such as APIs, parameters, and error messages.
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/mindstudio-docs/master): Natural language Q&A, quickly grasping the project architecture and module relationships.
🔹 [AI Q&A (ZRead)](https://zread.ai/mindstudio-docs/master): Better Chinese Q&A experience, precisely locating feature usage and details.

## 🛠️ Contribution Guide

Welcome to contribute to this project. Please read *[Contribution Guide](docs/en/contributing/contributing_guide.md)* before you start.

## ⚖️ Related Notes

🔹 *[Release Notes](https://gitcode.com/Ascend/msprobe/releases)*  
🔹 *[Developer Guide](docs/en/development_guide/develop_guide.md)*  
🔹 *[Security Statement](docs/en/legal/SECURITY.md)*  
🔹 *[Disclaimer](docs/en/legal/disclaimer.md)*  
🔹 *[License Notice](docs/en/legal/license_notice.md)*  

## 🤝 Suggestions and Communication

We welcome everyone to contribute to the community. If you have any questions or suggestions, please submit [issues](https://gitcode.com/Ascend/msprobe/issues), and we will respond as soon as possible. Thank you for your support.

| Real-time Interaction (WeChat Group) | Official Updates (Official Account) | More Support (Assistant/Forum) |
|:---:|:---:|:---|
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan the QR code to join the technical discussion group*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan the QR code to follow the official account*</sub> | Scan the QR codes to join the technical discussion group and follow the official account, to connect with the MindStudio user and developer community:<br> **Quick Q&A:** Discuss technical issues in real time with community members<br>**Stay Updated:** Receive version release and feature update notifications firsthand<br> **Share Experience:** Exchange best practices and hands-on insights with fellow developers  <br><br> **More Support Channels**: 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by the following departments of Huawei:
🔹 Ascend Computing MindStudio Development Department
🔹 Parallel Distributed Computing Laboratory

Thank you for every PR from the community. Contributions to msProbe are welcome!
