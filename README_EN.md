<h1 align="center">MindStudio Probe</h1>

<div align="center">
<p><b><span style="font-size:24px;">All-scenario Ascend AI Precision Debugging Tool</span></b></p>

[![QuickStart](https://badgen.net/badge/QuickStart/QuickStart/blue)](docs/en/quick_start/pytorch_quick_start.md)
[![Full-text Search](https://badgen.net/badge/Full-text%20Search/ReadTheDocs/blue)](https://msprobe.readthedocs.io/en/latest/)
[![AI Q&A(DeepWiki)](https://badgen.net/badge/AI%20Q&A/DeepWiki/blue)](https://deepwiki.com/mindstudio-docs/master)
[![AI Q&A(ZRead)](https://badgen.net/badge/AI%20Q&A/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
[![Ascend Community](https://badgen.net/badge/Ascend/Community/blue)](https://www.hiascend.com/en/)
[![Report Issues](https://badgen.net/badge/Report/Issues/blue)](https://gitcode.com/Ascend/msprobe/issues)

</div>

## ✨ What's New

<span style="font-size:14px;">

🔹 **[2026.03.28]**: [Notice of Deprecation: ADump Module in the msProbe Repository](https://gitcode.com/Ascend/msprobe/discussions/2)  
🔹 **[2026.03.20]**: Released the [Foundation Model Training Accuracy Debugging Guide](docs/en/wiki/train_debug_guide.md), [Foundation Model Inference Accuracy Debugging Guide](docs/en/wiki/infer_debug_guide.md), and [Common Framework Tool Instructions](docs/en/wiki/dump_enable_guide.md)  
🔹 **[2025.12.31]**: Released the open-source version of MindStudio Probe

</span>

## ℹ️ Overview

MindStudio Probe (msProbe) is a full-scenario precision debugging toolchain for Ascend AI processors. Designed for precision debugging during model development, it supports mainstream frameworks such as PyTorch and MindSpore, helping you significantly improve the efficiency of locating model precision problems.

## ⚙️ Features

| Scenario | Sub-mode/Sub-scenario | Function | Description | Reference |
|---|---|---|---|---|
| **vLLM Inference** | eager/graph mode | Data collection | Collect msProbe precision data. | [Data Collection](docs/en/dump/vllm_dump_instruct.md) |
| | | Data comparison | Compare the precision of the data dumped by msProbe to locate precision issues. | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/pytorch_visualization_instruct.md)<br>[Precision Comparison](docs/en/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | TorchAir graph mode | Data collection | Collect precision data by using the **set_ge_dump_config** API. | [Data Collection](docs/en/dump/torchair_dump_instruct.md) |
| | | Precision comparison | Compare the precision of the data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/accuracy_compare/torchair_compare_instruct.md) |
| **SGLang inference** | eager mode | Data collection | Collect msProbe precision data. | [Data Collection](docs/en/dump/sglang_eager_dump_instruct.md) |
| | | Data comparison | Compare the precision of the data dumped by msProbe to locate precision issues. | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/pytorch_visualization_instruct.md)<br>[Precision Comparison](docs/en/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| **ATB inference** | - | Data collection | Before running an ATB model, load the ATB dump module to collect the precision data during the running of the ATB model. | [Data Collection](docs/en/dump/atb_data_dump_instruct.md) |
| | | Precision comparison | Compare the precision of the data dumped by ATB to locate precision issues. | [Precision Comparison](docs/en/accuracy_compare/atb_data_compare_instruct.md) |
| | | Data conversion | Convert the precision data dumped by ATB into a NumPy (.npy) or PyTorch tensor (.pt) file. | [Data Conversion](docs/en/dump/data_parse_instruct.md) |
| **Offline model inference** | - | Data collection | Collect msProbe precision data. | [Data Collection](docs/en/dump/infer_offline_dump_instruct.md) |
| | | Precision comparison | Provide one-click offline model comparison by simply inputting a model without data collection in advance and generate results quickly. | [Precision Comparison](docs/en/accuracy_compare/infer_compare_offline_model_instruct.md) |
| | | Offline model data precision comparison | Compare the precision of an offline model by inputting the dump data of the offline model. | [Offline Model Data Precision Comparison](docs/en/accuracy_compare/offline_data_compare_instruct.md) |
| | | Data conversion | Convert the dump data of an offline model into a NumPy (.npy) or PyTorch tensor (.pt) file. | [Data Conversion](docs/en/dump/data_parse_instruct.md) |
| **PyTorch training** | - | Configuration check before training | Before training or precision comparison, compare the configuration differences that may affect training precision in the two environments. | [Configuration Check Before Training](docs/en/config_check_instruct.md) |
| | | Data collection | Configure the **config.json** file to collect msProbe precision data. | [Data Collection](docs/en/dump/pytorch_data_dump_instruct.md) |
| | | Precision pre-check | Scan all APIs in a training model running on Ascend NPUs and provide diagnostic and analytical insights into precision. | [Precision Pre-check](docs/en/accuracy_checker/pytorch_accuracy_checker_instruct.md) |
| | | Graph comparison in hierarchical visualization | Parse the precision data dumped by msProbe to restore the model graph structure and compare the precision data of each model layer. | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/pytorch_visualization_instruct.md) |
| | | Precision comparison | Compare the precision of the data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | | Training status monitoring | Collect and aggregate the intermediate values of the network layer, optimizer, and communication operators during model training, helping diagnose exceptions that occur during computing, communication, and optimization. | [Training Status Monitoring](docs/en/monitor_instruct.md) |
| | | Checkpoint comparison | During or after training, compare two different checkpoints to evaluate model similarity. | [Checkpoint Comparison](docs/en/checkpoint_compare_instruct.md) |
| | | First network overflow/underflow node analysis | In the multi-rank scenario, find the first node where NaN or INF occurs through data dumping. | [First Network Overflow/Underflow Node Analysis](docs/en/overflow_check/overflow_check_instruct.md) |
| | | Trend visualization | Visualize the data collected by msProbe or the training status monitoring statistics in terms of the number of iterations, rank, and tensor. | [Trend Visualization](docs/en/accuracy_compare/trend_visualization_instruct.md) |
| **MindSpore training** | - | Configuration check before training | Before training or precision comparison, compare the configuration differences that may affect training precision in the two environments. | [Configuration Check Before Training](docs/en/config_check_instruct.md) |
| | | Data collection | Configure the **config.json** file to collect msProbe precision data. | [Data Collection](docs/en/dump/mindspore_data_dump_instruct.md) |
| | | Precision pre-check | Scan all APIs in a training model running on Ascend NPUs and provide diagnostic and analytical insights into precision. | [Precision Pre-check](docs/en/accuracy_checker/mindspore_accuracy_checker_instruct.md) |
| | | Graph comparison in hierarchical visualization | Parse the precision data dumped by msProbe to restore the model graph structure and compare the precision data of each model layer. | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/mindspore_visualization_instruct.md) |
| | | Precision comparison | Compare the precision of the data dumped by msProbe to locate precision issues. | [Precision Comparison](docs/en/accuracy_compare/mindspore_accuracy_compare_instruct.md) |
| | | Training status monitoring | Collect and aggregate the intermediate values of the network layer, optimizer, and communication operators during model training, helping diagnose exceptions that occur during computing, communication, and optimization. | [Training Status Monitoring](docs/en/monitor_instruct.md) |
| | | Overflow/Underflow detection and parsing | Overflow/Underflow detection collects precision data from APIs/modules with overflow/underflow issues, while overflow/underflow analysis examines this data to determine whether the phenomenon is normal. | [Overflow/Underflow Detection and Parsing](docs/en/overflow_check/mindspore_overflow_check_instruct.md)<br>[Data Collection](docs/en/dump/mindspore_data_dump_instruct.md) |
| | | Checkpoint comparison | During or after training, compare two different checkpoints to evaluate model similarity. | [Checkpoint Comparison](docs/en/checkpoint_compare_instruct.md) |
| | | Trend visualization | Visualize the data collected by msProbe or the training status monitoring statistics in terms of the number of iterations, rank, and tensor. | [Trend Visualization](docs/en/accuracy_compare/trend_visualization_instruct.md) |
| **MSAdapter scenario** | - | Data collection | Configure the **config.json** file to collect msProbe precision data. | [Data Collection](docs/en/dump/msadapter_data_dump_instruct.md) |
| | | Checkpoint comparison | During or after training, compare two different checkpoints to evaluate model similarity. | [Checkpoint Comparison](docs/en/checkpoint_compare_instruct.md) |

## 🚀 Quick Start

An executable sample is provided to help you quickly get started with precision data collection and comparison. For details, see [Quick Start of msProbe in the PyTorch Scenario](docs/en/quick_start/pytorch_quick_start.md) or [Quick Start of msProbe in the MindSpore Scenario](docs/en/quick_start/mindspore_quick_start.md).

## 📦 Installation Guide

msProbe supports PyPI installation, WHL installation, and source code compilation. For details, see the [msProbe Installation Guide](docs/en/msprobe_install_guide.md).

## 📘 User Guide

msProbe supports various scenarios including training and inference. Select your scenario in the Functions section above, choose the corresponding feature, and refer to the linked documentation for detailed configuration.

## 💡 Best Practices

🔹 [Foundation Model Training Accuracy Debugging Guide](docs/en/wiki/train_debug_guide.md)  
🔹 [Foundation Model Inference Accuracy Debugging Guide](docs/en/wiki/infer_debug_guide.md)  
🔹 [Common Framework Tool Instructions](docs/en/wiki/dump_enable_guide.md)  

## 📚 Supplementary Materials

🔹 [Precision Data Collection Baseline Report in PyTorch](docs/en/baseline/pytorch_data_dump_perf_baseline.md)  
🔹 [Precision Pre-check Baseline Report in MindSpore](docs/en/baseline/mindspore_accuracy_checker_perf_baseline.md)  
🔹 [Precision Data Collection Baseline Report in MindSpore](docs/en/baseline/mindspore_data_dump_perf_baseline.md)  
🔹 [Standard Performance Baseline Report](docs/en/baseline/monitor_perf_baseline.md)  

## ❓ FAQ

For frequently asked questions and solutions, see the [FAQ](docs/en/faq.md).

## 🌌 Smart Search

To improve documentation efficiency, we provide multiple search options:  
🔹 [Full-text Search (ReadTheDocs)](https://msprobe.readthedocs.io/en/latest/): Keyword-based full-text search for interfaces, parameters, and error messages.  
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/mindstudio-docs/master): Natural language Q&A for a quick understanding of project architecture and module relationships.  
🔹 [AI Q&A (ZRead)](https://zread.ai/mindstudio-docs/master): Chinese Q&A with better user experience, pinpointing feature usage and details.  

## 🛠️ Contributing

We welcome contributions. See the [Contributing Guide](CONTRIBUTING.md).

## ⚖️ Additional Information

🔹 [Developer Guide](docs/en/developer_guide/development_guide.md)  
🔹 [Security Statement](docs/en/security_statement.md)  
🔹 [Disclaimer](docs/en/legal/disclaimer.md)  
🔹 [License Declaration](docs/en/legal/license_notice.md)  

## 🤝 Feedback and Support

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [Issues](https://gitcode.com/Ascend/msprobe/issues). We will reply as soon as possible. Thank you for your support.

| Instant Interaction (WeChat Group) | Official News (WeChat Account) | In-depth Support (Assistant/Forum) |
|:---:|:---:|:---|
| <img src="https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/zh/figures/readme/officialGroupChat.jpg" width="120"><br><sub>*Scan the QR code to join the group*</sub> | <img src="https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/zh/figures/readme/officialAccount.jpg" width="120"><br><sub>*Follow the official account*</sub> | Scan the QR codes above to join the WeChat group and follow the official account — the fastest way to connect with MindStudio users and developers:<br> **Quick Q&A:** Discuss technical issues with community members<br>**Stay Updated:** Receive release notes and feature update notifications<br> **Share Experience:** Exchange best practices with developers  <br><br> **More channels:** 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

msProbe is jointly developed by the following Huawei departments:  
🔹 Ascend Computing MindStudio Development Department  
🔹 Parallel Distributed Computing Laboratory  

Thank you to everyone in the community for your PRs. We warmly welcome contributions to msProbe!
