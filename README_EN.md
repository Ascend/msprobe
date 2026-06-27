<h1 align="center">MindStudio Probe</h1>
<div align="center">
  <p>🚀 <b>All-scenario Ascend AI precision debugging tool</b></p>

[![Docs](https://badgen.net/badge/Docs/readthedocs/green)](https://msprobe.readthedocs.io/zh-cn/latest/)
  [![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](https://raw.gitcode.com/Ascend/msprobe/raw/26.0.0/LICENSE) [![Version](https://badgen.net/badge/Version/26.0.0-alpha.1/green)](https://gitcode.com/Ascend/msprobe/releases/26.0.0-alpha.1) [![Ascend](https://img.shields.io/badge/Hardware-Ascend-orange.svg)](https://www.hiascend.com/)
</div>

## 📢 What's New

[2026.03.28]: [Notice of Deprecation: ADump Module in the msProbe Repository](https://gitcode.com/Ascend/msprobe/discussions/2)

[2026.03.20]: Released the [Foundation Model Training Accuracy Debugging Guide](./docs/en/wiki/train_debug_guide.md), [Foundation Model Inference Accuracy Debugging Guide](./docs/en/wiki/infer_debug_guide.md), and [Common Framework Tool Instructions](./docs/en/wiki/dump_enable_guide.md).

[2025.12.31]: Released the open-source version of MindStudio Probe.

## 📌 Overview

MindStudio Probe (msProbe) is a full-scenario precision toolchain for Ascend. It is designed for precision debugging during model development and help you significantly improve the efficiency of locating model precision problems.

## 🔍 Directory Structure

The key directories are as follows. For details, see [Project Directory](./docs/en/dir_structure.md).

```ColdFusion
MindStudio-probe
├── ccsrc                         # C/C++ source code directory
├── cmake                     # CMake files for the C-based components of msProbe
├── docs                         # Documentation directory
├── examples                   # Directory for tool configuration examples
├── output                       # Directory for generated deliverables
├── plugins                    # Entry to plugin code
├── python                      # Python source code directory
├── scripts                      # Directory for storing installation, uninstallation, and upgrade scripts
├── test                         # Test code directory 
├── setup.py                     # E2E packaging and build script
├── README.md                  # Repository description
├── LICENSE                       # License file
```

## 📝Version Description

|  Version  |Supported PyTorch Version|Supported MindSpore Version|Supported Python Version|Supported CANN Version|
|:-----:|:--:|:--:|:--:|:--:|
| 26.0.0 (under development)|2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|≥ CANN 8.3.RC1|
| 26.0.0-alpha.2 |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|≥ CANN 8.3.RC1|
| 26.0.0-alpha.1 |2.1/2.2/2.5/2.6/2.7/2.8|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.11|≥ CANN 8.3.RC1|

## 🛠️ Environment Setup

Install msProbe by referring to [msProbe Installation Guide](docs/en/msprobe_install_guide.md).

## 🚀 Quick Start

An executable sample is provided to describe the precision data collection and comparison functions of msProbe, helping you quickly get started. For details, see [Quick Start of msProbe in the PyTorch Scenario](./docs/en/quick_start/pytorch_quick_start.md) or [Quick Start of msProbe in the MindSpore Scenario](./docs/en/quick_start/mindspore_quick_start.md).

## 📖 Functions

| Scenario           |  Sub-mode/Sub-scenario  | Function         | Description                                                                                         | Reference                                                                                                                                                              |
|-----------------|:-----------:|--------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **vLLM Inference**     |   eager mode  | Data collection        | Collect msProbe precision data.                                                                            | [Data Collection](docs/en/dump/vllm_dump_instruct.md)                                                                                                                         |
|                 |             | Data comparison        | Compare the precision of the data dumped by msProbe to locate precision issues<br>via graph comparison in hierarchical visualization or precision comparison mode.                                     | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/pytorch_visualization_instruct.md)<br>[Precision Comparison](docs/en/accuracy_compare/pytorch_accuracy_compare_instruct.md)                  |
|                 | ACLGraph mode| Data collection        | Collect precision data by using the **acl_save** API.                                                                       | [Data Collection](docs/en/dump/aclgraph_dump_instruct.md)                                                                                                                    |
|                 | TorchAir graph mode| Data collection        | Collect precision data by using the **set_ge_dump_config** API.                                                             | [Data Collection](docs/en/dump/torchair_dump_instruct.md)                                                                                                                    |
|                 |             | Precision comparison        | Compare the precision of the data dumped by msProbe to locate precision issues                                                           | [Precision Comparison](docs/en/accuracy_compare/torchair_compare_instruct.md)                                                                                                     |
| **SGLang inference**   |   eager mode  | Data collection        | Collect msProbe precision data.                                                                            | [Data Collection](docs/en/dump/sglang_eager_dump_instruct.md)                                                                                                                |
|                 |             | Data comparison        | Compare the precision of the data dumped by msProbe to locate precision issues                                                           | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/pytorch_visualization_instruct.md)<br>[Precision Comparison](docs/en/accuracy_compare/pytorch_accuracy_compare_instruct.md)                  |
| **ATB inference**      |      -      | Data collection        | Before running an ATB model, load the ATB dump module to collect the precision data during the running of the ATB model.                                            | [Data Collection](docs/en/dump/atb_data_dump_instruct.md)                                                                                                                    |
|                 |             | Precision comparison        | Compare the precision of the data dumped by ATB to locate precision issues.                                                                | [Precision Comparison](docs/en/accuracy_compare/atb_data_compare_instruct.md)                                                                                                     |
|                 |             | Data conversion        | Convert the precision data dumped by ATB into a NumPy (.npy) or PyTorch tensor (.pt) file.                                         | [Data Conversion](docs/en/dump/data_parse_instruct.md)                                                                                                                       |
| **Offline model inference**     |      -      | Data collection        | Collect msProbe precision data.                                                                            | [Data Collection](docs/en/dump/infer_offline_dump_instruct.md)                                                                                                               |
|                 |             | Precision comparison        | Provide one-click offline model comparison by simplifying inputting a model without data collection in advance and generate results quickly.                                                   | [Precision Comparison](docs/en/accuracy_compare/infer_compare_offline_model_instruct.md)                                                                                          |
|                 |             | Offline model data precision comparison  | Compare the precision of an offline model by inputting the dump data of the offline model.                                                             | [Offline Model Data Precision Comparison](docs/en/accuracy_compare/offline_data_compare_instruct.md)                                                                                          |
|                 |             | Data conversion        | Convert the dump data of an offline model into a NumPy (.npy) or PyTorch tensor (.pt) file.                                           | [Data Conversion](docs/en/dump/data_parse_instruct.md)                                                                                                                       |
| **PyTorch training**  |      -      | Configuration check before training     | Before training or precision comparison, compare the configuration differences that may affect training precision in the two environments.                                                               | [Configuration Check Before Training](docs/en/config_check_instruct.md)                                                                                                                       |
|                 |             | Data collection        | Configure the **config.json** file to collect msProbe precision data.           | [Data Collection](docs/en/dump/pytorch_data_dump_instruct.md)  |
|                 |             | Precision pre-check        | Scan all APIs in a training model running Ascend NPUs and provide diagnostic and analytical insights into precision.                                                            | [Precision Pre-check](docs/en/accuracy_checker/pytorch_accuracy_checker_instruct.md)                                                                                             |
|                 |             | Graph comparison in hierarchical visualization   | Parse the precision data dumped by msProbe to restore the model graph structure and compare the precision data of each model layer.                                              | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/pytorch_visualization_instruct.md)                                                                                           |
|                 |             | Precision comparison        | Compare the precision of the data dumped by msProbe to locate precision issues                                                           | [Precision Comparison](docs/en/accuracy_compare/pytorch_accuracy_compare_instruct.md)                                                                                             |
|                 |             | Training status monitoring      | Collect and aggregate the intermediate values of the network layer, optimizer, and communication operators during model training, helping diagnose exceptions that occur during computing, communication, and optimization.                                 | [Training Status Monitoring](docs/en/monitor_instruct.md)                                                                                                                             |
|                 |             | Checkpoint comparison| During or after training, compare two different checkpoints to evaluate model similarity.                                                          | [Checkpoint Comparison](docs/en/checkpoint_compare_instruct.md)                                                                                                            |
|                 |             | First network overflow/underflow node analysis  | In the multi-rank scenario, find the first node where NaN or INF occurs through data dumping.                                                             | [First Network Overflow/Underflow Node Analysis](docs/en/overflow_check/overflow_check_instruct.md)                                                                                                   |
|                 |             | Trend visualization       | Visualize the data collected by msProbe or the training status monitoring statistics in terms of the number of iterations, rank, and tensor.                                      | [Trend Visualization](docs/en/accuracy_compare/trend_visualization_instruct.md)                                                                                                 |
| **MindSpore training**|      -      | Configuration check before training     | Before training or precision comparison, compare the configuration differences that may affect training precision in the two environments.                                                               | [Configuration Check Before Training](docs/en/config_check_instruct.md)                                                                                                                       |
|                 |             | Data collection        | Configure the **config.json** file to collect msProbe precision data.           | [Data Collection](docs/en/dump/mindspore_data_dump_instruct.md)|
|                 |             | Precision pre-check        | Scan all APIs in a training model running Ascend NPUs and provide diagnostic and analytical insights into precision.                                                            | [Precision Pre-check](docs/en/accuracy_checker/mindspore_accuracy_checker_instruct.md)                                                                                           |
|                 |             | Graph comparison in hierarchical visualization   | Parse the precision data dumped by msProbe to restore the model graph structure and compare the precision data of each model layer.                                              | [Graph Comparison in Hierarchical Visualization](docs/en/accuracy_compare/mindspore_visualization_instruct.md)                                                                                         |
|                 |             | Precision comparison        | Compare the precision of the data dumped by msProbe to locate precision issues                                                           | [Precision Comparison](docs/en/accuracy_compare/mindspore_accuracy_compare_instruct.md)                                                                                           |
|                 |             | Training status monitoring      | Collect and aggregate the intermediate values of the network layer, optimizer, and communication operators during model training, helping diagnose exceptions that occur during computing, communication, and optimization.                                 | [Training Status Monitoring](docs/en/monitor_instruct.md)                                                                                                                             |
|                 |             | Overflow/Underflow detection and parsing     | Overflow/Underflow detection collects precision data from APIs/modules with overflow/underflow issues, while overflow/underflow analysis examines this data to determine whether the phenomenon is normal.<br>It is recommended that the data collection function be triggered to collect statistics and detect overflow/underflow problems.| [Overflow/Underflow Detection and Parsing](docs/en/overflow_check/mindspore_overflow_check_instruct.md)<br>[Data Collection](docs/en/dump/mindspore_data_dump_instruct.md)                                    |
|                 |             | Checkpoint comparison| During or after training, compare two different checkpoints to evaluate model similarity.                                                          | [Checkpoint Comparison](docs/en/checkpoint_compare_instruct.md)                                                                                                            |
|                 |             | Trend visualization       | Visualize the data collected by msProbe or the training status monitoring statistics in terms of the number of iterations, rank, and tensor.                                      | [Trend Visualization](docs/en/accuracy_compare/trend_visualization_instruct.md)                                                                                                 |
| **MSAdapter scenario**|      -      | Data collection        | Configure the **config.json** file to collect msProbe precision data.           | [Data Collection](docs/en/dump/msadapter_data_dump_instruct.md)|
|                 |             | Checkpoint comparison| During or after training, compare two different checkpoints to evaluate model similarity.                                                          | [Checkpoint Comparison](docs/en/checkpoint_compare_instruct.md)                                                                                                            |

## 📚 Supplementary Materials

- [Precision Data Collection Baseline Report in PyTorch](docs/en/baseline/pytorch_data_dump_perf_baseline.md)

- [Precision Pre-check Baseline Report in MindSpore](docs/en/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [Precision Data Collection Baseline Report in MindSpore](docs/en/baseline/mindspore_data_dump_perf_baseline.md)

- [Standard Performance Baseline Report](docs/en/baseline/monitor_perf_baseline.md)

## 💬 FAQs

[FAQs](docs/en/faq.md) summarizes the problems that may occur when you use msProbe.

## 📝 Additional Information

- [Contributing Guide](CONTRIBUTING.md)
- [Security Statement](./docs/en/security_statement.md)
- [Disclaimer](./docs/en/legal/disclaimer.md)
- [License Declaration](./docs/en/legal/license_notice.md)

## 💬 Suggestions and Feedback

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [issues](https://gitcode.com/Ascend/msprobe/issues). We will reply as soon as possible. Thank you for your support.

|                                      📱 Follow the MindStudio WeChat Account                                      | 💬 Communication and Support Channels                                                                                                                                                                                                                                                                                                                                                                                                                    |
|:-----------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://raw.gitcode.com/Ascend/msot/files/master/docs/zh/figures/readme/officialAccount.png" width="120"><br><sub>*Scan the QR code to follow us and get the latest updates.*</sub>| 💡 **Join the WeChat group**:<br>Follow the WeChat account and reply "communication group" to obtain the QR code for joining the group.<br><br>🛠️ ️**Other channels**:<br><br>|

## 🤝 Acknowledgments

msProbe is jointly developed by the following Huawei departments:

- Ascend Computing MindStudio Development Department
- Parallel Distributed Computing Laboratory

Thank you to everyone in the community for your PRs. We warmly welcome contributions to msProbe!
