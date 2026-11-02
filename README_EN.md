<h1 align="center">MindStudio Probe</h1>

<div align="center">
<p><b><span style="font-size:24px;">A Powerful Full-Scenario Precision Debugging Tool for Ascend AI</span></b></p>

[![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](docs/en/quick_start/pytorch_quick_start.md)
[![Exact Search](https://badgen.net/badge/Exact%20Search/ReadTheDocs/blue)](https://msprobe.readthedocs.io/zh-cn/latest/)
[![AI Q&A (DeepWiki)](https://badgen.net/badge/AI%20Q&A/DeepWiki/blue)](https://deepwiki.com/mindstudio-docs/master)
[![AI Q&A (ZRead)](https://badgen.net/badge/AI%20Q&A/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
[![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/)
[![Report Issues](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://gitcode.com/Ascend/msprobe/issues)

</div>

English | [简体中文](./README.md)

## ✨ Latest News

<span style="font-size:14px;">

🔹 **[2026.03.28]**: [End-of-Life Notice for the ADump Module in the msProbe Repository](https://gitcode.com/Ascend/msprobe/discussions/2)<br>
🔹 **[2026.03.20]**: Released [Foundation Model Training Accuracy Debugging Guide](docs/en/best_practices/train_debug_guide.md), [Foundation Model Inference Accuracy Debugging Guide](docs/en/best_practices/infer_debug_guide.md), and [Enabling Tools for Common Frameworks](docs/en/best_practices/dump_enable_guide.md)<br>
🔹 **[2025.12.31]**: MindStudio Probe Precision Debugging Tool Fully Open-Sourced

</span>

## ℹ️ Introduction

MindStudio Probe (msProbe) is a full-scenario precision debugging toolchain built for Ascend AI processors, specifically designed for the precision debugging phase of model development. It supports mainstream frameworks such as PyTorch and MindSpore, significantly improving the efficiency of locating model precision issues.

## ⚙️ Features

msProbe provides three core capabilities around model precision debugging, covering the complete workflow of "data collection → precision comparison → visualization":

| Core Capability | Description | Quick Entry |
|---|---|---|
| Data Collection | Collect precision data during model training/inference with one click via `config.json` configuration. | [PyTorch Data Collection](docs/en/user_guide/dump/pytorch_data_dump_instruct.md) |
| Precision Comparison | Compare the collected precision data to locate the network layers/operators with precision differences. | [PyTorch Precision Comparison](docs/en/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| Visualization | Parse the dump data to reconstruct the model graph structure, and locate precision differences through hierarchical visualization. | [Graph Comparison in Hierarchical Visualization](docs/en/user_guide/accuracy_compare/pytorch_visualization_instruct.md) |

> The table above lists the most commonly used PyTorch scenario entries. For more frameworks and usage scenarios, see the [msProbe Usage Guide](docs/en/user_guide/README.md).

## 🚀 Quick Start

For a 10-minute one-stop experience of the complete workflow of data collection, NPU/GPU precision comparison, and graph comparison in hierarchical visualization mode, see [msProbe Quick Start](docs/en/quick_start/pytorch_quick_start.md).

## 📦 Installation Guide

msProbe supports three installation methods: PyPI, WHL, and compilation from source. For details, see [msProbe Installation Guide](docs/en/install_guide/msprobe_install_guide.md).

## 📘 Usage Guide

For detailed usage of the tool, see [msProbe Usage Guide](docs/en/user_guide/README.md).

## 💡 Typical Cases

To help users understand and master the tool through typical problem scenarios, see [msProbe Typical Cases](./docs/en/best_practices/README.md).

## ❓ FAQs

For a summary of frequently asked questions and solutions, see [FAQs](docs/en/support/faq.md).

## 🌌 Intelligent Search

To improve document search efficiency, we provide multiple efficient search methods:<br>
🔹 [Exact Search (ReadTheDocs)](https://msprobe.readthedocs.io/zh-cn/latest/): Full-text keyword search, directly accessing information such as APIs, parameters, and error messages.<br>
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/mindstudio-docs/master): Natural language Q&A, quickly grasping the project architecture and module relationships.<br>
🔹 [AI Q&A (ZRead)](https://zread.ai/mindstudio-docs/master): Better Chinese Q&A experience, precisely locating feature usage and details.

## 🛠️ Contribution Guide

Welcome to contribute to this project, see [Contribution Guide](docs/en/contributing/contributing_guide.md).

## ⚖️ Related Notes

🔹 [Release Notes](https://gitcode.com/Ascend/msprobe/releases)<br>
🔹 [Developer Guide](docs/en/development_guide/develop_guide.md)<br>
🔹 [Security Statement](docs/en/legal/SECURITY.md)<br>
🔹 [Disclaimer](docs/en/legal/disclaimer.md)<br>
🔹 [License Notice](docs/en/legal/license_notice.md)

## 🤝 Suggestions and Communication

We welcome everyone to contribute to the community. If you have any questions or suggestions, please submit [Issues](https://gitcode.com/Ascend/msprobe/issues), and we will respond as soon as possible. Thank you for your support.

🔹 [msProbe SIG Meeting](https://meeting.ascend.osinfra.cn/?sig=sig-sig-mstt): Regular community meetings, welcome to participate and discuss.

| Real-time Interaction (WeChat Group) | Official Updates (Official Account) | More Support (Assistant/Forum) |
|:---:|:---:|:---|
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan the QR code to join the technical discussion group*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan the QR code to follow the official account*</sub> | Scan the QR codes to join the technical discussion group and follow the official account, to connect with the MindStudio user and developer community:<br> **Quick Q&A:** Discuss technical issues in real time with community members<br>**Stay Updated:** Receive version release and feature update notifications firsthand<br> **Share Experience:** Exchange best practices and hands-on insights with fellow developers  <br> <br> **More Support Channels**: 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by Huawei Technologies Co., Ltd. and developers.

Thank you for every PR from the community. Contributions are welcome!
