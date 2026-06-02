<h1 align="center">MindStudio Probe</h1>
<div align="center">
  <p>🚀 <b>昇腾 AI 全场景精度调试利器</b></p>

[![Docs](https://badgen.net/badge/Docs/readthedocs/green)](https://msprobe.readthedocs.io/zh-cn/latest/)
  [![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](https://raw.gitcode.com/Ascend/msprobe/raw/master/LICENSE) [![Version](https://badgen.net/badge/Version/26.0.0-alpha.1/green)](https://gitcode.com/Ascend/msprobe/releases/26.0.0-alpha.1) [![Ascend](https://img.shields.io/badge/Hardware-Ascend-orange.svg)](https://www.hiascend.com/)
</div>

## 📢 最新消息

[2026.03.20]：上线[大模型训练精度定位指南](./zh/wiki/train_debug_guide.md)、[大模型推理精度定位指南](./zh/wiki/infer_debug_guide.md)及[常用框架工具使能指南](./zh/wiki/dump_enable_guide.md)

[2025.12.31]：MindStudio Probe精度调试工具全面开源。

## 📌 简介

MindStudio Probe（MindStudio精度调试工具，msProbe）是针对昇腾提供的全场景精度工具链，专为模型开发的精度调试环节设计，可显著提升用户定位模型精度问题的效率。

## 🔍 目录结构

关键目录如下，详细介绍参见[项目目录](./zh/dir_structure.md)。

```text
MindStudio-probe
├── csrc                         # C/C++源码目录
├── cmake                        # 存放解析C化部分cmake文件
├── docs                         # 文档目录
├── examples                     # 工具配置样例存放目录
├── output                       # 交付件生成目录
├── plugins                      # 插件类代码总入口
├── python                       # Python源码目录
├── scripts                      # 存放安装卸载升级脚本
├── test                         # 测试代码目录
├── setup.py                     # 端到端打包构建脚本
├── README.md                    # 整体仓代码说明
└── LICENSE                      # LICENSE文件
```

## 📝 版本说明

|  版本   |支持PyTorch版本|支持MindSpore版本|支持Python版本|支持CANN版本|
|:-----:|:--:|:--:|:--:|:--:|
| 26.0.0(在研版本) |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|大于等于 CANN 8.3.RC1|
| 26.0.0-alpha.2 |2.1/2.2/2.5/2.6/2.7/2.8/2.9|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.12|大于等于 CANN 8.3.RC1|
| 26.0.0-alpha.1 |2.1/2.2/2.5/2.6/2.7/2.8|2.4.0/2.5.0/2.6.0/2.7.1|3.8-3.11|大于等于 CANN 8.3.RC1|

## 🛠️ 环境部署

安装msProbe工具，具体请参见《[msProbe工具安装指南](./zh/msprobe_install_guide.md)》。

## 🚀 快速入门

msProbe工具快速入门，通过一个可执行样例，完成msProbe工具的精度数据采集和精度比对功能的快速上手。具体请参见《[PyTorch场景精度调试工具快速入门](./zh/quick_start/pytorch_quick_start.md)》或《[MindSpore场景精度调试工具快速入门](./zh/quick_start/mindspore_quick_start.md)》”。

## 📖 功能介绍

| 使用场景            |  子模式/细分场景   | 功能项          | 功能说明                                                                                          | 参考文档                                                                                                                                                               |
|-----------------|:-----------:|--------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **vLLM推理**      |  Eager/图模式   | 数据采集         | 完成msProbe精度数据采集操作                                                  | [数据采集](./zh/dump/vllm_dump_instruct.md)                                                 |
|                 |             | 数据比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题<br/>请参考分级可视化构图比对或精度比对                                      | [分级可视化构图比对](./zh/accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](./zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)                   |
|                 | torchair | 数据采集         | 通过set_ge_dump_config接口完成精度数据采集操作                                                              | [数据采集](./zh/dump/torchair_dump_instruct.md)                                                                                                                     |
|                 |             | 精度比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [精度比对](./zh/accuracy_compare/torchair_compare_instruct.md)                                                                                                      |
| **SGLang推理**    |   eager模式   | 数据采集         | 完成msProbe精度数据采集操作                                                                             | [数据采集](./zh/dump/sglang_eager_dump_instruct_new.md)                                                                                        |
|                 |             | 数据比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [分级可视化构图比对](./zh/accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](./zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)                   |
| **ATB推理**       |      -      | 数据采集         | 通过在ATB模型运行前，加载ATB dump模块的方式，实现对ATB模型运行过程中的精度数据的采集                                             | [数据采集](./zh/dump/atb_data_dump_instruct.md)                                                                                                                     |
|                 |             | 精度比对         | 将ATB dump的精度数据进行精度比对，进而定位精度问题                                                                 | [精度比对](./zh/accuracy_compare/atb_data_compare_instruct.md)                                                                                                      |
|                 |             | 数据转换         | 将ATB dump的精度数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件                                          | [数据转换](./zh/dump/data_parse_instruct.md)                                                                                                                        |
| **离线模型推理**      |      -      | 数据采集         | 完成msProbe精度数据采集操作                                                                             | [数据采集](./zh/dump/infer_offline_dump_instruct.md)                                                                                                                |
|                 |             | 精度比对         | 提供一键式离线模型比对功能，仅需输入模型即可完成比对，无需提前采集数据，快速输出结果                                                    | [精度比对](./zh/accuracy_compare/infer_compare_offline_model_instruct.md)                                                                                           |
|                 |             | 离线模型数据精度比对   | 提供离线模型数据比对功能，输入离线模型的dump数据进行精度比对                                                              | [离线模型数据精度比对](./zh/accuracy_compare/offline_data_compare_instruct.md)                                                                                           |
|                 |             | 数据转换         | 将离线模型的dump数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件                                            | [数据转换](./zh/dump/data_parse_instruct.md)                                                                                                                        |
| **PyTorch训练**   |      -      | 训练前配置检查      | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异                                                                | [训练前配置检查](./zh/config_check_instruct.md)                                                                                                                        |
|                 |             | 数据采集         | 通过config.json配置，完成msProbe精度数据采集操作           | [数据采集](./zh/dump/pytorch_data_dump_instruct.md)   |
|                 |             | 精度预检         | 在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析                                                             | [精度预检](./zh/accuracy_checker/pytorch_accuracy_checker_instruct.md)                                                                                              |
|                 |             | 分级可视化构图比对    | 将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对                                               | [分级可视化构图比对](./zh/accuracy_compare/pytorch_visualization_instruct.md)                                                                                            |
|                 |             | 精度比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [精度比对](./zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)                                                                                              |
|                 |             | 训练状态监测       | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况                                  | [训练状态监测](./zh/monitor_instruct.md)                                                                                                                              |
|                 |             | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度                                                           | [checkpoint比对](./zh/checkpoint_compare_instruct.md)                                                                                                             |
|                 |             | 整网首个溢出节点分析   | 多rank场景下通过dump数据找到首个出现Nan或Inf的节点                                                              | [整网首个溢出节点分析](./zh/overflow_check/overflow_check_instruct.md)                                                                                                    |
|                 |             | 趋势可视化        | 将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化                                       | [趋势可视化](./zh/accuracy_compare/trend_visualization_instruct.md)                                                                                                  |
| **MindSpore训练** |      -      | 训练前配置检查      | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异                                                                | [训练前配置检查](./zh/config_check_instruct.md)                                                                                                                        |
|                 |             | 数据采集         | 通过config.json配置，完成msProbe精度数据采集操作            | [数据采集](./zh/dump/mindspore_data_dump_instruct.md) |
|                 |             | 精度预检         | 在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析                                                             | [精度预检](./zh/accuracy_checker/mindspore_accuracy_checker_instruct.md)                                                                                            |
|                 |             | 分级可视化构图比对    | 将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对                                               | [分级可视化构图比对](./zh/accuracy_compare/mindspore_visualization_instruct.md)                                                                                          |
|                 |             | 精度比对         | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题                                                            | [精度比对](./zh/accuracy_compare/mindspore_accuracy_compare_instruct.md)                                                                                            |
|                 |             | 训练状态监测       | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况                                  | [训练状态监测](./zh/monitor_instruct.md)                                                                                                                              |
|                 |             | 溢出检测与解析      | 溢出检测用于采集溢出API或模块的精度数据，而溢出解析则是通过对溢出数据的分析，进一步判断是否为正常溢出<br/>推荐直接使用数据采集功能采集统计量信息，检测溢出问题，具体请参见数据采集 | [溢出检测与解析](./zh/overflow_check/mindspore_overflow_check_instruct.md)<br>[数据采集](./zh/dump/mindspore_data_dump_instruct.md)                                     |
|                 |             | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度                                                           | [checkpoint比对](./zh/checkpoint_compare_instruct.md)                                                                                                             |
|                 |             | 趋势可视化        | 将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化                                       | [趋势可视化](./zh/accuracy_compare/trend_visualization_instruct.md)                                                                                                  |
| **MSAdapter场景** |      -      | 数据采集         | 通过config.json配置，完成msProbe精度数据采集操作            | [数据采集](./zh/dump/msadapter_data_dump_instruct.md) |
|                 |             | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度                                                           | [checkpoint比对](./zh/checkpoint_compare_instruct.md)                                                                                                             |

## 📚 补充材料

- [PyTorch场景的精度数据采集基线报告](./zh/baseline/pytorch_data_dump_perf_baseline.md)

- [MindSpore场景的精度预检基线报告](./zh/baseline/mindspore_accuracy_checker_perf_baseline.md)

- [MindSpore场景的精度数据采集基线报告](./zh/baseline/mindspore_data_dump_perf_baseline.md)

- [训练状态监测工具标准性能基线报告](./zh/baseline/monitor_perf_baseline.md)

## 💬 FAQ

FAQ汇总了在使用msProbe工具过程中可能遇到的问题，具体请参见[FAQ](./zh/faq.md)。

## 📝 相关说明

- 《[开发者指南](./zh/developer_guide/development_guide.md)》
- 《[安全声明](./zh/security_statement.md)》
- 《[免责声明](./zh/legal/disclaimer.md)》
- 《[License声明](./zh/legal/license_notice.md)》

## 💬 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[Issues](https://gitcode.com/Ascend/msprobe/issues)，我们会尽快回复。感谢您的支持。

- 联系我们

| 💬 技术交流群 | 📢 官方公众号 | 🤝 更多加入渠道 |
| :---: | :---: | :--- |
| <img src="https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/zh/figures/readme/officialGroupChat.jpg" width="120"><br><sub>*扫码直接加入技术交流群*</sub> | <img src="https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/zh/figures/readme/officialAccount.jpg" width="120"><br><sub>*扫码关注获取最新动态*</sub> |欢迎扫码关注技术交流群跟官方公众号。这里是 MindStudio 用户与开发者最快捷的交流阵地：<br> **快速提问：** 与社区小伙伴即时探讨技术问题<br>**掌握动态：** 第一时间获取版本发布与功能更新通知<br> **经验共享：** 与其他开发者交流最佳实践  <br>🛠️ **其他渠道**：<br>👉 昇腾助手：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png)<br>👉 昇腾论坛：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🤝 致谢

msProbe由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 分布式并行计算实验室

感谢来自社区的每一个PR，欢迎贡献msProbe！

```{toctree}
:maxdepth: 2
:caption: 开始使用
:hidden:
简介 <zh/overview>
安装指南 <zh/msprobe_install_guide>
快速入门 <zh/quick_start/pytorch_quick_start>
常见问题 <zh/faq>
安全声明 <zh/security_statement>
```

```{toctree}
:maxdepth: 2
:caption: 功能指南-训练场景
:hidden:

训练前配置检查 <zh/config_check_instruct>
数据采集 <zh/dump/pytorch_data_dump_instruct>
分级可视化构图比对 <zh/accuracy_compare/pytorch_visualization_instruct>
精度比对 <zh/accuracy_compare/pytorch_accuracy_compare_instruct>
训练状态监测 <zh/monitor_instruct.md>
精度预检 <zh/accuracy_checker/pytorch_accuracy_checker_instruct>
```

```{toctree}
:maxdepth: 2
:caption: 功能指南-推理场景
:hidden:

# vLLM推理
vLLM 数据采集（Eager/图模式） <zh/dump/vllm_dump_instruct>
vLLM torchair数据采集 <zh/dump/torchair_dump_instruct>

# SGLang推理
SGLang eager模式数据采集 <zh/dump/sglang_eager_dump_instruct>

# ATB推理
ATB数据采集 <zh/dump/atb_data_dump_instruct>
ATB精度比对 <zh/accuracy_compare/atb_data_compare_instruct>
ATB和离线模型数据转换 <zh/dump/data_parse_instruct>

# 离线模型推理
离线模型数据采集 <zh/dump/infer_offline_dump_instruct>
离线模型比对 <zh/accuracy_compare/infer_compare_offline_model_instruct>
离线模型数据比对 <zh/accuracy_compare/offlline_data_compare_instruct>
```

```{toctree}
:maxdepth: 2
:caption: 定位指南
:hidden:

大模型训练精度定位指南 <zh/wiki/train_debug_guide>
大模型推理精度定位指南 <zh/wiki/infer_debug_guide>
常见框架dump工具使能 <zh/wiki/dump_enable_guide>
```
