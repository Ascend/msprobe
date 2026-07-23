<h1 align="center">MindStudio Probe</h1>

<div align="center">
<p><b><span style="font-size:24px;">昇腾 AI 全场景精度调试利器</span></b></p>

[![快速入门](https://badgen.net/badge/快速入门/QuickStart/blue)](./zh/quick_start/pytorch_quick_start.md)
[![精确搜索](https://badgen.net/badge/精确搜索/ReadTheDocs/blue)](https://msprobe.readthedocs.io/zh-cn/latest/)
[![AI问答(DeepWiki)](https://badgen.net/badge/AI问答/DeepWiki/blue)](https://deepwiki.com/mindstudio-docs/master)
[![AI问答(ZRead)](https://badgen.net/badge/AI问答/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
[![昇腾社区](https://badgen.net/badge/昇腾社区/Community/blue)](https://www.hiascend.com/)
[![报告问题](https://badgen.net/badge/报告问题/Issues/blue)](https://gitcode.com/Ascend/msprobe/issues)

</div>

## ✨ 最新消息

<span style="font-size:14px;">

🔹 **[2026.03.28]**：[msprobe 仓库 ADump 模块日落下线通知](https://gitcode.com/Ascend/msprobe/discussions/2)  
🔹 **[2026.03.20]**：上线[大模型训练精度定位指南](./zh/best_practices/train_debug_guide.md)、[大模型推理精度定位指南](./zh/best_practices/infer_debug_guide.md)及[常用框架工具使能指南](./zh/best_practices/dump_enable_guide.md)  
🔹 **[2025.12.31]**：MindStudio Probe 精度调试工具全面开源

</span>

## ℹ️ 简介

MindStudio Probe（MindStudio 精度调试工具，msProbe）是针对昇腾 AI 处理器打造的全场景精度调试工具链，专为模型开发的精度调试环节设计，支持 PyTorch、MindSpore 等主流框架，可显著提升用户定位模型精度问题的效率。

## ⚙️ 功能介绍

| 使用场景 | 子模式/细分场景 | 功能项 | 功能说明 | 参考文档 |
|---|---|---|---|---|
| **vLLM推理** | Eager/图模式 | 数据采集 | 完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/vllm_dump_instruct.md) |
| | | 数据比对 | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题<br/>请参考分级可视化构图比对或精度比对 | [分级可视化构图比对](./zh/user_guide/accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](./zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | torchair | 数据采集 | 通过set_ge_dump_config接口完成精度数据采集操作 | [数据采集](./zh/user_guide/dump/torchair_dump_instruct.md) |
| | | 精度比对 | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题 | [精度比对](./zh/user_guide/accuracy_compare/torchair_compare_instruct.md) |
| | 通用场景 | 推理异常检测 | 获取vLLM推理输出，感知异常问题 | [推理异常检测](./zh/user_guide/response_anomaly_instruct.md) |
| **SGLang推理** | eager模式 | 数据采集 | 完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/sglang_eager_dump_instruct_new.md) |
| | | 数据比对 | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题 | [分级可视化构图比对](./zh/user_guide/accuracy_compare/pytorch_visualization_instruct.md)<br>[精度比对](./zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| **ATB推理** | - | 数据采集 | 通过在ATB模型运行前，加载ATB dump模块的方式，实现对ATB模型运行过程中的精度数据的采集 | [数据采集](./zh/user_guide/dump/atb_data_dump_instruct.md) |
| | | 精度比对 | 将ATB dump的精度数据进行精度比对，进而定位精度问题 | [精度比对](./zh/user_guide/accuracy_compare/atb_data_compare_instruct.md) |
| | | 数据转换 | 将ATB dump的精度数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件 | [数据转换](./zh/user_guide/dump/data_parse_instruct.md) |
| **离线模型推理** | - | 数据采集 | 完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/infer_offline_dump_instruct.md) |
| | | 精度比对 | 提供一键式离线模型比对功能，仅需输入模型即可完成比对，无需提前采集数据，快速输出结果 | [精度比对](./zh/user_guide/accuracy_compare/infer_compare_offline_model_instruct.md) |
| | | 离线模型数据精度比对 | 提供离线模型数据比对功能，输入离线模型的dump数据进行精度比对 | [离线模型数据精度比对](./zh/user_guide/accuracy_compare/offline_data_compare_instruct.md) |
| | | 数据转换 | 将离线模型的dump数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件 | [数据转换](./zh/user_guide/dump/data_parse_instruct.md) |
| **PyTorch训练** | - | 训练前配置检查 | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异 | [训练前配置检查](./zh/user_guide/config_check_instruct.md) |
| | | verl超参比对与关键超参校验 | verl训练过程中或结束后，比对两台不同服务器上训练日志中采集到的真实超参配置，或者校验配置是否与关键超参取值相同，辅助用户高效比对真实超参值配置，加速定位因配置差异所引发的训练精度问题 | [verl超参比对与关键超参校验](./zh/user_guide/verl_param_compare_or_verify_instruct.md) |
| | | 数据采集 | 通过config.json配置，完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/pytorch_data_dump_instruct.md) |
| | | 精度预检 | 在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析 | [精度预检](./zh/user_guide/accuracy_checker/pytorch_accuracy_checker_instruct.md) |
| | | 分级可视化构图比对 | 将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对 | [分级可视化构图比对](./zh/user_guide/accuracy_compare/pytorch_visualization_instruct.md) |
| | | 精度比对 | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题 | [精度比对](./zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md) |
| | | 编译精度比对 | 对启用torch.compile的模型进行eager与compile逐模块精度对比，定位编译引入的前向、反向或loss差异 | [编译精度比对](./zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md) |
| | | 训练状态监测 | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况 | [训练状态监测](./zh/user_guide/monitor_instruct.md) |
| | | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度 | [checkpoint比对](./zh/user_guide/checkpoint_compare_instruct.md) |
| | | 整网首个溢出节点分析 | 多rank场景下通过dump数据找到首个出现Nan或Inf的节点 | [整网首个溢出节点分析](./zh/user_guide/overflow_check/overflow_check_instruct.md) |
| | | 趋势可视化 | 将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化 | [趋势可视化](./zh/user_guide/accuracy_compare/trend_visualization_instruct.md) |
| **MindSpore训练** | - | 训练前配置检查 | 训练前或精度比对前，对比两个环境下可能影响训练精度的配置差异 | [训练前配置检查](./zh/user_guide/config_check_instruct.md) |
| | | 数据采集 | 通过config.json配置，完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/mindspore_data_dump_instruct.md) |
| | | 精度预检 | 在昇腾NPU上扫描训练模型中的所有API，给出精度情况的诊断和分析 | [精度预检](./zh/user_guide/accuracy_checker/mindspore_accuracy_checker_instruct.md) |
| | | 分级可视化构图比对 | 将msProbe工具dump的精度数据进行解析，还原模型图结构，实现模型各个层级的精度数据比对 | [分级可视化构图比对](./zh/user_guide/accuracy_compare/mindspore_visualization_instruct.md) |
| | | 精度比对 | 将msProbe工具dump的精度数据进行精度比对，进而定位精度问题 | [精度比对](./zh/user_guide/accuracy_compare/mindspore_accuracy_compare_instruct.md) |
| | | 训练状态监测 | 收集和聚合模型训练过程中的网络层，优化器，通信算子的中间值，帮助诊断模型训练过程中计算，通信，优化器各部分出现的异常情况 | [训练状态监测](./zh/user_guide/monitor_instruct.md) |
| | | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度 | [checkpoint比对](./zh/user_guide/checkpoint_compare_instruct.md) |
| | | 趋势可视化 | 将msProbe工具数据采集或训练状态监测的统计量数据从迭代步数、节点rank和张量目标三个维度进行趋势可视化 | [趋势可视化](./zh/user_guide/accuracy_compare/trend_visualization_instruct.md) |
| **MSAdapter场景** | - | 数据采集 | 通过config.json配置，完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/msadapter_data_dump_instruct.md) |
| | | checkpoint比对 | 训练过程中或结束后，比较两个不同的checkpoint，评估模型相似度 | [checkpoint比对](./zh/user_guide/checkpoint_compare_instruct.md) |
| **Slime场景** | - | 数据采集 | 训推阶段完成msProbe精度数据采集操作 | [数据采集](./zh/user_guide/dump/slime_train_rollout_dump_instruct.md) |

## 🚀 快速入门

通过一个可执行样例，快速上手精度数据采集和精度比对功能，请参见《[PyTorch 场景精度调试工具快速入门](./zh/quick_start/pytorch_quick_start.md)》或《[MindSpore 场景精度调试工具快速入门](./zh/quick_start/mindspore_quick_start.md)》。

## 📦 安装指南

支持 PyPI 安装、WHL 安装、源码编译三种方式，具体请参见《[msProbe 安装指南](./zh/install_guide/msprobe_install_guide.md)》。

## 📘 使用指南

msProbe 的功能覆盖训练和推理等多种场景。请根据您的实际使用场景，在上方功能介绍中选择对应的功能项，并参考相应文档进行配置和使用。

## 💡 典型案例

🔹 [大模型训练精度定位指南](./zh/best_practices/train_debug_guide.md)  
🔹 [大模型推理精度定位指南](./zh/best_practices/infer_debug_guide.md)  
🔹 [常用框架工具使能指南](./zh/best_practices/dump_enable_guide.md)  

## 📚 补充材料

🔹 [PyTorch 场景的精度数据采集基线报告](./zh/baseline/pytorch_data_dump_perf_baseline.md)  
🔹 [MindSpore 场景的精度预检基线报告](./zh/baseline/mindspore_accuracy_checker_perf_baseline.md)  
🔹 [MindSpore 场景的精度数据采集基线报告](./zh/baseline/mindspore_data_dump_perf_baseline.md)  
🔹 [训练状态监测工具标准性能基线报告](./zh/baseline/monitor_perf_baseline.md)  

## ❓ FAQ

常见问题及解决方案汇总，请参见《[FAQ](./zh/support/faq.md)》。

## 🌌 智能检索

为提升文档查阅效率，我们提供多种高效检索方式：  
🔹 [精确搜索（ReadTheDocs）](https://msprobe.readthedocs.io/zh-cn/latest/)：关键词全文检索，直达接口、参数与报错等信息。  
🔹 [AI 问答（DeepWiki）](https://deepwiki.com/mindstudio-docs/master)：自然语言问答，快速把握项目架构与模块关系。  
🔹 [AI 问答（ZRead）](https://zread.ai/mindstudio-docs/master)：中文问答体验更优，精准定位功能用法与细节。  

## 🛠️ 贡献指南

欢迎参与项目贡献，请参见《[贡献指南](./zh/contributing/contributing_guide.md)》。

## ⚖️ 相关说明

🔹 《[开发者指南](./zh/development_guide/develop_guide.md)》  
🔹 《[安全声明](./zh/legal/SECURITY.md)》  
🔹 《[免责声明](./zh/legal/disclaimer.md)》  
🔹 《[许可证声明](./zh/legal/license_notice.md)》  

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交 [Issues](https://gitcode.com/Ascend/msprobe/issues)，我们会尽快回复。感谢您的支持。

| 即时互动（微信群） | 官方资讯（公众号） | 深度支持（助手/论坛） |
|:---:|:---:|:---|
| <img src="https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/zh/figures/readme/officialGroupChat.jpg" width="120"><br><sub>*扫码加入技术交流群*</sub> | <img src="https://raw.gitcode.com/Ascend/msprobe/raw/master/docs/zh/figures/readme/officialAccount.jpg" width="120"><br><sub>*扫码关注官方公众号*</sub> | 欢迎扫码关注技术交流群和官方公众号，直达 MindStudio 用户与开发者最快捷的交流平台：<br> **快速提问：** 与社区小伙伴即时探讨技术问题<br>**掌握动态：** 第一时间获取版本发布与功能更新通知<br> **经验共享：** 与广大开发者交流最佳实践与实战心得  <br><br> **更多支持渠道**：👉 昇腾助手：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 昇腾论坛：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 致谢

本工具由华为公司的下列部门联合贡献：  
🔹 昇腾计算 MindStudio 开发部  
🔹 分布式并行计算实验室  

感谢来自社区的每一个 PR，欢迎贡献 msProbe！

```{toctree}
:maxdepth: 2
:caption: 开始使用
:hidden:
安装指南 <zh/install_guide/msprobe_install_guide>
快速入门 <zh/quick_start/pytorch_quick_start>
常见问题 <zh/support/faq>
安全声明 <zh/legal/SECURITY>
```

```{toctree}
:maxdepth: 2
:caption: 功能指南-训练场景
:hidden:

训练前配置检查 <zh/user_guide/config_check_instruct>
数据采集 <zh/user_guide/dump/pytorch_data_dump_instruct>
分级可视化构图比对 <zh/user_guide/accuracy_compare/pytorch_visualization_instruct>
精度比对 <zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct>
编译精度比对 <zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct>
训练状态监测 <zh/user_guide/monitor_instruct>
趋势可视化 <zh/user_guide/accuracy_compare/trend_visualization_instruct>
精度预检 <zh/user_guide/accuracy_checker/pytorch_accuracy_checker_instruct>
```

```{toctree}
:maxdepth: 2
:caption: 功能指南-推理场景
:hidden:

# vLLM推理
vLLM 数据采集（Eager/图模式） <zh/user_guide/dump/vllm_dump_instruct>
vLLM torchair数据采集 <zh/user_guide/dump/torchair_dump_instruct>

# SGLang推理
SGLang eager模式数据采集 <zh/user_guide/dump/sglang_eager_dump_instruct>

# ATB推理
ATB数据采集 <zh/user_guide/dump/atb_data_dump_instruct>
ATB精度比对 <zh/user_guide/accuracy_compare/atb_data_compare_instruct>
ATB和离线模型数据转换 <zh/user_guide/dump/data_parse_instruct>

# 离线模型推理
离线模型数据采集 <zh/user_guide/dump/infer_offline_dump_instruct>
离线模型比对 <zh/user_guide/accuracy_compare/infer_compare_offline_model_instruct>
离线模型数据比对 <zh/user_guide/accuracy_compare/offlline_data_compare_instruct>
```

```{toctree}
:maxdepth: 2
:caption: 定位指南
:hidden:

大模型训练精度定位指南 <zh/best_practices/train_debug_guide>
大模型推理精度定位指南 <zh/best_practices/infer_debug_guide>
常见框架dump工具使能 <zh/best_practices/dump_enable_guide>
```
