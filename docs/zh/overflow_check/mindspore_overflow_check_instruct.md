# MindSpore场景溢出检测

## 简介

msProbe 工具提供静态图O2编译等级下的溢出检测功能。其中检测对象为 **kernel** 级别，对应 config.json 配置中的 **"L2"** level。

需要注意，本工具仅支持在 INF/NAN 模式<sup>a</sup>下进行溢出检测。INF/NAN 模式的使能方式如下：

```Shell
# 使能 CANN 侧 INF/NAN 模式
export INF_NAN_MODE_ENABLE=1
# 使能 MindSpore 框架侧 INF/NAN 模式
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
```

**a**：在处理浮点数计算溢出问题时，NPU 当前支持两种溢出模式：INF/NAN 模式与饱和模式。INF/NAN 模式遵循 IEEE 754 标准，根据定义输出 INF/NAN 的计算结果。与之对应的饱和模式在计算出现溢出时，饱和为浮点数极值（+-MAX）。对于 CANN 侧配置，Atlas 训练系列产品，默认为饱和模式，且不支持使用 INF/NAN 模式；Atlas A2训练系列产品，默认为 INF/NAN 模式，且不建议使用饱和模式。对于 MindSpore 框架侧配置，仅支持对 Atlas A2 训练系列产品进行设置，默认为 INF/NAN 模式。CANN 侧 与 MindSpore 框架侧配置须一致。

溢出检测任务的配置示例见[MindSpore静态图场景-task配置为overflow_check](../dump/config_json_introduct.md#task配置为overflow_check)。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**数据准备**

采集精度数据，详情请参见MindSpore场景精度数据采集中的"[接口介绍](../dump/mindspore_data_dump_instruct.md#接口介绍)"章节。

**约束**

仅支持MindSpore框架。

## 示例代码

溢出检测功能使用方式与数据采集任务一致，详见[MindSpore场景精度数据采集](../dump/mindspore_data_dump_instruct.md)。

## 溢出检测结果文件介绍

溢出检测结果文件目录结构与含义与数据采集任务一致，但仅保存溢出 API 或 kernel 的真实数据或统计信息。详见MindSpore场景精度数据采集中的[dump结果文件介绍](../dump/mindspore_data_dump_instruct.md#dump结果文件说明)章节。

**说明**：在静态图 O2 编译等级下，若 MindSpore 版本为 2.4，或者 MindSpore 版本为 2.5，且msProbe安装时使用的whl包在编译时未配置`--include-mod=adump`选项，则会产生 kernel_graph_overflow_check.json 中间文件，一般情况下无需关注。
