# Overflow/Underflow Detection in the MindSpore Scenario

## Overview

msProbe supports overflow/underflow detection at the O2 compilation level of static graphs. The detection object is at the **kernel** level, corresponding to the **L2** in the **config.json** file.

Note that this tool supports overflow/underflow detection only in INF/NaN mode<sup>a</sup>. To enable INF/NaN detection, perform the following steps:

```Shell
# Enable INF/NaN detection on CANN.
export INF_NAN_MODE_ENABLE=1
# Enable INF/NaN detection on MindSpore.
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
```

**a**: When processing floating-point number overflow/underflow, NPU supports two modes: INF/NaN mode and saturation mode. Inf/NaN mode complies with IEEE 754 and outputs the Inf/NaN compute result based on the definition. In saturation mode, when overflow/underflow occurs during computation, the value is saturated to the extreme value (±MAX) of the floating-point number. For CANN configurations in Atlas training products, the saturation mode is used by default, and the Inf/NaN mode is not supported. For Atlas A2 training products, the Inf/NaN mode is used by default, and the saturation mode is not recommended. For the MindSpore framework, the configuration applies only to Atlas A2 training products. The Inf/NaN mode is used by default. The configuration on the CANN side must match that on the MindSpore framework side.

For details about the configuration example of an overflow/underflow detection task, see [MindSpore Static Graph Scenario](../dump/config_json_introduct.md#task = overflow_check).

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Data Preparation**

Collect precision data by referring to [API Description](../dump/mindspore_data_dump_instruct.md#api-description) in *Precision Data Collection in MindSpore*.

**Constraints**

Only the MindSpore framework is supported.

## Sample Code

The usage of overflow/underflow detection is the same as that of data collection. For details, see [Precision Data Collection in MindSpore](../dump/mindspore_data_dump_instruct.md).

## Overflow/Underflow Detection Result File

The directory structure and meaning of the result file are the same as those of the data collection task. However, only the real data or statistics of the API or kernel with overflow/underflow issues is saved. For details, see [Dump Result File](../dump/mindspore_data_dump_instruct.md#dump-result-file) in *Precision Data Collection in MindSpore*.

**Note**: At the O2 compilation level of static graphs, if the MindSpore version is 2.4 or 2.5 and the **mindstudio-probe .whl** package with `--include-mod=adump` added is not used, the intermediate file **kernel_graph_overflow_check.json** is generated. Generally, this file can be ignored.
