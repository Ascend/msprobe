
# ATB场景精度比对

## 简介

msProbe工具提供ATB场景的精度比对功能，帮助定位精度问题发生点。

**基本概念**

* **余弦相似度**：两个非零向量之间夹角的余弦值。可以用于评估两个Tensor间的相似程度。

* **欧式距离**：在多维空间中两个点之间的绝对距离。可以用于评估两个Tensor间的相似程度。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**数据准备**

ATB模型dump数据。dump数据获取方式请参见[ATB场景精度数据采集](../dump/atb_data_dump_instruct.md)。

**约束**

仅支持基于CANN 8.5.0以上版本采集到的ATB模型精度数据的比对。

## 快速入门

以下通过一个简单的示例，展示如何使用msProbe工具进行ATB模型的精度数据比对。

先预采集ATB模型的标杆精度数据与待比对精度数据（有精度问题的数据），采集方式请参见[ATB场景精度数据采集](../dump/atb_data_dump_instruct.md)。然后执行以下比对命令进行精度比对。

```bash
# 请传入实际精度数据路径
msprobe compare -m atb -gp golden_data/atb_dump_data/data/0_39943/0/ -tp target_data/atb_dump_data/data/0_276107/0/
```

命令行参数介绍请参见[参数说明](#参数说明)章节。

## ATB精度数据比对功能介绍

### 功能说明

ATB精度数据比对功能用于ATB dump数据的精度比对，包括真实数据比对与统计量数据比对，比对结果最终保存在Excel表格中。

**注意事项**

* 采集ATB模型精度数据时，若"task"配置参数为"tensor"或"all"，则dump数据中都包含算子输入输出Tensor的真实数据，因此在比对时，均进行真实数据比对；若"task"配置参数为"statistics"，则dump数据中仅包含算子输入输出Tensor的统计量数据，因此在比对时，进行统计量数据比对。

* 比对的标杆数据与待比对数据必须同时为真实数据或同时为统计量数据。

* 真实数据比对当前仅支持bool、int8、int32、int64、bfloat16、float16、float32类型的Tensor数据。

### 命令格式

```bash
msprobe compare -m atb -gp <goldenDataPath> -tp <targetDataPath> [-o <outputPath>]
```

### 参数说明

| 参数 | 可选/必选 | 说明 |
| --- | --------- | --- |
| -m或--mode         | 必选 | 指定比对场景，必须为atb。 |
| -gp或--golden_path | 必选 | 指定标杆数据路径，必须指定到执行轮次级目录。ATB dump数据的目录结构介绍请参见《ATB场景精度数据采集》中的"[输出说明](../dump/atb_data_dump_instruct.md#输出说明)"。 |
| -tp或--target_path | 必选 | 指定待比对数据路径，必须指定到执行轮次级目录。ATB dump数据的目录结构介绍请参见《ATB场景精度数据采集》中的"[输出说明](../dump/atb_data_dump_instruct.md#输出说明)"。 |
| -o或--output_path  | 可选 | 指定比对结果输出路径，默认为当前工作目录下的output目录（工具会自动创建）。 |

### 使用示例

1. 准备标杆精度数据与待比对精度数据。

    ATB模型的精度数据采集方式请参见[ATB场景精度数据采集](../dump/atb_data_dump_instruct.md)。假设采集到的精度数据分别保存在golden_data/atb_dump_data、target_data/atb_dump_data目录下。

2. 执行比对命令。比对命令如下：

    ```bash
    # 请传入实际精度数据路径
    msprobe compare -m atb -gp golden_data/atb_dump_data/data/0_39943/0/ -tp target_data/atb_dump_data/data/0_276107/0/
    ```

### 输出说明

ATB精度数据比对输出件为Excel表格文件。

**真实数据比对输出件介绍**

真实数据精度比对得到的Excel表格文件的各列含义介绍如下：

| 列名 | 含义 |
| --- | ---- |
| Target Data Name          | 待比对数据名称，由op 名称、op ID、IO类型、索引组成。例如0_WordEmbedding/input.1。 |
| Golden Data Name          | 标杆数据名称，由op 名称、op ID、IO类型、索引组成。例如0_WordEmbedding/input.1。 |
| Target Device and PID     | 采集待比对数据时的device ID和进程号。 |
| Golden Device and PID     | 采集待标杆数据时的device ID和进程号。 |
| Target Execution Count    | 采集待比对数据时的op执行轮次。 |
| Golden Execution Count    | 采集标杆数据时的op执行轮次。 |
| Target Data Type          | 待比对数据的数据类型。 |
| Golden Data Type          | 标杆数据的数据类型。 |
| Target Data Shape         | 待比对数据的数据形状。 |
| Golden Data Shape         | 标杆数据的数据形状。 |
| Cosine                    | 余弦相似度。 |
| Euc Distance              | 欧式距离。 |
| Max Absolute Err          | 最大绝对误差。 |
| Max Relative Err          | 最大相对误差。 |
| One Thousandth Err Ratio  | 相对误差小于千分之一的比例。 |
| Five Thousandth Err Ratio | 相对误差小于千分之五的比例。 |
| Target Max                | 待比对数据的所有元素的最大值。 |
| Golden Max                | 标杆数据的所有元素的最大值。 |
| Target Min                | 待比对数据的所有元素的最小值。 |
| Golden Min                | 标杆数据的所有元素的最小值。 |
| Target Mean               | 待比对数据的所有元素的平均值。 |
| Golden Mean               | 标杆数据的所有元素的平均值。 |
| Target Norm               | 待比对数据的所有元素的Norm值。 |
| Golden Norm               | 标杆数据的所有元素的Norm值。 |

**统计量数据比对输出件介绍**

统计量精度比对得到的Excel表格文件的各列含义介绍如下：

| 列名 | 含义 |
| --- | ---- |
| Target Data Name          | 待比对数据名称，由op 名称、op ID、IO类型、索引组成。例如0_WordEmbedding/input.1。 |
| Golden Data Name          | 标杆数据名称，由op 名称、op ID、IO类型、索引组成。例如0_WordEmbedding/input.1。 |
| Target Device and PID     | 采集待比对数据时的device ID和进程号。 |
| Golden Device and PID     | 采集待标杆数据时的device ID和进程号。 |
| Target Execution Count    | 采集待比对数据时的op执行轮次。 |
| Golden Execution Count    | 采集标杆数据时的op执行轮次。 |
| Target Data Type          | 待比对数据的数据类型。 |
| Golden Data Type          | 标杆数据的数据类型。 |
| Target Data Shape         | 待比对数据的数据形状。 |
| Golden Data Shape         | 标杆数据的数据形状。 |
| Max Diff                  | 最大值绝对误差。 |
| Min Diff                  | 最小值绝对误差。 |
| Mean Diff                 | 平均值绝对误差。 |
| Norm Diff                 | Norm值绝对误差。 |
| Relative Err of Max(%)    | 最大值相对误差。 |
| Relative Err of Min(%)    | 最小值相对误差。 |
| Relative Err of Mean(%)   | 平均值相对误差。 |
| Relative Err of Norm(%)   | Norm值相对误差。 |
| Target Max                | 待比对数据的所有元素的最大值。 |
| Golden Max                | 标杆数据的所有元素的最大值。 |
| Target Min                | 待比对数据的所有元素的最小值。 |
| Golden Min                | 标杆数据的所有元素的最小值。 |
| Target Mean               | 待比对数据的所有元素的平均值。 |
| Golden Mean               | 标杆数据的所有元素的平均值。 |
| Target Norm               | 待比对数据的所有元素的Norm值。 |
| Golden Norm               | 标杆数据的所有元素的Norm值。 |
