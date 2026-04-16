# MindSpore动态图场景离线精度预检

## 简介

**MindSpore动态图精度预检**：通过扫描昇腾NPU环境上，用户训练MindSpore模型中的所有Mint API以及MSAdapter场景下迁移的MindSpore API，输出精度情况的诊断和分析。工具以模型中所有API前反向的dump结果为输入，构造相应的API单元测试，将NPU输出与标杆（CPU高精度）比对，计算对应的精度指标，从而找出NPU中存在精度问题的API。本工具支持**随机生成模式和真实数据模式**。

**基本概念**

- Mint API：MindSpore在动态图执行过程中产生的能与PyTorch对应的API。

- MSAdapter：用于PyTorch到MindSpore的兼容层，可使部分PyTorch代码在MindSpore上运行。

- 随机生成模式：通过值域自动构造输入数据，结果精度略低，用于快速定位可能的精度问题。 
- 真实数据模式：使用模型dump生成的真实输入数据比对，结果更可靠。

**离线预检流程**

操作流程如下：

1. 在NPU环境下安装msProbe。
2. 在NPU训练脚本内添加msProbe工具dump接口PrecisionDebugger，采集待预检数据。详见《[MindSpore场景精度数据采集](../dump/mindspore_data_dump_instruct.md)》，注意需要配置level="L1"。
3. 执行预检操作，查看预检结果文件，分析预检不达标的API。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

仅支持MindSpore动态图场景。

## 快速入门

### 数据准备

请在当前目录下创建一个`dump.json`文件，模拟dump输出文件，内容如下：

```json
{
    "task": "statistics",
    "level": "L1",
    "dump_data_dir": null,
    "framework": "mindspore",
    "data": {
        "Mint.where.0.forward": {
            "input_args": [
             {
              "type": "mindspore.Tensor",
              "dtype": "Bool",
              "shape": [
               1,
               4096
              ],
              "Max": false,
              "Min": false,
              "Mean": null,
              "Norm": null
             },
             {
              "type": "int",
              "value": 0
             },
             {
              "type": "int",
              "value": 1
             }
            ],
            "input_kwargs": {},
            "output": [
             {
              "type": "mindspore.Tensor",
              "dtype": "Int64",
              "shape": [
               1,
               4096
              ],
              "Max": 1.0,
              "Min": 1.0,
              "Mean": 1.0,
              "Norm": 64.0
             }
            ]
           }}}
```

### 运行命令行

```bash
msprobe acc_check -api_info ./dump.json
```

详细分析结果请参见[预检结果说明](#预检结果说明)。

## 离线精度预检功能介绍

### 使用acc_check执行预检

**功能说明**

acc_check命令用于对dump.json中记录的所有API执行单元测试，比较NPU与CPU输出差异，生成前向与反向的精度结果。适用于单卡场景下的API精度预检。

**注意事项**

- 预检依赖dump的真实数据，需确保dump配置为level="L1"或level="mix"。

- 随机数据模式下，-save_error_data会额外保存输入输出文件，请提前评估磁盘容量。

**命令格式**

```bash
msprobe acc_check -api_info <dump_json_path> [-o <out_path>] [-csv_path <result_csv_path>] [-save_error_data]
```

可选字段使用 [] 表示，变量使用 < > 表示。

**参数说明**

| 参数     | 可选/必选  | 说明                                                                                                                                                                   |
| ------------------------- | ---------------------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -api_info或--api_info_file | 必选    | 指定API信息文件dump.json，str类型。对其中的Mint API以及部分Tensor API进行预检，预检支持的Tensor API列表详见[预检支持列表](../../../python/msprobe/mindspore/api_accuracy_checker/checker_support_api.yaml)。 |
| -o或--out_path    | 可选    | 指定预检结果存盘路径，str类型，默认“./”。                                                                                                                                             |
| -csv_path或--result_csv_path | 可选    | 指定本次运行中断时生成的`accuracy_checking_result_{timestamp}.csv`文件路径，执行acc_check中断时，若想从中断处继续执行，配置此参数即可，str类型。需要指定为上次中断的`accuracy_checking_result_{timestamp}.csv`文件。详见[断点续检](#断点续检)。 |
| -save_error_data              | 可选    | 保存（随机数据模式）精度未达标的API输入输出数据。                                                                                                                                           |

**示例1：执行预检**

```bash
msprobe acc_check -api_info ./dump.json -o ./checker_result
```

acc_check执行结果在-o参数指定路径下生成，包括`accuracy_checking_result_{timestamp}.csv`和`accuracy_checking_details_{timestamp}.csv`两个文件。`accuracy_checking_result_{timestamp}.csv`属于API级，标明每个API是否通过测试。建议用户先查看`accuracy_checking_result_{timestamp}.csv`文件，对于其中没有通过测试的或者特定感兴趣的API，根据其API Name字段在`accuracy_checking_details_{timestamp}.csv`中查询其各个输出的达标情况以及比较指标。详细介绍请参见[预检结果说明](#预检结果说明)。

随机数据模式下，如果需要保存比对不达标的输入和输出数据，可以在acc_check执行命令结尾添加`-save_error_data`，例如：

**示例2：保存未达标输入输出数据**

```bash
msprobe acc_check -api_info ./dump.json -o ./checker_result -save_error_data
```

未达标API的数据将保存在：

```bash
{out_path}/error_data/
```

### 使用multi_acc_check执行多线程预检

**功能说明**

**multi_acc_check**可在多张NPU卡上并行执行**acc_check**操作，用于加速大规模模型的API精度预检，适合7B/13B/38B等大模型。

**注意事项**

- 所有参与的Device ID必须处于空闲状态，多卡acc_check会并发调用Python进程。

- 多卡输出结果不相互覆盖，会自动创建不同时间戳的子目录。

**命令格式**

```bash
msprobe multi_acc_check -api_info <dump_json_path> [-d device_list] [-o out_path] [-csv_path result_csv_path] [-save_error_data]
```

**参数说明**

| 参数名称                      | 说明                                                                                                                                                            | 参数类型      | 是否必选     |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------| ---------------------------------- |
| -api_info或--api_info_file | 指定API信息文件dump.json。对其中的Mint API以及部分Tensor API进行预检，预检支持的Tensor API列表详见[预检支持列表](../../../python/msprobe/mindspore/api_accuracy_checker/checker_support_api.yaml)。                    | str       | 是      |
| -o或--out_path             | 指定预检结果存盘路径，默认“./”。                                                                                                                                            | str       | 否      |
| -csv_path或--result_csv_path | 指定本次运行中断时生成的`accuracy_checking_result_{timestamp}.csv`文件路径，执行acc_check中断时，若想从中断处继续执行，配置此参数即可。需要指定为上次中断的`accuracy_checking_result_{timestamp}.csv`文件。详见[断点续检](#断点续检)。 | str       | 否      |
| -d或--device               | 指定DeviceID，选择acc_check代码运行所在的卡，默认值为0，支持同时指定0~Device数量-1，例如0 1 2 3 4。                                                                                          | List[int] | 否      |
| -save_error_data          | 保存（随机数据模式）精度未达标的API输入输出数据。                                                                                                                                    | 空         | 否      |

在不同卡数下，使用38B语言大模型的预检耗时基线参考[multi_acc_check耗时基线](../baseline/mindspore_accuracy_checker_perf_baseline.md)。

**使用示例**

```bash
msprobe multi_acc_check -api_info ./dump.json -d 0 1 2 3
```

未达标API的数据将保存在：

```bash
./ut_error_data{timestamp}/
```

**输出说明**

multi_acc_check预检执行后，会在每个Device下各自生成两个csv文件，详细介绍请参见[预检结果说明](#预检结果说明)。

### 断点续检

**功能说明**

断点续检用于在预检因环境或数据规模导致中断后，从中断位置继续执行，而无需重新进行全部比对。

**注意事项**

- 必须保证accuracy\_checking\_result\_*.csv与对应的accuracy\_checking\_details\_*.csv未被修改过。

- 必须使用上次中断生成的CSV文件，否则无法保证断点续检的正确性。

- 断点续检不会重新创建CSV文件，而是在原文件后追加结果。

**使用示例**

```bash
msprobe acc_check -api_info ./dump.json -csv_path xxx/accuracy_checking_result_{timestamp}.csv
```

须指定为上次预检中断的`accuracy_checking_result_{timestamp}.csv`文件。请勿修改`accuracy_checking_result_{timestamp}.csv`和`accuracy_checking_details_{timestamp}.csv`文件以及文件名，否则不对断点续检的结果负责。

## 预检结果说明

精度预检生成的`accuracy_checking_result_{timestamp}.csv`和`accuracy_checking_details_{timestamp}.csv`文件内容详情如下：

`accuracy_checking_details_{timestamp}.csv`

| 字段      | 含义       |
| ------------------- | ------------------------------------------------- |
| API Name            | API名称。                                |
| Bench Dtype         | 标杆数据的API数据类型。    |
| Tested Dtype        | 被检验数据的API数据类型。                         |
| Shape               | API的Shape信息。     |
| Cosine              | 被检验数据与标杆数据的余弦相似度。                 |
| MaxAbsErr           | 被检验数据与标杆数据的最大绝对误差。               |
| MaxRelativeErr      | 被检验数据与标杆数据的最大相对误差。               |
| Status              | API预检通过状态，pass表示通过测试，error表示未通过。 |
| Message             | 提示信息。            |

注意：PyTorch无法对dtype为整数类型的tensor进行反向求导，而MindSpore支持。反向过程的预检仅比较dtype为浮点型的输出。

`accuracy_checking_result_{timestamp}.csv`

| 字段                  | 含义      |
| --------------------- | ---------------- |
| API Name              | API名称。         |
| Forward Test Success  | 前向API是否通过测试，pass为通过，error为错误。 |
| Backward Test Success | 反向API是否通过测试，pass为通过，error为错误，如果是空白的话代表该API没有反向输出。 |
| Message               | 提示信息。         |

Forward Test Success和Backward Test Success是否通过测试是由`accuracy_checking_details_{timestamp}.csv`中的余弦相似度、最大绝对误差判定结果决定的。具体规则详见[API预检指标](#api预检指标)。
需要注意的是`accuracy_checking_details_{timestamp}.csv`中可能存在一个API的前向（反向）有多个输出，那么每个输出记录一行，而在`accuracy_checking_result_{timestamp}.csv`中的结果需要该API的所有结果均为pass才能标记为pass，只要存在一个error则标记error。

### API预检指标

   - API预检指标是通过对`accuracy_checking_details_{timestamp}.csv`中的余弦相似度、最大绝对误差的数值进行判断，得出该API是否符合精度标准的参考指标。
   - 余弦相似度大于0.99，并且最大绝对误差小于0.0001，标记“pass”，否则标记为“error”。
