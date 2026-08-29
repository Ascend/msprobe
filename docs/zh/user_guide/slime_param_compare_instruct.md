# slime超参比对与关键超参校验

## 简介

slime在NPU和标杆服务器上训练时，采集到训练日志，训练日志中包含真实超参的配置信息，如：训练优化器学习率、KL散度等。

slime超参比对（Slime Hyperparameter Compare）可比较两个不同服务器上训练日志中采集到真实超参配置，通过筛选出仅与配置相关的部分，保存为config配置文件，并比较两个配置文件参数配置。通过辅助用户高效比对真实超参值配置，加速定位因配置差异所引发的训练或推理精度问题。

## 使用前准备

安装msProbe工具，详情请参见《[msProbe安装指南](../install_guide/msprobe_install_guide.md)》。

## slime超参比对

### 功能介绍

**功能说明**

比较NPU和标杆（如：GPU）的训练日志，对日志进行配置解析，去除与配置不相关的打印信息，将配置信息保存成config文件，并通过对比NPU和标杆的config配置文件，输出训练过程中真实超参配置的比对结果，结果保存为csv文件。

**命令格式**

```bash
msprobe config_check -sc <NPU_log> <bench_log> [-o <compare_result>]
```

**参数说明**

| 参数名               | 可选/必选 | 参数说明                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| -sc或--slime-compare | 必选      | 执行比对操作，NPU_log和bench_log分别为两个待比对的slime训练日志路径，路径配置详细介绍请参见[log路径说明](#log_path)。 |
| -o或--output        | 可选      | 比对结果输出路径，默认为当前执行目录下./slime_param_compare_result，也可自定义文件夹路径。|

**log路径说明**<a name="log_path"></a>

对于--slime-compare参数的两个路径：

- NPU_log和bench_log分别为NPU和标杆服务器上训练slime时保存的训练日志。
- NPU_log和bench_log按顺序配置，自动选择以后面一个日志作为标杆日志进行对比。
- 待比对的NPU_log和bench_log训练日志文件格式只支持“.log”或“.txt”（如：slime_NPU.log 或 slime_NPU.txt），否则执行比对时会报错。
- 训练日志中要包含完整的slime配置信息，否则会解析失败。

**使用示例**

执行比对操作，示例命令如下：

```bash
msprobe config_check -sc NPU_log/training.log bench_log/training.log -o ./compare_result
```

**输出说明**

比对操作执行完成后，控制台会打印比对结果文件的输出路径，详细介绍请参见[输出结果文件说明](#output_file_desc1)。

### 输出结果文件说明<a name="output_file_desc1"></a>

比对结果输出路径会生成3个文件：

- bench_config.json：标杆日志中提取的config配置信息。
- NPU_config.json：NPU日志中提取的config配置信息。
- hyper_params_compare.csv：比对结果。里面会有1个sheet页，sheet中会包含超参名称，NPU生效值，bench生效值，是否一致。

slime超参比对结果文件hyper_params_compare.csv，内容如下示例：

| 超参名称                          | NPU生效值 | bench生效值 | 是否一致 |
| ----------------------------------------- | --------- | ----------- | -------- |
| use_kl_loss                    | TRUE      | FALSE     | 否       |
| eps_clip                           | 0.2           | 0.2           | 是       |

比对结果中名词解释：

| 统计值      | 解释                                                         |
| ----------- | ------------------------------------------------------------ |
| 超参名称    | slime日志中涉及的配置信息中真实超参名称                       |
| NPU生效值   | 对应超参名称在NPU训练日志中对应的真实值                      |
| bench生效值 | 对应超参名称在标杆训练日志中对应的真实值                     |
| 是否一致    | NPU生效值和bench生效值是否一致，如果一致则为“是”，反之为“否”，当结果为否时，需要用户进一步确认排查是否因为该超参不一致导致的精度差异 |
