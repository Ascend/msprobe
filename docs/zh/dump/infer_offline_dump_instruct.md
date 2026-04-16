# 推理离线模型数据采集

## 简介

提供了传统小模型场景下tensor数据dump功能，获得精度数据，用于模型精度定位。适用于ONNX、OM模型，用户只需要通过参数指定原始模型对应的离线模型。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

采集OM模型数据依赖aisbench包和aclruntime包，用户使用前可通过以下命令安装这两个依赖包。

```bash
 msprobe install_deps -m offline --no_check
```

> [!NOTE] 说明
>
> --no_check参数，会跳过检查目标网站的证书信息，有一定的安全风险，用户需要谨慎使用并自行承担后果。

**约束**

仅支持ONNX、OM模型数据采集。

OM模型dump依赖CANN功能，用户可通过环境变量ASCEND_TOOLKIT_HOME修改CANN路径，默认路径为/usr/local/Ascend/cann。

**安全风险提示**

在模型文件传给工具加载前，用户需要确保传入文件是安全可信的，若模型文件来源官方有提供SHA256等校验值，用户必须要进行校验，以确保模型文件没有被篡改。

## 数据采集

**功能说明**

离线模型精度数据采集。

**注意事项**

仅支持ONNX、OM模型数据采集。

**命令格式**

 ```sh
  msprobe offline_dump --model_path <model_path> [options]
 ```

--model_path为必选参数，[options]代表可选参数，具体参数介绍请参见以下**参数说明**。

**参数说明**

| 参数名               | 可选/必选 | 描述                                                         |
| -------------------- | --------- | ------------------------------------------------------------ |
| --model_path         | 必选      | .onnx或.om格式的模型文件路径，需使用绝对路径，且须指定到文件名，分别对应ONNX、OM模型。 |
| --input_data         | 可选      | 模型的输入数据路径，需使用绝对路径，且须指定到文件名，默认根据模型的input随机生成，多个输入以逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy。注意：使用aipp模型时该输入为OM模型的输入，且支持自动将npy文件转为bin文件。 |
| -o或--output_path    | 可选      | 输出文件路径，需使用绝对路径，默认为当前路径的output文件夹。 |
| --input_shape        | 可选      | 模型输入为静态shape时使用。模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300"，节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称。 |
| --rank               | 可选      | 指定运行设备[0,255]，可选参数，默认0。                       |
| --dym_shape_range    | 可选      | 模型输入为动态shape时使用。动态shape的阈值范围。如果设置该参数，那么将根据参数中所有的shape列表进行依次推理和精度比对。如果模型转换时指定了某个维度值为-1，比对时需要指定特定范围，该维度在比对时不能设置为-1<br/>配置格式为："input_name1:1,3,200\~224,224-230;input_name2:1,300"。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 |
| --output_size        | 可选      | 指定模型的输出size，有几个输出，就设几个值，每个值默认为**90000000**，如果模型输出超出大小，请指定此参数以修正。动态shape场景下，获取模型的输出size可能为0，用户需根据输入的shape预估一个较合适的值去申请内存。多个输出size用英文逗号隔开, 例如"10000,10000,10000"。 |
| --onnx_fusion_switch | 可选      | onnxruntime算子融合开关，默认开启算子融合，如存在onnx dump数据中因算子融合导致缺失的，建议关闭此开关。仅当model_path传入ONNX格式模型时生效。使用方式：--onnx_fusion_switch False。 |

**使用示例**

```bash
 msprobe offline_dump --model_path /model_path/model.onnx -o /dump_output_path
```

或

```bash
 msprobe offline_dump --model_path /model_path/model.om -o /dump_output_path
```

**输出说明**

dump完成后会打印提示信息msprobe ends successfully. 输出结果会放置在配置的输出路径中，没有配置输出路径默认会放在当前路径的output文件夹。

## 输出结果文件说明

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape}用来区分动态shape时不同的模型实际输入，静态shape时没有该层
├-- dump_data
│   ├-- npu                          # npu dump数据目录
│   │   ├-- {timestamp}              # 模型所有npu dump的算子输出
│   │   │   └-- 0                    # Rank设备ID号
│   │   │       └-- {om_model_name}  # 模型名称
│   │   │           └-- 1            # 模型ID号
│   │   │               ├-- 0        # 针对每个Task ID执行的次数维护一个序号，从0开始计数，该Task每dump一次数据，序号递增1
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {timestamp}
│   │   │   ├-- output_0.bin
│   │   │   └-- output_0.npy
│   │   └-- {timestamp}_summary.json
│   └-- {onnx} # 原模型dump数据存放路径，onnx分别对应ONNX模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
└-- model
    ├-- {om_model_name}.json                    # 离线模型OM模型(.om)通过atc工具转换后的json文件
    └-- new_{onnx_model_name}.onnx              # 把每个算子作为输出节点后新生成的ONNX模型

```
