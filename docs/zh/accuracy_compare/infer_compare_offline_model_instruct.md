# 推理离线模型一键式精度比对

## 简介

离线模型一键式精度比对（推理）功能将推理场景的精度比对做了自动化，适用于ONNX、OM模型，用户输入原始模型、对应的离线模型，输出整网比对的结果。离线模型为通过ATC工具转换的OM模型。<br>
支持动态shape模型精度比对；支持AIPP(Artificial Intelligence Pre-Processing)数据预处理功能。<br>
**注意**：请确保ATC工具转换的OM模型与当前运行环境使用的芯片型号一致。

## 使用前准备

**环境准备**

安装msProbe工具，具体请参见《[msProbe安装指南](../msprobe_install_guide.md)》。<br>
比对OM模型依赖aisbench包和aclruntime包，用户使用前可通过以下命令安装这两个依赖包。<br>

 ```sh
  msprobe install_deps -m offline [--no_check]
 ```

需要注意的是，--no_check参数，会跳过检查目标网站的证书信息，有一定的安全风险，用户需要谨慎使用并自行承担后果。

**约束**

仅支持ONNX、OM模型比对。

一键式精度比对依赖CANN功能，用户可通过环境变量ASCEND_TOOLKIT_HOME修改CANN路径，默认路径为/usr/local/Ascend/cann。

**安全风险提示**

在模型文件传给工具加载前，用户需要确保传入文件是安全可信的，若模型文件来源官方有提供SHA256等校验值，用户必须要进行校验，以确保模型文件没有被篡改。

## 推理离线模型一键式精度比对功能介绍

### 功能说明

使用命令行工具对离线模型进行一键式比对，只需输入模型，无需提前采集数据，输出比对结果。

### 注意事项

仅支持ONNX、OM模型比对。

一键式精度比对依赖CANN功能，用户可通过环境变量ASCEND_TOOLKIT_HOME修改CANN路径，默认路径为/usr/local/Ascend/cann。

### 命令格式

 ```sh
  msprobe compare -m offline_model -gp /golden_path/golden_model.onnx -tp /target_path/target_path.om -o /compare_output_path
 ```

### 参数说明

| 参数名称                 | 解释                                                                                                                                                                                                                                                                   | 是否必选 |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| -m或--mode            | 比对模式，需要指定为offline_model。                                                                                                                                                                                                                                             | 是    |
| -gp或--golden_path    | 模型文件[.onnx, .om]路径，分别对应ONNX、OM模型。                                                                                                                                                                                                                                    | 是    |
| -tp或--target_path    | 昇腾AI处理器的离线模型[.om]路径。                                                                                                                                                                                                                                                 | 是    |
| --input_data         | 模型的输入数据路径，路径须指定到具体文件名。默认根据模型的input随机生成。多个输入以英文逗号分隔，例如：/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy。注意：使用aipp模型时该输入为OM模型的输入，且支持自动将npy文件转为bin文件。                                                                                                           | 否    |
| -o或--output_path     | 输出文件路径，默认为当前路径的output文件夹。                                                                                                                                                                                                                                            | 否    |
| --input_shape        | 模型输入为静态shape时使用。模型输入的shape信息，默认为空，例如"input_name1:1,224,224,3;input_name2:3,300"，使用双引号，节点中间使用英文分号隔开。input_name必须是转换前的网络模型中的节点名称。                                                                                                                                      | 否    |
| --dym_shape_range    | 模型输入为动态shape时使用。动态shape的阈值范围。如果设置该参数，那么将根据参数中所有的shape列表进行依次推理和精度比对。如果模型转换时指定了某个维度值为-1，比对时需要指定特定范围，该维度在比对时不能设置为-1。<br/>配置格式为："input_name1:1,3,200\~224,224-230;input_name2:1,300"。<br/>其中，input_name必须是转换前的网络模型中的节点名称；"\~"表示范围，a\~b\~c含义为[a: b :c]；"-"表示某一位的取值。 <br/> | 否    |
| --rank               | 指定运行设备[0,255]，可选参数，默认0。                                                                                                                                                                                                                                              | 否    |
| --output_size        | 指定模型的输出size，有几个输出，就设几个值，每个值默认为**90000000**，如果模型输出超出大小，请指定此参数以修正。动态shape场景下，获取模型的输出size可能为0，用户需根据输入的shape预估一个较合适的值去申请内存。多个输出size用英文逗号隔开, 例如"10000,10000,10000"。                                                                                                       | 否    |
| --onnx_fusion_switch | onnxruntime算子融合开关，默认开启算子融合，如存在onnx dump数据中因算子融合导致缺失的，建议关闭此开关。使用方式：--onnx_fusion_switch False。                                                                                                                                                                        | 否    |

### 输出说明

比对完成则打印提示信息msprobe compare ends successfully.
在配置的输出路径中，生成dump_data文件夹、 input文件夹、model文件夹和.csv后缀的文件，csv文件名称基于时间戳自动生成，格式为：result_{timestamp}.csv。

## 输出结果文件说明

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape}用来区分动态shape时不同的模型实际输入，静态shape时没有该层
├-- dump_data
│   ├-- npu                          # npu dump数据目录
│   │   ├-- {timestamp}              # 模型所有npu dump的算子输出，dump为False情况下没有该目录
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
│   └-- {onnx} # 原模型dump数据存放路径，onnx对应ONNX模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- input_Add_100.0.1682148256368588.npy  # 如果是ONNX模型，则会dump输入数据，并增加对应的input前缀
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json                    # 离线模型OM模型(.om)通过atc工具转换后的json文件
│   └-- new_{onnx_model_name}.onnx              # 把每个算子作为输出节点后新生成的ONNX模型
└-- result_{timestamp}.csv                   # 比对结果文件
```

### 查看比对结果

请前往[对比结果说明](infer_compare_result.md)
