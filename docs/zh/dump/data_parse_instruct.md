# 数据转换功能

## 简介

本功能提供了将dump数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件的能力。支持将ATB场景的dump数据（.bin文件）或adump数据进行转换，便于后续的数据分析和处理。

## 使用前准备

**环境准备**

- 安装msProbe工具，详情请参考《[msProbe安装指南](../msprobe_install_guide.md)》。
- 当指定转换格式为PyTorch tensor(.pt)时，需安装PyTorch。
- 转换adump数据时需要确保安装配套版本的CANN Toolkit开发套件包并配置CANN环境变量，具体请参见[CANN快速安装](https://www.hiascend.com/cann/download)。

**约束**

- 支持.bin格式的ATB的dump数据转换。
- 支持adump的数据转换。

## 数据转换功能介绍

**功能说明**

将dump数据转换为numpy（.npy）或PyTorch tensor（.pt）格式文件。

**命令格式**

```sh
msprobe parse -d <dump_path> [-t <type>] [-o <output_path>]
```

**参数说明**

| 参数名 | 可选/必选 | 描述 |
|--------|------|------|
| -d或--dump_path | 必选 | 待转换的文件路径或目录路径。支持单个文件或目录输入：<br>&#8226; 单个文件：直接指定文件路径，须指定到文件名。<br>&#8226; 目录：指定dump文件所在目录。 | 是 |
| -t或--type | 可选 | 输出文件类型，支持以下两种格式：<br>&#8226; npy：输出为numpy（.npy）格式文件。<br>&#8226; pt：输出为PyTorch tensor（.pt）格式文件。<br>默认值为pt。 |
| -o或--output_path | 可选 | 输出文件路径，默认为当前路径下的output文件夹。 |

**使用示例（转换单个dump文件）**

```sh
msprobe parse -d /path/to/dump_file -o /path/to/output
```

**使用示例（转换整个目录下的dump文件）**

```sh
msprobe parse -d /path/to/dump_file_directory -o /path/to/output
```

**输出说明**

上述示例执行完成后，在--output_path参数指定路径下生成--type参数指定的格式文件。--dump_path参数指定为单个文件时只转换单个文件；参数指定为目录时，转换该目录下的所有文件。
