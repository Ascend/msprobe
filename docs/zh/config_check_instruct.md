# 训练前配置检查

## 简介

该工具主要适用于对比两个环境下可能影响训练精度的配置差异，包括：

- 环境变量
- 第三方库版本
- 训练超参
- 权重
- 数据集
- 随机操作

**工具使用流程**

1. 准备两台训练服务器。

2. 安装msProbe工具。

3. 在两台服务器上分别执行数据采集。

   可选择静态或动态两种采集方式：

   - 静态数据采集：通过命令行启动采集，仅支持环境变量，第三方库版本及训练超参采集。
   - 动态数据采集：通过在训练脚本添加接口启动采集，支持环境变量，第三方库版本、训练超参、权重、数据集和随机操作采集。

4. 数据比对。

5. 结果分析。

   根据[输出结果文件说明](#输出结果文件说明)判断比对结果所比对的属性是否都通过检查。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](./msprobe_install_guide.md)》。

**约束**

支持PyTorch和MindSpore框架。

## 数据采集

### 静态数据采集

**功能说明**

通过命令行启动采集，仅支持环境变量、第三方库版本及训练超参采集。

其中环境变量、第三方库版本默认采集，训练超参采集需要用户传入启动训练的shell脚本或yaml配置文件。

**注意事项**

静态数据采集功能只能获取到系统中的环境变量，shell脚本中解析的超参不支持复杂运算的数据还原，有类似问题时建议使用[动态采集方式](#动态数据采集)。

**命令格式**

```bash
msprobe config_check -d [<*.sh> <*.yaml>] [-o <output_file_path>]
```

**参数说明**

| 参数            | 可选/必选 | 说明|
|---------------|-------|----------------------------------------------|
| -d或--dump   | 必选  | 数据采集模式，可选择是否传入启动训练的用户shell脚本路径或yaml配置文件路径。支持同时传入shell脚本和yaml配置文件，表示采集训练超参、环境变量和第三方库版本，默认都不传入，表示只采集环境变量和第三方库信息。shell脚本和yaml配置文件均可采集训练超参，取决于训练超参所在文件的位置。 |
| -o或--output | 可选    | 采集结果文件输出路径，默认输出结果文件名为config_check_pack.zip的压缩包，可自定义配置文件名，配置路径末尾直接添加结果文件名，但必须以`.zip`后缀结尾。 |

**使用示例**

- 默认场景

  ```bash
  msprobe config_check -d
  ```

- 传入shell脚本

  ```bash
  msprobe config_check -d train.sh -o /xx/output_file_path/config_check_pack.zip
  ```

- 传入shell脚本和yaml配置文件

  ```bash
  msprobe config_check -d train.sh config.yaml -o /xx/output_file_path/config_check_pack.zip
  ```

**输出说明**

执行成功后，两个环境均输出采集结果文件config_check_pack.zip的保存路径。该结果文件用于后续[数据比对](#数据比对)。

### 动态数据采集

**功能说明**

通过在训练脚本添加接口启动采集，支持环境变量，第三方库版本、训练超参、权重、数据集和随机操作采集。

**注意事项**

在使用MindSpeed-LLM进行数据采集时，需要注意动态数据采集中的**apply_patches**函数需要在MindSpeed-LLM
框架pretrain_gpt.py的megatron_adaptor函数导入之后执行。

**使用示例**

1. 
   在训练流程执行到的第一个Python脚本开始处添加如下代码：

   ```Python
   from msprobe.core.config_check import ConfigChecker
   ConfigChecker.apply_patches(fmk)
   ```

   apply_patches：启动数据采集所需的各项patch，参数如下：

   - **fmk** (string)：可选参数，训练框架，string类型。可选"pytorch"和"mindspore"，默认未配置，表示传入"pytorch"。

2. 在模型初始化好之后添加如下代码：

   ```Python
   from msprobe.core.config_check import ConfigChecker
   ConfigChecker(model=model, shell_path="", output_zip_path="", fmk="")
   ```

   ConfigChecker对模型挂上数据采集所需的hook，会在每次模型前向将要被执行的一刻进行数据采集，参数如下：

   - **model** (Model)：可选参数，初始化好的模型，默认不会采集权重和数据集。
   - **shell_path** (list[])：可选参数，动态采集模式下支持**megatron**训练超参自动捕获，使用**megatron**时推荐不传入，其他情况下可选择是否传入启动训练的用户shell脚本路径或yaml配置文件路径。支持同时传入shell脚本和yaml配置文件，表示采集训练超参，默认都不传入，表示不采集训练超参。shell脚本和yaml配置文件均可采集训练超参，取决于训练超参所在文件的位置。
   - **output_zip_path** (string)：可选参数，采集结果文件输出路径，默认输出结果文件名为config_check_pack.zip的压缩包，可自定义配置文件名，配置路径末尾直接添加结果文件名，但必须以`.zip`后缀结尾。
   - **fmk** (string)：可选参数，训练框架，string类型。可选"pytorch"和"mindspore"，默认未配置，表示传入"pytorch"。

   采集完成后会得到一个zip包，里面包括各项[影响精度的配置](#简介)。会分rank和step存储，其中step为micro_step。

3. 在另一个环境上执行上述操作，得到另一个zip包。

**输出说明**

执行成功后，两个环境均输出采集结果文件config_check_pack.zip的保存路径。该结果文件用于后续[数据比对](#数据比对)。

## 数据比对

**功能说明**

将[数据采集](#数据采集)在两个训练环境下分别采集的zip包作为输入，执行数据比对操作。

**注意事项**

无

**命令格式**

```bash
msprobe config_check -c bench_zip_path cmp_zip_path [-o <output_path>]
```

**参数说明**

| 参数          | 可选/必选 | 说明                                                         |
| ------------- | --------- | ------------------------------------------------------------ |
| -c或--compare | 必选      | 数据比对。须同时配置bench_zip_path和cmp_zip_path两个参数，其中bench_zip_path为标杆环境采集到的数据，cmp_zip_path为待对比环境采集到的数据。 |
| -o或--output  | 可选      | 比对结果输出路径，默认为config_check_result。若重复执行比对，输出路径里原有的比对结果会被覆盖。 |

**使用示例**

将两个zip包拷贝到同一个环境下，执行如下命令进行比对：

```bash
msprobe config_check -c bench_zip_path cmp_zip_path
```

**输出说明**

比对命令执行完成后生成比对结果文件，详细介绍请参见[输出结果文件说明](#输出结果文件说明)。

## 输出结果文件说明

比对结果输出路径会生成2个目录和1个文件：

- bench：bench_zip_path里打包的数据。
- cmp：cmp_zip_path里打包的数据。
- result.xlsx：比对结果。里面会有多个sheet页，其中**summary**总览通过情况，其余页是具体检查项的详情。其中step为micro_step。

| file_name       |pass_check|
|-----------------|----------|
| env             |pass|
| pip             |pass|
| dataset         |pass|
| weights         |pass|
| hyperparameters |pass|
| random          |pass|

以上六项分别对应环境变量，第三方库版本，数据集，权重，训练超参，随机函数检查。

pass_check表示检查是否通过，pass为通过，error为不通过，warning表示存在对msProbe后续操作无影响的非关键第三方库版本不一致，建议用户查看详细信息分析。

前五项检查在**精度比对**前必须保证pass。
