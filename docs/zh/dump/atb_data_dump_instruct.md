
# ATB场景精度数据采集

## 简介

msProbe工具通过在ATB模型运行前，执行ATB dump模块加载脚本的方式，采集模型运行过程中的精度数据。

**注意**：

* 因精度数据的采集需要经过从NPU内存拷贝数据到主机内存、从主机内存写数据到磁盘等IO操作，所以将减慢ATB模型的运行速度。对模型性能的影响程度取决于采集数据大小以及环境IO性能。

* 以真实数据模式采集精度数据时，工具将会直接保存op的输入/输出Tensor，输出结果文件将会占用较大磁盘空间。

* 在执行ATB dump模块加载脚本前，ATB模型运行环境需已就绪，包括CANN Toolkit包、CANN NNAL包的安装与使能。

**基本概念**

* **ATB**：全称为Ascend Transformer Boost，是一款基于昇腾Ascend AI处理器，专门为Transformer模型设计的高效、可靠的加速库。详细介绍请参见《CANN商用版 ATB加速库用户指南》中的“[简介](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/acce/ascendtb/ascendtb_0001.html)”章节。

* **dump**：采集精度数据，并完成数据持久化的过程。

* **Operation**：ATB原生算子。ATB模型有多个Operation构成，每个Operation内又可包含一个或多个其它Operation。其中最外层的Operation又被称为layer级Operation，常见的有WordEmbedding、Prefill_layer、Decoder_layer、LmHead等。

* **Kernel**：ATB Operation调用的底层算子。名称中往往以“Kernel”结束，例如RmsNormKernel、AddBF16Kernel等。

* **op**：广义上的执行算子。包括layer级Operation、非layer级Operation以及Kernel。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**注意**：当前仅支持通过[源码编译安装](../msprobe_install_guide.md#编译安装)的方式安装带有ATB dump模块的msProbe工具，且编译时需通过`--include-mod`参数指定包含atb_probe模块。

**约束**

仅支持CANN 8.3.RC1及以上版本。

## 快速入门

以下通过一个简单的示例，展示如何使用msProbe工具进行ATB模型的精度数据采集。

1. 配置文件创建

    在当前目录下创建`config.json`文件，用于配置dump参数。内容如下：

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "all",
        "ids": "0",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    dump参数介绍请参见[参数说明](#参数说明)章节中的dump配置文件参数说明。

2. 命令执行

    通过`pip show mindstudio-probe`命令确定msProbe工具安装路径。假设安装路径为/usr/local/lib/python3.11/site-packages，则执行以下命令进行dump模块加载。

    ```bash
    MSPROBE_HOME_PATH=/usr/local/lib/python3.11/site-packages
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh --output=$PWD --config=$PWD/config.json
    ```

    命令行参数介绍请参见[参数说明](#参数说明)章节中的命令行参数说明。

3. ATB模型运行

    以使用MindIE镜像进行纯模型推理为例。

    ```bash
    cd $ATB_SPEED_HOME_PATH
    # 请自行准备模型权重文件，传入实际权重路径。并需自行保证权重文件安全可靠
    python examples/run_pa.py --model_path /path-to-weights
    ```

    模型执行过程中的精度数据将会保存在`--output`指定路径下的atb_dump_data目录。

4. ATB dump模块卸载

    采集完精度数据后，执行以下命令进行dump模块卸载。

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/unload_atb_probe.sh
    ```

## ATB dump功能介绍

### 功能说明

ATB dump功能用于ATB模型运行过程中的精度数据采集，包括模型结构信息、输入/输出Tensor的真实数据或统计量数据。并且支持在模型运行过程中修改dump配置文件，实现动态dump。

**注意事项**

* 为了减小dump模块解析配置文件带来的性能开销，在解析到配置文件中的"dump_enable"参数为true前，模块会处于睡眠状态，每间隔一定数量的op执行，才读取一次配置文件。这将可能导致初次将"dump_enable"参数改true后，部分op的精度数据丢失。因此，建议在采集精度数据前唤醒dump模块，进入调试状态，具体唤醒操作可参见[使用示例](#使用示例)中的“4. 唤醒ATB dump模块”步骤说明，或者在模型运行前，就将"dump_enable"参数设true，直接进入调试状态。

* dump模块进入调试状态后，每隔5秒解析一次配置文件。因此配置文件的修改可能在5秒后才可生效。

### 命令格式

ATB dump模块加载命令格式如下：

```bash
source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh [--output=<outputPath>] [--config=<configPath>]
```

ATB dump模块卸载命令格式如下：

```bash
source $MSPROBE_HOME_PATH/msprobe/scripts/atb/unload_atb_probe.sh
```

`$MSPROBE_HOME_PATH`为msProbe工具安装路径，可通过`pip show mindstudio-probe`命令获取。

### 参数说明

**命令行参数说明**

| 参数 | 可选/必选 | 说明 |
| --- | --------- | --- |
| --output | 可选 | 指定dump数据的输出保存路径，默认为当前工作目录。 |
| --config | 可选 | 指定dump配置文件路径，默认为`load_atb_probe.sh`加载脚本同级目录下的`config.json`文件路径。dump配置文件可在需采集数据时再创建。 |

**dump配置文件参数说明**

dump配置文件为JSON格式的文本文件，各配置参数介绍如下：

| 参数 | 可选/必选 | 说明 |
| --- | --------- | --- |
| task         | 可选 | 指定dump任务，str类型，默认为"tensor"。可选值：<br/> "tensor"：采集op的输入/输出Tensor的真实数据；<br/> "statistics"：采集op的输入/输出Tensor的统计量数据；<br/> "all"：采集op的输入/输出Tensor的真实数据与统计量数据。 |
| dump_enable  | 可选 | 指定是否允许dump数据，bool类型，默认为false。可选值：<br/> true：允许采集op的输入/输出Tensor的真实数据或统计量数据；<br/> false：不允许采集op的输入/输出Tensor的真实数据或统计量数据。 |
| exec_range   | 可选 | 指定需dump数据的op执行轮次范围，str类型，默认为"0,0"。可选值：<br/> "all"：dump op所有执行轮次的精度数据；<br/> "none"：op所有执行轮次的精度数据都不dump；<br/> "\<起始轮次\>,\<终止轮次\>"： dump op从起始轮次到终止轮次间的精度数据，包括起始轮次与终止轮次。<br/> **配置示例**："exec_range": "0,2"，表示dump op第1、2、3执行时的精度数据（第N次执行的执行轮次为N-1）。|
| ids          | 可选 | 指定需dump数据的op的ID，str类型，默认为""，表示dump所有layer级Operation的精度数据。需满足"\<ID1\>,\<ID2\>"格式，指定一个或多个ID。<br/> **配置示例**：<br/> "ids": "0"，表示dump ID为0的op的精度数据；<br/> "ids": "2_1"，表示dump ID为2的op下的ID为1的OP的精度数据；<br/> "ids": "0,2_1"，表示dump ID为0的op以及ID为2的op下的ID为1的OP的精度数据。 |
| op_name      | 可选 | 指定需dump数据的op的名称，str类型，默认为""，表示dump所有layer级Operation的精度数据。需满足"\<opName1\>,\<opName2\>"格式，指定一个或多个op名称。<br/> **配置示例**：<br/> "op_name": "word"，表示dump名称以"word"开头的op的精度数据（不区分大小写）。 |
| save_child   | 可选 | 指定是否dump op下的子op的精度数据，bool类型，默认为false。可选值：<br/> true：dump 指定op及内部子op的精度数据；<br/> false：仅dump 指定op的精度数据。 |
| device       | 可选 | 指定需dump数据的device ID，str类型，默认为""，表示dump 所有device上的精度数据。需满足"\<deviceID1\>,\<deviceID2\>"格式，指定一个或多个device ID。<br/> **配置示例**：<br/> "device": "0"，表示dump device0上的精度数据。 |
| filter_level | 可选 | 指定dump op的输入/输出Tensor的真实数据时的过滤等级，int类型，默认为1。该参数仅在指定layer级Operation，且"save_child"为true时生效。可选值：<br/> 0：采集op的输入/输出Tensor的真实数据时，不进行数据过滤；<br/> 1：采集op的输入/输出Tensor的真实数据时，相同Tensor仅保存一次；<br/> 2：在1基础上，过滤Kernel的输入/输出Tensor。 |

dump配置文件示例如下：

```json
{
    "task": "tensor",
    "dump_enable": false,
    "exec_range": "all",
    "ids": "",
    "op_name": "",
    "save_child": false,
    "device": "",
    "filter_level": 1
}
```

### 使用示例

以使用MindIE镜像进行服务化推理为例。请确保当前环境能成功拉起推理服务，且正确安装了msProbe工具。

1. 确定msProbe工具安装路径。查询命令如下：

    ```bash
    pip show mindstudio-probe
    ```

    假设安装路径为/usr/local/lib/python3.11/site-packages，执行以下命令将安装路径保存为环境变量：

    ```bash
    export MSPROBE_HOME_PATH=/usr/local/lib/python3.11/site-packages
    ```

2. 执行dump模块加载脚本。执行命令如下：

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh --output=$PWD --config=$PWD/config.json
    ```

3. 拉起推理服务。拉起命令如下：

    ```bash
    /usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon
    ```

    出现"Daemon start success!"回显信息后，说明服务拉起成功。

4. 唤醒ATB dump模块。

    首先，创建dump配置文件，文件路径为$PWD/config.json，文件内容如下：

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "none",
        "ids": "",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    配置文件中，"dump_enable"参数必须为true，"exec_range"参数需设为"none"，避免在唤醒过程中dump不必要精度数据。

    然后，在请求终端发送推理请求，请求示例如下：

    ```bash
    curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"model": "QWen2.5_7B", "messages": [{"role": "system", "content":{"type": "text", "text": "You are a helpful assistant"}}], "max_tokens": 20}' http://127.0.0.1:1025/v1/chat/completions
    ```

    "model"参数值、IP地址与端口号需根据实际配置修改。

    服务终端出现`"Ready to dump ATB data, the running speed of the model will be affected"`的回显信息后，说明dump模块被唤醒，进入调试状态。也可通过观察$PWD/atb_dump_data/data目录是否被创建，判断dump模块是否被唤醒。

5. dump模型精度数据。

    修改dump配置文件，修改后文件内容如下：

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "all",
        "ids": "2",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    修改并保存配置文件5秒后，在请求终端发送推理请求。dump模块将会采集推理服务在处理请求时的模型运行过程中的精度数据。

    **注意**：请求终端收到回复后仅能说明模型运行完成，不代表精度数据采集完成。需持续观察$PWD/atb_dump_data文件夹的体积大小，直到文件夹体积不再增大。

6. 停止推理服务，卸载dump模块。

    在采集完所需精度数据后，可在服务终端停止推理服务，并卸载dump模块。dump模块卸载命令如下：

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/unload_atb_probe.sh
    ```

### 输出说明

ATB dump输出文件的目录结构示例如下：

```lua
├── outputPath  # 执行模块加载脚本时执行的输出路径
│   ├── atb_dump_data  # 工具自动创建的固定目录
│   |   ├── data  # 工具自动创建的固定目录，存放op的输入/输出数据
│   |   │   ├── 0_39943  # device ID与进程号
|   |   |   |    ├── 0  #  op执行轮次
|   |   |   |    |   ├── 0_WordEmbedding  # layer级Operation ID与名称
|   |   |   |    |   |   ├── 0_GatherOperation  # 非layer级Operation ID与名称
|   |   |   |    |   |   |   ├── 0_Gather16I64Kernel  # Kernel ID与名称
|   |   |   |    |   |   |   |   ├── after  # 存放输出Tensor
|   |   |   |    |   |   |   |   |   |── outtensor0.bin
|   |   |   |    |   |   |   |   ├── before  # 存放输入Tensor
|   |   |   |    |   |   |   |   |   |── intensor0.bin
|   |   |   |    |   |   |   |   |   |── intensor1.bin
|   |   |   |    |   |   |   ├── after
|   |   |   |    |   |   |   └── ...
|   |   |   |    |   |   |   ├── before
|   |   |   |    |   |   |   └── ...
|   |   |   |    |   |   |   ├── op_param.json  # Operation参数数据
|   |   |   |    |   |   |...
|   |   |   |    |   ...
|   |   |   |    |   ├── 5_Prefill_layer
|   |   |   |    |   |   └── ...
|   |   |   |    |   ...
|   |   |   |    |   ├── statistic.csv  # op的输入/输出Tensor的统计量数据
|   |   |   |    ├── 1  #  op执行轮次
|   |   |   |    |   └── ...
|   |   |   |    ...
│   |   │   ├── 1_39945  # device ID与进程号
|   |   |   |   └── ...
|   |   |   ...
│   |   ├── info  # 工具自动创建的固定目录，存放结构信息
│   |   |   ├── layer  # 工具自动创建的固定目录，存放layer级Operation结构信息
│   |   │   |   ├── 0_39943  # device ID与进程号
│   |   │   |   |   ├── WordEmbedding_0.json
|   |   |   |   |   ...
|   |   |   |   ...
│   |   |   ├── model  # 工具自动创建的固定目录，存放模型结构信息
│   |   │   |   ├── 0_39943  # device ID与进程号
│   |   │   |   |   ├── DecoderModel_Decoder.json
│   |   │   |   |   ├── DecoderModel_Prefill.json
|   |   |   |   ...
```

无论配置文件中的"task"为何值，在采集模型精度数据时，statistic.csv均会生成。csv文件的各列含义介绍如下：

| 列名 | 含义 |
| --- | ---- |
| Device and PID  | device ID和进程号。 |
| Execution Count | 执行轮次。 |
| Op Name         | op名称，由父op名称和自身名称组成。例如Prefill_layer/Attention。 |
| Op Type         | op类型。 |
| Op Id           | op ID号，由父op ID和本身ID组成。例如2_0。 |
| Input/Output    | 输入Tensor还是输出Tensor。 |
| Index           | Tensor索引。 |
| Dtype           | Tensor的数据类型。例如bf16。 |
| Format          | Tensor的数据格式。例如nd。 |
| Shape           | Tensor形状。例如36x128。 |
| Max             | Tensor所有元素的最大值。"task"配置参数为"tensor"时，固定为"N/A"。 |
| Min             | Tensor所有元素的最小值。"task"配置参数为"tensor"时，固定为"N/A"。 |
| Mean            | Tensor所有元素的平均值。"task"配置参数为"tensor"时，固定为"N/A"。 |
| Norm            | Tensor所有元素的Norm值。"task"配置参数为"tensor"时，固定为"N/A"。 |
| Tensor Path     | Tensor的真实数据保存路径。"task"配置参数为"statistics"时，固定为"N/A"。|
