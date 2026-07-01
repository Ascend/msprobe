# 整网首个溢出节点分析

## 简介

在分析inf、nan的场景下，会采集多个rank下的多个step的dump数据，前面出现的异常会传播到同rank后续的节点，并通过通信算子传播到其他rank的后续节点中，因此如何分析首个nan出现的节点位置尤为重要。

整网首个溢出节点分析（overflow_check）可以对PyTorch的dump数据进行分析，在多卡场景下，检测到每张卡中产生inf/nan的节点。若是经过通信导致的inf/nan，可以分析并找出首个产生inf/nan的rank和节点。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../../install_guide/msprobe_install_guide.md)》。

**数据准备**

采集精度数据，详情请参见[PyTorch场景精度数据采集](../dump/pytorch_data_dump_instruct.md)。

**约束**

当前仅支持分析PyTorch场景的dump数据。

## 功能介绍

**功能说明**

对PyTorch的dump数据进行分析。

**命令格式**

```bash
msprobe overflow_check -i <input_path> -o <output_path>
```

**参数说明**

| 参数              | 可选/必选 | 说明                                                         |
| ----------------- | --------- | ------------------------------------------------------------ |
| -i或--input_path  | 必选      | dump数据的目录，需指定到step层级，如`-i /xxx/dump/step0`。   |
| -o或--output_path | 可选      | 输出文件的目录，默认未配置，表示在当前目录下创建`./output`目录。 |

**使用示例**

```commandline
msprobe overflow_check -i /xxx/dump/step0 -o ./output
```

**输出说明**

当打印如下日志时，分析认为不存在异常节点，不生成分析文件。

```ColdFusion
Cannot find any anomaly node, no need to generate analyze file.
```

存在异常节点时，生成`anomaly_analyze_{timestamp}.json`文件，结构为：

```json
{
  "rank_0": [  // 卡号
    {
      "op_name": "Tensor.op_name.0.forward",  // 节点名
      "data_info": {
        "input_args": [],  // input_args数据
        "input_kwargs": {},  // input_kwargs数据
        "output": []  // output数据
      },
      "construct_info": [],  // 节点层级数据
      "stack_info": {}   // 堆栈数据
    }
  ]
}
```

## 异常判定

**异常计算节点判定**

当某个计算节点的输入值正常，即Max或Min中不存在inf或nan，而输出值存在异常时认为从此节点开始产生了溢出，并有可能向后传递。

**异常通信节点判定**

通信节点按照功能分为有向节点，如`send`, `recv`, `scatter`, `gather`, `broadcast`, `reduce`等，以及无向节点，如`all_gather`, `all_reduce`, `reduce_scatter`, `all_to_all`等。

对于有向节点，当src节点的input存在异常时，通常认为传入的数据中本身就存在异常，因此考虑异常节点发生在src节点所在rank的上一个或多个计算节点中；当src节点的input正常而output存在异常值，或dst节点的output存在异常值时，考虑是通信节点本身的操作产生了异常数据。

对于无向节点，当节点input存在异常时，认为传入的数据中本身就存在异常，因此考虑异常节点发生在src节点所在rank的上一个或多个计算节点中；当input正常而output异常时，考虑是通信节点本身的操作产生了异常数据。

**顺序判定**

对于相连的有向通信算子，认为src节点的异常发生早于dst节点；对于无向通信算子，认为异常是同时发生的。

对于计算节点，按照dump的顺序排序。
