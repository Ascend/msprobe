# PyTorch场景kernel级精度数据采集

## 简介

本文主要介绍kernel级精度数据采集的配置示例和采集结果介绍，msProbe数据采集功能的详细使用参考《[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md)》。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe工具安装指南](../../install_guide/msprobe_install_guide.md)》。

**约束**

仅支持PyTorch框架。

## 快速入门

参考[PyTorch场景精度数据采集-快速入门](./pytorch_data_dump_instruct.md#快速入门)。

## kernel dump配置说明

使用kernel dump时，task需要配置为tensor，list必须填写一个API名称，kernel dump目前每个step只支持采集一个API的数据。

API名称填写参考L1 dump结果文件dump.json中的API名称，命名格式为：`{api_type}.{api_name}.{API调用次数}.{forward/backward}`。

配置示例如下：

```json
{
    "task": "tensor",
    "dump_path": "./dump_path",
    "level": "L2",
    "rank": [],
    "step": [],
    "tensor": {
        "scope": [],
        "list": ["Functional.linear.0.backward"]
    }
}
```

## dump结果文件介绍

### 采集结果说明

如果kernel级数据采集成功，会打印以下信息：

```bash
The kernel data of {api_name} is dumped successfully.
```

注意：如果打印该信息后，没有数据生成，参考[常见问题](#常见问题)进行排查。

如果kernel dump遇到不支持的API，会打印以下信息：

```bash
The kernel dump does not support the {api_name} API.
```

其中{api_name}是对应的API名称。

### 输出文件说明

kernel级数据采集成功后，会在指定的dump_path目录下生成如下文件：

```ColdFusion
├── /home/data_dump/
│   ├── step0
│   │   ├── 20241201103000    # 日期时间格式，表示2024-12-01 10:30:00
│   │   │   ├── 0             # 表示device id
│   │   │   │   ├──{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}    # kernel层算子数据
│   │   │  ...
│   │   ├── kernel_config_{device_id}.json    # kernel dump在接口调用过程中生成的中间文件，一般情况下无需关注
│   │  ...     
│   ├── step1
│  ...
```

## 附录

### 常见问题

#### 采集结果文件为空，有可能是什么原因？

1. 首先需要确认工具使用方式、配置文件内容和list填写的API名称格式是否都正确无误。
2. 其次需要确认API是否运行在昇腾NPU上，如果是运行在其他设备上则不会存在kernel级数据。
3. 如果排除上述两点仍然没有数据，您可以使用《[Ascend for PyTorch](https://gitcode.com/Ascend/pytorch)》提供的torch_npu.npu接口进行kernel级数据采集，工具的kernel dump也是基于其中的init_dump、set_dump和finalize_dump三个子接口实现的。torch_npu.npu接口详细描述见《[torch_npu.npu API 概述](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000192.html)》。
