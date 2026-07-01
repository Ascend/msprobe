# Checkpoint比对

## 简介

在模型训练过程中或结束后，可能保存一些检查点文件（checkpoint，简称ckpt）记录当前模型、优化器等训练状态。

Checkpoint比对（ckpt Compare，Checkpoint Compare）可比较两个不同的checkpoint，评估模型相似度。

当前支持Megatron-LM、MindSpeed（PyTorch&MindTorch）的ckpt比较。支持TP、PP、EP、VPP模型并行；支持megatron.core、megatron.legacy、TransformerEngine的模型实现。

## 使用前准备

安装msProbe工具，详情请参见《[msProbe安装指南](../install_guide/msprobe_install_guide.md)》。

## Checkpoint比对功能介绍

**功能说明**

比较两个不同的ckpt。

**注意事项**

- Megatron、MindSpeed的ckpt加载依赖megatron，请确保megatron已安装在Python环境中，或将megatron的代码保存在当前路径下。
- 在ckpt传给工具加载前，用户需要确保ckpt是安全可信的，若ckpt来源官方有提供SHA256等校验值，用户必须要进行校验，以确保ckpt没有被篡改。

**命令格式**

```bash
msprobe config_check --compare <ckpt_path1> <ckpt_path2> [-o <output_path.json>]
```

**参数说明**

| 参数名        | 可选/必选 | 参数说明                                                     |
| ------------- | --------- | ------------------------------------------------------------ |
| -c或--compare | 必选      | 执行比对操作，ckpt_path1和ckpt_path2为两个待比对的ckpt路径，路径配置详细介绍请参见[ckpt路径说明](#ckpt_path)。 |
| -o或--output  | 可选      | 比对结果输出路径，默认为./ckpt_similarity.json，可自定义文件名。输出路径存在时将报错终止。 |

**ckpt路径说明**<a name="ckpt_path"></a>

Megatron-LM和MindSpeed的ckpt目录结构示例如下：

```txt
directory_name/
├── iter_0000005/    # 某个iteration时的ckpt目录
│   └── mp_rank_xx_xxx/    # 单个rank的ckpt目录，xx_xxx为模型并行索引
│       └── model_optim_rng.pt    # 包含模型参数、随机状态等的PyTorch binary文件
├── iter_0000010/
├── latest_checkpointed_iteration.txt    # 记录最后一个保存的ckpt的纯文本文件
```

对于--compare参数的两个路径：

- 配置为directory_name时，工具通过latest_checkpointed_iteration.txt自动选择最后一个保存的ckpt进行比对。
- 配置为directory_name/iter_xxxxxxx时，工具使用指定iteration的ckpt进行比对。
- 暂不支持单个rank的比对。

**使用示例**

执行比对操作，示例命令如下：

```bash
msprobe config_check --compare ckpt_path1 ckpt_path2 -o output_path.json
```

**输出说明**

比对操作执行完成后，会打印比对结果json文件的输出路径，详细介绍请参见[输出结果文件说明](#输出结果文件说明)。

## 输出结果文件说明

Checkpoint比对结果以json文件输出，内容如下示例：

```json
{
    "decoder.layers.0.input_layernorm.weight": {
        "l2": 0.0, 
        "cos": 0.999999,
        "numel": 128,
        "shape": [
            128
        ]
    },
    "decoder.layers.0.pre_mlp_layernorm.weight": {
        "l2": 0.012, 
        "cos": 0.98,
        "numel": 128,
        "shape": [
            128
        ]
    }
}
```

统计量 | 解释 |
|-------|---------|
| l2 | 欧氏距离，$\|\|a-b\|\|_2$。 |
| cos | 余弦相似度， $\frac{<a,b>}{\|\|a\|\|_2\|\|b\|\|_2}$。 |
| numel | 参数的元素个数。 |
| shape | 参数的shape。 |
