# 对比结果说明

## 简介

本节以非量化昇腾AI处理器运行生成的dump数据与非量化onnx模型npy数据比对为例，介绍对比结果分析步骤，下文中参数说明均以该示例介绍，请根据您的实际情况进行替换。

## 对比输出结果说明

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
│   └-- {onnx}        # 原模型dump数据存放路径，onnx分别对应ONNX模型
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- input_Add_100.0.1682148256368588.npy  # 如果是ONNX模型，则会dump输入数据，并增加对应的input前缀
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                  # 随机输入数据，若指定了输入数据，则该文件不存在
├-- model
│   ├-- {om_model_name}.json
│   └-- new_{om_model_name}.onnx     # 把每个算子作为输出节点后新生成的ONNX模型
└-- result_{timestamp}.csv           # 比对结果文件
```

## 比对结果文件各字段含义说明

- **比对结果** 在文件 `result_{timestamp}.csv` 中，比对结果的含义与基础精度比对工具完全相同，其中每个字段的含义可参考 《CANN商用版 精度调试工具用户指南》中的“附录 > [完整比对结果参数说明](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/modelaccuracy/atlasaccuracy_16_0064.html)”章节。

* 下面简要介绍说明结果信息：

  |                  OpType |  NPUDump | DataType | Address | GroundTruth | DataType | TensorIndex|Shape|Overflow|CosineSimilarity|...|MeanRelativeError|CompareFailReason|IsNpuOps|IsOutputNode|IsPrecisionError|
  |------------------------:|---------:|---------:|--------:|------------:|---------:|-----------:|----:|-------:|---------------:|--:|----------------:|----------------:|-------:|-------:|-------:|
  |                      Sub|Sub_26Mul_28| float16 |    NaN |Sub_26,Mul_28|   float32|Sub_26Mul_28:output:0|[1,1,1,108]|NO|      1|...|         0.000364|                 |NO      |NO      |NO      |

如上所示的结果文件中主要关注以下几项:

 - [x] [NPUDump]：这个对应om模型中的算子，由于融合规则，可能会对应多个GPU/CPU算子。
 - [x] [DataType]：一共有两个，一个是NPU侧的数据类型，一个是CPU/GPU侧的数据类型，二者有所不同，可能会有精度损失问题。
 - [x] [GroundTruth]：om算子所对应的onnx模型算子。
 - [x] [Overflow]：数据是否出现上溢出或下溢出。
 - [x] [CompareFailReason]：比对失败原因，误差可能会因为除零非法或者不对应等原因造成无法计算，变为NaN值，会列出详细原因。
 - [x] [IsNpuOps]：用于过滤是否为npu独有节点。
 - [x] [IsOutputNode]：用于过滤是否为模型的整网输出节点。
 - [x] [IsPrecisionError]：用于过滤是否为精度异常的节点。
 - [x] [CosineSimilarity][RelativeEuclideanDistance]...[MeanRelativeError]：这是各类误差比对类型结果，主要需要看是否某一项超过精度阈值（即某项异常），若超过则需要重点关注。各对比算法说明如下：

|                  误差比对类型名称 |  说明 |
|:------------------------|:---------|
|CosineSimilarity         |进行余弦相似度算法比对出来的结果。取值范围为[-1,1]，比对的结果如果越接近1，表示两者的值越相近，越接近-1意味着两者的值越相反。|
|MaxAbsoluteError|进行最大绝对误差算法比对出来的结果。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。|
|AccumulatedRelativeError|进行累积相对误差算法比对出来的结果。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。|
|RelativeEuclideanDistance|进行欧氏相对距离算法比对出来的结果。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。|
|KullbackLeiblerDivergence|进行KL散度算法比对出来的结果。取值范围为[0, +∞)。KL散度越小，真实分布与近似分布之间的匹配越好。|
|StandardDeviation|进行标准差算法比对出来的结果。取值范围为[0, +∞)。标准差越小，离散度越小，表明越接近平均值。该列显示My Output和Ground Truth两组数据的均值和标准差，第一组展示My Output模型dump数据的数值（均值;标准差），第二组展示Ground Truth模型dump数据的数值（均值;标准差）。|
|MeanAbsoluteError|表示平均绝对误差。取值范围为[0, +∞)，MeanAbsoluteError趋于0，RootMeanSquareError趋于0，说明测量值与真实值越近似；MeanAbsoluteError趋于0，RootMeanSquareError越大，说明存在局部过大的异常值；MeanAbsoluteError越大，RootMeanSquareError等于或近似MeanAbsoluteError，说明整体偏差越集中；MeanAbsoluteError越大，RootMeanSquareError越大于MeanAbsoluteError，说明存在整体偏差，且整体偏差分布分散；不存在以上情况的例外情况，因为RMSE（RootMeanSquareError） ≥ MAE（MeanAbsoluteError）恒成立。|
|RootMeanSquareError|表示均方根误差。取值范围为[0, +∞)，MeanAbsoluteError趋于0，RootMeanSquareError趋于0，说明测量值与真实值越近似；MeanAbsoluteError趋于0，RootMeanSquareError越大，说明存在局部过大的异常值；MeanAbsoluteError越大，RootMeanSquareError等于或近似MeanAbsoluteError，说明整体偏差越集中；MeanAbsoluteError越大，RootMeanSquareError越大于MeanAbsoluteError，说明存在整体偏差，且整体偏差分布分散；不存在以上情况的例外情况，因为RMSE（RootMeanSquareError） ≥ MAE（MeanAbsoluteError）恒成立。|
|MaxRelativeError|表示最大相对误差。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。|
|MeanRelativeError|表示平均相对误差。取值范围为[0, +∞)，值越接近于0，表明越相近，值越大，表明差距越大。|

## 比对结果分析

- 精度指标

  | 误差对比算法                | 精度正常的参考标准   |
  | ------------------------- | ------ |
  | CosineSimilarity          | > 0.99  |
  | RelativeEuclideanDistance | < 0.05  |
  | KullbackLeiblerDivergence | < 0.005 |
  | RootMeanSquareError       | < 1.0   |
  | MeanRelativeError         | < 1.0   |

- **模型精度达标与否**首要的是看整网的输出结果是否精度达标，如果输出精度达标，则即使中间节点精度存在异常（**包括算子溢出**），也无需处理，否则需要逐个排查问题节点。
