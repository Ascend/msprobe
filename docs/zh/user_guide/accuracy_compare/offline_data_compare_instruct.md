# 离线模型dump数据精度比对

## 简介

本节主要介绍传统小模型场景的精度比对工具：该工具用于ONNX和TensorFlow框架模型的ATC模型转换前后的比对、TensorFlow训练场景的比对和离线模型不同版本之间的比对等场景。

## 使用前准备

**环境准备**

- 安装msProbe工具，详情请参见《[msProbe安装指南](../../install_guide/msprobe_install_guide.md)》。
- 安装配套版本的CANN Toolkit开发套件包和ops算子包并配置CANN环境变量，具体请参见[CANN快速安装](https://www.hiascend.com/cann/download)。

**约束**

支持Caffe、ONNX、TensorFlow和OM模型的离线dump数据。

## 功能介绍

**功能说明**

将离线模型dump的数据进行精度比对操作。

**注意事项**

无

**命令格式**

```shell
msprobe compare -m offline_data -tp <target_path> -gp <golden_path> [-fr <fusion_rule_file>] [-cfr <close_fusion_rule_file>] [-qfr <quant_fusion_rule_file>] [-mgf <model_graph_file>] [-o <output_path>]
```

**参数说明**

| 参数名                 | 可选/必选 | 说明                                                                                                                                             |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------|------|
| -m | 必选 | 比对模式，当前场景为offline_data，表示离线数据比对场景。 |
| -tp 或 --target_path | 必选 | 基于昇腾AI处理器运行生成的数据文件所在目录。由于dump数据文件是多个二进制文件，故须指定dump数据文件所在的父目录。如：$HOME/MyApp_mind/resnet50，其resnet50文件夹下直接保存dump数据文件。<br><br>训练场景下：<br><br>支持TensorFlow为原始训练网络的比对。<br>单个数据文件比对时，需指定数据文件所在的具体目录。<br>支持多个dump数据文件的批量比对，可指定固定路径为dump_path/time/，仅支持TensorFlow为原始训练网络的比对。指定的路径下可以存放多个dump数据文件，但要求每个dump数据文件拥有唯一路径，且路径命名规则为dump_path/time/device_id/model_name/model_id/dump_step/dump文件。 |
| -gp 或 --golden_path | 必选 | 基于GPU/CPU运行生成的原始网络数据文件所在目录。<br><br>由于npy文件是多个文件，故须指定npy文件所在的父目录。如：$HOME/Standard_caffe/resnet50，其中resnet50文件夹下直接保存npy数据文件。<br><br>当指定-cfr参数时，该参数指定的就是模型转换关闭算子融合功能下dump数据文件的目录。 |
| -o 或 --output_path  | 可选 | 比对数据结果存放路径，默认为当前路径。<br><br>不建议配置与当前用户不一致的其它用户目录，避免提权风险。<br><br>训练场景下：<br><br>单个数据文件比对时，结果文件名格式为`result_{timestamp}.csv`。<br>多个数据文件比对时，结果文件名格式为`{device_id}_{model_name}_{dump_step}_result_{timestamp}.csv`，批量比对将生成多个csv结果文件。 |
| -fr 或 --fusion_rule_file | 可选 | 全网层信息文件。<br><br>推理场景下：<br><br>通过使用ATC转换.om模型文件生成的json文件，文件包含整网算子的映射关系。<br>该参数指定的是默认开启算子融合功能情况下进行模型转换时生成的json文件；指定关闭算子融合功能情况下进行模型转换时生成的json文件使用-cfr参数。<br><br>训练场景下：<br><br>通过使用ATC转换.txt图文件生成的json文件。<br>单个数据文件比对时，该参数需指定具体的json文件；批量比对时，该参数可指定为多个json文件所在的目录。 |
| -qfr 或 --quant_fusion_rule_file   | 可选  | 量化信息文件（昇腾模型压缩输出的json文件）。<br><br>通过AMCT量化生成的量化信息文件（*.json），文件包含整网量化算子映射关系，用于精度比对时算子匹配。<br><br>Caffe非量化原始模型 vs 量化离线模型场景时，与-fr参数二选一；Caffe非量化原始模型 vs 量化原始模型场景时，仅使用本参数。<br><br>仅推理场景支持本参数。 |
| -cfr 或 --close_fusion_rule_file  | 可选 | 全网层信息文件（通过使用ATC转换.om模型文件生成的json文件，文件包含关闭算子融合功能情况下整网算子的映射关系）。<br><br> 仅推理场景支持本参数。 |
| -mgf 或 --model_graph_file | 可选 | 用于OM与OM直接比对结果按目标侧OM模型图中的算子顺序排序，不参与两侧dump的算子匹配。<br><br>默认不指定时，结果按Index排序。模型图JSON可通过`atc --mode=1 --om=<model.om> --json=<model.json>`生成；部分未匹配的dump算子排列在末尾，全部未匹配时停止比对并提示检查图文件和名称格式。不能与-fr、-qfr或-cfr参数组合使用，该互斥限制在内部detail/mapping模式下同样有效。 |

**使用示例**

1. 完成离线模型的数据dump。

   用户须自行准备GPU和NPU离线模型的dump数据。

2. 执行dump数据精度比对。示例命令如下：

   ```shell
   msprobe compare -m offline_data -tp ./data/target_path -gp ./data/golden_path -fr ./data/fusion_rule.json
   ```

   普通OM与OM同名比对按目标OM图顺序输出时，执行：

   ```shell
   msprobe compare -m offline_data -tp ./data/target_path -gp ./data/golden_path -mgf ./data/target_model.json
   ```

**输出说明**

默认情况下，比对命令执行完成后在命令执行的当前路径生成比对结果文件`result_{timestamp}.csv`。

不传 `-mgf` 时，最终结果按原 `Index` 升序排列。对于 OM 与 OM 同名直接比对，原编号按目标侧 dump 算子的时间戳顺序建立，标杆侧独有的算子随后补入；它不保证与模型图节点排列一致。传入 `-mgf` 才开启图排序，不能仅凭 CSV 的 `Index` 连续判断是否已开启。

指定`-mgf`时，结果先按目标OM图JSON中的`graph`数组顺序，再按各图`op`数组顺序展示，节点`id`不参与排序。CSV第一列`Index`在排序后按每条结果行重新编号，严格为`0、1、2……`，列结构保持不变。算子匹配关系和精度值保持不变：

- 每个图的匹配结果连续排列，图之间不交叉；图内op index指`op`数组下标，不是节点`id`。
- 同一算子的多条Tensor结果保持原相对顺序，但每行的新`Index`不同。编号在整个CSV内连续，不在每个图重新从0开始。
- 部分dump算子未匹配到图名称时，输出告警并将其放在末尾；全部未匹配时，停止本次比对，提示检查`-mgf`是否属于目标OM，以及dump与图JSON中的名称格式，不生成重新编号的结果。
- 节点`id`缺失、重复或大小倒置不影响顺序。名称中的`/`、`.`替换为`_`；普通模式复用dump名称处理规则，去除符合规则的`_lxsliceX`和`_sgt_graph_N`字段；FFTS模式保留这些字段。非法JSON结构、规范化后的无效名称及同图或跨图重名均报错。

该顺序用于沿模型图查看精度差异，不代表多流算子的实际执行时间顺序。适用于OM与OM按编译后名称直接匹配的dump，图侧名称处理遵循目标dump的普通/FFTS模式。若不同图节点经该规则变成同名，工具会报错以避免歧义。

内部`msaccucmp.py`的detail/mapping模式不应用此排序，传入`-mgf`时会提示忽略其排序用途；与融合参数的互斥校验仍然有效。外层参数`-fr/-qfr/-cfr`对应内层`-f/-q/-cf`。

未传`-mgf`时保持原有Index和CSV列结构。传入时，日志仍使用内部原编号，CSV中的Index为重新编号后的展示行号；Advisor中的Index对应最终CSV展示行号。新Index不是`--select/--range`的选择编号。

## 比对结果文件说明

`result_{timestamp}.csv`比对结果文件示例如下：

![offline_data_compare_result_1](../../figures/offline_data_compare_result_1.png)

![offline_data_compare_result_2](../../figures/offline_data_compare_result_2.png)

| 参数 | 说明 |
|------|------|
| Index | 默认表示原比对编号；指定-mgf时表示最终CSV的展示行号，从0逐行连续编号。 |
| OpSequence | 部分算子比对时算子运行的序列，即-fr参数指定的全网层信息文件中算子的ID。 |
| OpType | 算子类型。指定-fr参数时获取算子类型。 |
| NPUDump | 表示My Output模型的算子名。 |
| DataType | 表示NPU Dump侧数据算子的数据类型。 |
| Address | dump Tensor的内存地址。用于判断算子的内存问题。仅基于昇腾AI处理器运行生成的dump数据文件在整网比对时可提取该数据。 |
| GroundTruth | 表示Ground Truth模型的算子名。 |
| DataType | 表示Ground Truth侧数据算子的数据类型。 |
| TensorIndex | 表示基于昇腾AI处理器运行生成的dump数据的算子的input ID和output ID。 |
| Shape | 比对的Tensor的Shape。 |
| CosineSimilarity | 进行余弦相似度算法比对出来的结果，取值范围为[-1,1]，比对的结果如果越接近1，表示两者的值越相近，越接近-1意味着两者的值越相反。 |
| MaxAbsoluteError | 进行最大绝对误差算法比对出来的结果，取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| AccumulatedRelativeError | 进行累积相对误差算法比对出来的结果，取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| RelativeEuclideanDistance | 进行欧氏相对距离算法比对出来的结果，取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| KullbackLeiblerDivergence | 进行KL散度算法比对出来的结果，取值范围为0到无穷大。KL散度越小，真实分布与近似分布之间的匹配越好。 |
| StandardDeviation | 进行标准差算法比对出来的结果，取值范围为0到无穷大。标准差越小，离散度越小，表明越接近平均值。 |
| MeanAbsoluteError | 表示平均绝对误差。取值范围为0到无穷大，MeanAbsoluteError和RootMeanSquareError均趋于0，说明测量值与真实值越近似；MeanAbsoluteError趋于0，RootMeanSquareError越大，说明存在局部过大的异常值；MeanAbsoluteError越大，RootMeanSquareError等于或近似MeanAbsoluteError，说明整体偏差越集中；MeanAbsoluteError越大，RootMeanSquareError越大于MeanAbsoluteError，说明存在整体偏差，且整体偏差分布分散；不存在以上情况的例外情况，因为RootMeanSquareError ≥ MeanAbsoluteError恒成立。 |
| RootMeanSquareError | 表示均方根误差。取值范围为0到无穷大，MeanAbsoluteError趋于0，RootMeanSquareError趋于0，说明测量值与真实值越近似；MeanAbsoluteError趋于0，RootMeanSquareError越大，说明存在局部过大的异常值；MeanAbsoluteError越大，RootMeanSquareError等于或近似MeanAbsoluteError，说明整体偏差越集中；MeanAbsoluteError越大，RootMeanSquareError越大于MeanAbsoluteError，说明存在整体偏差，且整体偏差分布分散；不存在以上情况的例外情况，因为RootMeanSquareError ≥ MeanAbsoluteError恒成立。 |
| MaxRelativeError | 表示最大相对误差。取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| MeanRelativeError | 表示平均相对误差。取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| CompareFailReason | 算子无法比对的原因。<br><br>若余弦相似度为1，则查看该算子的输入或输出Shape是否为空或全部为1，若为空或全部为1则算子的输入或输出为标量，提示：this tensor is scalar。 |

**结果分析**

1. 确认命令是否传入目标 OM 导出的 `-mgf` 图文件，并检查未匹配告警；结合 `NPUDump` 名称核对结果与图节点的顺序。没有 dump 的节点不产生 CSV 行，不能用行号直接对应图节点下标。
2. 先检查 `NPUDump`、`GroundTruth`、`TensorIndex`、`DataType`、`Shape` 和 `CompareFailReason`，排除算子未匹配、Tensor 选错或无法比对的行，再判断数值误差。
3. 沿已确认的图序逐层检查 `CosineSimilarity`、`MaxAbsoluteError` 和相对误差指标，阈值按模型、数据类型和业务要求确定；不要先按误差大小重排而丢失图序。
4. 对首个可疑算子，核对其输入及上游输出：输入已不一致时继续向上游回溯，输入一致而输出异常时再重点检查该算子。结合模型图的数据依赖验证，CSV 中第一条异常行只是定位线索，不等同于根因。
