# 基于torch图模式（torchair）整网算子精度比对

## 简介

torchair图模式整网算子精度比对通过采集torchair图模式下模型中间算子的输入、输出数据，对比两次推理结果是否一致，从而判断模型在不同算子上的精度是否一致。torch图模式（torchair）整网算子精度比对主要支持GE数据与FX数据的比对以及GE开关融合数据的比对。

**基本概念**
参见《[基于torch图模式（torchair）推理场景](../dump/torchair_dump_instruct.md#基本概念)》。

## 使用前准备

**环境准备与通用约束**

参见《[基于torch图模式（torchair）推理场景](../dump/torchair_dump_instruct.md)》中的[环境准备](../dump/torchair_dump_instruct.md#环境准备)与[约束](../dump/torchair_dump_instruct.md#约束)。

**本章节额外约束**

- 两次GE dump或两次FX dump间，请指定不同的dump数据保存路径，否则会导致数据混乱，无法区分的问题，从而影响数据比对和分析。

## GE融合模式（默认） dump数据与FX dump数据精度比对

dump数据采集方式、接口参数、示例以及结果目录结构参见《[基于torch图模式（torchair）推理场景](../dump/torchair_dump_instruct.md)》中的[GE融合模式dump数据](../dump/torchair_dump_instruct.md#ge融合模式dump数据)与[FX模式dump数据](../dump/torchair_dump_instruct.md#fx模式dump数据)。

### compare精度比对

执行 `msprobe compare --target_path <GE_dump_data_path> --golden_path <FX_dump_data_path> [--output_path <output_path>] --mode torchair`，在指定的 `output_path` 路径下输出比对结果csv文件，若不使用 `--output_path` 参数，则默认保存在当前目录下。

```sh
# 使用Ascend Extension for PyTorch 7.1.0及以上版本
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden_path ${dump_path}/msprobe_fx_dump --mode torchair
```

```sh
# 使用Ascend Extension for PyTorch 7.1.0版本
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden_path ${dump_path}/msprobe_fx_dump/data_dump --mode torchair
```

```sh
# 使用Ascend Extension for PyTorch 7.1.0以下版本
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden_path data_dump --mode torchair
```

**注意**：使用Ascend Extension for PyTorch 7.1.0以下版本时，FX模式dump结果件中的token id目录的目录名比实际token id大1，因此在比对时会将token id目录名减1，作为真实token id。

## GE融合模式（默认）dump数据与GE关闭融合模式dump数据精度比对

GE融合与关闭融合的dump采集方法、示例及目录结构参见[GE融合模式dump数据](../dump/torchair_dump_instruct.md#ge融合模式dump数据)与[GE模式关闭融合dump数据](../dump/torchair_dump_instruct.md#ge模式关闭融合dump数据)。

### compare精度比对

执行 `msprobe compare --target_path <GE_dump_data_path> --golden_path <fusion_off_GE_dump_data_path> [--output_path <output_path>] --mode torchair`，在指定的 `output_path` 路径下输出比对结果csv文件，若不使用 `--output_path` 参数，则默认保存在当前目录下。

```sh
msprobe compare --target_path ${dump_path in GE dump}/msprobe_ge_dump --golden_path ${dump_path in fusion off GE dump}/msprobe_ge_dump --mode torchair
```

## 结果查看

精度比对结果的字段含义、判定标准与颜色标记等信息，请参见[附录](#附录)。

## 附录

### 将dump数据转化为指定信息以压缩数据量

> [!note] 说明
>
> 本节仅针对特定用户场景使用，若不涉及该场景请忽略。

dump过程中生成的数据量可能占用大量磁盘空间，可以在dump过程中启用后台进程，将完整的数据提取为指定的信息。以下参考脚本将数据转化为最大最小值，并删除原数据。

```py
#!/bin/env python3
import os
import time
import argparse

surfix = "_min_max"  # Converted data save surfix

# Define how single data is converted
def convert_data_to_info(data):
    return [data.min(), data.max()]

def convert(data_path):
    import numpy as np
    from components.utils.acc_cmp import parse_torchair_dump_data

    npz_surfix, npy_surfix = "{}.npz".format(surfix), "{}.npy".format(surfix)
    for cur_path, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(npy_surfix):  # already converted FX data
                continue

            cur = os.path.join(cur_path, file)
            if file.endswith(".npy"):  # FX saved npy data
                file_name = os.path.splitext(cur)[0]
                np.save(file_name + surfix, convert_data_to_info(np.load(cur)))
                os.remove(cur)
                print("Converted: {} -> {}{}".format(cur, file_name, npy_surfix))
            elif not file.endswith(npz_surfix) and not file.endswith(".txt") and not file.endswith(".swp"):
                inputs, outputs = parse_torchair_dump_data(cur)
                inputs = [convert_data_to_info(ii) for ii in inputs]
                outputs = [convert_data_to_info(ii) for ii in outputs]

                np.savez(cur + npz_surfix, inputs=inputs, outputs=outputs)
                os.remove(cur)
                print("Converted: {} -> {}{}".format(cur, cur, npz_surfix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="GE or FX data dump path")
    args = parser.parse_args()
    while True:
        convert(args.data_path)
        time.sleep(0.5)
        print("Waiting...")
```

在dump过程中后台执行该脚本，将dump数据转化为info数据，以减少内存占用。

```sh
# 将msprobe_ge_dump下的GE dump数据转化为info
python3 convert.py msprobe_ge_dump
```

```sh
# 使用Ascend Extension for PyTorch 7.1.0及以上版本时，将msprobe_fx_dump下的FX dump数据转化为info
python3 convert.py msprobe_fx_dump
```

```sh
# 使用Ascend Extension for PyTorch 7.1.0以下版本时，将data_dump下的FX dump数据转化为info
python3 convert.py data_dump
```

### 比对结果文件格式

torchair场景下的精度比对结果以CSV文件格式输出，包含以下主要列：

#### 基本信息

- **API Name**: 算子或API名称。
- **Stack Info**: 堆栈信息，用于定位代码位置。
- **Data Name**: 数据名称，格式为 [NPU真实数据名，Bench真实数据名]。

#### 真实数据模式指标

当dump数据模式为真实数据时，包含以下指标：

| 指标名称 | 含义 | 正常范围 |
|---------|------|----------|
| Cosine | 余弦相似度，衡量两个向量的方向相似性 | 0.99-1.0 |
| EucDist | 欧氏距离，衡量两个向量的绝对距离 | 越小越好 |
| MaxAbsErr | 最大绝对误差 | 越小越好 |
| MaxRelativeErr | 最大相对误差 | 一般 < 0.01 |
| One Thousandth Err Ratio | 相对误差小于千分之一的比例 | 越高越好 |
| Five Thousandth Err Ratio | 相对误差小于千分之五的比例 | 越高越好 |
| Requires_grad Consistent | 计算梯度是否一致 | True |

#### 统计数据模式指标

当dump数据模式为统计数据时，包含以下指标：

| 指标名称 | 含义 |
|---------|------|
| Max diff | 最大值差异 |
| Min diff | 最小值差异 |
| Mean diff | 平均值差异 |
| L2norm diff | L2范数差异 |
| MaxRelativeErr | 最大相对误差 |
| MinRelativeErr | 最小相对误差 |
| MeanRelativeErr | 平均相对误差 |
| NormRelativeErr | 范数相对误差 |

#### MD5模式指标

当dump数据模式为MD5时，包含以下指标：

| 指标名称 | 含义 |
|---------|------|
| NPU MD5 | NPU数据CRC-32值 |
| BENCH MD5 | 标杆数据CRC-32值 |

#### 结果判定信息

- **Result**: 比对结果（PASS/FAIL）
- **Accuracy Reached or Not**: 计算精度是否达标（Yes/No）
- **Err_message**: 错误信息提示

### 结果判定标准

#### 真实数据模式判定

- **PASS**: Cosine ≥ 0.99且MaxRelativeErr < 0.01
- **FAIL**: Cosine < 0.99或MaxRelativeErr ≥ 0.01

#### 统计数据模式判定

- **PASS**: 各项差异指标在可接受范围内
- **FAIL**: 存在显著差异

#### MD5模式判定

- **PASS**: NPU MD5 == BENCH MD5
- **FAIL**: NPU MD5 != BENCH MD5

### 颜色标记说明

当开启高亮颜色标记功能时：

- **红色**: 表示精度异常，需要重点关注。
- **黄色**: 表示精度可疑，需要进一步分析。
- **绿色**: 表示精度正常。

### 特殊值处理

- **N/A**: 表示无法计算该比对指标值。
- **NaN**: 表示计算结果为非数字，通常由于数据中存在NaN值。
- **inf**: 表示计算结果为无穷大，通常由于除零操作。

当dump数据中存在0或NaN时，比对结果中最大相对误差可能出现inf或NaN的情况，属于正常现象。

### 结果文件位置

比对结果CSV文件默认保存在当前目录，或通过 --output_path参数指定的目录中。
