# Comparison Result Description

## Overview

This document describes how to compare the dump data of a non-quantized model running on the Ascend AI Processor and .npy data of a non-quantized ONNX model. Replace parameters as required.

## Comparison Outputs

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape} is used to distinguish the actual inputs of different models in dynamic shape mode. This layer does not exist in static shape mode.
├-- dump_data
│   ├-- npu                          # Directory for storing NPU dump data
│   │   ├-- {timestamp}             # All NPU dump operator outputs of the model. This directory does not exist when dump is set to False.
│   │   │   └-- 0                    # Rank ID
│   │   │       └-- {om_model_name}  # Model name
│   │   │           └-- 1            # Model ID
│   │   │               ├-- 0        # Execution sequence number of each task ID, starting at 0. This value is increased by 1 every dump.
│   │   │               │   ├-- Add.8.5.1682067845380164
│   │   │               │   ├-- ...
│   │   │               │   └-- Transpose.4.1682148295048447
│   │   │               └-- 1
│   │   │                   ├-- Add.11.4.1682148323212422
│   │   │                   ├-- ...
│   │   │                   └-- Transpose.4.1682148327390978
│   │   ├-- {time_stamp}
│   │   │   ├-- output_0.bin
│   │   │   └-- output_0.npy
│   │   └-- {time_stamp}_summary.json
│   └-- {onnx}        # Path for storing the dump data of an ONNX model.
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- input_Add_100.0.1682148256368588.npy  # For an ONNX model, the input data is dumped and the corresponding input prefix is added.
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                  # Random input data. If the input data is specified, this file does not exist.
├-- model
│   ├-- {om_model_name}.json
│   └-- new_{om_model_name}.onnx     # New ONNX model generated with each operator serving as the output node.
└-- result_{timestamp}.csv           # Comparison result file
```

## Description of Fields in the Comparison Result File

- The comparison result is stored in the `result_{timestamp}.csv` file. The meaning of the comparison result is the same as that of Model Accuracy Analyzer. For details about the meaning of each field, see [Parameters in a Complete Model Comparison Result](<>) in *Model Accuracy Analyzer User Guide (CANN Commercial Edition)*.

* The following table briefly describes the result.

  |                  OpType |  NPUDump | DataType | Address | GroundTruth | DataType | TensorIndex|Shape|Overflow|CosineSimilarity|...|MeanRelativeError|CompareFailReason|IsNpuOps|IsOutputNode|IsPrecisionError|
  |------------------------:|---------:|---------:|--------:|------------:|---------:|-----------:|----:|-------:|---------------:|--:|----------------:|----------------:|-------:|-------:|-------:|
  |                      Sub|Sub_26Mul_28| float16 |    NaN |Sub_26,Mul_28|   float32|Sub_26Mul_28:output:0|[1,1,1,108]|NO|      1|...|         0.000364|                 |NO      |NO      |NO      |

Pay attention to the following items:

 - [x] [NPUDump]: This corresponds to the operator in the OM model. Due to the fusion pattern, it may correspond to multiple GPU/CPU operators.
 - [x] [DataType]: There are two data types: one is the NPU-side data type, and the other is the CPU/GPU-side data type. The two data types are different, which may cause precision drop.
 - [x] [GroundTruth]: ONNX operator corresponding to the OM operator.
 - [x] [Overflow]: Whether data overflow or underflow occurs.
 - [x] [CompareFailReason]: Comparison failure cause. Possible causes include invalid division by zero, which result in NaN and prevent error calculation. Detailed cause information is listed.
 - [x] [IsNpuOps]: Whether the node is an NPU-exclusive node.
 - [x] [IsOutputNode]: Whether the node is an output node of the entire model.
 - [x] [IsPrecisionError]: Whether the node has abnormal precision.
 - [x] [CosineSimilarity]...[MeanRelativeError]: These are the results for various error comparison types. Check whether any item exceeds the precision threshold, which would indicate an exception. See the table below for more details.

  |                  Error Comparison Type|  Description|
  |:------------------------|:---------|
  |CosineSimilarity         |Result of the cosine similarity comparison. The value range is [-1, 1]. A value closer to **1** indicates higher similarity, while a value closer to **-1** indicates lower similarity.|
  |MaxAbsoluteError|Result of the maximum absolute error comparison. The value range is [0, +∞). A value closer to 0 indicates higher similarity, while a larger value indicates greater difference.|
  |AccumulatedRelativeError|Result of the accumulated relative error comparison. The value range is [0, +∞). A value closer to 0 indicates higher similarity, while a larger value indicates greater difference.|
  |RelativeEuclideanDistance|Result of the Euclidean relative distance comparison. The value range is [0, +∞). A value closer to 0 indicates higher similarity, while a larger value indicates greater difference.|
  |KullbackLeiblerDivergence|Result of the Kullback-Leibler divergence comparison. The value range is [0, +∞). The smaller the Kullback-Leibler divergence, the closer the approximate distribution is to the true distribution.|
  |StandardDeviation|Result of the standard deviation comparison. The value range is [0, +∞). The smaller the standard deviation is, the smaller the dispersion is, and the closer the value is to the mean. The mean value and standard deviation of the dump data are displayed in the format of (mean;standard deviation). The first set of data is the result of **My Output**, and the second set is the result of **Ground Truth**.|
  |MeanAbsoluteError|Mean absolute error. The value range is [0, +∞). If values of both `MeanAbsoluteError` and `RootMeanSquareError` are close to 0, the measured value is more approximate to the actual value. If the value of `MeanAbsoluteError` is close to 0, a larger value of `RootMeanSquareError` indicates that some values are excessively large. A larger value of `MeanAbsoluteError` and `RootMeanSquareError` value equal to or approximate to that of `MeanAbsoluteError` indicate that the overall deviation is more centralized. A larger value of `MeanAbsoluteError` and `RootMeanSquareError` value larger than that of `MeanAbsoluteError` indicate that the overall deviation exists and its distribution is scattered. Other situations do not exist because "`RootMeanSquareError` ≥ `MeanAbsoluteError`" is always true.|
  |RootMeanSquareError|Root mean square error. The value range is [0, +∞). If values of both `MeanAbsoluteError` and `RootMeanSquareError` are close to 0, the measured value is more approximate to the actual value. If the value of `MeanAbsoluteError` is close to 0, a larger value of `RootMeanSquareError` indicates that some values are excessively large. A larger value of `MeanAbsoluteError` and `RootMeanSquareError` value equal to or approximate to that of `MeanAbsoluteError` indicate that the overall deviation is more centralized. A larger value of `MeanAbsoluteError` and `RootMeanSquareError` value larger than that of `MeanAbsoluteError` indicate that the overall deviation exists and its distribution is scattered. Other situations do not exist because "`RootMeanSquareError` ≥ `MeanAbsoluteError`" is always true.|
  |MaxRelativeError|Max. relative error. The value range is [0, +∞). A value closer to 0 indicates higher similarity, while a larger value indicates greater difference.|
  |MeanRelativeError|Mean relative error. The value range is [0, +∞). A value closer to 0 indicates higher similarity, while a larger value indicates greater difference.|

## Comparison Result Analysis

- Precision metrics

  | Error Comparison Algorithm               | Standard for Normal Precision  |
  | ------------------------- | ------ |
  | CosineSimilarity          | > 0.99  |
  | RelativeEuclideanDistance | < 0.05  |
  | KullbackLeiblerDivergence | < 0.005 |
  | RootMeanSquareError       | < 1.0   |
  | MeanRelativeError         | < 1.0   |

- To determine whether model precision meets requirements, check whether the network-wide output result meets requirements. If it does, ignore any issues, including operator overflow/underflow issues on intermediate nodes. Otherwise, examine each faulty node individually.
