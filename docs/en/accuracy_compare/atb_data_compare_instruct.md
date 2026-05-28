
# ATB Data Precision Comparison

## Overview

msProbe supports precision comparison in ATB scenarios to help locate precision issues.

**Concepts**

* **Cosine similarity**: cosine of the angle between two non-zero vectors. It can be used to evaluate the similarity between two tensors.

* **Euclidean distance**: absolute distance between two points in a multi-dimensional space. It can be used to evaluate the similarity between two tensors.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Data Preparation**

Obtain the dump data of an ATB model by referring to [Precision Data Collection in ATB](../dump/atb_data_dump_instruct.md).

**Constraints**

Only the precision data of an ATB model collected based on CANN 8.5.0 or later can be compared.

## Quick Start

The following uses a simple example to describe how to use msProbe to compare precision data of an ATB model.

You need to pre-collect the benchmark precision data and the precision data to be compared (data with precision issues) of an ATB model by referring to [Precision Data Collection in ATB](../dump/atb_data_dump_instruct.md). Then, run the following command to compare data precision.

```bash
# Input the actual precision data path.
msprobe compare -m atb -gp golden_data/atb_dump_data/data/0_39943/0/ -tp target_data/atb_dump_data/data/0_276107/0/
```

For details about the command line parameters, see [Parameters](#parameters).

## ATB Precision Data Comparison

### Function

This function can compare the precision of ATB dump data, including real data comparison and statistical data comparison. The comparison result is saved in an Excel file.

**Precautions**

* When collecting the ATB model precision data, if the **task** parameter is set to **tensor** or **all**, the dump data contains the real data of the input and output tensors of operators. Therefore, the real data is used for comparison. If **task** is set to **statistics**, the dump data contains only the statistical data of the input and output tensors of operators, and the statistical data is used for comparison.

* The benchmark data and the data to be compared must be both real data or both statistical data.

* Currently, only tensor data of the bool, int8, int32, int64, bfloat16, float16, and float32 types can be compared in real-data comparison mode.

### Syntax

```bash
msprobe compare -m atb -gp <goldenDataPath> -tp <targetDataPath> [-o <outputPath>]
```

### Parameters

| Parameter| Mandatory (Yes/No)| Description|
| --- | --------- | --- |
| -m or --mode        | Yes| Comparison scenario, which must be **atb**.|
| -gp or --golden_path| Yes| Benchmark data path, which must be specified to the execution round subdirectory. For details about the directory structure of ATB dump data, see [Output Description](../dump/atb_data_dump_instruct.md#output-description) in *Precision Data Collection in ATB*.|
| -tp or --target_path| Yes| Path of data to be compared, which must be specified to the execution round subdirectory. For details about the directory structure of ATB dump data, see [Output Description](../dump/atb_data_dump_instruct.md#output-description) in *Precision Data Collection in ATB*.|
| -o or --output_path | No| Output path of the comparison result. The default value is the **output** directory in the current working directory, which is automatically created by the tool.|

### Example

1. Prepare the benchmark precision data and the precision data to be compared.

    For details, see [Precision Data Collection in ATB](../dump/atb_data_dump_instruct.md). Assume that the collected precision data is stored in the **golden_data/atb_dump_data** and **target_data/atb_dump_data** directories.

2. Run the following command for comparison.

    ```bash
    # Input the actual precision data path.
    msprobe compare -m atb -gp golden_data/atb_dump_data/data/0_39943/0/ -tp target_data/atb_dump_data/data/0_276107/0/
    ```

### Output Description

The output of ATB precision data comparison is an Excel file.

**Output Description of Real Data Comparison**

The table below describes the Excel file obtained after real data precision comparison.

| Column| Description|
| --- | ---- |
| Target Data Name          | Name of the data to be compared, which consists of the op name, op ID, I/O type, and index, for example, **0_WordEmbedding/input.1**.|
| Golden Data Name          | Name of the benchmark data, which consists of the op name, op ID, I/O type, and index, for example, **0_WordEmbedding/input.1**.|
| Target Device and PID     | Device ID and process ID of the data to be compared.|
| Golden Device and PID     | Device ID and process ID of the benchmark data.|
| Target Execution Count    | Op execution round for the data to be compared.|
| Golden Execution Count    | Op execution round for the benchmark data.|
| Target Data Type          | Data type of the data to be compared.|
| Golden Data Type          | Data type of the benchmark data.|
| Target Data Shape         | Data shape of the data to be compared.|
| Golden Data Shape         | Data shape of the benchmark data.|
| Cosine                    | Cosine similarity.|
| Euc Distance              | Euclidean distance.|
| Max Absolute Err          | Max. absolute error.|
| Max Relative Err          | Max. relative error.|
| One Thousandth Err Ratio  | Proportion of elements with relative errors less than one per thousand.|
| Five Thousandth Err Ratio | Proportion of elements with relative errors less than five per thousand.|
| Target Max                | Max. value of all elements in the data to be compared.|
| Golden Max                | Max. value of all elements in the benchmark data.|
| Target Min                | Min. value of all elements in the data to be compared.|
| Golden Min                | Min. value of all elements in the benchmark data.|
| Target Mean               | Mean of all elements in the data to be compared.|
| Golden Mean               | Mean of all elements in the benchmark data.|
| Target Norm               | Norm value of all elements in the data to be compared.|
| Golden Norm               | Norm value of all elements in the benchmark data.|

**Output Description of Statistical Data Comparison**

The table below describes the Excel file obtained after statistical data comparison.

| Column| Description|
| --- | ---- |
| Target Data Name          | Name of the data to be compared, which consists of the op name, op ID, I/O type, and index, for example, **0_WordEmbedding/input.1**.|
| Golden Data Name          | Name of the benchmark data, which consists of the op name, op ID, I/O type, and index, for example, **0_WordEmbedding/input.1**.|
| Target Device and PID     | Device ID and process ID of the data to be compared.|
| Golden Device and PID     | Device ID and process ID of the benchmark data.|
| Target Execution Count    | Op execution round for the data to be compared.|
| Golden Execution Count    | Op execution round for the benchmark data.|
| Target Data Type          | Data type of the data to be compared.|
| Golden Data Type          | Data type of the benchmark data.|
| Target Data Shape         | Data shape of the data to be compared.|
| Golden Data Shape         | Data shape of the benchmark data.|
| Max Diff                  | Max. absolute error.|
| Min Diff                  | Min. absolute error.|
| Mean Diff                 | Mean absolute error.|
| Norm Diff                 | Absolute error of the norm value.|
| Relative Err of Max(%)    | Max. relative error.|
| Relative Err of Min(%)    | Min. relative error.|
| Relative Err of Mean(%)   | Mean relative error.|
| Relative Err of Norm(%)   | Relative error of the norm value.|
| Target Max                | Max. value of all elements in the data to be compared.|
| Golden Max                | Max. value of all elements in the benchmark data.|
| Target Min                | Min. value of all elements in the data to be compared.|
| Golden Min                | Min. value of all elements in the benchmark data.|
| Target Mean               | Mean of all elements in the data to be compared.|
| Golden Mean               | Mean of all elements in the benchmark data.|
| Target Norm               | Norm value of all elements in the data to be compared.|
| Golden Norm               | Norm value of all elements in the benchmark data.|
