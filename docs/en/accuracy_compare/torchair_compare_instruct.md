# Network-wide Operator Precision Comparison in TorchAir Mode

## Overview

In TorchAir mode (Torch graph mode), the input and output data of intermediate operators of a model is collected to check whether two inference results are consistent, thereby determining whether model precision is consistent on different operators. In this mode, network-wide operator precision comparison supports the comparison between GE data and FX data and data comparison in GE fusion mode.

**Concepts**
For details, see [Inference in TorchAir Mode](../dump/torchair_dump_instruct.md#basic-concepts).

## Preparations

**Environment Setup and General Constraints**

For details, see [Environment Setup](../dump/torchair_dump_instruct.md#environment-setup) and [Constraints](../dump/torchair_dump_instruct.md#constraints) of inference in TorchAir Mode.

**Additional Constraints**

- Specify different dump paths for two GE dump operations or two FX dump operations. Otherwise, data may be disordered and cannot be distinguished, affecting data comparison and analysis.

## Precision Comparison Between Dump Data in GE Fusion Mode (Default) and FX Dump Data

For details about the dump data collection method, API parameters, examples, and result directory structure, see [Dumping Data in GE Fusion Mode](../dump/torchair_dump_instruct.md#dumping-data-in-ge-fusion-mode) and [Dumping Data in FX Mode](../dump/torchair_dump_instruct.md#dumping-data-in-fx-mode).

### Precision Comparison

Run the `msprobe compare --target_path <GE_dump_data_path> --golden_path <FX_dump_data_path> [--output_path <output_path>] --mode torchair` command to generate the comparison result in CSV format in the path specified by `output_path`. If `--output_path` is not used, the result is saved in the current directory by default.

```sh
# Use Ascend Extension for PyTorch 7.1.0 or later
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden_path ${dump_path}/msprobe_fx_dump --mode torchair
```

```sh
# Use Ascend Extension for PyTorch 7.1.0.
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden_path ${dump_path}/msprobe_fx_dump/data_dump --mode torchair
```

```sh
# Use a version earlier than Ascend Extension for PyTorch 7.1.0.
msprobe compare --target_path ${dump_path}/msprobe_ge_dump --golden_path data_dump --mode torchair
```

**Note**: When a version earlier than Ascend Extension for PyTorch 7.1.0 is used, the name of the token ID directory in the dump result file in FX mode is one greater than the actual token ID. Therefore, during comparison, the directory name is decremented by one to obtain the correct token ID.

## Precision Comparison Between Dump Data in GE Fusion Mode (Default) and in GE Mode with Fusion Disabled

For details about the dump collection methods, examples, and directory structures for GE fusion and non-fusion modes, see [Dumping Data in GE Fusion Mode](../dump/torchair_dump_instruct.md#dumping-data-in-ge-fusion-mode) and [Dumping Data in GE Mode with Fusion Disabled](../dump/torchair_dump_instruct.md#dumping-data-in-ge-mode-with-fusion-disabled).

### Precision Comparison

Run the `msprobe compare --target_path <GE_dump_data_path> --golden_path <fusion_off_GE_dump_data_path> [--output_path <output_path>] --mode torchair` command to generate the comparison result in CSV format in the path specified by `output_path`. If `--output_path` is not used, the result is saved in the current directory by default.

```sh
msprobe compare --target_path ${dump_path in GE dump}/msprobe_ge_dump --golden_path ${dump_path in fusion off GE dump}/msprobe_ge_dump --mode torchair
```

## Result Viewing

For details about the fields, determination criteria, and color marks in the accuracy comparison result, see [Appendix](#appendix).

## Appendix

### Converting Dump Data into Targeted Information to Reduce Volume

> [!note] Note
>
> This section applies only to specific scenarios.

The data generated during the dump process may occupy a large amount of drive space. You can enable a background process during dump to extract complete data as specified information. The following script demonstrates how to convert the data into the maximum and minimum values and delete the original data.

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

Execute this script in the background during the dump process to convert the dump data into info, thereby reducing memory usage.

```sh
# Convert the GE dump data in msprobe_ge_dump into info.
python3 convert.py msprobe_ge_dump
```

```sh
# If Ascend Extension for PyTorch 7.1.0 or later is used, convert the FX dump data in msprobe_fx_dump into info.
python3 convert.py msprobe_fx_dump
```

```sh
# If an Ascend Extension for PyTorch version earlier than 7.1.0 is used, convert the FX dump data in data_dump into info.
python3 convert.py data_dump
```

### Comparison Result File Format

The precision comparison result in the TorchAir scenario is output in CSV format, including the following information:

#### Basic Information

- **API Name**: operator or API name.
- **Stack Info**: stack information, which is used to locate the code.
- **Data Name**: data name, in the format of [NPU data name, Bench data name].

#### Metrics in Real Data Mode

The table below lists metrics of dump data in real data mode.

| Metric| Meaning| Normal Range|
|---------|------|----------|
| Cosine | Cosine that measures the similarity between the directions of two vectors.| 0.99-1.0 |
| EucDist | Euclidean distance that measures the absolute distance between two vectors.| The smaller, the better.|
| MaxAbsErr | Maximum absolute error.| The smaller, the better.|
| MaxRelativeErr | Maximum relative error.| < 0.01 (generally)|
| One Thousandth Err Ratio | Proportion of elements with relative errors less than one per thousand.| The higher, the better.|
| Five Thousandth Err Ratio | Proportion of elements with relative errors less than five per thousand.| The higher, the better.|
| Requires_grad Consistent | Checks whether gradients are consistent.| True |

#### Metrics in Statistics Mode

The table below lists metrics of dump data in statistics mode.

| Metric| Meaning|
|---------|------|
| Max diff | Maximum value difference|
| Min diff | Minimum value difference|
| Mean diff | Mean difference|
| L2norm diff | L2 norm difference|
| MaxRelativeErr | Maximum relative error|
| MinRelativeErr | Minimum relative error|
| MeanRelativeErr | Mean relative error|
| NormRelativeErr | Norm relative error|

#### Metrics in MD5 Mode

The table below lists metrics of dump data in MD5 mode.

| Metric| Meaning|
|---------|------|
| NPU MD5 | CRC-32 value of NPU data|
| BENCH MD5 | CRC-32 value of benchmark data|

#### Result Information

- **Result**: comparison result (PASS/FAIL)
- **Accuracy Reached or Not**: precision meets the requirement or not (Yes/No)
- **Err_message**: error message

### Result Determination Criteria

#### Determination in Real Data Mode

- **PASS**: Cosine ≥ 0.99 and MaxRelativeErr < 0.01
- **FAIL**: Cosine < 0.99 or MaxRelativeErr ≥ 0.01

#### Determination in Statistics Mode

- **PASS**: difference metrics are within the acceptable range.
- **FAIL**: significant differences exist.

#### Determination in MD5 Mode

- **PASS**: NPU MD5 == BENCH MD5
- **FAIL**: NPU MD5 != BENCH MD5

### Color Marks

When highlight colors are used:

- **Red**: abnormal precision, requiring special attention.
- **Yellow**: suspicious precision, requiring further analysis.
- **Green**: normal precision.

### Special Value Processing

- **N/A**: Metric value cannot be calculated.
- **NaN**: The calculation result is not a number, usually because NaN values exist.
- **inf**: The calculation result is infinite, usually due to division by zero.

If the dump data contains 0 or NaN, the maximum relative error in the comparison result may be **inf** or **NaN**, which is normal.

### Result File Location

The comparison result CSV file is saved in the current directory by default or the directory specified by the `--output_path` parameter.
