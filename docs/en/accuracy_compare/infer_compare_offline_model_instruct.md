# One-Click Precision Comparison for Offline Inference Models

## Overview

The one-click precision comparison function for offline inference models automates precision comparison in inference scenarios. It supports ONNX and OM models, allowing you to input the original model along with its corresponding offline model and obtain the comparison results for the entire network. The offline model refers to an OM model converted using the ATC tool.<br>
Additionally, precision comparison of dynamic-shape models and the Artificial Intelligence Pre-Processing (AIPP) function are supported.<br>
**Note**: Ensure that the OM model converted by the ATC tool is compatible with the processor used in the current operating environment.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).<br>
The aisbench and aclruntime packages are required for comparing OM models. You can run the following command to install them:<br>

 ```sh
  msprobe install_deps -m offline [--no_check]
 ```

Note that the `--no_check` parameter skips the certificate information check of the target website, which poses security risks. Use this command with caution, as you assume full responsibility for any consequences.

**Constraints**

Only ONNX and OM models can be compared.

One-click precision comparison depends on CANN. You can use the environment variable `ASCEND_TOOLKIT_HOME` to modify the CANN path. The default path is `/usr/local/Ascend/cann`.

**Security Warning**

Before loading a model file into the tool, ensure it is secure and reliable. If the official file source provides a SHA256 verification value, you must verify the file to confirm it has not been tampered with.

## Function Description

### Function

The command-line tool is used to perform one-click comparison on offline models by simply inputting a model—no advance data collection required, and the comparison result will be output.

### Precautions

Only ONNX and OM models can be compared.

One-click precision comparison depends on CANN. You can use the environment variable `ASCEND_TOOLKIT_HOME` to modify the CANN path. The default path is `/usr/local/Ascend/cann`.

### Syntax

 ```sh
  msprobe compare -m offline_model -gp /golden_path/golden_model.onnx -tp /target_path/target_path.om -o /compare_output_path
 ```

### Parameters

| Parameter                | Description                                                                                                                                                                                                                                                                  | Mandatory (Yes/No)|
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| `-m` or `--mode`           | Comparison mode, which needs to be set to `offline_model`.                                                                                                                                                                                                                                            | Yes   |
| `-gp` or `--golden_path`   | Path of the model file (`.onnx` or `.om`), which corresponds to the ONNX or OM model, respectively.                                                                                                                                                                                                                                   | Yes   |
| `-tp` or `--target_path`   | Path of the offline model (`.om`) running on the Ascend AI Processor.                                                                                                                                                                                                                                                | Yes   |
| `--input_data`        | Path of the input data of the model. The path must be specified to a specific file name. By default, the value is randomly generated based on the model input. Use commas (,) to separate multiple inputs, for example, `/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy`. Note: For an AIPP model, the input is the same as that of an OM model, and a .npy file can be automatically converted to a .bin file.                                                                                                          | No   |
| `-o` or `--output_path`    | Output file path. The default value is the `output` folder in the current path.                                                                                                                                                                                                                                           | No   |
| --input_shape        | Used when the model input is a static shape. This parameter specifies the shape information of the model input, for example, `"input_name1:1,224,224,3;input_name2:3,300"`. Enclose the value in double quotation marks and separate multiple nodes with semicolons (;). The default value is empty. `input_name` must be the node name in the network model before model conversion.                                                                                                                                     | No   |
| `--dym_shape_range`   | Used when the model input is a dynamic shape. It specifies the dynamic shape range. If this parameter is set, inference and precision comparison are performed based on all shape lists in sequence. If a dimension is set to `-1` during model conversion, you need to specify a range for comparison. This dimension cannot be set to `-1` during comparison.<br>The format is as follows: `"input_name1:1,3,200\~224,224-230;input_name2:1,300"`.<br>`input_name` must be the node name in the network model before conversion. `\~` indicates the range. `a\~b\~c` indicates `[a: b :c]`. `-` indicates the value of a digit.<br>| No   |
| --rank               | Device. The value ranges from 0 to 255, and the default value is **0**.                                                                                                                                                                                                                                             | No   |
| `--output_size`       | Sizes of the model outputs. The number of values varies depending on the number of outputs. The default value for each output is **90000000**. If the size of an output exceeds the specified size, set this parameter to correct the size. In the dynamic shape scenario, the output size of a model may be 0. You need to estimate an appropriate value based on the input shape to request memory. Use commas (,) to separate multiple output sizes, for example, **10000,10000,10000**.                                                                                                      | No   |
| `--onnx_fusion_switch`| Operator fusion switch of ONNX Runtime. Operator fusion is enabled by default. If dumped ONNX data is missing due to operator fusion, it is recommended to disable this function. Example: `--onnx_fusion_switch False`                                                                                                                                                                       | No   |

### Output Description

If the comparison is complete, the message `msprobe compare ends successfully.` is displayed.
In the configured output path, the `dump_data`, `input`, and `model` folders and a .csv file are generated. The name of the .csv file is automatically generated based on the timestamp, in the format of `result_{timestamp}.csv`.

## Output File Description

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
│   └-- {onnx} # Path for storing the dump data of the original model. onnx corresponds to an ONNX model.
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- input_Add_100.0.1682148256368588.npy  # For an ONNX model, the input data is dumped and the corresponding input prefix is added.
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # File of random input data. If input data is specified, this file is not contained.
├-- model
│   ├-- {om_model_name}.json                    # JSON file converted from the offline model (.om) using the ATC tool.
│   └-- new_{onnx_model_name}.onnx              # New ONNX model generated with each operator serving as the output node.
└-- result_{timestamp}.csv                   # Comparison result file
```

### Viewing the Comparison Result

See [Comparison Result Description](infer_compare_result.md).
