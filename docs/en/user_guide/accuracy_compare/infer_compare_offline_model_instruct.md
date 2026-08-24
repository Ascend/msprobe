# One-Click Precision Comparison for Offline Inference Models

## Overview

The one-click precision comparison function for offline inference models automates precision comparison in inference scenarios. It supports ONNX and OM models, allowing you to input the original model along with its corresponding offline model and obtain the comparison results for the entire network. The offline model refers to an OM model converted using the ATC tool.<br>
Additionally, precision comparison of dynamic-shape models and the Artificial Intelligence Pre-Processing (AIPP) function are supported.<br>
**Note**: Ensure that the OM model converted by the ATC tool is compatible with the processor used in the current operating environment.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../../install_guide/msprobe_install_guide.md).

Collecting data for ONNX models depends on the `onnx` and `onnxruntime` packages, while collecting data for OM models depends on the `aisbench` and `aclruntime` packages. Users can install the required dependency packages using the following commands before use. If the dependency packages are already installed, you may skip this step.

```bash
 msprobe install_deps -m offline [--no_check]
```

Note that the `--no_check` parameter skips the certificate information check of the target websites of the `aisbench` and `aclruntime` packages, which poses security risks. Use this command with caution, as you assume full responsibility for any consequences.

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

| Parameter             | Optional/Required | Description                                                         |
| -------------------- | --------- | ------------------------------------------------------------ |
| `-m` or `--mode` | Required | Comparison mode. Must be set to `offline_model`. |
| `-gp` or `--golden_path` | Required | Path to the model file `[.onnx, .om]`, corresponding to ONNX and OM models respectively. |
| `-tp` or `--target_path` | Required | Path to the Ascend AI processor offline model `[.om]`. |
| `--input_data` | Optional | Path to the model input data. Must specify the exact file name. If not specified, data is randomly generated based on the model's inputs. Multiple inputs should be separated by commas, e.g., `/home/input_0.bin,/home/input_1.bin,/home/input_2.npy`. **Note:** When using an AIPP model, this input corresponds to the OM model input, and npy files are automatically converted to bin files. |
| `-o` or `--output_path` | Optional | Output file path. Defaults to the `output` folder in the current directory. |
| `--input_shape` | Optional | Used when the model input has a static shape. Specifies the shape information for model inputs. Default is empty, e.g., `"input_name1:1,224,224,3;input_name2:3,300"`. Use double quotes and separate nodes with semicolons. The `input_name` must match the node name in the original network model before conversion. |
| `--dym_shape_range` | Optional | Used when the model input has a dynamic shape. Specifies the threshold range for dynamic shapes. If set, inference and accuracy comparison are performed sequentially for all shape lists in the parameter. If a dimension was set to `-1` during model conversion, a specific range must be specified for comparison, and the dimension cannot be set to `-1` during comparison.<br/>Format: `"input_name1:1,3,200~224,224-230;input_name2:1,300"`.<br/>Where `input_name` must match the node name in the original network model; `~` denotes a range, `a~b~c` means `[a : b : c]`; `-` denotes a specific value. |
| `--rank` | Optional | Specifies the running device `[0, 255]`. Default is `0`. |
| `--output_size` | Optional | Specifies the output size(s) of the model. Set one value per output. Default is **90000000** for each. If the model output exceeds this size, specify this parameter to correct it. In dynamic shape scenarios, the model's output size may be 0; users should estimate an appropriate value based on the input shape to allocate memory. Multiple output sizes should be separated by commas, e.g., `"10000,10000,10000"`. |
| `--onnx_fusion_switch` | Optional | ONNX Runtime operator fusion switch. Operator fusion is enabled by default. If ONNX dump data is missing due to operator fusion, it is recommended to disable this switch. Usage: `--onnx_fusion_switch False`. |

### Output Description

If the comparison is complete, the message `msprobe compare ends successfully.` is displayed.

```ColdFusion
msprobe compare ends successfully.
```

In the configured output path, the `dump_data`, `input`, and `model` folders and a `.csv` file are generated. The name of the `.csv` file is automatically generated based on the timestamp, in the format of `result_{timestamp}.csv`.

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
