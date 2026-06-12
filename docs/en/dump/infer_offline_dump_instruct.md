# Inference Offline Model Data Collection

## Overview

Tensor dumping is supported in traditional small model scenarios to obtain precision data for locating precision issues. This functionality applies to ONNX and OM models, requiring only that you specify the offline model corresponding to the original model via parameters.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

The collection of ONNX model data depends on the onnx and onnxruntime packages, and the collection of OM model data depends on the aisbench and aclruntime packages. You can run the following command to install these packages. If the dependency packages are already installed, you can skip this step.

```bash
 msprobe install_deps -m offline [--no_check]
```

> [!NOTE]NOTE
>
> The **no_check** parameter skips the check of the certificate information of the target website for the aisbench and aclruntime packages, which poses security risks. Exercise caution when using this parameter and you shall bear the consequences.

**Constraints**

Only ONNX and OM model data can be collected.

OM model data dumping depends on CANN. You can use the environment variable **ASCEND_TOOLKIT_HOME** to change the CANN path. The default path is **/usr/local/Ascend/cann**.

**Security Warning**

Before loading a model file into the tool, ensure it is secure and reliable. If the official file source provides a SHA256 verification value, you must verify the file to confirm it has not been tampered with.

## Data Collection

**Function**

Collect precision data of an offline model.

**Precautions**

Only ONNX and OM model data can be collected.

**Syntax**

 ```sh
  msprobe offline_dump --model_path <model_path> [options]
 ```

--**model_path** is a mandatory parameter, and **[options]** indicates optional parameters. For details about the parameters, see the table below.

**Parameters**

| Parameter              | Required/Optional| Description                                                        |
| -------------------- | --------- | ------------------------------------------------------------ |
| --model_path         | Required     | Path of the .onnx or .om model file. The path must be an absolute path and must contain the file name. The .onnx and .om files correspond to the ONNX and OM models, respectively.|
| --input_data         | Optional     | Path of model input data. The path must be an absolute path and must contain the file name. The path is randomly generated based on model inputs by default. Separate multiple inputs with commas (,), for example, **/home/input\_0.bin,/home/input\_1.bin,/home/input\_2.npy**. Note: For an AIPP model, the input is the same as that of an OM model, and a .npy file can be automatically converted to a .bin file.|
| -o or --output_path   | Optional     | Output file path. The path must be an absolute path. The default value is the output folder in the current path.|
| --input_shape        | Optional     | Used when the model input is a static shape. Shape information of model inputs. Separate multiple nodes with semicolons (;), for example, **input_name1:1,224,224,3;input_name2:3,300**. By default, this parameter is left blank. **input_name** must be the node name in the network model before model conversion.|
| --rank               | Optional     | Device. The value ranges from 0 to 255, and the default value is **0**.                      |
| --dym_shape_range    | Optional     | Used when the model input is a dynamic shape. It specifies the dynamic shape range. If this parameter is set, inference and precision comparison are performed based on all shape lists in sequence. If a dimension is set to **-1** during model conversion, you need to specify a range for comparison. This dimension cannot be set to **-1** during comparison.<br>The format is as follows: "**input_name1:1,3,200\~224,224-230;input_name2:1,300**".<br>**input_name** must be the node name in the network model before conversion. **\~** indicates the range. **a\~b\~c** indicates [a: b :c]. **-** indicates the value of a digit.|
| --output_size        | Optional     | Sizes of the model outputs. The number of values varies depending on the number of outputs. The default value for each output is **90000000**. If the size of an output exceeds the specified size, set this parameter to correct the size. In the dynamic shape scenario, the output size of a model may be 0. You need to estimate an appropriate value based on the input shape to request memory. Use commas (,) to separate multiple output sizes, for example, **10000,10000,10000**.|
| --onnx_fusion_switch | Optional     | Operator fusion switch of ONNX Runtime. Operator fusion is enabled by default. If dumped ONNX data is missing due to operator fusion, it is recommended to disable this function. This parameter is valid only when **model_path** is set to an ONNX model path. Example: **--onnx_fusion_switch False**|

**Example**

```bash
 msprobe offline_dump --model_path /model_path/model.onnx -o /dump_output_path
```

Or

```bash
 msprobe offline_dump --model_path /model_path/model.om -o /dump_output_path
```

**Output Description**

After the dump is complete, the message "msprobe ends successfully" is displayed. The output result is stored in the configured output path. If no output path is configured, the result is stored in the output folder in the current path by default.

## Output File Description

```sh
{output_path}/{timestamp}/{input_name-input_shape}  # {input_name-input_shape} is used to distinguish the actual inputs of different models in dynamic shape mode. This layer does not exist in static shape mode.
├-- dump_data
│   ├-- npu                          # Directory for storing NPU dump data
│   │   ├-- {timestamp}              # All NPU dump operator outputs of the model
│   │   │   └-- 0                    # Rank ID
│   │   │       └-- {om_model_name}  # Model name
│   │   │           └-- 1            # Model ID
│   │   │               ├-- 0        # Execution sequence number of each task, starting at 0. This value is increased by 1 every dump.
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
│   └-- {onnx} # Path for storing the dump data of the original model. onnx indicates the path for storing the dump data of the ONNX model.
│       ├-- Add_100.0.1682148256368588.npy
│       ├-- ...
│       └-- Where_22.0.1682148253575249.npy
├-- input
│   └-- input_0.bin                          # File of random input data. If input data is specified, this file is not contained.
└-- model
    ├-- {om_model_name}.json                    # JSON file produced by converting the offline model (.om) via the ATC tool.
    └-- new_{onnx_model_name}.onnx              # ONNX model generated after setting each operator as an output node.

```
