
# Precision Data Collection in ATB

## Overview

msProbe collects precision data of an ATB model by executing the ATB dump module loading script during model running.

**Note**:

* The precision data collection requires I/O operations such as copying data from the NPU memory to the host memory and writing data from the host memory to the drive, which slows down ATB model running. The impact on model performance depends on the size of the collected data and the I/O performance of the environment.

* When precision data is collected in real data mode, the tool directly saves the input and output tensors of ops, resulting in output files that consume a large amount of drive space.

* Before executing the ATB dump module loading script, ensure that the ATB model operating environment is ready, including the installation and enabling of the CANN Toolkit and CANN NNAL packages.

**Concepts**

* **Ascend Transformer Boost (ATB)**: An efficient and reliable acceleration library designed for Transformer models based on the Ascend Ascend AI Processor. For details, see [ATB User Guide (CANN Commercial Edition)](<>).

* **dump**: a process of collecting precision data and completing data persistence.

* **Operation**: ATB native operator. An ATB model consists of multiple operations, and each operation can contain one or more other operations. The outermost operation is also called a layer-level operation, such as WordEmbedding, Prefill_layer, Decoder_layer, and LmHead.

* **Kernel**: bottom-layer operator called by an ATB operation. The name usually ends with "Kernel", for example, RmsNormKernel and AddBF16Kernel.

* **op**: execution operator in a broad sense. It includes layer-level operations, non-layer-level operations, and kernels.

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Note**: Currently, msProbe with the ATB dump module can be [compiled and installed from source code](../msprobe_install_guide.md#compilation-and-installation). During compilation, you need to use the `--include-mod` parameter to specify the **atb_probe** module.

**Constraints**

Only CANN 8.3.RC1 and later versions are supported.

## Quick Start

The following uses a simple example to describe how to use msProbe to collect precision data of an ATB model.

1. Create a configuration file.

    Create a `config.json` file in the current directory to configure dump parameters. The content is as follows:

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "all",
        "ids": "0",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    For details about the dump parameters, see [Parameters](#parameters).

2. Execution commands.

    Run `pip show mindstudio-probe` to determine the installation path of msProbe. Assume that the installation path is **/usr/local/lib/python3.11/site-packages**. Run the following command to load the dump module:

    ```bash
    MSPROBE_HOME_PATH=/usr/local/lib/python3.11/site-packages
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh --output=$PWD --config=$PWD/config.json
    ```

    For details about the command line parameters, see [Parameters](#parameters).

3. Run an ATB model.

    The following uses the MindIE image for pure model inference as an example.

    ```bash
    cd $ATB_SPEED_HOME_PATH
    # Prepare the model weight file and pass the actual weight path. Ensure that the weight file is secure and reliable.
    python examples/run_pa.py --model_path /path-to-weights
    ```

    The precision data generated during model execution is saved in the **atb_dump_data** directory in the path specified by `--output`.

4. Uninstall the ATB dump module.

    After collecting the precision data, run the following command to uninstall the dump module:

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/unload_atb_probe.sh
    ```

## ATB Dump Function

### Overview

The ATB dump function is used to collect precision data during ATB model running, including the model structure information and the actual data or statistical data of the input and output tensors. In addition, the dump configuration file can be modified during model running to implement dynamic dump.

**Precautions**

* To reduce the performance overhead caused by the dump module parsing the configuration file, the module remains in a sleep state until the **dump_enable** parameter in the configuration file is parsed as **true**. The configuration file is read only after a certain number of ops are executed. As a result, after the value of **dump_enable** is changed to **true** for the first time, the precision data of some ops may be lost. Therefore, it is advised to wake up the dump module before collecting precision data. For details, see [step 4 in "Usage Example"](#usage-example). Or, set the value of **dump_enable** to **true** before model running to directly enter the debugging state.

* After the dump module enters the debugging state, the configuration file is parsed every 5 seconds. Therefore, the modification of the configuration file may take effect 5 seconds later.

### Syntax

Load the ATB dump module:

```bash
source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh [--output=<outputPath>] [--config=<configPath>]
```

Uninstall the ATB dump module:

```bash
source $MSPROBE_HOME_PATH/msprobe/scripts/atb/unload_atb_probe.sh
```

`$MSPROBE_HOME_PATH` indicates the msProbe installation path, which can be obtained by running the `pip show mindstudio-probe` command.

### Parameters

**Command-Line Parameters**

| Parameter| Mandatory (Yes/No)| Description|
| --- | --------- | --- |
| --output | No| Specifies the output path of dump data. The default value is the current working directory.|
| --config | No| Specifies the dump configuration file path. The default value is the path of the `config.json` file in the same directory as the `load_atb_probe.sh` script. The dump configuration file can be created when data needs to be collected.|

**Parameters in the Dump Configuration File**

The dump configuration file is a text file in JSON format. The table below describes its parameters.

| Parameter| Mandatory (Yes/No)| Description|
| --- | --------- | --- |
| task         | No| Dump task; string; defaults to **tensor**. Possible values:<br> **tensor**: collects the actual data of the input/output tensor of an op.<br> **statistics**: collects the statistics of the input/output tensor of an op.<br> **all**: collects the actual data and statistics of the input/output tensor of an op.|
| dump_enable  | No| Whether to allow data dump; bool; defaults to **false**. Possible values:<br> **true**: allows the collection of the actual data or statistics of the input/output tensor of an op.<br> **false**: does not allow the collection of the actual data or statistics of the input/output tensor of an op.|
| exec_range   | No| Execution round of an op whose data is to be dumped; str; defaults to **0,0**. Possible values:<br> **all**: dumps the precision data of all execution rounds of an op.<br> **none**: does not dump the precision data of all execution rounds of an op.<br> **\<start round\>,\<end round\>**: dumps the op's precision data from the start round to the end round, inclusive.<br> Example: **"exec_range": "0,2"**, indicating that the precision data of the first, second, and third execution rounds of an op is dumped. (Note: the *N*th execution round is numbered *N* – 1.)|
| ids          | No| ID of an op whose precision data is to be dumped; string type. The default value is **""**, indicating that the precision data of all layer-level operations is dumped. The value must be in the "\<ID1\>,\<ID2\>" format. One or more IDs can be specified.<br> Example:<br> **"ids": "0"**: dumps the precision data of the op with ID 0.<br> **"ids": "2_1"**: dumps the precision data of the op with ID 1 under the op with ID 2.<br> **"ids": "0,2_1"**: dumps the precision data of the op with ID 0 and the op with ID 1 under the op with ID 2.|
| op_name      | No| Name of an op whose precision data is to be dumped; string type. The default value is **""**, indicating that the precision data of all layer-level operations is dumped. The value must be in the "\<opName1\>,\<opName2\>" format. One or more op names are supported.<br> Example:<br> **"op_name": "word"**: dumps the precision data of the op whose name starts with "word" (case-insensitive).|
| save_child   | No| Whether to dump the precision data of the sub-ops of an op. The value is of the bool type. The default value is **false**. Possible values:<br> **true**: dumps the precision data of the specified op and its internal sub-ops.<br> **false**: dumps only the precision data of the specified op.|
| device       | No| ID of the device whose data is to be dumped. The value is of the str type. The default value is **""**, indicating that the precision data of all devices is dumped. The value must be in the "\<deviceID1\>,\<deviceID2\>" format. One or more device IDs can be specified.<br> Example:<br> **"device": "0"**: dumps the precision data of device 0.|
| filter_level | No| Filtering level of the actual data of the input/output tensor of the dumped op. The value is of the int type. The default value is **1**. This parameter is valid only when the layer-level operation is specified and **save_child** is set to **true**. Possible values:<br> **0**: does not filter data when collecting the actual data of the input/output tensor of an op.<br> **1**: saves the same tensor only once when collecting the actual data of the input/output tensor of an op.<br> **2**: filters the input/output tensor of a kernel using a condition based on the value **1**.|

The following is an example of the dump configuration file:

```json
{
    "task": "tensor",
    "dump_enable": false,
    "exec_range": "all",
    "ids": "",
    "op_name": "",
    "save_child": false,
    "device": "",
    "filter_level": 1
}
```

### Usage Example

The following uses the MindIE image for serving inference as an example. Ensure that the inference service can be started in the current environment and msProbe is correctly installed.

1. Run the following command to query msProbe installation path.

    ```bash
    pip show mindstudio-probe
    ```

    If the installation path is **/usr/local/lib/python3.11/site-packages**, run the following command to save the installation path as an environment variable:

    ```bash
    export MSPROBE_HOME_PATH=/usr/local/lib/python3.11/site-packages
    ```

2. Run the following command to execute the dump module loading script.

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh --output=$PWD --config=$PWD/config.json
    ```

3. Run the following command to start the inference service.

    ```bash
    /usr/local/Ascend/mindie/latest/mindie-service/bin/mindieservice_daemon
    ```

    If "Daemon start success!" is displayed, the service is started successfully.

4. Wake up the ATB dump module.

    Create a dump configuration file in the **$PWD/config.json** directory. The file content is as follows:

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "none",
        "ids": "",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    In the configuration file, set **dump_enable** to **true** and **exec_range** to **none** to avoid dumping unnecessary precision data during the wakeup process.

    Then, send an inference request on the request terminal. The following is an example request:

    ```bash
    curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"model": "QWen2.5_7B", "messages": [{"role": "system", "content":{"type": "text", "text": "You are a helpful assistant"}}], "max_tokens": 20}' http://127.0.0.1:1025/v1/chat/completions
    ```

    Change the value of **model**, IP address, and port number as required.

    If `"Ready to dump ATB data, the running speed of the model will be affected"` is displayed on the service terminal, the dump module is woken up and enters the debugging state. You can also check whether the dump module is woken up by checking whether the **$PWD/atb_dump_data/data** directory is created.

5. Dump the model precision data.

    Modify the dump configuration file as follows:

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "all",
        "ids": "2",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    Five seconds after the configuration file is modified and saved, send an inference request from the request terminal. The dump module then collects precision data during model execution as the inference service processes the request.

    **Note**: After the request terminal receives a response, it only indicates that model execution is complete, but does not indicate that precision data collection is complete. Observe the size of the **$PWD/atb_dump_data** folder until its size does not increase.

6. Stop the inference service and uninstall the dump module.

    After collecting the required precision data, stop the inference service on the service terminal and uninstall the dump module by running the following command.

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/unload_atb_probe.sh
    ```

### Output Description

The following is an example of the directory structure for the ATB dump output:

```lua
├── outputPath  # Output path when the module loading script is executed
│   ├── atb_dump_data  # Fixed directory automatically created by the tool
│   |   ├── data  # Fixed directory automatically created by the tool, which stores the input/output data of an op.
│   |   │   ├── 0_39943  # Device ID and process ID
|   |   |   |    ├── 0  # Op execution round
|   |   |   |    |   ├── 0_WordEmbedding  # Layer-level operation ID and name
|   |   |   |    |   |   ├── 0_GatherOperation  # Non-layer-level operation ID and name
|   |   |   |    |   |   |   ├── 0_Gather16I64Kernel  # Kernel ID and name
|   |   |   |    |   |   |   |   ├── after  # Stores the output tensor.
|   |   |   |    |   |   |   |   |   |── outtensor0.bin
|   |   |   |    |   |   |   |   ├── before  # Stores the input tensor.
|   |   |   |    |   |   |   |   |   |── intensor0.bin
|   |   |   |    |   |   |   |   |   |── intensor1.bin
|   |   |   |    |   |   |   ├── after
|   |   |   |    |   |   |   └── ...
|   |   |   |    |   |   |   ├── before
|   |   |   |    |   |   |   └── ...
|   |   |   |    |   |   |   ├── op_param.json  # Operation parameter data
|   |   |   |    |   |   |...
|   |   |   |    |   ...
|   |   |   |    |   ├── 5_Prefill_layer
|   |   |   |    |   |   └── ...
|   |   |   |    |   ...
|   |   |   |    |   ├── statistic.csv  # Statistics on the op's input/output tensor
|   |   |   |    ├── 1  # Op execution round
|   |   |   |    |   └── ...
|   |   |   |    ...
│   |   │   ├── 1_39945  # Device ID and process ID
|   |   |   |   └── ...
|   |   |   ...
│   |   ├── info  # Fixed directory automatically created by the tool, which stores structure information.
│   |   |   ├── layer  # Fixed directory automatically created by the tool, which stores layer-level operation structure information.
│   |   │   |   ├── 0_39943  # Device ID and process ID
│   |   │   |   |   ├── WordEmbedding_0.json
|   |   |   |   |   ...
|   |   |   |   ...
│   |   |   ├── model  # Fixed directory automatically created by the tool, which stores model structure information.
│   |   │   |   ├── 0_39943  # Device ID and process ID
│   |   │   |   |   ├── DecoderModel_Decoder.json
│   |   │   |   |   ├── DecoderModel_Prefill.json
|   |   |   |   ...
```

The **statistic.csv** file is generated when model precision data is collected, regardless of the value of **task** in the configuration file. The table below describes the details of the .csv file.

| Column| Description|
| --- | ---- |
| Device and PID  | Device ID and process ID|
| Execution Count | Execution round|
| Op Name         | op name, which consists of the parent op name and its own name, for example, Prefill_layer/Attention.|
| Op Type         | op type|
| Op Id           | op ID, which consists of the parent op ID and its own ID, for example, 2_0.|
| Input/Output    | Input or output tensor|
| Index           | Tensor index|
| Dtype           | Tensor data type, for example, bf16.|
| Format          | Tensor data format, for example, nd.|
| Shape           | Tensor shape, for example, 36 × 128.|
| Max             | Maximum value of all tensor elements. If **task** is set to **tensor**, the value is fixed at **N/A**.|
| Min             | Minimum value of all tensor elements. If **task** is set to **tensor**, the value is fixed at **N/A**.|
| Mean            | Mean of all tensor elements. If **task** is set to **tensor**, the value is fixed at **N/A**.|
| Norm            | Norm value of all tensor elements. If **task** is set to **tensor**, the value is fixed at **N/A**.|
| Tensor Path     | Path for storing tensor data. If **task** is set to **statistics**, the value is fixed at **N/A**.|
