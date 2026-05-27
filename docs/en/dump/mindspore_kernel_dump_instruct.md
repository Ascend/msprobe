# Kernel Precision Data Collection in MindSpore Dynamic Graph Scenario

## Overview

This document describes the configuration examples and collection results of kernel dump data. For details about how to use the msProbe data collection function, see [Precision Data Collection in MindSpore](./mindspore_data_dump_instruct.md).

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

- Only the MindSpore framework is supported.
- When the msProbe data collection function is used, if **level** is set to **L2**, operator data at the kernel level is collected. This is supported only on the Ascend NPU platform.

## Usage Example

1. **Configuration file**

   When kernel dump is used, the **list** parameter in the **config.json** file must be set to an API name. Currently, kernel dump can collect data of only one API in each step.
   The API naming format is `{api_type}.{api_name}.{Number of API calls}.{forward/backward}`. For details, see the API name in [dump.json](./mindspore_data_dump_instruct.md).

   ```json
   {
       "task": "tensor",
       "dump_path": "/home/data_dump",
       "level": "L2",
       "rank": [],
       "step": [],
       "tensor": {
           "scope": [],
           "list": ["Functional.linear.0.backward"]
       }
   }
   ```

2. **Model script**

   Configure the tool enablement code in the model script by referring to "Creating a Model Script" in [Quick Start] (mindspore_dump_quick_start.md).

3. **Training script execution**

   Run the following command to execute the training script:

   ```bash
   python train.py
   ```
   
4. **Output description**

   If the API kernel-level data collection is successful, the following information is displayed:

   ```bash
   The kernel data of {api_name} is dumped successfully.
   ```

   Note: If no data is generated after the preceding information is printed, rectify the fault by referring to [FAQs](#faqs).

   If the kernel dump encounters an unsupported API, the following information is displayed:

   ```bash
   The kernel dump does not support the {api_name} API.
   ```

   *{api_name}* indicates the name of the API where the overflow/underflow was observed.

## Dump Result File

After kernel dump data is successfully collected, the following files are generated in the specified **dump_path** directory:

```ColdFusion
├── /home/data_dump/
│   ├── step0
│ │ ├── 20241201103000        # Date and time format, indicating 10:30:00, 2024-12-01.
│   │   │   ├── 0             # Device ID
│   │   │   │   ├──{op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}    # Kernel-layer operator data
│   │   │  ...
│   │   ├── kernel_config_{device_id}.json    # Intermediate file generated during kernel dump API call. Generally, you do not need to pay attention to it.
│   │  ...     
│   ├── step1
│  ...
```

## FAQs

Q: Why is the collection result file empty?

A: Perform the following steps:

Check whether the tool usage, configuration file content, and API name format in **list** are correct.
Check whether the API is running on the Ascend NPU. If the API is running on other devices, no kernel-level data exists.
