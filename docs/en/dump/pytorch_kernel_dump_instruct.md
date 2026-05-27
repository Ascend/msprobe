# Kernel-Level Precision Data Collection in PyTorch

## Overview

This document describes the configuration examples and collection results of kernel-level precision data collection. For details about how to use the msProbe data collection function, see [Precision Data Collection in PyTorch](./pytorch_data_dump_instruct.md).

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

Only the PyTorch framework is supported.

## Quick Start

See [PyTorch Precision Data Collection - Quick Start](./pytorch_data_dump_instruct.md#quick-start).

## Kernel Dump Configuration

When using kernel dump, set **task** to **tensor** and **list** to an API name. Currently, kernel dump can collect data of only one API in each step.

The API naming format is `{api_type}.{api_name}.{Number of API calls}.{forward/backward}`. For details, see the API name in the L1 dump result file **dump.json**.

Example:

```json
{
    "task": "tensor",
    "dump_path": "./dump_path",
    "level": "L2",
    "rank": [],
    "step": [],
    "tensor": {
        "scope": [],
        "list": ["Functional.linear.0.backward"]
    }
}
```

## Dump Result File

### Data Collection Results

If the kernel-level data collection is successful, the following information is displayed:

```bash
The kernel data of {api_name} is dumped successfully.
```

Note: If no data is generated after the preceding information is printed, rectify the fault by referring to [FAQs](#faqs).

If the kernel dump encounters an unsupported API, the following information is displayed:

```bash
The kernel dump does not support the {api_name} API.
```

**{api_name}** indicates the API name.

### Output File Description

After kernel-level data collection is successful, the following files are generated in the specified **dump_path** directory:

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

## Appendix

### FAQs

#### Why is the collection result file empty?

1. Check whether the tool usage, configuration file content, and API name format in **list** are correct.
2. Check whether the API is running on the Ascend NPU. If the API is running on other devices, no kernel-level data exists.
3. If the problem persists, use the **torch_npu.npu** API provided by the [Ascend Extension for PyTorch plugin](https://gitcode.com/Ascend/pytorch) to collect kernel-level data. The kernel dump function of the tool is implemented based on the **init_dump**, **set_dump**, and **finalize_dump** sub-interfaces. For details about the **torch_npu.npu** API, see [torch_npu.npu](<>).
