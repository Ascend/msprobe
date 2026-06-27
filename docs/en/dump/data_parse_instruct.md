# Data Conversion

## Overview

This document describes how to convert dump data into NumPy (.npy) or PyTorch tensor (.pt) files. The dump data (in .bin format) or adump data in the ATB scenario can be converted for subsequent data analysis and processing.

## Preparations

**Environment Setup**

- Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).
- To convert data into the PyTorch tensor (.pt) format, PyTorch also needs to be installed.
- To convert adump data, ensure that the CANN Toolkit of the matching version has been installed and the CANN environment variables have been configured. For details, see [CANN Software Installation Guide](https://www.hiascend.com/cann/download).

**Constraints**

- The .bin dump data of ATB can be converted.
- The adump data can be converted.

## Function Description

**Function**

This function can convert dump data into NumPy (.npy) or PyTorch tensor (.pt) files.

**Syntax**

```sh
msprobe parse -d <dump_path> [-t <type>] [-o <output_path>]
```

**Parameters**

| Parameter| Mandatory (Yes/No)| Description|
|--------|------|------|
| -d or --dump_path| Yes| Path of the file or directory to be converted. A single file or directory can be input:<br>&#8226; Single file: Specify the file path, including the file name.<br>&#8226; Directory: Specify the directory where the dump file is located.|
| -t or --type| No| Output file type. The following formats are supported:<br>&#8226; npy: The output file is in NumPy (.npy) format.<br>&#8226; pt: The output file is in PyTorch tensor (.pt) format.<br>The default value is **pt**.|
| -o or --output_path| No| Path of the output file. The default value is the **output** folder in the current path.|

**Example (converting a single dump file)**

```sh
msprobe parse -d /path/to/dump_file -o /path/to/output
```

**Example (converting dump files in the entire directory)**

```sh
msprobe parse -d /path/to/dump_file_directory -o /path/to/output
```

**Output Description**

After the preceding commands are executed, a file in the format specified by the **--type** parameter is generated in the path specified by the **--output_path** parameter. If **--dump_path** is set to a single file, only the specified file is converted. If **--dump_path** is set to a directory, all dump files in the directory are converted.
