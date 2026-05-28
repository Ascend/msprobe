# Configuration Check Before Training

## Overview

This tool is used to compare the configurations that may affect training precision in the two environments, including:

- Environment variables
- Third-party library versions
- Training hyperparameters
- Weights
- Datasets
- Random operations

**Tool Usage Process**

1. Prepare two training servers.

2. Install msProbe.

3. Collect data on the two servers.

   You can select the static or dynamic collection mode.

   - Static data collection: Start collection using the CLI. Only environment variables, third-party library versions, and training hyperparameters can be collected.
   - Dynamic data collection: Add an API to the training script to start collection. Environment variables, third-party library versions, training hyperparameters, weights, datasets, and random operations can be collected.

4. Compare data.

5. Analyze the result.

   Check whether attributes in the comparison result pass the check based on the [Result File Description](#Result File Description).

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](./msprobe_install_guide.md).

**Constraints**

The PyTorch and MindSpore frameworks are supported.

## Data Collection

### Static Data Collection

**Description**

Use the CLI to collect data. Only environment variables, third-party library versions, and training hyperparameters can be collected.

By default, environment variables and third-party library versions are collected. To collect training hyperparameters, you need to pass the shell script or YAML configuration file for starting training.

**Precautions**

The static data collection mode can obtain only environment variables in the system. The hyperparameters parsed in the shell script do not support data restoration of complex operations. In this case, you are advised to use [dynamic collection mode](#dynamic data collection).

**Syntax**

```bash
msprobe config_check -d [<*.sh> <*.yaml>] [-o <output_file_path>]
```

**Parameters**

| Parameter           | Required/Optional| Description|
|---------------|-------|----------------------------------------------|
| **-d** or **--dump**  | Required | Data collection mode. You can choose whether to pass the path of the shell script or YAML configuration file for starting training. The shell script and YAML configuration file can be passed at the same time, to collect training hyperparameters, environment variables, and third-party library versions. By default, neither of them is passed, indicating that only environment variables and third-party library information are collected. Both the shell script and YAML configuration file can be used to collect training hyperparameters, depending on the location of the file where training hyperparameters are included.|
| **-o** or **--output**| Optional   | Output path of the collection result file. By default, the output result file is a package named **config_check_pack.zip**. You can customize the file name and add the result file name to the end of the path. The file name extension must be `.zip`.|

**Example**

- Default scenario

  ```bash
  msprobe config_check -d
  ```

- Shell script passed

  ```bash
  msprobe config_check -d train.sh -o /xx/output_file_path/config_check_pack.zip
  ```

- Shell script and YAML configuration file passed

  ```Python
  msprobe config_check -d train.sh config.yaml -o /xx/output_file_path/config_check_pack.zip
  ```

**Output Description**

After command execution, **config_check_pack.zip** is output in both environments. The result file is used for subsequent [data comparison](#data comparison).

### Dynamic Data Collection

**Description**

Add an API to the training script to start collection. Environment variables, third-party library versions, training hyperparameters, weights, datasets, and random operations can be collected.

**Precautions**

When using MindSpeed-LLM for data collection, note that the **apply_patches** function in dynamic data collection mode must be executed
after the **megatron_adaptor** function in **pretrain_gpt.py** of MindSpeed-LLM is imported.

**Example**

1. 
   Add the following code to the beginning of the first Python script executed in the training process:

   ```Python
   from msprobe.core.config_check import ConfigChecker
   ConfigChecker.apply_patches(fmk)
   ```

   **apply_patches**: apply patches required for data collection.

   - **fmk** (string): (optional) training framework. The value can be **pytorch** or **mindspore**. By default, this parameter is not set, indicating that PyTorch is used.

2. Add the following code after the model is initialized:

   ```Python
   from msprobe.core.config_check import ConfigChecker
   ConfigChecker(model=model, shell_path="", output_zip_path="", fmk="")
   ```

   **ConfigChecker** attaches hooks required for data collection to the model. Data is collected each time the model is about to be executed forward.

   - **model** (Model): (optional) initialized model. By default, weights and datasets are not collected.
   - **shell_path** (list[]): (optional) In dynamic collection mode, Megatron training hyperparameters can be automatically captured. You are advised not to pass this parameter when using Megatron. In other cases, you can choose whether to pass the path of the shell script or YAML configuration file for starting training. The shell script and YAML configuration file can be passed at the same time, to collect training hyperparameters. By default, neither of them is passed, indicating that training hyperparameters are not collected. Both the shell script and YAML configuration file can be used to collect training hyperparameters, depending on the location of the file where training hyperparameters are included.
   - **output_zip_path** (string): (optional) output path of the collection result file. By default, the output result file is a package named **config_check_pack.zip**. You can customize the file name and add the result file name to the end of the configuration path. The file name extension must be `.zip`.
   - **fmk** (string): (optional) training framework. The value can be **pytorch** or **mindspore**. By default, this parameter is not set, indicating that PyTorch is used.

   After the collection is complete, a .zip package is generated, which contains [configurations that affect precision](#overview). The data is stored by rank and step (micro_step).

3. Perform the preceding operations in another environment to obtain another .zip package.

**Output Description**

After command execution, **config_check_pack.zip** is output in both environments. The result file is used for subsequent [data comparison](#data comparison).

## Data Comparison

**Description**

The .zip packages collected in the two training environments when [data collection](#data collection) is performed are used as inputs for data comparison.

**Precautions**

None

**Syntax**

```bash
msprobe config_check -c bench_zip_path cmp_zip_path [-o <output_path>]
```

**Parameters**

| Parameter         | Required/Optional| Description                                                        |
| ------------- | --------- | ------------------------------------------------------------ |
| **-c** or **--compare**| Required     | Compares data. Both **bench_zip_path** and **cmp_zip_path** must be configured. **bench_zip_path** indicates the data collected in the benchmark environment, and **cmp_zip_path** indicates the data collected in the environment to be compared.|
| **-o** or **--output** | Optional     | Specifies the output path of the comparison result. The default value is **config_check_result**. If comparison is performed repeatedly, the original comparison result in the output path will be overwritten.|

**Example**

Copy the two .zip packages to the same environment and run the following command to compare them:

```bash
msprobe config_check -c bench_zip_path cmp_zip_path
```

**Output Description**

After the comparison command is executed, a comparison result file is generated. For details, see [Output File Description](#Output File Description).

## Output File Description

Two directories and one file are generated in the comparison result output path:

- **bench**: data packaged in **bench_zip_path**.
- **cmp**: data packaged in **cmp_zip_path**.
- **result.xlsx**: comparison result. There are multiple sheets. The summary sheet shows the overall check result, and other sheets show the details of specific check items. The step means **micro_step**.

| file_name       |pass_check|
|-----------------|----------|
| env             |pass|
| pip             |pass|
| dataset         |pass|
| weights         |pass|
| hyperparameters |pass|
| random          |pass|

The preceding six items correspond to the environment variable, third-party library version, dataset, weight, training hyperparameter, and random operation check, respectively.

**pass_check** indicates whether the check is passed. The value can be **pass**, **error**, or **warning**. **warning** indicates non-key third-party library version inconsistency which does not affect subsequent msProbe operations. You are advised to view the details for analysis.

The first five items must pass the check before precision comparison.
