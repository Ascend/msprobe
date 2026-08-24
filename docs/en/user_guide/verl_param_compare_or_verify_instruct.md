# verl Hyperparameter Comparison and Key Hyperparameter Verification

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:41:22.074Z pushedAt=2026-08-11T02:51:07.143Z -->

## Introduction

When verl performs training on NPU and benchmark servers, training logs are collected. These training logs contain hyperparameter configurations, such as the training optimizer's learning rate and KL divergence.

verl Hyperparameter Compare compares the actual hyperparameter configurations collected from training logs on two different servers. It filters out only the configuration-related portions, saves them as configuration files, and compares the parameter configurations of the two files. By assisting users in efficiently comparing actual hyperparameter configurations, it accelerates the identification of training or inference precision issues caused by configuration discrepancies.

verl Hyperparameter Verify filters out only the configuration-related parts from training logs, saves them as configuration files, outputs the effective values of key hyperparameters, and determines whether the key hyperparameter requirements are met.

## Preparation Before Use

Install msProbe by referring to [*msProbe Installation Guide*](../install_guide/msprobe_install_guide.md).

## Verl Hyperparameter Compare

### Feature Description

**Overview**

The training logs of NPU and its benchmark (e.g., GPU) are compared. Configuration parsing is performed on the logs to remove print information unrelated to configuration, and the configuration information is saved as a file. By comparing the files of NPU and its benchmark, the comparison result of the actual hyperparameter configurations used during training is output and saved as a CSV file.

**Command Syntax**

```bash
msprobe config_check -vc <NPU_log> <bench_log> [-o <compare_result>]
```

**Parameter Description**

| Parameter                | Mandatory/Optional | Description                                                     |
| ------------------- | ------------------ | ------------------------------------------------------------ |
| `-vc` or `--verl-compare` | Mandatory          | Performs the comparison operation. `NPU_log` and `bench_log` are the paths to the two verl training logs to be compared. For detailed path configuration, see [Log Path Description](#log_path). |
| `-o` or `--output`        | Optional           | Output path for the comparison result. Defaults to `./verl_param_compare_resul`t under the current execution directory. A custom folder path may also be specified.|

**Log Path Description**<a name="log_path"></a>

For the two paths of the `--verl-compare` parameter:

- `NPU_log` and `bench_log` are the training logs saved during verl training on the NPU and benchmark servers, respectively. verl training prints logs to the console by default. The logs can be redirected, for example, `python -m verl.trainer.main_ppo ... 2>&1 | tee -a /your/custom/path/training.log`, in which case the log path is `/your/custom/path/training.log`
- `NPU_log` and `bench_log` are configured in order, and the latter log is automatically selected as the benchmark log for comparison.
- The training log files for `NPU_log` and `bench_log` to be compared only support the ".log" or ".txt" format (e.g., `verl_NPU.log` or `verl_NPU.txt`); otherwise, an error will be reported during comparison.
- The training log must contain complete verl configuration, i.e., it must include the section starting with "{'actor_rollout_ref'" and ending with the corresponding "}"; otherwise, parsing will fail. If the training log contains multiple configuration files, only the latest configuration will be extracted into the JSON file. If the latest configuration is incomplete, parsing will fail directly.

**Usage Example**

Example command for comparison:

```bash
msprobe config_check -vc NPU_log/training.log bench_log/training.log -o ./compare_result
```

**Output Description**

After the comparison is completed, the console prints the output path of the comparison result file. For a detailed description, see [Output Result File Description](#output_file_desc1).

### Output Result File Description<a name="output_file_desc1"></a>

Three files are generated in the comparison result output path:

- `bench_config.json`: Configuration extracted from the benchmark log.
- `NPU_config.json`: Configuration extracted from the NPU log.
- `hyper_params_compare.csv`: Comparison result. Since the profiler and ray-related hyperparameter configuration is irrelevant to the training itself, this CSV file does not include comparisons of hyperparameters related to profiler and ray. It contains one sheet page, which includes the hyperparameter name, effective values on NPU and benchmark, and consistency.

The following is the example content of `hyper_params_compare.csv`:

| Hyperparameter Name                                  | NPU Effective Value | Benchmark Effective Value | Value  |
| ----------------------------------------------------- | ------------------- | --------------------- | ----------- |
| `actor_rollout_ref`/`actor`/`calculate_entropy`             | `TRUE`                | `FALSE`                | No          |
| `actor_rollout_ref`/`actor`/`clip_ratio`                    | `0.2`                 | `0.2`                   | Yes         |

Glossary of terms in the comparison result:

| Term      | Explanation                                                                                                                                                                                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hyperparameter Name    | Actual hyperparameter name in the configuration involved in the verl log.                                                                                                                                                                                                               |
| NPU Effective Value    | Actual value corresponding to the hyperparameter name in the NPU training log.                                                                                                                                                                                                                      |
| Benchmark Effective Value  | Actual value corresponding to the hyperparameter name in the benchmark training log.                                                                                                                                                                                                                |
| Consistency            | Whether the NPU effective value and the benchmark effective value are consistent. If consistent, the value is "Yes"; otherwise, it is "No". When the result is "No", further confirmation and investigation are required to determine whether the precision discrepancy is caused by this hyperparameter inconsistency. |

## verl Key Hyperparameter Verification

### Feature Description

**Overview**

Parse the configuration from the training log, remove print information unrelated to the configuration, save the configuration as a file, output the effective values of key hyperparameters, and determine whether they meet the key hyperparameter requirements. The results are saved as a CSV file. The built-in [verl_hyper_params_verify.yaml](../../../python/msprobe/core/config_check/verl_param_compare/verl_hyper_params_verify.yaml) file can be used, or a custom YAML file can be passed in, with the format consistent with the built-in file.

**Command Syntax**

```bash
msprobe config_check -vv [<bench_config>] <tgt_log> [-o <compare_result>]
```

**Parameters**

| Parameter               | Optional/Mandatory | Description                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| `-vv` or `--verl-verify` | Mandatory      | Performs the key hyperparameter verification. `bench_config` specifies the key hyperparameter configuration file (optional), and `tgt_log` specifies the path to the verl training log to be verified (mandatory). For a detailed description of path configuration, see [Path Description](#path). |
| `-o` or `--output`        | Optional      | Output path for the verification result. Defaults to `./verl_param_verify_result` under the current execution directory. A custom folder path can also be specified. |

**Path Description**<a name="path"></a>

For the two paths of the `--verl-verify` parameter:

- `bench_config` is the key hyperparameter configuration file. It defaults to `verl_hyper_params_verify.yaml`. A configuration file can also be manually constructed according to the YAML format and passed in.
- `tgt_log` is the training log saved during verl training. verl training prints the log to the console by default. The log can be redirected, for example, `python -m verl.trainer.main_ppo ... 2>&1 | tee -a /your/custom/path/training.log`, in which case the log path is `/your/custom/path/training.log`.
- `bench_config` and `tgt_log` are configured in order, where `bench_config` is an optional parameter and `tgt_log` is a required parameter.
- The `tgt_log` training log file format only supports ".log" or ".txt" (e.g., `verl_NPU.log` or `verl_NPU.txt`); otherwise, an error will be reported during verification.
- The training log must contain complete verl configuration, i.e., it must include content starting with "{'actor_rollout_ref'" and ending with the corresponding "}"; otherwise, parsing will fail. If the training log contains multiple configuration files, only the latest configuration will be extracted into the JSON file. If the latest configuration is incomplete, parsing will fail directly.

**Usage Example**

Example command for verification:

```bash
msprobe config_check -vv /your/custom/path/training.log -o ./verify_result
```

**Output Description**

After the verification is completed, the console prints the output path of the verification result file. For a detailed description, see [Output Result File Description](#output_file_desc2).

### Output Result File Description<a name="output_file_desc2"></a>

The following files are output in the key hyperparameter verification result path:

- `tgt_config.json`: configuration extracted from the training log.
- `hyper_params_verify.csv`: verl key hyperparameter verification result file, with the following example content:

| Key Parameter Name  | Benchmark Effective Value | Target Effective Value | Consistency |
|-----------  |-----------|-------------|---------|
| `actor_rollout_ref`/`actor`/`calculate_entropy` | `TRUE` | `FALSE` | No   |
| `actor_rollout_ref`/`actor`/`clip_ratio`        | 0.2    | 0.2     | Yes   |

Glossary of terms in the verification result:

| Item      | Explanation|
|------------|-----|
| Key Parameter Name | Name of the key hyperparameter in the YAML configuration file |
| Benchmark Effective Value | Required value of the key hyperparameter |
| Target Effective Value | Actual value of the key hyperparameter in the training log |
| Consistency    | Whether the effective value matches the required value. "Yes" indicates consistency, and "No" indicates inconsistency. When the result is "No", further confirmation is needed to verify whether this critical parameter needs to be adjusted to align with the benchmark's effective value. |
