# Slime Hyperparameter Comparison and Key Hyperparameter Verification

## Introduction

When slime is trained on the NPU and the benchmark server, training logs are collected, and the training logs contain the real hyperparameter configuration information, such as the training optimizer learning rate, KL divergence, and so on.

Slime Hyperparameter Compare can compare the real hyperparameter configurations collected in the training logs on two different servers, filter out only the parts related to the configuration, save them as configuration files, and compare the parameter configurations of the two configuration files. By efficiently comparing real hyperparameter configurations, it accelerates the localization of training or inference precision problems caused by configuration differences.

## Preparations

Install the msProbe tool. For details, see [msProbe Installation Guide](../install_guide/msprobe_install_guide.md).

## Slime Hyperparameter Comparison

### Function

**Function Description**

Compare the training logs of the NPU and the benchmark (such as GPU), parse the configuration from the logs, remove the print information unrelated to the configuration, save the configuration information as a configuration file, and output the comparison result of the real hyperparameter configurations during training by comparing the configuration files of the NPU and the benchmark; the result is saved as a csv file.

**Command Format**

```bash
msprobe config_check -sc <NPU_log> <bench_log> [-o <compare_result>]
```

**Parameter Description**

| Parameter Name               | Optional/Mandatory | Parameter Description                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| -sc or --slime-compare | Mandatory      | Execute the comparison operation; NPU_log and bench_log are respectively the paths of the two slime training logs to be compared; for a detailed description of the path configuration, see [Log Path Description](#log_path). |
| -o or --output        | Optional      | Comparison result output path, defaulting to ./slime_param_compare_result under the current execution directory; a custom folder path can also be used.|

**Log Path Description**<a name="log_path"></a>

For the two paths of the `--slime-compare` parameter:

- `NPU_log` and `bench_log` are respectively the training logs saved when slime is trained on the NPU and the benchmark server.
- `NPU_log` and `bench_log` are configured in order; by default, the latter log is automatically selected as the benchmark log for comparison.
- The format of the `NPU_log` and `bench_log` training log files to be compared only supports ".log" or ".txt" (such as `slime_NPU.log` or `slime_NPU.txt`); otherwise an error will be reported when executing the comparison.
- The training log must contain the complete slime configuration information; otherwise parsing will fail.

**Usage Example**

Execute the comparison operation. An example command is as follows:

```bash
msprobe config_check -sc NPU_log/training.log bench_log/training.log -o ./compare_result
```

**Output Description**

After the comparison operation is executed, the console prints the output path of the comparison result file. For details, see [Output Result File Description](#output_file_desc1).

### Output Result File Description<a name="output_file_desc1"></a>

The comparison result output path generates 3 files:

- `bench_config.json`: the config configuration information extracted from the benchmark log.
- `NPU_config.json`: the config configuration information extracted from the NPU log.
- `hyper_params_compare.csv`: the comparison result. It contains 1 sheet, and the sheet contains the hyperparameter name, the NPU effective value, the bench effective value, and whether they are consistent.

Example content of the `hyper_params_compare.csv` file:

| Hyperparameter Name                          | NPU Effective Value | Bench Effective Value | Consistent |
| ----------------------------------------- | --------- | ----------- | -------- |
| use_kl_loss                    | TRUE      | FALSE     | No       |
| eps_clip                           | 0.2           | 0.2           | Yes       |

Term explanations in the comparison result:

| Statistical Value      | Explanation                                                         |
| ----------- | ------------------------------------------------------------ |
| Hyperparameter Name    | The real hyperparameter name in the configuration information involved in the slime log                       |
| NPU Effective Value   | The real value corresponding to the hyperparameter name in the NPU training log                      |
| Bench Effective Value | The real value corresponding to the hyperparameter name in the benchmark training log                     |
| Consistent    | Whether the NPU effective value and the bench effective value are consistent; if consistent it is "Yes", otherwise "No"; when the result is No, the user needs to further confirm and troubleshoot whether the precision difference is caused by this hyperparameter inconsistency |
