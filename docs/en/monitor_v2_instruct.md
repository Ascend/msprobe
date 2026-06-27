# Lightweight Training Status Monitoring Tool

## Overview

`Monitor V2` is a lightweight training status monitoring tool of msProbe. It enables you to collect key intermediate values (such as module input/output, weight gradient, optimizer momentum, and communication operator statistics) during training as required and flush them to drives in CSV format for training stability evaluation and exception locating.

**Tool Usage Process**

(1) Configure the monitoring items to be collected.
(2) Initialize `mon.step()` in the training code and call it once by step.
(3) View the CSV result in the output directory.

**Applicable Scenarios**

- When the loss increases or spikes occur, use `module` to observe the value distribution of the model forward input/output and backward gradients to determine whether the exception is caused by a specific layer or phase.
- When the gradient norm is abnormal, use `weight_grad` to locate the abnormal parameter and phase (`unreduced`: gradient accumulation phase; `reduced`: before gradient aggregation).
- When convergence deteriorates or oscillation occurs, use `optimizer` to check whether the momentum (`exp_avg/exp_avg_sq`) distribution changes abruptly.
- When a synchronization or communication exception occurs during distributed training, use `cc` to collect statistics on communication operators and narrow down the check scope based on code line filtering.

**Recommended Tool Enabling Policy**

- Enable `weight_grad` for long-term monitoring. Then, enable `module`/`cc` as needed when an issue occurs to reduce the overhead by filtering targets.

## Preparations

**Installation**

Install msProbe by referring to [msProbe Installation Guide](./msprobe_install_guide.md).

**Constraints**

- PyTorch: `torch >= 2.1`
- MindSpore: `mindspore >= 2.4.10` (A dynamic graph environment is required. If MSAdapter or MindTorch is used in the project, refer to the actual project requirements.)
- Monitor V2 currently supports only `format=csv`.

**Version Requirements and Restrictions**

To avoid confusion with [Monitor V1](./monitor_instruct.md), Monitor V2 must meet the following requirements:

- Output: Only `format=csv` is supported (`tensorboard`/`api` not supported).
- Configuration: `monitors.<name>` is used for organization, and naming of v1 such as`xy_distribution/wg_distribution/...` is not used.
- Toolchain: Toolchains of v1 such as `print_struct`, `stack_info`, `alert`, `csv2tensorboard`, and `csv2db in v1` are not supported.

## Quick Start

### Configuration File Preparation

Create a `monitor_v2_config.json` file in the directory where the training script is located. In the following example, only `weight_grad` is collected (a recommended configuration for long-term low-overhead monitoring).

For more configuration fields and their meanings, see [Detailed Configuration](#config-details).

```json
{
  "framework": "pytorch",
  "output_dir": "./monitor_v2_out",
  "rank": [0],
  "start_step": 0,
  "step_interval": 1,
  "step_count_per_record": 1,
  "collect_times": 100,
  "format": "csv",
  "monitors": {
    "weight_grad": {
      "enabled": true,
      "ops": ["min", "max", "mean", "norm", "nans"],
      "eps": 1e-8,
      "monitor_mbs_grad": false
    }
  }
}
```

### Example in PyTorch Scenario

First, initialize the monitor after the model and optimizer are ready. Then, call `mon.step()` once at the end of each training step. Finally, call `mon.stop()` to release resources when the training is complete. The key is to ensure that the function is called once for each step (usually after `optimizer.step()` and before or after `optimizer.zero_grad()`).

> [!NOTE]
>
> If `patch_optimizer_step=true` is configured (or `optimizer` is passed without explicit configuration), `optimizer.step()` is automatically wrapped to trigger data collection. In this case, do not manually call `mon.step()`. If you need to manually call it, explicitly set `patch_optimizer_step=false`.

```python
from msprobe.core.monitor_v2.trainer import TrainerMonitorV2

mon = TrainerMonitorV2("./monitor_v2_config.json", fr="pytorch")  # fr can be omitted. By default, config.framework is read.
mon.start(model=model, optimizer=optimizer, grad_acc_steps=grad_acc_steps)

for _ in range(num_steps):
    loss = forward(...)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    mon.step()

mon.stop()
```

### Example in MindSpore Scenario<a id="quickstart-mindspore"></a>

```python
from msprobe.core.monitor_v2.trainer import TrainerMonitorV2

mon = TrainerMonitorV2("./monitor_v2_config.json", fr="mindspore")
mon.start(model=model, optimizer=optimizer, grad_acc_steps=grad_acc_steps)
for _ in range(num_steps):
    ...
    mon.step()
mon.stop()
```

### Usage Tips

To improve locating efficiency, you are advised to narrow down the problem scope in the following sequence:

- If the loss is abnormal but the gradient norm is normal, preferentially enable `module` and use `targets` to focus on suspicious modules to observe the forward and backward input and output distributions.
- If the gradient norm is abnormal, preferentially enable `weight_grad` and use `unreduced` or `reduced` to determine whether the anomaly occurred during backward accumulation or before the step.
- If you suspect a communication-related issue, instantiate `TrainerMonitorV2` and invoke `start()` as early in the training process as possible. Doing so minimizes the risk of interception failure due to the acceleration library caching the original communication API. Also, use `cc_codeline` for filtering.

## Functions

Monitor V2 enables sub-functions as required by setting the `monitors` field in the configuration file. Each sub-function supports independent start, stop, and output operations, enabling flexible combinations tailored to specific problems.

### module (Module Input/Output and Backward Gradient)

**Function**

Collects the forward input/output and backward `grad_input/grad_output` statistics of a specified module to locate the layer and type of tensor where problems such as spikes, NaN/Inf, and abrupt scale changes occur.

**Precautions**

- The overhead is usually higher than that of `weight_grad`. You are advised to narrow down the monitoring scope (by using `targets` to monitor only suspicious modules).
- `module_name` in the output contains `input/output/grad_input/grad_output`, which is used to distinguish the collection scope.

**Example**

The following displays `monitors.module` configurations. For details about how to combine them into a complete `monitor_v2_config.json`, see [Configuration File Preparation](#configuration-file-preparation).

```json
{
  "monitors": {
    "module": {
      "enabled": true,
      "targets": ["encoder.layers.0", "mlp"],
      "ops": ["min", "max", "mean", "norm", "nans"],
      "eps": 1e-8
    }
  }
}
```

Configuration field description: See [Detailed Configuration > module](#config-module).

Code example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

**Output Description**

For details, see [module.csv](#output-module-csv).

### weight_grad (Weight Gradient)

**Function**

Collects weight gradient statistics to determine which parameter's gradient becomes abnormal first and whether the exception occurs during backward accumulation or before the step.

- `unreduced`: data collection during backpropagation (closer to the gradient generation process).
- `reduced`: data collection before `optimizer.step()` is called (closer to the final gradient form before the step).

**Precautions**

If gradient accumulation or micro-batch exists, you are advised to use `grad_acc_steps` or `micro_batch_number` to specify the number of micro-batches. Enable `monitor_mbs_grad` if micro-batch is required.

**Example**

The following displays `monitors.weight_grad` configurations. For details about how to combine them, see [Configuration File Preparation](#configuration-file-preparation).

```json
{
  "monitors": {
    "weight_grad": {
      "enabled": true,
      "monitor_mbs_grad": true,
      "grad_acc_steps": 8
    }
  }
}
```

Configuration field description: See [Detailed Configuration > weight_grad](#config-weight-grad).

Code example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

**Output Description**

See [weight_grad.csv](#output-weight-grad-csv).

### optimizer (Optimizer Momentum)

**Function**

Collects statistical metrics of the optimizer status. Currently, Adam momentum (`exp_avg`/`exp_avg_sq`) is mainly used to determine whether the optimizer status changes abruptly or whether abnormal distribution occurs.

**Precautions**

- This function takes effect only when `optimizer` is provided (`mon.start(model=..., optimizer=...)`).
- Currently, momentum information (`mv_distribution`) is mainly covered. Other extended capabilities depend on the version.

**Example**

The following displays `monitors.optimizer` configurations. For details about how to combine them, see [Configuration File Preparation](#configuration-file-preparation).

```json
{
  "monitors": {
    "optimizer": {
      "enabled": true,
      "mv_distribution": true,
      "ops": ["min", "max", "mean", "norm", "nans"]
    }
  }
}
```

Configuration field description: See [Detailed Configuration > optimizer](#config-optimizer).

Code example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

**Output Description**

See [optimizer.csv](#output-optimizer-csv).

### param (Parameter Distribution)

**Function**

Collects statistics on the distribution of parameters before and after the optimizer step to locate parameter exceptions or update exceptions.

**Precautions**

- This function takes effect only when <idp:inline displayname="code" id="code1018513631318">optimizer</idp:inline> is provided (<idp:inline displayname="code" id="code131857619132">mon.start(model=..., optimizer=...)</idp:inline>).
- `param_distribution` is enabled by default and can be disabled as required.

**Example**

The following displays `monitors.param` configurations. For details about how to combine them, see [Configuration File Preparation](#configuration-file-preparation).

```json
{
  "monitors": {
    "param": {
      "enabled": true,
      "param_distribution": true,
      "ops": ["min", "max", "mean", "norm", "nans"]
    }
  }
}
```

Configuration field description: See [Detailed Configuration > monitors.param](#config-param).

Code example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

**Output Description**

See [param.csv](#output-param-csv).

### cc (Communication Operator)

**Function**

Collects statistics and logs of communication operators during distributed training to locate problems such as abnormal communication calls, abnormal input and output, and communication irrelevant to training. You can also filter code lines to narrow down the check scope.

**Precautions**

- This function takes effect only after the distributed environment is initialized (for example, `torch.distributed.is_initialized()` in PyTorch is set to `true`).
- You are advised to perform instantiation and invoke `start()` as early as possible in the training process to prevent interception failure caused by some acceleration libraries caching the original communication API.
- `cc_log_only=true` is suitable for scenarios where logs are recorded before filtering rules are applied. It may interrupt training, so use it with caution.

**Example**

The following displays `monitors.cc` configurations. For details about how to combine them, see [Configuration File Preparation](#configuration-file-preparation).

```json
{
  "monitors": {
    "cc": {
      "enabled": true,
      "ops": ["min", "max", "mean", "norm", "nans"],
      "cc_codeline": [],
      "cc_pre_hook": false,
      "cc_log_only": false
    }
  }
}
```

The following displays `monitors.cc` configurations. For details about how to combine them, see [Configuration File Preparation](#configuration-file-preparation). Logs are printed for `cc_codeline` filtering.

```json
{
  "monitors": {
    "cc": {
      "enabled": true,
      "cc_log_only": true
    }
  }
}
```

Configuration field description: See [Detailed Configuration > cc](#config-cc).

Code example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

**Output Description**

See [cc.csv](#output-cc-csv).

## Output File Description

### Output Directory Description

The output directory is specified by `output_dir`. To facilitate multi-rank analysis, each rank outputs independently to `rank_<rank_id>/`.

Currently, only `format=csv` is supported. As each rank outputs independently to `rank_<rank_id>/`, each monitoring module has a CSV file.

Directory structure:

```ColdFusion
<output_dir>/
  rank_<rank_id>/
    module_step0-0.csv
    weight_grad_step0-0.csv
    optimizer_step0-0.csv
    param_step0-0.csv
    cc_step0-0.csv
```

> [!NOTE]
>
> CSV files are generated only for modules enabled in `monitors`.

### CSV Table Header and Field Description

To facilitate direct comparison of results across different monitoring items, the CSV output follows a unified rule:

- Each line contains at least `vpp_stage` and `step`.
- The common fields written by the monitoring module usually include `module_name`, `scope`, and `micro_step` (in some scenarios).
- The statistical metrics are expanded into columns by `ops`: `min/max/mean/norm/nans` (only the metrics configured/enabled by the user are written).

Therefore, a common CSV table header is as follows:

`vpp_stage | step | module_name | scope | micro_step (optional) | min | max | mean | norm | nans`

Field description:

- `step`: training step (increasing by `TrainerMonitorV2.step()`)
- `vpp_stage`: stage ID in the multi-model/multi-stage scenario (deduced from the name prefix `<idx><NAME_SEP>`; `0` by default if there is no prefix)
- `module_name`: tag of the monitored object (The tag rules vary depending on the module. For details, see [Fields Specific to Each Function](#output-module-csv).)
- `scope`: monitoring scope/phase (The semantics vary depending on the module. For details, see [Fields Specific to Each Function](#output-module-csv).)
- `micro_step`: This field is available only when micro-batch is enabled, for example, `weight_grad.monitor_mbs_grad=true`.

### Fields Specific to Each Function

#### module.csv<a id="output-module-csv"></a>

- `scope`: `forward`/`backward`
- `module_name`: The value is in the format of `<module_name>.<io_kind>.<idx>`, where `io_kind` may be `input/output/grad_input/grad_output`.
- Tips: Pay attention to whether the module in the abnormal step is abnormal in both `forward` and `backward` to determine whether the exception is caused by abnormal forward activation propagation or gradient link exception.

#### weight_grad.csv<a id="output-weight-grad-csv"></a>

- `scope`: `unreduced`/`reduced`
- `module_name`: parameter name (without the `scope` suffix). `micro_step` field is used to distinguish micro-batch.
- `micro_step`: records the current micro-batch index when micro-batch monitoring is enabled; records the total number of accumulated micro-batches (see `micro_batch_number/grad_acc_steps`) when micro-batch monitoring is disabled.
- Tips: If `unreduced` is normal but `reduced` is abnormal, the gradient likely changes or is modified before the step. If `unreduced` is abnormal, the exception likely occurs during the backward link.

#### optimizer.csv<a id="output-optimizer-csv"></a>

- `scope`: `exp_avg`/`exp_avg_sq` (same as the suffix of the `module_name` file)
- `module_name`: in the format of `<name>.exp_avg`/`<name>.exp_avg_sq`
- Tips: When the training fluctuates or does not converge, compare `exp_avg/exp_avg_sq` changes before and after the exception to determine whether it is caused by a sudden change in the optimizer status.

#### param.csv<a id="output-param-csv"></a>

- `scope`: `param_origin`/`param_updated`
- `module_name`: parameter name
- Tips: Compare the parameter distribution before and after the step to locate update exceptions or abrupt value changes.

#### cc.csv<a id="output-cc-csv"></a>

- `scope`: `comm`
- `module_name`: communication tag generated by communication monitoring (usually containing the communication operator, index, and code location information)
- Tips: Use `cc_log_only=true` to obtain communication logs, set `cc_codeline` to filter out communication irrelevant to training, and then enable statistics collection to locate the input and output distribution of abnormal communication.

## Public APIs

This section lists the main user-facing APIs that can be called by Monitor V2.
Note: "Prototype" is for documentation purposes only, where `Any/Optional/Dict/Type` is the Python's `typing` name.

### TrainerMonitorV2

- Function: Trains monitoring orchestrator, loads configurations, initializes the monitoring module, and collects and writes monitoring results in each step.
- Prototype:
  
  ```Python
  TrainerMonitorV2(config_path: str, fr: Optional[str] = None) -> TrainerMonitorV2
  ```

- Parameters:
  - `config_path`: configuration file path (JSON).
  - `fr`: framework type, which can be `pytorch` or `mindspore` or `pt/torch/ms`. If this parameter is not passed, the `framework` field in the configuration file is read.
- Returns: `TrainerMonitorV2` instance.
- Example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

### TrainerMonitorV2.start

- Function: Starts monitoring, that is, creates and starts the modules enabled in `monitors` based on the configuration, and establishes the write context.
- Prototype:
  
  ```Python
  TrainerMonitorV2.start(model: Any = None, optimizer: Any = None, **context: Any) -> None
  ```

- Parameters
  - `model`: model object to be monitored (`nn.Module` for PyTorch/`nn.Cell` for MindSpore; model list in some scenarios is also supported).
  - `optimizer`: optimizer object to be monitored. This parameter is mandatory when `weight_grad/optimizer` is enabled.
  - `context`: optional context information, which is used to supplement the running parameters required by the monitoring module.
    - `grad_acc_steps`/`micro_batch_number`: gradient accumulation/micro-batch count (affecting the `micro_step` semantics of `weight_grad`).
    - Other custom fields: Will be transparently transmitted to each monitoring module.
- Returns: None
- Example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

### TrainerMonitorV2.step

- Function: Advances the training step and triggers the collection and writing of the current step (controlled by `start_step/stop_step/step_interval/collect_times`).
- Prototype:
  
  ```Python
  TrainerMonitorV2.step() -> None
  ```

- Parameters: None
- Returns: None
- Example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

### TrainerMonitorV2.stop

- Function: Stops monitoring and releases resources (removes internal registration/interception operations within monitoring modules and closes the writer).
- Prototype:
  
  ```Python
  TrainerMonitorV2.stop() -> None
  ```

- Parameters: None
- Returns: None
- Example: See [Example in PyTorch Scenario](#example-in-pytorch-scenario) or [Example in MindSpore Scenario](#quickstart-mindspore).

## Detailed Configuration<a id="config-details"></a>

### monitor_v2_config.json

| Field| Mandatory (Yes/No)| Type| Description|
| --- | --- | --- | --- |
| `framework` | No| string | Framework type: `pytorch`/`mindspore` (also supports `pt/torch/ms` alias).|
| `output_dir` | No| string | Output directory. The default value is `./`.|
| `format` | No| string | Output format. Currently, only `csv` is supported.|
| `async_write` | No| bool | Reserved (synchronous writing in CSV is used).|
| `rank` | No| int / list[int] | Rank to be monitored. If this parameter is left empty or not set, all ranks are monitored.|
| `start_step` | No| int | Start step (included) for writing. The default value is `0`.|
| `stop_step` | No| int | End step (excluded) for writing. If this parameter is not set, it can be derived from `collect_times`.|
| `step_interval` | No| int | Write frequency: Write once every *N* steps (default value: `1`).|
| `step_count_per_record` | No| int | Number of steps to be combined into a CSV file (default value: `1`).|
| `patch_optimizer_step` | No| bool | Whether to automatically wrap `optimizer.step()` to trigger collection. If this parameter is not explicitly configured and `optimizer` is passed, it is enabled by default.|
| `collect_times` | No| int | Maximum number of write times. When the maximum number is reached, write operations stop. (The default value is large, meaning collection is effectively always performed.)|
| `monitors` | No| dict | Configuration set of monitoring modules. The key is the module name (see the following table).|

### monitors Configurations

The format of each subitem in `monitors` is as follows:

```json
{
  "enabled": true,
  "...": "Custom fields for each module"
}
```

The table below list common fields that are applicable to most modules.

| Field| Mandatory (Yes/No)| Type| Description|
| --- | --- | --- | --- |
| `enabled` | No| bool | Whether to enable the module. If this parameter is not set, `module` is enabled by default, and other modules are disabled by default.|
| `ops` | No| list[string] | Statistical metric. The value can be `min/max/mean/norm/nans`. If there is no valid item, the default value is used.|
| `eps` | No| number | Stable value. The default value is `1e-8`.|

### module (Module Input/Output and Backward Gradient) <a id="config-module"></a>

| Field| Mandatory (Yes/No)| Type| Description|
| --- | --- | --- | --- |
| `targets` | No| list[string] | Filters target modules. If this parameter is left empty, all modules are included. Otherwise, modules whose names contain the specified keyword are included.|
| `ops`/`eps` | No| - | See description in the "common fields" table.|

### weight_grad (Weight Gradient)<a id="config-weight-grad"></a>

| Field| Mandatory (Yes/No)| Type| Description|
| --- | --- | --- | --- |
| `monitor_mbs_grad` | No| bool | Whether to record micro-batch gradients. The default value is `false`.<br>In the PyTorch FSDP scenario, `weight_grad` automatically detects and collects gradients (`scope=unreduced`) before `reduce`. No separate `fsdp_grad` module is required.|
| `micro_batch_number` | No| int | Number of micro-batches, with a higher priority than `grad_acc_steps`.|
| `grad_acc_steps` | No| int | Number of gradient accumulation steps, which can be passed through `TrainerMonitorV2.start(..., grad_acc_steps=...)`.|
| <idp:inline displayname="code" id="code8652331599">ops</idp:inline>/<idp:inline displayname="code" id="code106528335916">eps</idp:inline>| No| - | See description in the "common fields" table.|

> [!NOTE]
>
> `weight_grad` records `unreduced` in the backward phase and captures and records `reduced` before `optimizer.step()` is called.

### optimizer (Optimizer Momentum)<a id="config-optimizer"></a>

| Field| Mandatory (Yes/No)| Type| Description|
| --- | --- | --- | --- |
| `mv_distribution` | No| bool | Whether to collect momentum (m/v, typically `exp_avg/exp_avg_sq` of Adam). The default value is `true`.|
| <idp:inline displayname="code" id="code1265519315590">ops</idp:inline>/<idp:inline displayname="code" id="code2655335597">eps</idp:inline>| No| - | See description in the "common fields" table.|

### param (Parameter Distribution)<a id="config-param"></a>

| Field                | Mandatory (Yes/No)| Type| Description                          |
| -------------------- | --------- | ---- | ------------------------------ |
| `param_distribution` | No     | bool | Whether to collect parameter distribution. The default value is `true`.|
| <idp:inline displayname="code" id="code18655431596">ops</idp:inline>/<idp:inline displayname="code" id="code4655163125920">eps</idp:inline>       | No     | -    | See description in the "common fields" table.                  |

> [!NOTE]
>
> `param` collects parameter distribution before and after `optimizer.step()` and outputs `scope=param_origin/param_updated`.

### cc (Communication Operator)<a id="config-cc"></a>

It takes effect only when the distributed environment has been initialized (for example, `torch.distributed.is_initialized()` in PyTorch is set to `true`).

| Field| Mandatory (Yes/No)| Type| Description|
| --- | --- | --- | --- |
| `cc_codeline` | No| list[string] | Monitors only the specified code lines (for example, `train.py[23]`). If this parameter is left empty, no filtering is performed.|
| `cc_log_only` | No| bool | Whether to print logs only without collecting data. (Some implementations may interrupt training after printing. Exercise caution when using this parameter.)|
| `cc_pre_hook` | No| bool | Whether to monitor communication input (pre-collection).|
| `module_ranks` | No| list[int] | This parameter takes effect only on specified ranks. (If this parameter is not set, the list is empty by default.)|
| `ops`/`eps`| No| - | See description in the "common fields" table.|

> [!NOTE]
>
> `monitors.cc` supports two formats: configuring the preceding fields directly, or nesting them within `cc_distribution` (for compatibility with the legacy structure).
