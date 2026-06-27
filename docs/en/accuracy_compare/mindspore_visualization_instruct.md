# Graph Comparison in Hierarchical Visualization (MindSpore)

## Overview

This function parses the precision data dumped by msProbe, restores the model graph structure, and compares the precision data at each model layer, helping you understand the model structure and analyze precision issues.

**Concepts**

- msProbe: short for MindStudio Probe, is a precision debugging toolkit that can locate precision issues during model training or inference.
- dump: a process of collecting precision data.

**Usage Process**

1. Install the tool and collect data. For details, see [Preparations](#preparations).
2. Use the command line tool to generate a graph structure file. For details, see [Hierarchical Visualization Instructions](#hierarchical-visualization-instructions).
3. Start the TensorBoard service. For details, see [Starting TensorBoard](#starting-tensorboard).
4. Use a browser to view the graph structure and analyze the model structure and precision data. For details, see [Viewing Results in Browser](#viewing-results-in-browser).

**Precautions**

In multi-rank scenarios, only data in the `step{Step ID}` or `rank{Rank ID}` directory is identified for graph construction.

**Tool Features**

- Supports model structure reconstruction.
- Supports comparison of the structure differences between two models.
- Supports comparison of the precision data between two models.
- Supports overflow/underflow detection of model data.
- Supports batch graph construction in multi-rank scenarios, associates communication nodes of each rank, and analyzes data transfer among ranks.
- Supports node name search, node filtering based on precision comparison results, and node filtering based on overflow/underflow detection results; automatically expands the level where a node is located.
- Supports cross-suite and cross-framework model comparison.
- Supports precision data comparison between two models under different [parallelism policies](#graph-merging-under-different-parallelism-policies).

![vis_show](../figures/visualization/vis_showcase.png)

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

If you choose to compile and install msProbe, you must configure `--include-mod tb_graph_ascend` in the compilation command to build the hierarchical visualization plugin.

**Data Preparation**

This function is supported only in dynamic graph scenarios. You need to set `level` to `L0` (cell information) or `mix` (cell and API information) to collect model structure data. That is, the content of the collection result file `construct.json` is not empty. For details, see [Precision Data Collection in MindSpore](../dump/mindspore_data_dump_instruct.md).

**Constraints**

MindSpore 2.4.0 or later is required.

## Hierarchical Visualization Instructions

### Single-Graph Construction

**Function**

Displays the model structure, precision data, and stack information, and provides the overflow/underflow detection function. It is applicable to scenarios where the model structure and data overflow/underflow need to be analyzed.

**Precautions**

The model structure data to be collected must be available. Ensure that the dump level is set to `L0` (module information) or `mix` (module and API information). The content of the collection result file `construct.json` cannot be empty.

**Syntax**

```bash
msprobe graph_visualize -tp <target_path> -o <output_path> [-oc] [-tensor_log] [-progress_log]
```

**Parameters**

| Parameter                                | Mandatory (Yes/No)| Description                                                        |
| -------------------------------------- | --------- | ------------------------------------------------------------ |
| `-tp` or `--target_path`                    | Yes     | Comparison path on the debugging side. The value is of the string type. The tool automatically performs single-rank build, multi-rank batch build, or multi-step batch build based on the path format.|
| `-o` or `--output_path`                     | Yes     | Directory for storing the graph construction result file. The value is of the string type. The file name is automatically generated based on the timestamp in the format of `build_{timestamp}.vis.db`.|
| `-oc` or `--overflow_check`                 | No     | Whether to enable overflow/underflow detection. After it is enabled, the overflow/underflow level of each overflowed/underflowed node is marked in the output .db file (`build_{timestamp}.vis.db`). If this parameter is configured, the function is enabled. By default, this parameter is not configured.|
| `-tensor_log` or `--is_print_compare_log`   | No     | Whether to enable log printing for a single module or API. Only the tensor data dumped by msProbe is supported. If this parameter is configured, the function is enabled. By default, this parameter is not configured.|
| `-progress_log` or `--is_print_progress_log`| No     | Whether to enable log printing for the detailed task progress. If this parameter is configured, the function is enabled. By default, this parameter is not configured.|

Example 1: Construct a single-rank graph.

```bash
msprobe graph_visualize -tp ./target_path -o ./output_path
```

The format of `target_path` must comply with the single-rank format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

Example 2: Construct multi-rank graphs in batches.

```bash
msprobe graph_visualize -tp ./target_path -o ./output_path
```

The format of `target_path` must comply with the multi-rank format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

Example 3: Construct multi-step graphs in batches.

```bash
msprobe graph_visualize -tp ./target_path -o ./output_path
```

The format of `target_path` must comply with the multi-step format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

Example 4: Perform overflow/underflow detection on a single graph.

```bash
msprobe graph_visualize -tp ./target_path -o ./output_path -oc
```

In the output result, each graph node is marked with an overflow/underflow detection metric. The metrics are as follows:

- `medium`: abnormal input; normal output
- `high`: abnormal input; abnormal output; the norm value of the output is abnormally increased compared with that of the input.
- `critical`: normal input; abnormal output

**Output Description**

In the configured output path, a `.vis.db` file is generated. The file name is automatically generated based on the timestamp in the format of `build_{timestamp}.vis.db`.

### Dual-Graph Comparison

**Function**

Displays model structure, structural differences, precision data, precision comparison metrics, and suspected precision issues (where larger differences in precision metrics appear in deeper colors). Additionally, cross-suite comparison, overflow/underflow detection, and fuzzy matching are supported.

Currently, three types of dump data are supported. The hierarchical visualization tool automatically determines the data type during comparison:

1. Statistics: Only the input and output data statistics of APIs and modules are dumped, which occupies a small amount of drive space.
2. Tensor: The input and output data statistics of APIs and modules are dumped, with tensors saved to drive. This type consumes significant drive space but provides more accurate comparison results.
3. MD5: The input and output data statistics and CRC-32 information of APIs and modules are dumped.

For details about how to configure the dump type, see [Configuration File Introduction](../dump/config_json_introduct.md).

**Precautions**

The model structure data to be collected must be available. Ensure that the dump level is set to `L0` (module information) or `mix` (module and API information). The content of the collection result file `construct.json` cannot be empty.

**Syntax**

```bash
msprobe graph_visualize -tp <target_path> -gp <target_path> -o <output_path> [-lm] [-oc] [-fm] [-tensor_log] [-progress_log]
```

Optional fields are enclosed in square brackets ([]), and variables are enclosed in angle brackets (<>).

**Parameters**

| Parameter                 | Mandatory (Yes/No)| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------- | -------- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-tp` or `--target_path`   | Yes    | Comparison path on the debugging side. The value is of the string type. The tool automatically performs single-rank comparison, multi-rank batch comparison, or multi-step batch comparison based on the path format. The value is of the string type.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `-gp` or `--golden_path`   | No (Mandatory in the dual-graph comparison scenario)    | Comparison path on the benchmark side. The value is of the string type. If this parameter is not set, single-graph construction is performed.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `-o` or `--output_path`    | Yes    | Directory for storing the graph construction result file. The value is of the string type. The file name is automatically generated based on the timestamp in the format of `compare_{timestamp}.vis.db`.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `-lm` or `--layer_mapping`| No | Cross-framework comparison, which is applicable to the comparison between MindSpore and PyTorch. If this parameter is configured, cross-framework layer comparison is enabled. After the layers in the model code are specified, the corresponding modules or APIs can be identified. You need to specify a custom mapping file in .yaml format. For details about the format and configuration of the custom mapping file, see [Custom Layer Mapping File](#custom-layer-mapping-file) and [Configuring Layer Mapping for Hierarchical Model Visualization](../examples/layer_mapping_example.md). After this parameter is configured, comparison is performed only by node name, and the type and shape of a node are ignored.<br><br>Node naming format: `{Cell}.{cell_name}.{class_name}.{forward/backward}.{number_of_calls}`<br>&#738226; If the values of `cell_name` are different, specify a custom mapping file using the `-lm` parameter, for example, `-lm mapping.yaml`.<br>&#738226; If the values of `cell_name` are the same but the values of `class_name` are different, directly configure the `-lm` parameter, for example, `-lm`.<br>&#738226; If the values of `cell_name` and `class_name` are the same, you do not need to configure the `-lm` parameter.|
| `-oc` or `--overflow_check`| No    | Whether to enable overflow/underflow detection. After it is enabled, the overflow/underflow level of each overflowed/underflowed node is marked in the output .db file (`compare_{timestamp}.vis.db`). If this parameter is configured, the function is enabled. By default, this parameter is not configured.                                                                                                                                                                                                                                                                                                                                                                    |
| `-fm` or `--fuzzy_match`   | No    | Whether to enable fuzzy matching. If this parameter is configured, the function is enabled. By default, this parameter is not configured. For details about the differences between fuzzy matching and default matching, see [Matching Description](#matching-description).                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `-tensor_log` or `--is_print_compare_log`   | No                      | Whether to enable log printing for a single module or API. Only the tensor data dumped by msProbe is supported. If this parameter is configured, the function is enabled. By default, this parameter is not configured.|
| `-progress_log` or `--is_print_progress_log`| No| Whether to enable log printing for the detailed task progress. If this parameter is configured, the function is enabled. By default, this parameter is not configured.|

Example 1: Perform single-rank graph comparison.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
```

The formats of `target_path` and `golden_path` must comply with the single-rank format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

Example 2: Perform multi-rank batch graph comparison.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
```

The formats of `target_path` and `golden_path` must comply with the multi-rank format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

Example 3: Perform multi-step batch graph comparison.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
```

The formats of `target_path` and `golden_path` must comply with the multi-step format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

Example 4: Perform cross-suite comparison.

If the node names on the debugging side are the same as those on the benchmark side, specify only the `-lm` parameter.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path -lm
```

If the node names on the debugging side are different from those on the benchmark side, you need to configure a custom mapping file. Pass the path of the custom mapping file to the `-lm` parameter. For details about how to configure the mapping file, see the provided parameter description.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path -lm ./mapping.yaml
```

Example 5: Perform overflow/underflow detection.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path -oc
```

In the output result, each graph node is marked with an overflow/underflow detection metric. The metrics are as follows:

- `medium`: abnormal input; normal output
- `high`: abnormal input; abnormal output. The norm value of the output is abnormally larger than that of the input.
- `critical`: normal input; abnormal output

Example 6: Perform fuzzy matching.

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path -fm
```

For details about the differences between fuzzy matching and default matching, see [Matching Description](#matching-description).

**Output Description**

In the configured output path, a `.vis.db` file is generated. The file name is automatically generated based on the timestamp in the format of `compare_{timestamp}.vis.db`.

### Model Structure Comparison

**Function**

Focuses on the model structure rather than the training process data. For example, this function ensures the consistency of the model structure before and after model migration, or determines whether the precision difference is caused by the model structure difference.

**Precautions**

When using msProbe to collect model data, collect only the model structure (`task=structure`). This configuration prevents the collection of model training process data, significantly reducing the collection time.

For details about the dump configuration, see [Dump Configuration Example](../dump/config_json_introduct.md#task--structure).

**Syntax**

See the syntax in [Dual-Graph Comparison](#dual-graph-comparison).

**Parameters**

See the parameter description in [Dual-Graph Comparison](#dual-graph-comparison).

**Example**

See examples 1, 2, and 3 in [Dual-Graph Comparison](#dual-graph-comparison).

**Output Description**

In the configured output path, a <idp:inline displayname="code" id="code142947418558">.vis.db</idp:inline> file is generated. The file name is automatically generated based on the timestamp in the format of <idp:inline displayname="code" id="code5294184105510">compare_{timestamp}.vis.db</idp:inline>.

### Graph Merging Under Different Parallelism Policies

**Function**

Different model parallelism policies lead to precision discrepancies between two models, requiring a network-wide data comparison. However, because partitioned data and model structures are distributed across multiple ranks, direct comparison is not possible. Therefore, the distributed data and model structures must be merged before comparison.

**Precautions**

- The supported model parallelism policies include Tensor Parallelism (TP), Pipeline Parallelism (PP), and Virtual Pipeline Parallelism (VPP). Context Parallelism (CP) and Expert Parallelism (EP) are not supported.
- Graph merging is supported for models based on Megatron and MindSpeed-LLM. The graph merging effect of models based on other suites is to be verified.
- Only the statistics data dumped by msProbe is supported. The `level` must be set to `L0` or `mix`.
- During comparison in graph merging mode, ensure that the Data Parallelism (DP) configuration is consistent. For example, with `rank=8 tp=1 pp=8`, the configuration `dp=1` produces a single merged graph. With `rank=8 tp=1 pp=4`, the corresponding `dp=2` produces two merged graphs. Currently, comparison between graphs of different quantities is not supported.

**Syntax**

```bash
msprobe graph_visualize -tp <target_path> [-gp <golden_path>] -o <output_path> [options]
```

**Parameters**

| Parameter                          | Mandatory (Yes/No)| Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|-------------------------------| ----- |-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-tp` or `--target_path`          | Yes | Comparison path on the debugging side. The value is of the string type. The tool automatically performs single-rank comparison, multi-rank batch comparison, or multi-step batch comparison based on the path format. The value is of the string type.|
| `-gp` or `--golden_path`          | No | Comparison path on the benchmark side. The value is of the string type. If this parameter is not set, single-graph construction is performed.|
| `-o` or `--output_path`           | Yes | Directory for storing the graph construction result file. The value is of the string type. The file name is automatically generated based on the timestamp in the format of `compare_{timestamp}.vis.db`.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `-lm` or `--layer_mapping`| No | Cross-framework comparison, which is applicable to the comparison between MindSpore and PyTorch. If this parameter is configured, cross-framework layer comparison is enabled. After the layers in the model code are specified, the corresponding modules or APIs can be identified. You need to specify a custom mapping file in .yaml format. For details about the format and configuration of the custom mapping file, see [Custom Layer Mapping File](#custom-layer-mapping-file) and [Configuring Layer Mapping for Hierarchical Model Visualization](../examples/layer_mapping_example.md). After this parameter is configured, comparison is performed only by node name, and the type and shape of a node are ignored.<br><br>Node naming format: `{Cell}.{cell_name}.{class_name}.{forward/backward}.{number_of_calls}`<br>&#738226; If the values of `cell_name` are different, specify a custom mapping file using the `-lm` parameter, for example, `-lm mapping.yaml`.<br>&#738226; If the values of `cell_name` are the same but the values of `class_name` are different, directly configure the `-lm` parameter, for example, `-lm`.<br>&#738226; If the values of `cell_name` and `class_name` are the same, you do not need to configure the `-lm` parameter.|
| `-oc` or `--overflow_check`       | No | Whether to enable overflow/underflow detection. After it is enabled, the overflow/underflow level of each overflowed/underflowed node is marked in the output .db file (`compare_{timestamp}.vis.db`). If this parameter is configured, the function is enabled. By default, this parameter is not configured.                                                                                                                                                                                                                                                                                                                                                                    |
| `-fm` or `--fuzzy_match`          | No | Whether to enable fuzzy matching. If this parameter is configured, the function is enabled. By default, this parameter is not configured. For details about the differences between fuzzy matching and default matching, see [Matching Description](#matching-description).                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `-tensor_log` or `--is_print_compare_log`   | No                  | Whether to enable log printing for a single module or API. Only the tensor data dumped by msProbe is supported. If this parameter is configured, the function is enabled. By default, this parameter is not configured.|
| `-progress_log` or `--is_print_progress_log`| No| Whether to enable log printing for the detailed task progress. If this parameter is configured, the function is enabled. By default, this parameter is not configured.|
| `--rank_size`                  | No (mandatory only in the graph merging scenario)| Number of accelerator cards used for model training. The value is of the int type. `rank_size=tp*pp*cp*dp`. CP is not supported currently. Therefore, `cp=1` is configured by default for graph merging.                                                                                                                            |
| `--tp`                         | No (mandatory only in the graph merging scenario)| TP size. The value is of the int type. In the actual training script, `--tensor-model-parallel-size T` needs to be specified. `T` indicates TP size, that is, the `tp` parameter required for graph merging (`tp=T`).                                                                                                 |
| --pp                          | No (mandatory only in the graph merging scenario)| Number of pipeline parallel stages. The value is of the int type. In the actual training script, `--pipeline-model-parallel-size P` needs to be specified. `P` indicates the number of pipeline parallel stages, that is, the `pp` parameter required for graph merging (`pp=P`).                                                                                           |
| --vpp                         | No| Number of virtual pipeline parallel stages. The value is of the int type. VPP depends on pipeline parallelism. In the actual training script, you need to specify `--num-layers-per-virtual-pipeline-stage V`, where `V` indicates the number of layers in each virtual pipeline stage, and specify `--num-layers L`, where `L` indicates the total number of model layers. For graph merging, `vpp=L/V/P` is required. The `vpp` parameter is optional. The default value is 1, indicating that VPP is disabled.|
| --order                       | No| Sorting order of model parallelism policies. The value is of the string type. The default value for Megatron is `tp-cp-ep-dp-pp`. If msProbe is used to dump data and the specified level is `L0`, and `order` in the actual training script is not the default value (for example, `--use-tp-pp-dp-mapping` is specified in the actual training script), pass the modified `order`. If the specified level for data dumping is `mix`, no modification is required.                        |

**Example**

Example 1: Comparison in graph merging mode with different TP size

`target_path`: eight ranks (`tp=8`); `golden_path`: four ranks (`tp=4`):

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path --rank_size 8 4 --tp 8 4 --pp 1 1
```

Example 2: Comparison in graph merging mode with different PP size

`target_path`: eight ranks (`pp=8`); `golden_path`: one rank (`pp=1`)

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path --rank_size 8 1 --tp 1 1 --pp 8 1
```

Example 3: Comparison in graph merging mode with different VPP sizes

`target_path`: eight ranks (`pp=8`); `golden_path`: eight ranks (`pp=8`, `vpp=2`)

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path --rank_size 8 8 --tp 1 1 --pp 8 8 --vpp 1 2
```

Example 4: Comparison in graph merging mode with different PP and TP sizes

`target_path`: eight ranks (`pp=8`); `golden_path`: eight ranks (`tp=8`)

```bash
msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path --rank_size 8 8 --tp 1 8 --pp 8 1
```

In all the preceding examples, the formats of `target_path` and `golden_path` must comply with the multi-rank or multi-step format listed in [Dump File Requirements for Graph Construction in Hierarchical Visualization](#dump-file-requirements-for-graph-construction-in-hierarchical-visualization).

## Starting TensorBoard

### Server with Direct Connectivity

Pass `out_path` where the `vis.db` file is generated to `--logdir`.

```bash
tensorboard --logdir out_path --bind_all
```

After the startup, the following log is printed:

![tensorboard_1](../figures/visualization/tensorboard_1.png)

In the preceding example, `ubuntu` is the server address, and `6008` is the port number.

Replace `ubuntu` with the actual server address. For example, if the actual server address is `10.123.12.12`, enter `http://10.123.12.12:6008` in the address box of the browser.

### Server Without Direct Connectivity

If the link is inaccessible (for example, the server cannot be directly connected and a VPN is required), try one of the following methods:

1. Manually set a proxy for the local computer network. For example, in Windows 10, add the server address (for example, `10.123.12.12`) in the manual proxy settings.

   ![proxy](../figures/visualization/proxy.png)

   Then, enter the following command on the server:

   ```bash
   tensorboard --logdir out_path --bind_all
   ```

   Finally, enter `http://10.123.12.12:6008` in the browser.

   If the firewall is enabled on the server, this method will not work. In this case, disable the firewall or try the following methods.

2. Use Visual Studio Code to connect to the server and enter the following command in the Visual Studio Code terminal:

   ```bash
   tensorboard --logdir out_path
   ```

   ![tensorboard_2](../figures/visualization/tensorboard_2.png)

   Hold `CTRL` and click the link.

3. Transfer the graph construction result file from the server to the local computer with the tb_graph_ascend plugin installed to view the result.

   Enter the following command on the computer:

   ```bash
   tensorboard --logdir out_path
   ```

   Hold `CTRL` and click the link.

## Viewing Results in Browser

### Open in Browser

Google Chrome is recommended. Enter the server address and port number in the address box of the browser and press `Enter` to access the TensorBoard page. The "/#graph_ascend" part is automatically appended.

![vis_browser_1](../figures/visualization/vis_browser_1.png)

If you have switched to another function of TensorBoard and want to return to the model hierarchical visualization page, click **GRAPH_ASCEND** in the upper left corner.

![vis_browser_2](../figures/visualization/vis_browser_2.png)

### Result Check

For details, see [Result Check](./pytorch_visualization_instruct.md#result-check) in *Graph Comparison in Hierarchical Visualization in PyTorch*.

## Graph Comparison Description

### Color

For details, see [Color Description](./pytorch_visualization_instruct.md#legend) in *Graph Comparison in Hierarchical Visualization in PyTorch*.

### Metric Description

Precision comparison evaluates API precision from three aspects: real data mode, statistics mode, and MD5 mode. The comparison results of these modes have different metrics.

**Common Metrics**

- `name`: parameter name, for example, `input.0`
- `type`: type, for example, `mindspore.Tensor`
- `dtype`: data type, for example, `BFloat32`
- `shape`: tensor shape, for example, `[32, 1, 32]`
- `Max`: maximum value
- `Min`: minimum value
- `Mean`: mean value
- `Norm`: L2 norm

**Metrics in Real Data Mode**

- `Cosine`: tensor's cosine similarity
- `EucDist`: tensor's Euclidean distance
- `MaxAbsErr`: tensor's maximum absolute error
- `MaxRelativeErr`: tensor's maximum relative error
- `One Thousandth Err Ratio`: proportion of elements in a tensor with a relative error less than 0.1%
- `Five Thousandth Err Ratio`: proportion of elements in a tensor with a relative error less than 0.5%

**Metrics in Statistics Mode**

- `(Max, Min, Mean, Norm) diff`: absolute error of the statistics
- `(Max, Min, Mean, Norm) RelativeErr`: relative error of the statistics

**Metrics in MD5 Mode**

- `md5`: CRC-32 value

## Appendixes

### Custom Layer Mapping File

The file name is the format of `\*.yaml`. The asterisk (*) indicates the file name, which can be customized.

The following is an example of the file content:

```yaml
ParallelAttention:                 # Layer name
  qkv_proj: query_key_value        # The content on the left of the colon (:) is the name of the layer nested in the MindSpore model code, and the content on the right of the colon (:) is the name of the layer nested in the PyTorch model code.
  out_proj: dense

ParallelTransformerLayer:
  attention: self_attention

Embedding:
  dropout: embedding_dropout

ParallelMLP:
  mapping: dense_h_to_4h
  projection: dense_4h_to_h

PipelineCell:
  model: module

Cell:
  network_with_loss: module
```

The layer name needs to be obtained from the model code.

In the YAML file, you only need to configure the layers that have the same functions but different names in the MindSpore and PyTorch model code. Layers with the same names will be automatically identified and mapped.

Model code example:

![ms_dump](../figures/ms_layer.png)

### Dump File Requirements for Graph Construction in Hierarchical Visualization

**Single-Rank Format**

`dump_path`: Must contain `dump.json`, `stack.json`, and `construct.json`. `construct.json` cannot be empty. If `construct.json` is empty, check whether the `level` parameter is set to `L0` or `mix` or not.

```ColdFusion
├── dump_path
│   ├── dump_tensor_data (available when the `task` parameter for dumping is set to `tensor`)
|   |    ├── MintFunctional.relu.0.backward.input.0.npy
|   |    ├── Mint.abs.0.forward.input.0.npy
|   |    ...
|   |    └── Cell.relu.ReLU.forward.0.input.0.npy
|   ├── dump.json         # Data information
|   ├── stack.json         # Call stack information
|   └── construct.json    # Hierarchical structure. When `level` is set to `L1`, the `construct.json` file is empty.
```

**Multi-Rank Format**

`dump_path`: Must contain only folders named in the format of `rank{number}`. Each `rank` folder must contain `dump.json`, `stack.json`, and `construct.json`. `construct.json` cannot be empty. If `construct.json` is empty, check whether the `level` parameter is set to `L0` or `mix` or not.

```ColdFusion
├── dump_path
|   ├── rank0
|   │   ├── dump_tensor_data (available only when the `task` parameter for dumping is set to `tensor`)
|   |   |    ├── MintFunctional.relu.0.backward.input.0.npy
|   |   |    ├── Mint.abs.0.forward.input.0.npy
|   |   |    ...
|   |   |    └── Cell.relu.ReLU.forward.0.input.0.npy
|   |   ├── dump.json         # Data information
|   |   ├── stack.json        # Operator call stack information
|   |   └── construct.json    # Hierarchical structure. When `level` is `L1`, the content of `construct.json` is empty.
|   ├── rank1
|   |   ├── dump_tensor_data
|   |   |   └── ...
|   |   ├── dump.json
|   |   ├── stack.json
|   |   └── construct.json
|   ├── ...
|   |
|   └── rankn
```

**Multi-Step Format**

`dump_path`: Must contain only folders named in the format of `step{number}`. Each `step` folder must contain only folders named in the format of `rank{number}`. Each `rank` folder must contain `dump.json`, `stack.json`, and `construct.json`. `construct.json` cannot be empty. If `construct.json` is empty, check whether the `level` parameter is set to `L0` or `mix` or not.

```ColdFusion
├── dump_path
│   ├── step0
│   |   ├── rank0
│   |   │   ├── dump_tensor_data (available only when `task` is set to `tensor`)
|   |   |   |    ├── MintFunctional.relu.0.backward.input.0.npy
|   |   |   |    ├── Mint.abs.0.forward.input.0.npy  
|   |   |   |    ...
|   |   |   |    └── Cell.relu.ReLU.forward.0.input.0.npy
│   |   |   ├── dump.json             # Data information
│   |   |   ├── stack.json            # Call stack information
│   |   |   └── construct.json        # Hierarchical structure. When `level` is `L1`, the content of `construct.json` is empty.
│   |   ├── rank1
|   |   |   ├── dump_tensor_data
|   |   |   |   └── ...
│   |   |   ├── dump.json
│   |   |   ├── stack.json
|   |   |   └── construct.json
│   |   ├── ...
│   |   |
|   |   └── rankn
│   ├── step1
│   |   ├── ...
│   ├── step2
```

### Matching Description

1. Default matching

   - The dump names of all nodes are the same.
   - The node levels are the same (the parent nodes are the same).

2. Fuzzy matching

   - For module nodes with identical dump names, matching is performed based on the consistent name and the call sequence within the module node, ignoring the call count of individual APIs under each node.

     ![fuzzy_match_pt.png](../figures/visualization/fuzzy_match_pt.png)

Dump names follow the format `Name + Number of calls` (e.g., in `Torch.matmul.2.forward`, `matmul` is the name and `2` is the call count).

## FAQs

For details, see [FAQs](./pytorch_visualization_instruct.md#faq) in *Graph Comparison in Hierarchical Visualization in PyTorch*.
