# Foundation Model Inference Accuracy Debugging Guide

## Introduction

Ensuring inference accuracy for large language models (LLMs) on Ascend NPUs aims to achieve semantic equivalence with their GPU-based counterparts. This is typically assessed through downstream task evaluation or direct comparison with GPU outputs. In practice, accuracy issues often stem from model structure errors, improper parameter configurations, or operator calculation deviations. These issues typically manifest as abnormal behaviors such as incoherent output (gibberish), unexpected generation interruption, repetitive generations, or significant deviation from expected results. Accuracy issues in model inference can be classified into two categories: **model accuracy issues** and **numerical precision issues**. These two types differ in their causes and impacts, and therefore require differentiated analysis and handling strategies.

**Model accuracy issues** are "structural deviations". They include data loading exceptions, incorrect configuration of training hyperparameters, incorrect implementation of network structures, or design defects within the framework itself. Such issues have a decisive impact on model convergence. Consequently, you must carefully examine each phase and make targeted adjustments based on the specific scenario.

**Numerical precision issues** are "computational deviations", arising from the inherent characteristics of floating-point operations. Limited word length prevents full representation of the value range, and differences in computation and communication order, as well as the approximation of mathematical expressions, can all introduce errors. It should be noted that although approximations in numerical computation may interfere with the convergence process, computational discrepancies do not necessarily lead to convergence failure. As the basic unit of computation, an operator's numerical precision is a key consideration. However, due to differences in hardware architecture (e.g., between GPU and CPU, or across different GPU versions), slight numerical deviations in the same computational logic are normal. As long as these deviations fall within a reasonable tolerance, they will not affect the final convergence of your model.

This guide is designed to help you effectively distinguish between normal computational deviations and genuine accuracy issues, and to accurately locate their root causes. To achieve this, it systematically describes the application scenarios and operational workflows of a dedicated toolset for accuracy issue diagnosis. The goal is to empower you to efficiently identify potential risks, either independently or with the support of technical documentation.

# 1. Common Inference Accuracy Issues

Accuracy issues in LLM inference typically manifest as model outputs that deviate from expectations. These symptoms can be classified into the following types:

1. Garbled characters in model responses: A large number of abnormal symbols such as �, \<unk\>, and â€™ are displayed in outputs, or a segment of characters in another language is suddenly inserted.

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/78a354cb-eec8-4670-aa11-39893d30b635/image.png 'image.png')

2. Repeated model responses: A model is stuck in a certain part and repeatedly generates the same or similar text segments.

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/f1f550d1-6c47-4cd5-a3f2-13b654544ebf/image.png 'image.png')

3. Semantic or logical errors: The output uses fluent language but the reasoning is either incoherent or broken.

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/5725d07b-d5c1-46f4-a2a7-2881e91409ae/image.png 'image.png')

4. Inconsistency and jitter: The outputs vary greatly among multiple requests.

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/657ad9a7-d3a9-4e98-b5b0-da8a83fd1d8d/image.png 'image.png')

5. Dataset evaluation discrepancy: The evaluation accuracy falls below the benchmark.

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/e831b0d3-1f41-4f1b-9055-47731a4f9495/image.png 'image.png')

# 2. Common Causes of Model Inference Accuracy Issues

Common accuracy issues in LLM inference can be classified into accuracy errors and practice errors. In inference deployment, practice errors are the primary cause of accuracy issues. The causes of practice errors are as follows:

(1) Model configuration error

Includes abnormal weights and incorrect model parameter configuration.

(a) Incorrect weights can affect the word embedding table, linear, layerNorm, and lmhead.

(b) Parameter configuration includes the padding mode and model `config` settings, such as `pad_token_id`, `eos_token_id`, and `max sequence`. You can avoid errors by aligning `config`.

(2) Model structure error

Includes code implementation error. Typically, this leads to obvious accuracy errors, manifesting as disordered or nonsensical English output, or even no output at all.

(3) Operator parameter pass error

(a) The attention mask is incorrectly passed. Generally, a model does not output disordered characters. However, there are cases where, while the model can answer questions, it may insert an excessive number of spaces. Consequently, the answers may not align with those produced on the GPU

(b) The RoPE rotation coefficient is incorrectly passed. Consequently, a model usually outputs disordered results.

(4) Operator implementation error

(a) Operator implementation postprocessing: Generally, the accuracy is normal during greedy search of a model, but the output is incorrect after the postprocessing sampling policy is added. For example, the output is repeated or similar to the greedy search result.

b. Model-sided operator: Single-operator detection passes but the overall model behaves abnormally. Potential causes may include memory corruption caused by interaction among multiple operators, and improper resetting of registers. You can ask the same question for multiple times after deterministic computing is enabled. If the results are inconsistent, the operator implementation may be incorrect.

(5) Environment version defects or differences

Generally, accuracy may be normal on one type of server but suddenly become abnormal on other servers or in other environments, or after an environment version change.  For example, accuracy may become abnormal when an x86 architecture is switched to an Arm architecture, or after CANN is replaced. For the solutions, see the requirements of different environment versions.

# 3. Locating Model Accuracy Issues Troubleshooting

The following figure shows the overall process of locating model inference accuracy issues.

![image.png](https://raw.gitcode.com/user-images/assets/7898473/79fb6a89-4944-4044-85a9-f2edd6a4098f/image.png 'image.png')

## 3.1 Checklist

In scenarios where a benchmark is available, it is critical to ensure that your configuration aligns with the benchmark setup. For model inference that involves benchmarking, two specific scenarios are typically considered:

1. NPU inference versus GPU inference of the same model, with GPU serving as the benchmark.
2. NPU inference of the current model versus that of a historical model version, with the historical baseline serving as the benchmark.

In the following description, these two scenarios are essentially a comparison between the problem scenario and the benchmark scenario.  Pay attention to the following items in the checklist:

- Comparison of inference hyperparameters and environment variables

You can use Beyond Compare to compare inference hyperparameters and environment variable settings in the inference logs or startup scripts in both scenarios.

- Comparison of third-party library versions

Check whether the versions of third-party libraries such as MindIE, vLLM, and Transformers are consistent with those in the benchmark version by `git branch`.

Check whether the versions of third-party libraries such as PyTorch and PTA are consistent with those in the benchmark version by `pip list`.

- Data reading check

Check the input data of model inference. Generally, you can directly print the input data in the code for checking. For example, in vLLM, you can obtain the original input text through `output.prompt` in the output of `llm.generate`.

- Model configuration check

You can directly print model configurations of both scenarios for comparison.

Example

```python
from vllm import LLM

#Initialize a model.
llm = LLM(
    model="/path/to/your/model", # or Hugging Face model name
    dtype="float16"
)

#View the model configuration.
print("Model configuration (Model Config)")
print("=" * 50)
print(llm.llm_engine.model_config)
```

Output example:

```python
Model configuration (Model Config)
==================================================
ModelConfig(model='meta-llama/Llama-2-7b-hf', tokenizer='meta-llama/Llama-2-7b-hf', tokenizer_mode=auto, trust_remote_code=True, dtype=torch.float16, seed=0, skip_tokenizer_init=False, use_v2_block_manager=False, ...
```

## 3.2 Preparations for Reproducing a Problem

After ensuring that the preceding information is consistent, you need to reproduce a problem. To minimize variables during fault localization, you should fix randomness and enable operator determinism.

### 3.2.1 Fixing Randomness

To reproduce a problem, you need to fix the random operations to ensure experimental reproducibility. These operations include model parameter initialization and dropout layers.
The involved operations are as follows:

- Fix the random seed, such as `np.random.seed`, `torch.manual_seed`, and `torch_npu.npu.manual_seed`.
- Disable the dropout layer.

### 3.2.2 Enabling Determinism

During problem reproduction, you are advised to enable operator computation determinism and communication determinism. Both of them need to be fixed as early as possible before the code starts training.

- Computation determinism:
`torch.use_deterministic_algorithms(True)`
- Communication determinism:
`export HCCL_DETERMINISTIC=TRUE`

To achieve the randomness fixing and determinism enabling operations described above, the msProbe toolkit provides the `seed_all` API. In PyTorch scenarios, this API can quickly fix all random seeds, dropout layers, and ensure deterministic operator computation and communication within the network.

How to use:

```python
from msprobe.pytorch import seed_all
seed_all(seed=1234, mode=True, rm_dropout=True, is_enhanced=False)
```

Parameter description:

| Parameter| Description|  Mandatory (Yes/No)|
|--|--|--|
|seed  |Random number seed. The default value is `1234`. |    No |
|mode  |Deterministic computation mode. The value can be `True` or `False` (default). This mode includes both operator computation determinism and communication determinism. |    No |
|rm_dropout |Switch for controlling dropout invalidation or validation. After this function is enabled, the dropout probability is automatically set to `0`. The value can be `True` (default) or `False`. |     No |
|is_enhanced|(Optional) Switch for enhancing randomness fixing. The value can be `True` or `False` (default), for example, `is_enhanced=True`. After it is enabled, the status of the built-in random number generators of PyTorch, NumPy, and Python is further fixed. When the same random API is executed multiple times in the same process or different processes, the random values generated each time can be the same. This helps achieve strict reproducibility in more complex random scenarios.|No|

## 3.3 Selecting Bad Cases

In model inference accuracy localization, two different models may exhibit different performance on the same dataset. For example, in a problem scenario, a model may answer a question incorrectly on a dataset evaluation, whereas the benchmark scenario produces the correct answer. Such an instance is referred to as a bad case.

Once a bad case is identified, subsequent issues can be localized as single-case problems. The subsequent operations will be described in the following section.

# 4. Accurate Issue Locating by Scenario

## 4.1 Reproducible Accuracy Issues in Single-Case Scenarios

In model inference, reproducible single-case accuracy issues typically refer to an extremely stable bad case. The details are as follows:

Fixed input: A specific prompt is given (for example, "What is the capital of China?").

Fixed parameters: `Temperature=0`, `Top_p=1`, fixed `Seed` fixed, and consistent `Max Tokens`

Incorrect output: For example, the model consistently answers "Shanghai" instead of the correct answer, "Beijing".

### 4.1.1 Locating Accuracy Issues in vLLM Scenarios

vLLM is a high-performance model inference framework developed by the team at UC Berkeley. It uses innovative memory management and scheduling policies to address the problems of low memory utilization, insufficient throughput, and low concurrent processing efficiency in traditional inference frameworks when deploying foundation models. The core advantages of vLLM lie in its unique memory management mechanism (PagedAttention) and Continuous Batching technology. These two innovations enable memory utilization to reach nearly 100% and the throughput to be 24 times that of traditional frameworks, making vLLM particularly suitable for real-time inference scenarios with high concurrency and low latency. The inference process of vLLM is divided into two main phases: prefill and decode. The prefill phase processes input prompts and generates the initial KV Cache, while the decode phase generates output tokens one by one and continuously updates the KV Cache. Throughout the process, vLLM uses its PagedAttention and Continuous Batching features to efficiently utilize memory resources and fully schedule computing resources.

#### 4.1.1.1 Tool Usage

To locate accuracy issues in vLLM scenarios, the dump and comparison capabilities of the msProbe tool are mainly used. vLLM involves multiple startup modes. The following describes how to enable the tool in each startup mode, using vLLM 0.9 as an example.

##### 4.1.1.1.1 V0 Scenario

- V0, offline mode, TP = 1

Obtain the model:
`model=llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()`

![image.png](https://raw.gitcode.com/user-images/assets/7898473/20631717-1da6-426d-8d03-52a84a0ed29c/image.png 'image.png')

For details about the configuration file, see [Configuration File Introduction](../dump/config_json_introduct.md). For details about msProbe interfaces, see documents related to precision data collection in torch scenarios. You can set the `token_range` parameter in the `start` interface to control the token data to be collected.

- V0, offline mode, TP > 1

The multiprocessing executor `MultiprocessingDistributedExecutor` is used, which introduces a process interval. In this setup, rank 0 resides in the main process, while all other ranks are in subprocesses.

Tool adding position: For data collection on rank 0, the tool can be directly added to the outermost layer of the `generate` function called by LLM. For data collection on other ranks, the tool needs to be added to the `_run_worker_process` function of the subprocess (`vllm/executor/multiproc_worker_utils.py`).

![image.png](https://raw.gitcode.com/user-images/assets/7898473/28a7eec9-1a46-4d7a-a3e7-91237f55b1fb/image.png 'image.png')

- V0, online mode, PP = 1 (TP and DP not limited; `--disable-frontend-multiprocessing` not set)

The multi-process client `MQLLMEngineClient` is used, causing a process interval. All ranks are in subprocesses.

Tool adding position: `run_engine_loop` function of the `MQLLMEngine` class in the subprocess (`vllm/engine/multiprocessing/engine.py`)

![image.png](https://raw.gitcode.com/user-images/assets/7898473/7e7f693f-e67f-4832-ae66-8afaa113069e/image.png 'image.png')

- V0, online mode, PP > 1 or `--disable-frontend-multiprocessing` specified

The multiprocessing executor `MultiprocessingDistributedExecutor` is used, which introduces a process interval. In this setup, rank 0 resides in the main process, while all other ranks are in subprocesses.

Tool adding position: See "V0, online, TP > 1" mode.

##### 4.1.1.1.2 V1 Scenario

- `v1 engine`, `eager(enforce_eager=True)`

###### 1. Add initialization

NPU: `NPUModelRunner.init` function in `vllm_ascend/worker/model_runner_v1.py`:

![image.png](https://raw.gitcode.com/user-images/assets/7898473/84fc8c8a-897e-4ac6-b791-aaf55654eda2/image.png 'image.png')

GPU: `GPUModelRunner.init` function in `vllm/v1/worker/gpu_model_runner.py`

![image.png](https://raw.gitcode.com/user-images/assets/7898473/a8bf2df4-d083-42a4-b146-4d4d5ec9279f/image.png 'image.png')

###### 2. Add the tool enabling code

Add the corresponding code according to the configuration (L0/L1).

```python
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        # L0 use 
        # self.debugger.start(self.model)
        # L1 use
        self.debugger.start()
```

NPU: `NPUModelRunner.execute_model` function in `vllm_ascend/worker/model_runner_v1.py`:

![image.png](https://raw.gitcode.com/user-images/assets/7898473/d3105c07-5c6c-430d-8db5-81121fdf91ca/image.png 'image.png')

Disable the function (place the code before the `return` statement in the `execute_model` function):
![image.png](https://raw.gitcode.com/user-images/assets/7898473/c12564be-8976-49db-98dc-e09313dda74e/image.png 'image.png')

GPU->vllm/v1/worker/gpu_model_runner.py  GPUModelRunner.execute_model
![image.png](https://raw.gitcode.com/user-images/assets/7898473/aa33f488-d3e1-4d5b-b3cb-439a23bdf0cc/image.png 'image.png')

![image.png](https://raw.gitcode.com/user-images/assets/7898473/2fafa44e-8a21-4488-8762-76e5e21eccfd/image.png 'image.png')

#### 4.1.1.2 Locating Process

Generally, there are three phases for locating accuracy problems that can be reproduced in a single case.

![image.png](https://raw.gitcode.com/user-images/assets/7898473/dd6bf3db-d722-43da-9df0-1daa297feffb/image.png 'image.png')

##### 4.1.1.2.1 Pre-locating Operations

The accuracy benchmark may come from either a GPU or a historical version of the NPU baseline that is known to have normal accuracy.

For details about model configuration check and randomness fixing, refer to [Checklist](#31-checklist) and [Preparations for Reproducing a Problem](#32-preparations-for-reproducing-a-problem). In the vLLM scenario, you also need to fix sampling randomness (`temperature` = `0`).
![image.png](https://raw.gitcode.com/user-images/assets/7898473/425c6827-f81f-4c7c-8a5b-5a4ffd8f4a80/image.png 'image.png')

##### 4.1.1.2.2 Locating Operations

- Confirming the first different token

You can print the output token ID sequence of the test case in vLLM. The following uses the V1 scenario as an example.
![image.png](https://raw.gitcode.com/user-images/assets/7898473/1151539d-5ad6-494a-b872-8e504eefe829/image.png 'image.png')

Then, you can easily compare the position of the first different token in the problem scenario with that in the benchmark scenario.

- Using msProbe to dump data

For details about how to enable the msProbe dump feature, see [Tool Usage](#4121-tool-usage). The mix level and statistics mode are recommended for data dump. The data in the following format can be obtained.

![image.png](https://raw.gitcode.com/user-images/assets/7898473/f5c9ea68-a488-41c1-a57a-3166a251edae/image.png 'image.png')

##### 4.1.1.2.3 Analysis of the Result

After the data dump, two files are obtained, one for the problem scenario and the other for the benchmark scenario.
You can use [msProbe](../accuracy_compare/pytorch_accuracy_compare_instruct.md) to compare the data. The following is an example:
`msprobe compare -tp <target_path> -gp <golden_path> [options]`
After the comparison is complete, a comparison table is generated. In the table, you can view the differences (`diff`) of four statistics. By examining the larger `diff`, you can locate the suspicious operator.

![image.png](https://raw.gitcode.com/user-images/assets/7898473/3ed1667c-40e6-44c3-af62-51b39c463f68/image.png 'image.png')

As shown in the preceding figure, matmul is the suspicious operator. You can reproduce the problem on this operator.

### 4.1.2 Locating Accuracy Problems in MindIE Scenarios

Mind Inference Engine (MindIE) is an inference acceleration suite provided by Ascend for various AI scenarios. Through layered open AI capabilities, it supports diverse AI service needs and empowers a large number of models by leveraging the compute of Ascend hardware. MindIE supports multiple mainstream AI frameworks and is compatible with different types of Ascend AI processors, providing multi-layer programming APIs to help users quickly build inference services based on the Ascend platform.

Currently, MindIE is often used together with the [Ascend Transformer Boost (ATB)](https://www.hiascend.com/document/detail/en/canncommercial/850/acce/ascendtb/ascendtb_0001.html) acceleration library to achieve optimal inference performance. The following uses MindIE + ATB as an example to describe how to locate accuracy problems in MindIE scenarios.

#### 4.1.2.1 Tool Usage

The following describes how to use the dump and comparison functions of the msProbe tool to locate accuracy problems in the MindIE scenario.

**Tool Installation**

Currently, the WHL msProbe installation package capable of dumping accuracy data for the ATB model can only be obtained by compiling the source code. The compilation and installation procedure is as follows:

```shell
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe

pip install setuptools wheel

python setup.py bdist_wheel --include-mod=atb_probe
cd ./dist
pip install ./mindstudio_probe*.whl
```

Notes:

(1) The third-party dependencies such as Git, curl, GCC 7.5 or later, and CMake 3.19.3 or later must be installed in the compilation environment.

(2) If the compilation fails due to a security certificate issue, you can temporarily disable the security certificate verification if the environment is secure. The compile command for disabling certificate verification is `python setup.py bdist_wheel --include-mod=atb_probe --no-check`.

**Usage**

1. Create a `config.json` file to set dump parameters. Example:

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "all",
        "ids": "",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    **Parameters in the Dump Configuration File**

    The dump configuration file is a text file in JSON format. The table below describes its parameters.

    | Parameter| Mandatory (Yes/No)| Description|
    | --- | --------- | --- |
    | task         | Optional| Dump task; string; defaults to **tensor**. Possible values:<br> **tensor**: collects the actual data of the input/output tensor of an op.<br> **statistics**: collects the statistics of the input/output tensor of an op.<br> **all**: collects the actual data and statistics of the input/output tensor of an op.|
    | dump_enable  | Optional| Whether to allow data dump; bool; defaults to **false**. Possible values:<br> **true**: allows the collection of the actual data or statistics of the input/output tensor of an op.<br> **false**: does not allow the collection of the actual data or statistics of the input/output tensor of an op.|
    | exec_range   | Optional| Execution round of an op whose data is to be dumped; string; defaults to **0,0**. Possible values:<br> **all**: dumps the precision data of all execution rounds of an op.<br> **none**: does not dump the precision data of all execution rounds of an op.<br> **\<start round\>,\<end round\>**: dumps the op's precision data from the start round to the end round, inclusive.<br> Example: **"exec_range": "0,2"**, indicating that the precision data of the first, second, and third execution rounds of an op is dumped. (Note: the *N*th execution round is numbered *N*-1.)|
    | ids          | Optional| ID of an operator whose precision data is to be dumped; string. The default value is **""**, indicating that the precision data of all layer-level operations is dumped. The value must be in the "\<ID1\>,\<ID2\>" format. One or more IDs can be specified.<br> Example:<br> **"ids": "0"**: dumps the precision data of the op with ID 0.<br> **"ids": "2_1"**: dumps the precision data of the op with ID 1 under the op with ID 2.<br> **"ids": "0,2_1"**: dumps the precision data of the op with ID 0 and the op with ID 1 under the op with ID 2.|
    | op_name      | Optional| Name of an op whose precision data is to be dumped; string type. The default value is **""**, indicating that the precision data of all layer-level operations is dumped. The value must be in the "\<opName1\>,\<opName2\>" format. One or more op names are supported.<br> Example:<br> **"op_name": "word"**: dumps the precision data of the op whose name starts with "word" (case-insensitive).|
    | save_child   | Optional| Whether to dump the precision data of the sub-ops of an op. The value is of the bool type. The default value is **false**. Possible values:<br> **true**: dumps the precision data of the specified op and its internal sub-ops.<br> **false**: dumps only the precision data of the specified op.|
    | device       | Optional| ID of the device whose data is to be dumped. The value is of the string type. The default value is **""**, indicating that the precision data of all devices is dumped. The value must be in the "\<deviceID1\>,\<deviceID2\>" format. One or more device IDs can be specified.<br> Example:<br> **"device": "0"**: dumps the precision data of device 0.|
    | filter_level | Optional| Filtering level of the actual data of the input/output tensor of the dumped op. The value is of the integer type. The default value is **1**. This parameter is valid only when the layer-level operation is specified and **save_child** is set to **true**. Possible values:<br> **0**: does not filter data when collecting the actual data of the input/output tensor of an op.<br> **1**: saves the same tensor only once when collecting the actual data of the input/output tensor of an op.<br> **2**: filters the input/output tensor of a kernel using a condition based on the value **1**.|

2. Run the script for loading the ATB dump module. Example commands:

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh --output=$OUTPUT_PATH --config=$CONFIG_PATH
    ```

    `$MSPROBE_HOME_PATH` indicates the msProbe installation path, `$OUTPUT_PATH` indicates the output path of the dump data, and `$CONFIG_PATH` indicates the path of the dump configuration file.

3. Perform an MindIE inference job.

4. msProbe automatically collects the accuracy data during ATB model running. After obtaining the dump data, you can use the comparison function of msProbe to compare and analyze the dump data. The comparison command is as follows:

    ```bash
    msprobe compare -m atb -gp <goldenDataPath> -tp <targetDataPath> [-o <outputPath>]
    ```

    **Comparison command parameters**

    | Parameter| Mandatory (Yes/No)| Description|
    | --- | --------- | --- |
    | -m or --mode        | Yes| Comparison scenario, which must be **atb**.|
    | -gp or --golden_path| Yes| Benchmark data path, which must be specified to the execution round subdirectory.|
    | -tp or --target_path| Yes| Path of the data to be compared. The path must be specified to the execution round subdirectory.|
    | -o or --output_path | Optional| Output path of the comparison result. The default value is the **output** directory in the current working directory, which is automatically created by the tool.|

For details about how to use the dump and comparison functions, see the [ATB Precision Data Collection](../dump/atb_data_dump_instruct.md) and [ATB Data Precision Comparison](../accuracy_compare/atb_data_compare_instruct.md).

#### 4.1.2.2 Locating Process

If a precision issue cannot be located, you can narrow down the scope hierarchically: first identify the problematic layer, and then locate the specific op within that layer. Here, an op includes both operation and kernel under the layer.

**Layer-level Problem Locating**

Set the `ids` parameter in the dump configuration file to `""` and `save_child` to `false` to dump only the input and output tensors of all layers. The following is an example of the dump output.

![atb_dump_layer.png](https://raw.gitcode.com/user-images/assets/7898473/0e57b323-4901-46c4-b97a-0375eee853ae/atb_dump_layer.png 'atb_dump_layer.png')

Compare the dump data between a layer with a precision issue and a layer without any precision issue. Then, identify the earliest layer where the precision issue first appears. The following is an example of the comparison result.

![atb_compare_layer.png](https://raw.gitcode.com/user-images/assets/7898473/baf5ebd9-8923-4991-8dd6-0204b1724e6c/atb_compare_layer.png 'atb_compare_layer.png')

As shown in the preceding figure, the first layer where a precision problem occurs is `5_Prefill_layer`.

**OP-level Problem Locating**

In the dump configuration file, set the `ids` parameter to the ID of the layer where a precision issue occurs (e.g., `5`) and set `save_child` to `true` to dump the input and output tensors of all operations and kernels under the specified layer. The following is an example of the dump output.

![atb_dump_operation.png](https://raw.gitcode.com/user-images/assets/7898473/61db7d44-55f0-493b-a590-c31c60225bb9/atb_dump_operation.png 'atb_dump_operation.png')

Compare op-level dump data from a normal run and a problematic run to pinpoint the operator likely causing the precision issue. The following is an example of the comparison result.

![atb_compare_operation.png](https://raw.gitcode.com/user-images/assets/7898473/86e3190b-a5f2-41b2-a151-8b8b422dddd5/atb_compare_operation.png 'atb_compare_operation.png')

As shown in the preceding figure, the `0_AddBF16Kernel` operator may have a precision issue.

After locating the suspicious operator, you can use the dump data to reproduce the problem in single-operator scenario. This helps to further determine whether the precision issue originates from the operator itself or from other factors, such as memory corruption.
