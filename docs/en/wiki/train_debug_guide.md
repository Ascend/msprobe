# Foundation Model Training Accuracy Debugging Guide

## 1 Accuracy Issue Overview

---

### 1.1 Definition

With the rapid development of large language model (LLM) technologies, particularly leading applications such as ChatGPT and DeepSeek, LLMs have quickly become a research hotspot in the AI field. Training LLMs requires substantial computing power and involves multiple aspects and technical levels, including data, models, frameworks, operators, and hardware. Due to the large scale of these models, the training process is complex and prone to accuracy issues.

Training accuracy issues are the result of multiple interacting factors. The main symptoms are that training convergence does not meet expectations, manifesting as loss misalignment, NaN values, spikes, and degradation in downstream task performance.

Training accuracy issues can be classified into model accuracy issues and numerical precision issues.

Model accuracy issues refer to problems originating from dataset read operations, training hyperparameters, model structure, or even framework design or usage. These issues have a significant impact on model convergence. Therefore, you need to carefully eliminate and analyze each factor, and adjust your model based on the actual situation.

Numerical precision issues, on the other hand, refer to approximate errors introduced during floating-point computation due to limited word length, computation order, communication order, or mathematical expression approximations. While such numerical approximations may affect model convergence, it should not be assumed that any difference in the computation process will definitely lead to a convergence problem. The numerical precision of operators forms the foundation of the computation process. Generally, operator precision issues are considered one of the sources of model accuracy issues and should therefore be given due attention. However, due to differences in implementation, the numerical computation results of the same process on different hardware platforms (e.g., GPU vs. CPU, or across different GPU versions) are typically not identical. Within a specific tolerance range, these differences do not affect the final model convergence.

To better analyze and locate model accuracy issues and numerical precision issues, and to correctly distinguish normal computational differences from abnormal differences that cause model accuracy issues, this document provides a detailed description of the application scenarios and steps for using the accuracy issue locating toolset. It is designed to help users identify potential accuracy issues either independently or under the guidance of this document.

### 1.2 Standards

When a customer raises a requirement for model training, refer to the standards provided by the product line. Only when these standards are not met should the issue be considered an accuracy issue.

- **Training standards** 
 When receiving customer feedback, verify whether the overall training meets the established training standards and whether the customer's requirement is reasonable. These standards include accuracy testing standards for different scenarios, such as pre-training, fine-tuning, and loss curves.
- **Operator precision standards** 
 When locating a suspicious operator, check whether it meets the operator precision standards, covering operators and framework APIs and providing multiple precision determination methods and testing standards.

### 1.3 Scenario Introduction

Training accuracy issues can be classified into two categories: those with a benchmark and those without.

The benchmark scenario corresponds to migration case, where users migrate a model trained on a benchmark platform (such as GPU or other training frameworks) to NPU for training. 
The scenario without a benchmark corresponds to native development, where users directly build and train models on NPU.

This document focuses on the mainstream migration scenario, which involves a benchmark. In this scenario, the training process and results on NPU are inconsistent with those on the benchmark (GPU or other NPU frameworks), and the deviation exceeds the acceptable tolerance threshold. This situation is referred to as misalignment. The scenario can be further classified into the following types:

- **Overflow or NaN**: Loss or grad norm overflow, or the occurrence of NaN values, happens more frequently than on the benchmark, as shown in the following figure. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/a5183818-b00b-4f5b-bb23-9b8277f0b004/image.png" alt="image.png" width="650"/>  
- **Loss difference in the first step**: Loss in the step 0 or the first several steps is different from that on the benchmark, and the average error is greater than 1%, as shown in the following figure. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/deb21294-900a-4411-9857-890ae31950cd/image.png" alt="image.png" width="400"/>  
- **Loss difference in long-term stable training**: The loss difference between the early-stage loss fitting and the benchmark gradually increases over time, and the average error exceeds 1%, as shown in the following figure. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/de267ea9-69ba-4f15-a179-54efc1a48528/image.png" alt="image.png" width="400"/>  
- **Spikes**: The loss or grad norm sharply increases and then quickly decreases more frequently compared to the benchmark, as shown in the following figure. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/eb705a99-b07c-4014-a966-4136825144a5/image.png" alt="image.png" width="400"/> 
- Compared to the benchmark, the loss difference during training is small, but the downstream task performance is poor, as shown in the following figure. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/e46ee64d-2c1d-4a76-8e99-eec2236165dd/image.png" alt="image.png" width="400"/> 

It is worth noting that even when the same type of problem occurs, the root causes are often complex and vary from case to case. For details, see [Root Causes](#51-root-cause-introduction). This document describes the overall fault locating approach and standard process for identifying model training accuracy issues, along with the usage of the training accuracy tools involved. It is designed to help users quickly become familiar with and master the accuracy locating process.

## 2. Steps for Locating Accuracy Issues

---
The following figure shows the overall process of locating model training accuracy issues. 
 <img src="https://raw.gitcode.com/user-images/assets/7898473/83c3651f-ebc3-4683-99fb-6b042d9df5f4/image.png " alt="image.png" width="800"/>  
It mainly describes the specific steps for locating accuracy issues in the migration scenario, helping you better understand the principles and methods of using the tool and apply them to other more specific and complex scenarios.

### 2.1 Checklist

Before locating benchmark accuracy issues, eliminate the interference of other non-operator factors. Currently, most accuracy issues are caused by factors such as model hyperparameters, third-party library versions, environment variables, data reading, and model structure inconsistency. To facilitate issue locating, check the training environment and preparations according to the following checklist. 
In addition, issue locating is primarily performed based on comparison between the benchmark device and the NPU device. Therefore, the prerequisite for locating is to have both the benchmark and NPU training environments ready.

- Comparison of training hyperparameters and environment variables 
 You can use Beyond Compare to compare the training hyperparameters and environment variables in the training logs or startup scripts of the two devices, or use the [script comparison tool](#32-scenario-specific-debugging-cases) to perform automatic comparison. For details about common training hyperparameters, see [Model Hyperparameters](#52-model-hyperparameters).
- Comparison of third-party library versions 
 Use `git branch` to check whether the versions of third-party libraries such as MindSpeed-LLM, Megatron and DeepSpeed are consistent with those of the benchmark.
 Use `pip list` to check whether the versions of third-party libraries such as PyTorch and PTA (PyTorch-NPU) are consistent with those of the benchmark. Alternatively, you can use the [script comparison tool](#48-script-comparison-tool) to perform automatic comparison.
- Data reading check 
 Examine the data that is read from the dataset and fed into the model for training. Generally, you can use the accuracy collection tool to capture the initial input data, or directly call the model's `forward` API in the code to save or print the input tensor for dataset verification. Also, you can use the [script comparison tool](#48-script-comparison-tool) for automatic comparison.
- Model structure check 
 Print the model structure in both training scenarios for comparison.
- Weight initialization alignment 
 Check whether the initialized weights are consistent before training. Ensure that the same pre-trained model is loaded or the same random seed is used for initialization. For details about how to fix randomness, see [Fixing Randomness](#221-fixing-randomness). During the check, you can use the [script comparison tool](#48-script-comparison-tool) for automatic comparison.
- Environment version update 
 Update the environment version only when conditions permit. Based on past troubleshooting for accuracy issues, most issues have been fixed in the latest version. Therefore, you are advised to install the CANN, driver, and PTA packages of the latest commercial version.

### 2.2 Preparations for Reproducing the Problem

Once non-operator factors are ruled out, move to the next step. First, make sure the problem is reproducible and the training process matches what the customer described.

#### 2.2.1 Fixing Randomness

To reproduce the problem, you need to fix the random operations to ensure experimental reproducibility. The steps with randomness include model parameter initialization, `dropout` layer, and data batch loading sequence. 
The involved operations are as follows:

- Fix random seeds, such as `np.random.seed`, `torch.manual_seed`, and `torch_npu.npu.manual_seed`.
- Disable the `dropout` layer.
- Disable `shuffle` during data loading (`shuffle` = `False`).

It is recommended to use the tool to automatically conduct these operations seeds. For details, see [seed_all Interface](#223-seed_all-interface).

#### 2.2.2 Enabling Determinism

During problem reproduction, you are advised to enable operator computation determinism and communication determinism. Both of them need to be fixed as early as possible before the code starts training.

- Computation determinism:

```python
torch.use_deterministic_algorithms(True)
```

- Communication determinism:

```bash
export HCCL_DETERMINISTIC=TRUE
```

Note: Some operators do not support deterministic computation. For such cases, see [Operator Determinism Issues](#2322-operator-determinism-issues) for troubleshooting.

#### 2.2.3 seed_all Interface

The msProbe toolkit provides the `seed_all` interface to quickly fix all random seeds, `dropout` layer, and operator computation and communication determinism on the network. In addition to using `seed_all`, you only need to manually disable `shuffle` for datasets.

Usage:

```python
from msprobe.pytorch import seed_all
seed_all(seed=1234, mode=True, rm_dropout=True) 
```

Parameters:

| Parameter| Description| Mandatory (Yes/No)|
| ---  | --- | --- |
|seed| Random seed The default value is `1234`.| No|
|mode| Deterministic computation mode. The value can be `True` or `False` (default). This mode includes both operator computation determinism and communication determinism.| No|
|rm_dropout| Switch for controlling dropout invalidation or validation. After this function is enabled, the dropout probability is automatically set to `0`. The value can be `True` (default) or `False`.| No|

Fixed random seeds: 
The following table lists the random seeds that can be fixed by `seed_all`.

|API|Random Seed|
| --- | --- |
|os.environ['PYTHONHASHSEED'] = str(seed)| Disables `hash` randomization in Python.|
|random.seed(seed)| Sets the seed of the random generator in `random`.|
|np.random.seed(seed)| Sets the seed of the random generator in NumPy.|
|torch.manual_seed(seed)| Sets the random seed of the current CPU.|
|torch.cuda.manual_seed(seed)| Sets the random seed of the current GPU.|
|torch.cuda.manual_seed_all(seed)| Sets the random seed of all GPUs.|
|torch_npu.npu.manual_seed(seed)| Sets the random seed of the current NPU.|
|torch_npu.npu.manual_seed_all(seed)| Sets the random seed of all NPUs.|
|torch.backends.cudnn.enable=False| Disables cuDNN.|
|torch.backends.cudnn.benchmark=False |Selects deterministic algorithm of cuDNN.|
|torch.backends.cudnn.deterministic=True| `cuDNN` uses only the deterministic convolution algorithm.|

#### 2.2.4 Scaling-Down the Size

For large clusters—such as those with thousands or even tens of thousands of cards—if accuracy issues occur, you need to scale down the cluster size to reproduce and locate the problems. 
A common practice is to keep partitioning parameters (e.g., `TP`, `PP`, `CP`, `SP`, and `EP`) unchanged while reducing the batch size or directly reducing the number of model layers. During this pruning process, ensure that the issues remain reproducible. Ultimately, select the training parameters with the smallest possible scale that still allow the issues to be reproduced.

### 2.3 Locating Accuracy Issues by Scenario

After performing the operations described in the previous section, most issues can be successfully reproduced. However, some difficult issues remain non-reproducible—for example, those where the occurrence location and symptoms are inconsistent, or issues that can only be reproduced intermittently.

- For stable and reproducible issues, the basic principle of accuracy issue analysis is to collect data during both benchmark and NPU training, perform fine-grained comparisons, and detect potentially abnormal computation APIs and their associated operators.
- For unstable or non-reproducible issues, the root causes are typically related to special operators that do not support deterministic computation, memory corruption, hardware-induced multi-bit flipping, or bugs in third-party libraries and frameworks. In such cases, you need to examine each potential cause systematically.

You are advised to use msProbe provided by the Ascend MindStudio Training Tools to collect data and analyze issues during training. For details, see [msProbe Usage](#4-msprobe-usage).

#### 2.3.1 Stable Reproduction

After the [checklist](#21-checklist) has been confirmed and the issue can be stably reproduced, follow the figure below to locate the issue. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/d7955340-9bde-479c-801e-f22573f50190/image.png" alt="Your image title" width="800"/>  

##### 2.3.1.1 Overflow or NaN

Symptom

If other non-operator factors are excluded according to the [checklist](#21-checklist) and the randomness is fixed, check whether the loss NaN or gradient overflow occurs more frequently than that in the benchmark. Generally, the loss scale continuously decreases. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/a5183818-b00b-4f5b-bb23-9b8277f0b004/image.png" alt="image.png" width="650"/>  

Troubleshooting

To locate an overflow issue, find the first step where NaN occurs:

- If NaN occurs on NPU but not on GPU, the first step where NaN occurs is considered as the problematic step.
- Both NPU and GPU have NaN errors, but the number of NaN errors in NPU is significantly greater than that in GPU. The first step where NPU has NaN errors but GPU does not is considered as the problematic step.

After determining the step to be analyzed, perform the following steps:

1. Collect the data of the step for analysis and comparison.

    - Use the [precision collection tool](#43-precision-collection-tool) to collect the forward and backward input and output of the overflowed step.
        - If the overflow disappears after the tool is added, it is suspected that memory corruption occurs. In this case, see [Memory Corruption](#2321-memory-corruption) for troubleshooting.
        - If the issue can still be reproduced after the tool is added:
            1. Use the overflow analysis function of the [hierarchical visualization tool](#44-hierarchical-visualization-tool) or manually search for Inf or NaN to locate the position where the overflow occurs first.
                - `weight`: The backward gradient in the previous step may be abnormal. In this case, switch to the previous step and collect the gradient data again.
                - `input`: There may be special operators that are not collected. In this case, analyze the source of `input` based on the code stack.
                - `output`: Analyze the operator where `output` locates. 
                For details about how to identify the `input`, `weight` and `output` positions in the `dump.json` file, see the description of the `dump.json` statistics in [precision collection tool](#43-precision-collection-tool).
            2. If no suspicious operator is located, the problem may be caused by accumulated errors. In this case, further analyze the problem by referring to "loss alignment" in [First Step Loss Difference](#2312-first-step-loss-difference)/[Loss Difference in Long-Term Stable Training](#2313-loss-difference-in-long-term-stable-training).
    - If the model scale is large, the precision collection tool takes a long time, or the problem cannot be stably reproduced, you can also use the [training status monitoring tool](#46-training-status-monitoring-tool) or manually mount hooks to collect gradients of each layer by referring to [Non-tool Method](#49-non-tool-method) to check whether there are layers with abnormal gradients.

2. For `NaN` errors, if the root cause is not found after the preceding checks, check the following special cases:

    - In Megatron and DeepSpeed models, `overlap` parameters (such as `overlap-param-gather` and `overlap-grad-reduce`) have high risks. You can disable these parameters first.
    - If a NaN error occurs when mixed precision is used in the FSDP framework, you are advised to first check for memory corruption caused by the PTA framework by switching the PTA version.
    - The FA (`npu_fusion_attention`) fusion operator is complex, and parameter passing errors may occur during its use. You can disable the FA branch to determine whether the fault is caused by FA. If it is, check whether there are usage errors by referring to the [official FA documentation](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fusion_attention.md).
    - Ensure that the Inf/NaN mode or non-saturation mode is enabled. For details, see [Appendix - Non-Saturation Mode](#53-non-saturation-mode).

##### 2.3.1.2 First Step Loss Difference

Symptom

After other non-operator factors are excluded according to the [checklist](#21-checklist) and the randomness is fixed, the loss for the first step or first several steps differs between GPU and NPU, with an average relative error exceeding 1%, as shown in the following figure. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/deb21294-900a-4411-9857-890ae31950cd/image.png" alt="image.png" width="400"/>  

Troubleshooting

Do as follows:

1. Use the [precision collection tool](#43-precision-collection-tool) to collect the step where the difference first occurs (with a relative error greater than 1%).

    - If loss in step 1 (corresponding to step 0 in training) is not aligned, use the tool to collect the statistics of NPU and benchmark in step 1, and check the forward computation result.
    - If loss in step 1 is aligned but not aligned in step 2, use the tool to check the backward computation result in step 1 and the forward computation result in step 2 of NPU and benchmark.
    - Use the same method to analyze steps 3 and 4. This method can be used to locate the issue as long as it occurs within a limited number of steps.

2. Use the analysis tool to compare the results and identify suspicious operators.

    - Use the [hierarchical visualization tool](#44-hierarchical-visualization-tool) to perform a graph comparison, and use the precision filtering function (color from dark to light) and the first node where the precision difference occurs for troubleshooting and analysis.
    - You can also use the [comparison tool](#45-precision-comparison-tool) to perform a table comparison and analyze the input and output precision differences for troubleshooting and analysis.

3. If no suspicious operator is identified, it is recommended to use the [precision pre-check tool](../accuracy_checker/pytorch_accuracy_checker_instruct.md) for further troubleshooting.

##### 2.3.1.3 Loss Difference in Long-Term Stable Training

Symptom

After other non-operator factors are excluded according to the [checklist](#21-checklist) and the randomness is fixed, loss can be aligned at the early stage of GPU and NPU training, but it cannot be aligned in the later stage of training. The average relative error in the later stage is greater than 1%, as shown in the following figure. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/de267ea9-69ba-4f15-a179-54efc1a48528/image.png" alt="image.png" width="400"/>  

Troubleshooting

1. Select a tool and collect data based on the actual situation.

    - [Precision collection tool](#43-precision-collection-tool): Applicable to scenarios where the number of steps is determined, for example, if loss or grad norm changes significantly during the process.

        (1) Determine the number of steps to be collected (take the smaller step number of the two).

            - loss jump: Collect the backward data of the previous step and the forward data of the current step. 
            - Grad norm jump: Collect the backward data of the current step. 

        (2) Determine the data volume to be collected.

            - For a small-scale model, directly collect `mix` (or API- and module-level) statistics. 
            - For a large-scale model, collect module-level statistics first, locate the module, and then collect API-level statistics internally. 

        (3) After identifying suspicious operators through visualization or comparison tool, collect the specific `tensor` values for further single-operator analysis.

    - [Training status monitoring tool](#46-training-status-monitoring-tool): Applicable to large-scale scenarios where the number of dump steps to be collected is uncertain.
        - If grad norm becomes abnormal first and then loss becomes abnormal, collect gradient data during training.
        - If loss is abnormal, collect activations and weight data during training.

2. For `dump` data analysis, refer to the previous section on using the hierarchical visualization tool or comparison tool. For training status monitoring data analysis, refer to [section 4.6](#46-training-status-monitoring-tool).
3. If benchmark data collection is costly and cannot be performed frequently due to uncertainty in the number of steps, use the precision pre-check tool and the non-benchmark tool for further analysis.
4. If the root cause is not found after the preceding checks, check the following two special cases:

    - Check the optimizer. For example, convert the Adam optimizer to SGD, replace the Adam fusion operator with smaller ones, or disable the optimizer to demarcate issues across forward and backward passes.
    - Check the `Matmul` staggering policy. If a suspicious `Matmul` operator is located but cannot be ruled on by single-operator verification, you can try disabling the staggering policy.

    ```bash
    export CLOSE_MATMUL_K_SHIFT=1
    ```

#### 2.3.2 Unstable Reproduction

If the issue cannot be stably reproduced after the [checklist](#21-checklist) has been confirmed and the pre-operations for reproducing the issue have been completed, follow the figure below to locate the issue. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/99f2e246-592c-4ee3-aa22-bc47fcceffc4/image.png" alt="Your image title" width="800"/>  

##### 2.3.2.1 Memory Corruption

Symptom

Memory corruption usually occurs in precision issues that are characterized by overflow or NaN. It can be further classified into the following two types:

- If the issue is stably resolved after enabling stream synchronization or using the precision collection tool, there is a high probability that the issue is caused by memory corruption in the computation or communication stream, incorrect calculation of memory allocation offsets within the framework, or reading of dirty data due to a lack of communication protection.
 When synchronization is added, operators and communication are completely isolated from each other, which prevents the issue from recurring. The command for enabling stream synchronization is as follows:

    ```bash
    export ASCEND_LAUNCH_BLOCKING=1
    ```

- However, memory corruption inside an operator cannot be avoided even with stream synchronization. This is more likely with non-integral block sizes, complex computation, or data type changes in core-based and UB-based computation.

Troubleshooting

1. Enable asynchronous dump. Specifically, add `async_dump: true` to `config.json` of the [precision collection tool](#43-precision-collection-tool). In addition, collect tensor data from the first NaN occurrence in two training runs: one with stream synchronization (no NaN) and one without (NaN present). If `async dump` still affects reproduction, manually mount hooks or use `print` statements.
2. Analyze whether the tensor difference pattern suggests memory corruption (e.g., regular corruption patterns: multiples of 2048, row-aligned, column-aligned).
3. Use msProf in conjunction with msInsight to examine the computation parallelism. For detailed instructions, refer to the [msProf User Guide](https://gitcode.com/Ascend/msprof) and [msInsight User Guide](https://gitcode.com/Ascend/msinsight).
4. Add pointer address printing. For NaN occurrence locations, intrusively modify PyTorch or PTA source code to add prints.
5. Use the [operator competition tool](https://gitcode.com/Ascend/mssanitizer/blob/26.0.0/docs/zh/quick_start/mssanitizer_quick_start.md) to check whether the operator's implementation (intra-pipeline for instruction execution, inter-pipeline for operator transfers, and inter-core for `aicube` and `aivector` parallelism) has anomalies indicating memory corruption.
6. If the root cause is still not found, refer to a more detailed memory issue debugging guide.

##### 2.3.2.2 Operator Determinism Issues

Issue Description

If results still differ between repeated training runs after using the tool's automatic fixing function and enabling stream synchronization, use the precision collection tool with `md5` mode to collect data from two repeated runs to identify the first anomalous operator. If the operator is a special random operator or has fixed position with small result variations, suspect an operator determinism issue.

Troubleshooting

Use the precision collection tool with `md5` mode to collect data from two repeated training runs. Compare to find the first operator with consistent input but inconsistent output:

- If the first discrepancy is in input, check the code stack for operators not captured.
- If the first discrepancy is in output, the operator is suspicious.

If the operator is a special random operator or has fixed position with small variations, try the following solutions:

- Special random operators (e.g. `torch.randn`) 
 Even with fixed random seeds, different hardware may generate different random numbers, as shown below. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/f690a76c-5a00-48ab-ae0b-f77512715275/image.png" alt="Your image title" width="500"/>  
    According to the figures, `torch.randn` generates different random tensors even after its random seed is fixed on NPU and GPU. In such cases, generate the random tensor on CPU first, then move it to the target device. This ensures the same tensor is generated on the host side and remains identical after being moved to NPU or GPU.
- Operators not yet supporting deterministic computation 
 Some operators besides random operators do not support deterministic computation yet, causing minor variations between runs. If such an uncommon operator (e.g., `MSDA`, `grid_sample`) is identified, try moving it to CPU or consult operator support for alternative implementations.

##### 2.3.2.3 Hardware Issues

**Issue Description**

For large-scale cluster training, occasional hardware faults (e.g., multi-bit flipping, power supply issues) may cause loss spikes or divergence. These issues typically disappear after switching devices.

Troubleshooting

For large cluster tasks, use hardware stress testing to identify anomalous nodes:

- Model stress testing: Run grouped single-node or multi-node tasks to find servers with inconsistent accuracy compared to most others.
- Operator stress testing: If instability occurs with a specific operator that cannot be reproduced in isolation, run single-operator stress tests on grouped single nodes.
- Command stress testing: Use the `ascend-dmi` command to repeatedly stress-test `aicore`:

 ```bash
 ascend-dmi -dg -i aicore -s -sc 60 -q
 ```

Additionally, gradually disable related communication links to debug hardware-related communication issues:

- Disable RoCE:

 ```bash
 export HCCL_INTRA_ROCE_ENABLE=0
 ```

- Disable PCIe:

 ```bash
 export HCCL_INTRA_PCIE_ENABLE=0
 ```

##### 2.3.2.4 Training Framework Debugging

**Issue Description**

For models using frameworks like Megatron or DeepSpeed, certain high-risk factors can be checked specifically.

Troubleshooting

- Megatron
    - Remove `overlap` parameters (common when training is normal but inference results are abnormal). For example, `overlap_param_gather` once caused incorrect checkpoint saving due to incomplete communication.
    - Simplify TP/PP/SP/CP/EP partition strategies.
- DeepSpeed
    - Remove `overlap` parameters.
    - Ensure `bucket_size` meet requirements.
    - Try different partition strategies (e.g. Zero1/2/3/offload).

##### 2.3.2.5 Training Phase/Model Phase/Version Debugging

**Issue Description**

If the root cause is still not found after all the above debugging steps, narrow down the scope by checking the training phase, model phase, or version.

Troubleshooting

- Training phase debugging
    - Disable the optimizer or set learning rate to 0 to demarcate forward/backward issues.
    - Replace the optimizer to demarcate optimizer issues (e.g., Adam -> SGD, Adam fusion operator-> small operators).
- Model phase debugging
    - Check the model in stages: layer dropping, binary search, attention excluding, etc.
    - Move to CPU: when unsure which operator causes the issue, use binary search to move parts of the network to CPU (if testing cost permits).
    - If the issue occurs during backward, freeze gradients using binary search.
- Version debugging 
 If the issue is clearly tied to a specific version but the root cause cannot be identified, use binary search on versions to identify the problematic change. For CANN, replace operator implementation with the corresponding version by replacing the merged .so file. For PyTorch, compile packages with specific changes. However, this method is less effective when many changes are included in a short period or a single change has broad impact.

#### 2.3.3 Reinforcement Learning (RL) Scenario

**Issue Description**

After ruling out non-operator factors, if RL training shows decreased `reward` relative to the benchmark or an increasing `logp_diff`, accuracy debugging is required. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/a9411220-551b-4810-9560-6e8d15194203/image.png" alt="Your image title" width="600"/>

The figure below shows the overall debugging process.
<img src="https://raw.gitcode.com/user-images/assets/7898473/c1041303-ad83-4b31-ab0d-bc46bbecc50a/image.png" alt="Your image title" width="700"/>  

##### 2.3.3.1 Basic Inference Debugging

In the RL pipeline, `response` is a key intermediate output, affecting both inference quality and subsequent `reward` operation and training. Therefore, basic inference correctness must be ensured first.

Check the inference module if any of the following conditions are met:

- First inference step produces gibberish.
- First 20 tokens of the first inference step differ from the benchmark.
- Inference dataset evaluation accuracy falls below the benchmark.

If none of the above occurs, proceed to [Reward Debugging](#2332-reward-debugging). Otherwise, collect inference data and compare it with benchmark data.

The inference framework in RL is typically vLLM (vLLM-Ascend on Ascend) or SGlang.

Data collection:

- For vLLM-Ascend inference data collection, refer to [msProbe Debugging Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html).
- For details about how to collect inference data of the SGLang framework, see [Precision Data Collection in SGLang](../dump/sglang_eager_dump_instruct.md).

When using the precision collection tool, set the collection level to `mix` or `L0` (at least module-level data needs to be collected) to enable layer-by-layer comparison. Example `config.json`:

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [0],
    "level": "mix",  # Can be switched to "L0"
    "statistics": {
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

Data comparison:

- Same framework (e.g., vLLM-Ascend vs vLLM) 
 Refer to analysis point 2 in [First Step Loss Difference](#2312-first-step-loss-difference).
- Different frameworks (e.g., vLLM-Ascend vs SGlang) 
 NPU and benchmark will have many layer name/structure differences. Use the [hierarchical visualization tool](#44-hierarchical-visualization-tool) with the "point-to-point matching" feature to manually align nodes that are not automatically matched. 
    <img src="https://raw.gitcode.com/user-images/assets/7898473/1c469d88-ebc5-45eb-b19d-6d40f1568755/image.png" alt="Your image title" width="800"/>  

For more details, refer to [Foundation Model Inference Accuracy Debugging Guide](../wiki/infer_debug_guide.md).

##### 2.3.3.2 Reward Debugging

In RL, `reward` takes `prompt` + `response` as input and scores the output. Anomalous `reward` directly affects the entire RL objective. Therefore, after inference, ensure that `reward` computes correctly.

Check `reward` if the following condition is met:

- First-step inference results are consistent (or instrumented), but `reward` differs significantly.

If not triggered, see [Basic Training Debugging](#2333-basic-training-debugging). If triggered, compare the `reward` calculation logic with the benchmark.

`reward` debugging depends on the dataset type.

|Dataset Type| Scoring Method| Corresponding Logic| Next Step|
| --- | --- | --- | --- |
|Math| Rule-based| Highly structured math answers; efficient rule-based scoring typically uses string matching.| Align rule-based scoring code with benchmark.|
|Code| Sandbox-based| Correctness cannot be determined by string matching; sandbox execution with test cases is required.|   Align sandbox execution and test case logic with benchmark.|
|General| Model-based| No single correct answer (e.g., chat, copywriting); uses a reward model learned from human preferences.| 1. Check if the scoring model has randomness enabled.<br>2. Refer to [First Step Loss Difference](#2312-first-step-loss-difference) for data collection and comparison.|

##### 2.3.3.3 Basic Training Debugging

In RL, the training phase updates weights and gradients. Basic training must be free from obvious anomalies in the first step.

Check the training module if the following condition is met:

- First-step inference results are consistent (or instrumented) and `reward` is normal, but `pg_loss` or `gnorm` differs significantly from the benchmark.

If not triggered, proceed to [Resharding Weight Synchronization](#2334-resharding-weight-synchronization). Otherwise, compare the first step of forward/backward training with the benchmark.

**Prerequisites**

In verl training, data shuffling or reorganization improves generalization and hardware utilization but can cause input misalignment (e.g. disordered data in a batch and `batch_size` changes) during comparison. Before debugging training-related issues, ensure the following switches are set/checked:

- Dataset `shuffle`: Randomly shuffles batches after reading.

  ```bash
  data.shuffle=False  # Must be disabled.
  ```

- `balance_batch`: Exchanges sequences across DP domains in the global batch to balance tokens per rank.

  ```bash
  The trainer.balance_batch=False  # Must match benchmark; must be disabled for training-inference alignment.
  ```

- `use_dynamic_bsz`: Dynamically adjusts micro-batch sizes based on `max_token_len` to prevent uneven token distribution.

  ```bash
  actor_rollout_ref.actor.use_dynamic_bsz=False  # Must match benchmark; must be disabled for training-inference alignment.
  ```
  
Training backends in RL typically use FSDP or Megatron (MindSpeed/MindSpeed-LLM with patches on Ascend).

Data Collection

- FSDP backend: The collection file in verl is `verl/workers/fsdp_workers.py`. Refer to [verl (FSDP)](../dump/verl_fsdp_consistency_preprocess_dump.md) for insertion points.
- Megatron backend: The collection file in verl is `verl/workers/megatron_workers.py`. Insertion is similar to FSDP.

Data Analysis

- Same framework (e.g., Megatron + MindSpeed vs Megatron)
    - For small-scale training, refer to [First Step Loss Difference](#2312-first-step-loss-difference) for data analysis.
    - For large-scale training, such as training with a large number of ranks or a large model, use the [Trend Visualization Tool](../accuracy_compare/trend_visualization_instruct.md) for analysis.
- Different frameworks (e.g., Megatron vs FSDP) 
 NPU and benchmark will have many layer name/structure differences. Use the [hierarchical visualization tool](#44-hierarchical-visualization-tool) with the "point-to-point matching" feature for comparison.

##### 2.3.3.4 Resharding Weight Synchronization

The core RL loop is: training updates weights → inference evaluates → `reward` feedback optimizes training. This requires accurate weight synchronization from the `actor` training side to the `rollout` inference side at each step. If weights used for inference are inconsistent with the latest weights used for training, the inference output will deviate from the expected result. 
This process is called resharding weight synchronization, with three core responsibilities:

- **Weight sharding adaptation**: Splits/merges multi-rank training weights according to the inference parallel strategy.
- **Weight read/write control**: Manages and synchronizes disk and HCCL read/write operations to prevent truncation, offset, or loss.
- **Weight name conversion**: Unifies weight names between training and inference to avoid loading failures.

In a distributed inference framework, `dummy` (fake data) can be used for the first execution to estimate the memory allocated and `capture size` of the computational graph for subsequent formal inference. In the verl framework (with `actor_rollout_ref.rollout.load_format` hyperparameter), inference supports two weight initialization formats:

- `safetensors`: Standard format; reads full pretrained weights.
- `dummy`: Random initialization based on shape; does not load real values.

Check resharding weight synchronization if either condition is met:

- Condition 1: `dummy` produces gibberish in the first inference step; switching to `safetensors` restores normal output.
- Condition 2: Both `dummy` and `safetensors` produce gibberish in the first step, but isolated inference (outside RL) is normal.

If neither condition is met, proceed to [Training-Inference Consistency Debugging](#2335-training-inference-consistency-debugging). Otherwise:

- Condition 1 – Check weight objects: 
   Generally, certain layer weights were not synchronized. During resharding, issues like truncation, offset, or pointer separation may occur. If the weight pointer registered in `parameters` differs from the weight pointer used in computation, printing `named_parameters()` may show normal updates while actual computation uses stale weights. In `safetensors` mode, this may not cause obvious issues in the first step, but in `dummy` mode, it leads to gibberish.
      
- Condition 2 – Check weight read/write and sharding: 
   Generally, weights were synchronized but with incorrect values. Try switching between disk and HCCL read/write configurations, or check whether partition parameters like TP/PP are aligned.

##### 2.3.3.5 Training-Inference Consistency Debugging

Mainstream RL algorithms are based on the `On-Policy` assumption, which requires that the behavioral policy for sampling (`rollout` inference) and the target policy for gradient computation (`actor` training) remain consistent to ensure unbiased gradient estimates and training stability. This consistency is referred to as training-inference consistency. 
The main metric for training-inference consistency is `logp_diff`. Anomalous `logp_diff` indicates a discrepancy between training and inference, requiring root cause analysis. The formula is:
<img src="https://raw.gitcode.com/user-images/assets/7898473/98058476-f95a-44ff-b5d0-7dfb79c1066c/image.png" alt="Your image title" width="400"/>  
where `M` is `response_mask`.

In verl:

- Enabling `logp_diff` monitoring:

  ```bash
  actor_rollout_ref.rollout.calculate_log_probs=True  # The default value is False.
  ```

- `logp_diff` implementation:

  ```python
  mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
  mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
  log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
  metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
  ```

Check training-inference consistency if:

- `logp_diff` is unusually large (>0.01) or differs significantly from the benchmark.

If not triggered, proceed to [Long-term Training Debugging](#2336-long-term-training-debugging). Otherwise:

- Large `logp_diff` in the first step 
 Refer to [Training-Inference Consistency Comparison](#23352-training-inference-consistency-comparison) for comparing NPU data at step 0 (assuming checks in sections 2.3.3.1–2.3.3.4 passed without issues).
- Normal `logp_diff` initially but increasing later 
 Forcibly set `LR` to `0` and observe `logp_diff` changes:
    - Still abnormal → issue unrelated to training backward; likely inference-side.
        - Refer to [KV Cache Debugging](#23351-kv-cache-debugging) to check KV cache read/write logic in decode.
        - Refer to [Training-Inference Consistency Comparison](#23352-training-inference-consistency-comparison) to compare NPU data at training steps that cause breakdown.
    - Becomes normal → possible undetected resharding issue.
        - Refer to [Resharding Weight Synchronization Debugging](#2334-resharding-weight-synchronization) to compare actual computation weights with saved weights to see numerical differences.

###### 2.3.3.5.1 KV Cache Debugging

In inference:

- Prefill: Clears previous KV cache and writes current context.
- Decode: Incrementally reads/writes KV cache.

Implementations use `state blocks` to manage the cache. If cache clearing is delayed, reuse is incorrect, or address offsets occur, decode-phase reads become abnormal, causing long-term drift and increased later `logp_diff`. 
Since KV cache is inference-specific, such issues inevitably trigger `logp_diff` inconsistency.

Debug KV cache by:

1. Comparing NPU and benchmark KV cache read/write code for logic differences.
2. Forcing per-statement KV cache clearing.

###### 2.3.3.5.2 Training-Inference Consistency Comparison

Current training-inference comparison only supports the prefill phase. Determine whether the inconsistency occurs only in decode or also in prefill as follows:

Set `max_response_len = 1` so the inference model runs only prefill, and observe `logp_diff`:

- Still abnormal → prefill also has issues; proceed with training-inference consistency comparison.
- Becomes normal → likely a decode-specific issue; check [KV cache read/write logic](#23351-kv-cache-debugging).

Comparison Prerequisites

In verl RL training, `batch` dimensions are flattened to `token` dimensions. For comparison, `token` and `feature` dimensions must match between training and inference.

Key differences between training and inference `forward token` sequences in standard RL training:

- Single prompt
    - Inference: prefill and *N**decode phases. 
    In the prefill phase, the `token` dimension of `forward` is `prompt_len`. In the `decode` phase, the `token` dimension of `forward` is `1`. After the phase ends, `prompt` + `response` is obtained.
    - Training: full `forward`.
      - If `pad` does not exist, the `token` dimension is `prompt_len` + `response_len`.
      - If `pad` exists, the `token` dimension is `max_prompt_len`.
- Multiple prompts
    - Inference: concatenates multiple prompts within a single request (up to budget). Prefill token dimension = sum of prompt lengths (e.g. `prompt1` + `prompt2`); Decode forward token dimension = 2. Then, `prompt1` + `response1` and `prompt2` + `response2` are obtained.
    - Training: may have mini-batch loops + gradient accumulation loops that split prompt data, making shape comparison impossible.

Additionally, `forward feature` dimensions may differ if partition strategies (e.g., TP) differ.

Prerequisites for training-inference comparison:

1. Training batch not split

    - Ensure that the number of `mini batch` used for gradient update in each training epoch is 1 (`mini_batch_num`= `1`).

    ```bash
    mini_batch_num = train_batch_size/train_ppo_mini_batch_size
    ```

    - Ensure that the number of gradient accumulation steps is 1 (`gac` = `1`).

    ```bash
    gac = train_ppo_mini_batch_size*n_resp_per_prompt/train_ppo_micro_batch_size_per_gpu/DP
    ```

    `DP` calculation formulas for different training backends:
    - FSDP (data parallelism): `DP = world_size`
    - Megatron (model parallelism): `DP = world_size/TP/PP/CP`

    Corresponding hyperparameters in verl:

    ```bash
    data.train_batch_size=${train_batch_size}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.ppo_epochs=1  # Default to 1.
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    ```

2. Disable padding and dynamic batching in training:

    ```bash
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=False
    ```

3. Identical partition strategies (TP, PP, CP, etc.) between training and inference
4. Inference runs prefill only (hyperparameter in verl script):

    ```bash
    data.max_response_length=1
    ```

5. No response difference between training and inference 
    Two alignment solutions:

    - Solution 1: Train on single prompts only.
    - Solution 2: After inference completes prefill + decode to get `prompt` + `response`, rerun prefill with `max_response=1`.
    
    Solution 2 involves repeated inference, which affects performance. Therefore, solution 1 is recommended.

    Implementation of Solution 1 
    In the verl framework, modify the training data input and adapt loss computation. For different backends:
    - FSDP: `verl/workers/actor/dp_actor.py`
    - Megatron: `verl/workers/actor/megatron_actor.py`

    The following uses FSDP as an example to describe code modifications:

    ```diff
    ...
    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """..."""
            response_length = micro_batch["responses"].size(-1)
    +        if "responses" in micro_batch and micro_batch["responses"] is not None:
    +            response_length = micro_batch["responses"].size(-1)
    +        else:
    +            response_length = 0
    multi_modal_inputs = {}
    ...

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """..."""
        # set to eval
        self.actor_module.eval()

    +        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
    +        if compute_prompts_only:
    +            if "responses" in data.batch:
    +                responses_len = data.batch["responses"].size(1)
    +                data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
    +               data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
    +                if data.batch["position_ids"].dim() == 3:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
    +                else:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
    +                # remove responses from batch
    +                data.batch["responses"] = None
    +                if "rollout_log_probs" in data.batch:
    +                    data.batch["rollout_log_probs"] = None
    +                if "response_mask" in data.batch:
    +                    data.batch["response_mask"] = None         
    + 

    micro_batch_size = data.meta_info["micro_batch_size"]
    ...

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

    +        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
    +        if compute_prompts_only:
    +            if "responses" in data.batch:
    +                responses_len = data.batch["responses"].size(1)
    +                data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
    +                data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
    +                if data.batch["position_ids"].dim() == 3:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
    +                else:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
    +                # remove responses from batch
    +                data.batch["responses"] = None
    +                if "rollout_log_probs" in data.batch:
    +                    data.batch["rollout_log_probs"] = None
    +                if "response_mask" in data.batch:
    +                    data.batch["response_mask"] = None
    + 
             select_keys = [
                 "responses",
                 "response_mask",
                 "input_ids",
                 "attention_mask",
                 "position_ids",
                 "old_log_probs",
                 "advantages",
             ]
             ...
                         ...
                         # Extract pre-computed rollout correction weights if present
                         # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                         rollout_is_weights = model_inputs.get("rollout_is_weights", None)

    +                    if response_mask is None:
    +                        prompt_mask = torch.ones_like(log_prob, dtype=torch.bool)
    +                        response_mask = prompt_mask
    + 
                         # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                         # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                         policy_loss_fn = get_policy_loss_fn(loss_mode)

                         # Compute policy loss (any function is expected to return 2 values)
                         pg_loss, pg_metrics = policy_loss_fn(
                             old_log_prob=old_log_prob,
                             log_prob=log_prob,
                             advantages=advantages,
                             response_mask=response_mask,
                             loss_agg_mode=loss_agg_mode,
                             config=self.config,
                             rollout_is_weights=rollout_is_weights,
                         )
                         micro_batch_metrics.update(pg_metrics)
                         ...
    ```

    As shown in the preceding code, the environment variable `PROMPTS_ONLY` is used to control the `single-prompt` training mode. If this variable is set to `1`, the `single-prompt` mode is enabled.

Comparison Details

For training-inference consistency comparison, many module name mismatches exist. For example, with `qwen2.5-0.5b`, inference uses vLLM and training uses FSDP. 
<img src="https://raw.gitcode.com/user-images/assets/7898473/3d0f7c0d-0f33-4850-af39-ec9b83792dad/image.png" alt="Your image title" width="800"/>

Differences:

1. Different module name prefixes and class names.
2. `qkv_proj` difference: inference uses merged `qkv_proj`; training uses separate `q_proj`, `k_proj`, and `v_proj`.
3. Rotary position: inference applies rotary inside each `self_attn` layer; training applies it once before decode.
4. `silu` differences: inference uses fused `AscendSiluAndMul`; training uses small operators not captured at L0 level.

Table Comparison Tool Adaptation

The original comparison tool fails to match many nodes in such cross-framework scenarios. The tool now supports automatic matching for common Qwen-series models by adding mapping lists and merging qkv. The `--consistent_check` hyperparameter is also used to improve matching success.

Single-rank scenario:

```bash
msprobe compare -tp /train_dump/dump.json -gp /infer_dump/dump.json --consistent_check --backend fsdp -o ./output
```

Multi-rank scenario:

```bash
msprobe compare -tp /train_dump/step0 -gp /infer_dump/step0 --consistent_check --backend fsdp -o ./output
```

Visual Comparison Tool Adaptation

The hierarchical visualization tool provides a "point-to-point matching" feature. Users can select and match two (gray) nodes manually in the browser interface for matching. This currently only supports `statistics` data mode.
![image.png](https://raw.gitcode.com/user-images/assets/7898473/6d7ed4aa-17d6-4d4a-a045-22f5c2e3e160/image.png 'image.png')  

##### 2.3.3.6 Long-term Training Debugging

Refer to [Loss Difference in Long-Term Stable Training](#2313-loss-difference-in-long-term-stable-training). Also check for alignment of:

- Learning rate
- Partition strategies
- Data type (`dtype`) precision

### 2.4 Solutions to API Issues

#### 2.4.1 Single-Operator Verification

For suspicious operators identified in the above debugging scenarios, perform single-operator verification on both NPU and the benchmark to confirm whether the API is problematic. Typically, use CPU as the absolute benchmark, compute Euclidean distances between NPU and CPU and between the benchmark and CPU, and compare whether the NPU-CPU gap is larger than the benchmark-CPU gap.

Procedure:

1. Use the precision collection tool to collect specific tensor values for the suspicious operator on NPU.
2. Generate a single-operator verification script using the single-operator API auto-generation tool.
3. Run the verification script on NPU and on the benchmark using the same NPU input, and compute Euclidean distances to CPU.
4. If the NPU-CPU gap is larger than the benchmark-CPU gap, the operator has a precision issue. Try the three precision fixes below.

#### 2.4.2 Three Precision Fixes

After the problematic API is identified, it does not necessarily mean it is the decisive factor for the network's accuracy anomaly. Therefore, it is necessary to further determine the impact of the problematic API on the overall training loss. Try the following three fixes for benchmark-equivalent replacement (replacing with a functionally equivalent, precision-problem-free implementation) and observe whether loss becomes normal.

- If loss becomes normal, root cause confirmed.
- If loss improves but still not meets standards, the API affects loss but is not the only factor.
- If loss unchanged, the API does not affect accuracy.

The three fixes:

- Increase precision 
 Try increasing API precision (e.g., FP16/BF16 -> FP32) to avoid half-precision issues. Note that some APIs (e.g., random noise generation) are not suitable for higher precision.
- Move common APIs to CPU 
 If the API exists on both NPU and CPU, move it to CPU to ensure correct computation. Example:

    ```python
    class ModuleOP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features=2, out_features=2) 
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x1 = self.linear(x) 
            r1 = self.relu(x1)
           return r1
    ```

    As shown in the `linear` layer, the input of the `forward` function is moved to CPU. According to the internal mechanism of PyTorch, the forward and backward operations of the API can be performed on CPU.

     ```python
        # Replace self.linear_1 with CPU to run class ModuleOP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features=2, out_features=2) 
            self.relu = nn.ReLU()
    
        def forward(self, x):
            # Move the input to CPU and make modifications accordingly.
            x = x.cpu()
           self.linear.cpu()
            x1 = self.linear(x)  # Run on CPU.
            x1 = x1.npu() # Move the output back to NPU.          r1 = self.relu(x1)
            return r1
     ```

- Replace fusion operators with small operators 
 If the problematic API is a fusion operator, control whether the model uses the fusion operators or reference implementation via branch variables for training. If precision is abnormal in the fusion operator scenario but normal in the standard logic scenario, it indicates an abnormality in the fusion operator's implementation. For example, in MindSpeed-LLM, `attention` can be controlled with:

    ```bash
    --use_fused_attn
    ```

If none of these fixes work, contact official Ascend operator support:

- For external users or field engineers, contact Ascend support to find the responsible person for the API.
- For internal users, map the API name to the corresponding operator (API names usually closely match operator names) and contact the operator owner.
 
## 3. Debugging Case Studies

---

### 3.1 Checklist Inconsistency Cases

#### 3.1.1 Configuration Inconsistency

Case: A speech recognition model showed significant WER differences after migrating from GPU to NPU.

Debugging method:

Compare NPU and benchmark training configurations (startup scripts or logs). 
Found that NPU used FSDP while GPU used DDP. Training Loss differences were small, but downstream metrics differed significantly. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/5b506a50-f3c3-406e-9c8e-a1e878709b64/image.png 'image.png')

Solution: Align GPU configurations.

Result: WER dropped and aligned with GPU.

#### 3.1.2  Data Inconsistency

Case: An LLM migrated from LLama-Factory NPU (benchmark) to ModelLink NPU showed loss misalignment. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/7efdae1a-4765-4914-9a7e-ffb41378aa1d/image.png 'image.png')

Debugging method:

Print and compare input tokens at the `forward_step` function (For ModelLink, print at the `forward_step` function in `modellink/pretrain_gpt.py`). 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/c29037c7-896a-4bc8-baba-4a35a61f3bfb/image.png 'image.png')  
As shown in the preceding figure, token tails are inconsistent.

Solution: Fix the data preprocessing code to ensure identical inputs.

Result: Aligned loss. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/655f0a54-b851-49f5-a750-cd6cbf32a691/image.png 'image.png')

#### 3.1.3 Model Structure Inconsistency

Case: An MOE model migrated from GPU to NPU showed loss misalignment. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/9cd35663-a6f7-4056-88a3-29051af73e60/image.png 'image.png')

Debugging method:

Compare model structure implementations. 
Found that residual ordering differed: NPU applied `residual` after `input_layernorm`, while GPU applied it before. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/880aab30-4973-43fb-a0cb-8e2914a63299/image.png 'image.png')

Solution: Place `input_layernorm` after `residual` on NPU.

Result: Aligned loss. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/08c4b411-aabd-486b-abe9-49a1ded42b87/image.png 'image.png')

### 3.2 Scenario-Specific Debugging Cases

#### 3.2.1 Stable Reproduction Scenarios

##### 3.2.1.1 Overflow or NaN

Case: A vision model migrated from GPU to ModelLink NPU showed gradient overflow from the first step. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/3127cf1c-086e-48d8-8092-7762c8f5e3f9/image.png 'image.png')  
As shown in the training screenshot, the gradient increases layer by layer until overflow occurs during gradient backpropagation in step 0.

Debugging method:

1. Collect data with `mix` level at step 0 (where overflow occurs). `config.json` example:

    ```json
    {
        "task": "statistics",
        "dump_path": "/home/data_dump",
        "rank": [],
        "step": [0],
        "level": "mix",
        "enable_dataloader": false,
        "statistics": {
            "scope": [], 
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics"
        }
    }
    ```

    According to the training screenshot, the gradient increases layer by layer after each `self_attn` backward pass. By examining the `self_attn` code, it is found that the FA operator is used. This operator has historically caused many precision issues due to incorrect usage. Therefore, it is important to first check the corresponding backward data in the dump. It is found that after each backward pass through the FA layer, the `norm` value increases significantly. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/6693c940-3bc0-452c-b25d-6c203fcf1aa8/image.png 'image.png')

2. Avoid the FA fusion operator in the training configuration of ModelLink by deleting the following hyperparameter:

    ```bash
    --use_fused_attn
    ```

    The overflow disappears. It is confirmed that the issue is introduced by the FA branch, but the performance deteriorates significantly. Further investigation is needed to determine the cause of the FA precision issue.
3. Analyze the specific usage of the FA operator in the code by referring to its official documentation.
    This issue occurs in a variable-length scenario. The original input has a batch size of 2, with `seq_len=3577` for input sequence 1 and `seq_len=1502` for input sequence 2. Both sequences are padded to a length of 3577. The original input shape is [2, 3577, 32, 128]. 
    Before FA computation, `batch_size` and `seq_len` are flattened, giving a shape of [7154, 32, 128]. The padding is then removed in the next step, reducing the input lengths for Q and KV to [5079, 32, 128]. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/a66e4909-82f0-437e-8fc4-f5b8d046d488/image.png 'image.png')  
    According to the official documentation, the attention mask should follow the rule [`maxSq`, `maxSkv`], i.e., [3577, 3577]. However, the customer's code actually uses [`query.shape[0]`, `key.shape[0]`], which evaluates to [5079, 5079]. This incorrect usage causes the operator to read row-wise during execution, resulting in numerical misalignment of 0s and 1s, ultimately leading to gradient overflow.

**Solution**: Correct `attention_mask` passed to FA.

**Result**: Overflow disappeared; Loss converged normally.

##### 3.2.1.2 First Step Loss Difference

Case: A speech model showed loss misalignment from the first step. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/64e451db-52a8-41b8-820c-b55c378769de/image.png 'image.png')

Debugging method:

1. Collect step 0 data at `mix` level with the precision collection tool. For details about `config.json`, see section [3.2.1.1](#3211-overflow-or-nan).
 The following provides an example of adding the precision collection tool to the code: 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/0434a5a0-46fe-45f0-98e9-a05884141ad8/image.png 'image.png')
2. Use the hierarchical visualization tool to analyze differences.
    The visualization command is as follows:

    ```bash
    msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
    ```

    A `.vis` file is generated in the output directory. Use TensorBoard to open the visualization page. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/99186c98-1b3f-47b8-bb6d-68113895e3f9/image.png 'image.png')  
    The GELU operator appears in red, signaling potential precision issues.
3. Use the precision comparison tool for comparison. 
    Run the following command to obtain the comparison result in CSV format:

    ```bash
    msprobe compare -tp /target_dump/step0 -gp /golden_dump/step0 -o ./output
    ```

    From the table, it is found that the input difference of the GELU operator is small, but the output difference is large. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/0c44ccee-3715-4dfc-b3c0-660e1d85db73/image.png 'image.png')
 
Solution: Move GELU computation to CPU (fixes precision but impacts performance). Contact the operator support personnel to obtain an official PTA GELU fix package.

**Result**: Precision meets the requirement without moving to CPU.

##### 3.2.1.3 Long-term Training Loss Difference

**Case**: A search model converted from FP32 to BF16 showed normal loss initially but diverged later. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/52b9c64d-a386-4a3a-8e5e-dc001fae004b/image.png 'image.png')

Debugging method:

The loss initially aligned but diverged later at a step with a large ID. Since the total amount of data to dump was large and the exact step was uncertain, the monitor tool was preferred for data collection.

1. Check the trends of grad norm and loss: 
    Consistent trend suggests gradient-induced loss spikes. Use the following `monitor_config.json` to collect gradient data.

    ```json
    {
      "targets": {},
      "wg_distribution": true, 
      "format": "csv",
      "ops": [
        "norm",
        "mean",
        "min",
        "max"
      ]
    }
    ```

    Code insertion: 
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/f42a70bb-0abb-4ad4-b59e-c547c831d444/image.png 'image.png')
    
2. After the collection, the `grad_unreduced-xx-xx.csv` and `grad_reduced-xx-xx.csv` files of each rank can be obtained. 
`xx` indicates the step ID. The following shows the gradient data of each layer before reduction from step 360. 
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/8b62626a-e35e-42c1-a2a2-178be0c0815c/image.png 'image.png')  
The horizontal axis represents the layer order during backpropagation, with the output layer on the left and the embedding layer on the right. It can be observed that the weight gradient norm values are larger near the embedding layer. In comparison, the gradient data for FP32 is relatively stable on the embedding layer as well. 
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/581924af-bb80-4eaa-88d3-fb962423c878/image.png 'image.png')  
Therefore, it is suspected that the gradient of the embedding layer is numerically less stable in BF16 than FP32.

Solution: Apply gradient clipping to the embedding layer.

Result: Loss converged normally without spikes.

#### 3.2.2 Unstable Reproduction Scenarios

##### 3.2.2.1 Memory Corruption

Case: A multi-modal model fine-tuned with FSDP showed loss NaN at step 2 after migration to NPU. 
Execution result on NPU: 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/751daf11-9dba-4536-a5cc-1d1a5eb6c10e/image.png 'image.png')  
Execution result on GPU: 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/b60f3448-308a-4421-95aa-068f4cae60fa/image.png 'image.png')

Debugging method:

1. Reduce scale:
 The model is trained on 128 ranks, which is costly. Therefore, the first step is to reduce the scale. After reducing the number of layers, the issue can be stably reproduced on a single node with 2 ranks.
2. Use the precision collection tool to collect the `mix`-level data of step 1 (the step where NaN first occurs) and analyze the data.
    After the tool is added, the NaN issue disappears. 
    Remove the tool and enable stream synchronization for further verification.

    ```bash
    export ASCEND_LAUNCH_BLOCKING=1
    ```

    After stream synchronization is enabled, the issue also disappears. 
    Based on the preceding two symptoms, it is suspected that memory corruption occurs during FSDP training.
3. Narrow down the check scope:
    The model consists of four parts: VAE, DIT, denoiser, and conditioner. 
    Narrow training down to `dit.transformer.layers`. If loss still contains NaN, the issue is caused by the `transformer.layers`.
4. Print gradients manually by mounting a hook:
    It is found that loss NaN is not occurred at step 1. Instead, NaN first appears at the backward gradient of `post_attention_layernorm` at step 0. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/9e98aad8-0158-4171-894e-6e3bb0778138/image.png 'image.png')  
    Compared with the gradient data without NaNs when stream synchronization is enabled, all parameters except the `weight` and `bias` of the `input_layernorm` and `post_attention_layernorm` layers match. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/2b958529-a687-42da-84c8-6f26a49979d4/image.png 'image.png')  
    The corresponding dump APIs are `Functional.layer_norm.10` and `Functional.layer_norm.11`.
5. Analyze the code:
    `post_attention_layernorm` is continuously called twice for both the image and text. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/e273c809-1669-4ce6-9ae1-2081b474b387/image.png 'image.png')  
    When the number of calls is changed to 1, the NaN issue disappears. This indicates that the issue occurs when the operator is repeatedly called. 
    The method of analyzing memory corruption is to check whether the abnormal data is regular and continuous. Therefore, you need to collect the corresponding data first.
6. Enable asynchronous dump:
    The reason adding the precision collection tool can prevent NaN is that calculating tensor statistics (like `min` and `max`) and flushing data to drives interfere with how operators work in streams, preventing NaN from reappearing. 
    By enabling asynchronous dump, the tool does not trigger synchronization during training. Instead, data is flushed to drives after the current step is complete, reducing the impact on the operator execution sequence and stream synchronization. 
    How to do: Add `async_dump: True` to the `config.json` file. 
    Collect the `Functional.layer_norm.10` and `Functional.layer_norm.11` data, as well as the `torch.split.192` reverse data in between. The NaN can be reproduced when a single operator is dumped.
7. Analyze the asynchronous dump data:
    According to the `dump.json file` without loss NaN, the input of `torch.split.192.backward` should be the output of `Functional.layer_norm.11`. However, when stream synchronization is disabled, the input of `torch.split.192.backward` of asynchronous dump is inconsistent with the output of `Functional.layer_norm.11`. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/f4c2607e-d9ff-42fd-9801-2f7f5d89bfe7/image.png 'image.png')  
    It was found that the memory was corrupted exactly at a size of 2048 (values in the 0-2047 range were largely unequal, while values in the 2048-3071 range were equal), which fits the pattern of memory corruption. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/d70832b1-c199-4ca8-9233-48ab0da20540/image.png 'image.png')
8. Print the operator memory address:
    Try to modify the torch_npu source code to print the `ptr` addresses and shapes corresponding to the input and output tensors of the operator. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/62c1f075-9756-4022-a450-c275882e551a/image.png 'image.png')  
    According to the logs, within two consecutive LayerNorms, memory corruption occurred where the output of a cast operator overlapped with the input of a concat operator (their memory addresses were identical). 
    Confirmation of memory corruption: 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/1c25de8b-129f-4762-951d-0248f8402591/image.png 'image.png')

Root cause: backend without `record` + multi-stream FSDP + consecutive LayerNorms caused memory corruption.

Solution: Add `record` on the FSDP `unshard` stream in torch_npu 2.3 to ensure tensor memory is not reused before the current operator completes.

Result: Loss NaN disappeared; convergence normal.

##### 3.2.2.2 Operator Determinism

Case: A vision model had determinism issues; repeated training runs gave inconsistent Loss.

Debugging method:

1. Enable determinism (computation and communication), set random seeds, disable dropout, and fix data reading order. 
    These operations (except for dataset reading) can be done by using `seed_all` of the msProbe tool.

    ```bash
    from msprobe.pytorch import seed_all
    seed_all(mode=True)
    ```

    Some ranks still showed inconsistent backward results with random occurrence positions.
2. Collect step 0 data at the `mix` level using the precision collection tool. Set `summary_mode` to `md5` to highlight minor differences. The `config.json` configuration is as follows:

    ```json
    {
        "task": "statistics",
        "dump_path": "/home/data_dump",
        "rank": [],
        "step": [0],
        "level": "mix",
        "enable_dataloader": false,
        "statistics": {
            "scope": [], 
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "md5"
        }
    }
    ```

    Compare the data collected twice. The input of `masked_fill.23` is abnormal first. 
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/23f89bbd-8f60-44ca-a777-575602508164/image.png 'image.png')  
    Trace input source to MMCV's MSDA operator based on the `stack.json` call stack and code in the dump result. 
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/46e01b68-e868-4574-9f06-91e29abba68c/image.png 'image.png')  
    It is confirmed with the operator support personnel that the MSDA operator does not support deterministic computing. It is recommended that the operator be replaced with small operators. 
    After replacement, `grid_sample` output still differed when the input of `masked_fill.23` was the same. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/7f726fa8-2cce-4a44-8789-20a17071775c/image.png 'image.png')  
 Confirm with the `grid_sample` operator technical support that the operator does not support deterministic computing.

Solution: Replace MSDA with small operators; move `grid_sample` to CPU.

Result: Randomness fixed; repeated runs gave identical results.

##### 3.2.2.3 Hardware Stress Testing

Case: In a cluster with nearly 5,000 ranks, the loss remains unmatched, and numerous grad norm spikes appear. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/8562b38b-a85e-4b44-99de-89e70595c56b/image.png 'image.png')  
Due to the large size of the cluster, hardware stress testing is prioritized to identify faulty nodes. 
Divide the 4,800 ranks into 100 groups of *(3 × 16 ranks)* tasks to run the same training task. Enable fixed randomness and deterministic computing to check whether any group is abnormal based on the final loss curve. If an abnormal group is found, perform the `dmi` stress testing on the group. 
Run the `ascend-dmi -dg -i aicore -s -sc 60 -q` command to perform a stress testing and check the result. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/c86efd04-6b4d-414c-99b3-f8c4bffdbc31/image.png 'image.png')  
The detection result shows that there are faulty nodes. Rule out these nodes, and the precision becomes normal. Specifically, there are not loss spikes in the later phase, and the grad norm spike frequency is significantly reduced.
![image.png](https://raw.gitcode.com/user-images/assets/7898473/91e3f75b-3c9b-4b25-86e4-789a76790ae0/image.png 'image.png')
 
## 4. msProbe Usage

---
This section describes how to install and use msProbe package. For details, see the [latest msProbe documentation](https://gitcode.com/Ascend/msprobe).

### 4.1 Installation

See [msProbe Installation Guide](../msprobe_install_guide.md).

### 4.2 Overview

The following table lists the main functional modules contained in the msProbe package and provides a brief introduction to each module.

|Module| Description|
| --- | --- |
|Precision collection tool<br>(dump)| Collects the forward and backward inputs and outputs  at API, Module, or mixed levels, along with call stacks.|
|Training status monitoring tool<br>(monitor)| Collects activations, weights, gradients, optimizer states, and communication operator intermediate values with low overhead.|
|Hierarchical visualization tool| Parses dump data to reconstruct the model graph and compare precision across hierarchy levels.|
|Precision comparison tool| Compares dump data from NPU and benchmark across various metrics.|
|Trend visualization tool|  Provides global, hierarchical, and step-based trend visualization analysis for data collected by dump/monitor.|
|Precision pre-check tool| Samples operator APIs, input shapes, data types, value distributions, or real input data to generate input tensors or executes real input data on GPU and NPU for comparison.|

The msProbe package contains two parts: data collection and data comparison. The overall usage logic is as follows:

- Data collection
  - Precision collection tool, applicable to scenarios where the problem step is known:
    - Collect statistics first.
        - For small-scale models, directly collect mix-level (API + module) statistics.
        - For large-scale models, collect module-level statistics first,then API-level within suspicious modules.
    - After identifying suspicious operators via statistics, collect specific tensor values.
  - Training status monitoring tool, suitable for large-scale scenarios where the problem step is uncertain.
- Data analysis:
  - Hierarchical visualization tool: parse the saved files and reconstructs model graph for precision comparison or overflow analysis of each model layer.
  - Precision comparison tool: compares dump data and computes error metrics.
  - Trend visualization tool: performs global, hierarchical, and step-based trend analysis on data saved by dump or monitor.
  - Precision pre-check tool: constructs consistent input data based on shapes and value ranges of operator APIs in dump files, and performs three-party comparison between NPU, benchmark, and CPU.

### 4.3 Precision Collection Tool

For issues with a known first occurrence step, this tool can be used to collect training data. msProbe can add a hook to the dump API in the training script and collect statistics or specific tensor values of the forward and backward input and output data at the API or module level when training starts.

Instructions

1. Configure the `config.json` file first.
    There are two common configurations for collecting statistics:
    - Collecting statistics for specified steps

      ```json
      {
          "task": "statistics",
          "dump_path": "/home/data_dump",
          "rank": [],
          "step": [0,1],
          "level": "mix",
          "statistics": {
              "scope": [], 
              "list": [],
              "data_mode": ["all"],
              "summary_mode": "statistics"
          }
      }
      ```

    - Collecting  tensors for specified steps

      ```bash
      {
          "task": "tensor",
          "dump_path": "/home/data_dump",
          "rank": [],
          "step": [0,1],
          "level": "mix",
          "tensor": {
              "scope": [],
              "list":[],
             "data_mode": ["all"]
          }
      }
      ```

    Common modifications:
    - Collection level: Change the collection level to `mix` (API + module), `L0` (module), or `L1` (API).
    - Statistics + MD5 mode: Change `summary_mode` to `md5`.
    - Specifying the step or rank to be collected: Modify `step` (or `rank`). `[]` indicates that all steps or ranks are collected. A value within the brackets indicates that the specific step or rank is collected. Multiple steps or ranks are separated by commas (,).
    - Filtering the target APIs:
        - Modify the `list` attribute. When this attribute is added, all APIs or modules whose names contain the specified string will be collected. Multiple collection objects are separated by commas (,). If the string represents a module type, the internal APIs of the module are also collected.
        - Modify the `scope` attribute and add the start and end APIs or modules. APIs or modules between the start and end range will be collected.
2. Insert in code:

    ```python
    from msprobe.pytorch import PrecisionDebugger
    ...
    debugger = PrecisionDebugger(config_path="./config.json")
    # Training step loop
    for step in ...:
        debugger.start(model=model)
        output = model(data) # Module forward, which may also be named train_step(...)
        # Operations such as loss calculation...
        loss.backward()
        debugger.stop()
        debugger.step()
    ```

    Notes:
    - Use `seed_all()` to fix randomness before collection starts. Also use `mode=True` to enable determinism. The earlier the code is placed, the better.
    - Initialize `PrecisionDebugger` before training starts. Do not repeatedly define it in the loop.
    - Place `debugger.start()` before forward (usually placed before `model.forward()` or `train_step()`). To collect L0 or mix-level data, pass `model`.
    - Place `debugger.stop()` after `loss.backward()` (usually placed after `loss.backward()` or `train_step()`).
    - Place `debugger.step()` at the end of an iteration and call it after the `stop` function.

3. Output format 
    The result is saved in the path specified by `dump_path` in the `config.json` file. The overall format is as follows: 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/239c76c4-241e-442f-ab09-0f00c61285a5/image.png 'image.png')  
    Notes:
    - `dump.json`: statistics, typically including `Max`, `Min`, `Mean`, and `Norm`.
    - `stack.json`: call stacks for dumped APIs.
    - `construct.json`: model structure file.
    - `dump_tensor_data`: specific tensor values.
    - `dump_tensor_data` contains data only when `task` is `tensor`.
    - `construct.json` contains data only when `level` is `L0` or `mix`.

4. `dump.json` details 
    Example of `dump.json`: 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/201776cc-0ca5-4f30-88e3-464e35df6ddf/image.png 'image.png')  
    As shown in the figure, the statistics of the linear layer are collected, including:
    - `input_args`: contains three values, corresponding to input, weight, and bias from top to bottom. The bias is empty, so its value is `null`.
    - `output`: contains one value, corresponding to the computation result of the linear layer.

How to Use

The following figure shows a typical scenario where the loss of the first few steps differs greatly from the benchmark. The loss of step 0 is completely the same, while the loss difference of step 1 increases by several times. Therefore, we can infer that the problem probably occurs in the backward pass of step 0 or the forward pass of step 1. 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/a6f1d7c5-8861-4178-8b72-c3f545158ed4/image.png 'image.png')  

1. Statistics collection:

    - If the model scale is small (e.g., single-server 16-rank training), directly use the precision collection tool to capture `mix`-level statistics for steps 0 and 1. Then, use the comparison tool and hierarchical visualization tool to analyze and locate suspicious operators.
    - If the model scale is large, first consider reducing the scale. If reduction is not feasible, follow the steps below:
        (1) Use the precision collection tool to collect `L0`-level statistics for steps 0 and 1 to locate suspicious modules.
        (2) For suspicious modules, use the `list` parameter in the `config.json` file to restrict collection to that module.
        (3) Collect `L1`-level statistics within the restricted module to locate suspicious operators.

    In addition, for some issues with fixed randomness, you can set `summary_mode` to `md5` during statistics collection to more accurately display minor differences.

2. Tensor collection and analysis
    Once the scope of suspicion has been narrowed down or the final suspicious operator has been identified, directly specify the `list` parameter to collect tensor data for the corresponding operator. Use this data for further single-operator analysis.

### 4.4 Hierarchical Visualization Tool

The collected dump data on NPU and benchmark can be visualized and compared using the hierarchical visualization tool for comparison. This tool reconstructs the graph structure, enables layer-by-layer precision comparison, and helps users understand model architecture and analyze precision issues.

Instructions

1. Graph construction
    - Graph build (`rank` for single-step single-rank, `step` for single-step multi-rank, and `dump_path` for multi-step)

      ```bash
      msprobe graph_visualize -tp ./target_path -o ./output_path
      ```

    - Graph comparison

      ```bash
      msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
      ```

      Once graph construction completes, a `.vis.db` file is generated with an auto-generated timestamp-based name: `build_{timestamp}.vis.db`.
  
2. Visualization
     
    - If direct server connection is available:

      ```bash
      tensorboard --logdir out_path --bind_all
      ```

    - If direct server connection is unavailable (using VSCode remote connection):

      ```bash
      tensorboard --logdir out_path
      ```

    After startup, the address and port number are printed. 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/6d5e914f-4841-4486-83a2-40a3f20e6bff/image.png 'image.png')

3. Browser analysis
Enter the address and port number in a browser to open the visualization interface.

How to Use

Note: The hierarchical visualization tool can only be used when the precision collection tool is configured with `L0` or `mix` level, meaning a `construct.json` model structure file must exist in the dump results. 
When comparing graphs, follow these methods:

- Model structure comparison: 
  In the left sidebar, check gray unmatched nodes. A list of all unmatched nodes will appear. Click a node to view its specific information in the network. 
  ![image.png](../figures/visualization/vis_unmatch_info.png 'image.png')  
  Clicking a missing node expands stack trace and input/output information. Locate the corresponding code using the stack trace:
  - If some model steps are missing after migration, add them.
  - If the issue is only due to module naming differences, use the point-to-point match button in the left sidebar to manually match nodes.
  ![image.png](https://raw.gitcode.com/user-images/assets/7898473/49b8538e-3047-424d-a7d8-395cc75dc66d/image.png 'image.png')  
- Node precision comparison 
 In addition to unmatched nodes, the left sidebar allows filtering by precision risk level to highlight high-risk nodes. After visualization, darker colors indicate larger precision differences and higher suspicion.
 Besides prioritizing nodes by color depth, also check the first node where a precision difference appears. 
 ![image.png](../figures/visualization/vis_precision_info.png 'image.png')
- For communication operator issues, right-click a node and select data sending or receiving to view data from other ranks. 
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/958e5083-6920-4d89-83cd-bc09c73c3c39/image.png 'image.png')  
- Overflow analysis 
 Overflow levels can be filtered in the left sidebar. Overflow detection nodes of a specific level are sorted sequentially. Start debugging from the darkest-colored nodes. Click a filter item to jump to the corresponding node. 
 ![image.png](../figures/visualization/vis_overflow_check.png 'image.png')  
    Overflow levels:
    - `medium`: Abnormal input, normal output. Low priority; check last.
    - `high`: Abnormal input and output, or sudden spike in metrics where scale exceeds threshold. Manually confirm whether an issue exists.
    - `critical`: Normal input, abnormal output. High priority; check first.

For more details, see the following figure.
 ![image.png](../figures/visualization/vis_show_info.png 'image.png')  

### 4.5 Precision Comparison Tool

For statistics or tensor data collected and saved on NPU and the benchmark using the precision collection tool, the precision comparison tool can be used for precision comparison across various evaluation metrics in addition to the visualization tool.
Instructions

1. Comparison command:

    ```bash
    msprobe compare -tp /target_dump/step0 -gp /golden_dump/step0 -o ./output
    ```

2. Output format
    Two types of files are generated:
    - `advisor_{timestamp}.txt`: Provides expert suggestions for APIs that may have precision issues.
    - `compare_result_{timestamp}.xlsx`: Lists detailed information and comparison results for all APIs that undergo precision comparison. Refer to the usage guidelines below for result analysis.

How to Use

This tool can compare the collected dump data on NPU and the benchmark and generate a CSV comparison result file. 
For non-`md5` statistics comparison: 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/a893f5a1-879b-46ea-a3e2-64d44aa6ace8/image.png 'image.png')  
The comparison results use color coding, comparison outcomes, and specific precision values for each metric. When analyzing results, focus on the `MeanRelativeErr` column. Pay special attention to operators with small input differences but large output differences. 
For `md5` statistics comparison: 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/90f090b5-6eed-4f5e-b8ba-57764fe467bd/image.png 'image.png')  
Directly filter the `Result` column, which indicates whether the md5 comparison passed (`Pass`) or showed differences (`Different`). Focus on operators with a `Different` result. 
For tensor comparison: 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/daf75315-e30b-43d6-adda-6910b89f8967/image.png 'image.png')  
The comparison results use color coding, comparison outcomes, and specific precision values for each metric. When analyzing results, focus on the `One Thousandth Err Ratio` and `Five Thousandth Err Ratio` columns. Pay special attention to operators with small input differences but large output differences.

### 4.6 Training Status Monitoring Tool

For large-scale training or accuracy issues where the exact problem step is unknown, if collecting dump data generates excessive drive I/O, use the lightweight training status monitoring tool for initial debugging. 
This tool collects and records intermediate values of activations, weight gradients, optimizer states, and communication operators during training with low performance overhead, presenting real-time training status. It also supports dynamic start/stop, allowing monitoring to be restarted and configurations/objects to be modified during training.

Instructions

1. Configure the `config.json` file before using the tool. 
    Common configurations for training monitoring (monitoring weight gradients):

    ```json
    {
      "targets": {},
      "wg_distribution": true, 
      "format": "csv",
      "ops": ["norm", "mean", "max", "min"],
      "ndigits": 16
    }
    ```

    Common modifications:
    - Collection type: Also supports activation gradients (`xy_distribution`), optimizer states (`mv_distribution`), and communication information (`cc_distribution`).
    - Start step: Add `start_step` to specify the step at which collection starts.
    - Collection times: Add `collect_times` to specify the number of steps to collect. The default value is `100000000`, indicating continuous collection.
    - Print model structure: Set `print_struct` to `True` to print the structure at step 1 and exit automatically.
    - Filtering collection modules: Configure specific modules in `targets`. Refer to the `print_struct` output for the correct format.
    - Output format: Common options include `csv` and `tensorboard`.
    - Dynamic start/stop: In addition to specifying a start step, enable dynamic monitoring for large-scale tasks or issues with uncertain symptoms by setting the environment variable `export DYNAMIC_MONITOR=True`. Once in dynamic mode, set the `dynamic_on` switch in `config.json` to `True` to start monitoring with the latest configuration (supporting mid-run modifications).

2. Code insertion (Example with `Megatron core_r0.6.0`, in the `pretrain` function of `training.py`)

    ```python
    model, optimizer, _ = setup_model_and_optimizer(model_provider, type)
    # Tool enabling
    from msprobe.pytorch import TrainerMon, seed_all
    seed_all(mode=True) 
    # Monitor initialization
    monitor = TrainerMon(
            config_file_path="./monitor_config.json",
            process_group=mpu.get_pipeline_model_parallel_group(),
            params_have_main_grad=True  # megatron=True, deepspeed=False
       )
       # Mount objects to be monitored.
       monitor.set_monitor(
            model[0],
            grad_acc_steps=args.global_batch_size // args.data_parallel_size // args.micro_batch_size,
            optimizer=optimizer
        )
    ```

3. Output format 
    The output path is set via the `MONITOR_OUTPUT_DIR` environment variable (default: `monitor_output`). File naming conventions (`xx` represents the step number):
    - Activations: `actv_xx-xx.csv`
    - Activation gradients: `actv_grad_xx-xx.csv`
    - Weight gradients before reduction: `grad_unreduced_xx-xx.csv`
    - Weight gradients after reduction: `grad_reduced_xx-xx.csv`
    - Optimizer state: `exp_avg_xx-xx.csv`
    - Weights: `param_xx-xx.csv` 
    The following is an example when the preceding items are collected: 
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/9ae12b93-113f-4ff9-a6a8-6ec05bbaae84/image.png 'image.png')

How to Use

Collect data based on the loss and grad norm when you use the monitor tool.

- If the issue manifests as grad norm spikes preceding loss spikes, prioritize collecting gradient data by setting `wg_distribution` in the `config.json` file.
- If the issue is primarily reflected in the loss, prioritize collecting activation and weight data during training by setting `xy_distribution` and `param_distribution` in the `config.json` file.

Note that the mean value must be included in the statistical options for subsequent analysis. 
Gradient data can be analyzed from the following aspects:

1. Check the differences between unreduced and reduced gradients:
    - Identify which rank has issues (e.g., where NaN data appears).
    - Reproduce the reduce calculation and verify whether the mean result meets expectations (e.g., differences caused by communication order).
2. Check gradient distribution across model layers:
    - Focus on steps or layers where gradients show significant differences from the benchmark.
    - For NaN issues, focus on layers with abnormally large grad norm values.
    - Focus on layers whose gradient trend over steps matches the overall gradient trend (especially spike patterns).

For tasks with clusters exceeding one thousand ranks where the problem step is unclear, even the training status monitoring tool may generate large amounts of data. Then, use dynamic start/stop in such cases: do not monitor initially; start monitoring only when an anomaly is observed to capture the first occurrence. Combining this with an exception rollback mechanism is recommended to directly capture the first occurrence.

### 4.7 Trend Visualization Tool

The trend visualization tool is suitable for large-scale collected data (e.g., many ranks, steps, or layers). Traditional data analysis methods risk local traps, while this tool provides a global, hierarchical perspective to assist users in judgment.

### 4.8 Script Comparison Tool

Prepare two training environments and use the tool to collect and compare configurations affecting accuracy.

1. Collect data. Perform the following operations in two of the environments: 
      Insert the following code at the beginning of the training script:

      ```python
      from msprobe.core.config_check import ConfigChecker
      ConfigChecker.apply_patches(fmk)
      ```   

      Insert the following code after model initialization:

      ```python
      from msprobe.core.config_check import ConfigChecker
      ConfigChecker(model, shell_path, output_zip_path, fmk)
      ```

      Parameters:
      - `model`: initialized model. If omitted, weights and dataset are not collected.
      - `shell_path`: training script path; list type. Pass one or more training configuration/startup scripts. If omitted, hyperparameters are not collected.
      - `output_zip_path`: path of the output ZIP package. If omitted, the default value `./config_check_pack.zip` is used.
      - `fmk` (optional): training framework. The value can be `pytorch` or `mindspore`. If not configured, the default value `pytorch` is used.

      To compare third-party library versions (excluding git-installed libraries), set `pip data` to `true` on both NPU and benchmark. The resulting ZIP package contains various configuration items affecting accuracy.
2. Upload the two ZIP packages to the same environment and run the following command to compare them:

      ```bash
      msprobe config_check -c bench_zip_path cmp_zip_path [-o output_path]
      ```

      In the command, `bench_zip_path` is the data collected from the benchmark side, `cmp_zip_path` is the data collected from the comparison side, and the `-o` parameter specifies the optional output path for comparison results. The default output path is `./config_check_result`. 
      The result contains two directories and one file:
      - `bench`: data packaged in `bench_zip_path`
      - `cmp`: data packaged in `cmp_zip_path`
      - `result.xlsx`: comparison result. There are multiple sheets in the file. The `summary` sheet provides an overview of the results, and other sheets provide details about each check item.

      The accuracy comparison includes: environment variables, third-party library versions, training hyperparameters, weights, and datasets.

### 4.9 Non-tool Method

In certain cases (e.g., tool errors, tool causing non-reproducibility, or slow collection), manually mounting hooks can be attempted. 
This method uses PyTorch's native hooks to obtain block-level data from the entire network. Compared to the precision tools, it is more lightweight and allows flexible addition of custom logic. However, it cannot capture communication data; hooks must be placed at specific locations to output such data. 
Sample code for manual hook mounting:

```python
def print_func(inputs, prefix):
      if isinstance(inputs, tuple) or isinstance(inputs, list):
          for i in inputs:
              print_func(i, prefix)
      elif isinstance(inputs, torch.Tensor):
          print(prefix, inputs.max())
      else:
          print(prefix, inputs)
def hook_func(name, module):
      def hook_function(module, inputs, outputs):
          print(module)
          print_func(inputs, name + ' inputs')
          print_func(outputs, name + ' outputs')

      return hook_function
for name, module in model.named_modules():
      if module is not None:
         module.register_backward_hook(hook_func('[forward]:' + name, module))
```

## 5. Appendix

---

### 5.1 Root Cause Introduction

#### 5.1.1 Operators

Most operator precision anomalies are caused by bugs in the operator design itself, typically occurring in forward/backward computation.

- Numerical computation errors: Bugs in computation logic lead to incorrect numerical results.
- Memory corruption: Precision errors caused by memory issues during operator computation. Common in core-based and UB-based computation with non-integral block sizes, complex computation, or data type changes. Framework memory allocation (e.g., misaligned addresses) or out-of-bounds reads are also common.
- Asynchronous computation within operators: Causes reading of dirty data. Common in fusion operators.
- Implementation differences from GPU: Some operators have different adapter layer implementations across PyTorch versions.
- Cache read logic anomalies: Some operators use caching for performance acceleration, but cache reads may have bugs in specific scenarios.
- Operator determinism: Deterministic computation is an NPU mechanism to ensure reproducible operator results ensuring all operator calculations are consistent and reproducible during debugging.

#### 5.1.2 Communication

Communication operations generally do not cause computation-related precision issues (though all reduce/reduce scatter involve addition). The most common issues are memory corruption during communication, data format inconsistencies, communication link problems, or lack of send/receive protection mechanisms. Typically this manifest as data inconsistency before and after communication.

- Memory corruption: Common when computation and communication streams execute in parallel and access the same memory.
- Data format inconsistency: In aggregate operators like allgather, different data formats during aggregation cause precision issues.
- Communication links: Precision anomalies caused by issues such as PCIe switch failures. This type of issues is rare, hardware-related, and difficult to locate.
- Lack of send/receive protection: Common; occurs when subsequent operations read or modify memory before communication completes.

#### 5.1.3 Service Code

In many practical debugging cases, the root cause is eventually identified as issues with environment variables, hyperparameters, data processing, weight conversion, model implementation, or evaluation methods after migration. These should be prioritized before deeper debugging. 
A key challenge: without a GPU baseline, there is currently no complete process or solution to self-certify correctness.

#### 5.1.4 Training Framework

Migration scenarios often involve switching training frameworks (e.g., from pure GPU Megatron to MindSpeed+Megatron or ModelLink+Megatron). If the framework itself has bugs, training accuracy issues may arise.

#### 5.1.5 Hardware

Hardware-induced precision issues are rare but the most difficult to locate. If an issue is strongly tied to a specific server and stably reproducible, it can be located relatively quickly. However, if it cannot be stably reproduced and is not server-specific, it is very difficult to locate.

1. Multi-bit flipping: Caused by unstable voltage in hardware, leading to numerical instability. No current DFX can detect this. The only detection method is repeated stress testing of AI Core.
2. Power supply failure: Power instability leads to numerical anomalies, encountered at multiple deployment sites. Such issues have clear error logs in iBMC and are relatively easy to identify.

### 5.2 Model Hyperparameters

![image.png](https://raw.gitcode.com/user-images/assets/7898473/e4b647f4-0f5d-4864-9ed7-c15645579147/image.png 'image.png')  
As shown in the figure, model hyperparameters typically include learning rate, batch size, parallel partition strategies, model parameters, fusion operator configurations, etc. Before comparing NPU and GPU accuracy, ensure all configurations are consistent on both sides.

1. Learning rate and warm-up: Different learning rate schedulers have different hyperparameters. For example, linear scheduling can start warm-up from an initial learning rate `lr-warmup-init`. Also, you can specify the proportion of warm-up steps. Parameter names vary across frameworks. For example, ModelLink supports the following learning rate parameters:
    - lr
    - min-lr
    - lr-warmup-init 1e-8
    - lr-warmup-fraction 0.01
2. Batch size: The batch size affects training speed and sometimes model accuracy.
    - `micro-batch size`: batch size processed on each rank.
    - `global-batch-size`: batch size required for a complete gradient update.
3. Partition strategies: DP, TP, PP, EP, and CP
    - Data parallelism (DP): Maintains full model and parameter on each rank/parallel group, each processing different data. It is suitable for large datasets.
    - Tensor parallelism (TP): Also known as intra-layer parallelism, it splits weights across devices to reduce memory consumption per device, enabling large model training. No extra device waiting time beyond communication overhead.
    - Pipeline parallelism (PP): Suitable for large-scale model training, it places different model layers on different computing devices and reduces memory consumption of a single device. PP, also known as inter-layer parallelism, involves device waiting due to dependency between layer inputs and outputs. By further splitting batches into micro-batches, and through careful placement of network layers across devices along with clever scheduling of forward and backward computations, device idle time (compute bubbles) can be minimized, thereby improving training efficiency.
    - Expert parallelism (EP): In a mixture of experts (MoE) model, different experts are placed on different computing devices, allowing each expert network to independently learn and process the input data. It enhances the scalability of the entire MoE model, computational efficiency, and generalization capabilities, making it a focus in the field of large-scale MOE models.
    - Context parallelism (CP): Splits data along the sequence dimension to support sequence-parallel attention layers and achieve load balancing among computing devices. In long-sequence data training tasks, CP can effectively reduce waiting time and improve throughput, making it an effective approach for handling large-scale datasets and complex models.
4. Model structure hyperparameters, including:
    - `num-layer`
    - `hidden-size`
    - `seq-length`
    - `ffn-hidden-size`
    - `num-attention-heads`
5. Fusion operator configurations:
    - `use-flash-attn`: FA fusion operator switch, which requires special attention.
    - `use-fused-swiglu`
    - `use-fused-rmsnorm`
    - `use-fused-rotary-pos-emb`

### 5.3 Non-saturation Mode

When Inf/NaN mode, also known as non-saturation mode, is enabled, an overflow result necessarily becomes Inf or NaN; otherwise, it may not. NVIDIA enables this by default. It is strongly recommended to enable it in NPU environments to align with the benchmark and expose overflow issues. To enable this mode, configure the following environment variable in the training shell script:

```bash
export INF_NAN_MODE_ENABLE=1
```

To check whether this mode is enabled, check for the environment variable in the shell script or print it in the Python training script:

```python
import os 
inf_nan_mode = os.environ.get("INF_NAN_MODE_ENABLE", False) 
print(f"******INF_NAN_MODE is {inf_nan_mode}******")
```

Note: Some products may not implement this mode. Refer to actual product documentation.
