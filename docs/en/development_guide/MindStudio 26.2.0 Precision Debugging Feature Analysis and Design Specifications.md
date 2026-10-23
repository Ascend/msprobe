# MindStudio 26.2.0 Precision Debugging Feature Analysis and Design Specifications

<table>
    <tr>
        <td>Affiliated SIG:</td>
        <td>mstt-sig</td>
    </tr>
    <tr>
        <td>Target Version:</td>
        <td>MindStudio 26.2.0</td>
    </tr>
    <tr>
        <td>Designer:</td>
        <td>wangchao</td>
    </tr>
    <tr>
        <td>Date:</td>
        <td>2026.08.03</td>
    </tr>
</table>
**Copyright © 2026 MindStudio Community**

Your reproduction, use, modification, and distribution of "this document" are governed by the Creative Commons Attribution-ShareAlike 4.0 International Public License (hereinafter referred to as "CC BY-SA 4.0").
For ease of understanding, you may visit <https://creativecommons.org/licenses/by-sa/4.0/> to review a summary of CC BY-SA 4.0 (which is not a substitute for the full license).
The full text of CC BY-SA 4.0 is available at the following URL: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.

**Revision History**

<table>
    <tr>
        <th>Date</th>
        <th>Revision</th>
        <th>Revision Description</th>
        <th>Author</th>
        <th>Review</th>
    </tr>
    <tr>
        <td>2026.08.03</td>
        <td>26.2.0</td>
        <td>Design Specification</td>
        <td>wangchao</td>
        <td>xxx</td>
    </tr>
</table>

# 1. Feature Overview

In recent years, large language models (LLMs), multimodal models (e.g., text-to-image/video), and mixture of experts (MoE) models have undergone rapid iteration, with model parameters surging from the billion-level to the trillion-level, while training data scale, context length, and the complexity of distributed parallelism strategies (TP/DP/PP) have increased simultaneously.
The underlying AI computing infrastructure has migrated from a GPU-centric ecosystem to heterogeneous Ascend NPU clusters, making cross-hardware, cross-framework (PyTorch/MindSpore), and cross-precision (FP32/FP16/BF16/INT8/FP8/FP4) deployment a norm.
Against this backdrop, precision stability has become the core lifeline for large model deployment: during training, issues such as loss oscillation/non-convergence, gradient explosion/vanishing, NaN/Inf overflow/Loss Spike, and training-inference consistency deviations frequently arise; during inference, problems including output drift, prediction accuracy degradation, generation quality deterioration, and repetition/garbled text are common; when migrating models from GPU to Ascend NPU, precision misalignment and numerical deviations in custom operators occur frequently. Moreover, these issues are triggered with randomness, concealment, and long-tail characteristics, rendering traditional debugging approaches inadequate.

Currently, large model precision troubleshooting generally relies on manual log printing, third-party comparison tools, and scattered scripts provided by hardware vendors. Within the Ascend NPU ecosystem, the pain points are particularly pronounced:

- Severe black-box nature with a lack of observability
- Fragmented tooling without unified closed-loop capability
- The proliferation of diverse large model frameworks results in poor adaptability and high performance overhead
- Emerging scenarios such as reinforcement learning and recommendation lack systematic precision localization methods

To address the aforementioned pain points and fill the gap in native precision debugging tools within the Ascend ecosystem, the Huawei Ascend team developed MindStudio-Probe (abbreviated as msProbe) based on the MindStudio training toolchain. It was officially open-sourced at the end of 2025 and is hosted in the GitCode Ascend/msprobe repository.
msProbe is positioned as a full-scenario precision debugging toolkit for the Ascend ecosystem. Its core objective is to adopt "lossless lightweight, full-link closed-loop, native NPU adaptation, and large model first" as its design philosophy, thereby building full-link capabilities spanning pre-training risk pre-checking, real-time monitoring during training, fine-grained collection of anomalous data, automated numerical comparison, overflow/NaN detection, interactive visual analysis, and root cause localization. This fundamentally breaks through the black-box barrier of traditional debugging, transforming "black-box training and inference" into "white-box observability," helping developers rapidly resolve large model precision issues, lowering the debugging threshold, shortening the development cycle, and ensuring the precision stability and performance advantages of models on Ascend NPUs.

Building on the capabilities of 26.1.0, version 26.2.0 focuses on continuous enhancement in five directions: agent-based intelligent localization, reinforcement learning training-inference consistency, recommendation scenario precision localization, completion of basic general capabilities, and inference scenario (aclgraph, dynamic graph) precision localization, building a full-link capability closed loop from "tool collection + comparison" to "agent-based intelligent assisted localization."

## 1.1 Scope

- Data Collection Capabilities
    - Supports module-level and API-level collection for PyTorch full-network training/inference, providing L0/L1/mix hierarchical dump capabilities.
    - Supports lightweight monitoring of training states such as weights, gradients, and optimizer states during model training/inference.
    - Supports filtered collection by module, device, keyword, and network layer to reduce data volume and improve collection performance.
    - Supports stack trace information collection and optimization, preserving the context of the precision issue site.
    - Supports aclgraph single-point data collection, full-network statistical/real data collection, and dynamic start/stop capabilities.
    - Supports data dump and disk write for ATB operators.
- Data Comparison Capabilities
    - Supports offline numerical comparison between Ascend and benchmark data.
    - Supports multi-dimensional precision difference analysis, including statistics, tensor, and MD5.
    - Supports numerical alignment verification across different frameworks, devices, ranks, and iteration steps.
    - Supports user-defined extensible comparison algorithms.
    - Supports rotated activation/weight restoration comparison capabilities in scenarios such as QuaRot.
    - Outputs standardized comparison results and error reports for locating the root cause of precision non-compliance.
- Data Visualization Capabilities
    - Provides hierarchical model visualization, graph structure display, and node matching views.
    - Supports switching between multiple DB files and trend visualization for time-series comparison analysis.
    - Provides visualization panels for error distribution, anomalous points, and matching results.
    - Supports visualization of task matching and tensor computation processes.
- Automated Analysis and Agent Capabilities
    - Automatically detects typical precision issues such as numerical overflow/gradient anomalies.
    - Automatically localizes the initial layer of precision differences, anomalous operators, and anomalous devices.
    - Supports agent-based localization capabilities for scenarios such as NPU vs GPU module-level automatic comparison, Loss misalignment analysis, training-inference operator implementation difference scanning, Loss Spike analysis, and inference garbled text/repetition.
    - Provides automated precision issue diagnosis and root cause assisted localization.
    - Supports full-process automated integration, adapting to CI/CD and batch localization scenarios.

## 1.2 Feature Requirements List

Feature Requirements List

<table>
    <tr>
        <th>Requirement ID</th>
        <th>Requirement Name</th>
        <th>Feature Description</th>
    </tr>
    <tr>
        <td>1</td>
        <td>[msprobe/msagent] Agent precision localization capability enhancement</td>
        <td>Supports scenario expansion for NPU vs GPU module-level automatic comparison analysis, inference garbled text/repetition, training loss misalignment, training-inference operator implementation difference scanning, and spike training spikes.</td>
    </tr>
    <tr>
        <td>2</td>
        <td>[msprobe] Reinforcement learning scenario precision localization capability enhancement</td>
        <td>Supports training and inference data collection in the slime training-inference consistency scenario, a systematic solution for training-inference weight synchronization comparison, module-level custom mock stubbing, and extension of the hyperparameter comparison framework to slime.</td>
    </tr>
    <tr>
        <td>3</td>
        <td>[msprobe] Recommendation scenario precision localization capability enhancement</td>
        <td>The PTAinductor_triton precision tool supports torch 2.10, Torch Inductor fused operator mapping visualization, and a precision detection tool POC.</td>
    </tr>
    <tr>
        <td>4</td>
        <td>[msprobe] Completion of basic general capabilities of the precision tool</td>
        <td>Precision comparison supports custom comparison algorithms; analysis migration and msprobe support pytorch2.9/pytorch2.10; monitor supports new L2 metrics of router weight similarity and per-token expert entropy; dump supports dumping the module input before the inplace operation under L0; supplementary collection of torch_npu.atb.xx operators; and usability optimization of tool documentation.</td>
    </tr>
    <tr>
        <td>5</td>
        <td>[msprobe] Inference scenario precision localization capability enhancement</td>
        <td>The aclgraph scenario supports custom API collection, full-network real data collection, low-precision data format collection, and dynamic start/stop; dump supports tensor collection slicing; precision comparison supports the QuaRot scenario; and the dynamic graph multi-batch supports collection for a specified request.</td>
    </tr>
</table>

# 2. Requirements Scenario Analysis

## 2.1 Feature Requirement Sources and Value Overview

[Precision Toolchain] By continuously enhancing capabilities in multiple aspects—including agent-based intelligent assisted localization, reinforcement learning training-inference consistency localization, recommendation scenario precision localization, completion of basic general capabilities, and inference scenario (aclgraph, dynamic graph) precision localization—the toolchain further improves large model precision debugging efficiency and training-inference reliability assurance, evolving from "tool collection + comparison" to "intelligent assisted localization."

## 2.2 Feature Scenario Analysis

The following scenarios are primarily supported:

- Agent precision localization: based on data collected by msprobe, automatically analyzes typical issues such as NPU vs GPU module differences, training loss misalignment, training-inference operator implementation differences, training spikes, and inference garbled text/repetition through agent-based intelligent assisted capabilities
- Reinforcement learning precision localization: extends training-inference consistency data collection, weight synchronization comparison, module-level mock stubbing, and hyperparameter comparison for the slime framework
- Recommendation scenario precision localization: supports completion of Inductor (triton backend) torch 2.10, Torch Inductor fused operator mapping visualization, and precision detection capabilities
- Completion of basic general capabilities: custom comparison algorithms, adaptation to new PyTorch versions, new monitor metrics, inplace collection, ATB operator collection, and tool documentation optimization
- Inference scenario precision localization: aclgraph scenario custom API/full-network real data/low-precision/dynamic start-stop capabilities, dump slicing, QuaRot restoration comparison, and dynamic graph multi-batch collection by request

# 3. [msprobe/msagent] Agent Precision Localization Capability Enhancement

## 3.1 Design Approach

This feature needs to support the following sub-scenario capabilities:

1. Scenario expansion: support NPU vs GPU module-level automatic comparison analysis
2. Scenario expansion: inference garbled text/repetition
3. Scenario expansion: training loss misalignment
4. Support training-inference operator implementation difference scanning
5. Scenario expansion: spike training spikes

## 3.2 Constraints

N/A.

## 3.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- Scenario expansion: support NPU vs GPU module-level automatic comparison analysis
  - Requirement background: In large model inference scenarios, NPU and GPU inference inconsistency frequently occurs, so module-level data comparison analysis is performed with GPU as the benchmark against NPU. For the same model, the NPU and GPU model implementations are not necessarily fully equivalent, and it currently requires manual layer-by-layer confirmation of the module matching relationship between the two, resulting in low localization efficiency.
  - Requirement content: msagent, as an agent tool collection repository, adds an NPU vs GPU module-level automatic comparison analysis Skill capability for data collected by msprobe. It automatically analyzes the modules that can be automatically matched between NPU and GPU, and locates and analyzes the differing module nodes based on dump data

- Scenario expansion: inference garbled text/repetition
  - Requirement background: In large language model online inference services, two typical abnormal phenomena—repetition and garbled text—frequently occur. With the existing dump comparison approach, the absence of valid benchmark data significantly affects localization efficiency.
  - Requirement content: Targeting stably reproducible repetition and garbled text issues, based on the agent localization capability for bad cases, it can automatically capture abnormal output samples and reason through a multi-dimensional evidence chain (input context, sampling parameters, Token ID sequence, probability distribution, etc.), ultimately outputting root cause classification and remediation suggestions. (Deliverables are not limited; they may be a skill or a summary of multiple cases.)

- Scenario expansion: training loss misalignment
  - Requirement background: In large model training scenarios, Loss misalignment is a typical issue affecting training precision consistency. Since Loss is generated by the gradual accumulation of computation results across multiple network layers, current localization methods usually require manual layer-by-layer investigation of intermediate Tensor differences, resulting in low localization efficiency; in complex model scenarios, it is difficult to identify the root cause of the Loss difference in a timely manner.
  - Requirement content: msagent, as an agent tool collection repository, adds a Loss misalignment analysis Skill capability for data collected by msprobe. It automatically analyzes the source of the Loss difference and locates the key modules and operators causing the Loss inconsistency.

- Support training-inference operator implementation difference scanning
  - Requirement background: During reinforcement learning training-inference consistency analysis, the training side and the inference side usually use different operator implementations. For example, the training phase uses single-operator computation, while the inference phase uses fused operators to improve performance, resulting in differences in operator granularity and implementation that increase the difficulty of localizing training-inference differences. Therefore, a training-inference operator implementation difference scanning capability is needed to assist in quickly localizing operator-level inconsistencies.
  - Requirement content: In the form of a skill, it supports automatic scanning of operator information on the training side and the inference side, and comparative analysis of training-inference operator implementation differences. It provides a one-click scanning entry point to reduce manual troubleshooting costs.

- Scenario expansion: spike training spikes
  - Requirement background: During large model training, Loss Spike (training spikes) is a common issue affecting training stability, and may be caused by factors such as gradient anomalies, data anomalies, and operator computation errors. Current precision localization capabilities mainly cover scenarios such as NaN and determinism, and lack automated analysis means for the issue of a sudden Loss increase during training, resulting in high manual troubleshooting costs.
  - Requirement content: msagent, as an agent tool collection repository, adds a Loss Spike analysis Skill capability for data collected by msprobe, supporting automatic analysis of the causes of training spikes and localization of the key nodes causing abnormal Loss fluctuations.

## 3.4 DFX Attribute Design

### 3.4.1 Performance Design

*As a diagnostic feature, it is insensitive to performance impact and is therefore not addressed.*

### 3.4.2 Security Design

#### 3.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are added (UI inputs, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team has been notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (which validations: length, format, category, threshold, null check, and whether path-type parameters are normalized and standardized before use) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method, whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether externally provided services have authentication, authorization; all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a company-recommended secure encrypted protocol, whether returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the standard open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information is involved (generation, use, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether secure function libraries are used | No |

# 4. [msprobe] Reinforcement Learning Scenario Precision Localization Capability Enhancement

## 4.1 Design Approach

This feature needs to support the following sub-scenario capabilities:

1. [Training-inference consistency] Support training and inference data collection in the slime training-inference consistency scenario
2. Systematic solution for training-inference weight synchronization comparison
3. msprobe supports module-level custom mock stubbing
4. Extension of the hyperparameter comparison framework to slime

## 4.2 Constraints

N/A.

## 4.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- [Training-inference consistency] Support training and inference data collection in the slime training-inference consistency scenario
  - Requirement background: In reinforcement learning scenarios, training-inference consistency issues often exist, and there are differences in the internal implementation details of the models used for inference and training (such as operator implementation, structural order, and naming differences), making data collection and comparison impossible and manual matching time-consuming and labor-intensive.
  - Requirement content: Support training and inference data collection in the slime training-inference consistency scenario, as follows:
    1. This requirement only supports inference and training collection in the prefill phase. The prerequisite is to ensure that the shapes of the inference and training data are kept within a reasonable range. Refer to the following prerequisite solution:
      a. Change training to a single prompt to ensure that the token dimensions on both sides are consistent
      b. Set max_response_len to 1 for inference; at this point the training token dimension is K (prompt), and the inference token dimension is K+1 (prompt+1response)

- Systematic solution for training-inference weight synchronization comparison
  - Requirement background: In the reinforcement learning training-inference integrated scenario, the training side and the inference side need to keep the model weights consistent. However, due to differences in parameter updates, weight synchronization processes, and format conversions, training-inference weight inconsistency issues are prone to occur. Current localization efficiency for weight synchronization issues is low, and a systematic analysis solution is lacking.
  - Requirement content: Design a systematic solution for training-inference weight synchronization comparison, quickly identify the source of weight inconsistency, and assist in localizing RL precision issues.

- msprobe supports module-level custom mock stubbing
  - Requirement background: During reinforcement learning training-inference consistency analysis, slight precision differences in preceding modules propagate layer by layer, affecting the localization of subsequent module issues. Currently, the method usually used is to save the input of a specified module on the training side and load and override it on the inference side to isolate preceding errors, but this is cumbersome and inefficient. Therefore, tool-based support is needed.
  - Requirement content:
    - Supports module-level custom Mock stubbing.
    - Supports loading already-collected module input Tensors and overriding the actual module inputs.

- Extension of the hyperparameter comparison framework to slime
  - Requirement background: When reinforcement learning frameworks are trained on NPUs and benchmarks, precision issues often arise due to differences in training/inference configurations. Judging solely from the launch sh script may result in missed validations or failure to identify default hyperparameters. Therefore, it is necessary to obtain the hyperparameters actually in effect in both the NPU and benchmark environments for subsequent comparison. Previously only the verl framework was supported; the slime framework needs to be extended.
  - Requirement content: Based on the training logs of the NPU and benchmark, parse the configurations in the logs and save them as config files, and perform differential comparison between the two.

## 4.4 DFX Attribute Design

### 4.4.1 Performance Design

*As a diagnostic feature, it is insensitive to performance impact and is therefore not addressed.*

### 4.4.2 Security Design

#### 4.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are added (UI inputs, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team has been notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (which validations: length, format, category, threshold, null check, and whether path-type parameters are normalized and standardized before use) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method, whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether externally provided services have authentication, authorization; all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a company-recommended secure encrypted protocol, whether returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the standard open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information is involved (generation, use, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether secure function libraries are used | No |

# 5. [msprobe] Recommendation Scenario Precision Localization Capability Enhancement

## 5.1 Design Approach

This feature needs to support the following sub-scenario capabilities:

1. Ascend 950PR&950DT products: the PTAinductor_triton precision tool supports torch 2.10
2. Torch Inductor fused operator mapping visualization (POC)
3. Torch Inductor fused operator precision detection tool (POC)

## 5.2 Constraints

N/A.

## 5.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- Ascend 950PR&950DT products: the PTAinductor_triton precision tool supports torch 2.10
  - Requirement background: It is necessary to compare the precision of eager mode and the full network after automatic operator generation, and identify nodes that do not meet the precision requirements.
  - Requirement content: (framework-type requirement)
    - Description: Support automatic precision comparison for Inductor (triton backend), and support setting precision thresholds.
    - Extend the version support of the original requirement.

- Torch Inductor fused operator mapping visualization (POC)
  - Requirement background: In large model scenarios, `torch.compile` compiles PyTorch models into efficient fused kernels (Fused Kernel) through the Inductor backend. The Validator tool has already implemented capabilities such as compile-time metadata extraction, runtime I/O capture, and per-kernel precision verification, and can detect the precision deviation (PASS/FAIL) of each fused operator. However, in the current verification process, when a fused operator is detected as FAIL (precision out of tolerance), users face the following pain points:
    1. Invisible fusion relationships: Users only know that a certain fused operator (such as `triton_poi_fused_relu_0`) is out of precision tolerance, but do not know which original FX operators (such as `aten::conv2d`, `aten::relu`, `aten::addmm`, etc.) this fused operator contains, and cannot determine which original operator's fusion introduced the precision degradation.
    2. Non-intuitive Buffer mapping relationships: The `buffer_to_fx_node` mapping and `arg_to_buffer` mapping extracted at compile time are program-oriented data structures, and users cannot intuitively understand "which `placeholder` node in the FX graph the kernel's `in_ptr0` corresponds to" or "which operator's output in the model the kernel's output `buf7` corresponds to".
    3. Lack of association between input/output data and pre-fusion nodes: For the kernel input/output tensors captured at runtime, users only know their buffer names (such as `buf0`, `buf3`), but cannot correspond these data to specific nodes in the pre-fusion FX graph, and thus cannot determine which pre-fusion operator produced the anomalous data.
    Therefore, based on the existing Validator tool capabilities, it is necessary to provide a mapping visualization capability between fused operators and pre-fusion FX operators, intuitively showing "which original FX nodes a fused operator contains", "which node in the FX graph each input/output buffer of the kernel corresponds to", and "the association relationship between runtime data and pre-fusion nodes", helping users quickly locate the root cause of precision degradation.
  - Requirement content: Based on the Validator's existing compile-time metadata (KernelInfo, buffer_to_fx_node, arg_to_buffer) and runtime I/O capture data, develop a mapping visualization module that presents the mapping relationship between fused operators and pre-fusion FX operators, as well as the association relationship between Buffers and data, in the form of structured text and visualization charts:
    1. Fused operator → original FX node mapping (Fusion Graph): Based on the `buffer_to_fx_node` already extracted by `BufferFxMapper` in `buffer_mapping.py` and the kernel's `input_buffer_names`/`output_buffer_names` recorded in hooks, construct the mapping relationship from each fused operator to the set of original FX operators it contains. Output in text form the list of original FX nodes contained in each fused operator, including node type (placeholder/call_function/output), operation name (such as `aten::addmm`, `aten::relu`), and the node's input/output buffer names.
    2. Kernel I/O Buffer → FX node mapping visualization: Based on the `_arg_to_buffer` in KernelInfo (such as `{"in_ptr0": "buf3", "out_ptr0": "buf7"}`) and `buffer_to_fx_node` (such as `{"buf3": fx.Node(addmm), "buf7": fx.Node(relu)}`), construct a complete-link visualization of "kernel parameter name → buffer name → FX node → FX operator name". Output as a tree or table form, clearly showing the complete mapping link for each fused operator's I/O.

- Torch Inductor fused operator precision detection tool (POC)
  - Requirement background: In large model scenarios, `torch.compile` compiles PyTorch models into efficient fused kernels (FusedKernel) through the Inductor backend. Operations such as operator fusion, memory optimization, and numerical transformation during the compilation process may introduce precision deviations. Currently, when locating precision issues, users usually determine whether a precision issue exists by comparing end-to-end outputs—this approach can only perceive the deviation of the final output, cannot locate the specific source of the precision issue, and cannot distinguish which fused operator caused the precision degradation.
  - Requirement content: Develop a Torch Inductor fused operator precision detection tool that is completely decoupled from PTA and adopts a three-phase solution of **compile-time metadata extraction + runtime I/O capture + per-kernel replay comparison** to implement precision detection for Inductor fused operators:
    1. Compile-time metadata extraction: Through monkey-patching Inductor's internal interfaces (`GraphLowering`, `define_kernel`), extract the metadata of each fused operator at model compile time—buffer names, parameter mappings, and input/output relationships. Using a single graph invocation (Invocation) as the verification unit, each compiled graph invocation is tracked independently, generates eager reference values independently, and is compared independently.
    2. Runtime I/O capture: Hook the autotuner's `run()` method to automatically capture the actual input/output tensors of each kernel call during model inference, and associate them with the current Invocation record. Supports capturing kernel calls in both the forward and backward phases.
    3. Per-kernel precision verification (at verification time): Run the FX Interpreter with the same graph inputs to obtain the eager reference value of each operator, replace the inputs of the compiled kernel with the eager reference values and re-execute, and compare the output differences. Provide a dtype-aware tolerance strategy: automatically select the appropriate atol/rtol threshold based on the tensor data type (int/float16/bfloat16/float32/float64/complex, etc.).

## 5.4 DFX Attribute Design

### 5.4.1 Performance Design

*As a diagnostic feature, it is insensitive to performance impact and is therefore not addressed.*

### 5.4.2 Security Design

#### 5.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are added (UI inputs, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team has been notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (which validations: length, format, category, threshold, null check, and whether path-type parameters are normalized and standardized before use) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method, whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether externally provided services have authentication, authorization; all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a company-recommended secure encrypted protocol, whether returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the standard open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information is involved (generation, use, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether secure function libraries are used | No |

# 6. [msprobe] Completion of Basic General Capabilities of the Precision Tool

## 6.1 Design Approach

This feature needs to support the following sub-scenario capabilities:

1. Precision comparison supports custom comparison algorithms
2. Analysis migration supports to PyTorch 2.9/PyTorch 2.10
3. msProbe supports PyTorch 2.10
4. monitor supports new L2 metrics: router weight similarity and per-token expert entropy
5. dump supports dumping the module input before the inplace operation under L0
6. Supplementary collection of torch_npu.atb.xx operators
7. Usability optimization of tool documentation

## 6.2 Constraints

N/A.

## 6.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- Precision comparison supports custom comparison algorithms
  - Requirement background: Tensor comparison currently mainly relies on cosine similarity and the dual-thousand metric. Facing complex and diverse user scenarios, the comparison metrics provided by the current tool may not satisfy users' specific scenarios, so it is planned to make this capability user-defined and extensible.
  - Requirement content: The comparison tool supports user-defined extensible comparison algorithms. The tool provides the comparison algorithm file, and the tool side gives specification definitions for the input and output of the algorithm. Users implement custom comparison algorithms based on the tool's specification requirements.

- Analysis migration supports to pytorch2.9/pytorch2.10
  - Requirement background: pytorch2.9/pytorch2.10 have been released, and the analysis migration tool needs to support the corresponding framework versions.
  - Requirement content: Analysis migration involves the full capabilities of the PyTorch framework and needs to be adapted to versions 2.9/2.10.

- msprobe supports pytorch2.10
  - Requirement background: pytorch2.10 has been released, and msprobe needs to support the corresponding framework version.
  - Requirement content: msprobe involves the full capabilities of the PyTorch framework and needs to be adapted to version 2.10.

- monitor supports new L2 metrics: router weight similarity and per-token expert entropy
  - Requirement background:
    1. Router weight similarity (Router Weight Cosine Similarity): Quantifies the degree of expert degradation by computing the average cosine similarity between the weight column vectors of all experts. When sim(W) → 1, all expert weights tend to be consistent, and the Router exists in name only.
       Formula: `sim(W) = E(i≠j)[cos(wi, wj)]`
    2. Per-Token Expert Entropy: By recording the softmax routing distribution of a token over multiple experts, observe Router health from the perspective of the "determinism" of routing decisions. If the routing distribution of every token is overly concentrated (entropy extremely low), it indicates that the Router has entered "singleton" mode.
       Formula: Entropy(S) = E_i[-Σ(j=1,n) s_ij log s_ij]
  - Requirement content: monitor L2 interpretability metrics add monitoring items for router weight similarity and per-token expert entropy

- dump supports dumping the module input before the inplace operation under L0
  - Requirement background: The current tool's module-level data collection is based on Hooks to obtain module input and output data. When the model has Inplace operations (such as add_), the module input Tensor is modified in place, causing the collected input data to actually have been overwritten, resulting in mis-collection problems such as "input equals output", which affects the accuracy of precision comparison results. Therefore, it is necessary to support collecting the module input data before the Inplace operation is executed.
  - Requirement content:
    - Supports collecting module input data before the Inplace operation is executed under L0 mode.
    - Optimizes the module-level data collection process to avoid input data being overwritten by Inplace operations.

- Supplementary collection of torch_npu.atb.xx operators
  - Requirement background: The current tool performs data collection according to an operator collection YAML list. As the vLLM-Ascend framework continues to evolve, a large number of `torch_npu.atb.*` series operators have been added, but the relevant operators have not yet been included in the collection list, causing some operators to fail to complete data collection and affecting the tool's coverage of new framework versions and the completeness of precision analysis. Therefore, it is necessary to supplement collection support for `torch_npu.atb.*` operators.
  - Requirement content:
    - Supplement the torch_npu.atb.* series operators into the collection YAML list.
    - Supports input/output data collection for the newly added ATB operators.
    - Keeps the existing operator collection logic and configuration method unchanged.

- Usability optimization of tool documentation
  - Requirement background: The current tool homepage README displays all functions in a flat manner, resulting in high information density and a lack of prominence for core functions. First-time users need to search for commonly used functions among a large amount of content, resulting in a high learning cost and affecting the tool's usability and usage efficiency. Therefore, it is necessary to optimize the information architecture of the homepage, highlight core capabilities, and lower the user onboarding threshold.
  - Requirement content: Optimize the usability of the tool homepage README, simplify the homepage display content, and optimize the information hierarchy. Specifically including:
    - The homepage retains only core content such as tool introduction, core functions, and quick start.
    - Migrate low-frequency or advanced content to second-level or deeper directories, and jump through links.
    - Optimize the page layout to make the homepage content more concise and highlight the key information needed by users when using it for the first time.

## 6.4 DFX Attribute Design

### 6.4.1 Performance Design

*As a diagnostic feature, it is insensitive to performance impact and is therefore not addressed.*

### 6.4.2 Security Design

#### 6.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are added (UI inputs, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team has been notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (which validations: length, format, category, threshold, null check, and whether path-type parameters are normalized and standardized before use) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method, whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether externally provided services have authentication, authorization; all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a company-recommended secure encrypted protocol, whether returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the standard open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information is involved (generation, use, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether secure function libraries are used | No |

# 7. [msprobe] Inference Scenario Precision Localization Capability Enhancement

## 7.1 Design Approach

This feature needs to support the following sub-scenario capabilities:

1. The aclgraph scenario supports custom API collection
2. dump supports tensor collection slicing (eager, aclgraph)
3. Precision comparison supports the QuaRot scenario and supports the restoration comparison capability of rotated activation weights
4. Aclgraph scenario dump supports dynamic start/stop
5. The aclgraph scenario supports full-network real data collection
6. The aclgraph scenario supports low-precision data format collection
7. Dynamic graph multi-batch supports collection for a specified request

## 7.2 Constraints

N/A.

## 7.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- The aclgraph scenario supports custom API collection
  - Requirement background: In large model inference scenarios, when performing precision alignment in ACLGraph mode, the msprobe aclgraph_dump tool currently provides only two capabilities: full-network statistical collection (AclGraphDumper) and single-point real data collection (acl_save). The current full-network collection capability covers all nn.Module blocks and torch APIs, but inference scenarios involve some user-defined functions/APIs (such as kvcache processing), and the tool needs to further expand support for collecting such APIs.
  - Requirement content: The aclgraph_dump tool extends the custom API collection and disk write capability in the full-network collection mode (AclGraphDumper). The tool needs to provide a generalized registration interface for users to support custom function collection configuration. Ultimately, it supports both statistical and real data collection.

- dump supports tensor collection slicing (eager, aclgraph)
  - Requirement background: In large model inference scenarios, an inference task is usually performed by assembling multiple requests into a batch. Therefore, the data obtained through msprobe dump is a concatenation of multiple requests. When locating precision issues, localization is performed at the request granularity, so a slicing capability needs to be provided at dump time.
  - Requirement content: The dump tool provides a configuration capability, supporting slicing for a specified shape. The collected data is subjected to statistical computation or disk write according to the shape after slicing, improving the overall usability of the tool.

- Precision comparison supports the QuaRot scenario and supports the restoration comparison capability of rotated activation weights
  - Requirement background: Both the activation values and weights quantized by the NPU involve rot matrices, and the dumped data cannot be directly compared with the GPU. Taking dsv4 as an example.
  - Requirement content: msprobe supports providing a configuration file for comparison restoration. The configuration file specifies the tensors that need restoration processing, the restoration algorithm, and the matrices that the restoration depends on. Thus, during comparison with the GPU, automatic restoration comparison is implemented.

- Aclgraph scenario dump supports dynamic start/stop
  - Requirement background: In large model inference scenarios, when using ACLGraph mode for precision localization, full-network collection (AclGraphDumper) injects collection stubs by calling `start(model)` before model compilation, and triggers data collection and disk write through `step()` in each replay after model graph compilation. The current tool's start/stop depends on runtime API control, and once `start()` is called to enable collection, all subsequent. In multi-step inference or long-stability test scenarios, users want to temporarily enable dump collection at a specific timing (such as when precision anomalies are found), and then disable it after localizing the issue, avoiding the Device-to-Host data copy overhead and large accumulation of disk files caused by full collection. Therefore, based on the existing AclGraphDumper, it is necessary to provide a dynamic start/stop mechanism based on a configuration file, supporting real-time control of the enabling and disabling of collection by externally modifying the configuration file during operation after collection injection, improving the tool's flexibility and O&M friendliness.
  - Requirement content: AclGraphDumper extends the capability of dynamic start/stop based on a configuration file. By adding a new configuration field `dump_enable` and combining it with a scheduled monitoring mechanism, it implements dynamic control of dump collection behavior during inference without restarting the process:
    - Add dump_enable configuration: Add a new `dump_enable` field (bool type, default value false) in config.json. When set to `true`, full-network collection (AclGraphDumper) normally collects data and writes to disk when step is executed; when set to `false`, the collection behavior is skipped when step is executed, without producing Device-to-Host data copy overhead or disk files.

- The aclgraph scenario supports full-network real data collection
  - Requirement background: In large model inference scenarios, when performing precision alignment in ACLGraph mode, the msprobe aclgraph_dump tool currently provides only two capabilities: full-network statistical collection (AclGraphDumper) and single-point real data collection (acl_save). Full-network statistical collection, based on the `acl_stat` interface, outputs only statistics such as the min/max/mean/norm of Tensors, lacking the real Tensor data of the input/output of each layer of the full network. When precision issue localization requires viewing the specific data distribution, element-by-element comparison, or troubleshooting specific numerical anomalies, the statistical information is insufficient to support fine-grained analysis. Although single-point collection `acl_save` can save real Tensor data, it relies on users manually inserting code point by point in the model, which is cumbersome and prone to omission in large-scale models, and cannot cover all intermediate results of the full network. Therefore, it is necessary to extend the real data disk write capability in the full-network collection mode, automatically collecting the real data of the input/output Tensors of each layer through configuration, improving the comprehensiveness and usability of full-network precision localization.
  - Requirement content: The aclgraph_dump tool extends the real data disk write capability in the full-network collection mode (AclGraphDumper), supporting automatic collection of real Tensor data at the input/output level of the full-network Module and API levels through configuration:
    1. Full-network real data collection: On the basis of the existing full-network statistical collection, add a new full-network real data disk write mode. Specify the collection target (Module/API level) through configuration, specify the input/output Tensor data of the module to .pt files, and keep the disk write path consistent with the existing dump directory structure (`dump_path/step{step_id}/{rank|pid}/{op_name}/`).
    2. Configuration item extension: In config.json, for the `task` configuration, support the following two collection methods:
       - `"statistics"` (default): Collects only statistical values, consistent with the existing behavior, without additional disk write overhead.
       - `"tensor"` (new): Collects real Tensor data and writes .pt files to disk. The format of each Tensor file is consistent with the .pt file generated by `acl_save`, and can be loaded through `torch.load`.
       - Supports enabling both `"statistics"` and `"tensor"` modes at the same time, outputting both statistical information and real data.

- The aclgraph scenario supports low-precision data format collection
  - Requirement background: In large model inference scenarios, to reduce memory usage and improve computation throughput, more and more models use low-precision data formats such as fp8/fp4 for inference. When performing precision alignment in ACLGraph mode, the msprobe aclgraph_dump tool provides two capabilities: full-network statistical collection (AclGraphDumper) and single-point Tensor disk write (acl_save). The current tool only supports the collection and analysis of data at conventional precisions such as FP16/BF16/FP32. When the model uses fp8/fp4 low-precision data formats, it is necessary to extend support for collecting fp8/fp4 low-precision data formats in aclgraph_dump, improving the tool's usability and accuracy in low-precision inference scenarios.
  - Requirement content: The aclgraph_dump tool extends the collection capability for low-precision data formats (fp8/fp4), supporting correct processing of fp8/fp4 format Tensors in full-network statistical collection (AclGraphDumper) and single-point collection (acl_save)

- Dynamic graph multi-batch supports collection for a specified request
  - Requirement background: In large model inference scenarios, an inference task is usually performed by assembling multiple requests into a batch. Therefore, the data obtained through msprobe dump is a concatenation of multiple requests. When locating precision issues, localization is performed at the request granularity, so a dump capability for a specified request needs to be provided at dump time.
  - Requirement content: The dump tool provides the request_id configuration capability through an interface form, supporting slicing for a specified request. The collected data is subjected to statistical computation or disk write according to the shape after slicing, improving the overall usability of the tool.

## 7.4 DFX Attribute Design

### 7.4.1 Performance Design

*As a diagnostic feature, it is insensitive to performance impact and is therefore not addressed.*

### 7.4.2 Security Design

#### 7.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are added (UI inputs, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team has been notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (which validations: length, format, category, threshold, null check, and whether path-type parameters are normalized and standardized before use) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method, whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether externally provided services have authentication, authorization; all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a company-recommended secure encrypted protocol, whether returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the standard open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information is involved (generation, use, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether secure function libraries are used | No |
