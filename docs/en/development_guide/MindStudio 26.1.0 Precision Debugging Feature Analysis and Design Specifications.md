# MindStudio 26.1.0 Precision Debugging Feature Analysis and Design Specification

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:43:33.335Z pushedAt=2026-08-11T02:51:07.164Z -->

<table>
    <tr>
        <td>Affiliated SIG:</td>
        <td>mstt-sig</td>
    </tr>
    <tr>
        <td>Target Version:</td>
        <td>MindStudio 26.1.0</td>
    </tr>
    <tr>
        <td>Designer:</td>
        <td>RanZheng</td>
    </tr>
    <tr>
        <td>Date:</td>
        <td>2026.05.07</td>
    </tr>
</table>
**Copyright © 2026 MindStudio Community**

Your reproduction, use, modification, and distribution of "this document" are governed by the Creative Commons Attribution-ShareAlike 4.0 International Public License (hereinafter referred to as "CC BY-SA 4.0"). 
For ease of understanding, you may visit <https://creativecommons.org/licenses/by-sa/4.0/> to review a summary of CC BY-SA 4.0 (which is not a substitute for the full license). 
The full text of CC BY-SA 4.0 is available at the following URL: <https://creativecommons.org/licenses/by-sa/4.0/legalcode>.

# 1. Feature Overview

In recent years, large language models (LLMs), multimodal models (e.g., text-to-image/video), and mixture of experts (MoE) models have undergone rapid iteration, with model parameters surging from the billion-level to the trillion-level, while training data scale, context length, and the complexity of distributed parallelism strategies (TP/DP/PP) have increased simultaneously. 
The underlying AI computing infrastructure has migrated from a GPU-centric ecosystem to heterogeneous Ascend NPU clusters, making cross-hardware, cross-framework (PyTorch/MindSpore), and cross-precision (FP32/FP16/BF16/INT8) deployment a norm. 
Against this backdrop, precision stability has become the core lifeline for large model deployment: during training, issues such as loss oscillation/non-convergence, gradient explosion/vanishing, and NaN/Inf overflow frequently arise; during inference, problems including output drift, prediction accuracy degradation, and generation quality deterioration are common; when migrating models from GPU to Ascend NPU, precision misalignment and numerical deviations in custom operators occur frequently. Moreover, these issues are triggered with randomness, concealment, and long-tail characteristics, rendering traditional debugging approaches inadequate.

Currently, large model precision troubleshooting generally relies on manual log printing, third-party comparison tools, and scattered scripts provided by hardware vendors. Within the Ascend NPU ecosystem, the pain points are particularly pronounced:

- Severe black-box nature with a lack of observability
- Fragmented tooling without unified closed-loop capability
- The proliferation of diverse large model frameworks results in poor adaptability and high performance overhead.

To address the aforementioned pain points and fill the gap in native precision debugging tools within the Ascend ecosystem, the Huawei Ascend team developed MindStudio-Probe (abbreviated as msProbe) based on the MindStudio training toolchain. It was officially open-sourced at the end of 2025 and is hosted in the GitCode Ascend/msProbe repository.
msProbe is positioned as a full-scenario precision debugging toolkit for the Ascend ecosystem. Its core objective is to adopt "lossless lightweight, full-link closed-loop, native NPU adaptation, and large model first" as its design philosophy, thereby building full-link capabilities spanning pre-training risk pre-checking, real-time monitoring during training, fine-grained collection of anomalous data, automated numerical comparison, overflow/NaN detection, interactive visual analysis, and root cause localization. This fundamentally breaks through the black-box barrier of traditional debugging, transforming "black-box training and inference" into "white-box observability," helping developers rapidly resolve large model precision issues, lowering the debugging threshold, shortening the development cycle, and ensuring the precision stability and performance advantages of models on Ascend NPUs.

## 1.1 Scope

- Data Collection Capabilities
    - Supports module-level and API-level collection for PyTorch full-network training and inference, providing L0/L1/mix hierarchical dump capabilities.
    - Supports lightweight monitoring of training states such as weights, gradients, and optimizer states during model training and inference.
    - Supports filtered collection by module, device, keyword, and network layer to reduce data volume and improve collection performance.
    - Supports stack trace information collection and optimization, preserving the context of the precision issue site.
    - Supports single-point data collection for aclgraph, and supports data dump and disk write for ATB operators.
- Data Comparison Capability
    - Supports offline numerical comparison between Ascend and benchmark data.
    - Supports multi-dimensional precision difference analysis, including statistics, tensor, and MD5.
    - Supports numerical alignment verification across different frameworks, devices, ranks, and iteration steps.
    - Outputs standardized comparison results and error reports for locating the root cause of precision non-compliance.
- Data Visualization Capabilities
    - Provides hierarchical model visualization, graph structure display, and node matching views.
    - Supports multi-DB file switching and trend visualization for convenient time-series comparative analysis.
    - Provides visualization panels for error distribution, anomaly points, and matching results.
    - Supports visual presentation of task matching and tensor computation processes.
- Automated Analysis Capabilities
    - Automatically detects typical precision issues such as numerical overflow and gradient anomalies.
    - Automatically locates the starting layer of precision discrepancies, abnormal operators, and abnormal devices.
    - Automated precision issue diagnosis and root cause assisted localization are provided.
    - Full-process automated integration is supported, adapting to CI/CD and batch localization scenarios.

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
        <td>msProbe tool usability, availability, and performance improvement initiative</td>
        <td>Acquisition support for device MD5, support for stack/construct switches, and support for acquisition blacklist; performance optimization for statistical value acquisition, overflow node detection, and comparison; visualization support for tensor-level point-to-point matching and matching functionality enhancement.</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Precision localization solution for PyTorch graph mode scenarios</td>
        <td>Supports automatic comparison for Inductor (Triton backend), MLIR, AKG, DVM, and eager->aot eager, and supports aclgraph full-graph data collection.</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Enhanced reinforcement learning precision localization capability</td>
        <td>Supports verl key hyperparameter comparison and verification, verl fully_async scenario, sglang inference, PD separation scenario data collection, and token-level logp_diff monitoring.</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Optimization of issue reproduction affected by msProbe tool enablement</td>
        <td>Support for register overflow status read, clear operator encapsulation, and tool adaptation</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Construction of serving sub-health status awareness capability</td>
        <td>Support for proactive awareness of duplicate and garbled output issues</td>
    </tr>
    <tr>
        <td>6</td>
        <td>msProbe Dump API Support Enhancement Special Initiative</td>
        <td>Enhanced dump capability for quantization operators, fusion operators, and custom operators</td>
    </tr>
</table>

# 2. Requirements Scenario Analysis

## 2.1 Feature Requirement Sources and Value Overview

[Precision Toolchain] By enhancing usability, reducing the impact on issue reproduction, strengthening collection and demarcation capabilities in reinforcement learning scenarios, filling API support gaps, and improving graph mode localization capabilities, the precision toolchain aims to boost model precision debugging efficiency and training/inference reliability assurance capabilities.

## 2.2 Feature Scenario Analysis

The following scenarios are primarily supported:

- Tool usability: addressing issues such as missed collection, incorrect collection, and slow collection
- PyTorch graph mode: comprehensively supplementing precision collection and analysis capabilities in graph mode
- Reinforcement learning precision localization: further extending support for the latest inference scenarios in reinforcement learning, and strengthening automatic configuration checking and delimitation capabilities
- Construction of serving sub-health status awareness capability: enhancing proactive awareness of repetition and garbled output patterns

# 3. Tool Usability, Availability, and Performance Enhancement Initiative

## 3.1 Design Approach

This feature is required to support the following sub-scenario capabilities:

1. Collection support for device MD5
2. Support for the stack/construct switch
3. Support for collection blacklist
4. Statistical value collection, overflow node detection, and comparison performance optimization;
5. Visualization support for tensor-level point-to-point matching and matching capability enhancement.

## 3.2 Constraints

N/A.

## 3.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- Device MD5 Collection Support
  - Requirement Background: The original msProbe tool only supported host MD5 computation, which was time-consuming and did not support asynchronous disk writing.
  - Requirement Description: Implement device-side MD5 computation through XOR and other methods, and integrate the device MD5 fused operator provided by the operator side.

- Stack/Construct Switch Support
  - Requirement Background: During network training, dump.json, stack.json, and construct.json are collected at each step by default. Among them, stack.json and construct.json contain substantial duplication across different steps, and it is not recommended to save them at every step.
  - Requirement Description: Stack and construct collection switches are supported. When these switches are turned off, stack and construct collection is not performed.
- Collection Blacklist Support
  - Requirement Background: Interference information exists in the network, such as empty, mask, and communication placeholder inputs and outputs. During overflow detection or deterministic comparison, such information can easily interfere with the user's ability to locate the first NaN node or the first divergent node.
  - Requirement Description: A blacklist YAML configuration is supported. The blacklist allows masking of an entire API or one or more specific inputs/outputs of that API. Results covered by the blacklist are replaced with null.

- Statistical value collection / overflow node detection / comparison performance optimization
  - Requirement background: Collection and analysis are time-consuming and require dedicated performance optimization.
  - Requirement content: Each shall be implemented independently.

- Visualization support for tensor-level point-to-point matching and matching capability enhancement
  - Requirement background: When using precision tools in reinforcement learning scenarios involving multi-framework comparison, a large number of nodes cannot be automatically matched due to name differences, structural ordering, and other factors, requiring manual intervention by the user. However, the current hierarchical visualization tool only supports point-to-point matching for statistics, and does not yet support manual matching for finer-grained tensor data.
  - Requirement: The hierarchical visualization backend shall provide a backend interface for tensor-level point-to-point matching, and the UI development shall be carried out in collaboration with the visualization team.

## 3.4 DFX Attribute Design

### 3.4.1 Security Design

#### 3.4.1.1 Security Design Confirmation

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

# 4. PyTorch Graph Mode Scenario Precision Debugging Capability Solution

## 4.1 Design Approach

This feature is required to support the following sub-scenario capabilities:

1. Support automatic precision comparison for Inductor (Triton backend), with configurable precision thresholds.
2. The precision comparison tool shall support MLIR, AKG, DVM, and multi-stage compilation (eager → AOT eager).
3. Support full-network data collection in aclgraph scenarios.

## 4.2 Constraints

N/A.

## 4.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

- Support automatic accuracy comparison for Inductor (Triton backend), with configurable accuracy thresholds
  - Requirement Background: It is necessary to compare the accuracy of the entire network between eager mode and after automatic operator generation, in order to identify nodes that fail to meet the accuracy criteria
  - Requirement Objective: Support automatic accuracy comparison for Inductor (Triton backend), with configurable accuracy thresholds
- The accuracy comparison tool supports mlir, akg, dvm, and automatic accuracy comparison across multiple compilation stages (eager → AOT eager)
  - Requirement background: Insufficient graph mode support
  - Requirement objective: Provide precision tool support for graph mode backends including MLIR, AKG, DVM, and multi-stage compilation (eager → AOT eager)
- Support full-network data collection in aclgraph scenarios
  - Requirement background: Currently, aclgraph only supports single-point data saving, requiring users to instrument data at multiple locations, which results in low localization efficiency
  - Requirement objective: Provide corresponding dump interfaces to enable module-level full-network data dumping, allowing users to perform custom comparisons based on the collected data

## 4.4 DFX Attribute Design

### 4.4.1 Performance Design

*As a diagnostic feature, it is insensitive to performance impact and is therefore not addressed.*

### 4.4.2 Security Design

#### 4.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are introduced (UI inputs, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team has been notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (what validations: length, format, type, threshold, empty check; whether path-type parameters are standardized and normalized before use, etc.) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method; whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether file operations exist | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether authentication and authorization are in place for externally provided services; all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a secure encrypted protocol recommended by the company, whether returned data is validated, whether a timeout mechanism exists) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved; whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved; whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used; whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved; whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved; whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the normal open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added; whether there is a dependency on a specific version (dependency on a specific version is generally not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used; whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information is involved (generation, usage, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether the secure function library is used | No |

# 5. Reinforcement Learning Precision Localization Capability Enhancement

## 5.1 Design Approach

This feature is required to support the following sub-scenario capabilities:

1. Support comparison and verification of key hyperparameters in verl
2. Support collection in verl for fully_async scenarios, sglang scenarios, and PD separation scenarios
3. Per-token-level logp_diff monitoring

## 5.2 Constraints

N/A.

## 5.3 Detailed Implementation (Module-Level or Process-Level Message Sequence Diagram from the User Entry Point)

With the widespread adoption of large model inference frameworks such as vLLM and SGLang, it is necessary to collect and analyze key data during the inference process to support the following:

- Support comparison and verification of key verl hyperparameters
  - Requirement background: When verl is trained on NPUs and GPUs, precision issues often arise due to configuration discrepancies. Relying solely on the launch shell script for diagnosis may result in missed validations or failure to identify default hyperparameters.
  - Requirement content: Based on training logs, parse the configurations that actually take effect and perform differential comparison. Additionally, for the provided list of key hyperparameters, assist users in conducting rapid rule-based verification.
- verl supports collection schemes for fully_async scenarios, sglang scenarios, and PD separation scenarios
  - Requirement background: After the iteration of the new verl version, the collection scheme may change when using paths such as fully_async, necessitating investigation and verification of the collection scheme.
  - Requirement content: Verification of the collection scheme under fully_async scenarios, and verification of the sglang collection scheme
- Per-token logp_diff monitoring
  - Requirement background: The current logp_diff metric in verl reinforcement learning only provides a single overall value at the end, making it impossible to distinguish differences at the prefill stage or at a specific decode step.
  - Requirement: A per-token level logp_diff metric is added, with support for its computation and analysis.

## 5.4 DFX Attribute Design

### 5.4.1 Performance Design

_Debugging-oriented features are insensitive to performance impact and are not covered here._

### 5.4.2 Security Design

#### 5.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new input is added (UI input, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation update is notified | Yes |
| 1.2 Whether security validation is designed for the input (which validations, length, format, category, threshold, whether empty, whether path-type parameters are standardized and normalized before use, etc.) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Whether the inter-process interaction method and communication method are trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated for output (whether the generated file permissions are correct, whether symbolic link validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether compression bombs are validated, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether authentication and authorization are provided for external services, all web attack patterns need attention, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a secure encrypted protocol recommended by the company, whether the returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the normal open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption and authentication exist (whether secure encryption algorithms are used, whether the encryption and decryption process is secure) | Not applicable |
| 9 Whether sensitive information exists (generation, use, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether the secure function library is used | No |

# 6. Optimization of Issue Reproduction Affected by msProbe Tool Enablement

## 6.1 Design Approach

This feature is required to support the following sub-scenario capabilities:

1. Torch framework adaptation development for the register overflow status read-and-clear operator
2. Torch framework adaptation development for the stub kernel operator
3. msProbe adaptation to the register overflow detection mode
4. msProbe Adaptation for Stub Kernel Dump Collection

## 6.2 Constraints

N/A.

## 6.3 Detailed Implementation

- Torch framework adaptation development for the register overflow status read/clear operator
  - Requirement background: When encountering NaN issues that are unstable to reproduce, a tensor collection interface that is as lightweight as possible is required. The runtime can provide such an interface, but it needs to be wrapped as a torch interface.
  - Requirement content: Encapsulate getfloatstatus and clearstatus as torch-callable interfaces.
- Torch framework adaptation development for the stub kernel operator
  - Requirement Background: When encountering non-deterministically reproducible NaN issues, a tensor collection interface that is as lightweight as possible is required. The runtime can provide such an interface, but it needs to be encapsulated as a torch interface.
  - Requirement Description: Encapsulate the stub kernel as a torch-callable interface.
- msProbe Adaptation to the Register Overflow Detection Mode
  - Requirement Background: Same as above.
  - Requirement Description: Support the register overflow detection mode, in which asynchronous data dumping of register states can be performed.
- msProbe adapts to the stub kernel dump collection capability
  - Requirement background: same as above
  - Requirement description: supports the stub kernel tensor collection mode, in which tensor data can be dumped to disk

## 6.4 DFX Attribute Design

### 6.4.1 Performance Design

_Debugging-oriented features are insensitive to performance impact and are not covered here._

### 6.4.2 Security Design

#### 6.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new inputs are added (UI inputs, CLI parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team is notified for updates | Yes |
| 1.2 Whether security validation is designed for inputs (what validations: length, format, type, threshold, null check, whether path-type parameters are standardized and normalized before use, etc.) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Inter-process interaction method, whether the communication method is trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated for output (whether generated file permissions are correct, whether symlink validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are validated, whether decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether listening on all zeros, whether the protocol uses a secure encrypted protocol, whether external services have authentication and authorization, all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URL is in the configuration file, whether the protocol used is a secure encrypted protocol recommended by the company, whether returned data is validated, whether there is a timeout mechanism) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, whether command injection risks are mitigated | Not applicable |
| 5.2 Whether HTML interfaces are involved, whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether JLabel controls are used, whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the normal open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, whether there are specific version dependencies (specific version dependencies are generally not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information exists (generation, usage, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether a secure function library is used | No |

# 7. Construction of Serving Sub-Health Status Awareness Capability

## 7.1 Design Approach

This feature is required to support the following sub-scenario capabilities:

1. The serving sub-health state supports garbled-text perception capability.
2. The serving sub-health state supports repetition perception capability.

## 7.2 Constraints

N/A.

## 7.3 Detailed Implementation

- The serving sub-health state supports garbled-text perception capability
  - Requirement Background: Currently, there is a lack of automated detection capability for garbled-text issues.
  - Requirement Content: The garbled-text perception capability is supported through a garbled-text detection algorithm based on character encoding validation and semantic coherence analysis.
- The serving sub-health state supports duplicate-content perception capability
  - Requirement Background: Currently, there is a lack of automated detection capability for repetitive issues.
  - Requirement Content: Repetition awareness capability is supported through a repetition detection algorithm based on token_id sequences.

## 7.4 DFX Attribute Design

### 7.4.1 Performance Design

_As a debugging-oriented feature, it is insensitive to performance impact and is therefore not applicable._

### 7.4.2 Security Design

#### 7.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new input is introduced (UI input, command-line parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether documentation update is notified | Yes |
| 1.2 Whether security validation is designed for the input (what validations: length, format, category, threshold, null check; whether path-type parameters are normalized and standardized before use, etc.) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | N/A |
| 2.1 Whether the inter-process interaction method and communication method are trusted | N/A |
| 2.2 Whether there is resource contention | N/A |
| 3 Whether there are file operations | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether symbolic link validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether compression bombs are validated, whether decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | N/A |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether authentication and authorization are provided for external services; all web attack patterns need attention, XSS, etc.) | N/A |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URLs are in the configuration file, whether the protocol used is a secure encrypted protocol recommended by the company, whether returned data is validated, whether there is a timeout mechanism) | N/A |
| 5 Whether injection risks are involved | N/A |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | N/A |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | N/A |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | N/A |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | N/A |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | N/A |
| 5.6 Whether SQL database injection is involved | N/A |
| 6 Whether third-party libraries are introduced | N/A |
| 6.1 Whether open source introduction follows the normal open source introduction process | N/A |
| 6.2 Whether new Python dependencies are added, and whether there are dependencies on specific versions (generally, depending on specific versions is not allowed) | N/A |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | N/A |
| 8 Whether encryption and authentication exist (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | N/A |
| 9 Whether sensitive information exists (generation, use, retention, and destruction of sensitive information) | N/A |
| 10 Whether the secure function library is used | No |

# 8. msProbe Dump API Support Completion Initiative

## 8.1 Design Approach

This feature is required to support the following sub-scenario capabilities:

1. Complementing the collection of quantization operators
2. Complementing the collection of fusion operators
3. Improving the usability of custom operator collection

## 8.2 Constraints

N/A.

## 8.3 Detailed Implementation

- Supplementing Collection of Quantization Operators
  - Requirement Background: Collection omissions exist for quantization operators.
  - Requirement Description: The operator collection list is supplemented based on the comprehensive quantization operator inventory provided by the framework side.
- Supplementing Collection of Fusion Operators
  - Requirement background: Collection omissions exist for fused operators.
  - Requirement content: Supplement the tool's collection list based on the comprehensive fused operator inventory provided by the framework side.
- Improve the usability of custom operator collection.
  - Requirement background: Collection omissions exist for custom operators.
  - Requirement content: Support batch custom operator registration capability.

## 8.4 DFX Attribute Design

### 8.4.1 Performance Design

_This is a debugging feature, which is insensitive to performance impact and is therefore not applicable._

### 8.4.2 Security Design

#### 8.4.2.1 Security Design Confirmation

| Checklist Item | Check Result |
| --- | --- |
| 1 Whether new input is added (UI input, CLI parameters, commands, HTTP interfaces) | Yes |
| 1.1 Whether the documentation team is notified for updates | Yes |
| 1.2 Whether security validation is designed for the input (what validations: length, format, type, threshold, empty check, whether path-type parameters are standardized and normalized before use, etc.) | Yes |
| 2 Whether there is (cross-trust-domain) inter-process interaction | Not applicable |
| 2.1 Whether the inter-process interaction method and communication method are trusted | Not applicable |
| 2.2 Whether there is resource contention | Not applicable |
| 3 Whether file operations exist | Yes |
| 3.1 Whether external files are read (whether file size is validated, whether read content is validated, whether deserialization is secure) | Yes |
| 3.2 Whether files are generated as output (whether generated file permissions are correct, whether soft link validation is performed) | Yes |
| 3.3 Whether temporary files are generated (whether they are cleaned up in a timely manner) | No |
| 3.4 Whether files are decompressed (whether zip bombs are checked, whether the decompression location is validated, whether decompression permissions are validated, etc.) | No |
| 4 Whether network communication is involved | Not applicable |
| 4.1 Whether a port is listened on (whether the communication matrix is updated, whether all-zero listening is used, whether the protocol uses a secure encrypted protocol, whether authentication and authorization are provided for external services, all web attack patterns must be considered, XSS, etc.) | Not applicable |
| 4.2 Whether external networks are accessed (whether the communication matrix is updated, whether the accessed URL is in the configuration file, whether the protocol used is a secure encrypted protocol recommended by the company, whether the returned data is validated, whether a timeout mechanism exists) | Not applicable |
| 5 Whether injection risks are involved | Not applicable |
| 5.1 Whether command execution is involved, and whether command injection risks are mitigated | Not applicable |
| 5.2 Whether an HTML interface is involved, and whether HTML injection risks are mitigated (XSS attacks) | Not applicable |
| 5.3 Whether the JLabel control is used, and whether HTML injection risks are mitigated | Not applicable |
| 5.4 Whether XML parsing is involved, and whether XML injection risks are mitigated | Not applicable |
| 5.5 Whether YAML parsing is involved, and whether a secure parsing interface is used | Not applicable |
| 5.6 Whether SQL database injection is involved | Not applicable |
| 6 Whether third-party libraries are introduced | Not applicable |
| 6.1 Whether open source introduction follows the normal open source introduction process | Not applicable |
| 6.2 Whether new Python dependencies are added, and whether there is a dependency on a specific version (dependency on a specific version is generally not allowed) | Not applicable |
| 7 Whether new binary deliverables are added (whether security compilation options comply with company requirements) | Not applicable |
| 8 Whether encryption or authentication exists (whether secure encryption algorithms are used, whether the encryption/decryption process is secure) | Not applicable |
| 9 Whether sensitive information exists (generation, usage, retention, and destruction of sensitive information) | Not applicable |
| 10 Whether secure function libraries are used | No |
