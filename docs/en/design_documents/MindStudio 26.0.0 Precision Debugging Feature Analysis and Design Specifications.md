# MindStudio 26.0.0 Precision Debugging Feature Analysis and Design Specifications

|                                           |                  |
| ----------------------------------------- | ---------------- |
| SIG group:                                | mstt-sig         |
| Incorporated into the following versions: | MindStudio26.0.0 |
| Designer:                                 | wangchao         |
| Date:                                     | 2026.1. 20       |

Copyright © 2026 MindStudio Community

Your reproduction, use, modification and distribution of this document is subject to the Creative Commons Attribution-ShareAlike 4.0 International Public License ("CC BY-SA 4.0"). For ease of understanding, you can visit thehttps://creativecommons.org/licenses/by-sa/4.0/Understand the overview (but not the replacement) of CC BY-SA 4.0. You can obtain the complete agreement of CC BY-SA 4.0 from the following website:https://creativecommons.org/licenses/by-sa/4.0/legalcode.

**Revision records**

| Date       | Revised version | Revision Description | Authors  | Audited |
| ---------- | --------------- | -------------------- | -------- | ------- |
| 2026.1. 20 | 26.0.0          | Design specification | wangchao | xxx     |

# 1. Feature Overview

To meet the computing power requirements of the fast-growing deep neural network, Huawei launched the Ascend series AI processors in 2018. With the rapid development of the AI field, new networks and operators emerge one after another. Operators need to be continuously developed and optimized based on Ascend chips. The operator itself is a formula of mathematical expression and supports data of different dimensions. After the operator is implemented, it is impossible to traverse all input and output. Therefore, the bug is not found. However, thousands of operators with different inputs are used in a large model, resulting in a single operator function problem becoming a network precision problem. At the same time, operator fusion may also have functional problems and ultimately affect the network precision. Precision debugging is the function debugging in the AI field. In addition, low precision is frequently used in AI (based on energy efficiency and performance considerations), and different AI chips have their own low precision expression design. These differences may ultimately affect the precision result of the network. At the same time, in large models, data accumulation may cause beyond the precision expression capability, and the overflow in the calculation process needs to be handled by automatic mixing precision. The arrival of large models further increases the pressure of automatic mixing precision. The precision debugging tool is used to ensure that the AI model can meet the expected precision on the Ascend chip. In order to achieve this goal, there are many methods, and this paper mainly describes the implementation of these methods.

Help locate the functional problems that occur when AI models are migrated from other acceleration chips to Ascend chips. The AI model is different from the application software service process. In the migration scenario, the model before the migration is used as the benchmark, and then the AI model nodes before and after the migration are compared to locate the problem.

Detects and fixes precision exceptions. The precision may be abnormal if the operator result overflows. Therefore, the precision debugging tool needs to detect the overflow. After the overflow occurs, the overflow problem is fixed by using the automatic mixing precision module. Precision debugging is divided into two functional domains: overflow detection (repair) and precision demarcation.

For overflow detection (repair), the overflow detection capability of the Ascend chip is exposed and integrated into the framework to shield the differences in the framework. In addition, considering the capability difference (mainly the data expression capability difference) between the Ascend chip and other acceleration chips, the overflow repair aspect needs to be enhanced. In a large model, the low-precision expression is more likely to cause overflow than the single-node model. Thus, the automatic precision compensation has higher requirements. For precision demarcation, there are operator implementation problems, entire network problems, and accumulated error problems based on the existing problems. For operator implementation, you need to understand the operator composition of the corresponding model. In this case, you do not need to pay attention to the upstream and downstream relationships of operators in the model. The problem of the whole network is essentially caused by the integration of CANN software stack in order to optimize the overall performance when scheduling operators. Accumulated errors are caused by the poor robustness of the Ascend chip hardware in terms of numerical expression capability compared with other accelerators. In the existing precision demarcation tools, only the first two problems can be solved. The cumulative error problem itself is a long-term possible problem in the whole system, and it needs to be solved fundamentally by model interpretability improvement. To locate operator implementation problems and network-wide problems, you need to understand the composition of the model, split the model into nodes and node associations, and analyze the node and node dependency behavior to find the specific problem points in the model.

## 1.1 Scope

Precision demarcation (network-wide problem demarcation)In the existing AI field, there are two scenarios: inference and training. The frameworks used by the two scenarios are greatly different. Therefore, the model expression is greatly different. Model expression differs greatly. As a result, model function problem demarcation tools differ greatly. However, the following functional components are indispensable: model expression, data analysis (comparison), and function problem demarcation. Identify problem points through data analysis, and then locate the initial node that causes the problem through functional problem demarcation. In this process, different model expressions need to be identified. The overall architecture is as follows:![img_1.png](img_1.png)    

Internal capabilities of the tool:

 * Data analysis: Identify nodes/edges with precision problems based on the comparison algorithm.
 * Demarcation and analysis: Locate the first faulty node based on the topology structure.
 * Topology management: The built-in basic data structure constructs node topology relationships based on the collected information and supports analysis of benchmark and target topology identification and comparison of nodes. (The interfaces in the API are used to maintain the topology.)
 * Information capture: Collect topology information and node value information.
 * Program control: Implements debugging control through sub-process control and interface hijacking, such as pause and running. (Program functions are not affected.)
 * Interface hijacking: There are two main tasks: a. Implement tool embedding process; b. Fixed input in the training process

## 1.2 Feature Requirement List

Feature Requirement List

| Requirement No. | Requirement name                                                                                          | Feature Description                                                                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1               | Precision debugging tool msprobe training and general capability enhancement                              | Real-time MD5 difference analysis dump, comparison result highlighting optimization, and monitor normalization reconstruction                         |
| 2               | \[Reinforcement learning\] Reinforcement learning training and promotion consistency positioning solution | Supports training and inference data comparison in the Verl training and recommendation consistency scenario. Supports the fsdp and megatron backend. |
| 3               | Basic capability coverage in the inference scenario                                                       | The basic capabilities of mindie, vllm, and sglang are supported.                                                                                     |
| 4               | \[Recommendation\] Data parsing and visualized analysis capabilities are enhanced.                        | Visualized analysis supports training data trend analysis.                                                                                            |

# 2. Requirement Scenario Analysis

## 2.1 Feature Requirement Source and Value Overview

\[Precision tool chain\] Real-time difference analysis, multi-backend consistency verification, inference engine coverage, and visualization enhancement are used to improve model precision debugging efficiency and reliability assurance capabilities.

## 2.2 Feature Scenario Analysis

The following scenarios are supported:

 * Precision debugging tool msprobe training and general capability enhancement: MD5 real-time difference analysis dump, comparison result highlighting optimization, and monitor normalization reconstruction.
 * \[Reinforcement learning\] Reinforcement learning training and push consistency locating solution: supports training and inference data comparison in the Verl training and push consistency scenario, and supports fsdp and megatron backends.
 * Basic capabilities in the inference scenario: Mindie, vllm, and sglang are supported.
 * \[Training and Promotion\] Enhanced data parsing and visualized analysis capabilities: Visualized analysis supports training data trend analysis.

# 3. Precision debugging tool msProbe training and general capability enhancement

## 3.1 Design Idea

This feature must support the following sub-scenarios:

1. MD5 real-time difference analysis dump
2. Optimize the indicator highlights in the comparison result.

## 3.2 Constraints

Not involved.

## 3.3 Detailed implementation (module-level or process-level message sequence diagram from user entry)

1. MD5 real-time difference analysis dump

Background: During model training on the NPU, deterministic computing is a common challenge. Currently, the MSProbe tool supports the MD5 dump function to check the consistency of output data on the entire network. However, the MSProbe tool lacks the real-time automatic analysis capability. As a result, the MSProbe tool cannot collect unstable data onsite. This requirement aims to quickly capture the real data of MD5 inconsistency by inheriting real-time monitoring and difference analysis.

Requirement description and implementation scheme: A copy of MD5 data of a model is preconfigured in advance. When the tool collects model data again, the tool can determine the difference between each tensor in the current task and the preset MD5 data in advance based on the preconfigured MD5 data. After the difference node is identified, dump the actual data.

## 3.4 DFX Attribute Design

### 3.4.1 Performance Design

*Commissioning features, which are not sensitive to performance impact and are not involved.*

### 3.4.2 Security Design

#### 3.4.2.1 Safety Design Qualification

| Checklist Content                                                                                                                                                                                                                                                                                                                       | Check Result  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| 1 Whether to add an input (interface input, command line parameters, commands, and HTTP interfaces)                                                                                                                                                                                                                                     | Yes           |
| 1.1 Whether to notify the update of information                                                                                                                                                                                                                                                                                         | Yes           |
| 1.2 Check whether security verification is designed for input. (Checks, length, format, type, threshold, whether the parameter is empty, and whether the input parameters of the path class are standardized before being used.)                                                                                                        | Yes           |
| 2. Check whether the processes interact with each other (cross trusted domains).                                                                                                                                                                                                                                                        | Not involved. |
| 2.1 Check whether the inter-process interaction mode and communication mode are reliable.                                                                                                                                                                                                                                               | Not involved. |
| 2.2 Whether resource competition exists                                                                                                                                                                                                                                                                                                 | Not involved. |
| 3. Check whether file operations exist.                                                                                                                                                                                                                                                                                                 | Yes           |
| 3.1 Read external files (Whether to verify the file size, whether to verify the read content, and whether deserialization is secure)                                                                                                                                                                                                    | Yes           |
| 3.2 Generate File Output (Check whether the file permission is correct and whether the soft connection is verified.)                                                                                                                                                                                                                    | Yes           |
| 3.3 Check whether temporary files are generated and cleared in time.                                                                                                                                                                                                                                                                    | No.           |
| 3.4 Decompress Files (Whether to verify the compression bomb, decompression location, and decompression permission)                                                                                                                                                                                                                     | No.           |
| 4. Whether network communication is involved                                                                                                                                                                                                                                                                                            | Not involved. |
| 4.1 Listening Port (Check whether the communication matrix is updated, whether all zeros are listened, whether the protocol uses the security encryption protocol, whether the external service provides authentication, authorization, and web attack mode.)                                                                           | Not involved. |
| 4.2 Whether to access the external network (Check whether the communication matrix is updated, whether the accessed website is in the configuration file, whether the protocol used is the security encryption protocol recommended by Huawei, and whether the returned data is verified.), whether the timeout mechanism is available) | Not involved. |
| 5. Involved injection risks                                                                                                                                                                                                                                                                                                             | Not involved. |
| 5.1 Check whether command execution is involved and whether command injection risks are mitigated.                                                                                                                                                                                                                                      | Not involved. |
| 5.2 Check whether HTML pages are involved and whether HTML injection risks are mitigated (XSS attacks).                                                                                                                                                                                                                                 | Not involved. |
| 5.3 Check whether the JLable control is used and whether the HTML injection risk is mitigated.                                                                                                                                                                                                                                          | Not involved. |
| 5.4 Check whether XML parsing is involved and whether XML injection risks are mitigated.                                                                                                                                                                                                                                                | Not involved. |
| 5.5 Check whether the YAML parsing is involved and whether the secure parsing interface is used.                                                                                                                                                                                                                                        | Not involved. |
| 5.6 SQL Database Injection Is Involved                                                                                                                                                                                                                                                                                                  | Not involved. |
| 6. Import third-party libraries                                                                                                                                                                                                                                                                                                         | Not involved. |
| 6.1 Check whether the open source introduction follows the normal open source introduction process.                                                                                                                                                                                                                                     | Not involved. |
| 6.2 Check whether the Python dependency is added and whether the Python dependency is specific to the version. (Generally, the Python dependency is not allowed to be specific to the version.)                                                                                                                                         | Not involved. |
| 7. Check whether binary deliverables are added (whether the compilation security options meet Huawei requirements).                                                                                                                                                                                                                     | Not involved. |
| 8. Whether encryption and authentication exist (whether secure encryption algorithms are used and whether the encryption and decryption process is secure)                                                                                                                                                                              | Not involved. |
| 9. Check whether sensitive information exists. (Generation, use, retention, and destruction of sensitive information)                                                                                                                                                                                                                   | Not involved. |
| 10 Whether to use the secure function library                                                                                                                                                                                                                                                                                           | No.           |

# 4. Reinforcement learning Reinforcement learning training and consistent positioning solution

## 4.1 Design Idea

This feature must support the following sub-scenarios:

1. Supports training inference data comparison in the Verl training and recommendation consistency scenario.
2. Supports data collection by the dataset loading module.

## 4.2 Constraints

Not involved.

## 4.3 Detailed implementation (module-level or process-level message sequence diagram from user entry)

1. Supports training and inference data comparison in the Verl training and recommendation consistency scenario.

Background: In the model training and inference consistency verification scenarios of the Verl framework, ensure that the intermediate data or final output data generated by the training process (forward propagation) and inference process are consistent under the same input conditions. This is critical for model debugging, precision verification, and deployment reliability.

Requirement objective: Develop a data comparison tool/module to compare key data generated during the training forward process and inference process under the Verl framework, identify and report differences, and help developers verify the consistency of training and promotion.

Core function requirements:

 * Data Collection

Supports automatic capture of key data points in the training forward process and inference process.

Configurable collection layers: layer-by-layer output, specific layer activation value, gradient information, and loss value

Supports multiple data types, such as tensors, scalars, and statistics.

 * Comparison Dimension

Value precision comparison (configurable error tolerance allowed)

Shape/Dimension Consistency Verification

 * Data Type Consistency Check

Check special values (abnormal values such as NaN and Inf)

 * Difference Analysis

Automatically identifies the difference location (layer name, tensor dimension, and index).

Quantify the degree of difference (absolute error, relative error, MSE, etc.)

Variance visualization support

 * Report generation

Generate a detailed comparison report in HTML/JSON format.

Variance Summary Statistics

Suggested Repair Direction

## 4.4 DFX Attribute Design

### 4.4.1 Performance Design

*Commissioning features, which are not sensitive to performance impact and are not involved.*

### 4.4.2 Security Design

#### 4.4.2.1 Safety Design Qualification

| Checklist Content                                                                                                                                                                                                                                                                                                                       | Check Result  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| 1 Whether to add an input (interface input, command line parameters, commands, and HTTP interfaces)                                                                                                                                                                                                                                     | Yes           |
| 1.1 Whether to notify the update of information                                                                                                                                                                                                                                                                                         | Yes           |
| 1.2 Check whether security verification is designed for input. (Checks, length, format, type, threshold, whether the parameter is empty, and whether the input parameters of the path class are standardized before being used.)                                                                                                        | Yes           |
| 2. Check whether processes interact with each other (cross trusted domains).                                                                                                                                                                                                                                                            | Not involved. |
| 2.1 Check whether the inter-process interaction mode and communication mode are reliable.                                                                                                                                                                                                                                               | Not involved. |
| 2.2 Whether resource competition exists                                                                                                                                                                                                                                                                                                 | Not involved. |
| 3. Check whether file operations exist.                                                                                                                                                                                                                                                                                                 | Yes           |
| 3.1 Read External Files (Whether to verify the file size, whether to verify the read content, and whether deserialization is secure)                                                                                                                                                                                                    | Yes           |
| 3.2 Generate File Output (Check whether the file permission is correct and whether the soft connection is verified.)                                                                                                                                                                                                                    | Yes           |
| 3.3 Check whether temporary files are generated and cleared in time.                                                                                                                                                                                                                                                                    | No.           |
| 3.4 Decompress Files (Whether to verify the compression bomb, decompression location, and decompression permission)                                                                                                                                                                                                                     | No.           |
| 4. Whether network communication is involved                                                                                                                                                                                                                                                                                            | Not involved. |
| 4.1 Listening Port (Check whether the communication matrix is updated, whether all zeros are listened, whether the protocol uses the security encryption protocol, whether the external service provides authentication, authorization, and web attack mode.)                                                                           | Not involved. |
| 4.2 Whether to access the external network (Check whether the communication matrix is updated, whether the accessed website is in the configuration file, whether the protocol used is the security encryption protocol recommended by Huawei, and whether the returned data is verified.), whether the timeout mechanism is available) | Not involved. |
| 5 Involved injection risks                                                                                                                                                                                                                                                                                                              | Not involved. |
| 5.1 Check whether command execution is involved and whether command injection risks are mitigated.                                                                                                                                                                                                                                      | Not involved. |
| 5.2 Check whether the HTML interface is involved and whether the HTML injection risk (XSS attack) is mitigated.                                                                                                                                                                                                                         | Not involved. |
| 5.3 Check whether the JLable control is used and whether the HTML injection risk is mitigated.                                                                                                                                                                                                                                          | Not involved. |
| 5.4 Check whether XML parsing is involved and whether XML injection risks are mitigated.                                                                                                                                                                                                                                                | Not involved. |
| 5.5 Check whether the YAML parsing is involved and whether the secure parsing interface is used.                                                                                                                                                                                                                                        | Not involved. |
| 5.6 SQL Database Injection Is Involved                                                                                                                                                                                                                                                                                                  | Not involved. |
| 6. Import third-party libraries                                                                                                                                                                                                                                                                                                         | Not involved. |
| 6.1 Check whether the open source introduction follows the normal open source introduction process.                                                                                                                                                                                                                                     | Not involved. |
| 6.2 Check whether the Python dependency is added and whether the Python dependency is specific to the version. Generally, the Python dependency is not allowed to be specific to the version.                                                                                                                                           | Not involved. |
| 7. Check whether binary deliverables are added (whether the security compilation options meet Huawei requirements).                                                                                                                                                                                                                     | Not involved. |
| 8. Whether encryption and authentication exist (whether secure encryption algorithms are used and whether the encryption and decryption process is secure)                                                                                                                                                                              | Not involved. |
| 9. Check whether sensitive information exists. (Generation, use, retention, and destruction of sensitive information)                                                                                                                                                                                                                   | Not involved. |
| 10 Whether to use the secure function library                                                                                                                                                                                                                                                                                           | No.           |

# 5. Basic capability coverage in the inference scenario

## 5.1 Design Ideas

This feature must support the following sub-scenarios:

1. Dynamically starting and stopping dump in the vllm scenario
2. Basic capability support for the sglang dynamic diagram scenario

## 5.2 Constraints

Not involved.

## 5.3 Detailed implementation (module level or process level message sequence diagram from user entry)

With the wide application of the large model inference frameworks vLLM and SGLang, key data needs to be collected and analyzed to support the following functions:

 * Model performance optimization analysis
 * Inference accuracy verification
 * Resource usage monitoring
 * Troubleshooting and Debugging

## 5.4 DFX Attribute Design

### 5.4.1 Performance Design

*This feature is a commissioning feature and has no impact on performance. Therefore, this feature is not involved.*

### 5.4.2 Security Design

#### 5.4.2.1 Safety Design Qualification

| Checklist Content                                                                                                                                                                                                                                                                                                                       | Check Result  |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| 1. Add new input (interface input, command line parameters, commands, and HTTP interfaces)                                                                                                                                                                                                                                              | Yes           |
| 1.1 Whether to notify the update of information                                                                                                                                                                                                                                                                                         | Yes           |
| 1.2 Check whether security verification is designed for input. (Checks, length, format, type, threshold, whether the parameter is empty, and whether the input parameters of the path class are standardized before being used.)                                                                                                        | Yes           |
| 2. Check whether processes interact with each other (cross trusted domains).                                                                                                                                                                                                                                                            | Not involved. |
| 2.1 Check whether the inter-process interaction mode and communication mode are reliable.                                                                                                                                                                                                                                               | Not involved. |
| 2.2 Whether resource competition exists                                                                                                                                                                                                                                                                                                 | Not involved. |
| 3. Check whether file operations exist.                                                                                                                                                                                                                                                                                                 | Yes           |
| 3.1 Read External Files (Whether to verify the file size, whether to verify the read content, and whether deserialization is secure)                                                                                                                                                                                                    | Yes           |
| 3.2 Generate File Output (Check whether the file permission is correct and whether the soft connection is verified.)                                                                                                                                                                                                                    | Yes           |
| 3.3 Check whether temporary files are generated and cleared in time.                                                                                                                                                                                                                                                                    | No.           |
| 3.4 Decompress Files (Whether to verify the compression bomb, decompression location, and decompression permission)                                                                                                                                                                                                                     | No.           |
| 4. Whether network communication is involved                                                                                                                                                                                                                                                                                            | Not involved. |
| 4.1 Listening Port (Check whether the communication matrix is updated, whether all zeros are listened, whether the protocol uses the security encryption protocol, whether the external service provides authentication, authorization, and web attack mode.)                                                                           | Not involved. |
| 4.2 Whether to access the external network (Check whether the communication matrix is updated, whether the accessed website is in the configuration file, whether the protocol used is the security encryption protocol recommended by Huawei, and whether the returned data is verified.), whether the timeout mechanism is available) | Not involved. |
| 5 Involved injection risks                                                                                                                                                                                                                                                                                                              | Not involved. |
| 5.1 Check whether command execution is involved and whether command injection risks are mitigated.                                                                                                                                                                                                                                      | Not involved. |
| 5.2 Check whether HTML pages are involved and whether HTML injection risks (XSS attacks) are mitigated.                                                                                                                                                                                                                                 | Not involved. |
| 5.3 Check whether the JLable control is used and whether the HTML injection risk is mitigated.                                                                                                                                                                                                                                          | Not involved. |
| 5.4 Check whether XML parsing is involved and whether XML injection risks are mitigated.                                                                                                                                                                                                                                                | Not involved. |
| 5.5 Check whether the YAML parsing is involved and whether the secure parsing interface is used.                                                                                                                                                                                                                                        | Not involved. |
| 5.6 SQL Database Injection Is Involved                                                                                                                                                                                                                                                                                                  | Not involved. |
| 6. Import third-party libraries                                                                                                                                                                                                                                                                                                         | Not involved. |
| 6.1 Check whether the open source introduction follows the normal open source introduction process.                                                                                                                                                                                                                                     | Not involved. |
| 6.2 Check whether the Python dependency is added and whether the Python dependency is specific to the specific version. Generally, the Python dependency is not allowed to be specific to the specific version.                                                                                                                         | Not involved. |
| 7. Check whether binary deliverables are added (whether the security compilation options meet Huawei requirements).                                                                                                                                                                                                                     | Not involved. |
| 8. Whether encryption and authentication exist (whether secure encryption algorithms are used and whether the encryption and decryption process is secure)                                                                                                                                                                              | Not involved. |
| 9. Check whether sensitive information exists. (Generation, use, retention, and destruction of sensitive information)                                                                                                                                                                                                                   | Not involved. |
| 10 Whether to use the secure function library                                                                                                                                                                                                                                                                                           | No.           |

# 6. Recommendation The data parsing and visualized analysis capabilities are enhanced

## 6.1 Design Idea

This feature must support the following sub-scenarios:

1. Visualized analysis supports training trend analysis.
2. Analysis of TP, PP, VPP, and DP simulation models

## 6.2 Constraints

Not involved currently

## 6.3 Detailed Implementation

As the Verl framework is widely used in deep learning training and inference scenarios, the current data analysis and visualization capabilities face the following core problems:

 * Data Parsing Limitations

Single format support: Currently, only the basic Tensor data format is supported. There is no complete parsing of complex nested structures, distributed training data, and mixed precision data.

The granularity is not fine enough: Only the data statistics can be collected at the overall model level. The data insights at the fine-grained and neuron levels are lacking.

Insufficient real-time performance: Data is collected in post-event analysis mode during training, and the change trend of key indicators cannot be monitored in real time.

Missing metadata: The parsed data lacks context metadata. (e.g. training steps, learning rate, gradient norm, etc.). It is difficult to analyze correlation.

## 6.4 DFX Attribute Design

### 6.4.1 Performance Design

*This feature is a commissioning feature and has no impact on performance. Therefore, this feature is not involved.*

### 6.4.2 Security Design

#### 6.4.2.1 Safety Design Qualification

| Checklist Content                                            | Check Result  |
| ------------------------------------------------------------ | ------------- |
| 1 Whether to add an input (interface input, command line parameters, commands, and HTTP interfaces) | Yes           |
| 1.1 Whether to notify the update of information              | Yes           |
| 1.2 Check whether security verification is designed for input. (Checks, length, format, type, threshold, whether the parameter is empty, and whether the input parameters of the path class are standardized before being used.) | Yes           |
| 2. Check whether processes interact with each other (cross trusted domains). | Not involved. |
| 2.1 Check whether the inter-process interaction mode and communication mode are reliable. | Not involved. |
| 2.2 Check whether resource competition exists.               | Not involved. |
| 3. Check whether file operations exist.                      | Yes           |
| 3.1 Read external files (Whether to verify the file size, whether to verify the read content, and whether deserialization is secure) | Yes           |
| 3.2 Generate File Output (Check whether the file permission is correct and whether the soft connection is verified.) | Yes           |
| 3.3 Check whether temporary files are generated and cleared in time. | No.           |
| 3.4 Decompress Files (Whether to verify the compression bomb, decompression location, and decompression permission) | No.           |
| 4. Whether network communication is involved                 | Not involved. |
| 4.1 Listening Port (Check whether the communication matrix is updated, whether all zeros are listened, whether the protocol uses the security encryption protocol, whether the external service provides authentication, authorization, and web attack mode.) | Not involved. |
| 4.2 Whether to access the external network (Check whether the communication matrix is updated, whether the accessed website is in the configuration file, whether the protocol used is the security encryption protocol recommended by Huawei, and whether the returned data is verified.), whether the timeout mechanism is available) | Not involved. |
| 5. Involved injection risks                                  | Not involved. |
| 5.1 Check whether command execution is involved and whether command injection risks are mitigated. | Not involved. |
| 5.2 Check whether HTML pages are involved and whether HTML injection risks (XSS attacks) are mitigated. | Not involved. |
| 5.3 Check whether the JLable control is used and whether the HTML injection risk is mitigated. | Not involved. |
| 5.4 Check whether XML parsing is involved and whether XML injection risks are mitigated. | Not involved. |
| 5.5 Check whether the YAML parsing is involved and whether the secure parsing interface is used. | Not involved. |
| 5.6 SQL Database Injection Is Involved                       | Not involved. |
| 6. Import the third-party library.                           | Not involved. |
| 6.1 Check whether the open source introduction follows the normal open source introduction process. | Not involved. |
| 6.2 Check whether the Python dependency is added and whether the Python dependency is specific to the specific version. Generally, the Python dependency is not allowed to be specific to the specific version. | Not involved. |
| 7. Check whether binary deliverables are added (whether the compilation security options meet Huawei requirements). | Not involved. |
| 8. Whether encryption and authentication exist (whether secure encryption algorithms are used and whether the encryption and decryption process is secure) | Not involved. |
| 9. Check whether sensitive information exists. (Generation, use, retention, and destruction of sensitive information) | Not involved. |
| 10 Whether to use the secure function library                | No.           |
