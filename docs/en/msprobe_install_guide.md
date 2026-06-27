# msProbe Installation Guide

## Environment and Dependencies

Before using msProbe, ensure that an executable AI application exists in the Ascend environment.

The CANN Toolkit and operator package of the matching version have been installed, and environment variables have been configured. For details, see [CANN Quick Installation](https://www.hiascend.com/cann/download).

## Installation Description

This document describes how to install msProbe. Currently, msProbe can be installed from PyPI, by downloading the .whl package, or by compilation.

It is recommended to use [miniconda](https://docs.anaconda.com/miniconda/) to manage environment dependencies.

```bash
conda create -n msprobe python=3.10
conda activate msprobe
```

## Constraints and Precautions

- All paths read and written by the tool, such as `config_path` and `dump_path`, can contain only letters, digits, underscores (_), slashes (/), periods (.), and hyphens (-).

- To ensure security and adhere to the principle of least privilege, you are advised to install and run this tool as a standard user rather than a high-privilege user (such as `root`).

- Ensure the execution user's **umask** value is set to **0027** or higher to prevent excessive permissions for generated accuracy data files and directories.

- Users must strictly adhere to the principle of least privilege; for example, input files must not be writable by "others." In some function scenarios with stricter security requirements, ensure that the input files are not writable by group users.

- It is recommended that the msProbe execution user be the same as the installation user. If `root` is used for execution, pay attention to the security risks caused by the high permissions of `root`.

## Installation from PyPI

```bash
pip install mindstudio-probe --pre
```

Currently, the msProbe version is a pre-release version. Add `--pre` to the end of the command for installation.

If the following information is displayed, msProbe is successfully installed:

`Successfully installed mindstudio-probe-{version}`

## Installation via the .whl Package

Download the msProbe .whl package by referring to [Release Notes](./release_notes.md).

After obtaining the .whl package, run the following command to install it:

```bash
sha256sum {name}.whl # Verify the .whl package. If the verification codes are the same, the .whl package is not damaged during download.
```

```bash
pip install ./mindstudio_probe-{version}-py3-none-any.whl # Install the .whl package.
```

If the following information is displayed, msProbe is successfully installed:

`Successfully installed mindstudio-probe-{version}`

To overwrite the existing installation, add `--force-reinstall` to the end of the command.

The .whl package does not contain the adump, aclgraph_dump, and atb_probe functions. To use these functions, download the source code and compile the .whl package by referring to [Compilation and Installation](#compilation-and-installation).

## Compilation and Installation

**Description**

The **setup.py** script can be used to compile the .whl software package of msProbe.

**Syntax**

```bash
python3 setup.py bdist_wheel [--include-mod=<include_mode>] [--no-check]
```

**Parameters**

| Parameter         | Required/Optional| Description                                                        |
| ------------- | :-------: | ------------------------------------------------------------ |
| --include-mod |   Optional   | Specifies a module. The options are as follows:<br>&#8226; **adump**: adds the adump module when compiling the .whl package. The adump module is used for L2 dump in MindSpore static graph scenarios. Only MindSpore 2.5.0 and later versions support the adump module.<br>&#8226; **tb_graph_ascend**: adds the hierarchical model visualization plugin when compiling the .whl package. Node.js v20.19.3 and Npm v10.8.2 are recommended for hierarchical model visualization. For details about the dependencies and functions of the plugin, see [Graph Comparison in Hierarchical Visualization in PyTorch](./accuracy_compare/pytorch_visualization_instruct.md) or [Graph Comparison in Hierarchical Visualization in MindSpore](./accuracy_compare/mindspore_visualization_instruct.md).<br>&#8226; **trend_analyzer**: adds the trend visualization plugin when compiling the .whl package. For details about the trend visualization plugin, see [Trend Visualization](./accuracy_compare/trend_visualization_instruct.md).<br>&#8226; **atb_probe**: adds the atb_probe module when compiling the .whl package. The atb_probe module is used to collect data in the ATB inference scenario.<br>&#8226; **aclgraph_dump**: adds the aclgraph_dump module when compiling the .whl package, to save .pt files using `acl_save` in the ACLGraph scenario. `torch` and `torch_npu` are also required.<br>By default, this parameter is not set, indicating that the basic tool package is compiled.<br>If multiple modules are specified, separate them with commas (,), for example, **adump,atb_probe**.<br>When the adump or atb_probe module is specified, the compilation environment must have third-party dependencies such as Git, curl, GCC 7.5 or later, and CMake 3.19.3 or later. When the adump module is specified, the `libadump_server.a` file must be contained in the enabled CANN environment.<br>The .whl package generated by configuring this parameter is available only for the Python version and processor architecture used during compilation.|
| --no-check    |   Optional   | Skips certificate verification. After `--include-mod` specifies a module, the third-party library dependency is downloaded. During the download, certificate verification is performed. You can configure this parameter to skip certificate verification.|

**Examples**

- Compile and install the basic tool package.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- Compile and install the basic tool package and the adump module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=adump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- Compile and install the basic tool package and the aclgraph_dump module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- Compile and install the basic tool package and the hierarchical visualization plugin.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- Compile and install the basic tool package and the trend visualization plugin.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=trend_analyzer --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- Compile and install the basic tool package and the hierarchical visualization and trend visualization plugins.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend,trend_analyzer --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- Compile and install the basic tool package and the atb_probe module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=atb_probe --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

**Output Description**

If the following information is displayed, msProbe is successfully installed:

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

## Uninstallation

Run the following command to uninstall msProbe:

```bash
pip uninstall mindstudio-probe
```

If the following information is displayed, msProbe is successfully uninstalled:

```ColdFusion
Successfully uninstalled mindstudio-probe-{version}
```

## Upgrade

msProbe cannot be directly upgraded. You need to [uninstall](#uninstallation) msProbe and then [install](#installation-description) it.

## msProbe Information

```bash
pip show mindstudio-probe
```

Example:

```ColdFusion
Name: mindstudio-probe
Version: 26.0.x
Summary: Ascend MindStudio Probe Utils
Home-page: https://gitcode.com/Ascend/MindStudio-Probe
Author: Ascend Team
Author-email: pmail_mindstudio@xx.com
License: Mulan PSL v2
Location: /home/xxx/miniconda3/envs/xxx/lib/python3.x/site-packages/
Requires: einops, matplotlib, numpy, onnx, onnxruntime, openpyxl, pandas, protobuf, pyyaml, rich, setuptools, skl2onnx, tensorboard, tqdm, wheel
Required-by:
```

## Ascend Ecosystem

### Installing PyTorch_NPU

For details, see [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch).

### Installing MindSpeed LLM

For details, see [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM).
