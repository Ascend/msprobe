# msProbe Installation Guide

## 1. Installation Description

Before using this tool, install CANN. For details, see [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the Ascend NPU driver and CANN software (including the Toolkit and operator package), and configure environment variables.

To separately upgrade this tool or use the latest version, you can install it in any of the following modes: [Online Installation](#21-online-installation), [Offline Installation](#22-offline-installation), or [Source Installation](#23-source-installation).

## 2. Installation Modes

### 2.1 Online Installation

```bash
pip install mindstudio-probe
```

If the following information is displayed, msProbe is successfully installed:

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

### 2.2 Offline Installation

1. Download the msProbe .whl package and the corresponding digital signature file (.sha256) by referring to [msProbe Release](https://gitcode.com/Ascend/msprobe/releases).

   By downloading this software, you agree to the terms and conditions of the [Huawei Enterprise Business End User License Agreement (EULA)](https://e.huawei.com/cn/about/eula).

2. Verify the integrity of the .whl package.

   1. Run the following command in the directory where the .whl package is located to obtain the sha256 verification code of the .whl package:

      ```bash
      sha256sum {name}.whl
      ```

      The following example information is displayed:

      ```ColdFusion
      {sha256} {name}.whl
      ```

   2. Open the digital signature file with a text editor to view the sha256 verification code.

   3. Compare the sha256 verification codes of the two files.

      If the two verification codes are the same, the correct software package is downloaded. If they are different, do not use the software package. If support and service are required, seek help in the forum or submit a technical service request.

3. Install the .whl package.

   ```bash
   pip install ./mindstudio_probe-{version}-py3-none-any.whl
   ```

   If the following information is displayed, msProbe is successfully installed:

   ```ColdFusion
   Successfully installed mindstudio-probe-{version}
   ```

   To overwrite the existing installation, add `--force-reinstall` to the end of the command.

   The preceding .whl package does not contain functions such as aclgraph_dump, atb_probe, and nan_check. To use these functions, download the source code and compile the .whl package by referring to [Source Installation](#23-source-installation).

### 2.3 Source Installation

**Prerequisites**

It is recommended to pull the Docker compilation image before source installation to ensure compilation environment consistency.

1. Pull the Docker image.

   ```bash
   docker pull swr.cn-north-4.myhuaweicloud.com/mindstudio-image/mindstudio-build:26.1.0-20260610
   ```

2. Start the container.

   ```bash
   docker run -it --name msprobe-compile \
   --network host \
   swr.cn-north-4.myhuaweicloud.com/mindstudio-image/mindstudio-build:26.1.0-20260610 \
   /bin/bash
   ```

**Function Description**

The **build.py** script can be used to compile the .whl software package of msProbe.

**Syntax**

```bash
# Full build command
python3 build.py [local] [-v <version>] [-e include-mod=<include_mode>] [-e no-check=true|false]
```

**Parameters**

| Parameter         | Required/Optional| Description                                                        |
| ------------- | :-------: | ------------------------------------------------------------ |
| local |   Optional    | Local build, that is, reusing existing local dependencies without proactively downloading third-party dependencies. |
| -v / --version    |   Optional    | Specifies the build version number. The default value is read from pyproject.toml.                   |
| -e / --extra      |   Optional    | Extra build options in KEY=VALUE format, which can be specified multiple times. Supported KEYs:<br>&#8226; include-mod: specifies optional modules. Available values:<br>&emsp;- tb_graph_ascend: adds the hierarchical model visualization plugin when compiling the .whl package. The dependencies and recommended versions for hierarchical model visualization are Node.js v20.19.3 and npm v10.8.2. For details about the dependencies and functions of the plugin, see [Graph Comparison in Hierarchical Visualization in PyTorch](./accuracy_compare/pytorch_visualization_instruct.md) or [Graph Comparison in Hierarchical Visualization in MindSpore](./accuracy_compare/mindspore_visualization_instruct.md).<br>&emsp;- trend_analyzer: adds the trend visualization plugin when compiling the .whl package. The dependencies and recommended versions for trend visualization are Node.js v20.19.3 and npm v10.8.2. For details about the trend visualization plugin, see [Trend Visualization](./accuracy_compare/trend_visualization_instruct.md).<br>&emsp;- atb_probe: adds the atb_probe module when compiling the .whl package. The atb_probe module is used to collect data in the ATB inference scenario.<br>&emsp;- aclgraph_dump: adds the aclgraph_dump module when compiling the .whl package, to save .pt files using `acl_save` in the ACLGraph scenario. `torch` and `torch_npu` are also required for the compilation environment.<br>&emsp;- nan_check: adds the nan_check module when compiling the .whl package, used for register overflow status monitoring in the nan_check scenario.<br>&emsp;- xor_checksum: adds the XOR checksum acceleration operator when compiling the .whl package, used to accelerate checksum collection when `summary_mode` is set to `xor` in the PyTorch scenario, bringing multiple times performance improvement. `torch` and `torch_npu` are also required for the compilation environment.<br>By default, this parameter is not set, indicating that the basic tool package is compiled.<br>If multiple modules are specified, separate them with commas (,), for example, tb_graph_ascend,trend_analyzer.<br>When the atb_probe module is specified, the compilation environment must have third-party dependencies such as Git, curl, GCC 7.5 or later, and CMake 3.19.3 or later.<br>The .whl package generated by configuring this parameter is available only for the Python version and processor architecture used during compilation.<br>&#8226; no-check: skips certificate verification. The value is true or false. After `include-mod` specifies a module, the third-party library dependency is downloaded. During the download, certificate verification is performed. You can configure this parameter to skip certificate verification. |

**Examples**

- Install the basic tool package.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package (with a specified custom version).

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -v 26.0.0
  cd ./artifacts
  pip install ./mindstudio_probe-26.0.0*.whl
  ```

- Install the basic tool package and the aclgraph_dump module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=aclgraph_dump -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the hierarchical visualization plugin.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=tb_graph_ascend -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the trend visualization plugin.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=trend_analyzer -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the hierarchical visualization and trend visualization plugins.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=tb_graph_ascend,trend_analyzer -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the atb_probe module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=atb_probe -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the nan_check module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=nan_check -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

<a id="install-xor-checksum"></a>

- Install the basic tool package and the xor_checksum acceleration operator.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=xor_checksum
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

**Output Description**

If the following information is displayed, msProbe is successfully installed:

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

## 3. Verifying the Installation

After the installation is complete, run the following command to verify that the tool is successfully installed:

```bash
pip show mindstudio-probe
```

If no error is reported and the tool information is displayed, the installation is successful.

If `pip show mindstudio-probe` prompts that the command does not exist, check whether the current terminal uses the Python environment in which `msProbe` is installed.

## 4. Uninstallation

Run the following command to uninstall msProbe:

```bash
pip uninstall mindstudio-probe
```

If the following information is displayed, msProbe is successfully uninstalled:

```ColdFusion
Successfully uninstalled mindstudio-probe-{version}
```

## 5. Upgrade

msProbe cannot be directly upgraded. You need to [uninstall](#4-uninstallation) it and then [install](#2-installation-modes) it again.

You can run `pip show mindstudio-probe` to view the version information of the current environment, and then select the version to upgrade. Pay attention to the version matching relationship during upgrade. For details, see [Release Notes](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.0.0/release_notes.md).

## 6. Appendix

### 6.1 Tool Constraints and Precautions

- All paths read and written by the tool, such as `config_path` and `dump_path`, can contain only letters, digits, underscores (_), slashes (/), periods (.), and hyphens (-).

- To ensure security and adhere to the principle of least privilege, you are advised to install and run this tool as a standard user rather than a high-privilege user (such as `root`).

- Ensure the execution user's **umask** value is set to **0027** or higher to prevent excessive permissions for generated accuracy data files and directories.

- Users must strictly adhere to the principle of least privilege; for example, input files must not be writable by "others." In some function scenarios with stricter security requirements, ensure that the input files are not writable by group users.

- It is recommended that the msProbe execution user be the same as the installation user. If `root` is used for execution, pay attention to the security risks caused by the high permissions of `root`.

### 6.2 Viewing msProbe Information

```bash
pip show mindstudio-probe
```

Example:

```ColdFusion
Name: mindstudio-probe
Version: 26.x.x
Summary: Ascend MindStudio Probe Utils
Home-page: https://gitcode.com/Ascend/MindStudio-Probe
Author: 
Author-email: Ascend Team <pmail_mindstudio@xx.com>
License-Expression: MulanPSL-2.0
Location: /home/xxx/miniconda3/envs/xxx/lib/python3.x/site-packages/
Requires: einops, matplotlib, numpy, openpyxl, pandas, psutil, pytz, pyyaml, rich, skl2onnx, tensorboard, tqdm, wheel
Required-by: 
```

### 6.3 Ascend Ecosystem Links

#### 6.3.1 Installing PyTorch_NPU

For details, see [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch).

#### 6.3.2 Installing MindSpeed LLM

For details, see [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM).
