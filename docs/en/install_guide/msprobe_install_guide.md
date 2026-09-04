# msProbe Installation Guide

## 1. Installation Description

Before using this tool, you need to install CANN. For details, see [CANN Quick Installation](https://www.hiascend.com/en/cann/download) to install the Ascend NPU driver and CANN software (including the Toolkit and ops), and configure environment variables.

If you need to upgrade this tool separately or use the latest version, you can install it in any of the following ways: [Online Installation](#21-online-installation), [Offline Installation](#22-offline-installation), and [Installation from Source](#23-installation-from-source).

## 2. Installation Methods

### 2.1 Online Installation

```bash
pip install mindstudio-probe==26.1.0.post1
```

If the following information is displayed, msProbe is successfully installed:

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

> [!NOTE]
>
> The package for online installation does not provide aclgraph_dump, nan_check, and xor_checksum functions. To use these functions, see [Installation from Source](#23-installation-from-source).

### 2.2 Offline Installation

1. Download the msProbe `.whl` package and the corresponding digital signature file (`.sha256`) by referring to [msProbe Release](https://gitcode.com/Ascend/msprobe/releases).

   Once you download this software, you agree to the terms and conditions of the [Huawei Enterprise End User License Agreement (EULA)](https://e.huawei.com/en/about/eula).

2. Verify the integrity of the `.whl` package.

   1. Run the following command in the directory where the `.whl` package is located to obtain the SHA256 checksum of the package.

      ```bash
      sha256sum {name}.whl
      ```

      The following information is displayed:

      ```ColdFusion
      {sha256} {name}.whl
      ```

   2. Open the digital signature file using Notepad to view the SHA256 checksum.

   3. Check whether the SHA256 checksums of the two files are the same.

      If they are the same, the downloaded software package is correct. If they are different, do not use the software package. For support and services, seek help in the forum or submit a technical service ticket.

3. Install the `.whl` package.

   ```bash
   pip install ./mindstudio_probe-{version}-py3-none-any.whl
   ```

   If the following information is displayed, msProbe is successfully installed:

   ```ColdFusion
   Successfully installed mindstudio-probe-26.1.0.post1
   ```

   To overwrite the existing installation, add `--force-reinstall` to the end of the command.

> [!NOTE]
>
> The package for offline installation does not provide aclgraph_dump, nan_check, and xor_checksum functions. To use these functions, see [Installation from Source](#23-installation-from-source).

### 2.3 Installation from Source

**Description**

The `build.py` script can be used to compile the `.whl` package of msProbe.

**Syntax**

```bash
# Build command
python3 build.py [local] [-v <version>] [-e include-mod=<include_mode>] [-e no-check=true|false]
```

**Parameters**

| Parameter         | Mandatory/Optional| Description                                                        |
| ------------- | :-------: | ------------------------------------------------------------ |
| local             |   Optional    | Local build, i.e., reuse existing local dependencies without actively downloading third-party dependencies. |
| -v / --version    |   Optional    | Specifies the build version number; defaults to reading from `pyproject.toml`. |
| -e / --extra      |   Optional    |Extra build options in `KEY=VALUE` format, can be specified multiple times. Supported keys: <br/>&#8226; `include-mod`: Specifies optional modules. Possible values:<br/>&emsp;- `tb_graph_ascend`: Includes the model hierarchy visualization plugin when building the whl package. Depends on Node.js v20.19.3 and npm v10.8.2. For details, see [Graph Comparison in Hierarchical Visualization (PyTorch)](../user_guide/accuracy_compare/pytorch_visualization_instruct.md) or [Graph Comparison in Hierarchical Visualization (MindSpore)](../user_guide/accuracy_compare/mindspore_visualization_instruct.md). <br/>&emsp;- `trend_analyzer`: Includes the trend visualization plugin when building the whl package. Depends on Node.js v20.19.3 and npm v10.8.2. For details, see [Trend Visualization](../user_guide/accuracy_compare/trend_visualization_instruct.md).<br/>&emsp;- `atb_probe`: Includes the `atb_probe` module when building the whl package, used for data collection in ATB inference scenarios. The build environment requires git, curl, GCC 7.5 or higher, and CMake 3.19.3 or higher.<br/>&emsp;- `aclgraph_dump`: Includes the `aclgraph_dump` module when building the whl package, used to save `.pt` files via `acl_save` in aclgraph scenarios. Requires additional dependencies `torch` and `TorchNPU`. <br/>&emsp;- `nan_check`: Includes the `nan_check` module when building the whl package, used for register overflow status monitoring. Requires additional dependencies `torch` and `TorchNPU`. <br/>&emsp;- `xor_checksum`: Includes the XOR checksum acceleration operator when building the whl package, used to accelerate checksum value collection in PyTorch scenarios with `summary_mode=xor`, delivering several-fold performance improvement. Requires additional dependencies `torch` and `TorchNPU`. <br/>By default, this parameter is not set, which means the base tool package is built.<br/>When specifying multiple modules, separate them with commas, e.g., `tb_graph_ascend,trend_analyzer`.<br/>When `atb_probe` is specified, the build environment requires git, curl, GCC 7.5 or higher, and CMake 3.19.3 or higher.<br/>The whl package generated with this parameter is only available for the Python version and processor architecture used at build time. <br/>&#8226; `no-check`: Skips certificate verification. The value is `true` or `false`. When `include-mod` specifies optional modules, third-party dependencies are downloaded, and certificate verification is performed during the download. Setting this parameter skips the certificate verification. |

**Example**

- Install the basic tool package.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```
  
- Install the basic tool package with the specific version.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -v 26.1.0
  cd ./artifacts
  pip install ./mindstudio_probe-26.1.0*.whl
  ```
  
- Install the basic tool package and the aclgraph_dump module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=aclgraph_dump -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```
  
- Install the basic tool package and the hierarchical visualization plugin.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=tb_graph_ascend -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the trend visualization plugin.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=trend_analyzer -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the hierarchical visualization and trend visualization plugins.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=tb_graph_ascend,trend_analyzer -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the atb_probe module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=atb_probe -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

- Install the basic tool package and the nan_check module.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=nan_check -e no-check=true
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

<a id="install-xor-checksum"></a>

- Install the basic tool package and the xor_checksum acceleration operator.

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git -b 26.1.0
  cd msprobe

  pip install uv

  python3 build.py -e include-mod=xor_checksum
  cd ./artifacts
  pip install ./mindstudio_probe*.whl
  ```

**Output Description**

If the following information is displayed, msProbe is successfully installed:

```ColdFusion
Successfully installed mindstudio-probe-26.1.0
```

## 3. Installation Verification

After the installation, check whether the tool has been successfully installed:

```bash
pip show mindstudio-probe
```

If no error is output and the tool information is displayed, the tool has been successfully installed.

If `pip show mindstudio-probe` prompts that it does not exist, check whether the Python environment with `msProbe` installed is used.

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

msProbe cannot be directly upgraded. You need to [uninstall](#4-uninstallation) msProbe and then [install](#2-installation-methods) it again.

You can use the `pip show mindstudio-probe` command to view the version information of the current environment, and then select the version to upgrade to. When upgrading the version, you need to pay attention to the version compatibility relationship. Please refer to the [Release Notes](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes_en.md).

## 6. Appendix

### 6.1 Constraints and Precautions

- All paths read and written by the tool, such as `config_path` and `dump_path`, can contain only letters, digits, underscores (_), slashes (/), periods (.), and hyphens (-).

- To ensure security and adhere to the principle of least privilege, you are advised to install and run this tool as a standard user rather than a high-privilege user (such as `root`).

- Ensure the execution user's `umask` value is set to `0027` or higher to prevent excessive permissions for generated accuracy data files and directories.

- You must follow the principle of least privilege. For example, the files fed to the tool must not be writable by `other` users. In scenarios with stricter security requirements, you must also ensure that the input files are not writable by group users.

- It is recommended that the msProbe execution user be the same as the installation user. If `root` is used for execution, pay attention to the security risks caused by the high permissions of `root`.

### 6.2 Viewing msProbe Information

```bash
pip show mindstudio-probe
```

Example:

```ColdFusion
Name: mindstudio-probe
Version: 26.1.0.post1
Summary: Ascend MindStudio Probe Utils
Home-page: https://gitcode.com/Ascend/MindStudio-Probe
Author: 
Author-email: Ascend Team <pmail_mindstudio@xx.com>
License-Expression: MulanPSL-2.0
Location: /xxx/xxx/miniconda3/envs/xxx/lib/python3.x/site-packages
Requires: einops, matplotlib, numpy, openpyxl, pandas, psutil, pytz, pyyaml, rich, skl2onnx, tensorboard, tqdm, wheel
Required-by: 
```

### 6.3 Ascend Ecosystem

#### 6.3.1 Installing TorchNPU

For details, see [Ascend for PyTorch](https://gitcode.com/Ascend/pytorch).

#### 6.3.2 Installing MindSpeed LLM

For details, see [MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM).
