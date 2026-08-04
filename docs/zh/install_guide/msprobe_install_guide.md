# msProbe 安装指南

<br>

## 1. 安装说明

使用本工具前需要安装CANN，具体操作请参见《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装昇腾NPU驱动和CANN软件（包含Toolkit和ops包），并配置环境变量。

如需单独升级本工具或使用最新版本，您可通过以下三种方式进行安装：[在线安装](#21-在线安装)、[离线安装](#22-离线安装)、[源码安装](#23-源码安装)。

## 2. 安装方式

### 2.1 在线安装

```bash
pip install mindstudio-probe
```

打印如下信息时，表示msProbe安装成功。

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

> [!NOTE]
>
> 在线安装的安装包不包含aclgraph_dump、nan_check和xor_checksum功能。若需要使用这些功能，请参见[源码安装](#23-源码安装)。

### 2.2 离线安装

1. 请参见[msProbe Release](https://gitcode.com/Ascend/msprobe/releases)下载msProbe的whl软件包和对应数字签名文件（.sha256）。

   下载本软件即表示您同意《[华为企业业务最终用户许可协议（EULA）](https://e.huawei.com/cn/about/eula)》的条款和条件。

2. 验证whl包的完整性。

   1. 在whl包所在目录执行如下命令获取whl软件包的sha256校验码。

      ```bash
      sha256sum {name}.whl
      ```

      打印如下示例信息。

      ```ColdFusion
      {sha256} {name}.whl
      ```

   2. 用记事本打开数字签名文件查看sha256校验码。

   3. 比对两个文件的sha256校验码是否一致。

      若两个校验码一致，则表示下载了正确的软件包；若不一致，请不要使用该软件包，如需支持与服务，请在论坛求助或提交技术工单。

3. 安装whl包。

   ```bash
   pip install ./mindstudio_probe-{version}-py3-none-any.whl
   ```

   打印如下信息时，表示msProbe安装成功。

   ```ColdFusion
   Successfully installed mindstudio-probe-{version}
   ```

   若覆盖安装，请在命令行末尾添加 `--force-reinstall` 参数。

> [!NOTE]
>
> 离线安装的安装包不包含aclgraph_dump、nan_check和xor_checksum功能。若需要使用这些功能，请参见[源码安装](#23-源码安装)。

### 2.3 源码安装

如需使用最新代码的功能，或对源码进行修改以增强功能，可下载本仓库代码，自行编译、打包工具并完成安装。

#### 2.3.1 环境准备

源码编译统一使用 MindStudio 标准构建环境。

- 日常开发或使用已发布镜像，请参考《[MindStudio工具开发环境安装指导](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/dev_env_setup.md)》。
- 需要从基础操作系统复现环境、执行源码构建验证或单元测试验证时，必须参考《[MindStudio统一构建镜像制作指南](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)》，从openEuler基础镜像现场构建环境镜像。

本文档后续的源码编译和单元测试命令，均在上述指定镜像容器或现场构建的环境镜像的容器中执行，CANN 软件包版本、GCC 版本和 Python 版本以统一镜像制作指南为准，本仓库不重复维护。

镜像构建完成后，必须使用统一镜像制作指南第 7 章给出的 `ctr_in.py` 命令，在交互式终端中启动并进入容器。不得使用普通 `docker run` 创建容器，也不得使用 `docker exec <容器名> bash -c '<命令>'` 替代交互式环境；否则可能跳过 Python、GCC 和 CANN 环境初始化。

进入 `ctr_in.py` 打开的交互式容器 Shell 后，执行如下命令克隆本仓库：

```bash
cd ~
git clone https://gitcode.com/Ascend/msprobe.git
cd ~/msprobe
```

#### 2.3.2 执行编译

##### 2.3.2.1 编译命令功能介绍

通过 build.py 脚本编译 msProbe 工具的 whl 软件包。

**命令格式**

```bash
# 完整构建命令
python3 build.py [local] [-v <version>] [-e include-mod=<include_mode>] [-e no-check=true|false]
```

**参数说明**

| 参数              | 可选/必选 | 说明                                                         |
| ----------------- | :-------: | ------------------------------------------------------------ |
| local             |   可选    | 本地构建，即复用本地已有依赖，不主动下载第三方依赖 |
| -v / --version    |   可选    | 指定构建版本号，默认从pyproject.toml读取。                   |
| -e / --extra      |   可选    | 额外构建选项，KEY=VALUE格式，可多次指定。支持的KEY：<br/>&#8226; include-mod：指定可选模块，可取值：<br/>&emsp;- all：表示安装所有插件，环境必须满足下列各组件的依赖项。<br/>&emsp;- tb_graph_ascend：表示在编译whl包时加入模型分级可视化插件。模型分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、npm v10.8.2。模型分级可视化插件的详细依赖及功能使用说明请参见[PyTorch场景分级可视化构图比对](../user_guide/accuracy_compare/pytorch_visualization_instruct.md)或[MindSpore场景分级可视化构图比对](../user_guide/accuracy_compare/mindspore_visualization_instruct.md)。<br/>&emsp;- trend_analyzer：表示在编译whl包时加入趋势可视化插件。趋势可视化构建相关依赖和推荐版本为Node.js v20.19.3、npm v10.8.2。趋势可视化插件的功能说明请参见[趋势可视化](../user_guide/accuracy_compare/trend_visualization_instruct.md)。<br/>&emsp;- atb_probe：表示在编译whl包时加入atb_probe模块。atb_probe模块用于ATB推理场景下的数据采集。<br/>&emsp;- aclgraph_dump：表示在编译whl包时加入aclgraph_dump模块，用于在aclgraph场景通过acl_save保存.pt文件。编译环境需要额外依赖`torch`和`TorchNPU`。<br/>&emsp;- nan_check：表示在编译whl包时加入nan_check模块，用于在nan_check场景下做寄存器溢出状态监测。编译环境需要额外依赖`torch`和`TorchNPU`。<br/>&emsp;- xor_checksum：表示在编译whl包时加入XOR校验加速算子，用于PyTorch场景下`summary_mode`配置为`xor`时加速校验值采集，可带来数倍性能提升。编译环境需要额外依赖`torch`和`TorchNPU`。<br/>默认未配置该参数，表示编译基础工具包。<br/>指定多个模块时，模块间以","连接，例如tb_graph_ascend,trend_analyzer。<br/>指定atb_probe模块时，编译环境需具备git、curl、GCC 7.5或以上版本、CMake 3.19.3或以上版本等第三方依赖软件。<br/>配置该参数生成的whl包，仅限编译时使用的Python版本和处理器架构可用。<br/>&#8226; no-check：跳过证书校验，值为true或false。include-mod指定可选模块后，会下载所依赖的第三方库包，下载过程会进行证书校验，配置本参数可以跳过证书校验。 |

##### 2.3.2.2 编译命令示例

- 编译基础工具包

  ```bash
  python3 build.py
  ```

- 编译基础工具包（指定自定义版本）

  ```bash
  python3 build.py -v 26.0.0
  ```

- 编译基础工具包和aclgraph_dump模块

  ```bash
  python3 build.py -e include-mod=aclgraph_dump -e no-check=true
  ```

- 编译基础工具包和分级可视化插件

  ```bash
  python3 build.py -e include-mod=tb_graph_ascend -e no-check=true
  ```

- 编译基础工具包和趋势可视化插件

  ```bash
  python3 build.py -e include-mod=trend_analyzer -e no-check=true
  ```

- 编译基础工具包和分级可视化、趋势可视化插件

  ```bash
  python3 build.py -e include-mod=tb_graph_ascend,trend_analyzer -e no-check=true
  ```

- 编译基础工具包和atb_probe模块

  ```bash
  python3 build.py -e include-mod=atb_probe -e no-check=true
  ```

- 编译基础工具包和nan_check模块

  ```bash
  python3 build.py -e include-mod=nan_check -e no-check=true
  ```

- 编译基础工具包和xor_checksum加速算子

  ```bash
  python3 build.py -e include-mod=xor_checksum
  ```

- 安装带上所有功能的工具包

  ```bash
  python3 build.py -e include-mod=all
  ```

**输出说明**

构建成功后，安装包将生成在 `artifacts/` 目录下。

#### 2.3.3 执行单元测试（可选）

此步骤非安装必需。如需验证代码基本功能，可执行单元测试：

```bash
cd ~/msprobe
python3 build.py test
```

#### 2.3.4 安装

将生成在 `artifacts/` 目录下的安装包拷贝到运行环境中，执行以下命令安装：

```bash
pip install mindstudio_probe*.whl
```

打印如下信息时，表示msProbe安装成功：

```text
Successfully installed mindstudio-probe-{version}
```

## 3. 验证安装

安装完成后，执行以下命令验证工具是否安装成功：

```bash
pip show mindstudio-probe
```

若输出不报错，且能显示工具信息，则表明安装成功。

若 `pip show mindstudio-probe` 提示命令不存在，请确认当前终端使用的是安装 `msProbe` 的 Python 环境。

## 4. 卸载

执行如下命令卸载msProbe工具。

```bash
pip uninstall mindstudio-probe
```

打印如下信息时，表示msProbe卸载成功。

```text
Successfully uninstalled mindstudio-probe-{version}
```

## 5. 升级

msProbe工具不支持直接升级，需要先完成[卸载](#4-卸载)后再重新[安装](#2-安装方式)。

可通过`pip show mindstudio-probe`命令查看当前环境的版本信息，再选择需要升级的版本。升级版本时需要关注版本配套关系，请参见《[版本说明](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/26.1.0/release_notes.md)》。

## 6. 附录

### 6.1 工具限制与注意事项

- 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

- 出于安全性及权限最小化角度考虑，本工具不应使用root等高权限账户，建议使用普通用户权限安装执行。

- 使用本工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

- 用户须自行遵循最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

- msProbe建议执行用户与安装用户保持一致，如果使用root执行，请自行关注root高权限触及的安全风险。

### 6.2 查看msProbe工具信息

```bash
pip show mindstudio-probe
```

示例如下：

```ColdFusion
Name: mindstudio-probe
Version: 26.x.x
Summary: Ascend MindStudio Probe Utils
Home-page: https://gitcode.com/Ascend/MindStudio-Probe
Author: 
Author-email: Ascend Team <pmail_mindstudio@xx.com>
License-Expression: MulanPSL-2.0
Location: /xxx/xxx/miniconda3/envs/xxx/lib/python3.x/site-packages/
Requires: einops, matplotlib, numpy, openpyxl, pandas, psutil, pytz, pyyaml, skl2onnx, tensorboard, tqdm, wheel
Required-by: 
```

### 6.3 Ascend生态链接

#### 6.3.1 安装TorchNPU

请参见[Ascend for PyTorch](https://gitcode.com/Ascend/pytorch)。

#### 6.3.2 安装MindSpeed LLM

请参见[MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)。
