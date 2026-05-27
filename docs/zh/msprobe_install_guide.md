# msProbe工具安装指南

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

      若两个校验码一致，则表示下载了正确的软件包；若不一致，请不要使用该软件包，需要支持与服务请在论坛求助或提交技术工单。

3. 安装whl包。

   ```bash
   pip install ./mindstudio_probe-{version}-py3-none-any.whl
   ```

   打印如下信息时，表示msProbe安装成功。

   ```ColdFusion
   Successfully installed mindstudio-probe-{version}
   ```

   若覆盖安装，请在命令行末尾添加 `--force-reinstall` 参数。

   以上提供的whl包链接不包含adump、aclgraph_dump、atb_probe和nan_check等功能，如果需要使用这些功能，请参见[源码安装](#23-源码安装)下载源码编译whl包。

### 2.3 源码安装

**功能说明**

通过setup.py脚本编译msProbe工具的whl软件包。

**命令格式**

```bash
# 方式1：使用默认版本编译
python3 setup.py bdist_wheel [--include-mod=<include_mode>] [--no-check]

# 方式2：指定自定义版本编译（前置设置环境变量WHL_VERSION）
WHL_VERSION=自定义版本号 python3 setup.py bdist_wheel [--include-mod=<include_mode>] [--no-check]
```

**参数说明**

| 参数          | 可选/必选 | 说明                                                         |
| ------------- | :-------: | ------------------------------------------------------------ |
| --include-mod |   可选    | 指定可选模块，可取值：<br/>&#8226; adump：表示在编译whl包时加入adump模块。adump模块用于MindSpore静态图场景L2级别的dump。仅MindSpore 2.5.0及以上版本支持adump模块。<br/>&#8226; tb_graph_ascend：表示在编译whl包时加入模型分级可视化插件。模型分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、npm v10.8.2。模型分级可视化插件的详细依赖及功能使用说明请参见[PyTorch场景分级可视化构图比对](./accuracy_compare/pytorch_visualization_instruct.md)或[MindSpore场景分级可视化构图比对](./accuracy_compare/mindspore_visualization_instruct.md)。<br/>&#8226; trend_analyzer：表示在编译whl包时加入趋势分级可视化插件。趋势分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、npm v10.8.2。趋势分级可视化插件的功能说明请参见[趋势可视化](./accuracy_compare/trend_visualization_instruct.md)。 <br/>&#8226; atb_probe：表示在编译whl包时加入atb_probe模块。atb_probe模块用于ATB推理场景下的数据采集。<br/>&#8226; aclgraph_dump：表示在编译whl包时加入aclgraph_dump模块，用于在aclgraph场景通过acl_save保存.pt文件。编译环境需要额外依赖`torch`和`torch_npu`。<br/>&#8226; nan_check：表示在编译whl包时加入nan_check模块，用于在nan_check场景下做寄存器溢出状态监测。<br/>默认未配置该参数，表示编译基础工具包。<br/>指定多个模块时，模块间以","连接，例如adump,atb_probe。<br/>指定adump或atb_probe模块时，编译环境需具备git、curl、GCC 7.5或以上版本、CMake 3.19.3或以上版本等第三方依赖软件。且指定adump模块时，使能的CANN环境下需包含`libadump_server.a`文件。<br/>配置该参数生成的whl包，仅限编译时使用的Python版本和处理器架构可用。 |
| --no-check    |   可选    | 跳过证书校验。--include-mod指定可选模块后，会下载所依赖的第三方库包，下载过程会进行证书校验，配置本参数可以跳过证书校验。 |

**使用示例**

- 安装基础工具包

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 安装基础工具包（指定自定义版本）

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  # 设置环境变量WHL_VERSION=2.1.0，自定义包版本为2.1.0
  WHL_VERSION=2.1.0 python3 setup.py bdist_wheel
  cd ./dist
  pip install ./mindstudio_probe-2.1.0*.whl
  ```
  
- 安装基础工具包和adump模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=adump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 安装基础工具包和aclgraph_dump模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 安装基础工具包和分级可视化插件

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 安装基础工具包和趋势可视化插件

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=trend_analyzer --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 安装基础工具包和分级可视化、趋势可视化插件

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend,trend_analyzer --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 安装基础工具包和atb_probe模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=atb_probe --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 安装基础工具包和nan_check模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=nan_check --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

**输出说明**

打印如下信息时，表示msProbe安装成功。

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

## 3. 卸载

执行如下命令卸载msProbe工具。

```bash
pip uninstall mindstudio-probe
```

打印如下信息时，表示msProbe卸载成功。

```ColdFusion
Successfully uninstalled mindstudio-probe-{version}
```

## 4. 升级

msProbe工具不支持直接升级，需要先完成[卸载](#3-卸载)后再重新[安装](#2-安装方式)。

## 5. 附录

### 5.1 工具限制与注意事项

- 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

- 出于安全性及权限最小化角度考虑，本工具不应使用root等高权限账户，建议使用普通用户权限安装执行。

- 使用本工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

- 用户须自行遵循最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

- msProbe建议执行用户与安装用户保持一致，如果使用root执行，请自行关注root高权限触及的安全风险。

### 5.2 查看msProbe工具信息

```bash
pip show mindstudio-probe
```

示例如下：

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

### 5.3 Ascend生态链接

#### 5.3.1 安装PyTorch_NPU

请参见[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)。

#### 5.3.2 安装MindSpeed LLM

请参见[MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)。
