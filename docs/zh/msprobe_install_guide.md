# msProbe工具安装指南

## 环境和依赖

使用msProbe工具前，要求已存在可执行的用户AI应用，其中要求昇腾环境：

- 可正常运行用户AI应用，详细设备型号请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。
- 已安装配套版本的CANN Toolkit开发套件包和算子包并配置环境变量，详情请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)》。

## 安装说明

本文主要介绍msProbe工具的安装。当前支持**从PyPI安装、下载whl包安装和编译安装**三种方式。

推荐使用[miniconda](https://docs.anaconda.com/miniconda/)管理环境依赖。

```bash
conda create -n msprobe python=3.10
conda activate msprobe
```

## 工具限制与注意事项

- 工具读写的所有路径，如`config_path`、`dump_path`等，只允许包含大小写字母、数字、下划线、斜杠、点和短横线。

- 出于安全性及权限最小化角度考虑，本工具不应使用root等高权限账户，建议使用普通用户权限安装执行。

- 使用本工具前请确保执行用户的umask值大于等于0027，否则可能会导致工具生成的精度数据文件和目录权限过大。

- 用户须自行保证使用最小权限原则，如给工具输入的文件要求other用户不可写，在一些对安全要求更严格的功能场景下还需确保输入的文件group用户不可写。

- msProbe建议执行用户与安装用户保持一致，如果使用root执行，请自行关注root高权限触及的安全风险。

## 从PyPI安装

```bash
pip install mindstudio-probe --pre
```

**目前msProbe工具版本为预发布版本，请在命令行末尾添加`--pre`参数进行安装。**

打印如下信息时，表示msProbe安装成功。

`Successfully installed mindstudio-probe-{version}`

## 下载whl包安装

请参考[版本说明](./release_notes.md)中的“版本配套说明”章节，下载msProbe的whl软件包。

获取到whl软件包后执行如下命令进行安装。

```bash
sha256sum {name}.whl # 验证whl包，若校验码一致，则whl包在下载中没有受损
```

```bash
pip install ./mindstudio_probe-{version}-py3-none-any.whl # 安装whl包
```

打印如下信息时，表示msProbe安装成功。

`Successfully installed mindstudio-probe-{version}`

若覆盖安装，请在命令行末尾添加 `--force-reinstall` 参数。

上面提供的whl包链接不包含adump、aclgraph_dump和atb_probe功能，如果需要使用这些功能，请参考[编译安装](#编译安装)下载源码编译whl包。

## 编译安装

**功能说明**

通过setup.py脚本编译msProbe工具的whl软件包。

**命令格式**

```bash
python3 setup.py bdist_wheel [--include-mod=<include_mode>] [--no-check]
```

**参数说明**

| 参数          | 可选/必选 | 说明                                                         |
| ------------- | :-------: | ------------------------------------------------------------ |
| --include-mod |   可选    | 指定可选模块，可取值：<br/>&#8226; adump：表示在编译whl包时加入adump模块。adump模块用于MindSpore静态图场景L2级别的dump。仅MindSpore 2.5.0及以上版本支持adump模块。<br/>&#8226; tb_graph_ascend：表示在编译whl包时加入模型分级可视化插件。模型分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、Npm v10.8.2。模型分级可视化插件的详细依赖及功能使用说明请参见[PyTorch场景分级可视化构图比对](./accuracy_compare/pytorch_visualization_instruct.md)或[MindSpore场景分级可视化构图比对](./accuracy_compare/mindspore_visualization_instruct.md)。<br/>&#8226; trend_analyzer：表示在编译whl包时加入趋势分级可视化插件。趋势分级可视化插件的功能说明请参见[趋势可视化](./accuracy_compare/trend_visualization_instruct.md)。 <br/>&#8226; atb_probe：表示在编译whl包时加入atb_probe模块。atb_probe模块用于ATB推理场景下的数据采集。<br/>&#8226; aclgraph_dump：表示在编译whl包时加入aclgraph_dump模块，用于在aclgraph场景通过acl_save保存.pt文件。编译环境需要额外依赖`torch`和`torch_npu`。<br/>默认未配置该参数，表示编译基础工具包。<br/>指定多个模块时，模块间以","连接，例如adump,atb_probe。<br/>指定adump或atb_probe模块时，编译环境需具备git、curl、GCC 7.5或以上版本、CMake 3.19.3或以上版本等第三方依赖软件。且指定adump模块时，使能的CANN环境下需包含`libadump_server.a`文件。<br/>配置该参数生成的whl包，仅限编译时使用的Python版本和处理器架构可用。 |
| --no-check    |   可选    | 跳过证书校验。--include-mod指定可选模块后，会下载所依赖的第三方库包，下载过程会进行证书校验，配置本参数可以跳过证书校验。 |

**使用示例**

- 编译安装基础工具包

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 编译安装基础工具包和adump模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=adump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 编译安装基础工具包和aclgraph_dump模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=aclgraph_dump --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```
  
- 编译安装基础工具包和分级可视化插件

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 编译安装基础工具包和趋势可视化插件

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=trend_analyzer --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 编译安装基础工具包，同时编译安装分级可视化和趋势可视化插件

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=tb_graph_ascend,trend_analyzer --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

- 编译安装基础工具包和atb_probe模块

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe
  
  pip install setuptools wheel
  
  python3 setup.py bdist_wheel --include-mod=atb_probe --no-check
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

**输出说明**

打印如下信息时，表示msProbe安装成功。

```ColdFusion
Successfully installed mindstudio-probe-{version}
```

## 卸载

执行如下命令卸载msProbe工具。

```bash
pip uninstall mindstudio-probe
```

打印如下信息时，表示msProbe卸载成功。

```ColdFusion
Successfully uninstalled mindstudio-probe-{version}
```

## 升级

msProbe工具不支持直接升级，需要先完成[卸载](#卸载)后再重新[安装](#msprobe工具安装指南)。

## 查看msProbe工具信息

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

## Ascend生态链接

### 安装PyTorch_NPU

请参见[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)。

### 安装MindSpeed LLM

请参见[MindSpeed LLM](https://gitcode.com/Ascend/MindSpeed-LLM)。
