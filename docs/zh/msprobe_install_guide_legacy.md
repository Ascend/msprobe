# msProbe 源码安装指南（即将废弃）

> **日落公告**
>
> 本文档描述的基于 `setup.py` 的源码安装方式**即将在2026-09-30废弃**，不再推荐使用。
> 请使用新的 `build.py` 构建方式，参见 [msProbe 工具安装指南](./msprobe_install_guide.md)。

## 1. 源码安装

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
| --include-mod |   可选    | 指定可选模块，可取值：<br/>&#8226; adump：表示在编译whl包时加入adump模块。adump模块用于MindSpore静态图场景L2级别的dump。仅MindSpore 2.5.0及以上版本支持adump模块。<br/>&#8226; tb_graph_ascend：表示在编译whl包时加入模型分级可视化插件。模型分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、npm v10.8.2。模型分级可视化插件的详细依赖及功能使用说明请参见[PyTorch场景分级可视化构图比对](./accuracy_compare/pytorch_visualization_instruct.md)或[MindSpore场景分级可视化构图比对](./accuracy_compare/mindspore_visualization_instruct.md)。<br/>&#8226; trend_analyzer：表示在编译whl包时加入趋势分级可视化插件。趋势分级可视化构建相关依赖和推荐版本为Node.js v20.19.3、npm v10.8.2。趋势分级可视化插件的功能说明请参见[趋势可视化](./accuracy_compare/trend_visualization_instruct.md)。 <br/>&#8226; atb_probe：表示在编译whl包时加入atb_probe模块。atb_probe模块用于ATB推理场景下的数据采集。<br/>&#8226; aclgraph_dump：表示在编译whl包时加入aclgraph_dump模块，用于在aclgraph场景通过acl_save保存.pt文件。编译环境需要额外依赖`torch`和`torch_npu`。<br/>&#8226; nan_check：表示在编译whl包时加入nan_check模块，用于在nan_check场景下做寄存器溢出状态监测。<br/>&#8226; xor_checksum：表示在编译whl包时加入XOR校验加速算子，用于PyTorch场景下`summary_mode`配置为`xor`时加速校验值采集，可带来数倍性能提升。编译环境需要额外依赖`torch`和`torch_npu`。<br/>默认未配置该参数，表示编译基础工具包。<br/>指定多个模块时，模块间以","连接，例如adump,atb_probe。<br/>指定adump或atb_probe模块时，编译环境需具备git、curl、GCC 7.5或以上版本、CMake 3.19.3或以上版本等第三方依赖软件。且指定adump模块时，使能的CANN环境下需包含`libadump_server.a`文件。<br/>配置该参数生成的whl包，仅限编译时使用的Python版本和处理器架构可用。 |
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

<a id="install-xor-checksum"></a>

- 安装基础工具包和xor_checksum加速算子

  ```bash
  git clone https://gitcode.com/Ascend/msprobe.git
  cd msprobe

  pip install setuptools wheel

  python3 setup.py bdist_wheel --include-mod=xor_checksum
  cd ./dist
  pip install ./mindstudio_probe*.whl
  ```

**输出说明**

打印如下信息时，表示msProbe安装成功。

```ColdFusion
Successfully installed mindstudio-probe-{version}
```
