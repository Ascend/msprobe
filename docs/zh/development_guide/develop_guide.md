# 开发者指南

本文面向 MindStudio Probe 的开发和维护人员，介绍源码目录、构建方式、功能开发流程、功能改动后的验证方法，以及资料联动更新要求。本文重点结合 MindStudio Probe 当前仓库和现有文档内容编写，适用于新增命令参数、扩展工具能力、增加交付件或维护软件包安装方式等场景。

## 1. MindStudio Probe 开发概述

MindStudio Probe 提供 AI 任务运行精度数据的采集、预检与比对等能力。围绕开发工作，通常可以分为以下几类：

| 开发对象 | 典型内容                                                     |
| -------- | ------------------------------------------------------------ |
| 采集能力 | 包括训练和推理模块的精度数据采集能力                         |
| 预检能力 | `msprobe acc_check`、`msprobe multi_acc_check`               |
| 比对能力 | `msprobe compare、msprobe compare -m atb、offline_model`、`msprobe graph_visualize`等 |
| 溢出检测 | `msprobe overflow_check`                                     |
| 扩展功能 | 训练前配置检查、训练状态监测、 checkpoint比对、趋势可视化等  |
| 文档资料 | 安装指南、快速入门、功能说明、数据文件参考、扩展功能         |

## 2. 代码目录

根据当前仓库资料，msProbe 项目主要目录如下：

| 目录                            | 说明                     |
| ------------------------------- | ------------------------ |
| `ccsrc`                         | C/C++源码目录            |
| `cmake`                         | 存放解析C化部分cmake文件 |
| `docs`                          | 文档目录                 |
| `examples`                      | 工具配置样例存放目录     |
| `output`                        | 交付件生成目录           |
| `plugins`                       | 插件类代码总入口         |
| `python/msprobe/core`           | 工具核心功能模块         |
| `python/msprobe/infer`          | 推理工具模块             |
| `python/msprobe/mindspore`      | MindSpore工具模块        |
| `python/msprobe/msaccucmp`      | msaccucmp工具模块        |
| `python/msprobe/overflow_check` | 溢出检测模块             |
| `python/msprobe/pytorch`        | PyTorch工具模块          |
| `python/msprobe/visualization`  | 可视化模块               |
| `scripts`                       | 存放安装卸载升级脚本     |
| `test`                          | 测试代码目录             |
| `docs/zh`                       | 中文文档                 |

## 3. 开发环境配置

### 3.1 基础软件

| 软件名                   | 版本要求       | 用途                   |
| ------------------------ | -------------- | ---------------------- |
| PyCharm（推荐）/ VS Code | 无硬性要求     | 编写和调试 Python 代码 |
| Python                   | 3.8 及以上     | 主开发环境             |
| pip                      | 与 Python 配套 | 安装依赖和本地包       |
| conda                    | 无硬性要求     | 隔离开发依赖           |
| wheel                    | 最新稳定版     | 构建 whl 包            |
| Git                      | 无硬性要求     | 拉取、管理和提交代码   |

### 3.2 开发依赖

基础依赖定义在 `docs/requirements.txt`。

其中核心运行依赖包括：

- einops
- matplotlib
- numpy
- openpyxl
- pandas
- pyyaml
- rich
- skl2onnx
- tensorboard
- tqdm
- wheel

### 3.3 推荐环境准备

建议在仓库根目录下使用虚拟环境进行开发：

```bash
conda create -n msprobe python=3.10
conda activate msprobe
```

## 4. 获取代码与构建

### 4.1 获取代码

```bash
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
```

### 4.2 编译安装基础工具包

```bash
pip install uv

python3 build.py
cd ./artifacts
pip install ./mindstudio_probe*.whl
```

编译工具包时还可以选择编译的功能模块，通过--include-mod参数配置，详见《[msProbe工具安装指南](../install_guide/msprobe_install_guide.md)》

安装完成后，建议立即校验：

```bash
which msprobe
msprobe --help
```

## 5. 测试与验证

仓库提供了统一的单元测试入口：

```bash
cd test/msprobe_test
bash run_test.sh
```

- 测试数据应该放在`test/`目录下的相应位置。
- 运行测试后，代码覆盖率报告生成在./report目录下。

## 6. 文档联动更新

功能开发完成后，若改动影响用户使用方式或输出结果，需要同步更新文档。

| 改动类型 | 需同步更新的文档 |
| --- | --- |
| 安装、编译、升级方式 | `docs/zh/install_guide/msprobe_install_guide.md` |
| 快速入门 | `docs/zh/quick_start` |
| 功能文档 | `docs/zh/user_guide/dump` |
| 性能基线文档 | `docs/zh/baseline` |
| 案例文档 | `docs/zh/best_practices` |
| FAQ | `docs/zh/support` |

若新增文档、截图或示意图：

1. 图片统一放在 `docs/zh/figures`。
2. 文件名应与功能语义对应。
3. 正文中的图标题、路径、说明文字要同步更新。

## 7. 提交流程建议

1. 在功能开发完成后，先执行本地安装验证。
2. 至少完成一轮 `UT`，必要时补充 `ST`。
3. 若涉及用户可见行为变化，同步补充文档和示例命令。
4. 若新增分析能力，说明其输入数据要求、输出文件和适用场景。
