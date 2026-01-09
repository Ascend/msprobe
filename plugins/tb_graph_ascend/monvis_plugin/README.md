# TensorBoard 训练监控可视化插件

一个用于可视化模型监控指标的 TensorBoard 插件，支持 step、rank 和 moduleName 指标的热力图和趋势图分析，插件支持将特定的数据库脚本生成可视化交互式界面。

## 核心功能

- **交互式热力图**：支持跨维度（训练步数、rank、模块/参数）指标统计量可视化
- **趋势分析**：查看选定维度的指标变化趋势

## 数据库结构

| 表名                 | 说明                 |
| -------------------- | -------------------- |
| `monitoring_targets` | 被监控模块/层信息    |
| `monitoring_metrics` | 可用指标列表         |
| `metric_stats`       | 指标的统计量类型     |
| `global_stats`       | 全局步数/rank 范围   |
| `metric_*_step_*`    | 分片存储的指标数据表 |

## 项目文件

```bash
hierarchy_plugin/ # 模型分级可视化插件
├── fe/ # 前端资源
├── server # 插件后端核心
monvis_plugin # 训练监控可视化插件
├── fe/ # 前端资源
├── server # 插件后端核心
setup.py # 安装配置

```

## 安装指南

1. 克隆项目仓库
   git clone https://gitee.com/ascend/mstt.git -b poc

2. 进入目录 `plugins/tensorboard-plugins/tb_graph_ascend` 下

```bash
// 进入前端目录
cd plugins/tensorboard-plugins/tb_graph_ascend/monvis_plugin/fe
// 安装前端依赖
npm install --force
// Windows系统
npm run buildWin
// Linux系统
npm run buildLinux
// 进入根目录
cd plugins/tensorboard-plugins/tb_graph_ascend
// 本地安装
pip install -e .
```

3. 插件安装后会被 TensorBoard 自动识别

## TensorBoard 使用

1. 启动服务：

```bash
tensorboard --logdir=./db --port=6008
```

访问路径：

浏览器打开 http://localhost:6008

选择顶部导航栏的 MonVis 标签页
