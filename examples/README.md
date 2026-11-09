# msProbe 示例

简体中文 | [English](README_EN.md)

首次使用建议从最小示例开始，再按设备数量和采集模式选择入口。

## PyTorch

| 场景 | 设备数量 | dump 模式 | 版本边界 | 入口 |
| --- | --- | --- | --- | --- |
| PyTorch 最小训练与精度比对 | 单进程 CPU | `tensor` / `L1` | `mindstudio-probe==26.2.0`，PyTorch `2.7.1` | [可运行示例](../docs/zh/quick_start/pytorch_minimal_quick_start.md) |
| ResNet-50 NPU/GPU 采集、比对与可视化 | 单卡 | `statistics` / `mix` | 当前 master 教程；依赖版本按教程安装 | [完整教程](../docs/zh/quick_start/pytorch_quick_start.md) |
| ResNet-50 预置 GPU 标杆数据 | 单进程数据，step0/step1 | `statistics` / `mix` | 配套当前 master 的 ResNet-50 教程，不作为任意模型或版本的标杆 | [数据目录](quick_start/gpu_dump)，[使用方法](../docs/zh/quick_start/pytorch_quick_start.md#23-在-gpu-环境采集标杆数据) |

多卡接入请参考 [PyTorch 数据采集](../docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)中的分布式训练埋点与 `rank` 配置，以及 [PyTorch 精度比对](../docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)中的整网多卡比对。这些是接入文档，当前目录未提供独立的多卡训练脚本；使用时应选择与安装版本一致的文档。

更换版本或框架前，请核对 [安装与版本配套](../docs/zh/install_guide/msprobe_install_guide.md)。NPU 环境还需配套的 CANN、驱动和 `torch_npu`；CPU 最小示例不验证 NPU/GPU 跨设备精度。
