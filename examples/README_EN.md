# msProbe Examples

English | [简体中文](README.md)

Start with the minimal example, then choose an entry by device count and dump mode.

## PyTorch

| Scenario | Device Count | Dump Mode | Version Scope | Entry |
| --- | --- | --- | --- | --- |
| Minimal PyTorch training and precision comparison | Single-process CPU | `tensor` / `L1` | `mindstudio-probe==26.2.0`, PyTorch `2.7.1` | [Runnable example](../docs/en/quick_start/pytorch_minimal_quick_start.md) |
| ResNet-50 NPU/GPU collection, comparison, and visualization | Single device | `statistics` / `mix` | Current master tutorial; install the dependencies listed there | [Full tutorial](../docs/en/quick_start/pytorch_quick_start.md) |
| Pre-collected ResNet-50 GPU reference data | Single-process data, step0/step1 | `statistics` / `mix` | For the current master ResNet-50 tutorial, not a reference for arbitrary models or versions | [Data directory](quick_start/gpu_dump), [usage](../docs/en/quick_start/pytorch_quick_start.md) |

For multi-device integration, see the distributed training instrumentation and `rank` configuration in [PyTorch Data Collection](../docs/en/user_guide/dump/pytorch_data_dump_instruct.md), and network-wide multi-device comparison in [PyTorch Precision Comparison](../docs/en/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md). These are integration guides; this directory does not provide a standalone multi-device training script. Use documentation matching the installed version.

Before changing the tool version or framework, check [installation and version compatibility](../docs/en/install_guide/msprobe_install_guide.md). NPU environments also require matching CANN, drivers, and `torch_npu`. The CPU minimal example does not validate cross-device NPU/GPU precision.
