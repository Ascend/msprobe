# msProbe PyTorch 场景最小示例

<br>

## 1. 概述

在 Linux CPU 上运行一个两层全连接网络的一次训练迭代，完成安装、埋点、dump、比对和结果解读。无需 CANN、`torch_npu`、GPU、模型权重或数据集下载。跨设备定位请参见《[msProbe PyTorch 场景快速入门](pytorch_quick_start.md)》。

## 2. 操作步骤

### 2.1 环境准备

本例固定使用 `mindstudio-probe==26.2.0`、PyTorch `2.7.1`，建议在 Python 3.11/3.12 的独立环境中运行。以下命令使用 Linux Bash：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install mindstudio-probe==26.2.0
git clone --depth 1 https://gitcode.com/Ascend/msprobe.git
cd msprobe/examples/quick_start/minimal
python -m pip show mindstudio-probe torch
msprobe --help
```

已检出仓库时，直接进入 `examples/quick_start/minimal`。软件包名是 `mindstudio-probe`，Python 导入名和命令均为 `msprobe`。NPU 场景的版本配套请参见《[msProbe 安装指南](../install_guide/msprobe_install_guide.md)》。

### 2.2 配置与埋点

[config.json](../../../examples/quick_start/minimal/config.json) 使用 `task=tensor` 保存完整 Tensor，`level=L1` 采集 API，`step=[0]` 只采集第一个迭代，`async_dump=false` 同步落盘。网络很小，适合直接体验真实 Tensor 比对。

[train.py](../../../examples/quick_start/minimal/train.py) 中的关键埋点为：

```python
debugger = PrecisionDebugger(config_path="config.json", dump_path="./dump")
debugger.start(model)
# 模型前向、loss.backward()、optimizer.step()
debugger.stop()
debugger.step()  # 每个训练迭代结束后调用一次
```

`train.py` 使用脚本旁的配置文件，命令行 `--dump-path` 覆盖配置中的输出目录。两次运行固定随机种子、模型初始化和输入；`--perturb` 在采集开始前给第二个 Linear 的权重加 `0.1`，人为制造差异。

### 2.3 运行并检查 dump

每次练习使用新目录，防止与旧 dump 混合：

```bash
RUN_DIR=$(mktemp -d "$PWD/run.XXXXXX")
python train.py --dump-path "$RUN_DIR/golden"
python train.py --dump-path "$RUN_DIR/target" --perturb
GOLDEN_JSON=$(find "$RUN_DIR/golden/step0" -name dump.json -print -quit)
TARGET_JSON=$(find "$RUN_DIR/target/step0" -name dump.json -print -quit)
test -n "$GOLDEN_JSON" && test -n "$TARGET_JSON"
find "$RUN_DIR" -name dump.json -o -name stack.json -o -name '*.pt'
```

单进程输出位于 `step0/proc{pid}/`：`dump.json` 记录 API 与 Tensor 元信息，`stack.json` 记录调用栈，`dump_tensor_data/` 保存 `.pt` 文件。两个进程的 PID 可以不同，不要手写 `proc{pid}`。多卡场景通常使用 `rank{id}` 目录。

### 2.4 执行精度比对

继续在同一终端执行：

```bash
msprobe compare -tp "$TARGET_JSON" -gp "$GOLDEN_JSON" -o "$RUN_DIR/compare"
find "$RUN_DIR/compare" -name '*.csv'
```

`-tp` 是待检查数据，`-gp` 是标杆数据。默认产生 `compare_result_{timestamp}.csv`。保留两侧完整的 `step0/proc{pid}` 目录，真实 Tensor 比对还需要其中的 `dump_tensor_data` 文件。

### 2.5 精度比对结果分析

1. 先看 `Err_message`，排除未匹配、形状或数据类型不一致、Tensor 文件缺失等问题。
2. 再看 `Result`、`Cosine` 和 `MaxAbsErr` 等指标。本例第二个 Linear 的激活输入应一致，其权重参数和前向输出开始出现差异，随后 loss 和反向数据也会受到影响。
3. 结合该行的 API 名称、`NPU_Stack_Info` 或目标侧 `stack.json` 回到 `train.py`，区分产生差异的位置与后续传播位置。CPU 数据仍沿用结果中的 `NPU Name` / `Bench Name` 列名，它们分别表示目标侧与标杆侧。

本例 `Functional.linear.1.forward.input.0` 的 `MaxAbsErr` 为 `0`，权重所在的 `input.1` 约为 `0.1`，`output.0` 约为 `0.115`。`Result` 是规则筛选结果，即使标为 `pass` 也可能存在非零误差，应结合这些数值判断。

可再运行一次不带 `--perturb` 的 target，并用新的输出目录比对，检查同模型、同输入的结果。这里的差异由示例主动注入，不能据此判断某个 CPU/NPU/GPU 算子有缺陷；统计摘要一致也不能替代 Tensor 逐元素比对。

指标定义与阈值请参见《[PyTorch 场景精度比对](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md#精度比对结果分析)》，其他单卡、多卡与 dump 模式请参见《[msProbe 示例](../../../examples/README.md)》。
