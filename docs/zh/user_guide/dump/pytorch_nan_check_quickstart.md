# nan_check 使用示例

本例采集一次前向、反向计算：

- 负数开平方使 loss 变为 NaN，再从落盘文件找到产生 NaN 的 API 和输入值。
- 使用单卡 PyTorch NPU、默认 `isnan` 模式。

先在`config.json`中配置采集范围，再用`PrecisionDebugger.start()`和`stop()`包住要检查的代码。运行后，查看`dump.json`中的NaN检测结果；需要看具体的输入、输出数据时，读取`.pt`文件；需要找调用位置时，查看`stack.json`。

## 1. 构建

本例需要`aclgraph_dump`模块，编译时添加`-e include-mod=aclgraph_dump`。环境准备、编译和安装步骤请参见[安装指南](../../install_guide/msprobe_install_guide.md)。

## 2. 运行采集

在本地克隆的`msprobe`目录中创建`config.json`和下方的`repro_nan.py`，并在该目录运行。默认已是`isnan`，无需额外配置检测模式或`tensor_list`：

```json
{
    "task": "nan_check",
    "dump_path": "./nan_check_dump",
    "level": "L1"
}
```

需要指定采集卡、step或API范围时，可在[完整配置示例及参数说明](./config_json_introduct.md#task配置为nan_check)中查看`rank`、`step`、`scope`、`list`、`data_mode`等字段；本例沿用默认值。

`nan_check.detail`默认为`true`，同时记录JSON和保存命中调用的Tensor。仅需Tensor文件时，配置`"nan_check": {"detail": false}`；此时不生成`dump.json`、`stack.json`和`construct.json`。`detail`仅适用于默认的`isnan`模式，不能与`slot`同时配置。

创建 `repro_nan.py`，绿色 `+` 行是工具新增代码，保存时去掉行首的 `+`：

```diff
import torch
import torch_npu
+from msprobe.pytorch import PrecisionDebugger

torch.npu.set_device(0)
+debugger = PrecisionDebugger(config_path="./config.json")
x = torch.tensor([4.0, -1.0], device="npu", requires_grad=True)

+debugger.start()
y = torch.sqrt(x)
z = torch.add(y, 1.0)
loss = z.sum()
loss.backward()
+debugger.stop()
+debugger.step()
+torch.npu.synchronize()

print("loss:", loss.item())
print("x.grad:", x.grad.detach().cpu().tolist())
```

```bash
python3 repro_nan.py
```

预期输出：

```text
loss: nan
x.grad: [0.25, nan]
```

## 3. 落盘结果

以下展示本例的落盘格式与预期内容，进程 ID、API 和采集序号以实际运行为准。目录树省略了`sum`和反向API文件；`proc<pid>`是进程目录，有分布式rank时使用`rank<ID>`。

```text
nan_check_dump/
└── step0/
    └── proc<pid>/
        ├── dump.json
        ├── stack.json
        ├── construct.json
        └── dump_tensor_data/
            ├── Torch.sqrt.0.forward.0.input.0.pt
            ├── Torch.sqrt.0.forward.0.output.0.pt
            ├── Torch.add.0.forward.0.input.0.pt
            ├── Torch.add.0.forward.0.output.0.pt
            └── ...
```

`dump.json`中，`sqrt`和`add`的部分检测结果如下：

```json
{
    "data": {
        "Torch.sqrt.0.forward": {
            "input_args": [{"shape": [2], "is_nan": 0}],
            "output": [{"shape": [2], "is_nan": 1}]
        },
        "Torch.add.0.forward": {
            "input_args": [{"shape": [2], "is_nan": 1}],
            "output": [{"shape": [2], "is_nan": 1}]
        }
    }
}
```

`is_nan` 是每个 tensor 的标志：`0` 不含 NaN，`1` 含 NaN，`null` 无法检测。前向输出含 NaN 时，输入、输出自动保存为 `.pt`；`stack.json` 按“API 名列表、调用栈列表”分组保存。

非空FP8/FP4 Tensor的检测结果为`null`，整数、bool和空Tensor为`0`。未知标志不触发保存；同次调用的其他输出含NaN时，仍保存全部原始输入和输出Tensor。

## 4. 读取文件并定位原因

下面的命令读取最近一次采集，打印 `sqrt` 的检测标志、保存的输入输出，以及对应调用栈：

```bash
python3 - <<'PY'
import json
from pathlib import Path
import torch

dump_file = max(Path("nan_check_dump").rglob("dump.json"), key=lambda p: p.stat().st_mtime)
records = json.loads(dump_file.read_text(encoding="utf-8"))["data"]
api = next(name for name in records if "sqrt" in name and name.endswith(".forward"))
record = records[api]
print(api, "input=", record["input_args"][0]["is_nan"], "output=", record["output"][0]["is_nan"])
for path in sorted((dump_file.parent / "dump_tensor_data").glob(f"{api}.*.pt")):
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    print(path.name, tensor.tolist())
stack = json.loads((dump_file.parent / "stack.json").read_text(encoding="utf-8"))
for api_names, frames in stack.values():
    if api in api_names:
        print("\n".join(frames))
PY
```

处理结果节选（路径和行号以实际输出为准）：

```text
Torch.sqrt.0.forward input= 0 output= 1
Torch.sqrt.0.forward.0.input.0.pt [4.0, -1.0]
Torch.sqrt.0.forward.0.output.0.pt [2.0, nan]
File .../repro_nan.py, line 10, in <module>,
 y = torch.sqrt(x)
```

结论：`sqrt` 输入不含 NaN，输出首次出现 NaN；保存的输入中有 `-1.0`，对应调用栈指向 `torch.sqrt(x)`。回到这行代码检查负数的来源和计算定义域。`add` 的输入、输出都含 NaN，是在传播上游结果。

下图将上述输入、输出值与检测标志放在一起，对照本例中`sqrt`首次产生NaN和`add`传播NaN的区别。

![本例的数值与检测标志对照：sqrt首次产生NaN，add传播上游NaN](../../figures/pytorch_nan_check_quickstart/nan_propagation.png)
