# nan_check 使用示例

本例采集一次前向、反向计算：

- 负数开平方使 loss 变为 NaN，再从落盘文件找到产生 NaN 的 API 和输入值。
- 使用单卡PyTorch NPU、eager执行场景和默认`isnan`模式，入口为`PrecisionDebugger`。AclGraph图模式请参见[NaN自动检测](./aclgraph_dump_instruct.md#nan-自动检测)。

先在`config.json`中配置采集范围，再用`PrecisionDebugger.start()`和`stop()`包住要检查的代码。运行后，查看`dump.json`中的NaN检测结果；需要看具体的输入、输出数据时，读取`.pt`文件；需要找调用位置时，查看`stack.json`。

## 1. 构建

本例需要`aclgraph_dump`模块，编译时添加`-e include-mod=aclgraph_dump`。环境准备、编译和安装步骤请参见[安装指南](../../install_guide/msprobe_install_guide.md)。

## 2. 运行采集

在本地克隆的`msprobe`目录中创建`config.json`和下方的`repro_nan.py`，并在该目录运行。本例使用默认的`isnan`模式：

```json
{
    "task": "nan_check",
    "dump_path": "./nan_check_dump",
    "level": "L1"
}
```

需要指定采集卡、step或API范围时，可在[完整配置示例及参数说明](./config_json_introduct.md#task配置为nan_check)中查看`rank`、`step`、`scope`、`list`、`data_mode`等字段；本例沿用默认值。

`nan_check.detail`默认为`true`；仅需Tensor文件时，配置`"nan_check": {"detail": false}`。两种取值的输出内容见[落盘结果](#3-落盘结果)。

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

以下是`dump.json`的`data`字段中`sqrt`和`add`两个前向API的完整记录，省略文件头及其他API记录。`add`的第二个位置参数为Python标量`1.0`，通过`type`和`value`字段记录：

```json
{
    "data": {
        "Torch.sqrt.0.forward": {
            "input_args": [
                {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [2],
                    "requires_grad": true,
                    "is_nan": 0
                }
            ],
            "input_kwargs": {},
            "output": [
                {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [2],
                    "requires_grad": true,
                    "is_nan": 1
                }
            ]
        },
        "Torch.add.0.forward": {
            "input_args": [
                {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [2],
                    "requires_grad": true,
                    "is_nan": 1
                },
                {
                    "type": "float",
                    "value": 1.0
                }
            ],
            "input_kwargs": {},
            "output": [
                {
                    "type": "torch.Tensor",
                    "dtype": "torch.float32",
                    "shape": [2],
                    "requires_grad": true,
                    "is_nan": 1
                }
            ]
        }
    }
}
```

### is_nan字段

`isnan`模式在`detail: true`时为每个输入、输出Tensor记录`is_nan`：`1`表示含NaN，`0`表示NaN检测通过，`null`表示检测结果未知。整数Tensor、bool Tensor和空Tensor的`is_nan`为`0`；元素数量大于0的FP8/FP4 Tensor、FakeTensor和meta Tensor的`is_nan`为`null`。Python整数、浮点数、bool等标量通过`type`和`value`字段记录。

### Tensor与附加文件

前向输出含NaN时，保存当前API调用在采集范围内的全部输入、输出Tensor；反向计算得到的梯度含NaN时，保存该次反向调用的输入、输出梯度。保存以输出Tensor的`is_nan: 1`为触发条件；同次调用的其他Tensor标记为`null`时，也会尝试保存该调用的全部原始输入、输出Tensor。每次API调用的文件分别保存，文件名中的API调用序号和保存序号以实际输出为准。

| detail | 落盘结果 |
| --- | --- |
| `true`（默认） | `dump.json`记录检测结果，输出含NaN的调用还会保存`.pt`文件。默认同时生成`stack.json`和`construct.json`；设置根级参数`extra_info: false`后，输出由`dump.json`和对应的`.pt`文件组成。 |
| `false` | 输出文件仅包含上述调用的`.pt`文件。 |

`stack.json`按“API名列表、调用栈列表”分组保存，供下文定位调用位置。

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

结论：`sqrt`输入的NaN检测通过，输出首次出现NaN；保存的输入中有 `-1.0`，对应调用栈指向 `torch.sqrt(x)`。回到这行代码检查负数的来源和计算定义域。`add` 的输入、输出都含 NaN，是在传播上游结果。

下图将上述输入、输出值与检测标志放在一起，对照本例中`sqrt`首次产生NaN和`add`传播NaN的区别。

![本例的数值与检测标志对照：sqrt首次产生NaN，add传播上游NaN](../../figures/pytorch_nan_check_quickstart/nan_propagation.png)

## 5. slot模式

eager场景还可使用`slot`模式，通过NPU寄存器状态检测API中的NaN、Inf或过程溢出。按[安装指南](../../install_guide/msprobe_install_guide.md)使用`-e include-mod=nan_check`构建安装，并将`config.json`替换为[check_mode参数配置说明](./config_json_introduct.md#check_mode参数配置说明)中的完整`slot`配置。`detail`的适用模式为`isnan`；在`slot`模式中显式配置该字段会报错。

运行前设置环境变量，再执行第2节的脚本：

```bash
export INF_NAN_MODE_FORCE_DISABLE=1
python3 repro_nan.py
```

`dump.json`以API为粒度记录`is_nan`检测结果。即使输出Tensor正常，也可能检测到过程溢出；int64运算的检测结果存在偏差风险，需结合实际数值复核。`tensor_list`指定的API发生溢出时，触发异常并将详细数据保存到`extra-info`，供[Tensor解析工具](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/user_guide/msaicerr/Dump_files_parsing.md)分析。
