# nan_check Usage Example

This example collects one forward and backward computation.

- Taking a square root of a negative value makes the loss NaN. Use the saved files to identify the API and its input.
- Run eagerly on a single PyTorch NPU with the default `isnan` mode through `PrecisionDebugger`. For AclGraph graph mode, see [Automatic NaN Detection](./aclgraph_dump_instruct.md#automatic-nan-detection).

Configure collection in `config.json`, then call `PrecisionDebugger.start()` before the code you want to check and `stop()` afterward. After the run, check `dump.json` for the NaN detection results, read the `.pt` files for the input and output data, and check `stack.json` for the call locations.

## 1. Build

This example requires the `aclgraph_dump` module. Add `-e include-mod=aclgraph_dump` when building. For environment setup, build, and installation steps, see the [Installation Guide](../../install_guide/msprobe_install_guide.md).

## 2. Data Collection

Create `config.json` and the `repro_nan.py` below in the locally cloned `msprobe` directory, and run there. This example uses the default `isnan` mode:

```json
{
    "task": "nan_check",
    "dump_path": "./nan_check_dump",
    "level": "L1"
}
```

To select ranks, steps, or an API range, see `rank`, `step`, `scope`, `list`, `data_mode`, and other fields in the [complete configuration example and parameter descriptions](./config_json_introduct.md#task--nan_check). This example uses their default settings.

`nan_check.detail` defaults to `true`. To save tensor files only, set `"nan_check": {"detail": false}`. See [Saved Files](#3-saved-files) for the output of each setting.

Create `repro_nan.py`. Green `+` lines are tool additions; remove the leading `+` before saving:

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

Expected output:

```text
loss: nan
x.grad: [0.25, nan]
```

## 3. Saved Files

The saved-file format and values below are illustrative and expected; process IDs, API indices, and collection indices depend on the actual run. The tree omits `sum` and backward API files. `proc<pid>` identifies the process, and distributed execution uses `rank<ID>`.

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

Below are the complete forward API records for `sqrt` and `add` in the `data` field of `dump.json`. The file header and other APIs are omitted. The second positional argument of `add` is the Python scalar `1.0`, recorded through `type` and `value` fields:

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

### is_nan Field

With `detail: true`, `isnan` records each input/output tensor's `is_nan`: `1` for NaN, `0` for a passed NaN check, or `null` for an unknown detection result. Integer, bool, and empty tensors record `0`; FP8/FP4 tensors with at least one element, FakeTensors, and meta tensors record `null`. Python scalar integers, floats, and bools are recorded through `type` and `value` fields.

### Tensors and Additional Files

When a forward output contains NaN, all input/output tensors of that API call within the collection range are saved. When gradients computed by a backward call contain NaN, that call's input/output gradients are saved. Saving is triggered by an output tensor with `is_nan: 1`; other tensors with `null` flags in the same call are also submitted for saving with all original input/output tensors. Files are saved separately for each API call; API-call and save indices depend on the actual output.

| detail | Saved files |
| --- | --- |
| `true` (default) | `dump.json` records detection results, and calls with NaN outputs also save `.pt` files. `stack.json` and `construct.json` are generated by default; setting the root parameter `extra_info: false` makes the output consist of `dump.json` and the applicable `.pt` files. |
| `false` | The output consists only of `.pt` files for the calls described above. |

`stack.json` stores groups of API-name lists and call-stack lists for locating calls in the next section.

## 4. File Reading and Cause Location

This command reads the latest collection and prints the `sqrt` flags, saved inputs and outputs, and matching call stack:

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

An excerpt of the processing output is as follows. Actual paths and line numbers vary.

```text
Torch.sqrt.0.forward input= 0 output= 1
Torch.sqrt.0.forward.0.input.0.pt [4.0, -1.0]
Torch.sqrt.0.forward.0.output.0.pt [2.0, nan]
File .../repro_nan.py, line 10, in <module>,
 y = torch.sqrt(x)
```

Conclusion: the `sqrt` input passes the NaN check, and its output first produces NaN. The saved input contains `-1.0`, and the call stack points to `torch.sqrt(x)`. Inspect the source of the negative value and the operation's input domain at that line. Both the input and output of `add` contain NaN, showing upstream propagation.

The figure below combines the values and detection flags above, contrasting the first NaN from `sqrt` with its propagation through `add` in this example.

![Numeric values and detection flags in this example: sqrt produces the first NaN and add propagates it](../../figures/pytorch_nan_check_quickstart/nan_propagation.png)

## 5. slot Mode

Eager execution also supports `slot` mode, which reads NPU register status for NaN, Inf, or intermediate overflow in APIs. Follow the [Installation Guide](../../install_guide/msprobe_install_guide.md) to build with `-e include-mod=nan_check`, and replace `config.json` with the complete `slot` configuration in [check_mode Configuration](./config_json_introduct.md#check_mode-configuration). `detail` applies to `isnan`; explicitly setting it in `slot` mode raises an error.

Set the environment variable before running the script from section 2:

```bash
export INF_NAN_MODE_FORCE_DISABLE=1
python3 repro_nan.py
```

`dump.json` records `is_nan` at API granularity. Intermediate overflow may be detected even when the output tensor is normal. Results for int64 operations carry a risk of inaccuracy and require checking against the actual values. Overflow in an API selected by `tensor_list` raises an exception and saves detailed data under `extra-info` for the [Tensor Parsing Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/troubleshooting/troubleshooting_0532.html).
