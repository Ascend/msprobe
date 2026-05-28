# FAQs

## Model Computation Result Change

**Symptom**

During model training, the `seed_all` interface is used to fix randomness and enable deterministic computation simultaneously. This ensures that the loss and gnorm results obtained after the model is executed twice are the same. However, the following causes may lead to the loss or gnorm deviation.

**Cause Analysis**

The computation result may change due to a synchronization operation introduced by the tool.

When the tool collects statistics, it sends the information to the CPU through item and then flushes the information to the JSON file after tensor computation on the device. The item is a synchronous operation, which may change the computation result of the model. Specifically, NaN occurs during model computation, which will not occur after the tool is used.

`ASCEND_LAUNCH_BLOCKING` is an environment variable that controls the operator execution mode in PyTorch training or online inference scenarios. If it is set to `1`, operators run in synchronous mode. If the computation result changes, you can set `ASCEND_LAUNCH_BLOCKING` to `1`. If the result still changes, the result change is caused by a synchronization operation. In this case, you need to reproduce the problem and locate the fault. You are advised to use the asynchronous dump function of msProbe by referring to the `async_dump` field in [Configuration](./dump/config_json_introduct.md).

Solution

Change the computation result using the hook mechanism.

The hook mechanism of PyTorch or MindSpore may change the gradient accumulation sequence in some special scenarios, affecting the gnorm result of model backward computation. Sample code:

```python
import random, os
import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.ln2 = nn.Linear(32, 32)

    def forward(self, x):
        x1 = self.ln1(x)

        x2 = self.bn1(x)
        x2 = self.ln2(x2)
        return x1 + x2


class BigNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = Net()
        self.net2 = Net()

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(out1)
        return out1, out2


def my_backward_hook(module, grad_input, grad_output):
    pass


if __name__ == "__main__":
    os.environ["HCCL_DETERMINISTIC"] = 'true'

    seed = 1234
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = BigNet()
    model.net2.register_full_backward_hook(my_backward_hook)
    inputs = torch.randn(3, 32)

    out1, out2 = model(inputs)
    loss = out1.sum() + out2.sum()
    loss.backward()

    for name, param in model.named_parameters():
        print(f"{name}: {param.grad.mean()}")

```

Run the preceding script again to obtain weight gradients of each model layer. Comment out `model.net2.register_full_backward_hook(my_backward_hook)` and run the script again, and it can be seen that the weight gradient of the BN layer has changed.

If the gnorm changes at the L0 or mix level, try to change the collection level to L1. If the gnorm does not change at the L1 level, there is a high probability that the gradient calculation result changes due to the hook mechanism.

## Data Collection

1. Why are `null` or `None`  displayed in the API or module statistics in `dump.json`?

   Common causes:

   - The input or output parameter is `None`.
   - If the input or output parameter type is not supported by the tool, a log will be printed to remind you.
   - When the `dtype` of the input or output tensor is `bool`, the `Mean` and `Norm` fields are `null`.

2. If the output of nn.Module is of the `namedtuple` type, the tool dumps the data of each field, but the output data type is converted to `tuple`. Why is that?
   - This is due to the PyTorch framework itself, which converts `namedtuple` data to `tuple` when registering the module's backward hook.

3. If an API is included in `support_wrap_ops.yaml` but the data of the API is not dumped, what is the reason?
   - Check whether the API call is within the collection scope, that is, within the scope covered by the `start` and `stop` APIs.
   - The tool patches the API only when it is called, so that the data can be dumped. Therefore, when the API is directly imported and called, the address of the API has been determined,
   and the tool cannot patch the API. As a result, the API data cannot be dumped. As shown in the following example, `relu` cannot be dumped.

   ```python
   import torch
   from torch import relu  # At this time, the relu address has been determined and cannot be modified.
   
   from msprobe.pytorch import PrecisionDebugger
   
   debugger = PrecisionDebugger(dump_path="./dump_data")
   x = torch.randn(10)
   debugger.start()   # In this case, the APIs under torch are patched, but the APIs imported cannot be patched.
   x = relu(x)          
   debugger.stop()
   ```

   In the preceding scenario, if you want to collect the `relu` data, change `relu(x)` to `torch.relu(x)`.

4. When L0 dump is used, some module data is not collected. Why is that?
   - Check whether the log contains `The {module_name} has registered deprecated register_backward_hook` information.
     This message indicates that the module is mounted with `register_backward_hook` that has been deprecated by the PyTorch framework. This conflicts with `register_full_backward_hook` used by the tool. Therefore, the tool skips the backward data collection of the module.
   - If you want to collect data of all modules, replace `register_backward_hook` used in the model with `register_full_backward_pre_hook` or `register_full_backward_hook` recommended by the PyTorch framework.

5. When data is dumped in the vLLM scenario, the error message `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and npu:0!` is displayed.
   - This occurs because the debugger is instantiated before the LLM. To solve this problem, instantiate the debugger after the LLM. The following is an example:

   ```python
   from vllm import LLM, SamplingParams
   from msprobe.pytorch import PrecisionDebugger
   prompts = [
      "Hello, my name is",
      "The president of the United States is",
      "The capital of France is",
      "The future of AI is",
   ]
   
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
   llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")
   
   debugger = PrecisionDebugger("./config.json")    # Instantiate the debugger after the LLM instantiation.
   
   debugger.start()
   outputs = llm.generate(prompts, sampling_params)
   debugger.stop()
   ```

6. When using msProbe to collect data in the PyTorch framework, ensure that `NPU_ASD_ENABLE` is set to `0`, to disable feature value detection. Due to tool conflicts, some API data may be missing when this function is enabled.

## Precision Pre-check (PyTorch)

1. Does the pre-check tool need to enable or disable `jit_compile` during the dump and `acc_check` processes?

   A: Yes.

2. Is the pre-check tool useful for APIs that involve data type conversion (e.g., `type_as`)?

   On CPU, the precision of these APIs first increases and then decreases. Therefore, their validity has limited reference value.

3. The error message `ERROR: Got unsupported ScalarType BFloat16` is displayed during the `acc_check` process.

   A: Use the tool of the latest version.

4. The randomness of the Dropout operator on CPU and NPU are different. Why are the results consistent?

   A: This is normal. The tool has special processing for this operator. It only determines that the proportion of positions whose value is `0` is approximately equal to the specified `p` value.

5. Why is `dtype` of the floating-point bench data different from that of the CPU?

   A: For FP16 data, the CPU performs calculations using higher-precision FP32, which aligns with operator precision. This use of higher precision yields results closer to the actual value.

6. What operations are performed by the magic functions of tensors?

   See the table below.

   | Tensor Magic Function | Operation        |
   | --------------- | ---------------- |
   | `__add__`       | +                |
   | `__and__`       | &                |
   | `__bool__`      | Returns the tensor Boolean value.|
   | `__div__`       | /                |
   | `__eq__`        | ==               |
   | `__ge__`        | >=               |
   | `__gt__`        | >                |
   | `__iadd__`      | +=               |
   | `__iand__`      | &=               |
   | `__idiv__`      | /=               |
   | `__ifloordiv__` | //=              |
   | `__ilshift__`   | <<=              |
   | `__imod__`      | %=               |
   | `__imul__`      | *=               |
   | `__ior__`       | \|=              |
   | `__irshift__`   | >>=              |
   | `__isub__`      | -=               |
   | `__ixor__`      | ^=               |
   | `__lshift__`    | <<               |
   | `__matmul__`    | Matrix multiplication        |
   | `__mod__`       | %                |
   | `__mul__`       | *                |
   | `__nonzero__`   | Returns a tensor Boolean value.|
   | `__or__`        | \|               |
   | `__radd__`      | + (backward)       |
   | `__rmul__`      | * (backward)       |
   | `__rshift__`    | >>               |
   | `__sub__`       | -                |
   | `__truediv__`   | /                |
   | `__xor__`       | ^                |

## Precision Comparison (PyTorch)

### Tool Usage

#### Dumping a Specified Fusion Operator

Currently, data collection supports the input and output of fusion operators. To use this function, add fusion operators to [support_wrap_ops.yaml] directory. In the following example, the softmax fusion operator is called.

```python
def npu_forward_fused_softmax(self, input_, mask):
    resl = torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)
    return resl
```

To dump the input and output information of the `npu_scaled_masked_softmax` operator, add the fusion operator to `torch_npu:` in `support_wrap_ops.yaml`.

```yaml
- npu_scaled_masked_softmax
```

The tool supports `npu_scaled_masked_softmax` dumping. This example is for reference only.

### Common Problems

1. Will there be a conflict if dump is performed multiple times in the same directory?

    If dump is performed multiple times in the same directory, the previous result will be overwritten. You can use `dump_path` to modify the dump directory.

2. How to dump operator-level data?
   
   Set `level` to `L2`.

3. Why can't the APIs of the NPU and benchmark data be completely aligned during tool comparison?

    There are difference between torch version and hardware, which is a normal phenomenon.

### Exception Handling

1. `error code: EI0006`

    The CANN version is too early and is incompatible. Upgrade the CANN software to the latest version.

2. `torch_npu._C._clear_overflow_npu() RuntimeError NPU error, error code is 107002.`

    If this error occurs when overflow/underflow detection is enabled, perform the following operations:

    If only one device is used, add the following code. `0` indicates the device ID. Select an idle device ID.

    ```python
    torch.npu.set_device('npu:0')
    ```

    If multiple devices are used, change the corresponding device ID in the code. For example, if the device ID used by the process is `{rank}`, add the following code:

    ```python
    torch.npu.set_device(f'npu:{rank}')
    ```

    If this error occurs during precision comparison, install the latest version of msProbe.

3. Are files generated after data dump (e.g., `VF_lstm_99_forward_input.1.0.npy` and `VF_lstm_99_forward_input.1.1.npy`) normal?

    It is normal that the .npy files have suffixes 1.0, 1.1, or 1.2. For example, when the input data is `[[tensor1, tensor2, tensor3]]`, such suffixes are generated.

4. `NameError: name 'torch_npu' is not defined` is displayed when the kernel-level data of a specified backward API is dumped.

   For an NPU environment, install torch_npu; for a GPU environment, the kernel-level data of a specified API cannot be dumped.

5. `[ERROR] The file path /home/xxx/dump contains special characters.` is reported after `dump_path` is configured.

   Check whether the absolute dump path contains special characters. Ensure that the path name contains only uppercase letters, lowercase letters, digits, underscores (_), slashes (/), dots (.), and hyphens (-). Note that if the path where the script is executed is `/home/abc++/` and `dump_path` is set to `./dump`, the absolute path `/home/abc++/dump` is verified by the tool. In this case, this error is reported because `++` is a special character.

6. Why can't the backward gradient data of the matmul weight be dumped?

   The matmul operation expects two-dimensional inputs. If the input is not two‑dimensional, it is expanded to two dimensions via a `view` operation before the matmul is executed. Consequently, during backward propagation, the backward hook captures gradient information from the `UnsafeViewBackward` operation rather than from `MmBackward`, meaning the backward gradient data of the weight is not obtained. A typical example is that when the input of the linear layer is not two-dimensional and there is no bias, `output = input.matmul(weight.t())` is called. Therefore, the backward gradient data of the weight of the linear layer cannot be obtained.

7. In the `dump.json` file, `dtype` of some APIs is float16, but `dtype` of these APIs displayed in the .npy file is float32.

    When msProbe dumps data, the original data is first transferred from the NPU to the CPU and then converted to NumPy format. The NPU-to-CPU transfer logic is the same as that for GPU-to-CPU. During this process, `dtype` may change from float16 to float32. In cases of inconsistency, `dtype` of the collected data follows that of the .pkl file.

8. `Exception("msprobe: exit after iteration {}". format(max(self.config.step))).` is raised after `dataloader` is called.

   This is normal. `dataloader` ends the program by raising an exception. The stack information can be ignored.

9. `activation_func must be F.gelu` or `ValueError(Only support fusion of gelu and swiglu)` is displayed after the data collection function of the msProbe is used.

    Such an error is commonly seen in acceleration libraries or model repositories such as Megatron, MindSpeed, and ModelLink. The reason is that the tool encapsulates torch APIs, which changes their type and address. For APIs that are already resolved before the tool is enabled, encapsulation is not possible. When an acceleration library subsequently verifies API types—differentiating between encapsulated and original APIs—an error is reported due to the inconsistency.
    The workarounds are as follows:

    - Place the instantiation of `PrecisionDebugger` at the beginning of the file, that is, after the import statement, to ensure that all APIs are encapsulated.

    - Comment out `-gelu` or `-silu` in the `MindStudio-Probe/python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml` file. The tool will skip collecting the API.

    - Comment out the type check that causes the error based on the error stack information.

10. After msProbe is added, an error related to the AsStrided operator or compilation is reported, for example, `Failed to compile Op [AsStrided]`.

     Comment out `- t` and `- transpose` under `Tensor:` in the `MindStudio-Probe/python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml` file in the tool directory.
