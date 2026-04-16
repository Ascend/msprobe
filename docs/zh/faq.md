# FAQ

## 模型计算结果改变原因分析

**问题现象**

在模型训练场景下，使用seed_all接口同时固定随机性和开启计算，通信确定性计算，能够保证模型执行两次得到的loss和gnorm结果完全一样。如果出现使能工具后loss或者gnorm出现偏差，可能是以下原因导致。

**原因分析**

工具引入同步导致计算结果变化：

工具采集统计量数据时，会涉及到将device上的tensor计算后的统计量信息通过item的时候传到CPU侧，再落盘到json文件中，item操作是一个同步的操作，可能会导致模型的计算结果出现变化。**一般的现象是模型计算出现NaN，且在未使用工具时问题会复现，使用工具后问题不再出现。**

ASCEND_LAUNCH_BLOCKING是一个环境变量，用于控制在PyTorch训练或在线推理场景中算子的执行模式。当设置为“1”时，算子将采用同步模式运行。因此如果出现加工具计算结果变化，可以设置ASCEND_LAUNCH_BLOCKING为1，如果结果仍然发生变化，则说明是由于同步引起的结果改变。这个时候需要复现问题现象完成问题定位，推荐使用msProbe工具的异步dump功能，具体使用方式可查看[config配置](./dump/config_json_introduct.md)中的async_dump字段。

**解决方案**

通过Hook机制改变计算结果：

PyTorch或MindSpore的hook机制会导致某些特殊场景下梯度计算的累加顺序产生变化，从而影响模型反向计算的gnorm结果。具体代码示例如下：

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

执行一遍以上脚本，可以打印得到模型中各层的权重梯度，注释掉model.net2.register_full_backward_hook(my_backward_hook)后再执行一次，可以看出bn层的权重梯度已经发生了变化。

**如果在L0，mix级别采集出现gnorm发生变化，可以尝试将采集级别改为L1，若L1级别gnorm不发生变化，则大概率是hook机制导致的梯度计算结果变化。**

## 数据采集

1. dump.json中API或Module统计信息里出现null或None值的原因是什么？

   dump.json里出现null或None值的可能性较多，常见的场景有：

   - 输入或者输出参数本身是一个None值。
   - 输入参数或输出参数类型当前工具不支持，会有日志打印提醒。
   - 输入或者输出tensor的dtype为bool时，Mean和Norm等字段为null。

2. 如果存在namedtuple类型的数据作为nn.Module的输出，工具会将各字段数据dump下来，但是输出数据类型会被转成tuple，原因是什么？
   - 这是由于PyTorch框架自身，在注册module的backward hook时，会将namedtuple类型转成tuple类型。

3. 如果某个API在dump支持列表support_wrap_ops.yaml中，但没有dump该API的数据，原因是什么？
   - 首先确认API调用是否在采集范围内，即需要在start和stop接口涵盖的范围内。
   - 其次，由于工具只在被调用时才对API进行patch，从而使得数据可以被dump下来。因此当API是被直接import进行调用时，由于该API的地址已经确定，
   工具无法再对其进行patch，故而该API数据无法被dump下来。如下示例，relu将无法被dump：

   ```python
   import torch
   from torch import relu    # 此时relu地址已经确定，无法修改
   
   from msprobe.pytorch import PrecisionDebugger
   
   debugger = PrecisionDebugger(dump_path="./dump_data")
   x = torch.randn(10)
   debugger.start()    # 此时会对torch下面的API进行patch，但已无法对import进来的API进行patch
   x = relu(x)          
   debugger.stop()
   ```

   在上述场景中，若希望采集relu数据，只需要将`relu(x)`修改为`torch.relu(x)`即可。

4. 在使用L0 dump时，发现有些module的数据没有采集下来，原因是什么？
   - 确认日志打印中是否存在`The {module_name} has registered deprecated register_backward_hook`信息，
     该信息说明module挂载了被PyTorch框架废弃的register_backward_hook，这与工具使用的register_full_backward_hook接口会产生冲突，故工具会跳过该module的反向数据采集。
   - 如果您希望所有module数据都能采集下来，可以将模型中使用的register_backward_hook接口改为PyTorch框架推荐的register_full_backward_pre_hook或register_full_backward_hook接口。

5. 在vllm场景下进行数据dump时，发现报错：`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and npu:0!`
   - 这是因为工具的debugger实例化早于LLM实例化导致的，解决方法就需要将debugger的实例化移至LLM实例化之后进行，可参考下方示例：

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
   
   debugger = PrecisionDebugger("./config.json")    # debugger实例化晚于LLM实例化
   
   debugger.start()
   outputs = llm.generate(prompts, sampling_params)
   debugger.stop()
   ```

6. 在使用msProbe进行PyTorch框架的数据采集功能时，请注意确认环境变量NPU_ASD_ENABLE=0，即关闭特征值检测功能。由于工具冲突，在该功能开启的情况下可能导致某些API数据采集的缺失。

## 精度预检（PyTorch）

1. 预检工具在dump和acc_check的过程中，是否需要同时开启或关闭jit编译（jit_compile）？

   答：是。

2. 预检工具对于type_as这类涉及数据类型转换操作的API，是否具有参考性？

   由于这类API在CPU侧存在精度先提升后下降的操作，因此这类API的有效性的参考价值有限。

3. acc_check过程中出现报错：ERROR: Got unsupported ScalarType BFloat16。

   答：请使用最新版本的工具。

4. Dropout算子，CPU和NPU的随机应该不一样，为什么结果比对是一致的？

   答：这个结果是正常的，工具对该算子有特殊处理，只判定位置为0的位置比例大约和设定p值相当。

5. 为什么浮点型数据bench和CPU的dtype不一致？

   答：对于fp16的数据，CPU会上升一个精度fp32去计算，这是和算子那边对齐的精度结论，CPU用更高精度去计算会更接近真实值。

6. Tensor魔法函数具体对应什么操作？

   答：

   | Tensor魔法函数  | 具体操作         |
   | --------------- | ---------------- |
   | `__add__`       | +                |
   | `__and__`       | &                |
   | `__bool__`      | 返回Tensor布尔值 |
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
   | `__matmul__`    | 矩阵乘法         |
   | `__mod__`       | %                |
   | `__mul__`       | *                |
   | `__nonzero__`   | 返回Tensor布尔值 |
   | `__or__`        | \|               |
   | `__radd__`      | +（反向）        |
   | `__rmul__`      | *（反向）        |
   | `__rshift__`    | >>               |
   | `__sub__`       | -                |
   | `__truediv__`   | /                |
   | `__xor__`       | ^                |

## 精度比对（PyTorch）

### 工具使用

#### dump指定融合算子

数据采集当前支持融合算子的输入输出，需要在[support_wrap_ops.yaml](../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml)中添加，比如以下代码段调用的softmax融合算子。

```python
def npu_forward_fused_softmax(self, input_, mask):
    resl = torch_npu.npu_scaled_masked_softmax(input_, mask, self.scale, False)
    return resl
```

如果需要dump其中调用的npu_scaled_masked_softmax算子的输入输出信息，需要在`support_wrap_ops.yaml`中的`torch_npu:`中自行添加该融合算子：

```yaml
- npu_scaled_masked_softmax
```

npu_scaled_masked_softmax融合算子工具已支持dump，本例仅供参考。

### 常见问题

1. 在同一个目录多次执行dump会冲突吗？

    答：会，同一个目录多次dump，会覆盖上一次结果，可以使用dump_path参数修改dump目录。

2. 如何dump算子级的数据？
   
   答：需要配置level为L2模式。

3. 工具比对发现NPU和标杆数据的API无法完全对齐？

    答：torch版本和硬件差异属于正常情况。

### 异常情况

1. HCCL报错： error code: EI0006。

    CANN软件版本较低导致不兼容。升级新版CANN软件版本即可。

2. torch_npu._C._clear_overflow_npu() RuntimeError NPU error，error code is 107002。

    如果运行溢出检测功能遇到这个报错，采取以下解决方法：

    如果是单卡运行，添加如下代码，0是卡号，选择自己空闲的卡号。

    ```python
    torch.npu.set_device('npu:0')
    ```

    如果多卡运行，请在代码中修改对应卡号，比如进程使用卡号为{rank}时可以添加如下代码：

    ```python
    torch.npu.set_device(f'npu:{rank}')
    ```

    如果运行精度比对功能遇到这个报错，尝试安装最新版本的msProbe。

3. dump得到的`VF_lstm_99_forward_input.1.0.npy`、`VF_lstm_99_forward_input.1.1.npy`类似的数据是否正常？

    带1.0/1.1/1.2后缀的npy是正常现象，例如，当输入数据为[[tensor1, tensor2, tensor3]]会生成这样的后缀。

4. dump指定反向API的kernel级别的数据报错：NameError：name 'torch_npu' is not defined。

   答：如果是NPU环境，请安装torch_npu；如果是GPU环境，暂不支持dump指定API的kernel级别的数据。

5. 配置dump_path后，使用工具报错：[ERROR] The file path /home/xxx/dump contains special characters。

   答：请检查你设置的dump绝对路径是否包含特殊字符，确保路径名只包含大小写字母、数字、下划线、斜杠、点和短横线；注意，如果执行脚本的路径为/home/abc++/，设置的dump_path="./dump"，工具实际校验的路径为绝对路径/home/abc++/dump，++为特殊字符，会引发本条报错。

6. 无法dump matmul权重的反向梯度数据。

   答：matmul期望的输入是二维，当输入不是二维时，会将输入通过view操作展成二维，再进行matmul运算，因此在反向求导时，backward_hook能拿到的是UnsafeViewBackward这步操作里面数据的梯度信息，取不到MmBackward这步操作里面数据的梯度信息，即权重的反向梯度数据。典型的例子有，当linear的输入不是二维，且无bias时，会调用output = input.matmul(weight.t())，因此拿不到linear层的weight的反向梯度数据。

7. dump.json文件中的某些API的dtype类型为float16，但是读取此API的npy文件显示的dtype类型为float32。

    答：msProbe工具在dump数据时需要将原始数据从NPU to CPU上再转换为numpy类型，NPU to CPU的逻辑和GPU to CPU是保持一致的，都存在dtype可能从float16变为float32类型的情况，如果出现dtype不一致的问题，最终采集数据的dtype以pkl文件为准。

8. 使用dataloader后raise异常Exception("msprobe: exit after iteration {}". format(max(self.config.step)))。

   答：正常现象，dataloader通过raise结束程序，堆栈信息可忽略。

9. 使用msProbe工具数据采集功能后，模型出现报错，报错信息为：`activation_func must be F.gelu`或`ValueError(Only support fusion of gelu and swiglu)`。

    答：这一类报错常见于Megatron/MindSpeed/ModelLink等加速库或模型仓中，原因是工具本身会封装torch的API（API类型和地址会发生改变），而有些API在工具使能前类型和地址就已经确定，此时工具无法对这类API再进行封装，而加速库中会对某些API进行类型检查，即会把工具无法封装的原始的API和工具封装之后的API进行判断，所以会报错。
    规避方式包含如下三种：

    - 将PrecisionDebugger的实例化放在文件的开始位置，即导包后的位置，确保所有API都被封装。

    - 注释`MindStudio-Probe/python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml`文件中的`-gelu`或者`-silu`，工具会跳过采集该API。

    - 可以考虑根据报错堆栈信息注释引发报错的类型检查。

10. 添加msProbe工具后触发与AsStrided算子相关、或者编译相关的报错，如：`Failed to compile Op [AsStrided]`。

     答：注释工具目录`MindStudio-Probe/python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml`文件中`Tensor:`下的`- t`和`- transpose`。
