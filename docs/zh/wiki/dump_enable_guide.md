# 常见框架dump工具使能

## 概述

本文档介绍常见框架中调试工具的使能方法，包括dump工具和monitor工具的配置位置。

## 工具使能

### 工具资料说明

- **文档地址**: [Data Dump for PyTorch](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/pytorch_data_dump_instruct.md)

### 工具添加位置查找方法

想要找该在哪添加工具的时候，我们可以任意打印一个API的调用栈信息出来看，从调用栈里可以看出工具的添加位置。

比如打印linear的调用栈代码有以下两种方式：

#### 方法1：在torch里直接打印调用栈

![image.png](https://raw.gitcode.com/user-images/assets/7898473/b304f147-c369-471d-a460-a7e3f9bee152/image.png 'image.png')

#### 方法2：在启动脚本里替换掉原有linear

```python
import torch
import torch.nn.functional as F
import traceback

# 保存原始函数
original_linear = F.linear

# 替换functional中的linear，使用*args和**kwargs适配所有参数
def custom_linear(*args, **kwargs):
    print("="*50)
    print("调用F.linear，调用栈如下:")
    traceback.print_stack()
    # 将接收的参数原封不动地传给原始函数
    return original_linear(*args, **kwargs)

F.linear = custom_linear
```

**说明**：有了调用栈之后就很好找工具的添加位置了。

![image.png](https://raw.gitcode.com/user-images/assets/7898473/0889ef2c-af39-494f-b302-c87f297eefb6/image.png 'image.png')

### 各框架工具添加位置

#### MindSpeed-LLM

![image.png](https://raw.gitcode.com/user-images/assets/7898473/0a8f98aa-cb39-4c3f-b772-eb7c36fc13a5/image.png 'image.png')

#### MindSpeed-MM

![image.png](https://raw.gitcode.com/user-images/assets/7898473/ddd75cf2-fabb-4ccf-a7cc-bbfd0a7bd6ff/image.png 'image.png')

#### LLaMA-Factory

![image.png](https://raw.gitcode.com/user-images/assets/7898473/1f1c7307-3f28-440a-8ee7-57e01a5c0fc1/image.png 'image.png')

#### accelerate + DeepSpeed

![image.png](https://raw.gitcode.com/user-images/assets/7898473/1de6a6f8-2a11-4a4b-97b3-d458f45f31cf/image.png 'image.png')

#### torchtitan (FSDP2后端)

![image.png](https://raw.gitcode.com/user-images/assets/7898473/0ddc5861-2d0f-4a2e-8cec-18c4268cc649/image.png 'image.png')

#### VERL (fsdp后端)

使能确定性的位置：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/7c9eb36a-096a-42e4-8120-e44075978cb6/image.png 'image.png')

generate_sequences  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/a86b2a58-59f9-4abd-838d-274f4bb65653/image.png 'image.png')  

以上使能方式只针对vllm eager模式后端，不同配置或使能方式可能会变化。

update_actor  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/7edade77-bd4b-4d0b-ae08-3c72c6e81078/image.png 'image.png')

compute_log_prob  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/f0a85328-2860-44a2-8051-5322628032af/image.png 'image.png')

compute_ref_log_prob  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/ce7f7a79-3f7a-425b-9172-6ac41d716954/image.png 'image.png')

#### VERL (sglang后端)

##### 示例版本说明

| 组件          | 版本                 |
|-------------|--------------------|
| CANN        | 8.5.0              |
| python      | 3.11               |
| torch       | 2.7.1              |
| torch_npu   | 2.7.1.post2        |
| verl        | 0.7.1              |
| sglang      | 0.5.8              |
| MindSpeed   | 2.3.0_core_r0.12.1 |
| Megatron-LM | core_v0.12.1       |

##### 1. 创建config.json文件，用于配置dump参数

config.json配置文件详细介绍请参见[配置文件介绍](../dump/config_json_introduct.md)。

##### 2. SGLang框架中使能msProbe工具

找到SGLang框架`ModelRunner`类所属文件：sglang/srt/model_executor/model_runner.py

- `ModelRunner`类的`__init__`方法中添加`PrecisionDebugger`接口，传入`config.json`文件真实路径。

    ```python
        from msprobe.pytorch import PrecisionDebugger, seed_all
        seed_all(mode=True)
        self.debugger = PrecisionDebugger(config_path="/home/config.json")
    ```
    
    ![image.png](https://raw.gitcode.com/user-images/assets/9721900/6569931a-8993-4761-8e27-6b96353723c9/image.png 'image.png')

- `ModelRunner`类的`forward`方法中添加`start`、`stop`和`step`接口。

  - `forward`方法开始处

    ```python
        if hasattr(self, 'debugger'):
            self.debugger.start(model=self.model, rank_id=self.gpu_id)
    ```
    
    ![image.png](https://raw.gitcode.com/user-images/assets/9721900/7f813099-9254-4198-9511-d0c880c2a7fe/image.png 'image.png')

  - `forward`方法结束处

    ```python
        if hasattr(self, 'debugger'):
            self.debugger.stop()
            self.debugger.step()
    ```

    ![image.png](https://raw.gitcode.com/user-images/assets/9721900/c6625e1c-e8ff-4131-8c0b-de81bfa2dd83/image.png 'image.png')
    
##### 3. 注意

1. 使能方式只针对eager模式后端，使能方式和位置可能会随版本变化，verl启动训推脚本需要添加

    ```diff
        actor_rollout_ref.rollout.enforce_eager=True
    +   actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend"
    +   actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True
    ```

2. verl中使用ray自动纳管npu会导致卡不可见，需要设置环境变量`export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`
