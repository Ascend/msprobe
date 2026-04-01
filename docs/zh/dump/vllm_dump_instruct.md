# vLLM采集

## 简介

msProbe工具支持在vLLM推理场景中采集模型执行过程中的中间过程精度数据。

对于不同实现，使用方式如下：

* 如果调用的是`vllm-ascend`，请优先参考vLLM Ascend官方文档《[MSProbe 调试指南](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/msprobe_guide.html)》。
* 如果调用的是社区原生vLLM，则可通过在`GPUModelRunner`中添加`PrecisionDebugger`接口并启动推理的方式完成数据采集，具体指导见本文后续章节。

dump "statistics"模式的性能膨胀大小与"tensor"模式采集的数据量大小，可以参考[dump基线](../baseline/pytorch_data_dump_perf_baseline.md)。

**注意**：

* 当前场景仅支持在vLLM eager模式下采集数据，启动服务时需要指定`--enforce-eager`参数。
* `vllm-ascend`场景推荐直接复用官方已集成能力，通过`--additional-config`传入`dump_config_path`。
* `PrecisionDebugger`的初始化位置应尽量靠前，建议放在`GPUModelRunner.__init__`中模型执行逻辑开始前的位置，确保相关API能够被工具封装。
* `execute_model`通常较长，建议采用最省事的方式，在`start`后使用`try...finally`统一包裹，并在`finally`中调用`stop`和`step`；如果不方便整体包裹，也可以在各个`return`前补充`stop`和`step`调用，否则可能导致`dump.json`等结果文件落盘不完整。
* 若需要采集L0或mix级别数据，`start`接口需要传入`model`参数；若仅采集L1级别统计数据，可直接调用`start()`。
* 如果遇到dynamo相关报错，NPU上可设置环境变量`export TORCHDYNAMO_DISABLE=1`全局关闭dynamo。
* 本工具提供固定的API支持列表，若需要删除或增加dump的
  API，可以在[support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml)文件内手动修改，如下示例：

  ```yaml
  functional:  # functional为算子类别，找到对应的类别，在该类别下按照下列格式删除或添加API
    - conv1d
    - conv2d
    - conv3d
  ```

  删除API的场景：部分模型代码逻辑会存在API原生类型校验，工具执行dump操作时，对模型的API封装可能与模型的原生API类型不一致，此时可能引发校验失败，详见《FAQ》中“[异常情况](../faq.md#异常情况)”的第10条。

## 使用前准备

**环境准备**

安装msProbe工具，详情请参见《[msProbe安装指南](../msprobe_install_guide.md)》。

**约束**

仅支持采集基于PyTorch框架实现的模型，暂不支持PyTorch版本>=2.7的dynamo场景。

## 快速入门

以下给出两类vLLM场景的使用方式。

### vllm-ascend场景

`vllm-ascend`已提供msProbe接入能力，启动服务时可直接通过`--additional-config`传入dump配置文件路径。官方文档当前给出的示例如下：

```shell
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{"dump_config_path": "/data/msprobe_config.json"}'
```

说明：

* 官方文档发布时间对应当前访问内容为 `vllm-ascend zh-cn latest` 页面，链接为：
  `https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/msprobe_guide.html`
* 该方式适用于已集成msProbe能力的`vllm-ascend`，无需再手工修改`GPUModelRunner`实现。

### 社区vLLM场景

以下通过一个简单的示例，展示如何在社区原生vLLM框架的`GPUModelRunner`中使用msProbe工具进行精度数据采集。

1. 配置文件创建

    在当前目录下创建`config.json`文件，用于配置dump参数。内容示例如下：

    ```json
      {
        "task": "statistics",
        "dump_path": "/home/data_dump",
        "rank": [],
        "step": [],
        "level": "mix",
        "async_dump": false,
        "statistics": {
          "scope": [],
          "list": [],
          "data_mode": [
            "all"
          ],
          "summary_mode": "statistics"
        }
      }
    ```

    config.json配置文件及配置样例详细介绍请参见[配置文件介绍](./config_json_introduct.md)。

2. vLLM框架中使能msProbe工具

    找到vLLM框架`GPUModelRunner`类所属文件：`vllm/v1/worker/gpu_model_runner.py`

    - `GPUModelRunner`类的`__init__`方法中添加`PrecisionDebugger`接口，传入`config.json`文件路径。

      ```text
      class GPUModelRunner:
          def __init__(
              self,
              vllm_config: VllmConfig,
              device: torch.device,
          ):
              ################################ msprobe ################################
              from msprobe.pytorch import PrecisionDebugger, seed_all
              seed_all(mode=True)
              self.debugger = PrecisionDebugger(config_path="./config.json")
              ################################ msprobe ################################
              self.vllm_config = vllm_config
              self.device = device
              ...
      ```

    - `GPUModelRunner`类的`execute_model`方法中添加`start`、`stop`和`step`接口。

      `execute_model`函数通常较长，推荐优先使用`try...finally`方式统一处理结束逻辑，改动最小，也更不容易遗漏返回分支：

      ```text
      @torch.inference_mode()
      def execute_model(
          self,
          scheduler_output: "SchedulerOutput",
          intermediate_tensors: Optional[IntermediateTensors] = None,
      ) -> Union[ModelRunnerOutput, torch.Tensor]:
          ################################ msprobe ################################
          if hasattr(self, "debugger"):
              self.debugger.start(model=self.model)
          ################################ msprobe ################################
  
          try:
              ...
              return output
          finally:
              ################################ msprobe ################################
              if hasattr(self, "debugger"):
                  self.debugger.stop()
                  self.debugger.step()
              ################################ msprobe ################################
      ```

      如果不方便使用`try...finally`整体包裹，也可以在对应的`return`前增加如下调用：

      ```text
      if hasattr(self, "debugger"):
          self.debugger.stop()
          self.debugger.step()
      return output
      ```

3. 启动vLLM服务并开始采集数据

    启动服务时需开启eager模式，示例如下：

    ```shell
    #!/bin/bash
    export TORCHDYNAMO_DISABLE=1
    
    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --dtype float16 \
      --enforce-eager \
      --host 0.0.0.0 \
      --port 8000
    ```

    服务启动后发送推理请求，请求执行过程中将自动触发dump：

    ```shell
    curl http://127.0.0.1:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "prompt": "Explain gravity in one sentence.",
            "max_tokens": 32,
            "temperature": 0
          }'
    ```

## 数据采集功能介绍

vLLM场景精度数据采集详细功能以及采集的dump数据结构与PyTorch场景一致，具体请参见《[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md#数据采集功能介绍)》。
