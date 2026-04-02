# SGLang精度数据采集

## 简介

msProbe工具通过在SGLang框架中负责模型前向传播执行的核心类`ModelRunner`中添加`PrecisionDebugger`接口并启动推理的方式，
采集模型在运行过程中的精度数据。

dump "statistics"模式的性能膨胀大小与"tensor"模式采集的数据量大小，可以参考[dump基线](../baseline/pytorch_data_dump_perf_baseline.md)。

**注意**：

* 采集数据前，需要指定SGLang框架的`--disable-cuda-graph`参数关闭图模式。
* 使用SGLang框架的在线模式采集数据，需要指定SGLang框架的`--skip-server-warmup`参数关闭warmup，避免采集到warmup阶段的数据。
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

以下通过一个简单的示例，展示如何在SGLang框架中使用msProbe工具进行精度数据采集。

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

    config.json配置文件详细介绍请参见[配置文件介绍](./config_json_introduct.md)。

2. SGLang框架中使能msProbe工具
  
    找到SGLang框架`ModelRunner`类所属文件：sglang/srt/model_executor/model_runner.py

    - `ModelRunner`类的`__init__`方法中添加`PrecisionDebugger`接口，传入`config.json`文件路径。

      ```python
      class ModelRunner(ModelRunnerKVCacheMixin):
          """ModelRunner runs the forward passes of the models."""
           
          def __init__(
              self,
              model_config: ModelConfig,
              mem_fraction_static: float,
              gpu_id: int,
              tp_rank: int,
              ...
          ):
              ################################ msprobe ################################
              from msprobe.pytorch import PrecisionDebugger, seed_all
              seed_all(mode=True)
              self.debugger = PrecisionDebugger(config_path="./config.json")
              ################################ msprobe ################################
              # Parse args
              self.mem_fraction_static = mem_fraction_static
              self.device = server_args.device
              self.gpu_id = gpu_id
              self.tp_rank = tp_rank
              self.tp_size = tp_size
              ...
      ```

    - `ModelRunner`类的`forward`方法中添加`start`、`stop`和`step`接口。
      
      SGLang框架默认场景:

      ```python
      class ModelRunner(ModelRunnerKVCacheMixin):
          """ModelRunner runs the forward passes of the models."""
          
          def __init__(
              ...
            
          def forward(
              self,
              forward_batch: ForwardBatch,
              skip_attn_backend_init: bool = False,
              pp_proxy_tensors: Optional[PPProxyTensors] = None,
              reinit_attn_backend: bool = False,
              split_forward_count: int = 1,
          ) -> ModelRunnerOutput:
              self.forward_pass_id += 1
              ################################ msprobe ################################
              if hasattr(self, 'debugger'):
                  self.debugger.start(model=self.model)
              ################################ msprobe ################################
              
              ...
              
              ################################ msprobe ################################ 
              if hasattr(self, 'debugger'):
                  self.debugger.stop()
                  self.debugger.step()
              ################################ msprobe ################################
          
              return output
      ```

      SGLang框架启用DP场景（`--dp-size`>1），需要在`start`接口中配置`rank_id`参数：

      ```python
       if hasattr(self, 'debugger'):
          self.debugger.start(model=self.model, rank_id=self.gpu_id)
      ```

3. 启动SGLang框架模型推理，开始采集数据

    - 在线模式
        1. 启动服务

           ```shell
           #!/bin/bash
           export TORCHDYNAMO_DISABLE=1
           
           python3 -m sglang.launch_server \
            --model-path Qwen/Qwen2.5-0.5B-Instruct \
            --host 127.0.0.1 \
            --port 1027 \
            --disable-cuda-graph \
            --skip-server-warmup
           ```

        2. 发送请求，自动开始dump

           ```shell
            curl -H "Content-type: application/json" \
             -X POST \
             -d '{
                 "model": "Qwen/Qwen2.5-0.5B-Instruct",
                 "messages": [
                     {
                         "role": "user",
                         "content": "Hello, my name is"
                     }
                 ],
                 "max_tokens": 10
             }' \
             http://127.0.0.1:1027/v1/chat/completions
           ```

    - 离线模式
      
      离线脚本示例如下，运行将自动开始dump：

      ```python
      import os
      import asyncio
        
      import sglang as sgl
      import sglang.test.doc_patch
      from sglang.utils import async_stream_and_merge, stream_and_merge
        
      def main():
          llm = sgl.Engine(model_path="Qwen/Qwen2.5-0.5B-Instruct", disable_cuda_graph=True)
        
          prompts = [
              "Hello, my name is",
              "The president of the United States is",
              "The capital of France is",
              "The future of AI is",
          ]
        
          sampling_params = {"temperature": 0.8, "top_p": 0.95}
        
          outputs = llm.generate(prompts, sampling_params)
          for prompt, output in zip(prompts, outputs):
              print("===============================")
              print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        
        
      if __name__ == '__main__':
          main()
      ```

## 数据采集功能介绍

SGLang精度数据采集详细功能以及采集的dump数据结构与PyTorch场景一致，具体请参见《[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md#数据采集功能介绍)》。
