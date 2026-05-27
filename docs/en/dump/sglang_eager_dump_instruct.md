# Precision Data Collection in SGLang

## Overview

msProbe can collect model precision data during mode execution by adding `PrecisionDebugger` to the core class `ModelRunner`
that is responsible for executing forward propagation in the SGLang framework and starting inference

For details about performance changes in dump statistics mode and data volume collected in dump tensor mode, see [Dump Baseline] (../baseline/pytorch_data_dump_perf_baseline.md).

**Note**:

* Before collecting data, specify `--disable-cuda-graph` of the SGLang framework to disable the graph mode.
* To collect data in online SGLang mode, specify `--skip-server-warmup` of the SGLang framework to disable warmup, preventing data in the warmup phase from being collected.
* If a dynamo-related error occurs, set `export TORCHDYNAMO_DISABLE=1` on NPU to disable dynamo globally.
* The tool provides a fixed API support list. If you need to delete or add dump APIs,
  manually modify [support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml). The following is an example:

  ```yaml
  functional:  # functional indicates the operator type. Find the corresponding type and delete or add APIs in the following format:
    - conv1d
    - conv2d
    - conv3d
  ```

  Scenario where an API is deleted: The code logic of some models may involve native API type verification. When the tool performs the dump operation, the API type encapsulated by the model may be different from the native API type. In this case, the verification may fail. For details, see [FAQs](../faq.md).

## Preparations

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

Only data of models implemented based on the PyTorch framework can be collected. Currently, the dynamo scenario where the PyTorch version is 2.7 or later is not supported.

## Quick Start

The following uses a simple example to describe how to use msProbe to collect precision data in the SGLang framework.

1. Create a configuration file.

    Create a `config.json` file in the current directory to configure dump parameters. The following is an example:

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

    For details about **config.json**, see [Configuration File Introduction](./config_json_introduct.md).

2. Enable msProbe in the SGLang framework.
  
    Find the file of the `ModelRunner` class in the SGLang framework: **sglang/srt/model_executor/model_runner.py**.

    - Add the `PrecisionDebugger` API to the `__init__` method of the `ModelRunner` class and pass the path of the `config.json` file.

      ```text
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

    - Add `start`, `stop`, and `step` APIs to the `forward` method of the `ModelRunner` class.
        
      Default scenario of the SGLang framework:

      ```text
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

      If DP is enabled in the SGLang framework (`--dp-size` > 1), configure `rank_id` in the `start` API.

      ```text
       if hasattr(self, 'debugger'):
          self.debugger.start(model=self.model, rank_id=self.gpu_id)
      ```

3. Start model inference in the SGLang framework and start collecting data.

    - Online mode
        1. Launch the server.

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

        2. Send a request to automatically start data dump.

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

    - Offline mode
      
      The following is an example of the offline script. When the script is executed, data dump automatically starts.

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

## Data Collection

SGLang uses the same precision data collection functions and data structure as PyTorch. For more details, see [Precision Data Collection in PyTorch](./pytorch_data_dump_instruct.md#data-collection).
