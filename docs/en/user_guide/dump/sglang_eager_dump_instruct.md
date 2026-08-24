# Precision Data Collection in SGLang (SGLang Version < 0.5.11)

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-13T02:00:21.209Z pushedAt=2026-08-13T02:12:06.192Z -->

## Introduction

The msProbe tool collects precision data during model execution by adding the `PrecisionDebugger` interface to the core class `ModelRunner`, which is responsible for model forward propagation execution in the SGLang framework, and starting inference.

For the performance overhead of the dump "statistics" mode and the amount of data collected in the "tensor" mode, see [dump baseline](../../baseline/pytorch_data_dump_perf_baseline.md).

**Notes**

* This document applies only to data collection for **SGLang version <0.5.11**, which requires enabling the msProbe tool capability through invasive modification of the SGLang source code. For SGLang version >=0.5.11, msProbe is built in natively, and you can directly specify the `--msprobe-dump-config` parameter for precision data collection. For details, see *[Precision Data Collection in SGLang (SGLang Version >=0.5.11)](./sglang_eager_dump_instruct_new.md)*.
* Before collecting data, you need to specify the `--disable-cuda-graph` parameter of the SGLang framework to disable graph mode.
* When collecting data in the online mode of the SGLang framework, you need to specify the `--skip-server-warmup` parameter of the SGLang framework to disable warmup, so as to avoid collecting data from the warmup stage.
* If you encounter dynamo-related errors, you can set the environment variable `export TORCHDYNAMO_DISABLE=1` to globally disable dynamo.
* When collecting data in the PD disaggregation mode of the SGLang framework, the Router sends a `/health` request upon startup, and `/health` triggers model forward. You need to set the environment variable `export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`, so that the `/health` request only returns `200` and does not trigger model forward. This prevents msProbe from collecting data from the unnecessary `/health` request stage.
* This tool provides a fixed API support list. If you need to delete or add APIs for dump, you can manually modify the [support_wrap_ops.yaml](../../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml) file, as shown in the following example:

  ```yaml
  functional:  # Operator category. Find the corresponding category, and under that category, delete or add APIs in the following format.
    - conv1d
    - conv2d
    - conv3d
  ```

Scenario for deleting APIs: Some model code logic performs native API type validation. When the tool performs a dump operation, the API wrapper applied to a model may be inconsistent with the model's native API type, which may cause validation failure. For details, see [FAQs](../../support/faq.md).

## Preparations Before Use

**Environment Setup**

Install msProbe by referring to [*msProbe Installation Guide*](../../install_guide/msprobe_install_guide.md).

**Constraints**

Only models implemented based on the PyTorch framework are supported for data collection. The dynamo scenario for PyTorch version >= 2.7 is not supported yet.

## Quick Start

The following uses a simple example to show how to use msProbe for precision data collection in the SGLang framework.

1. Create a configuration file.

    Create a `config.json` file in the current directory to configure the dump parameters.

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

    For details about the `config.json` file, see *[Configuration File Introduction](./config_json_introduct.md)*.

2. Enable msProbe in the SGLang framework.

    Locate the file containing the `ModelRunner` class in the SGLang framework: `sglang/srt/model_executor/model_runner.py`.

    - Add the `PrecisionDebugger` interface in the `__init__` method of the `ModelRunner` class, passing in the path to the `config.json` file.

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

    - Add the `start`, `stop`, and `step` interfaces in the `forward` method of the `ModelRunner` class.

      Default scenario of the SGLang framework:

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

    To enable the DP scenario in the SGLang framework (`--dp-size` > 1), configure the `rank_id` parameter in the `start` interface:

    ```python
     if hasattr(self, 'debugger'):
        self.debugger.start(model=self.model, rank_id=self.gpu_id)
    ```

3. Start model inference in the SGLang framework and begin data collection.

    - Online mode
        1. Start the service.

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

        2. Send a request to automatically start the dump.

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

      An example offline script is shown below. Running it automatically starts the dump:

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
              "The future of AI is"
          ]
        
          sampling_params = {"temperature": 0.8, "top_p": 0.95}
        
          outputs = llm.generate(prompts, sampling_params)
          for prompt, output in zip(prompts, outputs):
              print("===============================")
              print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        
        
      if __name__ == '__main__':
          main()
      ```

## Data Collection in PD Disaggregation Scenario

The following uses a simple example to show how to use msProbe for precision data collection in the **PD disaggregation scenario** of the SGLang framework.

1. Create configuration files.
   
    Create configuration files in the current directory to configure the dump parameters.

    - To collect data from both the prefill and decode stages, create `config_prefill.json` and `config_decode.json`. The `dump_path` in the two JSON configuration files must be different to avoid dump write conflicts.
    - To collect only the prefill stage data, create `config_prefill.json`.
    - To collect only the decode stage data, create `config_decode.json`.

    The content example is as follows:

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

    For a detailed introduction to the configuration files, see *[Configuration File Introduction](./config_json_introduct.md)*.

2. Enable msProbe in the SGLang framework.

    1. Locate the file containing the `ModelRunner` class in the SGLang framework: `sglang/srt/model_executor/model_runner.py`.

       Add the `PrecisionDebugger` interface in the `__init__` method of the `ModelRunner` class, passing in the path to the `config_prefill.json` or `config_decode.json` file.

       Based on the passed-in configuration file, you can choose to collect the data of `prefill and decode stages`, `prefill stage only`, or `decode stage only`.

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
               config_path = ""
               disagg_mode = server_args.disaggregation_mode
               if disagg_mode == "prefill":
                   config_path = "./config_prefill.json" # To skip collecting the prefill stage, change it to config_path = ""
               elif disagg_mode == "decode":
                   config_path = "./config_decode.json" # To skip collecting the decode stage, change it to config_path = ""
               if config_path:
                   self.debugger = PrecisionDebugger(config_path=config_path)
               ################################ msprobe ################################
               # Parse args
               self.mem_fraction_static = mem_fraction_static
               self.device = server_args.device
               self.gpu_id = gpu_id
               self.tp_rank = tp_rank
               self.tp_size = tp_size
             ...
       ```

       Add the `start`, `stop`, and `step` interfaces to the `forward` method of the `ModelRunner` class. For details, refer to [Quick Start](#quick-start).

    2. When the SGLang framework enables the DP scenario (`--dp-size` > 1), the `bootstrap_room` value must be fixed.

       - Background Description

         After SGLang enables PD disaggregation deployment, the prefill stage forcibly adopts the `follow_bootstrap_room` scheduling rule, which relies on `bootstrap_room` to assign requests to different ranks. The scheduling rule is `target_dp_rank = bootstrap_room % dp_size`. The default SGLang Router service automatically generates the `bootstrap_room` value randomly, and this value currently cannot be configured through startup parameters. When `dp-size` > 1 and multiple requests are processed in a single batch, the random value causes requests to be assigned to different ranks in a disordered manner, which leads to inconsistent data collection across multiple runs and unreproducible experiments.

         Therefore, it is necessary to manually modify `bootstrap_room` to a fixed value, lock the rank assignment for requests, unify the scheduling rule, and ensure fixed data collection and reproducible results.

       - How to Do

         Locate the SGLang framework source file `sglang/srt/managers/io_struct.py`, and reset the `bootstrap_room` value (considering that the source code uses `random.randint(0, 2**63 - 1)` to generate a 19-digit integer, a fixed 19-digit integer is used here as well).

         ```python
         @dataclass
         class GenerateReqInput(BaseReq):
             ...
         
             def _normalize_bootstrap_params(self, num):
                 """Normalize bootstrap parameters for batch processing."""
         
                 ...
         
                 # Normalize bootstrap_room
                 ################################ msprobe ################################
                 self.bootstrap_room = 6347036608774465186
                 ################################ msprobe ################################
                 if self.bootstrap_room is None:
                     self.bootstrap_room = [None] * num
                 elif not isinstance(self.bootstrap_room, list):
                     self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
                 elif isinstance(self.bootstrap_room, list):
                     self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num
         
                 ...
         
         ```

3. Start model inference in the PD disaggregation scenario of the SGLang framework and begin data collection. The following example shows the operation on an Ascend device.

    1. Start the prefill service.

        When using msProbe for dump, you need to add `export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`, `--disable-cuda-graph`, and `--skip-server-warmup`. The example is as follows:

        ```shell
        # Enabling CPU Affinity
        export SGLANG_SET_CPU_AFFINITY=1
        
        # When using msProbe to dump, avoid the Router sending health check requests that trigger model forward
        export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
        
        # PIP: recommended to config first Prefill Server IP
        # PORT: one free port
        # all sglang servers need to be config the same PIP and PORT,
        export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
        # if you use Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
        export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
        python3 -m sglang.launch_server \
            --model-path /home/models/Qwen2.5-0.5B-Instruct \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port 8995 \
            --attention-backend ascend \
            --device npu \
            --base-gpu-id 0 \
            --tp-size 1 \
            --host 127.0.0.1 \
            --port 8000 \
            --disable-cuda-graph \
            --skip-server-warmup
        ```

    2. Start the decode service.

        When using msProbe to dump, you need to add `export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`, `--disable-cuda-graph`, and `--skip-server-warmup`. An example is as follows:

        ```shell
        # When using msProbe to dump, avoid the Router sending health check requests that trigger model forward
        export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
        
        # PIP: recommended to config first Prefill Server IP
        # PORT: one free port
        # all sglang servers need to be config the same PIP and PORT,
        export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
        # if you use Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
        export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"
        python3 -m sglang.launch_server \
            --model-path /home/models/Qwen2.5-0.5B-Instruct \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend ascend \
            --attention-backend ascend \
            --device npu \
            --base-gpu-id 1 \
            --tp-size 1 \
            --host 127.0.0.1 \
            --port 8001 \
            --disable-cuda-graph \
            --skip-server-warmup
        ```

    3. Start the router.

        Example:

        ```shell
        python3 -m sglang_router.launch_router \
            --pd-disaggregation \
            --policy cache_aware \
            --prefill http://127.0.0.1:8000 8995 \
            --decode http://127.0.0.1:8001 \
            --host 127.0.0.1 \
            --port 6688
        ```

    4. Send a request to automatically start the dump.

        Example:

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
        http://127.0.0.1:6688/v1/chat/completions
        ```

## Data Collection Overview

The detailed features of precision data collection and the collected dump data structure in the SGLang scenario are consistent with the PyTorch scenario. For details, see *[Precision Data Collection in PyTorch](./pytorch_data_dump_instruct.md#data-collection)*.
