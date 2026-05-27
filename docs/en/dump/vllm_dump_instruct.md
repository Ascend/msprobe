# Precision Data Collection in the vLLM Scenario

## Overview

msProbe can collect intermediate precision data during model execution in the vLLM inference scenario.

The usage methods for different vLLM scenarios are as follows:

* For vLLM Ascend, see [msProbe Debugging Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html).
* For the native vLLM, add the `PrecisionDebugger` API to `GPUModelRunner` and start inference to collect data. For details, see the following sections.

For details about performance changes in dump statistics mode and data volume collected in dump tensor mode, see [Dump Baseline] (../baseline/pytorch_data_dump_perf_baseline.md).

**Note**:

* Currently, data can be collected only in vLLM eager mode. When starting the service, you need to specify the `--enforce-eager` parameter.
* In the vLLM-Ascend scenario, you are advised to directly reuse the capabilities that have been integrated in the official document and pass `dump_config_path` through `--additional-config`.
* `PrecisionDebugger` should be initialized as early as possible. It is recommended that the initialization be performed before the model execution logic in `GPUModelRunner.__init__` starts to ensure that related APIs can be encapsulated by the tool.
* Typically, `execute_model` takes a long time. You are advised to wrap the logic after `start` in a `try...finally` block, and call `stop` and `step` inside the `finally` clause. If wrapping the logic is inconvenient, you may also call `stop` and `step` before each `return` statement. Otherwise, result files such as `dump.json` may become incomplete.
* To collect `L0` or `mix` data, the `model` parameter needs to be passed to `start`. To collect only `L1` data, you can directly call `start()`.
* If a dynamo-related error occurs, set `export TORCHDYNAMO_DISABLE=1` on NPU to disable dynamo globally.
* The tool provides a fixed API support list. If you need to delete or add dump APIs,
  manually modify [support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml). The following is an example:

  ```yaml
  functional:  # functional indicates the operator type. Find the corresponding type and delete or add APIs in the following format:
    - conv1d
    - conv2d
    - conv3d
  ```

  Scenario where an API is deleted: The code logic of some models may involve native API type verification. When the tool performs the dump operation, the API type encapsulated by the model may be different from the native API type. In this case, the verification may fail. For details, see [Exceptions](../faq.md#exception-handling) in *FAQs*.

## Preparations

**Environment Preparation**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

**Constraints**

Only data of models implemented based on the PyTorch framework can be collected. Currently, the dynamo scenario where the PyTorch version is 2.7 or later is not supported.

## Quick Start

The following describes how to use the tool in two vLLM scenarios.

### vLLM-Ascend

In this scenario, you can directly use msProbe capabilities by simply using `--additional-config` to pass the dump configuration file path. The following is an example provided in the official document:

```shell
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{"dump_config_path": "/data/msprobe_config.json"}'
```

Note:

* The official document can be found on the vLLM-Ascend latest documentation page at the following link:
  `https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html`
* This method is applicable to vLLM Ascend that has integrated msProbe capabilities. You do not need to manually modify the `GPUModelRunner` implementation.

### Native vLLM

The following uses a simple example to show how to use msProbe to collect precision data in `GPUModelRunner` of the native vLLM framework.

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

    For details about the `config.json` file and configuration examples, see [Configuration File Introduction](./config_json_introduct.md).

2. Enable msProbe in the vLLM framework.

    Find the file to which the `GPUModelRunner` class of the vLLM framework belongs, for example, `vllm/v1/worker/gpu_model_runner.py`.

    - Add `PrecisionDebugger` to the `__init__` method of `GPUModelRunner` and pass the `config.json` path.

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

    - Add `start`, `stop`, and `step` APIs to the `execute_model` method of `GPUModelRunner`.

      `execute_model` typically takes a long time. You are advised to use `try...finally` to handle the end logic in a unified manner. This approach requires minimal code modification and is less likely to miss a `return` branch.

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

      If using `try...finally` to wrap the entire code block is inconvenient, you can also add the following call before each corresponding `return` statement.

      ```text
      if hasattr(self, "debugger"):
          self.debugger.stop()
          self.debugger.step()
      return output
      ```

3. Start the vLLM service and collect data.

    When starting the service, you need to enable the eager mode. The following is an example:

    ```shell
    #!/bin/bash
    export TORCHDYNAMO_DISABLE=1
    
    vllm serve Qwen/Qwen2.5-0.5B-Instruct \
      --dtype float16 \
      --enforce-eager \
      --host 0.0.0.0 \
      --port 8000
    ```

    After the service is started, send an inference request. During the request execution, dump is automatically triggered.

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

## Data Collection

vLLM uses the same precision data collection functions and data structure as PyTorch. For more details, see [Precision Data Collection in PyTorch](./pytorch_data_dump_instruct.md#data-collection).
