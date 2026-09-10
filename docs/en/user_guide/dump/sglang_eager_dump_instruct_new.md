# Precision Data Collection in SGLang (SGLang Version >= 0.5.11)

<!-- md-trans-meta sourceCommit=39b2fc369098168c030fb85ace3bf05ce60a34c5 translatedAt=2026-08-11T02:42:49.202Z pushedAt=2026-08-11T02:51:07.156Z -->

## Introduction

The msProbe tool collects precision data during model execution by adding the `PrecisionDebugger` interface to the `ModelRunner` class, which is the core class responsible for model forward propagation in the SGLang framework, and then launching inference.

For the performance inflation of the dump "statistics" mode and the data volume collected in the "tensor" mode, refer to the [dump baseline](../../baseline/pytorch_data_dump_perf_baseline.md).

**Notes**

* This document applies only to data collection for **SGLang versions >= 0.5.11**, which have msProbe natively built in, allowing you to directly specify the `--msprobe-dump-config` parameter for precision data collection. See the SGLang official documentation *[MSProbe Debugging Guide](https://docs.sglang.io/docs/developer_guide/msprobe_debugging_guide)*.
If the current SGLang version is earlier than 0.5.11, you need to enable msProbe by intrusively modifying the SGLang source code. For detailed operations, refer to *[Precision Data Collection in SGLang (SGLang Version < 0.5.11)](./sglang_eager_dump_instruct.md)*.
* If you encounter dynamo-related errors, you can set the environment variable `export TORCHDYNAMO_DISABLE=1` to globally disable dynamo.
* When collecting data in PD disaggregation mode of the SGLang framework, the Router sends a `/health` request upon startup, which triggers model forward. You need to set the environment variable `export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` so that the `/health` request only returns `200` without triggering model forward. This prevents msProbe from collecting unnecessary data from the `/health` request phase.
* This tool provides a fixed list of supported APIs. If you need to delete or add APIs for dumping, you can manually modify the [support_wrap_ops.yaml](../../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml) file, as shown in the following example:

  ```yaml
  functional:  # Operator category. Find the corresponding category, and delete or add APIs in that category using the following format.
    - conv1d
    - conv2d
    - conv3d
  ```

Scenarios for deleting APIs: Some model code logic includes native API type validation. When the tool performs a dump operation, the API encapsulation for a model may differ from the model's native API type, which may cause validation failures. For details, see [FAQs](../../support/faq.md).

## Preparations Before Use

**Environment Setup**

Install msProbe by referring to [*msProbe Installation Guide*](../../install_guide/msprobe_install_guide.md).

**Constraints**

Only models implemented based on the PyTorch framework are supported for collection. The dynamo scenario for PyTorch version >= 2.7 is not supported at this time.

## Quick Start

1. Create a configuration file.

    Create a `config.json` file in the current directory to configure dump parameters. The following is an example of its content:

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

    For a detailed introduction to the `config.json` file, see *[Configuration File Introduction](./config_json_introduct.md)*.

2. Enable msProbe in the SGLang framework.

   SGLang has natively integrated the msProbe tool. When starting the service, you can directly pass the dump configuration file path via `--msprobe-dump-config`. The official documentation currently provides the following example:

   ```shell
   python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 1027 \
    --msprobe-dump-config /home/msprobe-config.json
   ```

   **Note**

   * Official documentation for the SGLang msProbe operation guide: `https://docs.sglang.io/docs/developer_guide/msprobe_debugging_guide`
   * This method applies to `SGLang` that has already integrated the msProbe capability, eliminating the need to manually modify the `ModelRunner` implementation.
   * After the service is started, sending a request automatically begins the dump.

## Data Collection in PD Disaggregation Scenario

The following simple example demonstrates how to use msProbe for precision data collection in the **PD disaggregation scenario** of the SGLang framework.

1. Create configuration files.

    Create configuration files in the current directory to configure dump parameters.

    - To collect data from both the prefill and decode phases, create `config_prefill.json` and `config_decode.json`. The `dump_path` values in the two JSON configuration files must be different to avoid dump write conflicts.
    - To collect only the prefill phase data, create `config_prefill.json`.
    - To collect only the decode phase data, create `config_decode.json`.

    Example:

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

    For a detailed introduction to configuration files, see *[Configuration File Introduction](./config_json_introduct.md)*.

2. Enable msProbe in the SGLang framework.

    Specify the configuration file path `--msprobe-dump-config`. For details, see *[For MindStudio-probe(msProbe) dump](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/server_arguments.mdx#for-mindstudio-probemsprobe-dump)*.

3. Start model inference in the PD disaggregation scenario of the SGLang framework to collect data. The following example shows the operation on an NPU device.

    - Start the prefill service.

      When using msProbe for dump, you need to specify `--msprobe-dump-config` and add `export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`.

      ```shell
      # Enabling CPU Affinity
      export SGLANG_SET_CPU_AFFINITY=1
      
      # When using msProbe for dump, prevent the Router from sending health check requests that trigger model forward.
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
          --msprobe-dump-config your_path/config_prefill.json
      ```

    - Start the decode service

      When using msProbe for dump, you need to specify `--msprobe-dump-config` and add `export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`.

      ```shell
      # When using msProbe for dump, prevent the Router from sending health check requests that trigger model forward
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
          --msprobe-dump-config your_path/config_decode.json
      ```

    - Start the Router.

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

    - Send a request to automatically start dump.

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
