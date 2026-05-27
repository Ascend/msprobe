# SGLang精度数据采集（SGLang版本>=0.5.11）

## 简介

msProbe工具通过在SGLang框架中负责模型前向传播执行的核心类`ModelRunner`中添加`PrecisionDebugger`接口并启动推理的方式，
采集模型在运行过程中的精度数据。

dump "statistics"模式的性能膨胀大小与"tensor"模式采集的数据量大小，可以参考[dump基线](../baseline/pytorch_data_dump_perf_baseline.md)。

**注意**：

* 本文档仅适用于**SGLang版本>=0.5.11**的数据采集，此版本及更高版本已原生内置msProbe工具，可直接指定参数`--msprobe-dump-config`进行精度数据采集，请参考SGLang官方文档《[MSProbe Debugging Guide](https://docs.sglang.io/docs/developer_guide/msprobe_debugging_guide)》。
若当前SGLang版本<0.5.11，需通过侵入式修改SGLang源码的方式开启msProbe工具能力，具体操作请参考文档《[SGLang精度数据采集（SGLang版本<0.5.11）](./sglang_eager_dump_instruct.md)》。
* 如果遇到dynamo相关报错，可设置环境变量`export TORCHDYNAMO_DISABLE=1`全局关闭dynamo。
* 使用SGLang框架的PD分离模式采集数据，Router启动时会发送`/health`请求，`/health`会触发模型forward，需要设置环境变量`export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`，这样`/health`请求只返回`200`，不会触发模型forward，避免msProbe采集不需要的`/health`请求阶段的数据。
* 本工具提供固定的API支持列表，若需要删除或增加dump的API，可以在[support_wrap_ops.yaml](../../../python/msprobe/pytorch/dump/api_dump/support_wrap_ops.yaml)文件内手动修改，如下示例：
  
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

    config.json文件详细介绍请参见《[配置文件介绍](./config_json_introduct.md)》。

2. SGLang框架中使能msProbe工具

   `SGLang`已原生内置msProbe工具，启动服务时可直接通过`--msprobe-dump-config`传入dump配置文件路径。官方文档当前给出的示例如下：
   
   ```shell
   python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 1027 \
    --msprobe-dump-config /home/msprobe-config.json
   ```

   说明：

   * SGLang的msProbe操作指南官方文档链接为：`https://docs.sglang.io/docs/developer_guide/msprobe_debugging_guide`
   * 该方式适用于已集成msProbe能力的`SGLang`，无需再手工修改`ModelRunner`实现。
   * 启动服务后，发送请求，自动开始dump。

## PD分离场景数据采集

以下通过一个简单的示例，展示如何在SGLang框架**PD分离场景**中使用msProbe工具进行精度数据采集。

1. 配置文件创建

    在当前目录下创建config文件，用于配置dump参数。

    - 采集prefill和decode阶段的数据，则创建`config_prefill.json`和`config_decode.json`。其中两个json配置文件的"dump_path"要不同，避免dump写入冲突。
    - 仅采集prefill阶段的数据，则创建`config_prefill.json`。
    - 仅采集decode阶段的数据，则创建`config_decode.json`。

    内容示例如下：

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

    config文件详细介绍请参见《[配置文件介绍](./config_json_introduct.md)》。

2. SGLang框架中使能msProbe工具
  
    指定配置文件路径`--msprobe-dump-config`。具体可参考《[For MindStudio-probe(msProbe) dump](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md#for-mindstudio-probemsprobe-dump)》。
  
3. 启动SGLang框架PD分离场景模型推理，开始采集数据。以下示例基于NPU设备。

    - 启动Prefill服务
      
      使用msProbe工具dump时，需要指定`--msprobe-dump-config`，并添加`export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`。示例如下：

      ```shell
      # Enabling CPU Affinity
      export SGLANG_SET_CPU_AFFINITY=1
      
      # 使用msProbe工具dump时，避免Router发送健康检查请求触发模型forward
      export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
      
      # PIP: recommended to config first Prefill Server IP
      # PORT: one free port
      # all sglang servers need to be config the same PIP and PORT,
      export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
      # if you are Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
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
      
    - 启动Decode服务
    
      使用msProbe工具dump时，需要指定`--msprobe-dump-config`，并添加`export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`。示例如下：

      ```shell
      # 使用msProbe工具dump时，避免Router发送健康检查请求触发模型forward
      export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
      
      # PIP: recommended to config first Prefill Server IP
      # PORT: one free port
      # all sglang servers need to be config the same PIP and PORT,
      export ASCEND_MF_STORE_URL="tcp://PIP:PORT"
      # if you are Atlas 800I A2 hardware and use rdma for kv cache transfer, add this parameter
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
      
    - 启动路由
    
      示例如下：

      ```shell
      python3 -m sglang_router.launch_router \
          --pd-disaggregation \
          --policy cache_aware \
          --prefill http://127.0.0.1:8000 8995 \
          --decode http://127.0.0.1:8001 \
          --host 127.0.0.1 \
          --port 6688
      ```
    
    - 发送请求，自动开始dump
    
      示例如下：

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

## 数据采集功能介绍

SGLang精度数据采集详细功能以及采集的dump数据结构与PyTorch场景一致，具体请参见《[PyTorch场景精度数据采集](./pytorch_data_dump_instruct.md#数据采集功能介绍)》。
