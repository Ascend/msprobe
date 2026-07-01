# 大模型推理精度定位指南

## 引言

大语言模型在昇腾NPU上的推理精度保障，旨在确保模型能力与GPU保持语义等价，通常通过下游任务评估或与GPU输出对比来衡量。实践中，精度问题常源于模型结构错误、参数配置不当或算子计算偏差，表现为胡言乱语、输出中断、重复生成或结果偏离等异常现象。大模型推理中的精度问题可分为**模型精度问题**与**数值精度问题**两大类别，二者成因与影响程度各异，需采取差异化的分析与处置策略。

**模型精度问题**属于“结构性偏差”，涵盖数据加载异常、训练超参配置失误、网络结构实现错误或框架本身的设计缺陷。此类问题对模型收敛具有决定性影响，必须逐环节严密排查，结合实际场景进行针对性调整。

**数值精度问题**则属于“计算性偏差”，源于浮点运算的固有特性——有限字长效应对表示范围的限制、计算与通信顺序差异、以及数学表达式的近似处理均会引入误差。需明确的是，数值计算的近似性虽可能干扰收敛过程，但计算差异并不必然导致收敛失败。算子作为计算的基础单元，其数值精度固然是关键考量，然而由于硬件架构差异（如GPU与CPU之间、不同GPU版本之间），相同计算逻辑产生细微数值偏差属正常现象，只要控制在合理容限内，便不会影响模型最终收敛。

为有效甄别正常计算差异与异常精度问题，精准定位根因，本指南系统梳理了精度问题定位工具集的适用场景与操作流程，助力用户自主或在技术文档指导下高效排查潜在风险。

## 1. 推理常见精度问题现象

大语言模型推理的精度问题，常体现在模型输出的结果不符合预期，具体现象可以分为以下几类：

1. 模型回答乱码：输出中出现大量 �、\<unk\>、â€™ 等异常符号，或突然插入一段其他语言的字符。

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/78a354cb-eec8-4670-aa11-39893d30b635/image.png 'image.png')

2. 模型回答重复：模型在某个局部卡住，不断重复相同或相似的文本片段。

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/f1f550d1-6c47-4cd5-a3f2-13b654544ebf/image.png 'image.png')

3. 回答语义断裂与逻辑崩坏：模型输出的文字在局部可能是通顺的，但整体逻辑无法连贯，或者推理链条中断。

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/5725d07b-d5c1-46f4-a2a7-2881e91409ae/image.png 'image.png')

4. 不一致与抖动类：多次请求之间，输出结果差异巨大。

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/657ad9a7-d3a9-4e98-b5b0-da8a83fd1d8d/image.png 'image.png')

5. 数据集评测不达标类：数据集评测结果相较于标杆，正确率下降。

    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/e831b0d3-1f41-4f1b-9055-47731a4f9495/image.png 'image.png')

## 2. 模型推理常见精度问题总结

大语言模型推理常见精度问题，可以分为精度误差和实践错误，其中在推理部署中当前精度问题主要以实践错误为主。实践错误原因主要包括：

1. **模型配置**

   权重问题、模型参数配置问题等。

   1. 权重错误会影响wordembedding词表、linear、layerNorm、lmhead。

   2. 参数配置包括padding方式，模型config设置，例如pad_token_id、eos_token_id、max sequence等，可以通过对齐config来规避错误。

2. **模型结构错误（代码实现错误等）**

   一般会导致较明显的精度错误，使得模型输出无序英文或胡言乱语，或无法正常输出。

3. **算子传参错误**

   1. 例如attention mask传入错误。Attention mask传入错误，模型一般不会输出无序英文，且有时能回答问题，但可能输出大量空格，且回答无法与GPU对齐。

   2. RoPE旋转系数传入错误。该场景错误一般会导致模型直接无序输出。

4. **算子实现**

   1. 后处理算子实现问题，一般体现为模型greedy search时精度正常，增加后处理采样策略后输出不对，例如输出重复或接近greedy search。

   2. 模型侧算子问题，若单算子检测正常，模型侧异常，可能为多算子之间内存踩踏、寄存器未复位等原因导致。可以在开启确定性计算后，多次问同一个问题，若运行结果不一致可能为算子实现问题。

5. **环境版本缺陷或差异**

   一般体现为在某一类机器上精度正常，在其他机器或环境上突然异常，或更换环境版本后精度异常。例如x86换到arm架构出现精度异常，或更换CANN后精度异常。解决方案可以对齐环境依赖版本。

## 3. 模型精度问题定位思路

大模型推理整体精度定位流程如图所示：

![image.png](https://raw.gitcode.com/user-images/assets/7898473/79fb6a89-4944-4044-85a9-f2edd6a4098f/image.png 'image.png')

### 3.1 检查checklist

在有标杆对照的场景下，与标杆场景完成对齐配置检查至关重要。有标杆对照场景下的大模型推理，主要分为两大场景：

1. 相同模型NPU推理与GPU推理（GPU视为标杆）。
2. 相同模型NPU推理与历史基线版本NPU推理（历史基线视为标杆）。

针对上述两大场景，可以统一泛化为，**问题场景**与**标杆场景**的对照。后面主要以这两个词来描述。检查checklist，主要涉及以下几个排查点。

- **推理超参和环境变量比对**

可以使用Beyond Compare软件比对双方推理日志或启动脚本中的推理超参和环境变量设置。

- **三方库版本比对**

通过git分支检查MindIE、vLLM、Transformers等三方库版本是否与标杆对齐。

通过pip list检查PyTorch、torch-npu等第三方库版本是否与标杆对齐。

- **数据读取检查** 

检查模型推理的输入数据，一般可通过直接在代码中打印进行输入数据检查（如vLLM可以直接在llm.generate的输出中，通过output.prompt获取原始输入文本）。

- **模型配置检查**

在问题场景和标杆场景之间打印模型配置比对。

使用示例

```python
from vllm import LLM

#初始化模型
llm = LLM(
    model="/path/to/your/model",  # 或 Hugging Face 模型名
    dtype="float16"
)

#查看模型配置
print("模型配置 (Model Config)")
print("=" * 50)
print(llm.llm_engine.model_config)
```

输出示例：

```python
模型配置 (Model Config)
==================================================
ModelConfig(model='meta-llama/Llama-2-7b-hf', tokenizer='meta-llama/Llama-2-7b-hf', tokenizer_mode=auto, trust_remote_code=True, dtype=torch.float16, seed=0, skip_tokenizer_init=False, use_v2_block_manager=False, ...
```

### 3.2 问题复现前置操作

在确保上述配置环境等信息对齐后，需要进行问题复现，为了保证问题定位过程中的变量尽可能小，需要进行随机固定以及使能算子确定性。

#### 3.2.1 固定随机性

复现需要固定存在随机性的步骤，保证实验可重复性。存在随机性的步骤包括模型参数初始化，dropout层等。
涉及到的操作如下几项：

- 固定随机种子，如np.random.seed、torch.manual_seed、torch_npu.npu.manual_seed等。
- 关闭Dropout层。

#### 3.2.2 打开确定性

复现时建议打开算子计算确定性和通信确定性，两者都需要在训练开始的代码之前，尽早进行固定，具体可通过以下两项设置：

- 算子计算确定性：
`torch.use_deterministic_algorithms(True)`
- 通信确定性：
`export HCCL_DETERMINISTIC=TRUE`

针对以上**固定随机性**和**打开确定性**操作，msprobe工具包提供seed_all接口，在pytorch场景下快速固定网络中所有随机种子、Dropout层及算子计算和通信确定性。

使用方式：

```python
from msprobe.pytorch import seed_all
seed_all(seed=1234, mode=True, rm_dropout=True, is_enhanced=False)
```

参数说明：

| 参数名 | 可选/必选 | 说明 |
|--|--|--|
|seed  |  可选  |随机数种子。默认值为1234。  |
|mode  |  可选  |确定性计算模式。可配置True或False，默认值为False。该模式同时包含算子计算确定性和通信确定性。  |
|rm_dropout |   可选  |控制dropout失效的开关，开启后会自动将dropout概率设置为0。可配置 True 或 False，默认值为True。  |
|is_enhanced|可选|增强随机性固定的开关。可配置True或False，默认为False，非必选。参数示例：is_enhanced=True。开启该功能后，将进一步固定PyTorch、NumPy以及Python内置随机数生成器的状态。在同一个进程或不同进程中多次执行相同的随机性API，每次生成的随机值都完全相同。这有助于在更复杂的随机场景下实现严格的可复现性。|

### 3.3 挑选badcase

在大模型推理精度定位场景下，通常会出现两个模型在同一数据集下，表现不一致的情况。比如，问题场景下，经过数据集评测，发现有个问题出现回答错误的情况，但是同样的这个问题，其标杆场景下的结果却是正确的，这时我们就称这个问题为badcase。

在选取到badcase之后，后续的问题就可以衍变为单case问题进行定位。后续操作会在第四章节精度问题分场景定位中逐步体现。

## 4.  精度问题分场景定位

### 4.1 单case可复现精度问题

在大模型推理中，单case可复现精度问题，这通常指一个极其稳定的badcase。具体表现为：

输入固定：给定一个特定的Prompt（例如：“中国的首都是哪里？”）。

参数固定：Temperature=0，Top_p=1，Seed固定，Max Tokens相同。

输出结果每次都错：比如，模型每次都会把正确答案“北京”回答成错误答案“上海”。

#### 4.1.1 vLLM场景精度问题定位

vLLM 是由加州大学伯克利分校团队开发的高性能大模型推理框架，通过创新的显存管理和调度策略，解决了传统推理框架在部署大模型时面临的显存利用率低、吞吐量不足、并发处理效率低等问题。vLLM的核心优势在于其独特的PagedAttention显存管理机制和连续批处理技术，这两项创新使显存利用率提升至接近100%，吞吐量可达传统框架的24倍，特别适合高并发、低延迟的实时推理场景。vLLM的推理流程分为两个主要阶段：prefill阶段和decode阶段。prefill阶段处理输入提示词，生成初始的KV Cache；decode阶段则逐个生成输出token，持续更新KV Cache。整个过程中，vLLM通过其PagedAttention机制和Continuous Batching技术，实现了对显存资源的高效利用和对计算资源的充分调度。

##### 4.1.1.1 常用工具介绍

vLLM场景的精度问题定位，主要使用msprobe工具下的dump和比对能力进行问题定位。由于vLLM涉及多种拉起方式，以vLLM0.9版本为例下面逐一介绍各种拉起方式下的工具使能：

###### 4.1.1.1.1 V0场景

- **V0，离线模式，TP=1**

可直接通过以下方式获取model
`model=llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()`

![image.png](https://raw.gitcode.com/user-images/assets/7898473/20631717-1da6-426d-8d03-52a84a0ed29c/image.png 'image.png')

config 配置文件，具体字段的含义参见[config介绍](../user_guide//dump/config_json_introduct.md)，msprobe接口介绍参见[PyTorch场景精度数据采集](../user_guide/dump/pytorch_data_dump_instruct.md)，可以通过在start接口中设置token_range参数来控制需要采集的token数据。

- **V0，离线模式，TP>1**

使用多进程执行器MultiprocessingDistributedExecutor，存在进程间隔，0卡在主进程，其他卡在子进程

加工具位置：0卡采集可直接加在LLM调用generate的最外层，但其他卡采集需在子进程的_run_worker_process函数中vllm/executor/multiproc_worker_utils.py

![image.png](https://raw.gitcode.com/user-images/assets/7898473/28a7eec9-1a46-4d7a-a3e7-91237f55b1fb/image.png 'image.png')

- **V0,在线模式，PP=1（TP、DP不限制且不设置--disable-frontend-multiprocessing)**

使用多进程客户端MQLLMEngineClient，存在进程间隔，都在子进程里

加工具位置：子进程MQLLMEngine类的run_engine_loop函数中（vllm/engine/multiprocessing/engine.py）

![image.png](https://raw.gitcode.com/user-images/assets/7898473/7e7f693f-e67f-4832-ae66-8afaa113069e/image.png 'image.png')

- **V0，在线模式，PP>1或设置--disable-frontend-multiprocessing**

使用多进程执行器MultiprocessingDistributedExecutor，存在进程间隔，0卡在主进程，其他卡在子进程

加工具位置：等同于V0 在线模式 TP>1

###### 4.1.1.1.2 V1场景

- **v1 engine，eager（enforce_eager=True)**

**1. 添加初始化**

NPU -> 在model_runner_v1.py中添加：（位置 vllm_ascend/worker/model_runner_v1.py NPUModelRunner.init函数）

![image.png](https://raw.gitcode.com/user-images/assets/7898473/84fc8c8a-897e-4ac6-b791-aaf55654eda2/image.png 'image.png')

gpu -> vllm/v1/worker/gpu_model_runner.py  GPUModelRunner.init函数

![image.png](https://raw.gitcode.com/user-images/assets/7898473/a8bf2df4-d083-42a4-b146-4d4d5ec9279f/image.png 'image.png')

**2. 添加工具使能代码**

使能开始，按照配置（L0/L1）加入对应代码：

```python
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        # L0 use 
        # self.debugger.start(self.model)
        # L1 use
        self.debugger.start()
```

NPU-> 在model_runner_v1.py中添加：（位置 vllm_ascend/worker/model_runner_v1.py NPUModelRunner.execute_model函数）

![image.png](https://raw.gitcode.com/user-images/assets/7898473/d3105c07-5c6c-430d-8db5-81121fdf91ca/image.png 'image.png')

使能结束（放在execute_model函数return前即可）：
![image.png](https://raw.gitcode.com/user-images/assets/7898473/c12564be-8976-49db-98dc-e09313dda74e/image.png 'image.png')

GPU->vllm/v1/worker/gpu_model_runner.py  GPUModelRunner.execute_model
![image.png](https://raw.gitcode.com/user-images/assets/7898473/aa33f488-d3e1-4d5b-b3cb-439a23bdf0cc/image.png 'image.png')

![image.png](https://raw.gitcode.com/user-images/assets/7898473/2fafa44e-8a21-4488-8762-76e5e21eccfd/image.png 'image.png')

##### 4.1.1.2 定位流程

针对于单case可复现的精度问题的定位流程，可以总结为以下三个阶段：

![image.png](https://raw.gitcode.com/user-images/assets/7898473/dd6bf3db-d722-43da-9df0-1daa297feffb/image.png 'image.png')

###### 4.1.1.2.1 定位前置操作

其中精度标杆在该场景下，可能来自于GPU，也可能来自于历史精度正常版本的NPU基线。

模型配置检查和随机性固定，可以参考 [3.1 检查checklist](#31-检查checklist)和[3.2 问题复现前置操作](#32-问题复现前置操作)，针对于vLLM场景，需要同时设置固定采样随机性：temperature为0
![image.png](https://raw.gitcode.com/user-images/assets/7898473/425c6827-f81f-4c7c-8a5b-5a4ffd8f4a80/image.png 'image.png')

###### 4.1.1.2.2 定位过程操作

- 确认首差异token

可以通过如下方式在vLLM中打印试验case的输出token_id序列，以V1场景为例：
![image.png](https://raw.gitcode.com/user-images/assets/7898473/1151539d-5ad6-494a-b872-8e504eefe829/image.png 'image.png')

增加打印后，就可以很容易比较问题场景与标杆场景的首个差异token位置。

- 使用msprobe dump数据

msprobe dump的使用方式，可以参考[常用工具介绍](#4111-常用工具介绍)章节的常用工具介绍。dump级别推荐优先使用mix级别+statistics模式，可以得到如下形式的dump数据：

![image.png](https://raw.gitcode.com/user-images/assets/7898473/f5c9ea68-a488-41c1-a57a-3166a251edae/image.png 'image.png')

###### 4.1.1.2.3 定位结果分析

完成上述数据dump后，应该会得到问题场景和标杆场景的两份dump数据
可以使用[精度比对工具](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)，进行数据比对，示例如下：
`msprobe compare -tp <target_path> -gp <golden_path> [options]`
比对完成后会生成比对表格，在比对表格中，可以看到四种统计量的diff差异，从而可以依据较大的diff差异，怀疑问题点

![image.png](https://raw.gitcode.com/user-images/assets/7898473/3ed1667c-40e6-44c3-af62-51b39c463f68/image.png 'image.png')

如上图示例，matmul就是问题怀疑点，后续可以单算子复现确认问题

#### 4.1.2 MindIE场景精度问题定位

MindIE（Mind Inference Engine，昇腾推理引擎）是华为昇腾针对AI全场景业务的推理加速套件。通过分层开放AI能力，支撑用户多样化的AI业务需求，使能百模千态，释放昇腾硬件设备算力。MindIE向上支持多种主流AI框架，向下对接不同类型昇腾AI处理器，提供多层次编程接口，帮助用户快速构建基于昇腾平台的推理业务。

当前，MindIE往往与[ATB加速库](https://www.hiascend.com/document/detail/zh/canncommercial/850/acce/ascendtb/ascendtb_0001.html)结合使用，达到最佳的推理性能。下文以MindIE+ATB为例，介绍MindIE场景下的精度问题定位方法。

##### 4.1.2.1 常用工具介绍

主要使用msProbe工具的dump与比对功能进行MindIE场景的精度问题定位。

**工具安装**

当前仅支持通过源码编译的方式获取具有dump ATB模型精度数据能力的msProbe whl安装包。编译安装步骤如下：

```shell
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe

pip install setuptools wheel

python setup.py bdist_wheel --include-mod=atb_probe
cd ./dist
pip install ./mindstudio_probe*.whl
```

需要注意的是

1. 编译环境需安装git、curl、GCC 7.5或以上版本、CMake 3.19.3或以上版本等第三方依赖软件；
2. 若编译过程中因安全证书问题导致编译失败，在保证环境安全的情况下，可临时关闭安全证书验证。关闭证书校验的编译命令为`python setup.py bdist_wheel --include-mod=atb_probe --no-check`。

**工具使用**

1. 创建`config.json`配置文件，用于设置dump参数。配置示例如下：

    ```json
    {
        "task": "tensor",
        "dump_enable": true,
        "exec_range": "all",
        "ids": "",
        "op_name": "",
        "save_child": false,
        "device": "",
        "filter_level": 1
    }
    ```

    **dump配置文件参数说明**

    dump配置文件为JSON格式的文本文件，各配置参数介绍如下：

    | 参数 | 可选/必选 | 说明 |
    | --- | --------- | --- |
    | task         | 可选 | 指定dump任务，str类型，默认为"tensor"。可选值：<br/> "tensor"：采集op的输入/输出Tensor的真实数据；<br/> "statistics"：采集op的输入/输出Tensor的统计量数据；<br/> "all"：采集op的输入/输出Tensor的真实数据与统计量数据。 |
    | dump_enable  | 可选 | 指定是否允许dump数据，bool类型，默认为false。可选值：<br/> true：允许采集op的输入/输出Tensor的真实数据或统计量数据；<br/> false：不允许采集op的输入/输出Tensor的真实数据或统计量数据。 |
    | exec_range   | 可选 | 指定需dump数据的op执行轮次范围，str类型，默认为"0,0"。可选值：<br/> "all"：dump op所有执行轮次的精度数据；<br/> "none"：op所有执行轮次的精度数据都不dump；<br/> "\<起始轮次\>,\<终止轮次\>"： dump op从起始轮次到终止轮次间的精度数据，包括起始轮次与终止轮次。<br/> **配置示例**："exec_range": "0,2"，表示dump op第1、2、3执行时的精度数据（第N次执行的执行轮次为N-1）。|
    | ids          | 可选 | 指定需dump数据的op的ID，str类型，默认为""，表示dump所有layer级Operation的精度数据。需满足"\<ID1\>,\<ID2\>"格式，指定一个或多个ID。<br/> **配置示例**：<br/> "ids": "0"，表示dump ID为0的op的精度数据；<br/> "ids": "2_1"，表示dump ID为2的op下的ID为1的OP的精度数据；<br/> "ids": "0,2_1"，表示dump ID为0的op以及ID为2的op下的ID为1的OP的精度数据。 |
    | op_name      | 可选 | 指定需dump数据的op的名称，str类型，默认为""，表示dump所有layer级Operation的精度数据。需满足"\<opName1\>,\<opName2\>"格式，指定一个或多个op名称。<br/> **配置示例**：<br/> "op_name": "word"，表示dump名称以"word"开头的op的精度数据（不区分大小写）。 |
    | save_child   | 可选 | 指定是否dump op下的子op的精度数据，bool类型，默认为false。可选值：<br/> true：dump 指定op及内部子op的精度数据；<br/> false：仅dump 指定op的精度数据。 |
    | device       | 可选 | 指定需dump数据的device ID，str类型，默认为""，表示dump 所有device上的精度数据。需满足"\<deviceID1\>,\<deviceID2\>"格式，指定一个或多个device ID。<br/> **配置示例**：<br/> "device": "0"，表示dump device0上的精度数据。 |
    | filter_level | 可选 | 指定dump op的输入/输出Tensor的真实数据时的过滤等级，int类型，默认为1。该参数仅在指定layer级Operation，且"save_child"为true时生效。可选值：<br/> 0：采集op的输入/输出Tensor的真实数据时，不进行数据过滤；<br/> 1：采集op的输入/输出Tensor的真实数据时，相同Tensor仅保存一次；<br/> 2：在1基础上，过滤Kernel的输入/输出Tensor。 |

2. 执行ATB dump模块加载脚本。命令示例如下：

    ```bash
    source $MSPROBE_HOME_PATH/msprobe/scripts/atb/load_atb_probe.sh --output=$OUTPUT_PATH --config=$CONFIG_PATH
    ```

    `$MSPROBE_HOME_PATH`为msProbe工具安装路径，`$OUTPUT_PATH`为dump数据输出路径，`$CONFIG_PATH`为dump配置文件路径。

3. 正常进行MindIE推理任务。

4. msProbe工具会自动采集ATB模型运行过程中的精度数据。在得到dump数据后，即可使用msProbe工具的比对功能，对dump数据进行比对分析，比对命令如下：

    ```bash
    msprobe compare -m atb -gp <goldenDataPath> -tp <targetDataPath> [-o <outputPath>]
    ```

    **比对命令参数说明**

    | 参数 | 可选/必选 | 说明 |
    | --- | --------- | --- |
    | -m或--mode         | 必选 | 指定比对场景，必须为atb。 |
    | -gp或--golden_path | 必选 | 指定标杆数据路径，必须指定到执行轮次级目录。 |
    | -tp或--target_path | 必选 | 指定待比对数据路径，必须指定到执行轮次级目录。 |
    | -o或--output_path  | 可选 | 指定比对结果输出路径，默认为当前工作目录下的output目录（工具会自动创建）。 |

详细的dump、比对功能使用介绍请参见《[ATB场景精度数据采集指南](../user_guide//dump/atb_data_dump_instruct.md)》、《[ATB场景精度数据比对指南](../user_guide/accuracy_compare/atb_data_compare_instruct.md)》。

##### 4.1.2.2 定位流程

在不明确精度问题发生点大致位置时，一般可按照“先Layer再OP” 的顺序进行定位，这里的OP包括Layer下的Operation与Kernel。

**Layer级问题定位**

将dump配置文件中的"ids"参数设置为""，"save_child"参数设置为false，即可仅dump所有Layer的输入/输出Tensor，dump输出件示例如下：

![atb_dump_layer.png](https://raw.gitcode.com/user-images/assets/7898473/0e57b323-4901-46c4-b97a-0375eee853ae/atb_dump_layer.png 'atb_dump_layer.png')

比对无精度问题与发生精度问题时的Layer级dump数据，定位到最早发生精度问题的Layer。比对结果示例如下：

![atb_compare_layer.png](https://raw.gitcode.com/user-images/assets/7898473/baf5ebd9-8923-4991-8dd6-0204b1724e6c/atb_compare_layer.png 'atb_compare_layer.png')

从上图比对结果中可以看出，首个出现精度问题的Layer为“5_Prefill_layer”。

**OP级问题定位**

将dump配置文件中的"ids"参数设置为发生精度问题的Layer ID，例如"5"，"save_child"参数设置为true，dump指定Layer下所有Operation与Kernel的输入/输出Tensor，dump输出件示例如下：

![atb_dump_operation.png](https://raw.gitcode.com/user-images/assets/7898473/61db7d44-55f0-493b-a590-c31c60225bb9/atb_dump_operation.png 'atb_dump_operation.png')

比对无精度问题与发生精度问题时的OP级dump数据，定位到可能存在精度问题的OP算子。比对结果示例如下：

![atb_compare_operation.png](https://raw.gitcode.com/user-images/assets/7898473/86e3190b-a5f2-41b2-a151-8b8b422dddd5/atb_compare_operation.png 'atb_compare_operation.png')

从上图比对结果中可以看出，“0_AddBF16Kernel”算子可能存在精度问题。

找到可疑算子后，可使用dump数据，进行单算子复现，进一步确认是因为算子问题，还是因为内存踩踏等其它问题导致的精度问题。
