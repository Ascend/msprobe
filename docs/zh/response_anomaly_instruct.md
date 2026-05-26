# 推理异常检测

## 1 简介

推理异常检测（Response Anomaly）是由 msProbe 提供的基于模型输出的 token 和 logprobs 序列，在无侵入、零参照知识的条件下，实时检测推理过程中可能出现的异常响应，对企业级GenAI推理服务中出现的生僻字、乱码、重复输出等输出崩溃类故障进行在线实时、高准确率的异常检测。

- 生僻字：偶发性输出无意义字符，且不符合上下文语境。
- 乱码：模型持续输出生僻字，明显胡言乱语，文本无意义，无法正常对话。
- 重复：重复输出相同内容。

## 2 使用前准备

安装msProbe工具，详情请参见《[msProbe安装指南](./msprobe_install_guide.md)》。

## 3 快速入门

### 3.1 配置文件准备

Response Anomaly使用以下配置文件：

1. [config.yaml](../../python/msprobe/response_anomaly/configs/config.yaml) 配置检测算法阈值。

   config.yaml 已默认配置全部阈值，用户可根据需求自行调节。

   ```yaml
   # 全局配置
   window_size: 128
   stride: 64
   
   # rare-character 生僻字
   rare_character:
     explogp_sum_thresh: 0.4  # 单token topk的logprob exp总和，当小于阈值时，则可能为生僻字
     category_thresh: 2  # 单token的topk的类别统计，与explogp_sum_thresh综合判定
   
   # garbled 乱码
   garbled:
     top1_logp_thresh: -5 # 当window_ratio比例符合，但加上类别检测后其比例不符合时，如果其logprob超过该阈值，也可考虑为乱码
     window_ratio: 0.2 # 满足条件的字符占序列长度的比例
     window_thresh: 2 # 满足条件的窗口数
   
   # repetition 重复
   repetition:
     trajectory:   # 轨迹检测 N-gram
       n: 3
       distinct_n_thresh: 0.2  # n-gram阈值
       logp_thresh: -0.2  # top1的logprob阈值
   
     acf:   
       acf_threshold: 0.65  # 自相关阈值
       logp_thresh: -0.2  # top1的logprob阈值
   
     single_window_thresh: 14  # 当只有一个方法检出时，检出重复的窗口数需要超出该阈值才会退出，可根据需要修改
     multi_window_thresh: 2 # 当acf和trajectory两个方法同时检出时，检出重复的窗口数需超出该阈值才会退出，可根据需要修改
   ```

2. 配置 mtype_config.json 和 token_id 与字符类别的映射文件。

   - mtype_config.json主要保存模型名称和对应的 BOS、EOS 的 token_id ，用于后续检测中交叉验证用户调用的模型。
   - token_id 与字符类别的映射文件主要保存模型 token_id 对应的类别，用于后续生僻字和乱码的检测。

   > [!NOTE]
   >
   > msProbe提供了deepseekv3、glm4、qwen3模型的 [mtype_config.json](../../python/msprobe/response_anomaly/configs/mtype_config.json) 文件和 token_id 与字符类别的映射文件（[deepseekv3_128000.json](../../python/msprobe/response_anomaly/token2category/deepseekv3_128000.json)、[glm-4-7_151329.json](../../python/msprobe/response_anomaly/token2category/glm-4-7_151329.json)、[qwen3-30b-a3b_151643.json](../../python/msprobe/response_anomaly/token2category/qwen3-30b-a3b_151643.json)），若用户使用的是其他模型，请使用以下命令生成对应文件。

   ```bash
   # 进入工具所在路径
   cd /{msprobe_install_path}/python/msprobe/response_anomaly
   # 执行如下脚本
   python ./tools/gen_model_config.py --model-path /home/Qwen3-30B-A3B --model-name Qwen3-30B-A3B
   ```

   | 参数         | 可选/必选 | 说明                                                         |
   | ------------ | --------- | ------------------------------------------------------------ |
   | --model-path | 必选      | 模型所在路径。                                               |
   | --model-name | 必选      | 模型名称，决定映射文件的命名和 mtype_config.json 里的 key，请根据实际模型填写，格式如下：<br>&#8226; 命名规范：模型名称分隔符请选择`-_.`，如`Qwen3-30B-A3B`、`glm-4.7-FP8`。<br>&#8226; msProbe会将模型名称转为小写，并将模型名称中符号`-_.`均转为`-`。<br>&#8226; 请与后续[analyze_output_anomaly](#41-analyze_output_anomaly)接口的 model_configs 参数传入的模型名称保持一致。 |

   成功执行脚本后，生成的 mtype_config.json 会直接替换 response_anomaly/configs 目录下原本的文件内容， token_id 与字符类别的映射文件会生成在 response_anomaly/token2category 目录下。

### 3.2 启动检测

1. 添加Response Anomaly接口。

   以下高亮行是需要在推理应用脚本中添加的Response Anomaly相关代码：

   ```diff
   from vllm import LLM, SamplingParams
   +from msprobe.response_anomaly import analyze_output_anomaly
   
   # 定义输入提示
   prompts = "Hello, my name is"
   
   # 设置采样参数
   topk = 20 # 采集topk logprobs
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95,logprobs=topk,prompt_logprobs=1)
   
   # 初始化模型
   llm = LLM(model="/home/Qwen3-30B-A3B")
   
   # 执行推理
   outputs = llm.generate(prompts, sampling_params)
   
   +topk_logprobs = [
   +    {token_id:logprobs[token_id].logprob for token_id in logprobs}
   +        for logprobs in outputs[0].outputs[0].logprobs
   +]
   +tokens = outputs[0].outputs[0].token_ids
   
   +model_configs = 'Qwen3-30B-A3B'
   
   +# 调用接口，执行推理异常检测
   +result = analyze_output_anomaly([topk_logprobs], [tokens], [model_configs])
   +# 打印输出异常检测结果
   +print(f"is_ill:{result[0][0]},ill_type:{result[0][1]}")
   ```

2. 启动推理应用。

### 3.3 输出说明

Response Anomaly将会输出当前推理进程中检测到的异常，具体请参见[analyze_output_anomaly](#41-analyze_output_anomaly)接口的返回值说明。

## 4 接口介绍

### 4.1 analyze_output_anomaly

**功能说明**

实时检测推理过程中可能出现的异常响应。

**函数原型**

```Python
analyze_output_anomaly(topk_logprobs, tokens, model_configs)
```

**参数说明**

- **topk_logprobs** (`List[Dict[int, float]]`)：必选参数，每条请求获取的 topk logprobs。
- **tokens** (`List[List[int]]`)：必选参数，每条请求获取的 tokens 序列。
- **model_configs** (`List[Any]`)：必选参数，每条请求的模型名称。
  - 请与 --model-name 参数配置的模型名称保持一致。
  - 若为同一模型服务的多个请求，如拉起 Qwen3-30B-A3B 模型服务，输入3条推理数据，model_configs 应为['Qwen3-30B-A3B']*3

> [!NOTE]
>
> 当输入为单请求数据时，需要在输入数据外部嵌套一层 List。

**返回值说明**

返回结果格式：[[is_ill, ill_type], ...]

每个请求会返回下述两个参数：

- is_ill：bool类型，是否异常，True为异常，False为正常。
- ill_type：int类型，异常类型，0表示正常，1表示生僻字，2表示乱码，3表示重复。

**调用示例**

请参见[启动检测](#32-启动检测)章节。
