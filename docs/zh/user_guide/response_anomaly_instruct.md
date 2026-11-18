# 推理异常检测

## 1. 简介

推理异常检测（Response Anomaly）是由 msProbe 提供的在线/离线推理输出异常检测工具，在无侵入、零参照知识的条件下实时感知企业级 GenAI 推理服务中出现的输出崩溃类故障。模块支持**默认模式（纯Token序列）**与**兼容模式（传统topk_logprobs）**两条检测路径。

**功能特点**

- **默认模式（纯Token序列）**：无需采集 `topk_logprobs`，也不依赖额外的判别模型或离线词表生成脚本，仅凭生成的 Token ID 序列与模型分词器即可实时识别生僻字/乱码与重复生成异常。
- **兼容模式（传统topk_logprobs）**：显式传入 `topk_logprobs` 时，沿用基于 token 与概率分布滑窗的检测算法，识别生僻字、乱码、重复及 `NaN`/`Inf` 异常。
- **灵活接入方式**：提供单次或批量调用的便捷函数 `analyze_output_anomaly`，以及支持长驻服务初始化并复用词表缓存的 `ILLDetector` 类。

默认模式（纯Token序列）下的初始化耗时、单请求检测耗时与批量吞吐参考数据，请参见[推理异常检测工具性能基线报告](../baseline/response_anomaly_baseline.md)。

## 2. 使用前准备

**环境准备**

安装 msProbe 工具，详情请参见《[msProbe 安装指南](../install_guide/msprobe_install_guide.md)》。

**默认模式（纯Token序列）准备**

准备被测模型的分词器（支持本地模型目录路径、`tokenizer.json` 文件路径，或推理进程中已加载的内存分词器实例），用于解析模型词表；传入本地模型目录或分词器文件前，请确保其来源安全可信。工具已内置重复检测规则配置文件 [repetition.json](../../../python/msprobe/response_anomaly/configs/repetition.json)（可按需调整阈值）与生僻字评分模型权重文件 [rare_chars_model.npz](../../../python/msprobe/response_anomaly/tables/rare_chars_model.npz)，无需手动生成额外词表映射文件。

**兼容模式（传统topk_logprobs）准备**

仅在使用兼容模式（传统topk_logprobs）时需配置以下文件（已为 DeepSeek-V3、GLM-4、Qwen3 内置默认配置）：

1. [config.yaml](../../../python/msprobe/response_anomaly/configs/config.yaml) 配置检测算法阈值。

   ```yaml
   # 全局配置
   window_size: 128
   stride: 64

   # rare-character 生僻字
   rare_character:
     explogp_sum_thresh: 0.4  # 单token topk的logprob exp总和，当小于阈值时，则可能为生僻字
     category_thresh: 2  # 单token的topk的类别统计，与explogp_sum_thresh综合判定
     top1_logp_thresh: -6 # 当没有词表信息时，使用top1 的logp进行判定

   # garbled 乱码
   garbled:
     top1_logp_thresh: -5 # 当window_ratio比例符合，但加上类别检测后其比例不符合时，如果其logprob超过该阈值，也可考虑为乱码
     window_ratio: 0.2 # 满足条件的字符占序列长度的比例
     window_thresh: 0 # 当检出乱码的窗口数超过该阈值，则直接退出，报乱码异常。当为0时，即只要有一个窗口检出乱码即退出上报

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
     multi_window_thresh: 2 # 当acf和trajectory两个方法同时检出时，检出重复的窗口数需达到该阈值才会退出，可根据需要修改
   ```

2. 配置 `mtype_config.json` 和 `token_id` 与字符类别的映射文件。

   - `mtype_config.json` 主要保存模型名称和对应的 BOS、EOS 的 `token_id`，用于后续检测中交叉验证用户调用的模型。
   - `token_id` 与字符类别的映射文件主要保存模型 `token_id` 对应的类别，用于后续生僻字和乱码的检测。

   > [!NOTE]
   >
   > msProbe 提供了 deepseekv3、glm4、qwen3 模型的 [mtype_config.json](../../../python/msprobe/response_anomaly/configs/mtype_config.json) 文件和 `token_id` 与字符类别的映射文件（[deepseekv3_128000.json](../../../python/msprobe/response_anomaly/token2category/deepseekv3_128000.json)、[glm-4-7_151329.json](../../../python/msprobe/response_anomaly/token2category/glm-4-7_151329.json)、[qwen3-30b-a3b_151643.json](../../../python/msprobe/response_anomaly/token2category/qwen3-30b-a3b_151643.json)），若用户使用的是其他模型，可使用以下命令生成对应文件。

   ```bash
   # 进入工具所在路径
   cd /{msprobe_install_path}/msprobe/response_anomaly/tools
   # 执行如下脚本
   python gen_model_config.py --model-path /home/Qwen3-30B-A3B --model-name Qwen3-30B-A3B
   ```

   | 参数         | 可选/必选 | 说明                                                         |
   | ------------ | --------- | ------------------------------------------------------------ |
   | --model-path | 必选      | 模型所在路径。                                               |
   | --model-name | 必选      | 模型名称，决定映射文件的命名和 `mtype_config.json` 里的 key，请根据实际模型填写，格式如下：<br>&#8226; 命名规范：模型名称分隔符请选择 `-_.`，如 `Qwen3-30B-A3B`、`glm-4.7-FP8`。<br>&#8226; msProbe 会将模型名称转为小写，并将模型名称中符号 `-_.` 均转为 `-`。<br>&#8226; 请与后续 [analyze_output_anomaly](#51-analyze_output_anomaly) 接口的 `model_configs` 参数传入的模型名称保持一致。 |

   成功执行脚本后，生成的 `mtype_config.json` 会直接替换 `response_anomaly/configs` 目录下原本的文件内容，`token_id` 与字符类别的映射文件会生成在 `response_anomaly/token2category` 目录下。

## 3. 默认模式（纯Token序列）

**功能说明**

默认模式（纯Token序列）仅依赖模型生成的 Token ID 序列与对应分词器词表，支持识别以下输出异常：

- **生僻字 / 乱码（`ill_type = 1`）**：基于增量 UTF-8 解码与内置 GBDT 字符统计模型，识别截断乱码字节、罕见生僻字及异常跨语种突兀字符。
- **重复生成 / 兜圈推理（`ill_type = 3`）**：结合精确周期循环、超长单字符连击、滚动 N-gram 与 MinHash 句级重复检测重复生成与无效绕圈。
- **多异常触发仲裁**：当两类异常在同一条输出序列中均触发时，返回更早触发（最小 `fired_at`）的异常结果；若触发位置相同，优先返回生僻字/乱码异常（`ill_type = 1`）。
- **灵敏度档位与思考分段**：支持 `"intercept"`（拦截档，默认）与 `"monitor"`（监控档）两档灵敏度，并支持通过 `think_end="auto"` 自动识别思考结束 token 以区分思考阶段与回答阶段。

**使用示例**

1. 便捷函数入口 `analyze_output_anomaly`（无需开启 `logprobs`，支持单条或批量 token 序列，以下高亮行为需在推理脚本中添加的代码）：

   ```diff
    from vllm import LLM, SamplingParams
   +from msprobe.response_anomaly import analyze_output_anomaly

    # 定义输入提示
    prompts = "Hello, my name is"

    # 设置采样参数（无需采集 logprobs）
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
   +model_path = "/home/Qwen3-30B-A3B"

    # 初始化模型
   -llm = LLM(model="/home/Qwen3-30B-A3B")
   +llm = LLM(model=model_path)

    # 执行推理
    outputs = llm.generate(prompts, sampling_params)

   +tokens = outputs[0].outputs[0].token_ids
   +# 默认返回 [[is_ill, ill_type]]；传入 return_detail=True 可获取包含 fired_at 与 reason 的 DetectionResult 列表
   +result = analyze_output_anomaly([tokens], model_path)
   +print(f"is_ill:{result[0][0]},ill_type:{result[0][1]}")
   ```

2. 长驻推理服务高频检测场景，建议在初始化阶段创建一次 `ILLDetector` 实例并长期复用（支持传入可信的本地模型目录、`tokenizer.json` 路径或内存中的 `llm.get_tokenizer()`）：

   ```python
   from msprobe.response_anomaly import ILLDetector

   detector = ILLDetector("/home/Qwen3-30B-A3B")
   # 批量返回 [[is_ill, ill_type], ...]；传入 return_detail=True 返回 DetectionResult 列表
   results = detector.run([out.token_ids for out in outputs[0].outputs], return_detail=True)
   for res in results:
       print(res.is_ill, res.ill_type, res.fired_at, res.reason)
   ```

> [!NOTE]
>
> 传入本地模型目录或分词器文件路径时，请确保其来源安全可信。当目录下不含 `tokenizer.json` 而回退调用 `AutoTokenizer.from_pretrained` 时，工具固定以 `trust_remote_code=False` 离线加载，不执行自定义分词代码；若被测模型依赖自定义分词代码，请直接传入已在推理进程中加载的内存分词器实例（如 `llm.get_tokenizer()`）。

示例接口说明：

- [analyze_output_anomaly](#51-analyze_output_anomaly)：单次或批量推理输出异常检测函数入口。
- [ILLDetector](#52-illdetector)：封装词表缓存与检测逻辑的可复用检测器类。

更多接口详情请参见 [5. 接口介绍](#5-接口介绍) 章节。

**输出说明**

`analyze_output_anomaly` 与 `detector.run` 默认返回二维列表 `[[is_ill, ill_type], ...]`，示例 1 执行后的打印输出如下（以正常输出为例）：

```text
is_ill:False,ill_type:0
```

当传入 `return_detail=True`（或调用单条接口 `detector.detect` / `detector.detector`）时，返回 [DetectionResult](#53-detectionresult) 对象，示例 2 检出重复异常时的打印输出示例如下：

```text
True 3 192 vllm[32,2000]x3
```

默认模式（纯Token序列）下各输出字段含义：

- `is_ill`：`True` 表示存在异常，`False` 表示正常。
- `ill_type`：`0` 表示正常，`1` 表示生僻字或 UTF-8 乱码，`3` 表示重复生成或兜圈推理。
- `fired_at`：首次触发告警的检查点前缀长度（1-based token 数），正常时为 `-1`。
- `reason`：命中的具体规则描述，正常时为空字符串 `""`。

## 4. 兼容模式（传统topk_logprobs）

**功能说明**

当调用方显式传入 `topk_logprobs` 时，工具采用兼容模式（传统topk_logprobs），沿用基于 token 与概率分布滑窗的检测算法，支持识别以下异常：

- **生僻字（`ill_type = 1`）**：偶发性输出无意义字符，且不符合上下文语境。
- **乱码（`ill_type = 2`）**：模型持续输出生僻字，明显胡言乱语，文本无意义，无法正常对话。
- **重复（`ill_type = 3`）**：基于轨迹 N-gram 与自相关（ACF）滑窗检测重复输出相同内容。
- **NaN Value（`ill_type = 4`）**：`logprobs` 出现 `nan`/`inf` 值。

> [!NOTE]
> 兼容模式（传统topk_logprobs）下当 `logprobs` 出现 `nan`/`inf` 值时，会直接返回 NaN Value（`ill_type = 4`），表明此时模型推理已经出现异常，不再做生僻字、乱码、重复的异常检测。

**使用示例**

完成 [2. 使用前准备](#2-使用前准备) 中的兼容模式（传统topk_logprobs）配置后，在推理应用脚本中显式采集并传入 `topk_logprobs`（以下高亮行为需在推理脚本中添加的代码）：

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

**输出说明**

执行完成后，`analyze_output_anomaly` 默认返回二维列表 `[[is_ill, ill_type], ...]`，上述示例打印输出如下（以正常输出为例）：

```text
is_ill:False,ill_type:0
```

兼容模式（传统topk_logprobs）下各输出字段含义：

- `is_ill`：`True` 表示存在异常，`False` 表示正常。
- `ill_type`：`0` 表示正常，`1` 表示生僻字，`2` 表示乱码，`3` 表示重复，`4` 表示 NaN Value。
- 若通过 `ILLDetector` 传入 `return_detail=True` 获取 [DetectionResult](#53-detectionresult)，兼容模式（传统topk_logprobs）下 `fired_at` 固定为 `-1`，`reason` 固定为空字符串 `""`。

## 5. 接口介绍

### 5.1 analyze_output_anomaly

**功能说明**

实时检测推理过程中可能出现的异常响应。默认采用默认模式（纯Token序列）执行检测；当显式传入 `topk_logprobs`（且对应请求非 `None` / 非空）时，自动采用兼容模式（传统topk_logprobs）检测算法。

**函数原型**

```python
# 1) 默认模式（纯Token序列）原型（无需 logprobs）
analyze_output_anomaly(
    tokens,
    model_configs=None,
    *,
    tier="intercept",
    think_end="auto",
    repetition_config=None,
    rare_chars_model_path=None,
    return_detail=False,
)

# 2) 兼容模式（传统topk_logprobs）原型（兼容已有调用方式）
analyze_output_anomaly(topk_logprobs, tokens, model_configs)
```

**参数说明**

- **`tokens`** (`Sequence[int]` | `Sequence[Sequence[int]]` | `np.ndarray`)：必选，单条 1 维输出 `token_ids` 序列或 2 维批量序列。
- **`model_configs`** (`str` | `os.PathLike` | tokenizer 对象 | `List[Any]`)：
  - **默认模式（纯Token序列）下必选**：可传可信的本地模型目录路径（优先解析 `tokenizer.json`，无该文件时以 `trust_remote_code=False` 离线加载内置慢速分词器）、`tokenizer.json` 文件路径、内存分词器对象（HuggingFace fast/slow tokenizer、`tokenizers.Tokenizer`、`tiktoken` 等；依赖自定义分词代码的模型请传内存分词器对象）或对应的批大小列表；传入单值时自动广播到整个批次。
  - **兼容模式（传统topk_logprobs）下可选**：与 `--model-name` 配置的模型名称保持一致，如 `['Qwen3-30B-A3B']*3`（若直接传本地模型目录路径，也可同时兼顾两种模式）。
- **`topk_logprobs`** (`List[Optional[List[Dict[int, float]]]]`，可选)：每条请求采集的 topk logprobs。省略、传 `None` 或单条元素为 `None` 时，自动对该请求使用默认模式（纯Token序列）检测（需通过 `model_configs` 提供模型目录、`tokenizer.json` 路径或分词器实例，若仅传入裸模型名称将抛出 `ValueError`）。
- **`tier`** (`str`，可选，默认 `"intercept"`)：默认模式（纯Token序列）灵敏度档位，支持 `"intercept"`（拦截档）与 `"monitor"`（监控档）。
- **`think_end`** (`"auto"` | `int` | `None`，可选，默认 `"auto"`)：默认模式（纯Token序列）下思考结束 token ID；`"auto"` 自动从分词器识别，`None` 关闭思考/回答阶段划分。
- **`repetition_config`** (`dict` | `str` | `os.PathLike`，可选，默认 `None`)：默认模式（纯Token序列）下自定义重复检测规则字典或 JSON 配置文件路径，为 `None` 时使用内置 [repetition.json](../../../python/msprobe/response_anomaly/configs/repetition.json)。
- **`rare_chars_model_path`** (`str` | `os.PathLike`，可选，默认 `None`)：默认模式（纯Token序列）下自定义生僻字评分模型路径，为 `None` 时使用内置 [rare_chars_model.npz](../../../python/msprobe/response_anomaly/tables/rare_chars_model.npz)。
- **`return_detail`** (`bool`，可选，默认 `False`)：为 `False` 时返回 `[[is_ill, ill_type], ...]`；为 `True` 时返回 `List[DetectionResult]`。

> [!NOTE]
>
> 在兼容模式（传统topk_logprobs）下当输入为单请求数据时，需要在 `topk_logprobs` 与 `tokens` 外部嵌套一层 List。在默认模式（纯Token序列）下传入单条 1 维 `tokens` 序列时内部会自动包装为单元素批次。

**返回值说明**

默认返回格式：`[[is_ill, ill_type], ...]`（`return_detail=True` 时返回 `[DetectionResult(...), ...]`）。

每个请求包含：

- **`is_ill`**：`bool` 类型，是否异常，`True` 为异常，`False` 为正常。
- **`ill_type`**：`int` 类型，异常类型：
  - `0`：正常
  - `1`：生僻字（默认模式（纯Token序列）下包含生僻字与 UTF-8 乱码）
  - `2`：乱码（兼容模式（传统topk_logprobs））
  - `3`：重复
  - `4`：NaN Value（兼容模式（传统topk_logprobs））

**调用示例**

请参见 [3. 默认模式（纯Token序列）](#3-默认模式纯token序列) 与 [4. 兼容模式（传统topk_logprobs）](#4-兼容模式传统topk_logprobs) 中的使用示例。

### 5.2 ILLDetector

**功能说明**

封装配置文件加载、分词器词表缓存与双模式检测逻辑，适合在推理服务初始化阶段实例化并长期复用。

**函数原型**

```python
# 方式一：默认模式（纯Token序列）初始化（同时内置默认 config.yaml，支持兼容模式（传统topk_logprobs）调用）
detector = ILLDetector(
    model_or_tokenizer,
    *,
    think_end="auto",
    repetition_config=None,
    rare_chars_model_path=None,
)

# 方式二：兼容模式（传统topk_logprobs）配置路径初始化（不传参数则使用包内默认配置）
detector = ILLDetector(
    config_path="./configs/config.yaml",
    mtype_path="./configs/mtype_config.json",
    tk2cat_path="./token2category/",
)
```

**参数说明**

- **`model_or_tokenizer`**：可信的本地模型目录路径（不含 `tokenizer.json` 时固定以 `trust_remote_code=False` 离线加载内置慢速分词器）、`tokenizer.json` 文件路径或内存分词器对象（依赖自定义分词代码的模型请传入内存分词器对象），用于默认模式（纯Token序列）初始化。
- **`think_end`**：默认模式（纯Token序列）下思考结束 token ID，默认 `"auto"`。
- **`repetition_config`**：可选，自定义重复检测规则字典或 JSON 配置文件路径，默认使用内置 [repetition.json](../../../python/msprobe/response_anomaly/configs/repetition.json)。
- **`rare_chars_model_path`**：可选，自定义生僻字评分模型路径，默认使用内置 [rare_chars_model.npz](../../../python/msprobe/response_anomaly/tables/rare_chars_model.npz)。
- **`config_path`** / **`mtype_path`** / **`tk2cat_path`**：可选，兼容模式（传统topk_logprobs）下的阈值配置文件、模型元信息文件与词表类别映射目录路径。

**返回值说明**

`detector` 是 `ILLDetector` 类的实例对象，提供以下主要方法执行单条或批量异常检测：

- `detect(tokens, tier="intercept", *, model_config=None) -> DetectionResult`：默认模式（纯Token序列）检测单条序列。
- `detector(tokens, model_config=None, *, tier="intercept") -> DetectionResult`（或 `detector(topk_logprobs, tokens, model_config=None)`）：单请求统一入口，返回 [DetectionResult](#53-detectionresult)。
- `run(tokens, model_configs=None, *, tier="intercept", return_detail=False)`（或 `run(topk_logprobs, tokens, model_configs=None, *, return_detail=False)`）：批量检测入口。

> [!TIP]
>
> 首次词表构建耗时集中在进程启动期，建议长驻服务在初始化阶段创建并复用 `ILLDetector` 实例（`detect` / `run` 均无状态且可安全重复调用）。多模型场景下，缓存内存随不同分词器数量线性增长，但受进程级 LRU 双上限约束（最多 `64` 份，默认总字节预算 `256 MiB`，可通过环境变量 `MSPROBE_VOCAB_CACHE_MB` 调整）；单份超预算时仍可正常检测，被逐出的词表不影响已绑定它的活跃实例。

**调用示例**

请参见 [3. 默认模式（纯Token序列）](#3-默认模式纯token序列) 中的使用示例。

### 5.3 DetectionResult

**功能说明**

单条输出序列异常检测的结构化结果数据类。

**类型定义**

```python
@dataclass
class DetectionResult:
    is_ill: bool = False
    ill_type: int = 0
    fired_at: int = -1
    reason: str = ""
```

**属性说明**

- **`is_ill`**：是否检测到异常。
- **`ill_type`**：异常类别编码（`0`~`4`）。
- **`fired_at`**：默认模式（纯Token序列）下首次触发告警的检查点前缀长度（1-based token 数；生僻字/乱码按规则实际依赖前缀计算，重复检测按 64-token 块边界或序列尾端检查点计算）；正常或兼容模式（传统topk_logprobs）下为 `-1`。
- **`reason`**：默认模式（纯Token序列）下命中的规则描述；正常或兼容模式（传统topk_logprobs）下为空字符串 `""`。
