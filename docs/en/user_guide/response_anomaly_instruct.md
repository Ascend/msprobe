# Response Anomaly

<!-- md-trans-meta sourceCommit=d04bd615f0a6704fd5647163e7531d767f227e36 translatedAt=2026-08-11T02:42:13.097Z pushedAt=2026-08-11T02:51:07.151Z -->

## 1 Introduction

Response Anomaly is a feature provided by msProbe that leverages the token and logprobs sequences output by a model to detect, in real time and under non-intrusive, zero-reference-knowledge conditions, anomalous responses that may occur during inference. It performs online, real-time, high-accuracy anomaly detection for output crash failures—such as rare characters, garbled text, and repetitive output—in enterprise-grade GenAI inference services.

- Rare characters: Occasional output of meaningless characters that do not fit the conversational context.
- Garbled text: The model continuously outputs rare characters, producing obvious nonsense; the text is meaningless and normal conversation cannot proceed.
- Repetition: Repeated output of identical content.

## 2 Preparation Before Use

Install msProbe by referring to [*msProbe Installation Guide*](../install_guide/msprobe_install_guide.md).

## 3 Quick Start

### 3.1 Preparing Configuration Files

Response Anomaly uses the following configuration files:

1. [config.yaml](../../../python/msprobe/response_anomaly/configs/config.yaml) configures the detection algorithm thresholds.

   `config.yaml` has all thresholds configured by default, and you may adjust them as needed.

   ```yaml
   # Global configuration
   window_size: 128
   stride: 64
   
   # Rare characters
   rare_character:
     explogp_sum_thresh: 0.4  # Sum of topk logprob exponentials of a single token. When below the threshold, the token may be a rare character.
     category_thresh: 2  # Topk category statistics of a single token, used together with explogp_sum_thresh for combined judgment.
     top1_logp_thresh: -6 # When vocabulary information is unavailable, use the top1 logp for judgment.
   
   # Garbled text
   garbled:
     top1_logp_thresh: -5 # When the window_ratio condition is met but the category detection condition is not, if the logprob exceeds this threshold, it can also be considered garbled text.
     window_ratio: 0.2 # Ratio of characters meeting the condition to the sequence length.
     window_thresh: 2 # Number of windows meeting the condition.
   
   # Repetition
   repetition:
     trajectory:   # Trajectory detection N-gram
       n: 3
       distinct_n_thresh: 0.2  # n-gram threshold
       logp_thresh: -0.2  # top1 logprob threshold
   
     acf:   
       acf_threshold: 0.65  # Autocorrelation threshold
       logp_thresh: -0.2  # top1 logprob threshold
   
     single_window_thresh: 14  # When only one method detects a repetition, the number of detected repeated windows must exceed this threshold to trigger an exit. Adjust as needed.
     multi_window_thresh: 2 # When both the ACF and trajectory methods detect repetition simultaneously, the number of detected repeated windows must exceed this threshold to trigger an exit. Adjust as needed.
   ```

2. Configure the `mtype_config.json` file and the mapping file for `token_id` to the character category.

   - `mtype_config.json` primarily stores the model name and the corresponding BOS and EOS token IDs, which are used to cross-validate the model invoked in subsequent detection.
   - The token_id-to-character-category mapping file primarily stores the categories corresponding to the model's token IDs, which are used for subsequent detection of rare characters and garbled text.

   > [!NOTE]
   >
   > msProbe provides the [mtype_config.json](../../../python/msprobe/response_anomaly/configs/mtype_config.json) file and token ID-to-character-category mapping files for the DeepSeek-V3, GLM-4, and Qwen3 models ([deepseekv3_128000.json](../../../python/msprobe/response_anomaly/token2category/deepseekv3_128000.json), [glm-4-7_151329.json](../../../python/msprobe/response_anomaly/token2category/glm-4-7_151329.json), [qwen3-30b-a3b_151643.json](../../../python/msprobe/response_anomaly/token2category/qwen3-30b-a3b_151643.json)). If you are using a different model, run the following commands to generate the corresponding files.

   ```bash
   # Enter the tool directory.
   cd /{msprobe_install_path}/msprobe/response_anomaly/tools
   # Execute the following script.
   python gen_model_config.py --model-path /home/Qwen3-30B-A3B --model-name Qwen3-30B-A3B
   ```

   | Parameter         | Optional/Mandatory | Description                                                         |
   | ------------ | --------- | ------------------------------------------------------------ |
   | `--model-path` | Mandatory      | Path to the model.                                               |
   | `--model-name` | Mandatory      | Model name, which determines the naming of the mapping file and the key in `mtype_config.json`. Enter the actual model name in the following format:<br>&#8226; Naming convention: Use `-_.` as the model name separator, e.g., `Qwen3-30B-A3B`, `glm-4.7-FP8`.<br>&#8226; msProbe converts the model name to lowercase and replaces the symbols `-_.` in the model name with `-`.<br>&#8226; Ensure consistency with the model name passed via the `model_configs` parameter of the [analyze_output_anomaly](#41-analyze_output_anomaly) API. |

   After the script executes successfully, the generated `mtype_config.json` directly replaces the original file content in the `response_anomaly/configs` directory, and the token ID-to-character category mapping file is generated in the `response_anomaly/token2category` directory.

### 3.2 Starting Detection

1. Add the Response Anomaly-related code.

   The highlighted lines below are the Response Anomaly-related code that needs to be added to the inference script:

   ```diff
   from vllm import LLM, SamplingParams
   +from msprobe.response_anomaly import analyze_output_anomaly
   
   # Define the input prompt.
   prompts = "Hello, my name is"
   
   # Set the sampling parameters.
   topk = 20 # Collect topk logprobs.
   sampling_params = SamplingParams(temperature=0.8, top_p=0.95,logprobs=topk,prompt_logprobs=1)
   
   # Initialize the model.
   llm = LLM(model="/home/Qwen3-30B-A3B")
   
   # Execute inference.
   outputs = llm.generate(prompts, sampling_params)
   
   +topk_logprobs = [
   +    {token_id:logprobs[token_id].logprob for token_id in logprobs}
   +        for logprobs in outputs[0].outputs[0].logprobs
   +]
   +tokens = outputs[0].outputs[0].token_ids
   
   +model_configs = 'Qwen3-30B-A3B'
   
   +# Call the API to perform response anomaly detection.
   +result = analyze_output_anomaly([topk_logprobs], [tokens], [model_configs])
   +# Print the anomaly detection results.
   +print(f"is_ill:{result[0][0]},ill_type:{result[0][1]}")
   ```

2. Start inference.

### 3.3 Output Description

Response Anomaly outputs the anomalies detected in the current inference process. For details, refer to the return value description of the [analyze_output_anomaly](#41-analyze_output_anomaly) API.

## 4 API Introduction

### 4.1 analyze_output_anomaly

**Function**

Detects possible anomalous responses during inference in real time.

**Prototype**

```Python
analyze_output_anomaly(topk_logprobs, tokens, model_configs)
```

**Parameters**

- **topk_logprobs** (`List[Dict[int, float]]`): Required; obtained for each request.
- **tokens** (`List[List[int]]`): Required; token sequence obtained for each request.
- **model_configs** (`List[Any]`): Required; model name for each request.
  - Must be consistent with the model name configured via the `--model-name` parameter.
  - If multiple requests are served by the same model, for example, when launching the Qwen3-30B-A3B model service with three inference data entries, `model_configs` should be `['Qwen3-30B-A3B']*3`.

> [!NOTE]
>
> When the input is a single request, it must be wrapped in an outer List.

**Return Value**

Return result format: [[is_ill, ill_type], ...]

Each request returns the following two parameters:

- `is_ill`: bool type. Whether the output is anomalous. `True` indicates anomalous, and `False` indicates normal.
- `ill_type`: int type, anomaly type. `0` indicates normal, `1` indicates rare characters, `2` indicates garbled text, and `3` indicates repetition.

**Call Example**

See [Starting Detection](#32-starting-detection).
