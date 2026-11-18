# Response Anomaly

## 1. Introduction

Response Anomaly is an online/offline inference output anomaly detection tool provided by msProbe. Under non-intrusive, zero-reference-knowledge conditions, it detects output crash failures in enterprise-grade GenAI inference services in real time. The module supports two detection paths: **Default Mode (Token-Only Sequence)** and **Compatible Mode (Legacy topk_logprobs)**.

**Features**

- **Default Mode (Token-Only Sequence)**: Without collecting `topk_logprobs`, relying on auxiliary judge models, or running offline vocabulary generation scripts, it detects rare-character/garbled text and repetition anomalies using only generated token ID sequences and the model tokenizer.
- **Compatible Mode (Legacy topk_logprobs)**: When `topk_logprobs` are explicitly supplied, it runs the sliding-window token-and-logprob detection pipeline to identify rare characters, garbled text, repetition, and `NaN`/`Inf` anomalies.
- **Flexible integration**: Provides both the convenience function `analyze_output_anomaly` for single/batch calls and the `ILLDetector` class for reusing cached vocabulary tables in long-running services.

For initialization latency, single-request detection latency, and batch throughput reference data in Default Mode (Token-Only Sequence), see [Performance Baseline Report of the Response Anomaly Detection Tool](../baseline/response_anomaly_baseline.md).

## 2. Preparation Before Use

**Environment Preparation**

Install msProbe by referring to [*msProbe Installation Guide*](../install_guide/msprobe_install_guide.md).

**Default Mode (Token-Only Sequence) Preparation**

Prepare the tokenizer of the target model (a local model directory path, a `tokenizer.json` file path, or an in-memory tokenizer instance loaded in the inference process) for vocabulary parsing; before passing a local model directory or tokenizer file, ensure that it comes from a trusted source. The repetition detection rule configuration [repetition.json](../../../python/msprobe/response_anomaly/configs/repetition.json) (thresholds adjustable as needed) and the rare-character scoring model weights [rare_chars_model.npz](../../../python/msprobe/response_anomaly/tables/rare_chars_model.npz) are built into the package, so no manual vocabulary mapping generation is required.

**Compatible Mode (Legacy topk_logprobs) Preparation**

Configure the following files only when using Compatible Mode (Legacy topk_logprobs) (defaults for DeepSeek-V3, GLM-4, and Qwen3 are bundled):

1. [config.yaml](../../../python/msprobe/response_anomaly/configs/config.yaml) configures the detection algorithm thresholds.

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
     window_thresh: 0 # When the number of detected garbled windows exceeds this threshold, exit immediately and report garbled text. When 0, exit on the first detected window.

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

2. Configure the `mtype_config.json` file and the mapping file for `token_id` to character category.

   - `mtype_config.json` primarily stores the model name and the corresponding BOS and EOS token IDs, which are used to cross-validate the model invoked in subsequent detection.
   - The `token_id`-to-character-category mapping file primarily stores the categories corresponding to the model's token IDs, which are used for subsequent detection of rare characters and garbled text.

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
   | `--model-name` | Mandatory      | Model name, which determines the naming of the mapping file and the key in `mtype_config.json`. Enter the actual model name in the following format:<br>&#8226; Naming convention: Use `-_.` as the model name separator, e.g., `Qwen3-30B-A3B`, `glm-4.7-FP8`.<br>&#8226; msProbe converts the model name to lowercase and replaces the symbols `-_.` in the model name with `-`.<br>&#8226; Ensure consistency with the model name passed via the `model_configs` parameter of the [analyze_output_anomaly](#51-analyze_output_anomaly) API. |

   After the script executes successfully, the generated `mtype_config.json` directly replaces the original file content in the `response_anomaly/configs` directory, and the token ID-to-character category mapping file is generated in the `response_anomaly/token2category` directory.

## 3. Default Mode (Token-Only Sequence)

**Function Description**

Default Mode (Token-Only Sequence) uses only generated token ID sequences and the model tokenizer vocabulary to detect the following output anomalies:

- **Rare characters / garbled text (`ill_type = 1`)**: Incremental UTF-8 corruption, rare-script characters scored by a built-in GBDT model, and abrupt cross-script anomalies.
- **Repetition / circling (`ill_type = 3`)**: Periodic token prefix loops, long single-character runs, rolling n-gram reuse, and MinHash sentence-level circling loops.
- **Multi-anomaly arbitration**: When both detectors trigger on the same sequence, the anomaly with the smaller `fired_at` prefix position is reported (`ill_type = 1` wins exact ties).
- **Sensitivity tiers and reasoning segmentation**: Supports `"intercept"` (default) and `"monitor"` sensitivity tiers, and automatically identifies reasoning end markers via `think_end="auto"` to separate reasoning from final answers.

**Usage Example**

1. Convenience function `analyze_output_anomaly` (no `logprobs` required; accepts a 1D sequence or 2D batch; highlighted lines show the code added to the inference script):

   ```diff
    from vllm import LLM, SamplingParams
   +from msprobe.response_anomaly import analyze_output_anomaly

    # Define the input prompt.
    prompts = "Hello, my name is"

    # Set the sampling parameters (no logprobs needed).
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
   +model_path = "/home/Qwen3-30B-A3B"

    # Initialize the model.
   -llm = LLM(model="/home/Qwen3-30B-A3B")
   +llm = LLM(model=model_path)

    # Execute inference.
    outputs = llm.generate(prompts, sampling_params)

   +tokens = outputs[0].outputs[0].token_ids
   +# Returns [[is_ill, ill_type]] by default; pass return_detail=True for List[DetectionResult] with fired_at and reason.
   +result = analyze_output_anomaly([tokens], model_path)
   +print(f"is_ill:{result[0][0]},ill_type:{result[0][1]}")
   ```

2. In long-running inference services or across multi-request batches, instantiate and reuse `ILLDetector` at startup (accepts a trusted local model directory path, `tokenizer.json`, or `llm.get_tokenizer()`):

   ```python
   from msprobe.response_anomaly import ILLDetector

   detector = ILLDetector("/home/Qwen3-30B-A3B")
   results = detector.run([out.token_ids for out in outputs[0].outputs], return_detail=True)
   for res in results:
       print(res.is_ill, res.ill_type, res.fired_at, res.reason)
   ```

> [!NOTE]
>
> Ensure that any local model directory or tokenizer file path comes from a trusted source. When a directory does not contain `tokenizer.json` and falls back to `AutoTokenizer.from_pretrained`, it is loaded offline with `trust_remote_code=False` and never executes custom tokenization code; for models relying on custom tokenization code, pass an in-memory tokenizer instance already loaded in the inference process (e.g., `llm.get_tokenizer()`).

APIs used in the examples:

- [analyze_output_anomaly](#51-analyze_output_anomaly): Convenience entry point for single or batch inference output anomaly detection.
- [ILLDetector](#52-illdetector): Reusable detector class encapsulating vocabulary lookup caches and detection logic.

For more API details, see [5. API Introduction](#5-api-introduction).

**Output Description**

By default, `analyze_output_anomaly` and `detector.run` return `[[is_ill, ill_type], ...]`. Example 1 prints the following output when no anomaly is detected:

```text
is_ill:False,ill_type:0
```

When `return_detail=True` is passed (or when calling `detector.detect` / `detector.detector`), [DetectionResult](#53-detectionresult) objects are returned. Example 2 prints output similar to the following when a periodic repetition loop is detected:

```text
True 3 192 vllm[32,2000]x3
```

Output fields in Default Mode (Token-Only Sequence):

- `is_ill`: `True` if an anomaly is detected; `False` if normal.
- `ill_type`: `0` indicates normal, `1` indicates rare characters or UTF-8 corruption, and `3` indicates repetition or circling reasoning.
- `fired_at`: First checkpoint prefix length (1-based token count) where an anomaly triggers; `-1` when normal.
- `reason`: Matched detection rule description; `""` when normal.

## 4. Compatible Mode (Legacy topk_logprobs)

**Function Description**

When `topk_logprobs` are explicitly supplied, the tool runs Compatible Mode (Legacy topk_logprobs), applying the sliding-window token-and-logprob detection pipeline to identify the following anomalies:

- **Rare characters (`ill_type = 1`)**: Occasional output of meaningless characters that do not fit the conversational context.
- **Garbled text (`ill_type = 2`)**: The model continuously outputs rare characters, producing obvious nonsense where normal conversation cannot proceed.
- **Repetition (`ill_type = 3`)**: Trajectory n-gram and autocorrelation (ACF) sliding-window detection of repeated content.
- **NaN Value (`ill_type = 4`)**: `nan`/`inf` values appear in `logprobs`.

> [!NOTE]
> In Compatible Mode (Legacy topk_logprobs), when `nan`/`inf` values appear in `logprobs`, NaN Value (`ill_type = 4`) is returned directly, indicating that model inference is already abnormal, and rare character, garbled text, and repetition detection will not be performed.

**Usage Example**

After completing the Compatible Mode (Legacy topk_logprobs) configuration in [2. Preparation Before Use](#2-preparation-before-use), collect and pass `topk_logprobs` in the inference script (highlighted lines show the code added to the script):

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

**Output Description**

After execution, `analyze_output_anomaly` returns `[[is_ill, ill_type], ...]` by default. The preceding example prints the following output when no anomaly is detected:

```text
is_ill:False,ill_type:0
```

Output fields in Compatible Mode (Legacy topk_logprobs):

- `is_ill`: `True` if an anomaly is detected; `False` if normal.
- `ill_type`: `0` indicates normal, `1` indicates rare characters, `2` indicates garbled text, `3` indicates repetition, and `4` indicates NaN Value.
- When `return_detail=True` is passed via `ILLDetector` to obtain [DetectionResult](#53-detectionresult) objects, `fired_at` is fixed to `-1` and `reason` is fixed to `""` in Compatible Mode (Legacy topk_logprobs).

## 5. API Introduction

### 5.1 `analyze_output_anomaly`

**Function**

Detects anomalous responses during inference in real time. Runs Default Mode (Token-Only Sequence) by default; when non-empty `topk_logprobs` are supplied (including per-request mixed batches), automatically routes those requests through Compatible Mode (Legacy topk_logprobs).

**Prototype**

```python
# 1) Default Mode (Token-Only Sequence) signature (no logprobs required)
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

# 2) Compatible Mode (Legacy topk_logprobs) signature
analyze_output_anomaly(topk_logprobs, tokens, model_configs)
```

**Parameters**

- **`tokens`** (`Sequence[int]` | `Sequence[Sequence[int]]` | `np.ndarray`): Required. A 1D token ID sequence or 2D batch of sequences.
- **`model_configs`** (`str` | `os.PathLike` | tokenizer object | `List[Any]`):
  - **Required in Default Mode (Token-Only Sequence)**: path to a trusted local model directory (parses `tokenizer.json` first, or falls back to built-in slow tokenizers offline with `trust_remote_code=False`), a `tokenizer.json` file, an in-memory tokenizer object (HuggingFace fast/slow tokenizer, `tokenizers.Tokenizer`, `tiktoken`, etc.; pass an in-memory tokenizer object for models requiring custom tokenization code), or a batch-length list of them; a single target is broadcast across the batch.
  - **Optional in Compatible Mode (Legacy topk_logprobs)**: model name(s) consistent with `--model-name`, e.g., `['Qwen3-30B-A3B']*3` (passing a local model directory path works for both modes).
- **`topk_logprobs`** (`List[Optional[List[Dict[int, float]]]]`, optional): Per-request top-k logprobs. When omitted, `None`, or `None` for an individual request, Default Mode (Token-Only Sequence) is used for that request (requires a model directory, `tokenizer.json` path, or tokenizer instance in `model_configs`; passing only a bare model name raises `ValueError`).
- **`tier`** (`str`, optional, default `"intercept"`): Sensitivity tier for Default Mode (Token-Only Sequence) (`"intercept"` or `"monitor"`).
- **`think_end`** (`"auto"` | `int` | `None`, optional, default `"auto"`): Token ID separating reasoning from the final answer in Default Mode (Token-Only Sequence).
- **`repetition_config`** (`dict` | `str` | `os.PathLike`, optional, default `None`): Custom repetition detection rule dictionary or JSON file path in Default Mode (Token-Only Sequence); uses bundled [repetition.json](../../../python/msprobe/response_anomaly/configs/repetition.json) when `None`.
- **`rare_chars_model_path`** (`str` | `os.PathLike`, optional, default `None`): Custom rare-character scoring model path in Default Mode (Token-Only Sequence); uses bundled [rare_chars_model.npz](../../../python/msprobe/response_anomaly/tables/rare_chars_model.npz) when `None`.
- **`return_detail`** (`bool`, optional, default `False`): Returns `[[is_ill, ill_type], ...]` when `False`, or `List[DetectionResult]` when `True`.

> [!NOTE]
>
> In Compatible Mode (Legacy topk_logprobs), a single request must be wrapped in an outer `List`. In Default Mode (Token-Only Sequence), a 1D `tokens` sequence is automatically wrapped as a single-item batch.

**Returns**

Default return format: `[[is_ill, ill_type], ...]` (or `[DetectionResult(...), ...]` when `return_detail=True`).

Each request contains:

- `is_ill`: `bool`. Whether the output is anomalous (`True` indicates anomalous, `False` indicates normal).
- `ill_type`: `int`, anomaly type:
  - `0`: normal
  - `1`: rare characters (in Default Mode (Token-Only Sequence), covers both rare characters and UTF-8 corruption)
  - `2`: garbled text (Compatible Mode (Legacy topk_logprobs))
  - `3`: repetition
  - `4`: NaN Value (Compatible Mode (Legacy topk_logprobs))

**Example**

See the usage examples in [3. Default Mode (Token-Only Sequence)](#3-default-mode-token-only-sequence) and [4. Compatible Mode (Legacy topk_logprobs)](#4-compatible-mode-legacy-topk_logprobs).

### 5.2 `ILLDetector`

**Function**

Reusable detector holding configuration parameters, tokenizer lookup tables, and caches for high-throughput online or offline checks.

**Prototype**

```python
# 1) Default Mode (Token-Only Sequence) initialization (also supports Compatible Mode (Legacy topk_logprobs) calls)
detector = ILLDetector(
    model_or_tokenizer,
    *,
    think_end="auto",
    repetition_config=None,
    rare_chars_model_path=None,
)

# 2) Compatible Mode (Legacy topk_logprobs) config-path initialization (defaults to bundled configs when omitted)
detector = ILLDetector(
    config_path="./configs/config.yaml",
    mtype_path="./configs/mtype_config.json",
    tk2cat_path="./token2category/",
)
```

**Parameters**

- **`model_or_tokenizer`**: Path to a trusted local model directory (falls back offline with `trust_remote_code=False` when `tokenizer.json` is absent), path to a `tokenizer.json` file, or an in-memory tokenizer instance (pass an in-memory tokenizer instance for models requiring custom tokenization code) for Default Mode (Token-Only Sequence) initialization.
- **`think_end`**: Reasoning end token ID in Default Mode (Token-Only Sequence); defaults to `"auto"`.
- **`repetition_config`**: Optional custom repetition configuration dictionary or JSON path; defaults to bundled [repetition.json](../../../python/msprobe/response_anomaly/configs/repetition.json).
- **`rare_chars_model_path`**: Optional custom rare-character scoring model path; defaults to bundled [rare_chars_model.npz](../../../python/msprobe/response_anomaly/tables/rare_chars_model.npz).
- **`config_path`** / **`mtype_path`** / **`tk2cat_path`**: Optional configuration file, model metadata file, and token-to-category directory paths for Compatible Mode (Legacy topk_logprobs).

**Returns**

`detector` is an instance of `ILLDetector` providing the following methods for single-sequence or batch detection:

- `detect(tokens, tier="intercept", *, model_config=None) -> DetectionResult`: Runs Default Mode (Token-Only Sequence) detection on a single sequence.
- `detector(tokens, model_config=None, *, tier="intercept") -> DetectionResult` (or `detector(topk_logprobs, tokens, model_config=None)`): Single-request entry point returning [DetectionResult](#53-detectionresult).
- `run(tokens, model_configs=None, *, tier="intercept", return_detail=False)` (or `run(topk_logprobs, tokens, model_configs=None, *, return_detail=False)`): Batch entry point.

> [!TIP]
>
> Cold vocabulary parsing happens at startup; in long-running services, create and reuse `ILLDetector` instances (`detect` and `run` are stateless and safe to call repeatedly). In multi-model scenarios, cached lookup-table memory scales linearly with distinct tokenizers up to a process-wide dual-cap LRU limit (`64` entries and a default `256 MiB` byte budget, configurable via `MSPROBE_VOCAB_CACHE_MB`). A vocabulary exceeding the budget still works without evicting hot entries, and live detectors bound to an evicted vocabulary continue to function normally.

**Example**

See the usage examples in [3. Default Mode (Token-Only Sequence)](#3-default-mode-token-only-sequence).

### 5.3 `DetectionResult`

**Function**

Structured dataclass representing the anomaly detection result of a single output sequence.

**Definition**

```python
@dataclass
class DetectionResult:
    is_ill: bool = False
    ill_type: int = 0
    fired_at: int = -1
    reason: str = ""
```

**Attributes**

- **`is_ill`**: Whether an anomaly was detected.
- **`ill_type`**: Anomaly type code (`0`–`4`).
- **`fired_at`**: First checkpoint prefix length (1-based token count) where an anomaly triggers in Default Mode (Token-Only Sequence) (rule-dependency prefix for rare-character checks; 64-token block boundaries or sequence tail for repetition checks); `-1` when normal or in Compatible Mode (Legacy topk_logprobs).
- **`reason`**: Matched rule description in Default Mode (Token-Only Sequence); `""` when normal or in Compatible Mode (Legacy topk_logprobs).
