# Checkpoint Comparison

## Overview

During or after model training, some checkpoint files may be saved to record the model and optimizer training status.

Checkpoint comparison is to compare two different checkpoints and evaluate model similarity.

Currently, the checkpoints of Megatron-LM and MindSpeed (PyTorch & MindTorch) can be compared. TP, PP, EP, and VPP are supported, as well as `megatron.core`, `megatron.legacy`, and `TransformerEngine`.

## Preparations

Install msProbe by referring to [msProbe Installation Guide](msprobe_install_guide.md).

## Checkpoint Comparison Description

**Function**

Compares two different checkpoints.

**Precautions**

- The checkpoints of Megatron-LM and MindSpeed are loaded based on megatron. Ensure that megatron has been installed in the Python environment or the megatron code has been saved in the current path.
- Before passing checkpoints to the tool for loading, ensure that checkpoints are secure and reliable. If the official checkpoint source provides a verification value such as SHA256, you must verify checkpoints to ensure that they are not tampered with.

**Syntax**

```bash
msprobe config_check --compare <ckpt_path1> <ckpt_path2> [-o <output_path.json>]
```

**Parameters**

| Parameter       | Mandatory (Yes/No)| Description                                                    |
| ------------- | --------- | ------------------------------------------------------------ |
| `-c` or `--compare`| Yes     | Performs the comparison operation. `ckpt_path1` and `ckpt_path2` are the paths of the two checkpoints to be compared. For details about the path configuration, see [Checkpoint Path Description](#ckpt_path).|
| `-o` or `--output` | No     | Output path of the comparison result. The default value is `./ckpt_similarity.json`. You can customize the file name. If the output path already exists, an error will be reported and the operation will be terminated.|

**Checkpoint Path Description**<a name="ckpt_path"></a>

The following is an example of the checkpoint directory structure of Megatron-LM and MindSpeed:

```txt
directory_name/
├── iter_0000005/    # Checkpoint directory of a certain iteration
│   └── mp_rank_xx_xxx/    # Checkpoint directory of a single rank. *xx_xxx* indicates the model parallel index.
│       └── model_optim_rng.pt    # PyTorch binary file containing model parameters and random states.
├── iter_0000010/
├── latest_checkpointed_iteration.txt  # Plain text file that records the last saved checkpoint.
```

For the two paths specified by the `--compare` parameter:

- If this parameter is set to `directory_name`, the tool automatically selects the last saved checkpoint for comparison based on the `latest_checkpointed_iteration.txt` file.
- If this parameter is set to `directory_name/iter_xxxxxxx`, the tool uses the checkpoint of the specified iteration for comparison.
- Currently, comparison of a single rank is not supported.

**Example**

Run the following command to perform the comparison:

```bash
msprobe config_check --compare ckpt_path1 ckpt_path2 -o output_path.json
```

**Output Description**

After the comparison is complete, the output path of the comparison result JSON file is displayed. For details, see [Output File Description](#output-file-description).

## Output File Description

The checkpoint comparison result is exported to a JSON file. The following is an example:

```json
{
    "decoder.layers.0.input_layernorm.weight": {
        "l2": 0.0, 
        "cos": 0.999999,
        "numel": 128,
        "shape": [
            128
        ]
    },
    "decoder.layers.0.pre_mlp_layernorm.weight": {
        "l2": 0.012, 
        "cos": 0.98,
        "numel": 128,
        "shape": [
            128
        ]
    }
}
```

Statistics| Description|
|-------|---------|
| l2 | Euclidean distance, $\|\|a-b\|\|_2$.|
| cos | Cosine similarity, $\frac{<a,b>}{\|\|a\|\|_2\|\|b\|\|_2}$.|
| numel | Number of elements in a parameter.|
| shape | Shape of a parameter.|
