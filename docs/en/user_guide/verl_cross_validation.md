# verl Training-Inference Cross-Instrumentation
<!-- md-trans-meta sourceCommit=a8426a4012af02b7cd501d5d585adb0c841097a8 translatedAt=2026-08-11T02:42:10.823Z pushedAt=2026-08-11T02:51:07.149Z -->

## Feature Overview

During the two phases (inference and training) of the reinforcement learning training process, precision issues generally require locating the problematic phase, and then collecting precision data from that phase for analysis. This document provides the guidelines for the following features based on the approach of replacing the output of one phase via instrumentation for issue demarcation.

| Feature | Description |
|------|------|
| **Basic Rollout Skip** | Skip the inference phase and directly load saved inference data, saving training time |
| **Enhanced Rollout Skip** | Support multiple calls to `generate_sequences`, avoiding repeated use of the same data |
| **Cross-Validation** | Two training processes cross-use inference data and checkpoints to locate precision issues |

---

## Basic Rollout Skip

Rollout Skip is a basic functionality provided by verl for skipping the inference phase and directly loading saved inference data, suitable for scenarios where experiments with the same configuration need to be run repeatedly.

**Supported Version**: verl v0.6.0 to v0.7.1

### Usage

Add the following parameters to the training launch command:

```diff
python train.py \
  actor_rollout_ref.rollout.skip_rollout=True \
  actor_rollout_ref.rollout.skip_dump_dir="/tmp/rollout_dump"
```

### Dumped File

File naming format: `{experiment_name}_{project_name}_GBS{gbs}__N{n}`

- `experiment_name`: experiment name
- `project_name`: project name
- `gbs`: generation batch size
- `n`: number of responses generated per prompt

### Effect

- First run: execute inference and save data
- Subsequent runs: direct loading of saved inference data, skipping the inference phase

---

## Enhanced Rollout Skip

The current basic functionality only supports saving data from a single `generate_sequences` call to disk. When `generate_sequences` is called multiple times, the basic functionality causes subsequent calls to reuse the data from the first call, resulting in inconsistency with the actual inference data. The enhanced functionality resolves this issue by introducing indexing.

### Code Modification

Modify the file `verl/utils/rollout_skip.py` as follows:

**Add an index variable in the `__init__` method of `class RolloutSkip`**:

```diff
 class RolloutSkip:
     def __init__(self, config, rollout_wg):
         self.rollout_config = config.actor_rollout_ref.rollout
         self.exp_name = config.data.get("experiment_name", "")
         self.project_name = config.data.get("project_name", "")
         self.n = int(self.rollout_config.get("n", 0))
         self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))
+        self.index = 0
```

**Add an index increment method in `class RolloutSkip`**:

```diff
+    def _add_index(self):
+        self.index += 1
```

**Modify the `curr_path_dump` attribute in `class RolloutSkip` to add an index suffix**:

```diff
     @property
     def curr_path_dump(self):
-        return self.dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}_GBS{self.gbs}__N{self.n}").absolute()
+        return self.dumped_dir.joinpath(
+            f"{self.exp_name}_{self.project_name}_GBS{self.gbs}_N{self.n}_{self.index}").absolute()
```

**Add an index increment call in the `wrap_generate_sequences` function**:

```diff
 def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
     generate_sequences = rollout_wg.generate_sequences
 
     def wrap_fn(batch, **kwargs):
         gen_batch_output = rolloutskip.try_load()
 
         if gen_batch_output is None:
             # * 1. Generation
             gen_batch_output = generate_sequences(batch, **kwargs)
             # * 2. Dump
             rolloutskip.dump(gen_batch_output)
+        rolloutskip._add_index()
         return gen_batch_output
     return wrap_fn
```

### Dumped File

File naming format: `{experiment_name}_{project_name}_GBS{gbs}_N{n}_{index}`

- Add the `{index}` suffix, which auto-increments with each call.

### Effect

- **First run**:
  - 1st call: execute inference and save as `..._N{n}_0`
  - 2nd call: execute inference and save as `..._N{n}_1`
  - And so on, each call generates an independent file

- **Subsequent runs**:
  - 1st call: Load `..._N{n}_0`
  - 2nd call: Load `..._N{n}_1`
  - And so on, each call loads the file with the corresponding index

- Avoid the issue of multiple calls reusing the same data.

### Usage

After the code modification is complete, the usage is the same as the basic functionality, requiring no additional parameters, and the index increments automatically.

```bash
python train.py \
  actor_rollout_ref.rollout.skip_rollout=True \
  actor_rollout_ref.rollout.skip_dump_dir="/tmp/rollout_dump"
```

---

## III. Cross-Validation

Through **training-inference cross-instrumentation**, the sharing and reuse of training and inference data are achieved, helping to delimit whether precision issues occur in the training phase or the inference phase.

### Working Principle

Two working modes:

| Mode       | Inference Phase                         | Training Phase                |
| ---------- | --------------------------------------- | ----------------------------- |
| **ckpt**   | Execute inference → Save rollout data | Execute training → Load checkpoints |
| **rollout** | Load rollout data (skip inference)  | Execute training → Save checkpoints |

The two training processes exchange data through a shared data directory, enabling the separation and cross-validation of training and inference.

### Constraints

- Currently support the verl reinforcement learning framework.
- Require two training processes to use the same configuration and data, and both training processes must be able to be launched simultaneously.
- The shared data path must be accessible and have sufficient storage space.

### Parameter Description

| Parameter           | Value              | Description                                                  |
| -------------- | ------------------ | ------------------------------------------------------------ |
| `share_data`     | "rollout"/"ckpt" | Working mode.<br>&#8226; rollout: Load inference data and save checkpoints.<br>&#8226; ckpt: Save inference data and load checkpoints. |
| `share_data_dir` | str                | Shared data storage path. Default value: `/tmp/verl/share_data_dir`. |

### Code Modification

Taking verl v0.7.1 as an example, the modifications are as follows:

#### 1. Add a Utility Class

Create the file `verl/utils/share_data.py`:

```python
import torch
from pathlib import Path
from verl.protocol import DataProto
import time
from functools import wraps

class ShareDataTest:

    print_mark = "[ShareDataTest]"

    def __init__(self, config):
        self.rollout_config = config

        self.rollout_index = 0  # Need to add
        self.ckpt_index = 0
        self.share_flag = self.rollout_config.get("share_data", "")

        self.dumped_dir = Path(self.rollout_config.get("share_data_dir", "/tmp/verl/share_data_dir"))
        self.dumped_dir.mkdir(parents=True, exist_ok=True)

        # Check if path is in Ray temporary directory
        if str(self.dumped_dir.absolute()).startswith("/tmp/ray/session"):
            print(
                f"\033[33m{self.print_mark} Warning: \nUsing dump path ",
                f"'{self.dumped_dir.absolute()}' is not recommended ",
                "as it's located in /tmp/ray/session*\033[0m",
                flush=True,
            )

        print(
            f"{self.print_mark} dump path set to: ",
            f"{self.dumped_dir.absolute()}",
            flush=True,
        )

    def _add_rollout_index(self):
        self.rollout_index += 1

    def _add_ckpt_index(self):
        self.ckpt_index += 1

    @property
    def curr_path_rollout(self):
        return self.dumped_dir.joinpath(
            f"rollout_{self.rollout_index}").absolute()

    @property
    def curr_path_ckpt(self):
        return self.dumped_dir.joinpath(
            f"ckpt_{self.ckpt_index}").absolute()

    def try_load(self):
        if not self.curr_path_rollout.exists():
            print(
                f"{self.print_mark} No data dump found at {self.curr_path_rollout}.",
                flush=True,
            )
            return None

        try:
            # * Load
            ret_batch = DataProto.load_from_disk(self.curr_path_rollout)
            print(
                f"\033[32m{self.print_mark} Successfully load pre-generated data from {self.curr_path_rollout}\033[0m",
                flush=True,
            )
            return ret_batch
        except Exception as e:
            print(
                f"\033[31m{self.print_mark} Failed to load pre-generated data from {self.curr_path_rollout}",
                f"Error: {str(e)}\033[0m",
                flush=True,
            )
            return None

    def dump(self, outputs: DataProto):
        try:
            outputs.save_to_disk(self.curr_path_rollout)
            print(
                f"\033[32m{self.print_mark} Successfully dump data in {self.curr_path_rollout}\033[0m",
                flush=True,
            )
        except Exception as e:
            print(
                f"\033[31m{self.print_mark} Failed to dump data in {self.curr_path_rollout}: {e}\033[0m",
                flush=True,
            )


def wrap_generate_sequences(share_data: ShareDataTest, worker):
    """Wrap the generate_sequences method to support data sharing"""
    original_generate_sequences = worker.generate_sequences

    @wraps(original_generate_sequences)
    def wrapped_fn(*args, **kwargs):
        if share_data.share_flag == "rollout":
            # rollout mode: Load data from shared file.
            gen_batch_output = None
            while gen_batch_output is None:
                print(f"\033[32m{share_data.print_mark} Waiting for shared data...\033[0m", flush=True)
                time.sleep(20)
                gen_batch_output = share_data.try_load()
                
        elif share_data.share_flag == "ckpt":
            # ckpt mode: Generate data and save to shared file.
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            share_data.dump(gen_batch_output)
        else:
            # Default mode: Execute directly.
            gen_batch_output = original_generate_sequences(*args, **kwargs)
        
        # Add index.
        share_data._add_rollout_index()
        return gen_batch_output

    return wrapped_fn

def after_update_policy(share_data, load_func, dump_func):
    """
    Choose to load or save checkpoints based on the configuration
    """
    # If checkpoint needs to be read:
    if share_data.share_flag == "ckpt":
        while not share_data.curr_path_ckpt.exists():
            print(f"\033[32m{share_data.print_mark} waiting for {share_data.curr_path_ckpt}\033[0m", flush=True)
            time.sleep(20)
        time.sleep(60)
        succ = False
        while not succ:
            try:
                load_func(share_data.curr_path_ckpt)
                succ = True
                print(f"\033[32m{share_data.print_mark} Successfully load ckpt from {share_data.curr_path_ckpt}\033[0m", flush=True)
            except Exception as e:
                print(f"\033[31m{share_data.print_mark} Load ckpt failed from {share_data.curr_path_ckpt} because of {e} \033[0m", flush=True)
                time.sleep(20)
    # Save checkpoint if needed.
    if share_data.share_flag == "rollout":
        try:
            dump_func(share_data.curr_path_ckpt)
            print(f"\033[32m{share_data.print_mark} Successfully save ckpt to {share_data.curr_path_ckpt}\033[0m", flush=True)
        except Exception as e:
            print(f"\033[31m{share_data.print_mark} Failed to save ckpt to {share_data.curr_path_ckpt}: {e}\033[0m", flush=True)
    share_data._add_ckpt_index()
```

### 2. Modify the Trainer Class

Modify the file `verl/trainer/ppo/ray_trainer.py`:

**Add initialization logic in the `fit()` function and wrap the `generate_sequences` method**:

```diff
 if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
     rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
     rollout_skip.wrap_generate_sequences()
+if self.config.trainer.get("share_data", None):
+    from verl.utils.share_data import ShareDataTest, wrap_generate_sequences, after_update_policy
+    share_data_manager = ShareDataTest(self.config.trainer)
+    self.async_rollout_manager.generate_sequences = \
+        wrap_generate_sequences(share_data_manager, self.async_rollout_manager)
```

**Add checkpoint synchronization after the `_update_actor` call in the training loop of the `fit()` function**:

```diff
 if self.config.trainer.critic_warmup <= self.global_steps:
     # update actor
     with marked_timer("update_actor", timing_raw, color="red"):
         actor_output = self._update_actor(batch)
+    if self.config.trainer.get("share_data", None):
+        after_update_policy(
+            share_data_manager, 
+            self.actor_rollout_wg.load_checkpoint, 
+            self.actor_rollout_wg.save_checkpoint
+        )
```

### Usage

#### Result Analysis

Delimitation approach for cross-validation:

**Background**: Training process A crashes, but it is unclear whether the issue lies in the training phase or the inference phase. A normal training process B (as a benchmark) is needed to perform cross-validation under the same experimental configuration.

**Method**:

| Scenario | Training Process A | Training Process B | Result |
|------|----------|----------|----------|
| **Scenario 1** | Inference + Load checkpoint | Load rollout +  Training | If training A crashes, it indicates an issue occurs in **the inference phase of A**. |
| **Scenario 2** | Load rollout + Training | Inference + Load checkpoint | If training A crashes, it indicates an issue occurs in **the training phase of A**. |

**Scenario 1** (delimit the inference phase):

Training process A (`ckpt` mode):

```bash
python train.py \
  ++trainer.share_data="ckpt" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- Inference phase: Execute inference and save rollout data
- Training phase: Load the checkpoint of training process B
  
Training process B (`rollout` mode):

```bash
python train.py \
  ++trainer.share_data="rollout" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- Inference phase: Load the rollout data of training process A
- Training phase: Execute training and save the checkpoint

**Scenario 2** (delimiting the training phase):

Training process A (`rollout` mode):

```bash
python train.py \
  ++trainer.share_data="rollout" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- Inference phase: Load the rollout data of training process B
- Training phase: Execute training and save the checkpoint

Training process B (`ckpt` mode):

```bash
python train.py \
  ++trainer.share_data="ckpt" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- Inference phase: Execute inference and save rollout data
- Training phase: Load the checkpoint of training process A

In this way, you can quickly locate at which phase of training process A the issue occurs.

### Log Description

The following log information is printed during execution for monitoring the data sharing status:

| Log Information                                           | Meaning                              |
| -------------------------------------------------- | --------------------------------- |
| `dump path set to: {path}`         | Prints the shared data directory path during initialization      |
| `Warning: Using dump path...`      | Warning: Ray temporary directory is used, not recommended |
| `Waiting for shared data...`       | Waiting for shared data in rollout mode        |
| `No data dump found at {path}`     | Shared data file not found                |
| `Successfully load pre-generated data from {path}` | Successfully loaded inference data (green)          |
| `Failed to load pre-generated data from {path}`    | Failed to load inference data (red)          |
| `Successfully dump data in {path}`                 | Successfully saved inference data (green)          |
| `Failed to dump data in {path}`                    | Failed to save inference data (red)          |
| `waiting for {ckpt_path}`                          | Waiting for checkpoint file in `ckpt` mode         |
| `Successfully load ckpt from {path}`               | Successfully loaded checkpoint (green)            |
| `Load ckpt failed from {path}`                     | Failed to load checkpoint (red)            |
| `Successfully save ckpt to {path}`                 | Successfully saved checkpoint (green)            |
| `Failed to save ckpt to {path}`                    | Failed to save checkpoint (red)            |

### Data Description

File naming format in the shared data directory:

- **Inference data**: `rollout_{index}`, where `index` increments from 0
- **Checkpoint data**: `ckpt_{index}`, where `index` increments from 0

The two training processes maintain data synchronization through the same index.
