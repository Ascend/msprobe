# Precision Data Collection for Training and Inference in the slime Framework

## 1. Introduction

The slime framework adopts a training-inference decoupled architecture of "Megatron on the training side + SGLang on the inference side".

In the accuracy localization scenario of the slime framework, the precision data of the model forward process needs to be collected separately in the training and inference stages.

This document describes how to use the msProbe tool to complete the above two-stage precision data collection under the slime framework.

If you need to locate whether the source of the accuracy difference is the training or the inference stage, it is recommended to complete the training-inference consistency preprocessing before collection. For details, see "[Training-Inference Consistency Preprocessing and Data Collection in the slime Framework](./slime_consistency_preprocess_dump.md)".

## 2. Environment Preparation

Install the msProbe tool. For details, see [msProbe Tool Installation Guide](../../install_guide/msprobe_install_guide.md).

```shell
pip install mindstudio-probe
```

## 3. Creating the Configuration File

Both training and inference collection require creating a config.json configuration file to configure the collection behavior of the dump task. An example is as follows:

```json
{
    "task": "statistics",
    "dump_path": "/example_dump_path/dump",
    "rank": [],
    "step": [0],
    "level": "mix",
    "async_dump": false,
    "extra_info": true,
    "statistics": {
        "scope": [],
        "list": [],
        "tensor_list": [],
        "data_mode": [
            "all"
        ],
        "summary_mode": "statistics"
    }
}
```

Description of the main parameters:

- `task`: Collection task type. statistics indicates collecting statistics information; if real tensor data needs to be collected, it can be configured as tensor.
- `dump_path`: Save path of the dump data. Please modify it according to the actual situation.
- `step`: Number of steps to collect. For example, when only step0 is collected in the training stage, it is configured as [0]; for the usage in the inference stage, see the SGLang collection document referenced in section 5.
- `level`: Collection level. mix indicates collecting both Module-level and API-level data.

For detailed introduction and usage of the above configuration parameters, see "[Configuration File Introduction](../../user_guide/dump/config_json_introduct.md)".

## 4. Data Collection in the Training Stage

Precision data collection in the training stage is completed on the Megatron training backend of slime. Instantiate `PrecisionDebugger` in `slime/backends/megatron_utils/actor.py`, call the `start` and `stop` interfaces before and after the target computation stage to collect data, and call `step` to advance the step count after this collection ends; by modifying `dump_path`, the dump data of different stages can be saved to different subdirectories. For more introduction to the PrecisionDebugger interface, see [Precision Data Collection in the PyTorch Scenario](../../user_guide/dump/pytorch_data_dump_instruct.md).

### 4.1 Preparatory Operations

Enabling data collection in the training stage: slime starts training workers based on Ray, and environment variables need to be delivered through `runtime env` so that Ray subprocesses also take effect. Add the following environment variable to `env_vars` of the `runtime-env-json` of `ray job submit`:

```json
{
  "env_vars": {
    "DUMP_ON": "1",
    ...
  }
}
```

Among them, `DUMP_ON` is used to quickly switch the dump function: when set to 1, collection is enabled; when set to 0 or not set, the training process is completely consistent with the original logic.

### 4.2 Code Modification for slime

The following takes **simultaneously collecting the three stages of ref_log_prob, old_log_prob, and actor training** as an example to show the code modification method. Users can keep only the instrumentation of a certain stage or combine multiple stages as needed; each stage is written to disk in separate directories by modifying `dump_path`, and the calling position of `step()` is described at the end of this section.

The modified example code is highlighted as follows:

A. Instantiate `PrecisionDebugger` in the `init` method of `MegatronTrainRayActor`.

```diff
 class MegatronTrainRayActor(TrainRayActor):
     @with_defer(lambda: Timer().start("train_wait"))
     def init(
         self,
         args: Namespace,
         role: str,
         with_ref: bool = False,
     ) -> int | None:

+        # Modify in the init method
+        # Instantiate PrecisionDebugger
+        dump_flag = int(os.environ.get("DUMP_ON", 0))  # The environment variable DUMP_ON is used to quickly switch the dump function
+        if dump_flag:
+            from msprobe.pytorch import PrecisionDebugger, seed_all
+            seed_all(mode=True)  # Fix randomness to ensure that results are reproducible across multiple runs
+            self.debugger = PrecisionDebugger(config_path='/example_config_path/config.json')  # Replace with the actual config.json path
+            self.dump_path_prefix = self.debugger.config.dump_path
+        else:
+            self.debugger = None

         monkey_patch_torch_dist()

         super().init(args, role, with_ref)
         ...
```

B. In the `train_actor` method, insert the `start`/`stop` interfaces before and after the target computation stage. The following example covers the three stages of `ref_log_prob`, `old_log_prob`, and `actor` training at the same time.

```diff
     def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
         # Create data iterator for log_probs and train.
         data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
         ...
         with inverse_timer("train_wait"), timer("train"):
             if self.args.compute_advantages_and_returns:
                 if "ref" in self.weights_backuper.backup_tags:
                     ...
                     self._switch_model("ref")

+                    # Enable collection before ref_log_prob computation
+                    if self.debugger:
+                        self.debugger.service.config.dump_path = os.path.join(
+                            self.dump_path_prefix, "ref_compute_log_prob"
+                        )  # ref_log_prob data is saved in the ref_compute_log_prob folder
+                        self.debugger.start(model=self.model)

                     rollout_data.update(
                         self.compute_log_prob(
                             data_iterator,
                             num_microbatches,
                             store_prefix="ref_",
                         )
                     )

+                    # Stop collection after ref_log_prob computation
+                    if self.debugger:
+                        self.debugger.stop()

                 self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                 if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                     ...

+                    # Enable collection before old_log_prob computation
+                    if self.debugger:
+                        self.debugger.service.config.dump_path = os.path.join(
+                            self.dump_path_prefix, "old_log_prob"
+                        )  # old_log_prob data is saved in the old_log_prob folder
+                        self.debugger.start(model=self.model)

                     rollout_data.update(
                         self.compute_log_prob(
                             data_iterator,
                             num_microbatches,
                             store_prefix="",
                         )
                     )

+                    # Stop collection after old_log_prob computation
+                    if self.debugger:
+                        self.debugger.stop()

                     ...
             ...
             # Train
             if self.args.use_routing_replay:
                 os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"

+            # Enable collection before actor training
+            if self.debugger:
+                self.debugger.service.config.dump_path = os.path.join(
+                    self.dump_path_prefix, "actor_train"
+                )  # actor training data is saved in the actor_train folder
+                self.debugger.start(model=self.model)

             with timer("actor_train"):
                 train(
                     rollout_id,
                     self.model,
                     self.optimizer,
                     self.opt_param_scheduler,
                     data_iterator,
                     num_microbatches,
                 )

+            # Stop collection after actor training, and advance the step count
+            if self.debugger:
+                self.debugger.stop()
+                self.debugger.step()

             self.prof.step(rollout_id=rollout_id)
         ...
```

**Note**:

- The above example is for **simultaneously collecting** the three stages of `ref_log_prob`, `old_log_prob`, and `actor` training: each stage controls the collection interval through `start`/`stop`, and writes to disk in separate directories by modifying dump_path; `debugger.step()` is called only once after the stop of the last collection stage (actor training).
- If **only a single stage is collected**, keep the start/stop instrumentation of that stage, call `debugger.step()` immediately after stop, and delete the instrumentation of the other stages and the step call at the end of the example. For example, when only `ref_log_prob` is collected, add `self.debugger.step()` after its stop (about line 141 of the example code).

## 5. Data Collection in the Inference Stage

The inference side of the slime framework uses the SGLang engine to perform rollout generation. Precision data collection in the inference stage is completed on the SGLang side. Please select the corresponding document according to the SGLang version currently in use:

- **SGLang version < 0.5.11**: Intrusively modify the SGLang source code and insert the `PrecisionDebugger` interface in `ModelRunner`. For details, see [SGLang Precision Data Collection (SGLang Version < 0.5.11)](../../user_guide/dump/sglang_eager_dump_instruct.md).
- **SGLang version >= 0.5.11**: The msProbe tool is natively built in, and precision data collection can be performed directly by specifying the parameter `--sglang-msprobe-dump-config` in `SGLANG_ARGS`.

**Note**:

- When slime starts the inference engine, the `--disable-cuda-graph` parameter of the SGLang framework needs to be specified to disable graph mode; when the SGLang version is < 0.5.11, `--skip-server-warmup` also needs to be specified to disable warmup, avoiding collecting data from the warmup stage.
- If a dynamo-related error occurs, you can set the environment variable `export TORCHDYNAMO_DISABLE=1` to globally disable dynamo.

## 6. Introduction to the dump Result Files

The following introduces the directory structure of the dump results in the **training stage**. For the format of the dump results in the inference stage, see the SGLang collection document referenced in section 5 and [Precision Data Collection in the PyTorch Scenario](../../user_guide/dump/pytorch_data_dump_instruct.md).

If the three stages are collected at the same time as in the example of section 4.2, the directory structure is as follows; if only a single stage is collected, only the corresponding subdirectory (such as `ref_compute_log_prob`) is generated.

```shell
${dump_path}
├── ref_compute_log_prob        # ref_log_prob collection data
│   └── step0
│       └── rank{ID}
│           ├── dump.json
│           ├── stack.json
│           └── construct.json
├── old_log_prob                # old_log_prob collection data
│   └── ...
└── actor_train                 # actor training collection data
    └── ...
```

Description of directories and files at each level:

- `rank{ID}`: Device ID. The data of each rank is saved in the corresponding `rank{ID}` directory.
- `dump.json`: Saves the statistics information of the forward and backward data of the API or Module, including the API name or Module name of the dump data, the dtype, shape, max, min, mean, L2norm (L2 norm) statistics information of each data, and the verification value output according to the `summary_mode` configuration (`md5` corresponds to the `CRC-32` field, `xor` corresponds to the XOR verification field). For details, see the `dump.json` file description section of [Precision Data Collection in the PyTorch Scenario](../../user_guide/dump/pytorch_data_dump_instruct.md#dumpjson-description).
- `stack.json`: Call stack information of the API/Module.
- `construct.json`: Hierarchical structure information. When level is configured as L1, the content of `construct.json` is empty.
