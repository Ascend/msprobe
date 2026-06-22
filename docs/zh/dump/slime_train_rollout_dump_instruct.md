# slime框架训推精度数据采集

## 1. 简介

slime 框架采用训推分离架构：训练侧使用 Megatron 训练后端，推理侧使用 SGLang 引擎。在 slime 框架精度定位场景中，需分别在训练与推理阶段采集模型前向过程的精度数据。

本文介绍如何使用 msProbe 工具，在 slime 框架下完成上述两阶段的精度数据采集。

## 2. 环境准备

安装 msProbe 工具，详情请参见《[msProbe工具安装指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/msprobe_install_guide.md)》。

```shell
pip install mindstudio-probe
```

## 3. 创建配置文件

训推采集均需创建 config.json 配置文件，用于配置 dump 任务的采集行为，示例如下：

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

主要参数说明：

- task：采集任务类型，statistics 表示采集统计量信息；若需采集真实张量数据，可配置为 tensor。
- dump_path：dump 数据的保存路径，请根据实际情况修改。
- step：采集步数，训练阶段例如仅采集 step0 时配置为 [0]；推理阶段用法参见第 5 节引用的 SGLang 采集文档。
- level：采集级别，mix 表示同时采集 Module 级和 API 级数据。

以上配置参数的详细介绍和使用方法请参见《[配置文件介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/config_json_introduct.md)》。

## 4. 训练阶段数据采集

训练阶段精度数据采集在 slime 的 Megatron 训练后端完成。在 `slime/backends/megatron_utils/actor.py` 中实例化 `PrecisionDebugger`，并在目标计算阶段前后调用 `start`、`stop` 接口采集数据，并在该次采集结束后调用 `step` 推进步数；通过修改 `dump_path` 可将不同阶段的 dump 数据保存至不同子目录。PrecisionDebugger 接口的更多介绍请参见《[PyTorch场景精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/pytorch_data_dump_instruct.md)》。

### 4.1 前置操作

使能训练阶段数据采集：slime 基于 Ray 拉起训练 Worker，环境变量需通过 runtime env 下发，才能保证 Ray 子进程同样生效。在 ray job submit 的 runtime-env-json 的 env_vars 中增加如下环境变量：

```json
{
  "env_vars": {
    "DUMP_ON": "1",
    ...
  }
}
```

其中 DUMP_ON 用于快速开关 dump 功能：置为 1 时开启采集，置为 0 或不设置时训练流程与原始逻辑完全一致。

### 4.2 slime代码修改

以下以**同时采集 ref_log_prob、old_log_prob 与 actor 训练**三个阶段为例，展示代码修改方式。用户可按需仅保留某一阶段的插桩，或组合多个阶段；各阶段通过修改 `dump_path` 分目录落盘，`step()` 的调用位置见本节末尾说明。

修改示例代码高亮显示如下：

A. 在 MegatronTrainRayActor 的 init 方法中实例化 PrecisionDebugger。

```diff
 class MegatronTrainRayActor(TrainRayActor):
     @with_defer(lambda: Timer().start("train_wait"))
     def init(
         self,
         args: Namespace,
         role: str,
         with_ref: bool = False,
     ) -> int | None:

+        # init方法中修改
+        # 实例化PrecisionDebugger
+        dump_flag = int(os.environ.get("DUMP_ON", 0))  # 环境变量DUMP_ON用于快速开关dump功能
+        if dump_flag:
+            from msprobe.pytorch import PrecisionDebugger, seed_all
+            seed_all(mode=True)  # 固定随机性，保证多次运行结果可复现
+            self.debugger = PrecisionDebugger(config_path='/example_config_path/config.json')  # 替换为实际config.json路径
+            self.dump_path_prefix = self.debugger.config.dump_path
+        else:
+            self.debugger = None

         monkey_patch_torch_dist()

         super().init(args, role, with_ref)
         ...
```

B. 在 train_actor 方法中，于目标计算阶段前后插入 start/stop 接口。以下示例同时覆盖 ref_log_prob、old_log_prob 与 actor 训练三个阶段。

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

+                    # ref_log_prob计算前开启采集
+                    if self.debugger:
+                        self.debugger.service.config.dump_path = os.path.join(
+                            self.dump_path_prefix, "ref_compute_log_prob"
+                        )  # ref_log_prob数据保存在ref_compute_log_prob文件夹
+                        self.debugger.start(model=self.model)

                     rollout_data.update(
                         self.compute_log_prob(
                             data_iterator,
                             num_microbatches,
                             store_prefix="ref_",
                         )
                     )

+                    # ref_log_prob计算后停止采集
+                    if self.debugger:
+                        self.debugger.stop()

                 self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                 if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                     ...

+                    # old_log_prob计算前开启采集
+                    if self.debugger:
+                        self.debugger.service.config.dump_path = os.path.join(
+                            self.dump_path_prefix, "old_log_prob"
+                        )  # old_log_prob数据保存在old_log_prob文件夹
+                        self.debugger.start(model=self.model)

                     rollout_data.update(
                         self.compute_log_prob(
                             data_iterator,
                             num_microbatches,
                             store_prefix="",
                         )
                     )

+                    # old_log_prob计算后停止采集
+                    if self.debugger:
+                        self.debugger.stop()

                     ...
             ...
             # Train
             if self.args.use_routing_replay:
                 os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"

+            # actor训练前开启采集
+            if self.debugger:
+                self.debugger.service.config.dump_path = os.path.join(
+                    self.dump_path_prefix, "actor_train"
+                )  # actor训练数据保存在actor_train文件夹
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

+            # actor训练后停止采集，并推进step计数
+            if self.debugger:
+                self.debugger.stop()
+                self.debugger.step()

             self.prof.step(rollout_id=rollout_id)
         ...
```

**说明**：

- 以上示例为**同时采集** ref_log_prob、old_log_prob 与 actor 训练三个阶段：各阶段通过 start/stop 控制采集区间，并通过修改 dump_path 分目录落盘；`debugger.step()` 仅在最后一个采集阶段（actor 训练）的 stop 之后调用一次。
- 若**仅采集单个阶段**，保留该阶段的 start/stop 插桩，在 stop 之后立即调用 `debugger.step()`，并删去其余阶段插桩及示例末尾的 step 调用。例如，仅采集 ref_log_prob 时，需在其 stop 之后（示例代码约第 141 行）增加 `self.debugger.step()`。

## 5. 推理阶段数据采集

slime 框架推理侧使用 SGLang 引擎执行 rollout 生成。推理阶段的精度数据采集在 SGLang 侧完成，请根据当前使用的 SGLang 版本选择对应文档：

- **SGLang 版本 < 0.5.11**：需侵入式修改 SGLang 源码，在 `ModelRunner` 中插入 `PrecisionDebugger` 接口。详情请参见《[SGLang精度数据采集（SGLang版本<0.5.11）](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct.md)》。
- **SGLang 版本 >= 0.5.11**：SGLang 已原生内置 msProbe 能力，启动服务时通过 `--msprobe-dump-config` 指定配置文件即可。详情请参见《[SGLang精度数据采集（SGLang版本>=0.5.11）](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/sglang_eager_dump_instruct_new.md)》。

**注意**：

- slime 启动推理引擎时，需指定 SGLang 框架的 `--disable-cuda-graph` 参数关闭图模式；SGLang 版本 < 0.5.11 时还需指定 `--skip-server-warmup` 关闭 warmup，避免采集到 warmup 阶段的数据。
- 若遇到 dynamo 相关报错，可设置环境变量 `export TORCHDYNAMO_DISABLE=1` 全局关闭 dynamo。

## 6. dump结果文件介绍

以下介绍**训练阶段** dump 结果的目录结构。推理阶段 dump 结果格式详见第 5 节引用的 SGLang 采集文档及《[PyTorch场景精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/pytorch_data_dump_instruct.md)》。

若按 4.2 节示例同时采集三个阶段，目录结构如下；若仅采集单个阶段，则仅生成对应子目录（如 `ref_compute_log_prob`）。

```shell
${dump_path}
├── ref_compute_log_prob        # ref_log_prob采集数据
│   └── step0
│       └── rank{ID}
│           ├── dump.json
│           ├── stack.json
│           └── construct.json
├── old_log_prob                # old_log_prob采集数据
│   └── ...
└── actor_train                 # actor训练采集数据
    └── ...
```

各级目录及文件说明：

- rank{ID}：设备 ID，每张卡的数据保存在对应的 rank{ID} 目录下。
- dump.json：保存 API 或 Module 前反向数据的统计量信息。包含 dump 数据的 API 名称或 Module 名称，各数据的 dtype、shape、max、min、mean、L2norm（L2 范数）统计信息，以及根据 summary_mode 配置输出的校验值（md5 对应 CRC-32 字段，xor 对应 XOR 校验字段）。具体介绍请参见《[PyTorch场景精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/dump/pytorch_data_dump_instruct.md#dumpjson%E6%96%87%E4%BB%B6%E8%AF%B4%E6%98%8E)》的dump.json 文件说明章节。
- stack.json：API/Module 的调用栈信息。
- construct.json：分层分级结构信息，level 配置为 L1 时，construct.json 内容为空。
