# VeRL 训推交叉打桩功能指南

## 功能概述

在强化学习训练过程中的两个阶段（推理、训练），精度问题一般需要先定界出问题的阶段，再去采集该阶段精度数据进行分析。本文基于打桩替换掉一个阶段输出的思路进行定界，提供以下功能的使用指南：

| 功能 | 说明 |
|------|------|
| **Rollout Skip 基础功能** | 跳过推理阶段，直接加载已保存的推理数据，节省训练时间 |
| **Rollout Skip 增强功能** | 支持多次调用 generate_sequences，避免重复使用相同数据 |
| **交叉验证** | 两个训练过程交叉使用推理数据和检查点，实现精度定界 |

---

## 一、Rollout Skip 基础功能

Rollout Skip是VeRL 提供的基础功能，用于跳过推理阶段，直接加载已保存的推理数据，适用于需要反复运行相同配置实验的场景。

**版本支持**：VeRL v0.6.0 - v0.7.1

### 使用方式

在训练启动命令中添加以下参数：

```diff
python train.py \
  actor_rollout_ref.rollout.skip_rollout=True \
  actor_rollout_ref.rollout.skip_dump_dir="/tmp/rollout_dump"
```

### 落盘文件

文件命名格式：`{experiment_name}_{project_name}_GBS{gbs}__N{n}`

- `experiment_name`：实验名称
- `project_name`：项目名称
- `gbs`：生成批次大小
- `n`：每个 prompt 生成的回复数量

### 使用效果

- 首次运行：执行推理并保存数据
- 后续运行：直接加载已保存的推理数据，跳过推理阶段

---

## 二、Rollout Skip 增强功能

当前基础功能仅支持一次 `generate_sequences` 调用的数据落盘，当 `generate_sequences` 被多次调用时，基础功能会导致后续调用重复使用第一次的数据，和实际的推理数据不一致。增强功能通过添加索引解决此问题。

### 代码修改

修改文件 `verl/utils/rollout_skip.py`：

**在 `class RolloutSkip` 的 `__init__` 方法中添加索引变量：**

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

**在 `class RolloutSkip` 中添加索引递增方法：**

```diff
+    def _add_index(self):
+        self.index += 1
```

**在 `class RolloutSkip` 中修改 `curr_path_dump` 属性，添加索引后缀：**

```diff
     @property
     def curr_path_dump(self):
-        return self.dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}_GBS{self.gbs}__N{self.n}").absolute()
+        return self.dumped_dir.joinpath(
+            f"{self.exp_name}_{self.project_name}_GBS{self.gbs}_N{self.n}_{self.index}").absolute()
```

**在 `wrap_generate_sequences` 函数中添加索引递增调用：**

```diff
 def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
     generate_sequences = rollout_wg.generate_sequences
 
     def warp_fn(batch, **kwargs):
         gen_batch_output = rolloutskip.try_load()
 
         if gen_batch_output is None:
             # * 1. Generation
             gen_batch_output = generate_sequences(batch, **kwargs)
             # * 2. Dump
             rolloutskip.dump(gen_batch_output)
+        rolloutskip._add_index()
         return gen_batch_output
     return warp_fn
```

### 落盘文件

文件命名格式：`{experiment_name}_{project_name}_GBS{gbs}_N{n}_{index}`

- 增加了 `{index}` 后缀，每次调用自动递增

### 使用效果

- **首次运行**：
  - 第1次调用：执行推理并保存为 `..._N{n}_0`
  - 第2次调用：执行推理并保存为 `..._N{n}_1`
  - 以此类推，每次调用生成独立文件

- **后续运行**：
  - 第1次调用：加载 `..._N{n}_0`
  - 第2次调用：加载 `..._N{n}_1`
  - 以此类推，每次调用加载对应索引的文件

- 避免多次调用重复使用相同数据的问题

### 使用方式

代码修改完成后，使用方式与基础功能相同，无需额外参数，索引自动递增：

```bash
python train.py \
  actor_rollout_ref.rollout.skip_rollout=True \
  actor_rollout_ref.rollout.skip_dump_dir="/tmp/rollout_dump"
```

---

## 三、交叉验证功能

通过**训推交叉打桩**的方式，实现训练和推理数据的共享与复用，帮助定界精度问题发生在训练还是推理阶段。

### 工作原理

两种工作模式：

| 模式        | 推理阶段                       | 训练阶段                |
| ----------- | ------------------------------ | ----------------------- |
| **ckpt**    | 执行推理 → **保存**rollout数据 | 执行训练 → **加载**ckpt |
| **rollout** | **加载**rollout数据(跳过推理)  | 执行训练 → **保存**ckpt |

两个训练过程通过共享数据目录进行数据交换，实现训练和推理的分离与交叉验证。

### 约束限制

- 当前支持VeRL强化学习框架
- 需要两个训练过程使用相同的配置和数据，且两个训练过程能同时拉起
- 共享数据路径需要可访问且具有足够存储空间

### 参数说明

| 参数           | 选项值             | 参数说明                                                     |
| -------------- | ------------------ | ------------------------------------------------------------ |
| share_data     | "rollout" / "ckpt" | 工作模式选择。<br>&#8226; rollout：加载推理数据，保存检查点。<br>&#8226; ckpt：保存推理数据，加载检查点。 |
| share_data_dir | str                | 共享数据存储路径。默认值为：`/tmp/verl/share_data_dir`。     |

### 代码修改说明

以VeRL v0.7.1 版本为例，修改如下：

#### 1. 新增工具类

创建文件 `verl/utils/share_data.py`:

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

        self.rollout_index = 0  # 需要添加
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
    """包装generate_sequences方法，支持数据共享"""
    original_generate_sequences = worker.generate_sequences

    @wraps(original_generate_sequences)
    def wrapped_fn(*args, **kwargs):
        if share_data.share_flag == "rollout":
            # rollout模式：从共享文件加载数据
            gen_batch_output = None
            while gen_batch_output is None:
                print(f"\033[32m{share_data.print_mark} Waiting for shared data...\033[0m", flush=True)
                time.sleep(20)
                gen_batch_output = share_data.try_load()
                
        elif share_data.share_flag == "ckpt":
            # ckpt模式：生成数据并保存到共享文件
            gen_batch_output = original_generate_sequences(*args, **kwargs)
            share_data.dump(gen_batch_output)
        else:
            # 默认模式：直接执行
            gen_batch_output = original_generate_sequences(*args, **kwargs)
        
        # 增加索引
        share_data._add_rollout_index()
        return gen_batch_output

    return wrapped_fn

def after_update_policy(share_data, load_func, dump_func):
    """
    根据配置选择加载ckpt还是保存ckpt
    """
    # 如果需要读ckpt
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
    # 如果需要存ckpt
    if share_data.share_flag == "rollout":
        try:
            dump_func(share_data.curr_path_ckpt)
            print(f"\033[32m{share_data.print_mark} Successfully save ckpt to {share_data.curr_path_ckpt}\033[0m", flush=True)
        except Exception as e:
            print(f"\033[31m{share_data.print_mark} Failed to save ckpt to {share_data.curr_path_ckpt}: {e}\033[0m", flush=True)
    share_data._add_ckpt_index()
```

### 2. 修改Trainer类

修改文件 `verl/trainer/ppo/ray_trainer.py`:

**在 `fit()` 函数中添加初始化逻辑，并包装 `generate_sequences` 方法：**

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

**在 `fit()` 函数的训练循环中，`_update_actor` 调用之后添加检查点同步：**

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

### 使用步骤

#### 结果分析

交叉验证的定界思路：

**背景**：训练过程A出现崩溃，但无法确定是训练阶段还是推理阶段的问题。需要借助一个正常的训练过程B（作为标杆），在相同实验配置下进行交叉验证。

**定界方法**：

| 场景 | 训练过程A | 训练过程B | 结果判断 |
|------|----------|----------|----------|
| **场景1** | 自身推理 + 加载ckpt | 加载rollout + 自身训练 | 如果训练A崩溃，说明**A的推理阶段**有问题 |
| **场景2** | 加载rollout + 自身训练 | 自身推理 + 加载ckpt | 如果训练A崩溃，说明**A的训练阶段**有问题 |

**场景1测试**（定界推理阶段）：

训练过程A（`ckpt` 模式）：

```bash
python train.py \
  ++trainer.share_data="ckpt" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- 推理阶段：执行推理并保存rollout数据
- 训练阶段：加载训练过程B的检查点

训练过程B（`rollout` 模式）：

```bash
python train.py \
  ++trainer.share_data="rollout" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- 推理阶段：加载训练过程A的rollout数据
- 训练阶段：执行训练并保存检查点

**场景2测试**（定界训练阶段）：

训练过程A（`rollout` 模式）：

```bash
python train.py \
  ++trainer.share_data="rollout" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- 推理阶段：加载训练过程B的rollout数据
- 训练阶段：执行训练并保存检查点

训练过程B（`ckpt` 模式）：

```bash
python train.py \
  ++trainer.share_data="ckpt" \
  ++trainer.share_data_dir="/root/autodl-tmp/share_data"
```

- 推理阶段：执行推理并保存rollout数据
- 训练阶段：加载训练过程A的检查点

通过这种方式，可以快速定位问题发生在训练过程A的哪个阶段。

### 日志说明

运行过程中会打印以下日志信息，用于监控数据共享状态：

| 日志信息                                           | 含义                              |
| -------------------------------------------------- | --------------------------------- |
| `dump path set to: {path}`         | 初始化时打印共享数据目录路径      |
| `Warning: Using dump path...`      | 警告：使用了 Ray 临时目录，不推荐 |
| `Waiting for shared data...`       | rollout 模式下等待共享数据        |
| `No data dump found at {path}`     | 未找到共享数据文件                |
| `Successfully load pre-generated data from {path}` | 成功加载推理数据（绿色）          |
| `Failed to load pre-generated data from {path}`    | 加载推理数据失败（红色）          |
| `Successfully dump data in {path}`                 | 成功保存推理数据（绿色）          |
| `Failed to dump data in {path}`                    | 保存推理数据失败（红色）          |
| `waiting for {ckpt_path}`                          | ckpt 模式下等待检查点文件         |
| `Successfully load ckpt from {path}`               | 成功加载检查点（绿色）            |
| `Load ckpt failed from {path}`                     | 加载检查点失败（红色）            |
| `Successfully save ckpt to {path}`                 | 成功保存检查点（绿色）            |
| `Failed to save ckpt to {path}`                    | 保存检查点失败（红色）            |

### 数据说明

共享数据目录下的文件命名格式：

- **推理数据**：`rollout_{index}`，index 从 0 开始递增
- **检查点数据**：`ckpt_{index}`，index 从 0 开始递增

两个训练过程通过相同的 index 保持数据同步。
