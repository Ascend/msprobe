# vLLM KV Cache 记录与校验

## 简介

KV Cache 用于保存模型生成过程中各 token 的 Key/Value 数据。vLLM 按缓存块（block）管理这些数据，一个缓存块包含多个 token 的存储槽位（slot）。调度器为每个请求安排使用的缓存块，模型执行时通过 `block_table` 指定读取的缓存块，通过 `slot_mapping` 指定各 token 的写入槽位。

KV Trace 记录请求标识、缓存块编号、token 位置和读写槽位，并检查调度器分配的缓存块与模型输入中指定的读写位置是否一致。例如，某个 token 应写入 slot 80，模型输入指定 slot 96 时，日志会给出请求、rank、token 索引及两者的差异，辅助定位问题。

## 使用前准备

**环境准备**

在启动 vLLM 的 Python 环境中执行以下命令，从源码安装 msprobe：

```shell
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
pip install --no-deps .
```

**约束**

- 本功能适用于 vLLM Ascend，支持 Eager 和图模式。
- 模型推理采用 Model Runner V2（MRV2），在启动前配置 `VLLM_USE_V2_MODEL_RUNNER=1`。

## 功能介绍

**功能说明**

记录每轮调度的请求、缓存块、读写位置和结果，检查请求标识、token 数量、缓存块编号和写入槽位的一致性。用户可从服务日志查看校验结果，并按请求、rank 和 KV Cache 组定位异常。

**命令格式**

```shell
vllm serve <model_path> --enforce-eager --additional-config '<additional_config>'
```

**参数说明**

| 参数名 | 可选/必选 | 参数说明 |
| --- | --- | --- |
| `<model_path>` | 必选 | 模型路径 |
| `--additional-config` | 必选 | 启用本功能的 JSON 配置，包含 `kv_trace` 对象。配置示例：`--additional-config '{"kv_trace":{"enabled":true,"action":"log","slot":true}}'`。各字段含义见 [kv_trace 对象参数](#kv_trace_parameters) |

**kv_trace 对象参数**<a name="kv_trace_parameters"></a>

| 参数名 | 可选/必选 | 参数说明 |
| --- | --- | --- |
| `enabled` | 可选 | 布尔值，`true` 开启 KV Trace，`false` 关闭。服务默认关闭 KV Trace；提供 `kv_trace` 配置后，此字段默认值为 `true` |
| `action` | 可选 | `log`（默认）：记录校验异常并继续推理；`raise`：记录校验异常并抛出 `KVGuardError`，推理引擎结束运行 |
| `slot` | 可选 | 布尔值，默认 `false`。`true` 开启 slot 地址计划检查，配合 `enabled=true` 使用。适用场景见 [slot 参数说明](#slot_parameter) |

**slot 参数说明**<a name="slot_parameter"></a>

开启 `slot` 后，KV Trace 核对调度器给每个 token 分配的缓存槽位与模型输入指定的写入位置。slot 检查适用于采用普通批次（DBO 关闭）的仅解码器模型常规解码。

| 场景 | 配置与检查内容 |
| --- | --- |
| 本地 KV Cache | 支持预填充上下文并行（PCP）和解码上下文并行（DCP）。PCP 核对各进程的 token 索引、写入掩码与全局槽位，DCP 按当前 rank 的缓存分片核对写入槽位 |
| 预填充与解码分离（PD） | 使用单个全注意力缓存组，TP、PP、PCP、DCP 均设为 1。P 端将 `MooncakeConnectorV1` 配置为 `kv_producer`，D 端配置为 `kv_consumer`。P 端核对导出块与请求绑定，D 端在读取返回后核对整次传输的源地址、目标地址和覆盖范围 |
| 抢占换出与恢复换入 | 使用 `PreemptOffloadConnector` 的 `kv_consumer` 角色，采用单个全注意力缓存组，TP、PP、PCP、DCP 均设为 1。检查每批传输的源块、目标块、缓存容量及完成事件 |

启动日志出现 `parameter=enable_dbo, action: resetting to False` 时，服务按普通批次运行，slot 检查结果也按普通批次解读。

**使用示例**

如果通过 `VLLM_PLUGINS` 指定启用的插件，将 `msprobe_kv_guard` 加入已有列表。以下示例仅选择本插件，在启动服务前执行：

```shell
export VLLM_PLUGINS=msprobe_kv_guard
```

同时启用其他插件时，各插件名称用英文逗号分隔。

以下示例使用 Eager 模式开启 KV Trace 和 slot 检查，并将服务日志保存到 `server.log`：

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/model \
  --enforce-eager \
  --additional-config '{"kv_trace":{"enabled":true,"action":"log","slot":true}}' \
  > server.log 2>&1
```

排查 token 写入槽位偏移、尾块越界或请求间槽位冲突时，使用 `slot=true` 开启检查。

将 `/path/to/model` 替换为模型路径。使用图模式时，删除上述命令中的 `--enforce-eager` 参数，并沿用服务现有的编译配置。

日常排查可使用 `log` 收集上下文，需要在校验失败后停止推理时使用 `raise`。张量采集、统计值采集和 NaN 检测按各自配置启用。

**输出说明**

KV Trace 日志输出到进程标准输出，上述启动命令通过重定向将日志保存为 `server.log`。容器部署时，也可从容器日志中检索 `[KV_GUARD]`。开启 slot 检查后，`kv.enabled` 记录 `slot=true` 和 `evidence_scope=model_runner_plan`；slot 校验异常通过带有 `audit=slot` 的 `kv.violation` 记录输出。日志格式和字段含义请参见[日志输出结果说明](#日志输出结果说明)。

## 日志输出结果说明

### 日志格式与示例

每条记录由日志前缀、`[KV_GUARD]`、事件名和 JSON 字段组成。常规记录为 INFO 级别，`kv.violation` 和 `kv.error` 为 ERROR 级别。

以下为格式说明示例，展示分析所需的部分字段，省略时间、进程号等日志前缀。示例请求 `req-1` 在本轮处理两个 token，使用缓存块 5：

```text
[KV_GUARD] kv.enabled {"rank":0,"action":"log","boundary":"worker.init_device.return","slot":true,"evidence_scope":"model_runner_plan"}
[KV_GUARD] kv.bind {"trace_id":"0123456789abcdef0123456789abcdef","evidence":"scheduler_plan","num_scheduled_tokens":{"req-1":2},"total_num_scheduled_tokens":2,"scheduled_new_reqs":[{"req_id":"req-1","block_ids":[[5]],"num_computed_tokens":0,"replay_start":0}]}
[KV_GUARD] kv.write {"trace_id":"0123456789abcdef0123456789abcdef","rank":0,"req_ids":["req-1"],"evidence":"model_runner_plan","execution_mode":"NONE","group_id":0,"block_size":16,"kernel_block_size":16,"num_tokens":2,"query_start_loc":[0,2],"positions":[0,1],"slot_mapping_scope":"local_tokens","slot_mapping":[80,81]}
[KV_GUARD] kv.read {"trace_id":"0123456789abcdef0123456789abcdef","rank":0,"req_ids":["req-1"],"evidence":"model_runner_plan","execution_mode":"NONE","group_id":0,"block_size":16,"kernel_block_size":16,"seq_lens":[2],"block_table":[[5]]}
[KV_GUARD] boundary {"trace_id":"0123456789abcdef0123456789abcdef","rank":0,"boundary":"worker.output","phase":"observed","evidence":"native_result","req_ids":["req-1"],"req_id_to_index":{"req-1":0},"num_output_tokens":{"req-1":1}}
```

### 事件含义

| 事件 | 用户可查看的信息 |
| --- | --- |
| `kv.enabled` | 功能已启用，以及当前进程的 rank、异常处理方式 |
| `kv.bind` | 本轮调度的请求、block 绑定、token 预算，以及结束、抢占、恢复、复制、清零等调度信息 |
| `kv.write` | 模型输入中每组 KV Cache 指定的写入 slot、token 位置；占位输入（例如陪跑）的记录使用 `evidence=dummy_plan` |
| `kv.read` | 模型输入中每组 KV Cache 指定的读取 block table 和序列长度 |
| `boundary` | 推理各阶段的输入输出；通过 `boundary`、`evidence` 和 `phase` 查看记录来源 |
| `kv.violation` | KV Trace 检测到的校验异常，`reason` 给出异常类型，包括请求对应、读写位置、采样计数和占位输入等问题 |
| `kv.error` | 推理执行过程中抛出的异常，`error` 及可用的 `error_type` 提供错误信息 |

### 关键字段

| 字段 | 含义与读法 |
| --- | --- |
| `trace_id` | 一轮调度的关联标识，用于汇总调度器（Scheduler）、执行进程（Worker）和读写位置记录 |
| `req_id` / `req_ids` / `request_id` | 分别出现在调度项、批次列表和部分异常记录中的请求标识 |
| `rank` | 分布式执行进程的逻辑编号，用于区分执行该请求的进程 |
| `dp_rank` | 数据并行实例编号，每个实例处理各自的请求批次 |
| `group_id` | KV Cache 组号，从 0 开始，用于区分模型中的不同缓存组 |
| `num_scheduled_tokens` | 每个请求本轮的 token 调度预算 |
| `query_start_loc` | 请求在批次 token 数组中的分界。例如 `[0,2,5]` 表示两个请求分别占用 `[0,2)`、`[2,5)` |
| `positions` / `seq_lens` | token 在请求序列中的位置，以及本轮的序列长度 |
| `block_ids` / `block_table` | 调度记录的缓存块，以及模型输入中指定读取的缓存块表；按 KV Cache 组和请求行对应查看 |
| `slot_mapping` | 模型输入中每个 token 指定的写入槽位；`-1` 表示填充槽位（PAD） |
| `block_size` | 调度器管理的每个缓存块容纳的 token 数，`block_ids` 使用这种块编号 |
| `kernel_block_size` | 模型读取时每个缓存块容纳的 token 数，`block_table` 使用这种块编号 |
| `evidence` / `phase` | 记录的上下文，例如调度记录 `scheduler_plan`、模型执行输入 `model_runner_plan`、陪跑输入 `dummy_plan`、结果观测 `observed` |
| `coverage` | KV 传输记录的字段完整性。`selected_public_fields` 表示已记录可用的公开字段；`not_expanded` 表示仅提供元数据类型及可用的请求标识 |

| `execution_mode` | 本轮模型输入采用的执行模式：`NONE` 表示 Eager，`FULL` 表示完整图模式。FULL 服务的预填充阶段也可采用 `NONE` |
| `slot` / `evidence_scope` | 在 `kv.enabled` 中查看 slot 检查开关及检查对象；`slot=true`、`evidence_scope=model_runner_plan` 表示检查模型输入指定的写入位置 |
| `audit` | `kv.violation` 中的检查类别；`slot` 表示 slot 校验异常 |
| `expected_slot` / `actual_slot` | 请求获准使用的槽位与模型输入指定的槽位，结合 `request_id` 和 `token_index` 定位差异 |
| `slot` / `capacity` | 在异常记录中查看冲突或越界槽位及缓存总容量。例如 `capacity=1024` 时，有效槽位为 0～1023 |
| `source` | 异常位置的来源：`scheduler` 为调度记录给出的目标位置，`execution` 为模型输入指定的位置 |

## 结果分析

### 1. 确认启用并查找异常

```shell
grep -F '[KV_GUARD] kv.enabled ' server.log
grep -E '\[KV_GUARD\] kv\.(violation|error) ' server.log
```

出现 `kv.enabled` 表示对应进程已开启 KV Trace。`kv.violation` 从 `reason` 入手，`kv.error` 结合 `error` 和服务异常堆栈分析。

### 2. 按调度、请求和 rank 汇总上下文

从异常记录取得 `trace_id`，筛选该轮日志：

```shell
grep -F '"trace_id":"0123456789abcdef0123456789abcdef"' server.log
```

将示例值替换为实际标识，再根据 `request_id`、`req_ids`、`rank` 和 `group_id` 找到相关记录。需要查看请求的多轮变化时，可检索请求标识：

```shell
grep -F '"req-1"' server.log
```

多个进程的日志可能交错输出，按关联字段组织记录。`trace_id` 为 `null` 时，使用该条记录的进程、rank、`boundary` 和时间信息查看上下文。

### 3. 对照调度与读写位置

按 `kv.bind → kv.write / kv.read → boundary` 阅读同轮记录：

1. 从 `kv.bind` 确认请求、token 预算和所属 KV Cache 组的 `block_ids`。查看持续调度的请求时，将前序块列表与 `scheduled_cached_reqs.new_block_ids` 合并，得到当前块列表；查看 `resumed_req_ids` 中的恢复请求时，使用本轮块列表。
2. 从 `kv.write` 的 `req_ids` 和 `query_start_loc` 找到请求对应的 token 区间，对照 `positions` 与 `slot_mapping`。
3. 从 `kv.read` 查看同一请求、rank、组号的 `block_table` 和 `seq_lens`。`block_size` 与 `kernel_block_size` 相同时，可直接对照缓存块编号；两者有差异时，先按各自块大小换算。
4. 从结果 `boundary` 查看请求和输出 token 数。`worker.output` 的 `phase=observed` 表示 Worker 已观察到结果，服务最终状态结合校验日志与异常堆栈判断。

以上单卡示例的块大小为 16，位置 0、1 位于 block 5，对应 slot 为 `5 × 16 + 0 = 80`、`5 × 16 + 1 = 81`；`seq_lens=[2]` 表示该请求本轮序列长度为 2。

对于两种块大小有差异的单卡场景，例如 `block_size=16`、`kernel_block_size=8`，调度 block 5 对应 `block_table` 中的两个块 10、11，分别覆盖槽位 80～87、88～95。此时将 `block_ids=[[5]]` 与 `block_table=[[10,11]]` 对照，位置 0、1 的写入槽位仍为 80、81。

使用滑动窗口（只保留最近一段上下文）时，`block_table` 中超出窗口的旧前缀可用 0 占位；结合当前 `positions` 和校验日志分析这些项。

### 4. 根据异常类型定位差异

假设上述示例中，模型输入的 `slot_mapping` 将首个 token 的写入槽位从 80 改为 96，则可能出现以下校验日志：

```text
[KV_GUARD] kv.violation {"reason":"write_target_mismatch","rank":0,"evidence":"boundary_plan","trace_id":"0123456789abcdef0123456789abcdef","request_id":"req-1","group_id":0,"token_index":0,"expected_slot":80,"actual_slot":96,"audit":"slot"}
```

这条日志表明 rank 0、组 0 中，`req-1` 对应的批次 token 索引 0 应写入 slot 80，模型输入指定 slot 96。结合 `kv.bind` 中的缓存块编号和 `kv.write` 中的 token 位置、写入槽位，即可定位该请求的差异。

| `reason` 示例 | 重点查看 |
| --- | --- |
| `write_target_mismatch` | `expected_slot`、`actual_slot`、`token_index`，以及对应的 block 绑定、token 位置 |
| `read_block_mapping_mismatch` / `write_block_mapping_mismatch` | 调度记录中的缓存块编号与 `kv.read` 中的 block table |
| `batch_request_mismatch` / `result_request_not_scheduled` | 本轮调度请求、执行批次请求和结果请求的对应关系 |
| `execution_range_mismatch` / `read_length_mismatch` | token 预算、`query_start_loc`、`positions` 和 `seq_lens` |
| `padded_write_target` / `dummy_write_target` | padding 或陪跑记录的写入槽位 |

### 5. 分析 slot 检查结果

先确认 `kv.enabled` 中的 `slot=true`，再筛选 slot 异常记录：

```shell
grep -F '"audit":"slot"' server.log
```

按异常中的 `trace_id`、请求标识、`rank` 和 `group_id` 汇总同轮日志，结合下表查看异常位置及处理步骤。

| `reason` | 结果含义 | 查看与处理 |
| --- | --- | --- |
| `write_target_mismatch` | 模型输入指定的写入槽位与请求获准使用的槽位存在差异 | 对照 `expected_slot`、`actual_slot` 和 `token_index`，再查看同轮 `kv.bind` 的缓存块与 `kv.write` 的位置记录 |
| `slot_collision` | 两个请求在同一轮使用同一写入槽位 | 按 `trace_id` 汇总 `kv.bind` 和 `kv.write`，结合 `slot`、`source` 查找涉及的请求及写入位置 |
| `duplicate_slot_write` | 同一请求的多个 token 使用同一写入槽位 | 按 `trace_id` 找到该请求的 `kv.write`，对照 `slot_mapping` 和 `positions` 中重复槽位对应的 token 位置 |
| `slot_out_of_capacity` | 写入槽位超出缓存容量范围 | 有效范围为 0～`capacity-1`，结合 `actual_slot` 和 `token_position` 定位越界位置 |
| `padded_write_target` | 填充区域出现有效写入槽位 | 查看 `slot_mapping_scope` 和 `execution_mode`，按 `graph_padding` 或业务 token 区域汇总该轮日志及启动配置 |
| `slot_observation_missing` / `slot_metadata_missing` / `slot_padding_metadata_missing` / `slot_metadata_invalid` | 校验输入完整性检查触发异常 | 按 `action` 配置记录或抛出异常；确认 `VLLM_USE_V2_MODEL_RUNNER=1`，保存完整启动参数、服务版本和异常日志 |
| `unsupported_slot_layout` | 当前缓存布局触发 slot 适用条件检查 | 查看异常日志中的缓存组信息；查看当前场景的 KV Trace 记录时，将 `slot` 设为 `false` |
| `unsupported_slot_null_redirect` | 预热过程使用保留空块作为写入位置 | 将 `slot` 设为 `false`，通过 KV Trace 查看预热场景的请求与读写位置 |
| `slot_release_binding_stale` | 请求释放后保留缓存绑定 | 按 `request_id` 对照前后 `kv.bind` 与服务日志中的释放记录 |
| `slot_release_inflight` | 请求的 KV 传输仍在进行时触发释放 | 对照 `kv_connector_output` 的发送、接收完成信息；`raise` 在释放前抛出异常，`log` 记录后继续 |

`log` 记录槽位及校验输入异常并继续推理。`raise` 锁存首次校验异常，在结果检查时抛出 `KVGuardError`；排查时结合对应日志确认运行配置。

### 6. 按场景补充查看

以下场景通过 KV Trace 日志分析请求及缓存使用情况。投机解码和空块重定向预热场景使用 `slot=false`。

数据并行（Data Parallel，DP）由多个实例分别处理不同请求批次；陪跑是实例以占位输入参与并行计算的过程。相关概念请参见 vLLM 官方《[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)》。

上下文并行将长序列的计算或缓存分布到多个执行进程：预填充上下文并行（Prefill Context Parallel，PCP）分摊输入处理；解码上下文并行（Decode Context Parallel，DCP）分布存储生成阶段使用的 KV Cache。相关概念请参见 vLLM Ascend 官方《[Context Parallel Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html)》。

| 场景 | 分析要点 |
| --- | --- |
| 抢占、恢复、复制、清零 | 从 `kv.bind` 查看 `preempted_req_ids`、`resumed_req_ids`、`kv_cache_block_copies`、`new_block_ids_to_zero`，结合前后轮绑定分析 |
| DP 陪跑 | 查看 `evidence=dummy_plan` 的 `slot_count`（槽位总数）、`non_pad_slots`（`[索引,值]` 列表，PAD 项省略）。普通陪跑使用 PAD 槽位；`allow_null_block=true` 表示允许使用预留占位块的槽位，范围为 `0 <= slot < block_size` |
| 推测解码（候选 token 生成与验证） | 对照 `kv.bind` 的 `speculative` 预算与 `evidence=sample_postprocess` 的 `num_sampled`（采样数）、`num_rejected`（拒绝数）、`query_start_loc`；`association=current_step` 表示当前轮结果，`observed_call_only` 表示仅记录本次调用，此时 `trace_id=null` |
| PCP / DCP | 结合各执行进程的 PCP/DCP 字段分析。`slot_mapping_scope=pcp_frame` 表示整组 PCP 的写入槽位列表，结合 `evidence=parallel_context` 的全局请求、token 对应关系和写入标记（mask）对照；DCP 结合 `dcp_local_seq_lens` 查看该进程的读取长度 |
| KV 传输 | 从 `evidence=connector_plan` 查看请求及本地、远端缓存块列表；从结果的 `kv_connector_output` 先检查 `failed_recving`（接收失败请求）和 `invalid_block_ids`（异常缓存块编号），再对照 `finished_sending`（发送完成请求）、`finished_recving`（接收完成请求）核验状态 |

在 Mooncake PD 分离场景开启 `slot=true` 后，按 `request_id` 关联 `evidence=connector_plan` 与 slot 异常。P 端在导出时核对块绑定。D 端在读取返回后核对整次传输的源地址、目标地址和覆盖范围，识别错位、遗漏及重复读取。`raise` 锁存检查异常并中止后续处理，`log` 记录后继续。发送、接收完成状态通过 `kv_connector_output` 查看。启动服务前设置 `VLLM_LOGGING_LEVEL=DEBUG`，即可在 Mooncake 传输日志中查看地址明细。

| `reason` | 查看与处理 |
| --- | --- |
| `pd_slot_transfer_mismatch` | 请求的源/目标地址区间或覆盖次数出现差异；对照请求传输计划与原生传输地址记录 |
| `pd_slot_source_plan_mismatch` / `pd_slot_ownership_mismatch` | 对照请求的导出、接收块计划与当前缓存归属 |
| `pd_slot_stale_transfer` | 请求结束、抢占或重新绑定后，旧传输仍然返回；按请求关联前后调度记录 |

抢占换出/换入可使用以下完整 Eager 命令：

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/model --enforce-eager \
  --kv-transfer-config '{"kv_connector":"PreemptOffloadConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":268435456,"enable_offload_prefix_caching":false}}' \
  --additional-config '{"scheduler_config":{"recompute_scheduler_enable":true},"kv_trace":{"enabled":true,"action":"log","slot":true}}' \
  > server.log 2>&1
```

在 `server.log` 中按 `swap_slot` 筛选异常，结合 `direction` 与 `event_id` 查看整批提交。`D2H` 表示换出，`H2D` 表示换入；恢复后的 GPU 槽位通过 `kv.write` 与前后 `kv.bind` 核对。

| `reason` | 查看与处理 |
| --- | --- |
| `swap_slot_plan_mismatch` | 源/目标块列表、容量或同步方式出现差异；结合事件编号与换入/换出方向定位 |
| `swap_slot_observation_missing` | 当前批次的换入/换出计划保留待执行项；对照调度与原生传输日志 |
| `swap_slot_completion_missing` | 原生完成事件与本次提交出现差异；对照事件编号及服务异常堆栈 |

## 结果解读

KV Trace 的校验依据是日志中的调度记录、模型输入的缓存块与槽位，以及推理结果，适合定位这些信息之间的一致性问题。缓存块的物理使用情况可结合设备侧追踪记录分析；KV 数值正确性可通过张量采集和精度比对确认。

`finished_req_ids` 表示请求结束信息，复制、清零和传输字段表示相应的调度安排。排查时将调度记录与执行结果、服务状态结合，形成完整的问题上下文。
