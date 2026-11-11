# vLLM KV 出入口追踪

KV Trace 通过 vLLM 的 general plugin entry point 加载，记录 Scheduler、Worker 和 Ascend Model Runner V2 之间的 KV 边界信息，并检查调度声明与执行地址计划的一致性。配置字段名为 `kv_trace`；不重建 allocator 状态，不检查 KV 数值。

## 启用

在启动 vLLM 的 Python 环境中正常安装 msprobe，使 entry point 可被发现；单独设置 `PYTHONPATH` 不会注册入口。

```shell
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/model \
  --additional-config '{"kv_trace":{"enabled":true,"action":"log"}}'
```

Eager 模式可增加 `--enforce-eager`；图模式沿用原生 compilation 配置。不需要 `dump_config`、`PrecisionDebugger` 或 `AclGraphDumper`。未配置 `kv_trace` 或设置 `enabled:false` 时关闭；`enabled` 必须是布尔值，`action` 只接受 `log`（默认）或 `raise`，不接受其他字段。如果显式配置了原生 `VLLM_PLUGINS` 白名单，需要包含 `msprobe_kv_guard`。

`log` 记录 `kv.violation` 并继续原生流程；`raise` 在调度出口、Worker 结果边界或 `_dummy_run` 结果出口抛出 `KVGuardError`，沿原生执行链使 EngineCore 失败退出。原始记录先保留，再执行阻断；检查不保证发生在本轮 kernel 执行前，也不回滚设备写入。

本功能独立于原有 tensor/statistics/NaN dump；这些能力仍按各自原配置使用。

## 阶段一：核心出入口

```text
Scheduler.schedule → SchedulerOutput + trace_id → Worker.execute_model
  → model_state.prepare_attn 输入计划 → Worker 观察到结果 → Scheduler 收到结果
```

| 记录 | 边界与含义 |
| --- | --- |
| `kv.bind` | 原生新请求 block 表、cached `new_block_ids` 增量和本轮调度字段，不是完整 allocator 状态 |
| `kv.write` / `kv.read` | `model_state.prepare_attn` 输入的每组 slot、block table、长度、请求和位置描述 |
| `boundary` | Worker 输入、观察到的输出、Scheduler 收到结果及后续公共边界；由 `boundary`、`evidence`、`phase` 区分来源 |
| `kv.enabled` | 启用进程、rank 与 action |
| `kv.error` | 原生边界异常，记录后原样抛出 |
| `kv.violation` | 边界计划不一致；按 `action` 记录或阻断结果 |

KV 操作只使用 `kv.bind/write/read`；启用、原生异常和违规是诊断，不是新的 KV 生命周期状态。每轮 `trace_id` 沿原对象传递，只用于关联记录，不是设备完成序号。`kv.bind` 合并 `num_scheduled_tokens`、`total_num_scheduled_tokens`；不另发调度预算事件。读写记录标记 `evidence=model_runner_plan`，附原生 `execution_mode`、`block_size` 和 `kernel_block_size`。整行 block table 是输入地址表，不表示每个元素都被 kernel 读取。

Worker 结果边界使用 `phase=observed`：正常调用链已经观察到原生输出，但不代表工具放行。即使随后 `guard.check()` 因 `action=raise` 抛错，这条原始记录仍可能存在。工具不提前或额外调用异步 `get_output()`。Scheduler 收到结果也不等于处理成功或设备完成，不另发重复的 `output_processed` 事件。

设备地址描述使用异步 D2H 快照；复用缓冲前完成上一份快照，在原生输出获取或下一次计划边界打印并校验。Scheduler 只维护一份请求到 block 表的映射，将本轮不可变计划随原生调度对象传给 Worker：new/resumed 替换，cached 增量追加，finished/preempt 删除。它用于将设备计划与调度来源比较，而非只检查设备表自身是否自洽。

校验请求对应关系、token 数量和声明长度、block 路由与写 slot。本轮写入的调度 block 必须位于原生 `kv_cache_config.num_blocks` 范围内；普通请求不能写 null block 0。保留最小异步快照和首个失败原因，不保存物理分配 epoch、独立 owners 索引或有效前缀状态，也不安装逐层 Attention hook。

滑窗退役前缀允许原生 block table 零洞，只严格检查本轮写入所用的表列；其他非零路由与调度绑定比较。因此不承诺识别所有零洞错误或真实未写读。推测/自适应压缩后的有效 query 可以小于 `num_tokens` 缓冲预算，未执行尾部必须为 PAD slot，不将旧位置缓冲解释成实际写入。

## 阶段二：本地特殊路径的边界

相同调度对象的 `finished_req_ids`、`preempted_req_ids`、cached `resumed_req_ids`、`new_block_ids_to_zero`、`kv_cache_block_copies` 及推测预算字段直接并入 `kv.bind`，标记 `evidence=scheduler_plan`，不增加 KV 操作状态。finished/preempt 是请求指令，不等于物理 block 已释放。copy 保留原生源、目标等字段，zero/copy 计划不证明设备操作完成。推测字段保留 `draft_counts`、`num_spec_tokens_to_schedule`、`skip_drafts` 和 `num_invalid_spec_tokens`，不导出候选 token 值。

`kv.bind` 中的 computed 前缀和 `replay_start` 按原值记录，不据此推断 prefix hit 或发生重算；普通 `replay_start=0` 不标为重算。cached 增量追加，resumed 替换旧绑定；不将 CoW 的源 block 全局替换为目标 block。

DP 陪跑等原生 dummy 调用在 `model_state.prepare_attn` 输入记录 `kv.write`，标记 `evidence=dummy_plan`、`phase=dummy`。同时快照实际传入的 slot frame 和完整原生持久 slot 缓冲，包括尾部 padding；以 `slot_count` 和带原始索引和值的 `non_pad_slots` 表示内容，未列出的项均为 PAD `-1`。普通 dummy 的写 slot 必须全为 `-1`，否则记录 `dummy_write_target`。每次边界动态读取原生 `redirect_writes_to_null_block`；仅该标志显式启用时允许写入组内 null block 的 slot 范围 `0 <= slot < block_size`。FULL 服务期 dummy 即使传入 `for_capture=True` 仍检查；CPP profiling 单独跳过。

`runner._dummy_run` 使用独立 `trace_id`，在返回边界完成待处理快照并按 `action` 检查，不借用上一真实请求的调度计划。初始化图捕获未经过该函数时可记录 `trace_id=null`；没有真实调度 trace 的原生 warmup 不与旧计划比较。这些检查针对 slot 写计划，不验证 Mamba state table、backend 内部派生 metadata、实际写入内容或性能。

原生 `runner.postprocess_sampled` 返回使用普通 `boundary`，记录 `idx_mapping`、`num_sampled`、`num_rejected` 和 query 边界，标记 `evidence=sample_postprocess`。不导出 token 值，不反算接受提交状态或有效前缀；普通非推测生成不增加这些计数复制。

对原生有效行检查计数形状、非负和采样容量；有 query 边界时，仅检查实际请求对应的边界前缀及拒绝数，允许 FULL 图附加 padding 边界。原始日志保留完整 query 边界数组。只有最后 PP 的当步参数与调度预算比较，不要求“采样数加拒绝数等于预算”。原生负索引的屏蔽行按原生语义跳过。

非最后 PP rank 的后处理可能属于旧步骤，此时保留 `trace_id=null`、`observed_trace_id` 和 `association=observed_call_only`，不推测原步骤关联。

## 阶段三：CP 与 connector 出入口

`kv.write/read` 附原生 PCP/DCP rank、size 和 interleave。DCP 读取描述保留 `dcp_local_seq_lens`。PCP 的写 slot 保留完整原生 frame，并标记 `slot_mapping_scope=pcp_frame`，不能把整个 frame 与仅本地的请求列表逐项配对。

原生 `pcp_manager.build_attention_context` 返回使用普通 `boundary`，记录全局请求/位置、全局表与 slot、本地到全局请求/token 索引及原生写 mask。它与本轮输入计划共用 `trace_id`，不自行构造 context、地址投影或 collective。单卡不增加这份快照。

校验全局计划与 Scheduler 绑定、局部 token/请求映射和完整写 frame。保留原生 `padded_gather_idx`，因为只有本 rank 的 token 索引不足以描述完整 frame；按该索引和 write mask 对照全局 slot，不重新推演 PCP 分段。允许重复请求行、空 rank 以及 FULL 图的原生占位 query。DCP 本地长度按输入边界的原生 CP 坐标检查。

Scheduler 输出和 Worker 输入的同一原生 connector metadata 使用普通 `boundary`，标记 `evidence=connector_plan`。对公开 `requests` / `requests_to_send` 接口，只选择本地/远端 block、computed/external token 数、远端 engine/request ID 和发送请求 ID。Mooncake 可直接提供这些字段；未知类型标记 `coverage=not_expanded`，部分投影标记 `selected_public_fields`，无法展开的请求保留 `unexpanded_req_ids`。

原生结果中的 `kv_connector_output` 保留 `finished_sending`、`finished_recving`、`failed_recving` 和 `invalid_block_ids`。失败接收也可能包含在 finished 列表中，不能仅看 finished 就判断成功；没有结果保持原生 null，而非伪造完成。工具不额外调用任何会清空队列的 `get_finished/get_transfer_results/get_block_ids_with_load_errors`。

上述记录是计划和原生报告结果，不是传输线程实际提交、远端内容有效或所有 stream 完成的证明。没有模型 token、KV 数值、内部线程对象或自建传输协议；无法转为日志字段的对象不输出，不因格式适配失败阻断原生推理。

## 日志与范围

按 `trace_id` 关联一轮调度，再按请求、rank 和 group 核对 `kv.bind → kv.write/read` 及普通边界。不要仅按 stdout 行序推定设备执行顺序。没有真实 `alloc/release` 事件，finished 不等于物理 free。

未知坐标或缺少原生映射时，不运行依赖这些信息的检查，不报告覆盖缺口，也不视为失败。只有 `kv.violation` 表示检查错误并触发配置的 `log/raise`；原生异常仍记录 `kv.error` 并照常抛出。

日志由 msprobe logger 输出至进程 stdout，启动脚本或容器日志系统可收集为 `server.log`；不另建 KV Trace 文件。每行 `[KV_GUARD]` 后为事件名和 JSON 字段，正常记录为 INFO，原生异常为 ERROR。

当前边界契约参考 vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58` 和 vllm-ascend `28e9e97ba254b076a88275e8a275556cb9a11664`。entry point 内兼容 Ascend 对额外工具配置的校验和 spawn 启动顺序，包装代码全部位于 msprobe；不修改原 `additional_config`。

校验针对可见的声明和地址计划；不能证明实际设备访存、KV 内容正确、物理释放完成，或同一 block ID 复用后的旧引用。prefix 命中、copy、zero 与远端加载的计划不证明内容有效。没有原生数据的内部操作不伪造事件，不提供物理分配 epoch/owners/valid_end 状态机的保证。
