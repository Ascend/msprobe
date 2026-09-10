# Logic Buffer 工具链（monkey-patch 版）— 技术交接文档

> 供维护者 / 下一位 agent 快速恢复上下文。所有结论均在本机（昇腾 NPU）实测过。
> 本包只含 **monkey-patch 采集管线**：不改 torch 源码。硬编码补丁（`patch.py`）与回归/验证测试集不在此包中（仍在开发仓库 `/home/qc/PTA_210/logic_buffer/`）。
> 最后整理：2026-09-04。

---

## 1. 项目是什么

一套给 **torch.compile + torch_npu（昇腾 NPU）** 用的"逻辑 buffer"追踪/对比/可视化工具：

- **采集（monkey-patch）**：进程内 wrap inductor 的 14 个宿主方法，编译时采集 **FX 节点 ↔ logic buffer ↔ kernel 调用** 映射 + 运行时 dump 张量。不复制文件、不落盘改 torch。
- **对比（run_compare）**：进程内重放 FX runnable + `output_code.py`，把 **FX 中间值 vs kernel 实际输出** 逐 buffer 数值对齐，产出 `compare_report.*`。
- **可视化（xla_style_viz2）**：FX 图与 kernel 调用链的双栏交互 HTML（graphviz / node wasm 渲染）。

包内运行文件：

| 文件 | 角色 |
|---|---|
| `logic_monkeypatch.py` | 入口：`install()` 装 14 个 hook；`_load_helper()` 注册 helper；`maybe_auto_viz()` 自动出图 |
| `logic_buffer_trace.py` | helper：记录函数 + dump。以 `torch._inductor.logic_buffer_trace` 名义加载 |
| `run_compare.py` | 重放 + 对比（CLI 入口 `main()`，argparse） |
| `xla_style_viz.py` | 单栏 XLA DOT；`xla_style_viz2` 依赖其 `XLA_COLORS/build_dot/html_escape/opname/parse_fx` |
| `xla_style_viz2.py` | 双栏交互图（主推），`import run_compare as rc` |

---

## 2. 原理

monkey-patch 不改 torch，而是把要采样的宿主方法**整包 wrap**：调用 `orig()` 前读输入、之后读输出，再调用 helper 里的同一记录函数 → 与硬改 torch 源码在**同一时机**采到**同一信息**。helper 用 `importlib` 以 `torch._inductor.logic_buffer_trace` 名义 exec 进来（保留其 `from . import ir` / `from .virtualized import V` 等相对导入），**不复制到 site-packages**。

---

## 3. 14 处 hook 对照表（`logic_monkeypatch.install()`）

| # | 宿主方法 | 时机 | 调用的 helper 记录器 |
|---|---|---|---|
| 1 | `GraphLowering.run_node` | 每个节点 lower 后 | `record_lowering_boundary` |
| 2 | `Scheduler.__init__` | scheduler 建立 / DCE 后 | `record_live_fx_buffer_map_after_dce` |
| 3 | `ir.ExternKernel.copy_input` | extern 输入物化 | `record_copy_input_materialization` |
| 4 | `ir.ExternKernelOut.codegen` | extern kernel 输出 codegen | `collect_extern_kernel_call` + `codegen_kernel_runtime_dump` |
| 5 | `ir.ExternKernelAlloc.codegen` | 同上（alloc 型） | 同上 |
| 6 | `ir.MultiOutput.codegen` | 多输出 extern（tuple） | `collect_single_output_multi_output_call` + dump |
| 7 | `TritonKernel.call_kernel` | triton kernel 调用前 | `collect_triton_kernel_call` |
| 8 | （同上）| 调用后 | `codegen_kernel_runtime_dump` |
| 9 | `PythonWrapperCodegen.write_args` | 写重放输入 | 注入 `save_replay_args` |
| 10–11 | （预留/聚合）| — | — |
| 12 | `DebugContext.__enter__` | debug 目录建好后 | 设 `INDUCTOR_LOGIC_BUFFER_(ACTIVE_)DIR` = `<torch_dir>/logic_buffer_run`；trace 关闭时兜底自建目录 |
| 13 | `DebugFormatter.fx_graph_transformed` | transformed 图落盘 | 补写 `fx_graph_transformed_runnable.py`（`save_graph_repro`） |

> 数字沿用历史编号（含曾并行的版本，中间跳号属正常）。`#`号即 README / 历史记录里的 hook #12 / #13。

**helper 只在 `INDUCTOR_LOGIC_BUFFER_TRACE ∈ {1,true,yes,on}` 时采样**（`_trace_on()`）。`LOGIC_MONKEYPATCH_DEBUG=1` 打印 hook 内异常 traceback（`_safe` 吞掉避免打断编译）。

---

## 4. helper（`logic_buffer_trace.py`）关键函数

- **记录映射**
  - `record_lowering_boundary(graph, node, result)` — lowering 期 FX 节点 ↔ buffer。
  - `record_live_fx_buffer_map_after_dce(scheduler)` — DCE 后存活 buffer ↔ fx 映射。**live 名取 `scheduler.name_to_buf.keys()`（真 buffer 名）**，不是 `node.get_name()`（torch 2.10 里是 `op<N>`）——这是 DCE 后能捕获惰性融合→buffer 映射的关键。
  - `collect_extern_kernel_call(name, self, output_ref)` / `collect_single_output_multi_output_call` / `collect_triton_kernel_call(name, self, call_args, precompile_args)` — kernel ↔ buffer/arg 映射，写 `kernel_arg_map.jsonl`。
  - `record_copy_input_materialization` — extern 输入物化，写 `copy_input_materializations.jsonl`。
- **运行时 dump**
  - `codegen_kernel_runtime_dump(wrapper, name, call_id, entries)` — 向生成的代码里埋 dump（写 `kernel_values.jsonl` + `kernel_tensors/`）。
  - `dump_fx_tensor(name, tensor)` — FX 中间值 dump（写 `fx_values.jsonl` + `fx_tensors/`）。run_compare 重放时在 instrumented 图里注入对它的一次 import。
  - `save_replay_args(args, graph_key, arg_names)` — 按编译图写
    `replay_args_<graph_id>_*.pt` 和索引 `replay_args.jsonl`。
- **trace 落位**：helper 读 `INDUCTOR_LOGIC_BUFFER_(ACTIVE_)DIR` 决定输出目录（hook #12 负责把它指向 model 目录下的 `logic_buffer_run`）。

---

## 5. 环境与 gotchas

- **版本**：torch 2.10.0+cpu / torch_npu 2.10.0.post5.dev20260829 / triton 3.6.0 / Python `/usr/local/python3.11.13/bin/python3`。
- 别在 `site-packages/torch` 目录里跑 python（`torch/__future__.py` 遮蔽标准库）。
- torch.compile 会命中全局缓存 → **每 case 独立 `TORCHINDUCTOR_CACHE_DIR` + 独立子进程**，cwd=工作目录（debug base_dir 在 import torch 时快照 cwd）。
- NPU 瞬态错误 `rtGetDevMsg context is null` / 编译超时 → 重跑即成功。
- 可视化：无系统 dot，用 node 26 + `@hpcc-js/wasm`（`Graphviz` 大写 API）。

---

## 6. 怎么跑（速查）

见 `README.md`「快速开始」。要点：

```python
from msprobe.logic_buffer_trace import install, maybe_auto_viz
install()                            # torch.compile 之前、同进程

# 环境：INDUCTOR_LOGIC_BUFFER_TRACE=1；可选 INDUCTOR_LOGIC_BUFFER_DIR / AUTO_VIZ
import torch, torch_npu
out = torch.compile(model)(x)

# forward 之后，若要自动出图：
maybe_auto_viz()                     # 需要 INDUCTOR_LOGIC_BUFFER_AUTO_VIZ=1
```

- `python3 -m msprobe.logic_buffer_trace.run_compare ...` 会自动注册 helper。viz2 也内置 `_ensure_runtime_dumps`（dump 缺失时经 `run_fx`/`run_output` 在 trace 原生设备补重放，自动 `_load_helper()`）。

---

## 7. 关键修复记录（按组件）

### hook 层

- **#12（2026-09-01）wrap `DebugContext.__enter__` 而非 `__init__`**：`__init__` 时 torch 的 `self._path` 还是 None（目录在 `__enter__` 才建），旧 hook 只能自建目录，且与 torch 的 `DebugContext._counter` 各自从 0 数 → 撞车分到 `.0`/`.1`（trace 与 codegen 分开）。改为 `__enter__` 后：torch 建好目录（或 trace 关闭时 hook 兜底建）再设 `INDUCTOR_LOGIC_BUFFER_DIR = self.filename("logic_buffer_run")` → **trace 与 output_code.py / fx_graph_transformed* 同目录**。
- **#13（2026-09-01）真实宿主是 `DebugFormatter`，不是 `DebugContext`**：torch 2.10 里 fx_* 调试方法在 `DebugFormatter`（`DebugContext.__getattr__` 在 trace enabled 时转发给它）。旧 wrap `DebugContext.fx_graph_transformed` → `hasattr` False → `fx_graph_transformed_runnable.py` 永不产出 → 重放回退 `fx_graph_runnable.py`（节点名对不上 transformed 图）→ 大量 `missing_fx_value`。修复：wrap `DebugFormatter.fx_graph_transformed`，`save_dir` 取 `INDUCTOR_LOGIC_BUFFER_DIR`（makedirs 兜底）。
- **`MultiOutput` 多输出 extern**（max_pool2d 等返回 tuple）原吞 call_id → 经 `collect_single_output_multi_output_call` 正常记录。
- **helper 内 alias pass**：记录 `reinterpret_tensor` 别名的 fx 映射——但**未能覆盖 memory-planning 之后**才生成的别名（时机不对），实际不生效、保留无害（unet/deltanet 的切片别名仍可能 no_fx，见 §8）。

### run_compare（共享工具层）

1. `_entry_output_logicals`：in-place（input_output）kernel 不再把输入 buffer 当输出。
2. `_parse_alias_map` / `_resolve_alias`：解析 output_code 的 `reinterpret_tensor` / getitem 别名（stack/cat 切片 → 基 buffer）。
3. `build_kernel_call_graph`：producers 覆盖改**并集**（多 kernel 写同一基 buffer 边全连上）。
4. `parse_output_kernel_calls`：支持 `torch.ops.npu.*`（如 `npu_fusion_attention`）输入不丢。
5. `_symint_arg_names` + `parse_fx_graph`：过滤动态模型 symint 形状占位符。
6. `_kernel_true_output_logicals`：kernel 输出若解析到自己输入（就地融合）则不算产出；但输出 buffer **有自身 `buffer_to_fx` 映射时保留原名**（存储别名≠值别名），避免就地融合/多输出 kernel 的 `kernel_fx_nodes` 变空。
7. `--shape-override "sN=VAL"`：动态 shape 换 shape 重放。
8. `_is_pure_view_fx` 过滤：buffer 有计算生产者时丢弃纯 reshape/view 重复映射（避免同 buffer 两行）。

### xla_style_viz / viz2

- `parse_fx` 过滤 symint；`build_dot` 支持 fx_stats（坏节点标红）。
- 双栏图 + `--cpu`（旧名 `--fx-device cpu` 兼容）= CPU 重放 FX（`replay_fx_on_cpu` 含 `device(type='npu')→device(type='cpu')` 替换，gpt2 等能跑 CPU 的关键）。
- `_ensure_runtime_dumps`：比较前若 fx/kernel dump 缺失，先在 trace 原生设备补重放 dump（避免静默退化全 `missing_fx_value`）。
- **cat/stack 并行生产者 → operand 级映射**：`_parse_alias_slices`（从 output_code 的 `reinterpret_tensor` offset/size 解析切片）+ `_build_cat_operand_map`（从 transformed FX 图 cat 的 operand 列表解析归属）；切片 i ↔ 第 i 个 operand，标注 `cat[0:64]`…。**索引 bug（2026-09-01）**：原用 `off // size` 算切片号，3D cat 下 size=总元素而 offset 按 cat 维步进 → 全落 operand 0；改按 **offset 排序秩** 分配。
- **映射策略（2026-09-01）**：kernel→FX 映射（含 hover/对比）**全静态**（lowering map + alias/slice/operand 解析）；**动态数值相等不作映射依据**。

---

## 8. 已知限制 / 待办

1. **`npu_fusion_attention` 无 CPU 版**：NPU 硬件算子，CPU 重放需等价实现（原生 attention），当前不做。
2. **cat/stack 并行生产者 no_fx**：多个 kernel 各写 `cat` 输出一段（语义合理，已映射到 operand）；但 unet 3 / deltanet 8 / csa_moe 10 这类 buffer 在对比报告仍 `no_fx_for_logical_buffer`（静态映射不含 codegen 切片别名）。
3. **helper alias pass** 不覆盖 memory-planning 后的别名 → 需 codegen/memory-planning 阶段新 hook（未做）。
4. **动态 shape**：FX 已滤 symint 占位符；报告对比用 `--shape-override`。
5. **版本漂移风险**：monkey-patch 按类/方法名 patch，torch 大版本变更新方法签名/宿主即失效（失效会报错/不触发，不会静默写坏）。
6. **helper 注册依赖**：run_compare 重放需 `torch._inductor.logic_buffer_trace` 可 import（见 §6 / README FAQ Q1）。

---

## 9. 验证状态（历史记录，2026-09-01）

- 8 模型（vgg16/unet/bert/gpt2/csa_moe/deltanet/mamba/kda）× 静态/动态 = 16 case：monkey-patch 采集与代码生成一致性全过，compare 全 ok，`no_fx_for_logical_buffer` 计数与当时另一变体完全一致。
- 另 8 个额外模型（t5/vit/mnv3/dlrm/hca/engram/latent_moe/dspark）NPU 采集：`missing_fx_value` 全 0，max_abs ≤ 2e-4；发现 ViT/HCA attention 融合为 `npu_fusion_attention`、dlrm 交互 bmm 折叠为 `aten.mm` 等。
- `compare_report.json` 是 **list**（每行一个 buffer 对比），不是 dict。
- 本包移除测试集后，复现验证需自行在开发仓库 `/home/qc/PTA_210/logic_buffer/` 的 `tests/` 里跑（或按 README 最小示例自建）。
