# Logic Buffer 工具链（monkey-patch 版）

一套用于 **`torch.compile` + torch_npu（昇腾 NPU）** 的"逻辑 buffer"追踪与数值对比工具。

> **本包只用 monkey-patch 采集（不改 torch 源码）**。不包含硬编码补丁（`patch.py`）与回归/验证测试集——它们属于开发仓库 `/home/qc/PTA_210/logic_buffer/`（monkey-patch 模式可单独运行，无 torch 副作用）。

- **采集**：编译时追踪 **FX 节点 ↔ 逻辑 buffer ↔ kernel 调用** 的三层映射，并 dump 运行时张量。
- **对比**：重放 FX 计算图 + 生成的 `output_code.py`，把 **FX 中间值 vs kernel 实际输出** 逐 buffer 数值对齐，产出 `compare_report`。
- **可视化**：把 FX 图与 kernel 调用链渲染成交互式 HTML（双栏 XLA 风格）。

```text
torch.compile(模型)
   │  inductor 编译期（进程内 wrap，不改源码）
   ▼
采集层 ──► FX↔buffer↔kernel 映射 + 运行时 dump 张量
   │
   ▼
run_compare ──► 重放 FX + output_code，逐 buffer 数值对比 ──► compare_report.*
   │
   ▼
xla_style_viz2 ──► 交互式 HTML 图
```

---

## 目录结构

```text
python/msprobe/logic_buffer_trace/
├── README.md                 本文件（文档 + 使用说明）
├── requirements.txt          环境版本说明
├── logic_monkeypatch.py      monkey-patch 采集入口（install()，进程内 wrap 14 个宿主方法）
├── logic_buffer_trace.py     采集 helper（由 monkey-patch 以 torch._inductor.logic_buffer_trace 名义加载）
├── run_compare.py            重放 + 数值对比（CLI + 库）
├── xla_style_viz.py          单栏 FX 图（XLA 风格 DOT；xla_style_viz2 的底层函数依赖）
├── xla_style_viz2.py         双栏 FX+kernel 图（XLA 风格 + 交互，主推）
└── PROJECT_HANDOFF.md        技术交接文档（hook 对照表 / 修复记录 / 已知限制）
```

> 依赖关系：`xla_style_viz2.py` 会导入同包的 `xla_style_viz.py`（单栏文件必须保留）。

---

## 环境要求

实测环境（本工具链在此栈上验证）：

| 组件 | 版本 |
|---|---|
| Python | `/usr/local/python3.11.13/bin/python3` |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post5.dev20260829 |
| triton | 3.6.0 |
| NPU | `torch.npu.is_available()` = True（1 device） |
| 可视化（可选） | node 26 + `@hpcc-js/wasm`（无系统 dot 时用 graphviz 渲染） |

```bash
python3 -c "import torch, torch_npu; print(torch.__version__, torch.npu.is_available())"
```

**⚠️ 重要 gotchas：**

1. **别在 `site-packages/torch` 目录内运行 python**——torch 自带 `torch/__future__.py` 会遮蔽标准库 `__future__`，site 初始化崩溃。
2. **torch.compile 缓存**：命中全局缓存会导致不重新生成 debug 目录、不触发采集。每个 case 用**独立 `TORCHINDUCTOR_CACHE_DIR`**（放空目录跑）。
3. **debug 目录**：`torch._dynamo.config.debug_dir_root` 在 import torch 时快照 cwd → 每个 case 建议**独立子进程**（cwd=工作目录），否则多个 case 的 model 目录会互相干扰。
4. **NPU 瞬态错误**：编译期偶发 `rtGetDevMsg ... context is a null pointer` / 超时，属昇腾环境问题，重跑即成功。

---

## 快速开始

### 最小端到端（mm + relu + add，不依赖任何模型库）

写一个独立脚本（本包不再附带冒烟测试文件，把这段存成 `smoke.py` 即可）：

```python
import os
from pathlib import Path
from collections import Counter

WORK = Path("/tmp/lb_smoke"); WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)                      # torch 在此快照 debug base_dir
os.environ.update({
    "INDUCTOR_LOGIC_BUFFER_TRACE": "1",        # 开采集
    "INDUCTOR_LOGIC_BUFFER_DIR": str(WORK / "logic_buffer_run"),  # trace 落位
    "TORCHINDUCTOR_CACHE_DIR": str(WORK / "cache"),               # 独立缓存
})
from msprobe.logic_buffer_trace import install
install()                           # 必须在 torch.compile 之前、同进程调用

import torch, torch_npu

def model(x):
    a = x @ x.transpose(1, 2)
    a = torch.relu(a)
    return a + x

x = torch.randn(2, 4, 4, device="npu")
with torch.no_grad():
    torch.compile(model)(x)

# 定位 codegen 目录（目录名称因模型而异）
md = next(path.parent for path in WORK.rglob("output_code.py"))
print("model_dir:", md)
```

跑完后在 `logic_buffer_run/` 有 trace，codegen 目录下有 `output_code.py`、
`fx_graph_transformed_runnable.py`。随后运行下文的 `run_compare` 模块命令即可；
它会自动注册 trace helper。

期望：2 个 buffer（`bmm`、`add`）全部 `ok`（max_abs = 0.0）。

### 自动出 HTML

采集脚本 `forward` 之后调 `maybe_auto_viz()`，并设 `INDUCTOR_LOGIC_BUFFER_AUTO_VIZ=1`，则自动生成 `<model_dir>/logic_buffer_run/graph.html`。默认关闭，避免拖慢只采集的用法。

---

## 采集（monkey-patch 模式）

```python
from msprobe.logic_buffer_trace import install
install()                            # 必须在 torch.compile 之前、同一编译进程内调用

import torch, torch_npu
# ... 你的 torch.compile 代码 ...
```

环境变量：

```bash
export INDUCTOR_LOGIC_BUFFER_TRACE=1               # 开采集（install() 会 wrap，但只有置 1 才采样）
export INDUCTOR_LOGIC_BUFFER_DIR=/abs/path/to/logic_buffer_run   # trace 落位（建议显式设）
export LOGIC_MONKEYPATCH_DEBUG=1                   # 可选：打印 hook 异常 traceback
export INDUCTOR_LOGIC_BUFFER_AUTO_VIZ=1            # 可选：采集完自动出 HTML
```

- **trace 落位**：不设 `INDUCTOR_LOGIC_BUFFER_DIR` 时，hook #12 自动把 trace
  指到 codegen 目录的 `logic_buffer_run` 子目录，与 `output_code.py` 和
  `fx_graph_transformed*` 位于同一 codegen 目录。显式设置后则使用指定路径。
- 每个 case 用独立 `TORCHINDUCTOR_CACHE_DIR` + 独立进程，避免缓存命中不采集。

> **为什么只 wrap 不硬改？** 每个采集点在 torch 源码里都是"函数中间的只写采样"。monkey-patch 取该宿主方法整包 wrap，把采样移到 `orig()` 调用前（读输入）或后（读输出），信息等价。详见 [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。

---

## 对比（run_compare）

```bash
python3 -m msprobe.logic_buffer_trace.run_compare \
    --fx <fx_runnable.py> \
    --output <output_code.py> \
    --trace-dir <logic_buffer_run_dir> \
    [--quiet] [--skip-run] [--shape-override "sN=VAL" ...]
```

- 产出：`compare_report.{json,csv,txt}` + `logic_buffer_graph.html`（交互图）。
- `--skip-run`：只对比已有 dump，不重放。
- `--shape-override`：动态 shape 换 shape 重放（更新 shape 标量 arg + 输入张量维度）。
- `run_compare` 和 `xla_style_viz2` 都会自动注册 `torch._inductor.logic_buffer_trace` helper。

---

## 可视化

```bash
# 双栏 FX+kernel 图（主推）
python3 -m msprobe.logic_buffer_trace.xla_style_viz2 <model_dir> --out out.html          # NPU 重放
python3 -m msprobe.logic_buffer_trace.xla_style_viz2 <model_dir> --out out.html --cpu    # CPU 重放 FX（NPU kernel vs CPU FX）

# 单栏 FX 图
python3 -m msprobe.logic_buffer_trace.xla_style_viz <model_dir> --out out.html
```

---

## 产物说明

一次编译产生的 trace 目录（`logic_buffer_run/`）：

```text
kernel_arg_map.jsonl            kernel 调用 ↔ buffer/arg 映射
kernel_values.jsonl             kernel 输出张量路径 + 元数据
kernel_tensors/                 按 kernel 输出的张量 .pt
fx_values.jsonl                 FX 节点 dump 的中间值
fx_tensors/                     按 FX 节点 dump 的张量 .pt
lowering_fx_buffer_map.jsonl    lowering 时 FX 节点 ↔ buffer 映射
lowering_live_fx_buffer_map.jsonl  DCE 后存活 buffer
copy_input_materializations.jsonl  extern copy_input 物化记录
replay_args_<graph_id>_*.pt     按编译图区分的重放输入
compare_report.{json,csv,txt}   run_compare 输出（逐 buffer 对比结果）
logic_buffer_graph.html         run_compare 生成的交互图
```

model 目录（codegen）另有：`output_code.py`（生成的 kernel 调用代码）、`fx_graph_transformed_runnable.py`（可重放的 transformed 图）、`ir_pre_fusion.txt` 等。

---

## 已知问题 / FAQ

**Q1. run_compare 报 `cannot import name 'logic_buffer_trace'`？**
请通过 `python3 -m msprobe.logic_buffer_trace.run_compare` 启动。该入口与可视化入口
都会自动注册 `torch._inductor.logic_buffer_trace` helper。

**Q2. monkey-patch 的 trace 和 codegen 会分到两个目录吗？**
不会了（2026-09-01 修复）。不显式设置 `INDUCTOR_LOGIC_BUFFER_DIR` 时，hook #12
会让 trace 与 `output_code.py`、`fx_graph_transformed*` 使用同一个 codegen 目录。

**Q3. 编译偶发 `rtGetDevMsg context is null` / 超时？**
NPU 瞬态错误，非工具问题，重跑即可。

**Q4. 采集时没生成 debug 目录 / 没 trace？**
`TORCHINDUCTOR_CACHE_DIR` 命中缓存或 cwd 不对。用独立缓存目录 + 独立进程（cwd=工作目录）。

**Q5. 动态模型有 `missing_fx_value`？**
必须用 `fx_graph_transformed_runnable.py` 重放（run_compare `--fx` 传它；若 `resolve_fx_replay_path` 找不到才回退 `fx_graph_runnable.py`，此时 transformed 图节点名对不上会出现 `missing_fx_value`——hook #13 已修，见 handoff）。

**Q6. bert 等模型没有 CPU 版？**
`npu_fusion_attention` 是 NPU 硬件算子，CPU 重放无实现（需原生 attention 等价实现），当前不做；其余模型 `xla_style_viz2 --cpu` 可跑。

**Q7. `no_fx_for_logical_buffer` 是什么？**
某些 buffer（cat/stack 并行生产者、memory-planning 后别名）静态映射不到 FX 节点。这是已知语义限制（unet 3 / deltanet 8 / csa_moe 10）。

**Q8. 映射是静态还是看数值？**
纯静态：lowering map + alias/slice/operand 解析；**动态数值相等不作为任何映射依据**（见 handoff §映射策略）。cat/stack 并行生产者映射到各自的 operand 节点 + 切片标注（如 `squeeze`+`cat[0:64]`、…、`squeeze_7`+`cat[448:512]`）。

---

## 技术细节

内部实现、14 处插桩对照表、helper 函数清单、修复记录与已知限制见 **[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)**。
