# UWC（训推权重一致性验证）使用指南

> **适用模型与环境（本文档整体以此为准）**：`Qwen2.5-0.5B-Instruct` 模型，**Ascend NPU** 环境，
> **GRPO** 训练算法，**veRL + Megatron-core（mcore）训练 + vLLM 推理** 的训推一体场景。
> 下述代码段与命令均基于该组合编写，其他模型/框架需按注释调整（如命名归一化规则、`torch.npu` 同步）。

## 一、模块概述

### 1.1 模块做了什么

UWC（Unified Weight Consistency）用于验证 RL 训推一体场景下 **训练侧（veRL + Megatron-core）→ 推理侧（vLLM）权重同步链路** 的数值一致性。整个链路被拆成三个互相独立、可单独开关的阶段：

```
训练侧更新权重
   ↓ ① Bridge 转换（HF ↔ mcore 格式）      —— 阶段1：P0 vs P2 + G0/G1 前向
   ↓ ② 传输（共享内存 / IPC）              —— 阶段2：P2_sender vs P5
   ↓ ③ vLLM 加载（HF → 内部分片格式）      —— 阶段3：dummy vs safetensors 差分 + data_ptr 指针偏移
推理侧实际用的权重
```

三个阶段互相独立，可单独运行，通过环境变量控制采集开关。采集零侵入、默认关闭、可随时开启，不影响生产训练。

### 1.2 解决了什么问题

- **权重不一致来源难以定位**：训练侧权重经过 bridge 转换、跨进程传输、vLLM 加载三道工序才到达推理侧，任一环节出错都会导致 RL 推理乱码，且极难排查。UWC 把链路拆成三段，每段产出独立的比对报告，能**快速定位不一致发生在哪一段**（bridge 转换 / 传输通道 / vLLM 加载）。
- **"dummy 权重"难以区分**：通过 sender 侧 `vs_P0_max_diff` 实时抽样诊断 + receiver 侧数值比对，能区分"传输了错误权重"与"漏同步了某些层"。
- **漏同步定位到具体层**：推理侧 data_ptr 指针偏移验证，能精确到"哪一层的参数没有被 `load_weights` 覆盖"。
- **黑盒行为兜底**：dummy vs safetensors 差分测试不依赖内部实现，直接对比两次 RL 首步推理输出，作为行为级校验。

## 二、采集位点与链路

| 位点 | 含义 | rank 维度 | 插入位置 |
|------|------|----------|---------|
| P0 | 原始 HF 权重（磁盘 safetensors） | 每 rank 完整 | `transformer_impl.py` init，`load_weights` 前 |
| P2 | bridge.export_weights 后的 HF 权重（init 时刻） | 每 rank 完整 | `transformer_impl.py` init，`load_weights` 后 |
| P2_sender | 同步时刻 sender 实际发送的 HF 权重 | 每 rank 完整 | `transformer_impl.py` `get_per_tensor_param` |
| P5 | Receiver 收到的 HF 权重 | 每 rank 完整 | `vllm_rollout/utils.py` `_update_weights` 入口 |
| G0/G1 | transformers/mcore 前向 logits | - | 离线 `forward_verify.py` |
| ptr_before/after | 模型参数 data_ptr（推理侧指针偏移验证） | - | `_update_weights` load_weights 前后 |

> **范围说明**：推理侧只做**指针偏移（data_ptr）验证**和 **dummy vs safetensors 差分测试**，**不做**分片反向还原（P6/P7）的验证与实现。

**关键设计**：P0/P2/P2_sender/P5 每个 rank 内容相同，**比对时只需 rank 0 的数据**，采集也只落盘 rank 0，避免多 rank 存储浪费。

---

## 三、代码修改清单

### 3.1 采集侧（运行时代码，具体修改代码段）

采集侧一共改了 5 个文件，均为**环境变量开关 + try-except 包裹**的 best-effort 采集，不改变原有权重加载 / 训练 / 同步逻辑。

#### 3.1.1 `verl/workers/engine/megatron/transformer_impl.py`

**① init 阶段采集 P0/P2（`UWC_ENABLE=1` + `UWC_STAGE1=1` 时）**

在模型初始化、`load_weights` 前后插入采集。关键点：`load_weights` 必须在守卫**之外**无条件执行（每个引擎都要加载），采集只做一次、任何异常都不影响加载。

```python
# transformer_impl.py —— init / 权重加载阶段
if self.vanilla_bridge:
    if os.environ.get("UWC_ENABLE", "0") == "1" and os.environ.get("UWC_STAGE1", "1") == "1":
        global _init_weights_dumped
        if not _init_weights_dumped:
            _init_weights_dumped = True
            # P0 采集（best-effort，失败不影响加载）
            try:
                from verl.utils.uwc.collectors.stage1_bridge_collector import (
                    collect_p0, collect_p2,
                )
                dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
                collect_p0(self.bridge, self.model_config.local_path, dump_dir, step=0)
            except Exception as e:
                print(f"[UWC stage1] P0 collect failed, continue training: {e}", flush=True)
            # bridge 转换（必须执行，不能放进采集守卫）
            self.bridge.load_weights(module, self.model_config.local_path)
            # P2 采集（best-effort）
            try:
                collect_p2(self.bridge, module, dump_dir, step=0)
            except Exception as e:
                print(f"[UWC stage1] P2 collect failed, continue training: {e}", flush=True)
        else:
            # 已 dump 过，只加载权重，不再重复采集
            self.bridge.load_weights(module, self.model_config.local_path)
    else:
        self.bridge.load_weights(module, self.model_config.local_path)
```

**② `get_per_tensor_param`：export 物化 + 实时诊断 + 落盘 P2_sender**

同步时刻 sender 侧核心改动：把 `export_weights` 返回的 generator **立即物化并 clone**（避免延迟访问读到被 offload/复用的数据）；对前 3 个 tensor 打印统计，并与磁盘 P0 实时对比 `vs_P0_max_diff`（判定 export 是否为 dummy）；`UWC_STAGE2=1` 时把实际发送的权重落盘为 `P2_sender`。

```python
# transformer_impl.py —— get_per_tensor_param（同步时刻，sender 侧）
uwc_enable = os.environ.get("UWC_ENABLE", "0") == "1"
if uwc_enable:
    materialized = []
    p0_map = None
    if os.environ.get("UWC_STAGE2", "1") == "1":
        from verl.utils.uwc.collectors.stage1_bridge_collector import load_p0_weights
        try:
            p0_map = load_p0_weights(self.model_config.local_path)
        except Exception as e:
            print(f"[UWC export diag] P0 load failed: {e}", flush=True)

    for i, (name, t) in enumerate(per_tensor_param):
        if i < 3:
            tf = t.float()
            line = (f"[UWC export diag] {name}: device={t.device}, dtype={t.dtype}, "
                    f"shape={t.shape}, mean={tf.mean().item():.6f}, std={tf.std().item():.6f}")
            if p0_map is not None:
                key = name.replace("model.model.", "model.")
                if key in p0_map:
                    max_diff = (tf.cpu() - p0_map[key].float()).abs().max().item()
                    line += f", vs_P0_max_diff={max_diff:.6f}"
                else:
                    line += ", vs_P0=key_not_found"
            print(line, flush=True)
        materialized.append((name, t.clone()))
    per_tensor_param = materialized

    # UWC 阶段2：sender 侧落盘（实际发送的权重，与接收端 P5 一一比对）
    if os.environ.get("UWC_STAGE2", "1") == "1":
        try:
            from verl.utils.uwc.collectors.stage2_transfer_collector import collect_p2_sender
            uwc_dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
            collect_p2_sender(per_tensor_param, uwc_dump_dir, step=0)
        except Exception as e:
            print(f"[UWC stage2] P2_sender collect failed: {e}", flush=True)
else:
    # 非诊断模式也物化+clone，避免 generator 延迟访问问题
    per_tensor_param = [(name, t.clone()) for name, t in per_tensor_param]
```

#### 3.1.2 `verl/workers/rollout/vllm_rollout/utils.py`

**`_update_weights` 入口：采集 P5 + data_ptr before/after**

receiver 侧核心改动：入口处落盘 P5（`UWC_STAGE2`）；`UWC_PTR_CHECK=1` 时在 `load_weights` 前后采集 data_ptr（阶段3 推理侧指针偏移验证）。

```python
# vllm_rollout/utils.py —— _update_weights 入口（receiver 侧）
def _update_weights(self, weights, peft_config, base_sync_done):
    uwc_enable = os.environ.get("UWC_ENABLE", "0") == "1"
    uwc_stage2 = uwc_enable and os.environ.get("UWC_STAGE2", "1") == "1"
    uwc_stage3 = uwc_enable and os.environ.get("UWC_STAGE3", "1") == "1"

    # UWC 阶段2：采集 P5（Receiver 接收到的 HF 权重）
    if uwc_stage2:
        from verl.utils.uwc.collectors.stage2_transfer_collector import collect_p5
        uwc_dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
        try:
            collect_p5(weights, uwc_dump_dir, step=0)
        except Exception as e:
            logger.warning(f"[UWC stage2] P5 collect failed: {e}")

    # UWC 阶段3（可选）：load_weights 前记录各参数 data_ptr
    if uwc_stage3 and os.environ.get("UWC_PTR_CHECK", "0") == "1":
        from verl.utils.uwc.collectors.stage3_vllm_collector import collect_ptr_before
        uwc_dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
        try:
            _uwc_ptr_before = collect_ptr_before(self.model_runner.model)
        except Exception as e:
            logger.warning(f"[UWC stage3 ptr] ptr_before collect failed: {e}")
            _uwc_ptr_before = None
    else:
        _uwc_ptr_before = None

    # ...（原有 load_weights / add_lora 逻辑完全不变）...

    # UWC 阶段3（可选）：load_weights 后判定漏同步层（指针未变化 => 漏同步）
    # 用真值判断（采集失败返回 None 或空 dict 时跳过），避免空结果被当作"全部通过"
    if uwc_stage3 and os.environ.get("UWC_PTR_CHECK", "0") == "1" and _uwc_ptr_before:
        try:
            from verl.utils.uwc.collectors.stage3_vllm_collector import collect_ptr_after
            report = collect_ptr_after(self.model_runner.model, _uwc_ptr_before, uwc_dump_dir, step=0)
            if report.get("status") == "fail":
                logger.warning(f"[UWC stage3 ptr] {report.get('not_synced_layers')} layers not synced")
        except Exception as e:
            logger.warning(f"[UWC stage3 ptr] ptr_after collect failed: {e}")
```

#### 3.1.3 `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py`

**sender 侧诊断日志：`use_shm` 状态 + 前 3 个权重统计 + buffer 内容统计**

```python
# bucketed_weight_transfer.py —— send() 中，逐权重写 buffer 的同时打诊断
uwc_enable = os.environ.get("UWC_ENABLE", "0") == "1"
uwc_stage2 = uwc_enable and os.environ.get("UWC_STAGE2", "1") == "1"
if uwc_stage2:
    print(f"[UWC stage2 sender] use_shm={self.use_shm}, bucket_size={self.bucket_size} bytes, "
          f"zmq={self.zmq_handle}", flush=True)

diag_count = 0
async for name, weight in ensure_async_iterator(weights):
    # ...（原有：装桶 / 发送 bucket 逻辑不变）...
    old_offset = offset                              # 记录当前权重起始偏移（诊断用，offset 随后会自增）
    self.buffer[old_offset : old_offset + weight.nbytes].copy_(
        weight.view(-1).view(torch.uint8), non_blocking=True)
    offset += weight.nbytes

    # UWC stage2: sender diag on the first few weights
    if uwc_stage2 and diag_count < 3:
        diag_count += 1
        wf = weight.float()
        get_torch_device().synchronize()             # 确保 non_blocking 拷贝完成后再读 buffer，避免读到旧/半拷贝数据
        buf_float = self.buffer[old_offset : old_offset + weight.nbytes].float()
        print(f"[UWC stage2 sender diag] {name}: device={weight.device}, dtype={weight.dtype}, "
              f"shape={list(weight.shape)}, mean={wf.mean().item():.6f}, std={wf.std().item():.6f}, "
              f"data_ptr={weight.data_ptr()}, "
              f"sender_buf mean={buf_float.mean().item():.6f}, std={buf_float.std().item():.6f}", flush=True)
```

#### 3.1.4 `verl/workers/engine_workers.py`

**`update_weights` send 前打印 `per_tensor_param` 前 3 个 tensor 统计**

```python
# engine_workers.py —— update_weights 中，get_per_tensor_param 返回后、send 之前
per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
    layered_summon=self.layered_summon, base_sync_done=True
)

if os.environ.get("UWC_ENABLE", "0") == "1":
    try:
        for i, (name, t) in enumerate(per_tensor_param):
            if i >= 3:
                break
            tf = t.float()
            print(f"[UWC engine diag] {name}: device={t.device}, dtype={t.dtype}, "
                  f"shape={list(t.shape)}, mean={tf.mean().item():.6f}, std={tf.std().item():.6f}, "
                  f"data_ptr={t.data_ptr()}", flush=True)
    except Exception as e:
        print(f"[UWC engine diag] error: {e}", flush=True)
```

> `compute_log_prob` 中另有一段**默认注释关闭**的训练权重 dump（调试用），需要时去掉注释即可用 `collect_p2` 在推理步 dump 训练侧权重与推理侧比对。

#### 3.1.5 `verl/workers/rollout/vllm_rollout/vllm_rollout.py`

**新增 `VERL_FORCE_USE_SHM` 环境变量，强制走共享内存传输（绕过 NPU IPC 误判）**

```python
# vllm_rollout.py —— __init__ 中确定传输方式
force_shm = os.environ.get("VERL_FORCE_USE_SHM", "0") == "1"
self.use_shm = force_shm or not is_support_ipc()
if self.use_shm:
    reason = "VERL_FORCE_USE_SHM=1 (forced)" if force_shm else "IPC is not supported on your devices"
    logger.warning(f"{reason}. Falling back to shared memory for weight transfer, "
                   "which may cause performance degradation. If you are using Ascend NPUs, "
                   "please ensure your software and CANN toolkit versions meet the IPC requirements.")
```

### 3.2 UWC 工具包（新增 `verl/utils/uwc/`，参考示例）

工具包目录结构如下，**采集器（collectors）是给用户扩展采集点的参考示例**——新采集点按同样模式写即可（环境变量开关 + try-except + 仅 rank 0 落盘）。

```
verl/utils/uwc/
├── comparator.py                        # 同格式权重数值比对（exact / atol）
├── logits_comparator.py                 # logits 余弦相似度比对
├── name_normalizer.py                   # 命名归一化 + QKV/gate_up 合并拆分
├── collectors/
│   ├── stage1_bridge_collector.py       # P0 / P2 / load_p0_weights（诊断用）
│   ├── stage2_transfer_collector.py     # P2_sender / P5
│   └── stage3_vllm_collector.py         # data_ptr 指针偏移验证（推理侧）
└── offline/
    ├── uwc_compare.py                   # 一键离线比对（stage1/2/3，产出 report.json）
    ├── forward_verify.py                # 阶段1 load 前向校验（G0 vs G1）
    ├── stage3_diff_test.py              # 阶段3 dummy vs safetensors 差分
    └── uwc_summary.py                   # 全链路总结 summary.json
```

> 注：推理侧不做分片反向还原，故无 `vllm_weight_restorer.py` / `stage3_inference_collector.py`。

**参考示例 1：`collectors/stage1_bridge_collector.py`（P0 / P2 采集）**

```python
def collect_p0(bridge, local_path, dump_dir, step=0):
    """采集 P0：原始 HF 权重（bridge.load_weights 之前），仅 rank 0 落盘。"""
    rank = _get_rank()
    if rank != 0:
        return None
    from safetensors.torch import load_file
    from transformers import AutoConfig
    AutoConfig.from_pretrained(local_path, trust_remote_code=True)   # 预加载 config
    weights = {}
    for fname in sorted(os.listdir(local_path)):
        if fname.endswith(".safetensors"):
            weights.update(load_file(os.path.join(local_path, fname)))
    filtered = {n: t.cpu() for n, t in weights.items()
                if "rope_freqs" not in n and "rotary_emb.inv_freq" not in n}  # 过滤非权重项
    out_dir = os.path.join(dump_dir, "stage1_bridge")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"P0_hf_original_step{step}_rank{rank}.pt")
    torch.save(filtered, out_path)
    logger.info(f"[UWC stage1] P0 saved to {out_path}, keys={len(filtered)}")
    return out_path


def collect_p2(bridge, module, dump_dir, step=0):
    """采集 P2：bridge.export_weights 之后的 HF 权重。

    注意：export_weights 内部包含 PP/TP 集合通信（all_gather/broadcast），
    所有 rank 都必须消费这个 generator，否则非 rank 0 进程会卡在集合通信上
    （死锁）。因此所有 rank 迭代 generator，只让 rank 0 落盘。
    """
    rank = _get_rank()
    weights = {}
    try:
        for name, tensor in bridge.export_weights(module):
            if rank == 0:
                weights[name] = tensor.clone().cpu()      # 物化 + clone，避免延迟访问
    except Exception as e:
        logger.warning(f"[UWC stage1] P2 export failed: {e}")
        return None
    if rank != 0:
        return None
    out_dir = os.path.join(dump_dir, "stage1_bridge")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"P2_hf_exported_step{step}_rank{rank}.pt")
    torch.save(weights, out_path)
    logger.info(f"[UWC stage1] P2 saved to {out_path}, keys={len(weights)}")
    return out_path
```

> 同步时刻 sender 侧还会用 `load_p0_weights(local_path)` 从磁盘加载 P0（按 `local_path` 缓存，多次 sync 不重复读盘），用于实时计算 `vs_P0_max_diff`。

**参考示例 2：`collectors/stage2_transfer_collector.py`（P2_sender / P5 采集）**

```python
def collect_p2_sender(weights, dump_dir, step=0):
    """采集 P2_sender：Sender 实际发送的 HF 权重，仅 rank 0 落盘。"""
    rank = _get_rank()
    if rank != 0:
        return None
    _sync_device()   # NPU 上 IPC/SHM 是异步的，先同步再拷贝
    saved = []
    for name, t in weights:
        cpu_t = torch.empty_like(t, device="cpu", dtype=t.dtype)
        cpu_t.copy_(t)
        saved.append((name, cpu_t))
    out_dir = os.path.join(dump_dir, "stage2_transfer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"P2_sender_step{step}_rank{rank}.pt")
    torch.save(saved, out_path)
    print(f"[UWC stage2] P2_sender saved to {out_path}, keys={len(saved)}", flush=True)
    return out_path


def collect_p5(weights, dump_dir, step=0):
    """采集 P5：Receiver 接收到的 HF 权重（_update_weights 入口参数）。

    与 collect_p2_sender 一致，仅 rank 0 落盘（每 rank 内容相同，避免多 rank
    重复写盘与存储浪费）。
    """
    rank = _get_rank()
    if rank != 0:
        return None
    _sync_device()   # 同步后再 clone，避免读到未同步内存
    saved = []
    for name, t in weights:
        cpu_t = torch.empty_like(t, device="cpu", dtype=t.dtype)
        cpu_t.copy_(t)
        saved.append((name, cpu_t))
    out_dir = os.path.join(dump_dir, "stage2_transfer")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"P5_receiver_step{step}_rank{rank}.pt")
    torch.save(saved, out_path)
    print(f"[UWC stage2] P5 saved to {out_path}, keys={len(saved)}", flush=True)
    return out_path


def _sync_device():
    """NPU 上 IPC/SHM 传输是异步的，先同步再拷贝；CUDA 环境同理兜底。"""
    try:
        torch.npu.synchronize()
    except (AttributeError, RuntimeError):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

**参考示例 3：`collectors/stage3_vllm_collector.py`（data_ptr 指针偏移验证）**

```python
def collect_ptr_before(model):
    """load_weights 前记录各参数 data_ptr。

    采集失败返回 None（而非空 dict）：调用侧用真值判断 _uwc_ptr_before，
    避免空 dict 恒被判 'pass'，把采集失败静默伪装成"全部同步"。
    """
    ptr_before = {}
    try:
        for name, param in model.named_parameters():
            ptr_before[name] = param.data_ptr()
    except Exception as e:
        logger.warning(f"[UWC stage3 ptr] collect_ptr_before failed: {e}")
        return None
    return ptr_before


def collect_ptr_after(model, ptr_before, dump_dir, step=0) -> dict:
    """load_weights 后比对 data_ptr：ptr 未变化 => 该层漏同步。"""
    report = {"status": "pass", "checked_layers": len(ptr_before),
              "synced_layers": 0, "not_synced_layers": 0, "details": []}

    if not ptr_before:
        # before 采集失败/为空：显式告警并置 error，避免误报 pass
        report["status"] = "error"
        report["error"] = "ptr_before 为空（采集失败或模型无参数），data_ptr 判定被跳过"
        logger.warning(f"[UWC stage3 ptr] {report['error']}")
    else:
        not_synced = []
        for name, param in model.named_parameters():
            if name not in ptr_before:
                continue
            synced = param.data_ptr() != ptr_before[name]
            report["synced_layers" if synced else "not_synced_layers"] += 1
            report["details"].append({"name": name, "ptr_before": ptr_before[name],
                                      "ptr_after": param.data_ptr(), "synced": synced})
            if not synced:
                not_synced.append(name)
        report["status"] = "pass" if not not_synced else "fail"
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    out_path = os.path.join(dump_dir, "stage3_vllm", f"ptr_before_after_step{step}_rank{rank}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
```

---

## 四、环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `UWC_ENABLE` | 总开关，1 启用 | 0 |
| `UWC_STAGE1` | 阶段1 采集开关（init 时刻 P0/P2） | 1（`UWC_ENABLE=1` 时） |
| `UWC_STAGE2` | 阶段2 采集开关（P2_sender / P5） | 1（`UWC_ENABLE=1` 时） |
| `UWC_STAGE3` | 阶段3 开关（data_ptr 指针偏移 + dummy/safetensors 差分） | 1（`UWC_ENABLE=1` 时） |
| `UWC_DUMP_DIR` | 采集数据存放目录 | /tmp/uwc_dump |
| `UWC_PTR_CHECK` | 阶段3 data_ptr 指针偏移采集（推理侧漏同步精确定位） | 0 |
| `VERL_FORCE_USE_SHM` | 强制共享内存传输，绕过 NPU IPC 误判 | 0 |

> 建议在脚本里显式设置各 `UWC_STAGE*`，避免默认值触发不想要的采集路径。

---

## 五、使用指南

### 5.1 阶段1：Bridge 转换验证（验证训练侧 HF ↔ mcore 格式转换无损）

```bash
# 1. 开启 stage1（init 时自动采集 P0/P2）
export UWC_ENABLE=1 UWC_STAGE1=1 UWC_STAGE2=0 UWC_STAGE3=0
export UWC_DUMP_DIR=/path/to/uwc_dump
# 2. 运行训练（Qwen2.5-0.5B-Instruct + GRPO，启动即可，无需跑完整步）
# 3. 离线比对
python -m verl.utils.uwc.offline.uwc_compare --dump_dir $UWC_DUMP_DIR --stage 1 \
    --model_config /path/to/config.json   # P2 是合并格式（qkv_proj/gate_up_proj）时需要
```

阶段1 可选的前向校验（load 语义，G0 vs G1，验证 load_weights 后 mcore 前向与 transformers 一致）：

```bash
python -m verl.utils.uwc.offline.forward_verify \
    --local_path /path/to/model_safetensors \
    --dump_dir $UWC_DUMP_DIR --input_ids 1,2,3,4,5
# 需在 NPU + mbridge 环境、单卡运行；fp32 / TP=1 / PP=1 / SP 关闭 / eval 模式
```

### 5.2 阶段2：传输一致性验证（验证 P2_sender → P5 跨进程传输无损）

```bash
export UWC_ENABLE=1 UWC_STAGE1=0 UWC_STAGE2=1 UWC_STAGE3=0
export UWC_DUMP_DIR=/path/to/uwc_dump
# 运行训练，至少跑 1 个训练步（触发首次 update_weights → P2_sender + P5 落盘）
# 结束后比对（P2_sender vs P5，exact=True）
python -m verl.utils.uwc.offline.uwc_compare --dump_dir $UWC_DUMP_DIR --stage 2
```

日志里应看到：

- `[UWC stage2] P2_sender saved ... keys=N`
- `[UWC stage2] P5 saved ... keys=N`
- `[UWC export diag] ... vs_P0_max_diff` —— step0 应为 0 左右，若极大说明 sender 侧是 dummy

### 5.3 阶段3：vLLM 指针偏移验证（推理侧加载验证）

> 推理侧只做**指针偏移（data_ptr）验证** + **dummy vs safetensors 差分测试**，不做分片反向还原。

**① dummy vs safetensors 差分测试**（黑盒行为校验，"某些层漏同步"）：

```bash
# 分别用 load_format=dummy / load_format=safetensors 各跑一次 RL 首步，
# 把两次推理输出存为 json，然后离线判定
python -m verl.utils.uwc.offline.stage3_diff_test \
    --dummy /path/to/dummy_rl_first_step_output.json \
    --safetensors /path/to/safetensors_rl_first_step_output.json \
    --dump_dir $UWC_DUMP_DIR
```

判定矩阵：

| dummy 乱码 | safetensors 乱码 | 结论 |
|-----------|----------------|------|
| 是 | 否 | 某些层漏同步 → 查 vLLM load_weights 覆盖范围 |
| 是 | 是 | 同步的是错误值 → 回查阶段 1/2 |
| 否 | 否 | 阶段 3 通过 |
| 否 | 是 | （罕见）safetensors 加载问题或同步覆盖坏了正确权重 |

**② data_ptr 指针偏移验证**（精确定位具体哪层漏同步）：

```bash
export UWC_ENABLE=1 UWC_STAGE3=1 UWC_PTR_CHECK=1
export UWC_DUMP_DIR=/path/to/uwc_dump
# load_weights 前后采集各参数 data_ptr()，未同步的层 ptr 不变（指针没分离）
# 报告输出到 $UWC_DUMP_DIR/stage3_vllm/ptr_before_after_*.json
# 汇总阶段3 报告（差分测试 + data_ptr）：
python -m verl.utils.uwc.offline.uwc_compare --dump_dir $UWC_DUMP_DIR --stage 3
```

### 5.4 全链路总结

```bash
python -m verl.utils.uwc.offline.uwc_summary --dump_dir $UWC_DUMP_DIR
# 汇总 stage1/2/3 报告，产出 summary.json
```

---

## 六、采集数据结构

```
$UWC_DUMP_DIR/
├── stage1_bridge/
│   ├── P0_hf_original_step0_rank0.pt     # 原始 HF 权重（dict）
│   ├── P2_hf_exported_step0_rank0.pt     # init 时刻 export 后 HF 权重（dict）
│   ├── G0_logits_gt.pt                   # 前向校验：transformers logits（可选）
│   └── G1_logits_mcore.pt                # 前向校验：mcore logits（可选）
├── stage2_transfer/
│   ├── P2_sender_step0_rank0.pt          # 同步时刻 sender 实际发送权重（list[(name,tensor)]）
│   └── P5_receiver_step0_rank0.pt        # 接收端收到的权重（list[(name,tensor)]）
├── stage3_vllm/
│   ├── diff_test_report.json             # dummy vs safetensors 差分报告
│   └── ptr_before_after_*.json           # data_ptr 指针偏移报告（UWC_PTR_CHECK=1）
├── stage1/report.json                    # 比对报告
├── stage2/report.json
├── stage3/report.json
└── summary.json                          # 全链路总结
```

---

## 七、比对容差与使用约束

### 7.1 比对容差

| 阶段 | 比对 | 容差 |
|------|------|------|
| 阶段1 export | P0 vs P2 | `atol=1e-3`（bf16 量级） |
| 阶段1 load | G0 vs G1 | avg_cos>0.99 且 max/min_cos>=0.95 |
| 阶段2 传输 | P2_sender vs P5 | `exact=True`（`max_abs_diff==0`） |
| 阶段3 推理侧 | dummy vs safetensors 差分 + data_ptr 指针偏移 | 行为判定（差分矩阵）+ ptr 是否变化 |

> 阶段3 不做分片反向还原（P5 vs P7 数值比对不在范围内）。

### 7.2 使用约束

- **阶段2 必须用同步时刻的 P2_sender**，不能用 init 时刻的 P2——训练后权重已变化，init 时刻 P2 会合法误报
- **阶段1 前向校验约束**（`forward_verify.py`）：fp32 跑前向、固定 input_ids、TP=1 PP=1、关闭 sequence_parallel、两侧都 eval 模式关闭 dropout、attention 实现 / RoPE / norm eps / 激活函数两侧对齐（建议从同一份 config.json 派生）
- **P0/P2/P2_sender/P5 每 rank 内容相同**，比对只用 rank 0，采集也只落盘 rank 0
- **阶段3 推理侧只做指针偏移验证**：data_ptr 判定"该层参数是否被 load_weights 覆盖"（ptr_after==ptr_before → 漏同步），配合 dummy vs safetensors 差分做行为校验；**不做**分片反向还原的验证与实现

### 7.3 已知局限

- 阶段3 差分测试抓不住"首步后才坏"的指针复用问题（`param.data` 别名传入 buffer 后被复用），需要配合 data_ptr 或多步跨 sync 测试
- 阶段2 只比发送端与接收端，定位不到"拷贝操作错"还是"传输通道错"；需要时可加中间采集点 P2' 做三点比对
- 阶段1 的 P0 vs P2 命名归一化依赖模型结构（qkv/gate_up 合并规则），不同架构需核对 `name_normalizer.py`

---

## 八、验证结果与结论

以下结果在 **Qwen2.5-0.5B-Instruct，Ascend NPU，GRPO** 环境下实测得到：

| 比对 | 结果 |
|------|------|
| P0 vs P2（阶段1 export） | ✅ 290/290，max_abs_diff=0 |
| P2_sender vs P5（阶段2 传输） | ✅ 290/290，max_abs_diff=0 |
| export vs P0 实时抽样（vs_P0_max_diff） | step0=0.000000；训练1步后=0.000002（lr=1e-6） |

**结论**：在 Qwen2.5-0.5B-Instruct / NPU / GRPO 场景下，Bridge export 与传输链路数值零差异，sender 侧发送的即为正确训练权重，训练→推理权重同步链路无损。使用时只需按第五节的命令依次跑阶段1/2/3，任一阶段 report 显示 fail 即可把问题定位到对应环节。
