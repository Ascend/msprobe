# Training-Inference Weight Consistency Verification and Localization Guide

## 1. Overview

### 1.1 Definition

The Unified Weight Consistency (UWC in this document), a training-inference weight consistency verification method, aims to verify the numerical consistency of the data at each stage in the weight synchronization link from the training side (veRL + Megatron-core) to the inference side (vLLM) in the reinforcement learning (RL) training-inference integrated scenario.

### 1.2 Decomposition of the UWC Verification Link

UWC decouples the complete weight synchronization link into three mutually independent verification stages that can be enabled separately. Each stage controls the collection switch through environment variables; it is disabled by default and can be enabled at any time without affecting production training performance.

> [!NOTE]
>
> In each stage below, the data collected at each rank node is the same, so only the data of rank 0 needs to be saved during comparison, to save storage.

#### Stage 1: Training-Side Format Conversion Verification (Bridge)

This stage verifies whether the conversion code between the HF format and the mcore format is correct, ensuring "logical equivalence".

This stage contains a two-layer defense mechanism:

1. Execute Bridge: the action is to first perform a load_weights (converting the HF-format model weights into the mcore format), and then perform an export_weights (converting the mcore-format model weights back to the HF format). Before load_weights, the original HF-format model weights are P0; after export_weights, the converted HF-format model weights P2 are obtained.
   - Verification point: P0 vs P2
   - Meaning: Compare whether the original weights P0 and the converted weights P2 are consistent.
   - Purpose: Determine the correctness of the format conversion code by verifying whether the model weights before and after Bridge are consistent.
2. transformers loads the initial HF-format model weights and performs one forward propagation on the same input, with the output logits being G0; veRL loads the mcore-format model weights converted by load_weights and performs one forward propagation (or the forward computation in the Rollout stage) on the same input, with the output logits being G1.
   - Verification point: G0 vs G1
   - Meaning: Compare whether G0 and G1 are consistent.
   - Purpose: Verify the correctness of the load_weights stage. If the load_weights conversion is correct and the HF model and mcore model architectures are exactly the same, then G0 ≈ G1 (allowing tiny floating-point precision differences, such as the 10⁻⁵ level); if G0 and G1 differ greatly, it indicates that the load_weights conversion is wrong, or the model architecture and configuration do not match.

#### Stage 2: Transport Integrity Verification (Transport)

Shared memory or IPC may cause data truncation or misalignment under some extreme concurrency or memory alignment conditions. This stage verifies the integrity of the transport.

- Action: Transfer the weights from the training process to the inference process through shared memory or IPC (inter-process communication).
- Verification point: P2_sender (the HF weights actually sent by the sender at the synchronization moment) vs P5 (the HF weights received by the Receiver)
- Meaning: Compare whether P2_sender and P5 are consistent.
- Purpose: Ensure that after cross-process and cross-node transfer, the weight values are exactly consistent. Ensure that the transport channel has not corrupted the data.

#### Stage 3: Inference-Side Loading Verification (Loading)

When the training side (veRL) switches to the inference side (vLLM), this is the final verification step for the correctness of model weight loading.

This stage contains a two-layer defense mechanism:

1. dummy vs safetensors differential test: Load the model with `load_format=dummy` and `load_format=safetensors` respectively and run the first-step inference of RL, saving the outputs as `dummy_rl_first_step_output.json` and `safetensors_rl_first_step_output.json`.
   - Verification point: `dummy_rl_first_step_output.json` vs `safetensors_rl_first_step_output.json`
   - Meaning: Compare the output generated using the dummy loading logic with the output generated using the real safetensors weight loading.
   - Purpose: Determine whether the model is loaded correctly by comparing whether the dummy output and the safetensors output are different; a black-box behavior check to determine whether missing synchronization occurs.
2. data_ptr pointer offset verification: Start the vLLM service, record the data_ptr of each parameter before loading (ptr_before), and trigger load_weights to record the data_ptr of each parameter after loading (ptr_after).
   - Verification point: ptr_before vs ptr_after
   - Meaning: Compare whether the weight loaded by the parameter before and after load_weights is the same. If it is the same (but the layer should actually have been updated), the layer is judged to be missing synchronization, so it must be ensured that both ptr_before and ptr_after of these layers are different.
   - Purpose: Verify that when vLLM loads the safetensors file and builds the internal KV Cache structure, the memory pointer mapping is correct; precisely locate the layers with missing synchronization.

> [!NOTE]
>
> The inference side only performs the dummy vs safetensors differential test and the data_ptr pointer offset verification, and does not perform shard reverse restoration.

### 1.2 Core Value and Problems Solved

- **The source of weight inconsistency is hard to locate**: The training-side weights go through three processes—bridge conversion, cross-process transfer, and vLLM loading—before reaching the inference side. An error in any step will cause RL inference garbling, and it is extremely hard to troubleshoot. UWC splits the link into three segments, and each segment produces an independent comparison report, which can **quickly locate which segment the inconsistency occurs in** (bridge conversion/transport channel/vLLM loading).
- **"dummy weights" are hard to distinguish**: Through the real-time sampling diagnosis of `vs_P0_max_diff` on the sender side + numerical comparison on the receiver side, it can distinguish "wrong weights were transferred" from "some layers were missing synchronization".
- **Missing synchronization located to a specific layer**: The data_ptr pointer offset verification on the inference side can precisely identify "which layer's parameters are not covered by `load_weights`".
- **Black-box behavior fallback**: The dummy vs safetensors differential test does not depend on the internal implementation and directly compares the outputs of two RL first-step inferences as a behavior-level check.

## 2. Applicable Models and Environment

This document as a whole is based on the following combination:

- **Model**: `Qwen2.5-0.5B-Instruct`
- **Hardware environment**: **Ascend NPU**
- **Training algorithm**: **GRPO**
- **Training-inference framework**: the training-inference integrated scenario of **veRL + Megatron-core (mcore) training + vLLM inference**

> The following code snippets and commands are written based on this combination; other models/frameworks need to be adjusted according to the comments (such as naming normalization rules and `torch.npu` synchronization).

## 3. UWC Environment Variables

| Variable | Description | Default Value |
|------|------|--------|
| `UWC_ENABLE` | Master switch, 1 to enable | 0 |
| `UWC_STAGE1` | Stage 1 collection switch (P0/P2 at init time) | 1 (when `UWC_ENABLE=1`) |
| `UWC_STAGE2` | Stage 2 collection switch (P2_sender/P5) | 1 (when `UWC_ENABLE=1`) |
| `UWC_STAGE3` | Stage 3 switch (data_ptr pointer offset + dummy/safetensors differential) | 1 (when `UWC_ENABLE=1`) |
| `UWC_DUMP_DIR` | Directory for storing collected data | /tmp/uwc_dump |
| `UWC_PTR_CHECK` | Stage 3 data_ptr pointer offset collection (precise localization of missing synchronization on the inference side) | 0 |
| `VERL_FORCE_USE_SHM` | Force shared memory transfer, bypassing NPU IPC misjudgment | 0 |

> It is recommended to explicitly set each `UWC_STAGE*` in the script, to avoid the default values triggering unwanted collection paths.

---

## 4. Code Changes

A total of 5 files are modified on the collection side, all of which are best-effort collection wrapped with environment variable switches + try-except, without changing the original weight loading, training, and synchronization logic.

### 4.1 verl/workers/engine/megatron/transformer_impl.py

(1) Collect P0/P2 in the init stage (when `UWC_ENABLE=1` + `UWC_STAGE1=1`)

Insert collection before and after model initialization and `load_weights`. Key point: `load_weights` must be executed unconditionally **outside** the guard (every engine must load), and collection is performed only once, with any exception not affecting loading.

```python
# transformer_impl.py —— init/weight loading stage
if self.vanilla_bridge:
    if os.environ.get("UWC_ENABLE", "0") == "1" and os.environ.get("UWC_STAGE1", "1") == "1":
        global _init_weights_dumped
        if not _init_weights_dumped:
            _init_weights_dumped = True
            # P0 collection (best-effort, failure does not affect loading)
            try:
                from verl.utils.uwc.collectors.stage1_bridge_collector import (
                    collect_p0, collect_p2,
                )
                dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
                collect_p0(self.bridge, self.model_config.local_path, dump_dir, step=0)
            except Exception as e:
                print(f"[UWC stage1] P0 collect failed, continue training: {e}", flush=True)
            # bridge conversion (must be executed, cannot be put inside the collection guard)
            self.bridge.load_weights(module, self.model_config.local_path)
            # P2 collection (best-effort)
            try:
                collect_p2(self.bridge, module, dump_dir, step=0)
            except Exception as e:
                print(f"[UWC stage1] P2 collect failed, continue training: {e}", flush=True)
        else:
            # Already dumped; only load the weights, no more repeated collection
            self.bridge.load_weights(module, self.model_config.local_path)
    else:
        self.bridge.load_weights(module, self.model_config.local_path)
```

(2) `get_per_tensor_param`: export materialization + real-time diagnosis + persisting P2_sender

Core changes on the sender side at the synchronization moment: **immediately materialize and clone** the generator returned by `export_weights` (to avoid delayed access reading offloaded/reused data); print statistics for the first 3 tensors and compare them in real time with the on-disk P0 via `vs_P0_max_diff` (to determine whether export is dummy); when `UWC_STAGE2=1`, persist the actually sent weights as `P2_sender`.

```python
# transformer_impl.py —— get_per_tensor_param (synchronization moment, sender side)
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

    # UWC stage 2: persist on the sender side (the actually sent weights, compared one-to-one with P5 on the receiver side)
    if os.environ.get("UWC_STAGE2", "1") == "1":
        try:
            from verl.utils.uwc.collectors.stage2_transfer_collector import collect_p2_sender
            uwc_dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
            collect_p2_sender(per_tensor_param, uwc_dump_dir, step=0)
        except Exception as e:
            print(f"[UWC stage2] P2_sender collect failed: {e}", flush=True)
else:
    # In non-diagnostic mode, also materialize + clone to avoid the generator delayed access problem
    per_tensor_param = [(name, t.clone()) for name, t in per_tensor_param]
```

### 4.2 verl/workers/rollout/vllm_rollout/utils.py

`_update_weights` entry: collect P5 + data_ptr before and after

Core changes on the receiver side: persist P5 at the entry (`UWC_STAGE2`); when `UWC_PTR_CHECK=1`, collect data_ptr before and after `load_weights` (stage 3 inference-side pointer offset verification).

```python
# vllm_rollout/utils.py —— _update_weights entry (receiver side)
def _update_weights(self, weights, peft_config, base_sync_done):
    uwc_enable = os.environ.get("UWC_ENABLE", "0") == "1"
    uwc_stage2 = uwc_enable and os.environ.get("UWC_STAGE2", "1") == "1"
    uwc_stage3 = uwc_enable and os.environ.get("UWC_STAGE3", "1") == "1"

    # UWC stage 2: collect P5 (the HF weights received by the Receiver)
    if uwc_stage2:
        from verl.utils.uwc.collectors.stage2_transfer_collector import collect_p5
        uwc_dump_dir = os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump")
        try:
            collect_p5(weights, uwc_dump_dir, step=0)
        except Exception as e:
            logger.warning(f"[UWC stage2] P5 collect failed: {e}")

    # UWC stage 3 (optional): record the data_ptr of each parameter before load_weights
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

    # ... (the original load_weights/add_lora logic is completely unchanged) ...

    # UWC stage 3 (optional): determine the missing-sync layers after load_weights (pointer unchanged => missing sync)
    # Use a truth-value check (skip when collection fails and returns None or an empty dict), to avoid an empty result being treated as "all passed"
    if uwc_stage3 and os.environ.get("UWC_PTR_CHECK", "0") == "1" and _uwc_ptr_before:
        try:
            from verl.utils.uwc.collectors.stage3_vllm_collector import collect_ptr_after
            report = collect_ptr_after(self.model_runner.model, _uwc_ptr_before, uwc_dump_dir, step=0)
            if report.get("status") == "fail":
                logger.warning(f"[UWC stage3 ptr] {report.get('not_synced_layers')} layers not synced")
        except Exception as e:
            logger.warning(f"[UWC stage3 ptr] ptr_after collect failed: {e}")
```

### 4.3 verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py

Sender-side diagnostic log: `use_shm` status + statistics of the first 3 weights + buffer content statistics.

```python
# bucketed_weight_transfer.py —— in send(), print diagnostics while writing weights to the buffer one by one
uwc_enable = os.environ.get("UWC_ENABLE", "0") == "1"
uwc_stage2 = uwc_enable and os.environ.get("UWC_STAGE2", "1") == "1"
if uwc_stage2:
    print(f"[UWC stage2 sender] use_shm={self.use_shm}, bucket_size={self.bucket_size} bytes, "
          f"zmq={self.zmq_handle}", flush=True)

diag_count = 0
async for name, weight in ensure_async_iterator(weights):
    # ... (original: bucketing/sending bucket logic unchanged) ...
    old_offset = offset                              # Record the current weight start offset (for diagnostics; offset will be incremented afterwards)
    self.buffer[old_offset : old_offset + weight.nbytes].copy_(
        weight.view(-1).view(torch.uint8), non_blocking=True)
    offset += weight.nbytes

    # UWC stage2: sender diag on the first few weights
    if uwc_stage2 and diag_count < 3:
        diag_count += 1
        wf = weight.float()
        get_torch_device().synchronize()             # Ensure the non_blocking copy is complete before reading the buffer, to avoid reading stale/half-copied data
        buf_float = self.buffer[old_offset : old_offset + weight.nbytes].float()
        print(f"[UWC stage2 sender diag] {name}: device={weight.device}, dtype={weight.dtype}, "
              f"shape={list(weight.shape)}, mean={wf.mean().item():.6f}, std={wf.std().item():.6f}, "
              f"data_ptr={weight.data_ptr()}, "
              f"sender_buf mean={buf_float.mean().item():.6f}, std={buf_float.std().item():.6f}", flush=True)
```

### 4.4 verl/workers/engine_workers.py

In `update_weights`, print statistics for the first 3 tensors of `per_tensor_param` before `send`.

```python
# engine_workers.py —— in update_weights, after get_per_tensor_param returns and before send
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

> **Note**: There is also a commented-out training weight dump in `compute_log_prob` (for debugging); when needed, uncomment it to use `collect_p2` to dump the training-side weights at the inference step and compare them with the inference side.

### 4.5 verl/workers/rollout/vllm_rollout/vllm_rollout.py

Add the `VERL_FORCE_USE_SHM` environment variable to force transfer through shared memory (bypassing NPU IPC misjudgment).

```python
# vllm_rollout.py —— determine the transfer method in __init__
force_shm = os.environ.get("VERL_FORCE_USE_SHM", "0") == "1"
self.use_shm = force_shm or not is_support_ipc()
if self.use_shm:
    reason = "VERL_FORCE_USE_SHM=1 (forced)" if force_shm else "IPC is not supported on your devices"
    logger.warning(f"{reason}. Falling back to shared memory for weight transfer, "
                   "which may cause performance degradation. If you are using Ascend NPUs, "
                   "please ensure your software and CANN toolkit versions meet the IPC requirements.")
```

### 4.6 UWC collectors Configuration

```text
verl/utils/uwc/
├── comparator.py                        # Same-format weight numerical comparison (exact/atol)
├── logits_comparator.py                 # logits cosine similarity comparison
├── name_normalizer.py                   # Naming normalization + QKV/gate_up merge/split
├── collectors/
│   ├── stage1_bridge_collector.py       # P0/P2/load_p0_weights (for diagnostics)
│   ├── stage2_transfer_collector.py     # P2_sender/P5
│   └── stage3_vllm_collector.py         # data_ptr pointer offset verification (inference side)
└── offline/
    ├── uwc_compare.py                   # One-click offline comparison (stage1/2/3, produces report.json)
    ├── forward_verify.py                # Stage 1 load forward verification (G0 vs G1)
    ├── stage3_diff_test.py              # Stage 3 dummy vs safetensors differential
    └── uwc_summary.py                   # Full-link summary summary.json
```

> **Note**: The inference side does not perform shard reverse restoration, so there is no `vllm_weight_restorer.py`/`stage3_inference_collector.py`.

#### 4.6.1 P0 and P2 Collection

Add collection in the `collectors/stage1_bridge_collector.py` file. The example code is as follows:

```python
def collect_p0(bridge, local_path, dump_dir, step=0):
    """Collect P0: the original HF weights (before bridge.load_weights); only rank 0 persists to disk."""
    rank = _get_rank()
    if rank != 0:
        return None
    from safetensors.torch import load_file
    from transformers import AutoConfig
    AutoConfig.from_pretrained(local_path, trust_remote_code=True)   # Preload config
    weights = {}
    for fname in sorted(os.listdir(local_path)):
        if fname.endswith(".safetensors"):
            weights.update(load_file(os.path.join(local_path, fname)))
    filtered = {n: t.cpu() for n, t in weights.items()
                if "rope_freqs" not in n and "rotary_emb.inv_freq" not in n}  # Filter out non-weight entries
    out_dir = os.path.join(dump_dir, "stage1_bridge")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"P0_hf_original_step{step}_rank{rank}.pt")
    torch.save(filtered, out_path)
    logger.info(f"[UWC stage1] P0 saved to {out_path}, keys={len(filtered)}")
    return out_path


def collect_p2(bridge, module, dump_dir, step=0):
    """Collect P2: the HF weights after bridge.export_weights.

    Note: export_weights internally contains PP/TP collective communication (all_gather/broadcast),
    so all ranks must consume this generator; otherwise non-rank-0 processes will hang on the collective
    communication (deadlock). Therefore, all ranks iterate the generator, and only rank 0 persists to disk.
    """
    rank = _get_rank()
    weights = {}
    try:
        for name, tensor in bridge.export_weights(module):
            if rank == 0:
                weights[name] = tensor.clone().cpu()      # Materialize + clone to avoid delayed access
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

> **Note**: At the synchronization moment, the sender side also uses `load_p0_weights(local_path)` to load P0 from disk (cached by `local_path`, so repeated syncs do not re-read the disk), used to compute `vs_P0_max_diff` in real time.

#### 4.6.2 P2_sender and P5 Collection

Add collection in the `collectors/stage2_transfer_collector.py` file. The example code is as follows:

```python
def collect_p2_sender(weights, dump_dir, step=0):
    """Collect P2_sender: the HF weights actually sent by the Sender; only rank 0 persists to disk."""
    rank = _get_rank()
    if rank != 0:
        return None
    _sync_device()   # On NPU, IPC/SHM is asynchronous; synchronize first, then copy
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
    """Collect P5: the HF weights received by the Receiver (the _update_weights entry parameter).

    Consistent with collect_p2_sender, only rank 0 persists to disk (the content is the same for each
    rank, avoiding repeated disk writes and storage waste across multiple ranks).
    """
    rank = _get_rank()
    if rank != 0:
        return None
    _sync_device()   # Clone after synchronization to avoid reading unsynchronized memory
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
    """On NPU, IPC/SHM transfer is asynchronous; synchronize first, then copy; the CUDA environment is handled as a fallback."""
    try:
        torch.npu.synchronize()
    except (AttributeError, RuntimeError):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

#### 4.6.3 data_ptr Pointer Offset Verification Collection

Add collection in the `collectors/stage3_vllm_collector.py` file. The example code is as follows:

```python
def collect_ptr_before(model):
    """Record the data_ptr of each parameter before load_weights.

    Return None on collection failure (rather than an empty dict): the caller uses a truth-value check on
    _uwc_ptr_before, so an empty dict is not always judged 'pass', silently disguising a failed collection.
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
    """Compare data_ptr after load_weights: an unchanged ptr => this layer is missing synchronization."""
    report = {"status": "pass", "checked_layers": len(ptr_before),
              "synced_layers": 0, "not_synced_layers": 0, "details": []}

    if not ptr_before:
        # before collection failed/is empty: explicitly warn and set error, to avoid a false pass
        report["status"] = "error"
        report["error"] = "ptr_before is empty (collection failed or the model has no parameters); the data_ptr check is skipped"
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

## 5. Comparison Verification

### 5.1 Bridge Conversion Verification (Verifying That Training-side HF ↔ mcore Format Conversion Is Lossless)

```bash
# 1. Enable stage1 (P0/P2 are automatically collected at init)
export UWC_ENABLE=1 UWC_STAGE1=1 UWC_STAGE2=0 UWC_STAGE3=0
export UWC_DUMP_DIR=/path/to/uwc_dump
# 2. Run training (Qwen2.5-0.5B-Instruct + GRPO; just start it, no need to run a full step)
# 3. Offline comparison
python -m verl.utils.uwc.offline.uwc_compare --dump_dir $UWC_DUMP_DIR --stage 1 \
    --model_config /path/to/config.json   # Required when P2 uses the merged format (qkv_proj/gate_up_proj)
```

Optional forward verification for stage 1 (load semantics, G0 vs G1, verifying that the mcore forward after load_weights is consistent with transformers):

```bash
python -m verl.utils.uwc.offline.forward_verify \
    --local_path /path/to/model_safetensors \
    --dump_dir $UWC_DUMP_DIR --input_ids 1,2,3,4,5
# Must be run in the NPU + mbridge environment, single NPU; fp32/TP=1/PP=1/SP disabled/eval mode
```

### 5.2 Transport Consistency Verification (Verifying That P2_sender → P5 cross-process Transfer Is Lossless)

```bash
export UWC_ENABLE=1 UWC_STAGE1=0 UWC_STAGE2=1 UWC_STAGE3=0
export UWC_DUMP_DIR=/path/to/uwc_dump
# Run training for at least 1 training step (triggering the first update_weights → P2_sender + P5 persistence)
# Compare after completion (P2_sender vs P5, exact=True)
python -m verl.utils.uwc.offline.uwc_compare --dump_dir $UWC_DUMP_DIR --stage 2
```

The following should be seen in the log:

- `[UWC stage2] P2_sender saved ... keys=N`
- `[UWC stage2] P5 saved ... keys=N`
- `[UWC export diag] ... vs_P0_max_diff` —— step0 should be around 0; if it is extremely large, the sender side is dummy

### 5.3 vLLM Loading Verification (Inference-side Loading Verification)

> The inference side only performs the **pointer offset (data_ptr) verification** + the **dummy vs safetensors differential test**, and does not perform shard reverse restoration.

**dummy vs safetensors differential test** (black-box behavior check, "some layers are missing synchronization"):

```bash
# Run the RL first step once each with load_format=dummy/load_format=safetensors,
# save the two inference outputs as json, then determine offline
python -m verl.utils.uwc.offline.stage3_diff_test \
    --dummy /path/to/dummy_rl_first_step_output.json \
    --safetensors /path/to/safetensors_rl_first_step_output.json \
    --dump_dir $UWC_DUMP_DIR
```

Decision matrix:

| Dummy Garbled | Safetensors Garbled | Conclusion |
|-----------|----------------|------|
| Yes | No | Some layers are missing synchronization → check the vLLM load_weights coverage |
| Yes | Yes | The synchronized values are wrong → check back to stages 1/2 |
| No | No | Stage 3 passes |
| No | Yes | (Rare) safetensors loading problem or the synchronization overwrote the correct weights |

**data_ptr pointer offset verification** (precisely locate which specific layer is missing synchronization):

```bash
export UWC_ENABLE=1 UWC_STAGE3=1 UWC_PTR_CHECK=1
export UWC_DUMP_DIR=/path/to/uwc_dump
# Collect the data_ptr() of each parameter before and after load_weights; the ptr of unsynchronized layers does not change (the pointer is not separated)
# The report is output to $UWC_DUMP_DIR/stage3_vllm/ptr_before_after_*.json
# Summarize the stage 3 report (differential test + data_ptr):
python -m verl.utils.uwc.offline.uwc_compare --dump_dir $UWC_DUMP_DIR --stage 3
```

### 5.4 Full-Link Summary

```bash
python -m verl.utils.uwc.offline.uwc_summary --dump_dir $UWC_DUMP_DIR
# Summarize the stage1/2/3 reports and produce summary.json
```

---

## 6. Result File Introduction

```ColdFusion
$UWC_DUMP_DIR/
├── stage1_bridge/
│   ├── P0_hf_original_step0_rank0.pt  # Original HF weights (dict)
│   ├── P2_hf_exported_step0_rank0.pt  # HF weights after export at init time (dict)
│   ├── G0_logits_gt.pt                # Forward verification: transformers logits (optional)
│   └── G1_logits_mcore.pt             # Forward verification: mcore logits (optional)
├── stage2_transfer/
│   ├── P2_sender_step0_rank0.pt       # Weights actually sent by the sender at the synchronization moment (list[(name,tensor)])
│   └── P5_receiver_step0_rank0.pt     # Weights received by the receiver (list[(name,tensor)])
├── stage3_vllm/
│   ├── diff_test_report.json          # dummy vs safetensors differential report
│   └── ptr_before_after_*.json        # data_ptr pointer offset report (UWC_PTR_CHECK=1)
├── stage1/report.json                 # Comparison report
├── stage2/report.json
├── stage3/report.json
└── summary.json                       # Full-link summary
```

---

## 7. Comparison Tolerance and Usage Constraints

### 7.1 Comparison Tolerance

| Stage | Comparison | Tolerance |
|------|------|------|
| Stage 1 export | P0 vs P2 | `atol=1e-3` (bf16 magnitude) |
| Stage 1 load | G0 vs G1 | avg_cos>0.99 and max/min_cos>=0.95 |
| Stage 2 transport | P2_sender vs P5 | `exact=True` (`max_abs_diff==0`) |
| Stage 3 inference side | dummy vs safetensors differential + data_ptr pointer offset | behavior determination (differential matrix) + whether ptr changes |

> Stage 3 does not perform shard reverse restoration (the P5 vs P7 numerical comparison is out of scope).

### 7.2 Usage Constraints

- **Stage 2 must use the P2_sender at the synchronization moment**, and cannot use the P2 at init time—the weights have changed after training, and the P2 at init time will cause a legitimate false alarm
- **Stage 1 forward verification constraints** (`forward_verify.py`): run the forward with fp32, fix input_ids, TP=1 PP=1, disable sequence_parallel, disable dropout in eval mode on both sides, and align the attention implementation/RoPE/norm eps/activation functions on both sides (it is recommended to derive from the same config.json)
- **The content of P0/P2/P2_sender/P5 is the same for each rank**; comparison uses only rank 0, and collection also persists only rank 0 to disk
- **Stage 3 inference side only performs pointer offset verification**: data_ptr determines "whether the parameters of this layer are covered by load_weights" (ptr_after==ptr_before → missing synchronization), combined with the dummy vs safetensors differential for a behavior check; it **does not** verify or implement shard reverse restoration

### 7.3 Known Limitations

- The stage 3 differential test cannot catch the pointer reuse problem that "breaks only after the first step" (`param.data` is passed as an alias into the buffer and then reused); it needs to be combined with data_ptr or a multi-step cross-sync test
- Stage 2 only compares the sender and the receiver, and cannot locate whether "the copy operation is wrong" or "the transport channel is wrong"; if needed, an intermediate collection point P2' can be added for a three-point comparison
- The naming normalization for P0 vs P2 in stage 1 depends on the model structure (qkv/gate_up merge rules); different architectures need to check `name_normalizer.py`

---

## 8. Verification Results and Conclusion

The following results are measured in the **Qwen2.5-0.5B-Instruct, Ascend NPU, GRPO** environment:

| Comparison | Result |
|------|------|
| P0 vs P2 (stage 1 export) | ✅ 290/290, max_abs_diff=0 |
| P2_sender vs P5 (stage 2 transport) | ✅ 290/290, max_abs_diff=0 |
| export vs P0 real-time sampling (vs_P0_max_diff) | step0=0.000000; after 1 training step=0.000002 (lr=1e-6) |

**Conclusion**: In the Qwen2.5-0.5B-Instruct/NPU/GRPO scenario, the Bridge export and the transport link have zero numerical difference, the sender sends the correct training weights, and the training-to-inference weight synchronization link is lossless.
