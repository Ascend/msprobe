# Performance Baseline Report of the Response Anomaly Detection Tool

## Environment Information

CPU: Apple M5 Max (18 CPU cores, single-process CPU)

Memory: 128 GB

Python: 3.11.15

NumPy: 2.4.6

Test vocabulary: Synthetic ByteLevel BPE `tokenizer.json` (`vocab_size = 151,643`, aligned with mainstream large-vocabulary LLMs)

This baseline serves as a performance reference for the Response Anomaly detection tool in Default Mode (Token-Only Sequence). The performance baseline test data in the following scenarios comes from statistical measurements across multiple runs; actual runtime performance may slightly fluctuate depending on hardware configuration and environment status.

## Initialization Latency and Memory Footprint Baseline

This baseline serves as a reference for first-time vocabulary loading (cold start), in-process cache reuse (warm start), and retained memory footprint in Default Mode (Token-Only Sequence).

| Metric | Measured Result | Notes |
| :--- | :--- | :--- |
| **Cold `ILLDetector(tokenizer.json)`** | `12,552.1 ms` (~`12.6 s`) | First-time decoding of 151.6k vocabulary and rare-character / repetition lookup-table construction |
| **Warm `ILLDetector(tokenizer.json)`** | p50 `0.091 ms` / p95 `0.106 ms` | Hitting the in-process vocabulary cache |
| **Retained Memory (`tracemalloc`)** | `15.27 MB` | Includes one-time shared scoring model load (~`12.23 MB` per vocabulary bundle); peak during build: `54.51 MB` |

## Single-Request Detection Latency Baseline

This baseline serves as a reference for single-request full-scan detection latency across different output sequence lengths on normal non-anomalous text in Default Mode (Token-Only Sequence).

| Sequence Length (tokens) | Rare-Char Detection p50 | Repetition Detection p50 | `detector.detect` p50 / p95 | Per-Token Cost | `analyze_output_anomaly` p50 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | `0.021 ms` | `0.333 ms` | `0.354 ms` / `0.370 ms` | `2.77 μs/tok` | `0.495 ms` |
| 512 | `0.026 ms` | `1.463 ms` | `1.472 ms` / `1.552 ms` | `2.88 μs/tok` | `1.632 ms` |
| 1024 | `0.035 ms` | `3.053 ms` | `3.073 ms` / `3.168 ms` | `3.00 μs/tok` | `3.249 ms` |
| 2048 | `0.051 ms` | `6.524 ms` | `6.648 ms` / `6.865 ms` | `3.25 μs/tok` | `6.728 ms` |
| 4096 | `0.090 ms` | `12.266 ms` | `12.287 ms` / `12.792 ms` | `3.00 μs/tok` | `12.556 ms` |
| 8192 | `0.153 ms` | `20.709 ms` | `20.507 ms` / `20.930 ms` | `2.50 μs/tok` | `20.870 ms` |

Tier and stress workload reference (sequence length `2048` tokens): Normal sequences show nearly identical latency between `tier="intercept"` (p50 `6.599 ms`) and `tier="monitor"` (p50 `6.614 ms`); under a 5% cross-script stress workload triggering the GBDT gate, detection latency is p50 `6.961 ms` / p95 `10.365 ms`.

## Batch Detection Throughput Baseline

This baseline serves as a reference for batch detection latency and single-core CPU throughput across different batch sizes when calling `detector.run` on sequences of `2048` tokens in Default Mode (Token-Only Sequence).

| Batch Size | Total Tokens | Batch Latency (p50 / p95) | Amortized Per Request | Single-Core Throughput |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 2,048 | `6.63 ms` / `6.84 ms` | `6.634 ms` | `308,693 tok/s` |
| 8 | 16,384 | `52.67 ms` / `54.05 ms` | `6.584 ms` | `311,064 tok/s` |
| 32 | 65,536 | `218.92 ms` / `230.98 ms` | `6.841 ms` | `299,367 tok/s` |
