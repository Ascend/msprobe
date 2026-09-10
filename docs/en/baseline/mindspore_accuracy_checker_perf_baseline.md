# Precision Pre-check Baseline in MindSpore

## Precision Pre-check Time Reference Baseline in "multi_acc_check" Mode

This baseline is the time reference baseline for precision pre-check in `multi_acc_check` mode in the MindSpore framework. It shows the time changes of a 38B language model across different rank configurations.

### Time Changes

| Number of Ranks | Total (Minute)| Remarks      |
| ----- |----------|---------- |
| 1 | 21.0     | Single-rank baseline   |
| 2 | 11.5     | Dual-rank baseline   |
| 4 | 6.7      | Four-rank baseline   |
| 8 | 3.5      | Eight-rank baseline   |
