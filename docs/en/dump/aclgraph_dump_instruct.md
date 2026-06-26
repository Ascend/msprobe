# aclgraph_dump Instructions

## Overview

**aclgraph_dump** provides the `acl_save` API that saves tensors as `.pt` files, which is applicable to the aclgraph scenario.

## Preparations

**Environment Setup**

1. Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).
2. Include the `aclgraph_dump` module during compilation and installation from source code.

   ```bash
   python3 build.py -e include-mod=aclgraph_dump -e no-check=true
   ```

3. Install and correctly configure Ascend Extension for PyTorch (torch_npu) and CANN (following msProbe installation requirements).

**Constraints**

- Only the PyTorch framework is supported.
- torch_npu is required for compiling `aclgraph_dump`. If this module is not included, `msprobe.lib.aclgraph_dump_ext` cannot be imported.

## Quick Start

The following example shows how to save a tensor in the forward process:

```python
import torch
import torch_npu

from msprobe.pytorch import acl_save

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x):
        y = self.linear(x)
        # Save the intermediate tensor.
        acl_save(x, "./dump/linear_out.pt")
        return y

if __name__ == "__main__":
    model = ToyMode()
    x = torch.randn(2, 8)
    x.to("npu:4")
    out = model(x)
```

## Data Collection

### Function Overview

`acl_save` is used to save tensor data. After this API is called, a `.pt` file is saved.

### API Description

**Prototype**

```python
acl_save(x: torch.Tensor, path: str) -> torch.Tensor
```

**Parameters**

| **Parameter**| **Type**| **Description**| **Mandatory (Yes/No)**|
| --- | --- | --- | --- |
| x | torch.Tensor | Tensor to be saved.| Yes|
| path | String| Saving path (relative or absolute path), which is of the string type. The name of the actual flushed file is in the format of `{base}_{seq}.pt`. For example, if `./dump/act.pt` is passed, `./dump/act_0.pt` and `./dump/act_1.pt` are actually flushed. (The parent directory `./dump/` must be an existing directory.)| Yes|

**Returns**

Returns a tensor with the same shape as the input, which has no specific meaning and is used only to trigger the saving operation.

### Examples

#### 1. Single-point saving during inference

```python
from msprobe.pytorch import acl_save

logits = model(x)
acl_save(logits, "./dump/logits.pt")
```

#### 2. Saving the file called for multiple times

```python
for step in range(3):
    y = model(x)
    acl_save(y, "./dump/act.pt")
```

`./dump/act_0.pt`, `./dump/act_1.pt`, and `./dump/act_2.pt` are generated.

## Output Description

### Dump Result File

After `acl_save` is called, `.pt` files are generated in the directory specified by `path`.

### Data Parsing

Data is saved in the PyTorch `.pt` format (pickle serialization) and can be read using `torch.load`.

```python
import torch

tensor = torch.load("./dump/act_0.pt")
```

## Appendix

### FAQs

Failed to import msprobe.lib.aclgraph_dump_ext

Check:

- whether `--include-mod=aclgraph_dump` has been included during compilation and installation.
- whether torch_npu has been installed and the environment variables have been correctly configured.
- whether the system is Linux.

`Allocate SQ failed`

In versions earlier than CANN 8.5 (excluding), `Allocate SQ failed` may occur because SQ is not reused in earlier versions. You are advised to change `CurrentNPUStream` to `DefaultNPUStream` in `ccsrc/aclgraph_dump/aclgraph_dump.cpp` or upgrade CANN to 8.5.0 or later.
