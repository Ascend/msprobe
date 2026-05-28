# Enabling Tools for Common Frameworks

## Overview

This document describes how to enable debugging tools in common frameworks, including the dump tool and monitor tool.

## Tool Enablement

### Reference

- [Data Dump for PyTorch](<>)

### Method for Locating the Position to Add Tools

To locate the target position, you can print the call stack information of any API and find the position from the call stack.

For example, you can print the call stack of `linear` in either of the following ways:

#### Method 1: Printing the call stack in torch

![image.png](https://raw.gitcode.com/user-images/assets/7898473/b304f147-c369-471d-a460-a7e3f9bee152/image.png 'image.png')

#### Method 2: Replacing the original `linear` in the startup script

```python
import torch
import torch.nn.functional as F
import traceback

#  Save the original function.
original_linear = F.linear

#  Replace linear in functional and use *args and **kwargs to accommodate all parameters.
def custom_linear(*args, **kwargs):
    print("="*50)
    print("Call F.linear. The call stack is as follows:")
    traceback.print_stack()
    #  Pass the received parameters to the original function without any modification.
    return original_linear(*args, **kwargs)

F.linear = custom_linear
```

**Note**: Once the call stack is obtained, finding the position to add tools becomes easy.

![image.png](https://raw.gitcode.com/user-images/assets/7898473/0889ef2c-af39-494f-b302-c87f297eefb6/image.png 'image.png')

### Tool Adding Position in Common Frameworks

#### MindSpeed-LLM

![image.png](https://raw.gitcode.com/user-images/assets/7898473/0a8f98aa-cb39-4c3f-b772-eb7c36fc13a5/image.png 'image.png')

#### MindSpeed-MM

![image.png](https://raw.gitcode.com/user-images/assets/7898473/ddd75cf2-fabb-4ccf-a7cc-bbfd0a7bd6ff/image.png 'image.png')

#### LLaMA-Factory

![image.png](https://raw.gitcode.com/user-images/assets/7898473/1f1c7307-3f28-440a-8ee7-57e01a5c0fc1/image.png 'image.png')

#### accelerate + DeepSpeed

![image.png](https://raw.gitcode.com/user-images/assets/7898473/1de6a6f8-2a11-4a4b-97b3-d458f45f31cf/image.png 'image.png')

#### TorchTitan (FSDP2)

![image.png](https://raw.gitcode.com/user-images/assets/7898473/0ddc5861-2d0f-4a2e-8cec-18c4268cc649/image.png 'image.png')

#### verl (FSDP)

Positions where deterministic computing is enabled: 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/7c9eb36a-096a-42e4-8120-e44075978cb6/image.png 'image.png')

generate_sequences
![image.png](https://raw.gitcode.com/user-images/assets/7898473/a86b2a58-59f9-4abd-838d-274f4bb65653/image.png 'image.png')  
The preceding enabling modes are only applicable to the vLLM eager backend, which may vary depending on configurations.

update_actor
![image.png](https://raw.gitcode.com/user-images/assets/7898473/7edade77-bd4b-4d0b-ae08-3c72c6e81078/image.png 'image.png')

compute_log_prob
![image.png](https://raw.gitcode.com/user-images/assets/7898473/f0a85328-2860-44a2-8051-5322628032af/image.png 'image.png')

compute_ref_log_prob
![image.png](https://raw.gitcode.com/user-images/assets/7898473/ce7f7a79-3f7a-425b-9172-6ac41d716954/image.png 'image.png')

**Note**: The preceding enabling modes are only applicable to the vLLM eager backend. Positions and enabling modes may vary depending on configurations.
