import torch_npu
import torch
import torch.nn as nn

m=nn.Conv3d(1, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))
input = torch.randn(8, 1, 16, 96, 96).float()
output = m(input)
print(output.size())


m=nn.Conv3d(1, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1)).npu()
input = torch.randn(8, 1, 16, 96, 96).float().npu()
output = m(input)
print(output)
print(output.size())




