import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import ascend_function
device = torch.device('npu' if torch.npu.is_available() else 'cpu')

def dataloader(batch_size):
  dataroot="/content/drive/My Drive/celeba"
  transform=transforms.Compose([ transforms.Resize(64),transforms.CenterCrop(64),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
  dataset=torchvision.datasets.MNIST(root=dataroot, train=True,transform=transform, download=True)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return data_loader
