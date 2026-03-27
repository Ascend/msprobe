# MindSpore动态图精度数据采集快速入门

## 概述

本文将展示如何在MindSpore动态图模式下使用msProbe工具进行精度数据采集。

**环境准备**

安装msProbe工具，详情请参见《[msProbe工具安装指南](../msprobe_install_guide.md)》。

## 操作步骤

1. 创建配置文件。

   请在当前目录下创建一个名为`config.json`的配置文件，将配置内容拷贝到文件中。

   ```json
   {
       "task": "statistics",
       "dump_path": "./output",
       "rank": [],
       "step": ["0-2"],
       "level": "L1",
       "statistics": {
           "scope": [],
           "list": [],
           "data_mode": [
               "all"
           ],
           "summary_mode": "statistics"
       }
   }
   ```

   以上配置参数详细介绍和使用请参见《[配置文件介绍](./config_json_introduct.md)》。

2. 创建模型脚本。

   在当前目录下创建一个Python脚本文件，例如`alexnet_model.py`，将如下代码拷贝到文件中。

   ```python
   import os
   import numpy as np
   import mindspore as ms
   from mindspore import nn, ops
   from mindspore import context, _device, set_deterministic
   from mindspore import Tensor
   from msprobe.mindspore import PrecisionDebugger, seed_all
   
   # 设置随机种子以确保结果可重现
   seed_all(seed=1234, mode=False, rm_dropout=True)
   
   # 配置文件路径
   script_dir = os.path.dirname(os.path.abspath(__file__))
   config_path = os.path.join(script_dir, 'config.json')
   
   # 初始化精度调试器
   debugger = PrecisionDebugger(config_path=config_path)
   
   # 设置MindSpore设备上下文
   context.set_context(mode=ms.PYNATIVE_MODE)
   
   context.set_context(device_target="Ascend", device_id=0)
   
   set_deterministic(True)
   print("Context set successfully. Please wait for the training task.")
   
   # 定义卷积层
   def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
       return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                        has_bias=has_bias, pad_mode=pad_mode)
   
   # 定义全连接层
   def fc_layer(input_channels, out_channels, has_bias=True):
       return nn.Dense(input_channels, out_channels, has_bias=has_bias)
   
   
   class AlexNet(nn.Cell):
       """
       AlexNet 模型定义
   
       参数:
       - num_classes: 分类数量
       - channel: 输入通道数（图像的颜色通道数）
       - phase: 模型运行阶段（'train' 或 'test'）
       - include_top: 是否包含全连接层的顶部（最后的分类层）
       """
       def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
           super(AlexNet, self).__init__()
   
           # 卷积层
           self.conv1 = conv_layer(channel, 64, 11, stride=4, pad_mode="same")
           self.conv2 = conv_layer(64, 128, 5, pad_mode="same")
           self.conv3 = conv_layer(128, 192, 3, pad_mode="same")
           self.conv4 = conv_layer(192, 256, 3, pad_mode="same")
           self.conv5 = conv_layer(256, 256, 3, pad_mode="same")
   
           # 激活函数和池化层
           self.relu = nn.ReLU()
           self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
   
           # 如果包括顶部（全连接层）
           self.include_top = include_top
           if self.include_top:
               self.flatten = nn.Flatten()
               self.fc1 = fc_layer(256 * 28 * 28, 4096)
               self.fc2 = fc_layer(4096, 4096)
               self.fc3 = fc_layer(4096, num_classes)
   
           # 数学操作
           self.add = ops.Add()
           self.mul = ops.Mul()
   
       def construct(self, x):
           """定义前向传播过程"""
   
           x = self.conv1(x)
           x = self.add(x, 0.1)    # 偏置加法
           x = self.mul(x, 2.0)    # 乘法操作
           x = self.relu(x)    # ReLU激活函数
           x = ops.celu(x) 
           x = x + 2
   
           # 打印每层输出形状，调试时可使用
           print(f"After Conv1: {x.shape}")
   
           x = self.max_pool2d(x)    # Max pooling 操作
           print(f"After MaxPool: {x.shape}")    # 打印池化后的形状
   
           x = self.conv2(x)
           x = self.relu(x)
   
           x = self.conv3(x)
           x = self.relu(x)
   
           x = self.conv4(x)
           x = self.relu(x)
   
           x = self.conv5(x)
           x = self.relu(x)
   
           # 打印卷积层后的形状，调试时使用
           print(f"After Conv5: {x.shape}")
   
           # 可选的全连接层部分
           if self.include_top:
               x = self.flatten(x)
               x = self.fc1(x)
               x = self.fc2(x)
               x = self.fc3(x)
   
           return x
   
   # 前向函数
   def forward_fn(data, label):
       out = net(data)
       loss = criterion(out, label)
       return loss
   
   # 训练步骤
   def train_step(data, label):
       loss, grads = grad_fn(data, label)
       optimizer(grads)
       return loss
   
   # 测试模型
   if __name__ == "__main__":
       net = AlexNet()
       optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
       criterion = nn.MSELoss()
   
       grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
   
       # 生成数据和标签
       batch_size = 1
       num_classes = 10
       data = np.random.normal(1, 1, (batch_size, 3, 227, 227)).astype(np.float32)
       label = np.random.randint(0, num_classes, (batch_size,)).astype(np.float32)    # 注意此处类型应为float32
   
       # 转换为MindSpore张量
       data = Tensor(data)
       label = Tensor(label)
   
       steps = 5
       for i in range(steps):
           debugger.start(net)    # 启动调试器
           loss = train_step(data, label)    # 执行训练步骤
           print(f"Step {i}, Loss: {loss}")
           debugger.stop()    # 停止调试器
           debugger.step()    # 计数步数
   ```

3. 运行训练脚本。

   在命令行中执行如下命令：

   ```bash
   python alexnet_model.py
   ```

4. 查看采集结果。

   执行训练命令后，工具会将模型训练过程中的精度数据采集下来。

   日志中打印出现如下信息表示数据采集成功，即可手动停止模型训练查看采集数据。

   ```markdown
   ****************************************************************************
   *                        msprobe ends successfully.                        *
   ****************************************************************************
   ```

5. 数据分析。

   在`dump_path`参数指定的路径下（本例中为`./output`），会出现如下目录结构，后续精度数据分析操作可使用msProbe工具的精度预检和精度比对等功能，详细流程请参见《[MindStudio Probe](../../../README.md)》。

   ```ColdFusion
   output/
   └── step0
       └── proc{pid}
           ├── construct.json             # level为L0时，保存Cell的层级关系信息。当前场景为空
           ├── dump.json                  # 保存API前反向输入输出数据的统计量信息
           └── stack.json                 # 保存API的调用栈
       ......
   ```
