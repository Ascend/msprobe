# Precision Data Collection in MindSpore Dynamic Graph Mode

## Overview

This document describes how to use msProbe to collect precision data in MindSpore dynamic graph mode.

**Environment Setup**

Install msProbe by referring to [msProbe Installation Guide](../msprobe_install_guide.md).

## Procedure

1. Create a configuration file.

   Create a `config.json` file in the current directory and copy the configuration to the file.

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

   For details about the preceding parameters, see [Configuration File Introduction](./config_json_introduct.md).

2. Create a model script.

   Create a Python script file in the current directory, for example, `alexnet_model.py`, and copy the following code to the file.

   ```python
   import os
   import numpy as np
   import mindspore as ms
   from mindspore import nn, ops
   from mindspore import context, _device, set_deterministic
   from mindspore import Tensor
   from msprobe.mindspore import PrecisionDebugger, seed_all
   
   # Set a seed to ensure that the results can be reproduced.
   seed_all(seed=1234, mode=False, rm_dropout=True)
   
   # Configuration file path
   script_dir = os.path.dirname(os.path.abspath(__file__))
   config_path = os.path.join(script_dir, 'config.json')
   
   # Initialize the precision debugger.
   debugger = PrecisionDebugger(config_path=config_path)
   
   # Set the MindSpore context.
   context.set_context(mode=ms.PYNATIVE_MODE)
   
   context.set_context(device_target="Ascend", device_id=0)
   
   set_deterministic(True)
   print("Context set successfully. Please wait for the training task.")
   
   # Define a convolutional layer.
   def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid", has_bias=True):
       return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                        has_bias=has_bias, pad_mode=pad_mode)
   
   # Define a fully connected layer.
   def fc_layer(input_channels, out_channels, has_bias=True):
       return nn.Dense(input_channels, out_channels, has_bias=has_bias)
   
   
   class AlexNet(nn.Cell):
       """
       Define the AlexNet model.
   
       Parameters:
       - num_classes: number of classes
       - channel: number of input channels (number of color channels of an image)
       - phase: model running phase ('train' or 'test')
       - include_top: whether to include the top of the fully-connected layer (the last class layer)
       """
       def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
           super(AlexNet, self).__init__()
   
           # Convolutional layer
           self.conv1 = conv_layer(channel, 64, 11, stride=4, pad_mode="same")
           self.conv2 = conv_layer(64, 128, 5, pad_mode="same")
           self.conv3 = conv_layer(128, 192, 3, pad_mode="same")
           self.conv4 = conv_layer(192, 256, 3, pad_mode="same")
           self.conv5 = conv_layer(256, 256, 3, pad_mode="same")
   
           # Activation function and pooling layer
           self.relu = nn.ReLU()
           self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
   
           # If the top (fully-connected layer) is included:
           self.include_top = include_top
           if self.include_top:
               self.flatten = nn.Flatten()
               self.fc1 = fc_layer(256 * 28 * 28, 4096)
               self.fc2 = fc_layer(4096, 4096)
               self.fc3 = fc_layer(4096, num_classes)
   
           # Mathematical operations
           self.add = ops.Add()
           self.mul = ops.Mul()
   
       def construct(self, x):
           """Define the forward propagation process."""
   
           x = self.conv1(x)
           x = self.add(x, 0.1)    # Bias addition
           x = self.mul(x, 2.0)    # Multiplication
           x = self.relu(x)    # ReLU
           x = ops.celu(x) 
           x = x + 2
   
           # Print the output shape of each layer for debugging.
           print(f"After Conv1: {x.shape}")
   
           x = self.max_pool2d(x)   # Max pooling
           print(f"After MaxPool: {x.shape}")   # Print the shape after pooling.
   
           x = self.conv2(x)
           x = self.relu(x)
   
           x = self.conv3(x)
           x = self.relu(x)
   
           x = self.conv4(x)
           x = self.relu(x)
   
           x = self.conv5(x)
           x = self.relu(x)
   
           # Print the shape after the convolutional layer for debugging.
           print(f"After Conv5: {x.shape}")
   
           # Optional fully-connected layer
           if self.include_top:
               x = self.flatten(x)
               x = self.fc1(x)
               x = self.fc2(x)
               x = self.fc3(x)
   
           return x
   
   # Forward function
   def forward_fn(data, label):
       out = net(data)
       loss = criterion(out, label)
       return loss
   
   # Training procedure
   def train_step(data, label):
       loss, grads = grad_fn(data, label)
       optimizer(grads)
       return loss
   
   # Model testing
   if __name__ == "__main__":
       net = AlexNet()
       optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
       criterion = nn.MSELoss()
   
       grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
   
       # Data and label generation
       batch_size = 1
       num_classes = 10
       data = np.random.normal(1, 1, (batch_size, 3, 227, 227)).astype(np.float32)
       label = np.random.randint(0, num_classes, (batch_size,)).astype(np.float32)   # Note that the type must be float32.
   
       # Convert the data to a MindSpore tensor.
       data = Tensor(data)
       label = Tensor(label)
   
       steps = 5
       for i in range(steps):
           debugger.start(net)   # Start the debugger.
           loss = train_step(data, label)   # Execute training steps.
           print(f"Step {i}, Loss: {loss}")
           debugger.stop()   # Stop the debugger.
           debugger.step()   # Count the number of steps.
   ```

3. Run the training script.

   Run the following command to execute the training script:

   ```bash
   python alexnet_model.py
   ```

4. Check the collection result.

   After the training command is executed, the tool collects precision data during model training.

   If the following information is displayed in the log, the data is successfully collected. You can manually stop model training and view the collected data.

   ```markdown
   ****************************************************************************
   *                        msprobe ends successfully.                        *
   ****************************************************************************
   ```

5. Analyze the data.

   The following directory structure is generated in the path specified by `dump_path` (`./output` in this example). You can use the precision pre-check and comparison functions of msProbe to analyze precision. For details, see [MindStudio Probe](../../../README.md).

   ```ColdFusion
   output/
   └── step0
       └── proc{pid}
           ├── construct.json            # When the level is L0, the hierarchical relationship information of the cell is saved. This field is empty in the current scenario.
           ├── dump.json                # Forward/backward inputs/outputs of the API
           └── stack.json                 # API call stack
       ......
   ```
