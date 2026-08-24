# msProbe PyTorch Quick Start

<!-- md-trans-meta sourceCommit=d8769dcb657f88517a1877f5ff6691464e86ec45 translatedAt=2026-08-11T11:42:51.498Z pushedAt=2026-08-11T11:47:06.110Z -->

<br>

## 1. Overview

MindStudio Probe (msProbe) is an AI model precision debugging tool. This document uses ResNet-50 model training as an example to demonstrate the complete workflow of NPU/GPU data collection, precision comparison, and graph comparison in hierarchical visualization, helping you master the troubleshooting methods and analysis approaches for typical precision issues such as numerical overflow, loss anomalies, and model non-convergence.

**Experience Map (Core Operations Take Approximately 10 Minutes)**

| Step | Phase | Core Tool | Operation Time | Principle Learning |
| :---: | :--- | :--- | :---: | :---: |
| 1 | Environment preparation | CANN container | 5 min | 5 min |
| 2 | NPU data collection | PrecisionDebugger | 1 min | 10 min |
| 3 | GPU benchmark data collection | PrecisionDebugger | 0.5 min | 5 min |
| 4 | Precision comparison | msProbe compare | 1 min | 10 min |
| 5 | Visualized graph comparison | graph_visualize / TensorBoard | 2 min | 10 min |

> 👉 This tutorial is based on the PyTorch framework. If you need to use it in the MindSpore scenario, please refer to *[Quick Start of msProbe in the MindSpore Scenario](mindspore_quick_start.md)*.

## 2. Procedure

### 2.1 Environment Preparation (Required)

🛑 **This step is mandatory! Skipping it may cause subsequent operations to fail.**

The NPU-side operations in this tutorial are **only supported** in standardized CANN containers and are not supported for execution directly on bare metal, virtual machines, or other non-standard container environments.

#### 2.1.1 Prerequisites

Before starting, ensure that your server meets the following requirements:

| Item | Requirement | Verification Method |
| --- | --- | --- |
| **Hardware computing power** | The Linux server is equipped with at least one NPU, with drivers and firmware installed. | Execute `npu-smi info` and confirm that the NPU status is normal. |
| **Container runtime** | Docker is installed and running (recommended version ≥ 18.0). | Execute `docker ps`. No error indicates that the service is running normally. |
| **Script execution** | Python 3 is installed on the host. | Execute `python3 -V` on the host. Version information output indicates that it is installed. |
| **Network communication** | curl is installed (any version). | Execute `curl -V`. Version information output indicates that it is installed. |

> 👉 When these requirements are met and if the environment has public network access, all NPU-side commands in this chapter can be directly executed by **Copy/Paste** without manual typing or concatenation. This can avoid command execution failures caused by input errors.

#### 2.1.2 Host: Automatically Identify and Configure Image Environment Variables

Execute the following command on the host (this command sequentially reads the NPU PCI ID, matches the image version, and writes environment variables for subsequent processes):

```bash
source /dev/stdin <<< "$(dev_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2); case "$dev_id" in 'd500' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=310P";; 'd802' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=910B";; 'd803' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=A3";; 'd806' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-950-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=950";; * ) echo "unset MY_STUDY_VAR_CANN_IMAGE MY_CHIP_NAME; echo >&2; echo -e '\033[31m[FAIL] Get device ID: $dev_id. Learning is not supported in the current environment.\033[0m' >&2";; esac)"
[ -n "$MY_STUDY_VAR_CANN_IMAGE" ] && echo -e "\e[32m[PASS] Successfully identified chip [$MY_CHIP_NAME] and auto-selected image:\n    $MY_STUDY_VAR_CANN_IMAGE\e[0m"
```

> [!NOTE]
>
> **Command Principle**
>
> Obtain the NPU PCI ID via `lspci`, automatically match the CANN official image, and assign the image address to the environment variable `MY_STUDY_VAR_CANN_IMAGE` for subsequent use.
> All images are from the CANN official images published on Huawei Cloud AscendHub. For image details, please refer to [Official CANN Image Repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884).

If `[PASS]` is output after command execution, it indicates success; if `[FAIL]` is output, possible causes are as follows:

1. The hardware is not within the scope of this tutorial: this learning environment only supports Ascend 310P, A2, A3, and 950 products. Please switch to a compatible hardware environment and retry.
2. Underlying environment anomaly: `lspci` is not installed, or the current user cannot query the NPU PCI ID via `lspci -n -D`. Contact the environment administrator to verify the underlying environment.

#### 2.1.3 Host: Pull the Image

Execute on the host:

```bash
docker pull ${MY_STUDY_VAR_CANN_IMAGE}
```

If the pull fails due to being on a corporate intranet, please refer to the solution in [Section 3.1](#31-how-to-obtain-docker-images-in-an-isolated-intranet).

#### 2.1.4 Host: Download Container Startup Script

Execute on the host:

```bash
cd ~ && curl -fLO --retry 3 https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py && chmod +x ctr_in.py
```

If download fails due to network restrictions, please refer to the solution in <a href="#32-transfer-container-startup-script">Section 3.2</a>.

#### 2.1.5 Host: Start Container

On the host, execute the following command and confirm the container creation information according to the terminal prompt:

```bash
~/ctr_in.py ${MY_STUDY_VAR_CANN_IMAGE}
```

**Expected result**: The terminal displays a root shell prompt similar to the following, indicating that the container has been successfully started and entered:

```text
[root@xxxxxx ~]#
```

If an error is prompted or a container selection interface appears, go back to [Section 2.1.2](#212-host-automatically-identify-and-configure-image-environment-variables) and confirm that the command outputs `[PASS]`, then restart the container.

#### 2.1.6 Inside Container: Install Python Dependencies and msProbe

Execute the following command inside the container:

```bash
pip3 install networkx==3.6.1 pillow==12.2.0
pip3 install https://inst.obs.cn-north-4.myhuaweicloud.com/env/mirror/$(arch)/download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_$(arch).whl
pip3 install https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.7.1/torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_$(arch).whl
pip3 install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install -U mindstudio-probe
```

If the installation fails due to being on a corporate intranet, please refer to the solution in <a href="#33-offline-installation-of-python-dependencies">Section 3.3</a>.

#### 2.1.7 Inside Container: Verify Environment Installation Correctness

After installation, execute the environment check command:

```bash
python3 -c 'import torch, torch_npu; assert torch.npu.is_available(), "NPU is unavailable"; import msprobe; print("PyTorch:", torch.__version__)' && msprobe --help >/dev/null && tensorboard --help >/dev/null && echo -e "\e[32m[PASS] NPU environment, msProbe and TensorBoard check passed.\e[0m"
```

If `[PASS]` is displayed, it indicates that the NPU environment, Python dependencies, msProbe, and TensorBoard are all properly configured, and you can proceed to the next step.

### 2.2 Collect Data to Be Debugged in NPU Environment

#### 2.2.1 Prepare Collection Configuration

Execute the following command inside the container to write the collection configuration to `~/config.json`:

```bash
cat > ~/config.json << EOF
{
    "task": "statistics",
    "dump_path": "${HOME}/msprobe_dump_npu",
    "rank": [],
    "step": [0, 1],
    "level": "mix",
    "async_dump": false,
    "statistics": {
        "scope": [],
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
EOF
```

This configuration collects the forward and backward input/output statistics at the Module and API levels for the 0th and 1st training iterations. The collection results can be used for both precision comparison and hierarchical visualized graph comparison. Since `task` is set to `statistics`, only Tensor statistics are saved, not the complete Tensor data, which reduces disk usage.

#### 2.2.2 Prepare Model Training Code

Execute the following command inside the container to write the training code to `~/precision_sample.py`. The script uses fixed random data to train a ResNet-50 model and collects precision data through `PrecisionDebugger`. The model includes typical structures such as convolution, normalization, activation, residual connection, pooling, and fully connected layers:

```python
cat > ~/precision_sample.py << 'EOF'
import os, argparse, torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets, torchvision.models as models, torchvision.transforms as transforms
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all(seed=1234, mode=True)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')
torch.cuda.set_device(args.gpu)
model = models.resnet50().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
train_loader = DataLoader(datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor()), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor()), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
debugger = PrecisionDebugger(config_path=os.path.expanduser("~/config.json"))

global_step = 0
total_epochs = 2
total_steps = total_epochs * len(train_loader)

for epoch in range(total_epochs):
    model.train()
    for i, (images, target) in enumerate(train_loader):
        debugger.start(model)
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        loss = criterion(model(images), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        debugger.stop()
        
        if global_step % 10 == 0:
            print(f"Current Step: {global_step} (Progress: {global_step / total_steps:.2%})\tLoss: {loss.item():.4e}")

        debugger.step()
        global_step += 1

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            correct += model(images).argmax(dim=1).eq(target).sum().item()
            total += target.size(0)
    print(f" * Finished Epoch Pool - Evaluation Acc@1: {100.0 * correct / total:.3f}%")
    scheduler.step()
EOF
```

#### 2.2.3 Start Training and Data Collection

Execute the following command inside the container:

```bash
python3 ${HOME}/precision_sample.py --gpu 0
```

> Card 0 is used by default. If this card is unavailable or you need to specify another card, replace the number in `--gpu 0` with the target card ID.

When the log outputs the following information, it indicates that the precision data collection for step0/step1 is complete. At this point, subsequent training iterations (step 2, 3, 4, etc.) will continue to execute. You can safely terminate the process by pressing `Ctrl + C` to save time. Early termination will not affect the integrity of the collected step0/step1 data:

```text
2026-07-15 02:08:30 (2596) [INFO] dump.json is at /root/msprobe_dump_npu/step1.
2026-07-15 02:08:31 (2596) [INFO] ****************************************************************************
2026-07-15 02:08:31 (2596) [INFO] *                        msprobe ends successfully.                        *
2026-07-15 02:08:31 (2596) [INFO] ****************************************************************************
```

> [!NOTE]
> 
> **Log Output and Manual Termination Principle**  
> Training begins immediately after the script starts. Since `"step": [0, 1]` is configured in `config.json`, msProbe only triggers data collection and outputs related logs at the 0th and 1st training iterations. Starting from the 2nd iteration, msProbe stops collecting data, and the terminal only outputs logs from the training script itself (e.g., `Current Step: 10 (Progress: 0.01%)`). At this point, the precision data for step0 and step1 has been completely written to disk, and the training process can be safely terminated.

#### 2.2.4 View Collection Results

Execute the following command to automatically locate the `dump.json` generated at training iteration 0 (`step0`) and view the directory structure:

```bash
NPU_DUMP_JSON=$(find "${HOME}/msprobe_dump_npu/step0" -type f -name dump.json | head -n 1)
echo "${NPU_DUMP_JSON}"
tree -L 3 "${HOME}/msprobe_dump_npu"
```

If the `dump.json` path is successfully output, it indicates that data collection is normal.

In single-device training, precision data is typically saved in the `proc{pid}` directory; in multi-device training, it is saved in the `rank{id}` directory. The common structure is as follows:

```text
msprobe_dump_npu
├── step0
│   └── proc{pid}
│       ├── construct.json
│       ├── dump.json
│       └── stack.json
└── step1
    └── proc{pid}
        ├── construct.json
        ├── dump.json
        └── stack.json
```

| File | Description |
| :--- | :--- |
| `construct.json` | Records Module hierarchy information |
| `dump.json` | Contains input/output statistics and overflow information of Modules and APIs during forward and backward processes, serving as the core input for subsequent precision comparison |
| `stack.json` | Records API call stack information, used to trace back from suspicious APIs to the training code |

### 2.3 Collect Benchmark Data in GPU Environment

This quick start is intended for experiencing the core functionality of msProbe. Collecting GPU data on your own offers limited value for understanding the tool, so it is recommended to directly use the preset sample data:

```bash
cd ~
git clone --depth 1 --single-branch https://gitcode.com/Ascend/msprobe.git -b 26.1.0
cp -rf ~/msprobe/examples/quick_start/gpu_dump ~/msprobe_dump_gpu
```

> [!NOTE]
> 
> The preset GPU data already covers typical precision issues, which can significantly shorten your experience time and allow you to focus on the core analysis capabilities of msProbe rather than environment setup.
> If you wish to experience the GPU data collection process, please refer to the instructions in [Chapter 4](#4-appendix-b-train-the-model-and-collecting-data-in-a-gpu-environment) for self-guided exploration.

### 2.4 NPU and GPU Precision Comparison

#### 2.4.1 Prepare Comparison Data

Execute the following command inside the NPU container to relocate the dual-end data paths:

```bash
NPU_DUMP_JSON=$(find "${HOME}/msprobe_dump_npu/step0" -type f -name dump.json | head -n 1)
GPU_DUMP_JSON=$(find "${HOME}/msprobe_dump_gpu/step0" -type f -name dump.json | head -n 1)
echo "NPU: ${NPU_DUMP_JSON}"
echo "GPU: ${GPU_DUMP_JSON}"
```

Confirm that both variables output the actual `dump.json` path before proceeding with the comparison.

#### 2.4.2 Execute Precision Comparison

Execute the following command inside the NPU container:

```bash
msprobe compare -tp "${NPU_DUMP_JSON}" -gp "${GPU_DUMP_JSON}" -o "${HOME}/accuracy_compare"
```

If the following information is output, the comparison is successful:

```text
************************************************************************************
*                        msprobe compare ends successfully.                        *
************************************************************************************
```

#### 2.4.3 View Precision Comparison Results

Execute the following command to view the generated result file:

```bash
tree -L 1 "${HOME}/accuracy_compare"
```

In a single-device scenario, a `compare_result_{timestamp}.csv` (or xlsx format) file is generated, which lists the APIs participating in the comparison, data types, tensor shapes, statistic errors, comparison conclusions, and error messages:

![Precision Comparison Result Diagram](../figures/compare_result_quick_start.png)
<div style="text-align: center;">
<strong>Figure 1</strong> Example of Precision Comparison Result File Content
</div>

When reviewing the results, it is recommended to analyze them in the following order:

1. **Filter anomalies**: Filter APIs that failed based on the `Result` column.
2. **Troubleshoot errors**: Check `Err_Message` to determine whether there are issues such as unmatched APIs, inconsistent data types, or shape mismatches.
3. **Compare statistics**: For APIs that are matched but have significant precision differences, compare statistics such as Max, Min, Mean, and L2 Norm, as well as relative errors.
4. **Trace back to code**: Use `NPU_Stack_Info` or the `stack.json` file on the NPU side to locate the training code corresponding to the suspicious API.

For more metric definitions and result interpretation methods, see *[Precision Comparison Result Analysis](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md#precision-comparison-result-analysis)*.

### 2.5 Graph Comparison in Hierarchical Visualization

This feature reconstructs the Module and API hierarchy of both models and maps precision differences onto graph nodes, making it suitable for locating suspicious nodes layer by layer from the overall model structure.

#### 2.5.1 Generate Dual-Graph Comparison Files

Execute the following command inside the NPU container:

```bash
msprobe graph_visualize -tp "${HOME}/msprobe_dump_npu" -gp "${HOME}/msprobe_dump_gpu" -o "${HOME}/graph_visualize_output"
```

After execution, check the output results:

```bash
tree -L 1 "${HOME}/graph_visualize_output"
```

The following files will be generated in the output directory:

```text
graph_visualize_output
└── compare_{timestamp}.vis.db
```

If the model structure is empty, verify that the `level` in both NPU and GPU collection configurations is either `mix` or `L0`, and check whether the `construct.json` file content on both sides is empty.

#### 2.5.2 Start TensorBoard

Execute the following command inside the NPU container:

```bash
tensorboard --logdir "${HOME}/graph_visualize_output" --bind_all
```

The terminal will output an access address similar to the following (the hostname and port are subject to the actual log):

```text
TensorBoard 2.x.x at http://hostname:6006/ (Press CTRL+C to quit)
```

Access `http://<Server IP>:6006/` in a browser. If direct access is blocked by firewall restrictions, you can access it via VS Code port forwarding or SSH port forwarding. For details, please refer to [Section 5.3](#53-how-to-access-tensorboard-when-the-port-is-blocked-by-a-firewall).

#### 2.5.3 View Visualized Comparison Results

After successfully opening TensorBoard, you can see the following dual-graph comparison results:

![Hierarchical visualized graph comparison results](../figures/vis_quick_start.png)
<div style="text-align: center;">
<strong>Figure 2</strong> NPU and GPU hierarchical visualized graph comparison
</div>

It is recommended to analyze in the following order:

1. **Confirm data**: In the data selection area, verify that the selected NPU data, GPU data, training step, and process correspond to each other;
2. **Expand level by level**: Expand modules level by level from the top of the model, prioritizing nodes with darker colors or those marked as having suspicious accuracy.
3. **Search and locate**: Use the node search function to quickly locate suspicious APIs identified in the Precision Comparison result file.
4. **Analyze deviations**: After selecting a node, compare the statistics, precision metrics, and call stacks on both sides to determine the first significant deviation position.

For more methods on node matching, precision filtering, overflow detection, and cross-framework comparison, please refer to *[PyTorch Scenario Hierarchical Visualization Graph Comparison](../user_guide/accuracy_compare/pytorch_visualization_instruct.md)*.

### 2.6 Next Steps

Congratulations on completing the msProbe quick start experience! You have now mastered the basic usage of msProbe. For a deeper understanding of its features, please refer to:

- *[Pre-Training Configuration Check](../user_guide/config_check_instruct.md)*
- *[Training Status Monitoring](../user_guide/monitor_instruct.md)*
- *[Data Collection in PyTorch](../user_guide/dump/pytorch_data_dump_instruct.md)*
- [*Precision Comparison in PyTorch*](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [*Graph Comparison in Hierarchical Visualization in PyTorch*](../user_guide/accuracy_compare/pytorch_visualization_instruct.md)

## 3. Appendix A: Solution for Intranet Environments Without Public Network Access

### 3.1 How to Obtain Docker Images in an Isolated Intranet

**Solution 1: Configure Docker Proxy for Direct Image Pull**

This solution applies to most Linux distributions with Docker version ≥ 18.0 (compatibility is not guaranteed for all scenarios). If exceptions occur, adjust the configuration according to the actual situation.

Edit the Docker service proxy configuration file `/etc/systemd/system/docker.service.d/http-proxy.conf`. The following is an example of its content (replace the username, password, proxy address, and port according to your actual environment):

```text
[Service]
Environment="HTTP_PROXY=http://username:password@proxy.example.com:8080"
Environment="HTTPS_PROXY=http://username:password@proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

After saving, reload and restart the Docker service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

You can then execute `docker pull` normally.

**Option 2: Import the CANN Image Offline**

If the proxy solution is not feasible, first execute [Section 2.1.2](#212-host-automatically-identify-and-configure-image-environment-variables) on the intranet NPU server and record the complete value of `MY_STUDY_VAR_CANN_IMAGE`. Then log in to a relay machine that has public network access and the same CPU architecture, replace the value of `CANN_IMAGE` below with the image address you just recorded, and execute the command:

```bash
CANN_IMAGE='Complete_image_address'
docker pull "${CANN_IMAGE}"
docker save -o cann.tar "${CANN_IMAGE}"
```

After transferring `cann.tar` to the intranet server via a USB drive or other means, execute the following command on the intranet server to load it:

```bash
docker load -i cann.tar
docker images | grep cann
```

After loading is complete, proceed to <a href="#32-transfer-container-startup-script">Section 3.2</a>, then return to <a href="#215-host-start-container">Section 2.1.5</a> to start the container. If you have switched the host shell, re-execute the command in Section 2.1.2 to restore the image environment variable.

### 3.2 Transfer Container Startup Script

Enter the following link in a browser that can access the current web page, download the `ctr_in.py` script file, and manually copy it to the `~/` directory on the intranet server:

```text
https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py
```

After copying, execute the following on the host of the intranet server:

```bash
cd ~
chmod +x ctr_in.py
ls -l ctr_in.py
```

After confirming that `ctr_in.py` exists and has execution permissions, return to <a href="#215-host-start-container">Section 2.1.5</a> to start the container.

### 3.3 Offline Installation of Python Dependencies

Prioritize using an internal network pip source to install dependencies. If no internal network software source is available, download the required installation packages as follows in a transfer environment that has public network access and shares the same CPU architecture and Python version as the internal network NPU server:

```bash
mkdir -p offline_wheels
python3 -m pip download xxx --dest offline_wheels
```

Transfer the `offline_wheels` directory to the internal network server, copy it to the user home directory inside the container, and then execute inside the container:

```bash
pip3 install --no-index --find-links="${HOME}/offline_wheels" xxx
```

After installation is complete, return to [Section 2.1.7](#217-inside-container-verify-environment-installation-correctness) to execute the verification command. There is no need to execute the online installation command again.

## 4. Appendix B: Train the Model and Collecting Data in a GPU Environment

If you need to collect data on your own, ensure that the GPU environment has PyTorch installed with the same version as the NPU environment, and install msProbe:

```bash
pip3 install -U mindstudio-probe
```

Execute the following command in the GPU environment to create the collection configuration:

```bash
cat > ~/config.json << EOF
{
    "task": "statistics",
    "dump_path": "${HOME}/msprobe_dump_gpu",
    "rank": [],
    "step": [0, 1],
    "level": "mix",
    "async_dump": false,
    "statistics": {
        "scope": [],
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
EOF
```

Refer to [Section 2.2.2](#212-host-automatically-identify-and-configure-image-environment-variables) to write the training code into `~/precision_sample.py`, then start training and data collection:

```bash
python3 ${HOME}/precision_sample.py --gpu 0
```

After collection is complete, execute the following command to package the data for transfer to the NPU container:

```bash
tar -czvf ${HOME}/msprobe_dump_gpu.tar.gz ${HOME}/msprobe_dump_gpu
```

## 5. FAQs

### 5.1 How to Re-enter the Container After Exiting?

On the host, choose either of the following methods to re-enter the container:

**Method 1 (Recommended): Use the Container Startup Script**

```bash
~/ctr_in.py
```

Select the target container as prompted. If there is only one accessible container, the script will automatically enter that container.

**Method 2: Use the Docker Native Command**

```bash
docker exec -it alice_YYMMDD_HHMMSS bash
```

Please replace `alice_YYMMDD_HHMMSS` with the actual container name. You can first execute `docker ps` to view the running containers and their names.

### 5.2 How to Handle "permission denied" When Executing Docker Commands?

The current user may not have been added to the Docker user group. Execute the following on the host with root privileges:

```bash
sudo usermod -aG docker "${USER}"
```

After execution, log out of the current user session and log in again, or run the following command to apply the user group change immediately:

```bash
newgrp docker
```

After completion, run `docker ps` to verify that Docker commands work properly. It is not recommended to perform daily operations as the root user. The Docker user group has elevated system privileges; add only trusted users to this group.

### 5.3 How to Access TensorBoard When the Port Is Blocked by a Firewall?

When the server firewall restricts direct access to the TensorBoard port, you can use **VS Code port forwarding** or **SSH local port forwarding**. Both methods require only SSH port connectivity and do not require opening port `6006` separately.

#### Method 1: VS Code Port Forwarding (Recommended)

If you have already connected to the server using VS Code Remote-SSH, you can quickly complete port mapping through the graphical interface:

1. Stop the TensorBoard process started above, and instead start TensorBoard in the VS Code remote terminal:

    ```bash
    tensorboard --logdir "${HOME}/graph_visualize_output" --bind_all
    ```

2. VS Code usually automatically detects the port listening information in the terminal and displays a prompt in the lower-right corner. Click the link directly to access it.
3. If no prompt appears, you can also click the **"Ports"** tab in the bottom panel of VS Code, select **"Forward a Port"**, enter `6006`, and confirm.
4. After the forwarding is successful, click the **"Local Address"** link generated in the list (such as `http://localhost:6006`) to access it directly in your local browser.

#### Method 2: SSH Command-Line Port Forwarding

If VS Code is not used, you can establish a tunnel using native SSH commands:

1. Execute the following command in a local terminal (Windows PowerShell/CMD/Linux/macOS), and enter the password when prompted:

    ```bash
    ssh -L 6006:localhost:6006 your_username@192.168.1.1
    ```

    > Please replace `your_username` and `192.168.1.1` with the actual username and server IP, and keep this SSH session connected.

2. Open the following address in a local browser:

    ```text
    http://localhost:6006/
    ```
