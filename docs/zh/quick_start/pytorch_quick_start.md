# msProbe PyTorch 场景快速入门

<br>

## 1. 概述

msProbe（MindStudio Probe）是 AI 模型精度调试工具。本文以 ResNet-50 模型训练为例，演示 NPU/GPU 数据采集、精度比对及分级可视化构图比对的完整流程，帮助您掌握数值溢出、Loss 异常、模型不收敛等典型精度问题的排查方法与分析思路。

**体验地图（核心操作约需 10 分钟）**

| 步骤 | 环节 | 核心工具 | 操作耗时 | 原理学习 |
| :---: | :--- | :--- | :---: | :---: |
| 1 | 环境准备 | CANN 容器 | 5 min | 5 min |
| 2 | NPU 数据采集 | PrecisionDebugger | 1 min | 10 min |
| 3 | GPU 标杆采集 | PrecisionDebugger | 0.5 min | 5 min |
| 4 | 精度比对 | msProbe compare | 1 min | 10 min |
| 5 | 可视化构图比对 | graph_visualize / TensorBoard | 2 min | 10 min |

> 👉 本教程基于 PyTorch 框架。如需在 MindSpore 场景下使用，请参阅《[MindSpore 场景精度调试工具快速入门](mindspore_quick_start.md)》。

## 2. 操作步骤

### 2.1 环境准备（必做）

🛑 **本节为强制前置步骤！跳过本节可能导致后续多项操作失败。**

本教程的 NPU 侧操作**仅支持**在标准化 CANN 容器中执行，不支持直接在裸机、虚拟机或其他非标准容器环境中执行。

#### 2.1.1 前置条件

开始前，请确认服务器满足以下要求：

| 项目 | 要求 | 验证方法 |
| --- | --- | --- |
| **硬件算力** | Linux 服务器配备至少 1 张 NPU 卡，驱动与固件已安装 | 执行 `npu-smi info`，确认 NPU 卡状态正常 |
| **容器运行** | 已安装并运行 Docker（建议版本 ≥ 18.0） | 执行 `docker ps`，无报错即表示服务正常启动 |
| **脚本执行** | 宿主机已安装 Python 3 | 在宿主机执行 `python3 -V`，有版本信息输出即表示已安装 |
| **网络通信** | 已安装 curl（任意版本） | 执行 `curl -V`，有版本信息输出即表示已安装 |

> 👉 确认前置条件满足后，若环境具备公网访问能力，本章 NPU 侧命令可全程直接 **Copy/Paste** 执行，无需手动输入或拼接，以避免因输入错误导致命令执行失败。

#### 2.1.2 宿主机：自动识别并配置镜像环境变量

在宿主机执行以下命令（该命令依次完成：读取 NPU PCI ID，匹配镜像版本，写入环境变量供后续流程使用）：

```bash
source /dev/stdin <<< "$(dev_id=$(lspci -n -D | grep -o '19e5:d[0-9a-f]\{3\}' | head -n1 | cut -d: -f2); case "$dev_id" in 'd500' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-310p-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=310P";; 'd802' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=910B";; 'd803' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=A3";; 'd806' ) echo "export MY_STUDY_VAR_CANN_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-950-openeuler24.03-py3.11-devel; export MY_CHIP_NAME=950";; * ) echo "unset MY_STUDY_VAR_CANN_IMAGE MY_CHIP_NAME; echo >&2; echo -e '\033[31m[FAIL] Get device ID: $dev_id. Learning is not supported in the current environment.\033[0m' >&2";; esac)"
[ -n "$MY_STUDY_VAR_CANN_IMAGE" ] && echo -e "\e[32m[PASS] Successfully identified chip [$MY_CHIP_NAME] and auto-selected image:\n    $MY_STUDY_VAR_CANN_IMAGE\e[0m"
```

> [!NOTE]说明
>
> **命令原理**
>
> 通过 `lspci` 获取 NPU 的 PCI ID，自动匹配 CANN 官方镜像，并将镜像地址赋给环境变量 `MY_STUDY_VAR_CANN_IMAGE`，供后续使用。  
> 所有镜像均来自华为云 AscendHub 上发布的 CANN 官方镜像。如需了解镜像详情，请参阅 [CANN 官方镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)。

若命令执行后输出 `[PASS]`，则表示执行成功；若输出 `[FAIL]`，可能原因如下：

1. 硬件不在本教程支持范围内：本学习环境仅支持昇腾 310P、A2、A3 及 950 系列产品，请切换至兼容的硬件环境后重试；
2. 底层环境异常：未安装 `lspci`，或当前用户无法通过 `lspci -n -D` 查询 NPU PCI ID，请联系环境管理员确认底层环境。

#### 2.1.3 宿主机：拉取镜像

在宿主机执行：

```bash
docker pull ${MY_STUDY_VAR_CANN_IMAGE}
```

若因处于企业内网导致拉取失败，请参考 [第 3.1 节](#31-docker-镜像在隔离内网的获取方法) 的解决方案。

#### 2.1.4 宿主机：下载容器启动脚本

在宿主机执行：

```bash
cd ~ && curl -fLO --retry 3 https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py && chmod +x ctr_in.py
```

若因网络限制无法下载，请参考 [第 3.2 节](#32-传输容器启动脚本) 的解决方案。

#### 2.1.5 宿主机：启动容器

在宿主机执行以下命令，并根据终端提示确认容器创建信息：

```bash
~/ctr_in.py ${MY_STUDY_VAR_CANN_IMAGE}
```

**预期结果**：终端显示类似以下的 root Shell 提示符，表示容器已成功启动并进入容器：

```text
[root@xxxxxx ~]#
```

若提示错误或出现容器选择界面，请返回 [第 2.1.2 节](#212-宿主机自动识别并配置镜像环境变量)，确认命令输出 `[PASS]`，再重新启动容器。

#### 2.1.6 容器内：安装 Python 依赖和 msProbe

在容器内执行以下命令：

```bash
pip3 install networkx==3.6.1 pillow==12.2.0
pip3 install https://inst.obs.cn-north-4.myhuaweicloud.com/env/mirror/$(arch)/download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_$(arch).whl
pip3 install https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.7.1/torch_npu-2.7.1.post4-cp311-cp311-manylinux_2_28_$(arch).whl
pip3 install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install -U mindstudio-probe
```

若因处于企业内网导致安装失败，请参考 [第 3.3 节](#33-离线安装-python-依赖) 的解决方案。

#### 2.1.7 容器内：检查环境安装正确性

安装完成后执行环境检查命令：

```bash
python3 -c 'import torch, torch_npu; assert torch.npu.is_available(), "NPU is unavailable"; import msprobe; print("PyTorch:", torch.__version__)' && msprobe --help >/dev/null && tensorboard --help >/dev/null && echo -e "\e[32m[PASS] NPU environment, msProbe and TensorBoard check passed.\e[0m"
```

若显示 `[PASS]`，表示 NPU 环境、Python 依赖、msProbe 和 TensorBoard 均已正常配置，可以继续进行下一步操作。

### 2.2 在 NPU 环境采集待调试数据

#### 2.2.1 准备采集配置

在容器内执行以下命令，将采集配置写入 `~/config.json`：

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

本配置采集第 0、1 两个训练迭代中 Module 和 API 层级的前向、反向输入输出统计量。采集结果可同时用于精度比对和分级可视化构图比对。由于 `task` 设置为 `statistics`，仅保存 Tensor 统计量，不保存完整的 Tensor 数据，可降低磁盘占用。

#### 2.2.2 准备模型训练代码

在容器内执行以下命令，将训练代码写入 `~/precision_sample.py`。脚本使用固定随机数据训练 ResNet-50 模型，并通过 `PrecisionDebugger` 采集精度数据。该模型包含卷积、归一化、激活、残差连接、池化和全连接等典型结构：

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

#### 2.2.3 启动训练和采集

在容器内执行以下命令：

```bash
python3 ${HOME}/precision_sample.py --gpu 0
```

> 默认使用 0 号卡。若该卡不可用或需指定其他卡，请将 `--gpu 0` 中的数字替换为目标卡 ID。

当日志输出如下信息时，表明 step0/step1 精度数据采集已完成。此时后续训练迭代（step 2、3、4 等）仍会继续执行，可按 `Ctrl + C` 安全终止进程以节省时间，提前终止不会影响已采集的 step0/step1 数据完整性：

```text
2026-07-15 02:08:30 (2596) [INFO] dump.json is at /root/msprobe_dump_npu/step1.
2026-07-15 02:08:31 (2596) [INFO] ****************************************************************************
2026-07-15 02:08:31 (2596) [INFO] *                        msprobe ends successfully.                        *
2026-07-15 02:08:31 (2596) [INFO] ****************************************************************************
```

> [!NOTE]说明
> 
> **日志输出与手动终止原理**  
> 脚本启动后即开始训练，由于 `config.json` 中配置 `"step": [0, 1]`，msProbe 仅在第 0、1 个训练迭代触发采集并输出相关日志；从第 2 个迭代起，msProbe 停止采集，终端仅输出训练脚本自身的日志（如 `Current Step: 10 (Progress: 0.01%)`）。此时 step0 与 step1 的精度数据已完整落盘，可安全终止训练进程。

#### 2.2.4 查看采集结果

运行以下命令，自动定位第 0 个训练迭代（`step0`）生成的 `dump.json` 并查看目录结构：

```bash
NPU_DUMP_JSON=$(find "${HOME}/msprobe_dump_npu/step0" -type f -name dump.json | head -n 1)
echo "${NPU_DUMP_JSON}"
tree -L 3 "${HOME}/msprobe_dump_npu"
```

若成功输出 `dump.json` 路径，则表明数据采集正常。

单卡训练中，精度数据通常保存在 `proc{pid}` 目录；多卡训练则保存在 `rank{id}` 目录。常见结构如下：

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

| 文件 | 说明 |
| :--- | :--- |
| `construct.json` | 记录 Module 层级关系信息 |
| `dump.json` | 包含 Module 和 API 在前向、反向过程中的输入输出统计量及溢出信息，是后续精度比对的核心输入 |
| `stack.json` | 记录 API 调用栈信息，用于从可疑 API 回溯至训练代码 |

### 2.3 在 GPU 环境采集标杆数据

本快速入门旨在体验 msProbe 核心功能，自行采集 GPU 数据对理解工具价值有限，建议直接使用预置示例数据：

```bash
cd ~
git clone --depth 1 --single-branch https://gitcode.com/Ascend/msprobe.git
cp -rf ~/msprobe/examples/quick_start/gpu_dump ~/msprobe_dump_gpu
```

> [!NOTE]说明
> 
> 预置 GPU 数据已覆盖典型精度问题特征，可将您的体验时间大幅缩短，聚焦于 msProbe 核心分析能力而非环境搭建。  
> 如您想体验 GPU 采集数据过程，请参考 [第 4 章](#4-附录-b在-gpu-环境下训练模型并采集数据) 中的提示自主探索操作。

### 2.4 NPU 与 GPU 精度比对

#### 2.4.1 准备比对数据

在 NPU 容器内执行以下命令，重新定位双端数据路径：

```bash
NPU_DUMP_JSON=$(find "${HOME}/msprobe_dump_npu/step0" -type f -name dump.json | head -n 1)
GPU_DUMP_JSON=$(find "${HOME}/msprobe_dump_gpu/step0" -type f -name dump.json | head -n 1)
echo "NPU: ${NPU_DUMP_JSON}"
echo "GPU: ${GPU_DUMP_JSON}"
```

确认两个变量均输出实际的 `dump.json` 路径后再继续执行比对。

#### 2.4.2 执行精度比对

在 NPU 容器内执行以下命令：

```bash
msprobe compare -tp "${NPU_DUMP_JSON}" -gp "${GPU_DUMP_JSON}" -o "${HOME}/accuracy_compare"
```

若输出如下信息，则表明比对成功：

```text
************************************************************************************
*                        msprobe compare ends successfully.                        *
************************************************************************************
```

#### 2.4.3 查看精度比对结果

执行以下命令查看生成的结果文件：

```bash
tree -L 1 "${HOME}/accuracy_compare"
```

单卡场景会生成 `compare_result_{timestamp}.csv`（或 xlsx 格式），该文件列出参与比对的 API、数据类型、Tensor 形状、统计量误差、比对结论和错误信息等：

![精度比对结果示意图](../figures/compare_result_quick_start.png)
<div style="text-align: center;">
<strong>图 1</strong> 精度比对结果文件内容示例
</div>

查看结果时，建议按以下顺序分析：

1. **筛选异常**：根据 `Result` 列筛选未通过的 API；
2. **排查错误**：查看 `Err_Message`，判断是否存在 API 未匹配、数据类型或形状不一致等问题；
3. **比对统计**：针对已匹配但精度差异较大的 API，对比 Max、Min、Mean、L2 Norm 等统计量及相对误差；
4. **回溯代码**：结合 `NPU_Stack_Info` 或 NPU 侧 `stack.json`，定位可疑 API 对应的训练代码。

更多指标定义和结果解读方法，请参见《[精度比对结果分析](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md#精度比对结果分析)》。

### 2.5 分级可视化构图比对

分级可视化构图比对会还原两侧模型的 Module 和 API 层级结构，并将精度差异映射到图节点上，适合从整体模型结构逐层定位可疑节点。

#### 2.5.1 生成双图比对文件

在 NPU 容器内执行以下命令：

```bash
msprobe graph_visualize -tp "${HOME}/msprobe_dump_npu" -gp "${HOME}/msprobe_dump_gpu" -o "${HOME}/graph_visualize_output"
```

执行完成后查看输出结果：

```bash
tree -L 1 "${HOME}/graph_visualize_output"
```

输出目录中将生成如下文件：

```text
graph_visualize_output
└── compare_{timestamp}.vis.db
```

若提示模型结构为空，请确认 NPU 和 GPU 采集配置中的 `level` 均为 `mix` 或者均为 `L0`，并检查两侧 `construct.json` 文件内容是否为空。

#### 2.5.2 启动 TensorBoard

在 NPU 容器内执行以下命令：

```bash
tensorboard --logdir "${HOME}/graph_visualize_output" --bind_all
```

终端将输出类似如下访问地址（主机名和端口以实际日志为准）：

```text
TensorBoard 2.x.x at http://hostname:6006/ (Press CTRL+C to quit)
```

在浏览器中访问 `http://服务器IP:6006/`。若因防火墙限制导致无法直接访问，可通过 VS Code 端口转发或 SSH 端口转发访问，具体方法请参考 [第 5.3 节](#53-tensorboard-端口被防火墙拦截时如何访问)。

#### 2.5.3 查看可视化比对结果

成功打开 TensorBoard 后，可看到如下双图比对结果：

![分级可视化构图比对结果](../figures/vis_quick_start.png)
<div style="text-align: center;">
<strong>图 2</strong> NPU 与 GPU 分级可视化构图比对
</div>

建议按以下顺序分析：

1. **确认数据**：在数据选择区确认所选 NPU 数据、GPU 数据、训练迭代（Step）及进程相互对应；
2. **逐级展开**：从模型顶层逐级展开 Module，优先关注颜色较深或标记为精度可疑的节点；
3. **搜索定位**：利用节点搜索功能，快速定位精度比对结果文件中发现的可疑 API；
4. **分析偏差**：选中节点后对比两侧统计量、精度指标及调用栈，判断首个显著偏差位置。

更多节点匹配、精度筛选、溢出检测和跨套件比对方法，请参见《[PyTorch 场景分级可视化构图比对](../user_guide/accuracy_compare/pytorch_visualization_instruct.md)》。

### 2.6 后续进阶路径

恭喜您完成 msProbe 快速入门体验，您已掌握 msProbe 的基础使用方法。如需深入了解其功能用法，请参考：

- 《[训练前配置检查](../user_guide/config_check_instruct.md)》
- 《[训练状态监测](../user_guide/monitor_instruct.md)》
- 《[PyTorch 场景数据采集](../user_guide/dump/pytorch_data_dump_instruct.md)》
- 《[PyTorch 场景精度比对](../user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)》
- 《[PyTorch 场景分级可视化构图比对](../user_guide/accuracy_compare/pytorch_visualization_instruct.md)》

## 3. 附录 A：内网环境无公网访问权限的应对方案

### 3.1 Docker 镜像在隔离内网的获取方法

**方案一：配置 Docker 代理直接拉取**

适用于大多数 Linux 发行版且 Docker 版本 ≥ 18.0 的环境（不保证所有场景兼容）。若遇异常，请结合实际情况调整。

编辑 Docker 服务代理配置文件 `/etc/systemd/system/docker.service.d/http-proxy.conf`，内容示例如下（请根据实际环境替换用户名、密码、代理地址及端口）：

```text
[Service]
Environment="HTTP_PROXY=http://username:password@proxy.example.com:8080"
Environment="HTTPS_PROXY=http://username:password@proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"
```

保存后重载并重启 Docker 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

随后即可正常执行 `docker pull`。

**方案二：离线导入 CANN 镜像**

如果代理方案不可行，请先在内网 NPU 服务器上执行 [第 2.1.2 节](#212-宿主机自动识别并配置镜像环境变量)，并记录 `MY_STUDY_VAR_CANN_IMAGE` 的完整值。然后登录一台具备公网访问能力且 CPU 架构相同的中转机，将下方 `CANN_IMAGE` 的值替换为刚才记录的镜像地址并执行命令：

```bash
CANN_IMAGE='完整镜像地址'
docker pull "${CANN_IMAGE}"
docker save -o cann.tar "${CANN_IMAGE}"
```

将 `cann.tar` 通过 U 盘等方式传输至内网服务器后，在内网服务器执行以下命令加载：

```bash
docker load -i cann.tar
docker images | grep cann
```

加载完成后，继续完成 [第 3.2 节](#32-传输容器启动脚本)，再返回 [第 2.1.5 节](#215-宿主机启动容器) 启动容器。如果已切换宿主机 Shell，请重新执行第 2.1.2 节中的命令以恢复镜像环境变量。

### 3.2 传输容器启动脚本

在可访问当前网页的浏览器中输入如下链接，下载 `ctr_in.py` 脚本文件，并将其手动复制至内网服务器的 `~/` 目录：

```text
https://inst.obs.cn-north-4.myhuaweicloud.com/env/ctr_in.py
```

复制完成后，在内网服务器的宿主机上执行：

```bash
cd ~
chmod +x ctr_in.py
ls -l ctr_in.py
```

确认 `ctr_in.py` 存在且具有执行权限后，返回 [第 2.1.5 节](#215-宿主机启动容器) 启动容器。

### 3.3 离线安装 Python 依赖

优先使用内网 pip 源安装依赖。若没有可用的内网软件源，请在具备公网访问能力、与内网 NPU 服务器的 CPU 架构和 Python 版本均相同的中转环境中，按以下方式下载所需安装包：

```bash
mkdir -p offline_wheels
python3 -m pip download xxx --dest offline_wheels
```

将 `offline_wheels` 目录传输到内网服务器并复制到容器的用户主目录，然后在容器内执行：

```bash
pip3 install --no-index --find-links="${HOME}/offline_wheels" xxx
```

安装完成后，返回 [第 2.1.7 节](#217-容器内检查环境安装正确性) 执行验证命令，无需再次执行联网安装命令。

## 4. 附录 B：在 GPU 环境下训练模型并采集数据

若需自行采集，请确保 GPU 环境已安装与 NPU 环境版本一致的 PyTorch，并安装 msProbe：

```bash
pip3 install -U mindstudio-probe
```

在 GPU 环境执行以下命令创建采集配置：

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

参照 [第 2.2.2 节](#222-准备模型训练代码) 将训练代码写入 `~/precision_sample.py`，随后启动训练与数据采集：

```bash
python3 ${HOME}/precision_sample.py --gpu 0
```

采集完成后，执行以下命令打包数据，以便传输至 NPU 容器：

```bash
tar -czvf ${HOME}/msprobe_dump_gpu.tar.gz ${HOME}/msprobe_dump_gpu
```

## 5. 常见问题（FAQ）

### 5.1 退出容器后如何重新进入？

在宿主机上选择以下任一方法重新进入容器：

**方法一（推荐）：使用容器启动脚本**

```bash
~/ctr_in.py
```

根据提示选择目标容器；若仅有一个可进入的容器，脚本会自动进入该容器。

**方法二：使用 Docker 原生命令**

```bash
docker exec -it alice_YYMMDD_HHMMSS bash
```

请将 `alice_YYMMDD_HHMMSS` 替换为实际容器名称。可先执行 `docker ps` 查看正在运行的容器及其名称。

### 5.2 执行 Docker 命令时提示 permission denied 如何处理？

当前用户可能未加入 Docker 用户组。可使用 root 权限在宿主机执行：

```bash
sudo usermod -aG docker "${USER}"
```

执行后需退出当前用户会话并重新登录，或执行以下命令使用户组变更立即生效：

```bash
newgrp docker
```

完成后执行 `docker ps` 验证 Docker 命令是否可正常使用。不建议以 root 用户身份进行日常操作。Docker 用户组具有较高的系统权限，请仅将可信用户加入该用户组。

### 5.3 TensorBoard 端口被防火墙拦截时如何访问？

当服务器防火墙限制直接访问 TensorBoard 端口时，可使用 **VS Code 端口转发** 或 **SSH 本地端口转发**。这两种方式仅需 SSH 端口连通，无需额外开放 6006 端口。

#### 方法一：VS Code 端口转发（推荐）

若已使用 VS Code Remote-SSH 连接服务器，可通过图形化界面快速完成端口映射：

1. 中止上面启动的 TensorBoard 进程，改为在 VS Code 远程终端中启动 TensorBoard：

    ```bash
    tensorboard --logdir "${HOME}/graph_visualize_output" --bind_all
    ```

2. VS Code 通常会自动检测终端中的端口监听信息并在右下角弹出提示，直接点击链接即可访问；
3. 若没有弹出提示，也可以点击 VS Code 底部面板的 **“端口 (Ports)”** 选项卡，选择 **“转发端口”**，输入 `6006` 并确认；
4. 转发成功后，点击列表中生成的 **“本地地址”** 链接（如 `http://localhost:6006`），即可在本地浏览器中直接访问。

#### 方法二：SSH 命令行端口转发

若未使用 VS Code，可通过原生 SSH 命令建立隧道：

1. 在本地终端（Windows PowerShell / CMD / Linux / macOS）执行以下命令，按提示输入密码：

    ```bash
    ssh -L 6006:localhost:6006 your_username@192.168.1.1
    ```

    > 请将 `your_username` 和 `192.168.1.1` 替换为实际的用户名和服务器 IP，并保持该 SSH 会话处于连接状态。

2. 在本地浏览器中打开：

    ```text
    http://localhost:6006/
    ```
