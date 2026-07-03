# 大模型训练精度定位指南

## 1 精度问题概述

---

### 1.1 定义

随着大语言模型技术的迅速发展，尤其是在诸如`ChatGPT`、`DeepSeek`等应用的引领下，大模型迅速成为AI界的研究热点。大模型训练需要强大的算力支撑，涉及数据、模型、框架、算子、硬件等诸多环节和技术层面。由于模型规模巨大，训练过程复杂且多变，经常出现精度问题。

训练精度问题是多种因素共同作用的结果，主要表现为训练收敛不及预期，如`Loss`不对齐、`NaN`、尖刺、下游任务效果变差等。

**训练精度问题一般可分为模型精度问题和数值精度问题。**

模型精度问题主要指模型从数据集中读取的数据、模型的训练超参数、模型结构甚至框架本身设计或使用过程等出现问题。模型精度问题对收敛有非常大的影响，需逐项仔细排除、分析，并结合实际情况进行调整。

数值精度问题主要指由于浮点数计算过程的有限字长效应、计算序、通信序或各种计算的数学表达式所带来的近似误差。计算数值的近似性一定概率上会影响模型的收敛性，但不能简单地认为计算过程差异一定会导致模型收敛出现问题。算子的数值精度是计算过程的基础，通常认为算子精度问题是大模型精度问题的来源之一，需要引起重视。但由于实现过程差异，不同硬件（如`GPU`和`CPU`之间，`GPU`各版本之间）同样的计算过程，数值计算结果通常会有一定差异，在特定容限范围内，不会影响模型最终收敛。

为了更好地分析和定位模型精度问题和数值精度问题，正确区分正常计算差异和引起模型精度问题的异常差异，本指南详细介绍了精度问题定位工具集详细的使用场景和使用步骤，方便用户自行或在文档指导下排查潜在的精度问题。

### 1.2 标准

当客户对模型训练提出要求时，需要参照以下产品线提供的精度标准来进行判断，只有不满足精度标准时才认为是精度问题。

- **训练交付标准**  
 当接收到客户反馈的问题时，需首先判断整体训练是否满足训练交付标准，确认要求是否合理。该标准提供了不同场景下的精度测试标准，例如预训练、微调、`Loss`曲线等。
- **算子精度标准**  
 在定位可疑算子时，需参照算子精度标准判断该算子是否满足算子精度标准。该标准详细介绍了算子和框架`API`的分类，并提供了多种精度判定方法和对应的测试标准。

### 1.3 场景介绍

**训练精度问题发生的场景可分为有标杆和无标杆两类。**

有标杆对应迁移场景，即用户将原本在标杆（如`GPU`、其他训练框架）上训练的模型迁移到`NPU`上进行训练。  
无标杆对应原生开发场景，即用户直接在`NPU`上进行模型搭建及训练。

其中，本文聚焦主流的有标杆迁移场景，主要表现为`NPU`训练过程和结果与标杆（`GPU`或`NPU`的其他框架）上的训练过程和结果不一致且偏差超过容忍阈值，我们称之为不对齐。该场景具体可再细分为以下几类现象：

- **溢出或NaN**，即相较于标杆更频繁地出现`Loss`或`Grad Norm`溢出、`NaN`，如下图所示：  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/a5183818-b00b-4f5b-bb23-9b8277f0b004/image.png" alt="image.png" width="650"/>  
- **首Step Loss差异**，即第0步或前几步`Loss`与标杆出现差异，平均误差大于1%，如下图所示：  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/deb21294-900a-4411-9857-890ae31950cd/image.png" alt="image.png" width="400"/>  
- **长稳Loss差异**，即前期`Loss`拟合，后期与标杆差异逐渐变大，平均误差大于1%，如下图所示：  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/de267ea9-69ba-4f15-a179-54efc1a48528/image.png" alt="image.png" width="400"/>  
- **尖刺**，即相较于标杆更频繁地出现`Loss`或`Grad Norm`陡增又快速跌落的现象，如下图所示：  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/eb705a99-b07c-4014-a966-4136825144a5/image.png" alt="image.png" width="400"/> 
- **训练中Loss与标杆相比差异较小但下游任务表现差**，如下图所示：  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/e46ee64d-2c1d-4a76-8e99-eec2236165dd/image.png" alt="image.png" width="400"/> 

**值得注意的是，哪怕属于同一类问题现象，其根因也复杂各异**，具体可参考[根因介绍](#51-根因介绍)。本文将介绍大模型训练精度问题定位时的整体定位思路和标准流程，以及在定位过程中涉及的训练精度工具使用方法，旨在帮助用户快速熟悉和掌握精度定位流程。

## 2 精度问题定位具体步骤

---
大模型训练整体精度定位流程如图所示。  
 <img src="https://raw.gitcode.com/user-images/assets/7898473/83c3651f-ebc3-4683-99fb-6b042d9df5f4/image.png " alt="image.png" width="800"/>  
本章主要针对迁移场景介绍精度问题定位的具体步骤，旨在帮助用户更快理解工具使用的原理和方法，并举一反三到其他更加具体和复杂的问题场景。

### 2.1 检查CheckList

在定位有标杆精度问题之前，需先排除其他非算子因素的干扰。目前大部分精度问题是由于模型超参、三方库版本、环境变量、数据读取、模型结构不一致等因素导致的，为了在定位过程中少走弯路，需在定位前先对训练环境及前置准备根据下列`CheckList`做有效排查。  
另外问题定位主要基于标杆设备和`NPU`设备的对比，因此定位的前置条件是需要分别准备标杆和`NPU`训练环境。

- **训练超参和环境变量比对**  
 可以使用`Beyond Compare`软件比对双方训练日志或启动脚本中的训练超参和环境变量设置，也可以使用[脚本比对工具](#48-脚本比对工具)进行自动比对。其中，常见训练超参可参考[附录-模型超参数](#52-模型超参数)。
- **三方库版本比对**  
 通过`git`分支检查`MindSpeed-LLM`、`Megatron`、`DeepSpeed`等三方库版本是否与标杆对齐。
  通过`pip list`检查`PyTorch`、torch-npu等第三方库版本是否与标杆对齐，也可以使用[脚本比对工具](#48-脚本比对工具)进行自动比对。
- **数据读取检查**  
 检查从数据集中读取后并送入模型训练的数据，一般可通过精度采集工具采集最开始的输入数据或直接在代码中调用模型`forward`时保存或打印传入的具体`tensor`来进行数据集检查，也可使用[脚本比对工具](#48-脚本比对工具)进行自动比对。
- **模型结构检查**  
 通过在双方训练中直接打印模型结构并进行比对。
- **权重初始化对齐**  
 需要确认训练前的初始化权重是否一致，需保证加载同一个预训练模型或使用一样的初始化随机种子，[固定随机性](#221-固定随机性)可参考问题复现章节，检查时可以使用[脚本比对工具](#48-脚本比对工具)进行自动比对。
- **环境版本更新**  
 这一项仅在条件允许的情况下进行，根据之前的精度问题定位经验，很多问题都是旧版本上的问题，在新的版本上已经解决。因此，在条件允许的情况下，推荐安装最新版本的`CANN`、驱动以及`torch-npu`包。

### 2.2 问题复现前置操作

根据上一章节排除其他非算子因素后，进入下一个定位环节。首先需要尽可能明确问题能复现，跑出和客户描述相同的训练过程。

#### 2.2.1 固定随机性

复现需要固定存在随机性的步骤，保证实验可重复性。存在随机性的步骤包括模型参数初始化，`Dropout`层，数据`batch`加载顺序等。  
涉及到的操作如下几项：

- 固定随机种子，如`np.random.seed`、`torch.manual_seed`、`torch_npu.npu.manual_seed`等；
- 关闭`Dropout`层；
- 数据加载关闭`shuffle`，将其设置为`shuffle=False`。

此处涉及操作较多，建议结合[工具固定（seed_all）](#223-工具固定)进行自动设定。

#### 2.2.2 打开确定性

复现时建议打开算子计算确定性和通信确定性，两者都需要在训练开始的代码之前，尽早进行固定，具体可通过以下两项设置：

- 算子计算确定性：

```python
torch.use_deterministic_algorithms(True)
```

- 通信确定性：

```bash
export HCCL_DETERMINISTIC=TRUE
```

注：不是所有的算子都支持确定性计算，对于一些特殊的暂未提供确定性计算特性的算子，参考[算子确定性问题](#2322-算子确定性问题)进行定位。

#### 2.2.3 工具固定

`msprobe`工具包提供`seed_all`接口快速固定网络中所有随机种子、`Dropout`层及算子计算和通信确定性，除`seed_all`之外只需再手动关闭数据集`shuffle`。

**使用方式：**

```python
from msprobe.pytorch import seed_all
seed_all(seed=1234, mode=True, rm_dropout=True) 
```

**参数说明：**  

| 参数名 | 可选/必选 | 说明 |
| ---  | --- | --- |
|seed| 可选 | 随机数种子。默认值为`1234`。|
|mode| 可选 | 确定性计算模式。可配置`True`或`False`，默认值为`False`。该模式同时包含算子计算确定性和通信确定性。|
|rm_dropout| 可选 | 控制`dropout`失效的开关，开启后会自动将`dropout`概率设置为0。可配置`True` 或 `False`，默认值为`True`。|

**固定随机数范围：**  
`seed_all`函数可固定随机数的范围如下表。

|API|固定随机数|
| --- | --- |
|os.environ['PYTHONHASHSEED'] = str(seed)| 禁止`Python`中的`hash`随机化。|
|random.seed(seed)| 设置`random`随机生成器的种子。|
|np.random.seed(seed)| 设置`numpy`中随机生成器的种子。|
|torch.manual_seed(seed)| 设置当前`CPU`的随机种子。|
|torch.cuda.manual_seed(seed)| 设置当前`GPU`的随机种子。|
|torch.cuda.manual_seed_all(seed)| 设置所有`GPU`的随机种子。|
|torch_npu.npu.manual_seed(seed)| 设置当前`NPU`的随机种子。|
|torch_npu.npu.manual_seed_all(seed)| 设置所有`NPU`的随机种子。|
|torch.backends.cudnn.enable=False| 关闭`cuDNN`。|
|torch.backends.cudnn.benchmark=False |`cuDNN`确定性选择算法。|
|torch.backends.cudnn.deterministic=True| `cuDNN`仅使用确定性的卷积算法。|

#### 2.2.4 缩小规模

对于一些大集群如千卡甚至万卡训练出现的精度问题，需要将集群规模缩小进行复现定位。  
常见的做法是保持`TP, PP, CP, SP, EP`等切分参数不变，将`Batch Size`缩小或直接减少模型层数，裁剪时需伴随实验确保能复现，最终选择规模尽可能小且可复现的训练参数。

### 2.3 精度问题分场景定位

通过完成上一节操作后，大部分问题可成功复现，但仍有小部分疑难问题无法复现如出现位置、现象不固定或只能概率复现等。

- 对于稳定复现的问题，精度问题分析的基本原理是通过采集标杆和`NPU`训练过程中数据并进行细粒度对比，发现可能异常的计算`API`及相关算子。
- 对于不稳定复现的问题，一般是涉及不支持确定性计算的特殊算子、内存踩踏、硬件层面导致的多比特翻转或三方库和框架`bug`，需进行逐一排查。

我们推荐使用昇腾`MindStudio Training Tools`工具链下提供的精度调试工具包`msprobe`在训练过程中采集数据并分析问题，详细使用方式可参考[msprobe工具定位](#4-msprobe工具定位)。

#### 2.3.1 稳定复现场景

已经确认排除[CheckList](#21-检查checklist)且能稳定复现后，整体定位流程如下图所示：  
<img src="https://raw.gitcode.com/user-images/assets/7898473/d7955340-9bde-479c-801e-f22573f50190/image.png" alt="Your image title" width="800"/>  

##### 2.3.1.1 溢出或NaN

**问题描述**

在根据[CheckList](#21-检查checklist)排除其他非算子因素，并且固定随机性的情况下，首先确认该问题是否相较于标杆出现了更频繁的`Loss NaN`或梯度溢出，一般还伴随着`Loss Scale`的持续降低。  
<img src="https://raw.gitcode.com/user-images/assets/7898473/a5183818-b00b-4f5b-bb23-9b8277f0b004/image.png" alt="image.png" width="650"/>  

**排查思路**

溢出一般需要先找到首个进行`NaN`分析的问题`Step`：

- `NPU`上出现`NaN`但`GPU`上没有，把第一个出现`NaN`的步数认为是问题`Step`做定位。
- `NPU`和`GPU`都出现了`NaN`，但`NPU`出现`NaN`的次数明显多于`GPU`，把第一个`NPU`有`NaN`而`GPU`无`NaN`的步数认为是问题`Step`做定位。

确定需要分析的`Step`后，按照如下步骤进行排查：

1. 采集该`Step`的数据做分析和比对：

    - 使用[精度采集工具](#43-精度采集工具)溢出步的前反向输入输出：
        - 若加入工具后溢出消失，怀疑为内存踩踏，参考[内存踩踏](#2321-内存踩踏)进行排查。
        - 若加入工具后仍能复现：
            1. 结合[分级可视化工具](#44-分级可视化工具)溢出分析功能或手动搜索`Inf/NaN`查看溢出最先发生的位置：
                - 若为`weight`，则怀疑上一步反向梯度先出现问题，切换定位`Step`为上一步重新采集梯度数据。
                - 若为`input`，则怀疑存在未被采集的特殊算子，需对照代码栈分析`input`来源。
                - 若为`output`，则`output`所在算子需重点分析。  
                其中`dump.json`文件中`input`、`weight`、`output`位置识别方法可参考[精度采集工具](#43-精度采集工具)中的`dump.json`统计量结果详解。
            2. 若没有定位到可疑算子，可能为累积误差导致，参照`Loss`对齐问题（[首Step Loss差异](#2312-首step-loss差异)/[长稳Loss差异](#2313-长稳训练loss差异)）进一步分析。
    - 若模型规模较大，精度采集工具所需时长过长或者不稳定复现，也可考虑使用[训练状态监测工具](#46-训练状态监测工具)或[非工具手段补充](#49-非工具手段补充)中手动挂`Hook`的方式来采集各层梯度，看是否有梯度异常的层。

2. 在`NaN`问题中，若根据以上排查没有找到根因，可补充排查以下几种特殊情况：

    - 在`Megatron`、`DeepSpeed`类模型中，`overlap`参数（如`overlap-param-gather`、`overlap-grad-reduce`等）存在较高风险，可先关闭该类参数。
    - 在`FSDP`框架下使用混合精度出现的`NaN`问题，建议优先排查`torch-npu`框架导致的内存踩踏，可尝试切换torch-npu版本。
    - `FA`（`npu_fusion_attention`）融合算子功能复杂，使用时易出现传参规范错误的问题，可先关闭`FA`分支，定界是否为`FA`导致。若定界确实为`FA`分支导致后，请参照[FA官网文档](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000762.html)来排查是否存在使用规范错误。
    - 确保打开了`Inf`/`NaN`模式或者非饱和模式，参照[附录-非饱和模式](#53-非饱和模式)。

##### 2.3.1.2 首Step Loss差异

**问题描述**

在根据[CheckList](#21-检查checklist)排除其他非算子因素，且固定随机性的情况下，`GPU`和`NPU`的第一步或前几步的`Loss`就已经出现差异，平均相对误差大于`1%`。如下图所示。  
<img src="https://raw.gitcode.com/user-images/assets/7898473/deb21294-900a-4411-9857-890ae31950cd/image.png" alt="image.png" width="400"/>  

**排查思路**

按如下步骤进行定位：

1. 使用[精度采集工具](#43-精度采集工具)，采集最先出现差异的步数（相对误差大于`1%`）：

    - 若第1步`Loss`（对应训练为第0步）不对齐，则使用工具采集第一步`NPU`与标杆的统计量数据，并排查其正向的计算结果。
    - 若第1步`Loss`对齐但第2步`Loss`不对齐，则使用工具排查`NPU`与标杆在第1步反向+第2步正向的计算结果。
    - 第3、4步同理第2步，只要是在有限几步内出现的都可以用这种方式进行排查。

2. 使用分析工具进行比对，寻找可疑算子：

    - 使用[分级可视化工具](#44-分级可视化工具)做画图比对，结合精度筛选功能（颜色深到浅）+首个精度出现差异的节点来进行排查分析。
    - 也可使用[精度比对工具](#45-精度比对工具)做表格比对，通过分析输入输出精度差异来进行排查分析。

3. 若没有定界到可疑算子，建议使用[精度预检工具](../user_guide/accuracy_checker/pytorch_accuracy_checker_instruct.md)进行辅助排查。

##### 2.3.1.3 长稳训练Loss差异

**问题描述**

在根据[CheckList](#21-检查checklist)排除其他非算子因素，且固定随机性的情况下，`GPU`和`NPU`训练初期`Loss`能对齐，但是后期训练`Loss`对不齐，后期平均相对误差大于`1%`。如下图所示。  
<img src="https://raw.gitcode.com/user-images/assets/7898473/de267ea9-69ba-4f15-a179-54efc1a48528/image.png" alt="image.png" width="400"/>  

**排查思路**

1. 根据情况选择工具并进行数据采集：

    - [精度采集工具](#43-精度采集工具)，适用于问题第一现场步数确定的场景，如中途`Loss`或`Grad Norm`存在明显跳变，使用思路如下：  

        1) 确定采集步数（两者取最小步数）：  

            - `Loss`跳变采集上一步反向+当前步正向的数据。  
            - `Grad Norm`跳变采集当前步反向的数据。  

        2) 确定采集规格：  

            - 模型规模小，直接采集`mix`级（`API`级+`Module`模块级）统计量。  
            - 模型规模大，先采集模块级统计量，定位到模块后再内部采集`API`级统计量。  

        3) 在统计量上通过使用可视化或精度比对工具分析到可疑算子后，采集具体`tensor`值进行进一步的单算子分析。  

    - [训练状态监测工具](#46-训练状态监测工具)，适合大规模、不确定采集`dump`步数的场景，使用思路如下：  
        - 若问题同时表现为`Grad Norm`先`Loss`后，则优先采集训练过程中梯度数据。  
        - 若问题主要表现在`Loss`上，则优先采集训练过程中激活值和权重数据。  

2. `dump`数据可参照上一节使用分级可视化工具或精度比对工具进行分析，训练状态监测数据参考[4.6节使用思路](#46-训练状态监测工具)进行结果分析。
3. 若标杆数据采集成本较大，在不确定步数的情况下无法频繁采集，可补充精度预检工具+无标杆工具进行排查。
4. 在长稳问题中，若根据以上排查没有找到根因，可补充排查以下两种特殊情况：

    - 排查优化器，如尝试`Adam`优化器转`SGD`、替换融合`Adam`为小算子、关闭优化器定界问题前反向等。
    - 排查`Matmul`错峰策略，当定位到可疑的`Matmul`算子但无法通过单算子验证排除时，可以尝试关闭错峰策略。

    ```bash
    export CLOSE_MATMUL_K_SHIFT=1
    ```

#### 2.3.2 不稳定复现场景

已经确认排除[CheckList](#21-检查checklist)且完成问题复现的前置操作后仍不能稳定复现，整体定位流程如下图所示。  
<img src="https://raw.gitcode.com/user-images/assets/7898473/99f2e246-592c-4ee3-aa22-bc47fcceffc4/image.png" alt="Your image title" width="800"/>  

##### 2.3.2.1 内存踩踏

**问题描述**

内存踩踏一般多发在现象为溢出或NaN的精度问题中，其可进一步分为以下两类情况：

- 若开启流同步或加入精度采集工具后，问题能够稳定消失，对于这一类问题，大概率是因为计算流/通信流的踩踏、框架内部内存分配偏移量计算错误、通信未做保护导致读入脏数据等。
 因此加入同步后算子和算子之间、通信和算子之间被完全隔离开来，不再复现。其中开启流同步命令如下：

 ```bash
 export ASCEND_LAUNCH_BLOCKING=1
 ```

- 但对于算子内部出现踩踏的问题，哪怕加入流同步也无法规避。特别是在分核计算和分`ub`计算，出现非整块、计算复杂和数据类型变化时，易出现算子内部踩踏。

**排查思路**

1. [精度采集工具](#43-精度采集工具)改`异步dump`，具体操作为在`config.json`文件中加入`async_dump: true`的配置项，同时采集开启流同步无`NaN`+不开流同步有`NaN`的两组训练中最先出现`NaN`的`tensor`数据。对于异步`dump`仍影响问题复现的，使用手动挂`Hook`或者`print`的方式采集数据。
2. 分析`tensor`差异特征是否满足内存踩踏的规律性，一般内存踩踏时踩踏区域较为规整，如为按整倍踩（如`2048`）、按行踩、按列踩等。
3. 使用`profiling`结合`insight`工具查看计算并行关系，具体操作参考[profiling使用文档](https://gitcode.com/Ascend/msprof/blob/master/docs/zh/quick_start/msprof_quick_start.md)和[insight使用文档](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/overview.md)。
4. 添加`ptr`内存地址打印，针对`NaN`出现的位置侵入式修改`PyTorch`或`torch-npu`源码添加打印。
5. 使用[算子检测工具](https://gitcode.com/Ascend/mssanitizer/blob/master/docs/zh/quick_start/mssanitizer_quick_start.md)排查算子流水内（不同指令间执行）、流水间（算子搬运操作）和核间（`aicube`和`aivector`并行）的实现是否存在异常，来判断该算子是否存在内存踩踏。
6. 若以上排查仍未能定位根因，可进一步参考更详细的内存问题定位指南。

##### 2.3.2.2 算子确定性问题

**问题描述**

对于一些使用了工具自动固定之后，开启流同步重复训练结果仍不一致的，优先使用精度采集工具结合`md5`模式采集两次重复训练的数据，排查首个异常算子，若该算子为特殊随机算子或位置固定且计算虽有差异但结果相近，则考虑为算子确定性问题。

**排查思路**

结合精度采集工具+`md5`模式重复采集2次训练的结果，比对排查找到首个输入一致输出不一致的算子：

- 若首个出现的为输入不一致，则需对照代码栈查看是否有漏采集的算子。
- 若首个出现的为输入一致输出不一致，则该算子为可疑算子。

若该算子为特殊随机算子或位置固定且计算虽有差异但结果相近，根据以下方案来尝试解决：

- 特殊随机算子（如`torch.randn`）  
 尽管工具内部对于随机的控制是通过设定统一的随机种子进行随机性固定的，但是由于硬件的差异，可能会导致同样的随机种子在不同硬件上生成的随机数不同，如下示例：  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/f690a76c-5a00-48ab-ae0b-f77512715275/image.png" alt="Your image title" width="500"/>  
    图中可见，特殊随机算子`torch.randn`在`NPU`和`GPU`上固定随机种子后，仍然生成不同的随机张量。对于此类场景，用户需要将网络中的`randn`在`CPU`上生成后再转对应`device`，这样在host侧生成的随机张量能够保证一样，搬移到`NPU`或者`GPU`设备上仍然一样。
- 算子暂不支持确定性计算  
 除随机性算子之外，还有部分算子暂不支持确定性计算的特性，因此每次运行结果会有细微差异。因此若排查到一些不常见的特殊算子（如`MSDA`、`grid_sample`）可通过转`CPU`或联系算子支撑人员咨询是否有替代实现来进行规避。

##### 2.3.2.3 硬件问题

**问题描述**

对于一些大集群的任务，可能会出现硬件故障（如多`bit`翻转、电源故障）带来的纯偶然事件，一般表现为`Loss`跑飞或尖刺，且通常换设备后问题消失。

**排查思路**

对于大集群任务可采用硬件压测来排除精度异常节点，压测可通过如下方法进行：

- 模型压测：使用分组的单机或多机任务训练找到与其他大部分卡或机器精度不一致的机组。
- 算子压测：若不稳定现象出现在某个特定算子且单算子无法复现，可分组单机做单算子压测。
- 命令压测：使用`ascend-dmi`命令对`aicore`进行重复压测，命令如下：

 ```bash
 ascend-dmi -dg -i aicore -s -sc 60 -q
 ```

除此之外，也可以通过以下命令逐步禁用相关通信链路，排查通信相关的硬件问题。

- 禁用ROCE：

 ```bash
 export HCCL_INTRA_ROCE_ENABLE=0
 ```

- 禁用PCIE：

 ```bash
 export HCCL_INTRA_PCIE_ENABLE=0
 ```

##### 2.3.2.4 训练框架排查

**问题描述**

对于使用了`Megatron`、`DeepSpeed`类训练框架的模型，可以针对性地排查一些高危因素。

**排查思路**

- Megatron
    - 去除`overlap`类参数，常出现在训练正常、推理结果异常的现象中，如`overlap_param_gather`参数曾因保存`checkpoint`时通信未完成导致保存权重有误。
    - 简化`TP/PP/SP/CP/EP`切分策略。
- DeepSpeed
    - 去除`overlap`类参数。
    - `bucket_size`参数需满足一定条件。
    - 切换`Zero1/2/3/offload`切分策略。

##### 2.3.2.5 训练阶段/模型阶段/版本排查

**问题描述**

对于使用以上所有排查手段后，仍未找到原因的，可通过训练阶段排查或模型阶段排查来缩小排查范围。

**排查思路**

- 训练阶段排查
    - 关闭优化器或设学习率为0来定界前反向。
    - 替换优化器来定界优化器本身的问题，如`Adam`换`SGD`、融合算子`Adam`换小算子拼接等。
- 模型阶段排查
    - 模型分阶段排查：如减层、二分、排除`attention`等。
    - 挪`device`：在不确定哪个算子引发的问题，在测试成本允许的情况下可以通过二分把整网的算子放到`CPU`上来执行。
    - 明确问题出现在反向时，可通过二分法冻结梯度来进行尝试。
- 版本排查  
 在明确现象和问题点跟特定的版本绑定，但是无法定位到根因时，可采用版本二分的办法来确认问题合入。例如`CANN`可以通过替换对应合入的`so`来将算子实现替换到对应版本。`PyTorch`同理，可以对特定合入进行编译软件包来排查问题。但该方法不适用于短期内版本合入过多或某个合入影响过大的情况。

#### 2.3.3 强化学习场景

**问题描述**

在根据[CheckList](#21-检查checklist)排除其他非算子因素的情况下，强化学习训练中若出现了`reward`相对标杆下降，或伴随`logp_diff`上升等现象，则需对其展开精度排查。  
<img src="https://raw.gitcode.com/user-images/assets/7898473/a9411220-551b-4810-9560-6e8d15194203/image.png" alt="Your image title" width="600"/>

**总体排查流程如下**    
<img src="https://raw.gitcode.com/user-images/assets/7898473/c1041303-ad83-4b31-ab0d-bc46bbecc50a/image.png" alt="Your image title" width="700"/>  

##### 2.3.3.1 基础推理排查

在强化学习链路中，`response`是关键中间输出，既决定推理效果，又直接影响`reward`计算与后续训练，因此必须优先保证推理输出基本正确。

为提升定位效率，满足以下任一条件时，需对推理模块进行基础排查：

- **首步推理有乱码。**
- **首步推理前20个`token`与标杆不一致。**
- **推理本身数据集评测不达标。**

若以上现象皆未触发，则直接跳转[reward排查](#2332-reward排查)，否则针对推理本身与标杆进行数据采集与比对分析。

强化学习中的推理框架一般采用`vLLM`（昇腾上使用`vLLM-Ascend`代替）或`SGlang`，对应的排查步骤如下：

**数据采集：**

- 对于`vLLM-Ascend`框架的推理数据采集参考[vLLM-Ascend精度数据采集](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html)。
- 对于`SGlang`框架的推理数据采集参考[SGlang精度数据采集](../user_guide/dump/sglang_eager_dump_instruct.md)。

注意在使用[精度采集工具](#43-精度采集工具)时采集配置需指定`level`为`mix`或`L0`，即需至少包含`Module`级别数据，保证后续可做逐层比对。基础配置config.json样例如下：

```json
{
    "task": "statistics",
    "dump_path": "/home/data_dump",
    "rank": [],
    "step": [0],
    "level": "mix",
    "statistics": {
        "list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

其中"level"取值可切换为"L0"。

**数据比对：**

- 同框架推理（如`vLLM-Ascend` vs `vLLM`）  
 数据采集后可参考[首Step Loss差异章节](#2312-首step-loss差异)中的第2点分析思路进行比对。
- 跨框架推理（如`vLLM-Ascend` vs `SGlang`）  
 数据采集后NPU会与标杆存在大量的层级名称/结构差异，优先以分级可视化工具进行比对，并通过[分级可视化工具](#44-分级可视化工具)的`点点匹配`功能，对因模块名称差异导致未自动匹配的节点进行手动对齐，方便后续进行对比分析。  
    <img src="https://raw.gitcode.com/user-images/assets/7898473/1c469d88-ebc5-45eb-b19d-6d40f1568755/image.png" alt="Your image title" width="800"/>  

除此之外，更详细的操作可参考[推理精度定位指南](./infer_debug_guide.md)。

##### 2.3.3.2 reward排查

在强化学习中，`reward`以`prompt`+`response`为输入，对回答的结果进行打分。`reward`异常将直接导致整个强化学习目标跑偏，因此在推理后需优先保证`reward`模块计算正确。

满足以下条件时，需对`reward`进行排查：

- **首步推理结果一致（或插桩），但`reward`差异大。**

若未触发，则跳转[基础训练排查](#2333-基础训练排查)，否则针对`reward`计算逻辑与标杆进行比对排查。

`reward`打分器排查根据其训练数据集类型不同可分为如下几类：

|数据集类型| 打分方式| 对应逻辑| 下一步排查操作|
| --- | --- | --- | --- |
|Math| 规则打分| 数学题答案高度结构化，规则打分是最高效的打分方式，一般直接使用字符串匹配。| 对齐标杆的规则打分代码。 |
|Code| 沙箱打分| 代码的正确性无法靠字符串匹配（如不同写法实现同一功能），必须通过沙箱执行结合测试用例验证。|   对齐沙箱执行与用例判定逻辑。|
|通用| 模型打分| 通用场景（如闲聊、文案）无唯一标准答案，只能靠`Reward Model`学习人工偏好。| 1. 排查打分模型是否开启了随机性配置。<br>2. 参考[首Step差异章节](#2312-首step-loss差异)进行数据采集与比对。|

##### 2.3.3.3 基础训练排查

在强化学习中，训练阶段负责权重与梯度更新，需保障首步训练无明显异常。

满足以下条件时，需对训练进行基础排查：

- **首步推理结果一致（或插桩），且`reward`正常，但`pg_loss`、`gnorm`与标杆差异大。**

若未触发，则跳转[resharding权重同步排查](#2334-resharding权重同步排查)，否则针对训练前向/反向与标杆进行首步比对排查。

**前置条件**

在`verl`框架训练中，为了提高训练泛化性、硬件利用率等，会通过对输入数据进行打乱或重组来实现，从而导致数据比对时输入数据无法对齐（`batch`内数据顺序错乱、`batch_size`变化）等。因此在定位训练相关问题前，请先设置/检查如下开关：

- 数据集`shuffle`开关：读取数据集后对所有`batch`进行随机顺序打乱

  ```python
  data.shuffle=False  # 必须关闭
  ```

- `balance_batch`开关：对`global batch`内的`DP`域交换顺序，保证每张卡上分到的`token`数量相对均分

  ```python
  trainer.balance_batch=False  # 训练与标杆比对时保持一致即可，训推一致比对时必须关闭
  ```

- `use_dynamic_bsz`开关：防止不同`micro batch`之间`token`数差异大导致`micro batch`间时而满载时而空闲，`mini_batch`拆分`micro batch`时支持动态`batch`开关，开启后`micro batch`的大小不再固定，而是凑齐`max_token_len`再组`micro batch`

  ```python
  actor_rollout_ref.actor.use_dynamic_bsz=False  # 训练与标杆比对时保持一致即可，训推一致比对时必须关闭
  ```
  
强化学习中的训练后端一般采用`FSDP`或`Megatron`（昇腾上会使用`MindSpeed/MindSpeed-LLM`打`patch`代替），对应的排查步骤如下：  

**数据采集**

- 对于FSDP框架的训练数据采集，在`verl`框架中的采集文件位置为`verl/workers/fsdp_workers.py`，工具在该文件中具体插入位置参考[VERL(FSDP后端)采集](./dump_enable_guide.md#verl-fsdp后端)。
- 对于Megatron类框架的训练数据采集，在`verl`框架中的采集文件位置为`verl/workers/megatron_workers.py`，工具在该文件中具体插入位置与`FSDP`类似，此处不赘述。

**数据分析**

- 同框架训练（如`Megatron + MindSpeed` vs `Megatron`）  
    - 小规模训练，可直接参考[首Step Loss差异章节](#2312-首step-loss差异)进行数据分析。
    - 大规模训练，如卡数较多、模型较大的，建议结合[趋势可视化工具](../user_guide/accuracy_compare/trend_visualization_instruct.md)进行分析。
- 跨框架训练（如`Megatron` vs `FSDP`）  
 数据采集后NPU会与标杆存在大量的层级名称/结构差异，建议优先使用[分级可视化工具](#44-分级可视化工具)结合`点点匹配`功能进行比对。

##### 2.3.3.4 resharding权重同步排查

强化学习的核心逻辑是「训练更新权重→推理验证效果→基于推理结果的`reward`反馈再优化训练」，这一闭环要求训练侧更新后的权重必须精准同步到推理侧，每个`step`的`rollout`都需做一次从`actor`训练到`rollout`推理的权重参数同步。若推理使用的权重与训练最新权重不一致，会导致推理输出偏离预期。  
这一步骤我们称之为`resharding`权重同步，其需兼顾三大核心职责：  

- **权重分片适配**：将训练侧的多卡分片权重，按推理侧的并行策略重新拆分 / 合并，适配推理的显存与计算需求。
- **权重读写管控**：负责权重在磁盘（disk）、多卡间（hccl）的读写与同步，确保权重从训练节点写入、推理节点读取的全过程无截断、无偏移、无丢失。
- **权重名称转换**：统一训练与推理侧的权重模块命名（如消除分布式训练的权重后缀、对齐训推模块名称映射），避免因名称不匹配导致权重加载失败。

分布式推理框架中一般可配置使用`dummy`假数据来进行首次执行，用以估算后续正式推理所分配的显存以及计算图的`capture size`。在`verl`框架中（`actor_rollout_ref.rollout.load_format`超参），推理初始化权重格式有如下两种：

- **`safetensors`**：标准的权重存储格式，会读取完整预训练权重。
- **`dummy`**：仅根据权重形状进行随机初始化，不加载真实数值。

满足以下任一条件时，需对`resharding`权重同步进行专项排查：

- **现象1：`dummy`时首步推理出现乱码，切换`safetensors`后首步推理恢复正常。**
- **现象2：`dummy`和`safetensors`首步皆存在乱码，但将推理从强化学习中隔离后进行单推无异常。**

若未触发以上现象，跳转[训推一致排查](#2335-训推一致排查)，否则需重点检查：

- 现象1 — 检查权重对象：  
   一般发生此种现象，代表**某些层权重未同步**，`resharding`过程中通常存在截断、偏移、指针分离等问题。若某层在`parameters`属性中注册的`weight`指针与其实际计算使用的`weight`指针出现分离，会导致即使打印`named_parameters()`里的权重正常更新，但实际计算使用权重无变化。若首步推理权重无变化，在`safetensors`模式下权重仍为初始化加载的正常权重，在首步（甚至前几步）不会表现出明显异常，但在`dummy`模式下，将导致推理侧权重仍为随机初始化权重，直接表现为推理输出乱码。  
   
- 现象2  — 检查权重读写/切分：  
   一般发生此种现象，**代表权重已进行同步，但同步成错误值**。读写可切换`disk`和`hccl`两种读写配置看是否存在差异，或查看切分检查`TP/PP`等配置是否对齐等。

##### 2.3.3.5 训推一致排查

目前主流RL算法是基于`On-Policy`前提展开的，`On-Policy`理论要求采样数据的行为策略与梯度计算的目标策略基本保持一致，才能确保梯度估计是无偏的，从而使训练过程更平稳。在强化学习中采样我们称之为`rollout`推理，梯度计算则对应`actor`训练，当推理与训练策略保持一致，即称为训推一致。  
训推一致性的主要指标为`logp_diff`，当`logp_diff`异常时表示强化学习中训练和推理存在一定程度的差异，需进行训推差异的根因查找，其计算公式为：    
<img src="https://raw.gitcode.com/user-images/assets/7898473/98058476-f95a-44ff-b5d0-7dfb79c1066c/image.png" alt="Your image title" width="400"/>  
其中`M`为`response_mask`。

在`verl`框架中，对应行为如下：  

- 开启`logp_diff`监测的配置超参：

  ```python
  actor_rollout_ref.rollout.calculate_log_probs=True  # 默认为False
  ```

- `logp_diff`的具体代码实现：

  ```python
  mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)
  mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)
  log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
  metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
  ```

当该指标满足以下条件时，需启动训推一致排查：

- **`logp_diff`异常偏大（>0.01）或与标杆差异明显。**

若未触发，直接跳转[长稳训练排查](#2336-长稳训练排查)，否则按如下操作进一步排查：

- 首步`logp_diff`较大  
 参考[训推一致比对](#23352-训推一致比对)对第0步进行NPU本身的训推比对（前提已经进行过`2.3.3.1`~`2.3.3.4`节的基础排查但未触发现象或未查明原因）。
- 首步`logp_diff`正常但后期变大  
 强行设`LR`=`0`，观察`logp_diff`变化：
    - 仍异常 → 判定与训练反向无关，推理侧影响可能性大，可按如下两个方向进行排查：
        - 参考[kv cache排查](#23351-kv-cache排查)排查推理`decode`部分的`kv cache`读写逻辑。
        - 参考[训推一致比对](#23352-训推一致比对)对后期训崩的步数进行`NPU`本身的训推比对。
    - 变正常 → 判定`resharding`仍存在未检出问题
        - 需在[resharding权重同步排查](#2334-resharding权重同步排查)的基础上进行加强验证，具体为采集具体计算时的权重与保存的权重进行对比看是否出现数值差异。

###### <span style="font-size:16px;">2.3.3.5.1 kv cache排查</span>

推理流程中：

- `prefill`：清除上一轮`kv cache`，写入当前上下文。
- `decode`：增量读写`kv cache`。

在实际工程实现中，通常通过 `state block` 管理缓存，而非逐句强制清除，若出现缓存清理不及时、复用错误、地址偏移等问题，会导致`decode`阶段读取异常，进而引发推理长稳漂移，最终导致后期`logp_diff`变大。  
此种情况下，由于`kv cache`为推理时特有的逻辑，与训练无关，因此在强化学习训练中必会触发`logp_diff`不一致。

对于kv cache的排查可通过如下两种方案：

1. 对比`NPU`与标杆的`kv cache`代码读写部分是否存在逻辑差异。
2. 强制逐句清除`kv cache`缓存。

###### <span style="font-size:16px;">2.3.3.5.2 训推一致比对</span>

目前训推一致比对只支持`prefill`阶段，所以需要先定界一致性差异只在`decode`阶段发生还是`prefill`阶段也存在，具体验证实验如下：

设`max_response_len`=`1`，让推理模型只跑`prefill`，同时观察`logp_diff`情况：

- 仍异常 → 单`prefill`也有问题，可进行训推一致性比对（触发比对指标）。
- 变正常 → 大概率为推理`decode`模块特有的问题，可以重点参考排查上一节的[kv cache](#23351-kv-cache排查)读取与写入逻辑。

**比对前置条件**

在`verl`强化学习训练时，`batch`维度被`flat`到`token`维度，因此训推比对时有个前提条件，需保证模型`forward`时的`token`、`feature`维度一致才能进行训练与推理双方统计值、双千比对。

在常规强化学习训练中，推理和训练的模型`forward token`序列维度通常存在如下差异：

- 单`prompt`为例
    - 推理分为`prefill`+`N*decode`  
    `prefill`阶段`forward`的`token`维度为`prompt_len`，`decode`阶段`forward`的`token`维度为`1`，结束后得到`prompt`+`response`
    - 训练则为整体`forward`
      - 若不存在`pad`，`token`维度为`prompt_len`+`response_len`
      - 若存在`pad`，`token`维度为`max_prompt_len`
- 多`prompt`为例
    - 推理会对接收到的单次请求中的多条`prompt`进行拼接，在不超过预算长度的情况下，会拼成`prompt1`+`prompt2`的`token`维度进行`prefill`计算，`decode`阶段`forward`的`token`维度为`2`，结束后得到`prompt1`+`response1`、`prompt2`+`response2`
    - 训练还可能存在`mini batch`循环+梯度累积循环对`prompt`数据进行拆分，一旦进行拆分则与推理不再可能`shape`一致，因此前提需要保证训练中无`batch`循环拆分。

在常规强化学习训练中，推理和训练的模型`forward feature`特征维度在切分配置不一致时（如开启不同策略的`TP`等）也会存在差异。

综上所述，要保证训推可比，前提条件为：

1. 保证训练`batch`未被拆分

    - 需保证每轮训练中用于梯度更新的`mini batch`个数`mini_batch_num`= `1`，计算公式为:

    ```python
    mini_batch_num = train_batch_size/train_ppo_mini_batch_size
    ```

    - 需保证梯度累计步骤数gac =1， 计算公式为：

    ```python
    gac = train_ppo_mini_batch_size*n_resp_per_prompt/train_ppo_micro_batch_size_per_gpu/DP
    ```

    其中，不同训练后端的`DP`计算公式为：
    - `fsdp`是数据并行，`DP=world_size`
    - `megatron`有模型并行，`DP=world_size/TP/PP/CP`

    在`VERL`框架脚本中以上所有值对应的具体超参为：

    ```python
    data.train_batch_size=${train_batch_size}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.ppo_epochs=1  # 默认为1
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    ```

2. 关闭训练中`pad`与动态组`batch`，在`VERL`框架脚本中对应的具体超参为：

    ```python
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.use_dynamic_bsz=False
    ```

3. 训练与推理的切分配置一致，保证如`TP、PP、CP`等切分策略完全一致。
4. 推理只执行prefill，在`VERL`框架脚本中对应的具体超参为：

    ```python
    data.max_response_length=1
    ```

5. 训练与推理不带`response`差异  
    有如下两种实现对齐的方案：

    - **方案1：将训练变单`prompt`。**
    - **方案2：将推理的prefill带上`response`**，具体为推理做完`prefill`和`decode`拿到完整的`prompt`+`response`后，设`max_response`=`1`重做一次`prefill`(`prompt`+`response`)  。
    
    由于后者存在重复推理影响性能，因此本指南优先以前方案1进行操作。

    **方案1具体实现**  
    在verl框架中针对进行训练的输入数据改动并适配`loss`计算保证不报错，对于不同后端，分别对应的修改文件如下：
    - `fsdp`：`verl/workers/actor/dp_actor.py`
    - `megatron`：`verl/workers/actor/megatron_actor.py`

    以`fsdp`为例，具体代码修改处如下：

    ```diff
    ...
    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """..."""
            response_length = micro_batch["responses"].size(-1)
    +        if "responses" in micro_batch and micro_batch["responses"] is not None:
    +            response_length = micro_batch["responses"].size(-1)
    +        else:
    +            response_length = 0
    multi_modal_inputs = {}
    ...
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """..."""
        # set to eval
        self.actor_module.eval()
    
    +        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
    +        if compute_prompts_only:
    +            if "responses" in data.batch:
    +                responses_len = data.batch["responses"].size(1)
    +                data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
    +                data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
    +                if data.batch["position_ids"].dim() == 3:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
    +                else:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
    +                # remove responses from batch
    +                data.batch["responses"] = None
    +                if "rollout_log_probs" in data.batch:
    +                    data.batch["rollout_log_probs"] = None
    +                if "response_mask" in data.batch:
    +                    data.batch["response_mask"] = None         
    + 
    
    micro_batch_size = data.meta_info["micro_batch_size"]
    ...
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
    
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
    
    +        compute_prompts_only = int(os.getenv("PROMPTS_ONLY", "0"))
    +        if compute_prompts_only:
    +            if "responses" in data.batch:
    +                responses_len = data.batch["responses"].size(1)
    +                data.batch["input_ids"] = data.batch["input_ids"][:, :-responses_len]
    +                data.batch["attention_mask"] = data.batch["attention_mask"][:, :-responses_len]
    +                if data.batch["position_ids"].dim() == 3:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :, :-responses_len]
    +                else:
    +                    data.batch["position_ids"] = data.batch["position_ids"][:, :-responses_len]
    +                # remove responses from batch
    +                data.batch["responses"] = None
    +                if "rollout_log_probs" in data.batch:
    +                    data.batch["rollout_log_probs"] = None
    +                if "response_mask" in data.batch:
    +                    data.batch["response_mask"] = None
    + 
             select_keys = [
                 "responses",
                 "response_mask",
                 "input_ids",
                 "attention_mask",
                 "position_ids",
                 "old_log_probs",
                 "advantages",
             ]
             ...
                         ...
                         # Extract pre-computed rollout correction weights if present
                         # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                         rollout_is_weights = model_inputs.get("rollout_is_weights", None)
    
    +                    if response_mask is None:
    +                        prompt_mask = torch.ones_like(log_prob, dtype=torch.bool)
    +                        response_mask = prompt_mask
    + 
                         # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                         # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                         policy_loss_fn = get_policy_loss_fn(loss_mode)
    
                         # Compute policy loss (any function is expected to return 2 values)
                         pg_loss, pg_metrics = policy_loss_fn(
                             old_log_prob=old_log_prob,
                             log_prob=log_prob,
                             advantages=advantages,
                             response_mask=response_mask,
                             loss_agg_mode=loss_agg_mode,
                             config=self.config,
                             rollout_is_weights=rollout_is_weights,
                         )
                         micro_batch_metrics.update(pg_metrics)
                         ...
    ```

    如上方代码所示，通过环境变量`PROMPTS_ONLY`来对`单prompt`的训练模式进行开关控制，设为`1`则表示打开`单prompt`模式。

**比对详情**

对于训推一致的比对，除了`shape`之外，还存在较多的模块层级名称不一致，以`qwen2.5-0.5b`为例，若推理使用`vllm`后端、训练使用`fsdp`后端，双方对应的模块名称如下：  
<img src="https://raw.gitcode.com/user-images/assets/7898473/3d0f7c0d-0f33-4850-af39-ec9b83792dad/image.png" alt="Your image title" width="800"/>

可见存在如下差异：

1. 模块名称大量差异，不同前缀、不同类名。
2. `qkv_proj`双方存在拆分差异，推理为`qkv_proj`合并，训练则拆分为`q_proj`、`k_proj`、`v_proj`。
3. `rotary`位置差异，推理`rotary`放在每一层的`self_attn`内部，而训练则在进`decode`之前统一做`rotary`。
4. `silu`激活函数实现差异，推理使用`AscendSiluAndMul`融合计算`Silu`和`Mul`，而训练由于使用小算子计算未被`L0`层级采集到。

**表格比对工具适配特性**

原始比对工具在这类训推比对的情况下，会出现大量无法匹配的现象，为此，比对工具专门针对较常用的`Qwen`系列模型的训推一致比对进行了自动匹配，主要在于自动加入`mapping`列表、`qkv`合并等，在比对时加入超参`--consistent_check`，即可大幅加强比对成功率。

**单卡场景**

```bash
msprobe compare -tp /train_dump/dump.json -gp /infer_dump/dump.json --consistent_check --backend fsdp -o ./output
```

**多卡场景**

```bash
msprobe compare -tp /train_dump/step0 -gp /infer_dump/step0 --consistent_check --backend fsdp -o ./output
```

**可视化比对工具适配特性**

在可视化比对工具中，提供`点点匹配`功能，用户可通过浏览器界面，鼠标选择两个待匹配的灰色节点进行匹配。当前仅支持统计值数据模式。 
![image.png](https://raw.gitcode.com/user-images/assets/7898473/6d7ed4aa-17d6-4d4a-a045-22f5c2e3e160/image.png 'image.png')  

##### 2.3.3.6 长稳训练排查

按照[长稳训练Loss差异](#2313-长稳训练loss差异)进行排查。同时关注如下相关部分是否对齐：

- 学习率。
- 切分策略。
- Dtype精度差异。

### 2.4 定界到API后的处理方案

#### 2.4.1 单算子验证

对于以上分场景定位中排查到的可疑算子，需在`NPU`和标杆上进行单算子验证进一步确认该算子是否为问题`API`。单算子验证时通常将`CPU`当做绝对标杆，通过分别计算`NPU`与`CPU`、标杆与`CPU`之间的欧氏距离进行三方比对，来判断`NPU`与`CPU`的差距是否大于标杆与`CPU`的差距。

具体操作步骤如下：

1. 使用精度采集工具采集`NPU`上可疑算子计算过程中的具体`tensor`值。
2. 借助单算子`API`自动生成脚本工具生成单算子验证脚本。
3. 在`NPU`和标杆上用同一份`NPU`输入，分别运行单算子验证脚本，计算与`CPU`的欧氏距离。
4. 比较`NPU`与`CPU`的差距是否大于标杆与`CPU`的差距，若大于，则进一步验证该算子存在精度问题，可尝试下一节精度修复三板斧进行处理。

#### 2.4.2 精度修复三板斧

明确问题`API`后，并不一定代表该`API`对网络精度的异常起到决定性作用，因此需要通过一些手段进一步确定问题`API`对整体训练`Loss`的影响。在这种情况下，可尝试使用精度修复三板斧进行标杆等价替换（替换为同功能，无精度问题的实现），观测替换后`Loss`是否正常：

- 若替换后Loss结果正常，说明为根因问题。
- 若`Loss`有变好趋势，但仍不达标，说明该`API`对`Loss`有影响，但不是唯一因素。
- 若`Loss`无变化，说明该`API`对精度本身没产生影响。

精度修复三板斧主要分如下三种：

- 升精度  
 首先尝试升高`API`精度，如`FP16`、`BF16`升`FP32`，尝试规避半精度带来的精度问题。但需要注意有些`API`不适合升精度，如`random`噪声生成。
- 常规`API`转`CPU`  
 常规`API`指的是该`API`在`NPU`和`CPU`侧都有对应实现，因此可通过将该`API`替换到`CPU`侧运行，确保`API`自身计算没问题。

    ```python
    class ModuleOP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features=2, out_features=2) 
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x1 = self.linear(x) 
            r1 = self.relu(x1)
            return r1
    ```

    如上图模型中的`linear`层，替换到`CPU`运行的具体操作主要是在`forward`函数中将其输入替换到`CPU`上，根据`PyTorch`内部机制，就可将该`API`的前反向均在`CPU`侧运行。

     ```python
        # 替换self.linear_1到CPU运行 class ModuleOP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features=2, out_features=2) 
            self.relu = nn.ReLU()
    
        def forward(self, x):
            # 输入搬运到CPU, 如果有多处需要更改,则每处都要修改
            x = x.cpu()
            self.linear.cpu()
            x1 = self.linear(x)  # 在CPU上运行 
            x1 = x1.npu()  # 输出搬运回NPU
            r1 = self.relu(x1)
            return r1
     ```

- 小算子代替融合算子  
 若定位到问题`API`为融合算子，可通过分支变量控制模型训练走融合算子或常规实现。若融合算子场景精度异常，常规逻辑精度正常，则说明融合算子实现异常。如`MindSpeed-LLM`中关于`attention`常规实现和融合算子实现的逻辑控制可通过如下命令行参数控制。

    ```bash
    --use_fused_attn
    ```

若使用以上三种仍无法修复，请联系昇腾官方算子支撑人员：

- 对于外部用户或现场`PAE`、`FAE`来说，可联系昇腾相关支撑人员获取`API`责任田，找到对应责任人进行定位。
- 对于内部用户，可根据`API`名称确定对应算子（通常情况下，`API`名称与算子名称高度相似），据此寻找算子责任人进行定位。

## 3 定位流程对应案例

---

### 3.1 CheckList不一致案例

#### 3.1.1 配置项不一致

**案例**：某语音识别模型，从GPU迁移到NPU训练后，下游指标WER差异较大

**定位方法：**

根据启动脚本或训练日志对比NPU和标杆的训练配置。  
比对发现NPU用的fsdp配置，GPU用的ddp配置，训练Loss差异不大但下游指标差异大。  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/5b506a50-f3c3-406e-9c8e-a1e878709b64/image.png 'image.png')

**解决方案**：同步GPU配置。

**结果**：修复后WER下降，与GPU对齐。

#### 3.1.2 读取数据不一致

**案例**：某大语言模型，从LLama-Factory NPU（标杆）迁移到ModelLink NPU训练，Loss对不齐  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/7efdae1a-4765-4914-9a7e-ffb41378aa1d/image.png 'image.png')

**定位方法：**

打印比较输入的tokens等信息，具体位置需结合训练代码（如ModelLink可直接在modellink/pretrain_gpt.py的forward_step函数中添加打印），如图所示：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/c29037c7-896a-4bc8-baba-4a35a61f3bfb/image.png 'image.png')  
可以看到读取的tokens数据末尾存在不一致的问题。

**解决方案**：修复数据预处理代码，使其输入一致。

**结果**：修复后Loss对齐。  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/655f0a54-b851-49f5-a750-cd6cbf32a691/image.png 'image.png')

#### 3.1.3 模型结构不一致

**案例**：某MOE模型，从GPU迁移到NPU后，Loss对不齐  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/9cd35663-a6f7-4056-88a3-29051af73e60/image.png 'image.png')

**定位方法：**

可以通过查看具体代码实现或打印模型结构比较。  
查看代码发现，NPU中residual是input_layernorm后的，GPU上是input_layernorm前的，两者模型顺序结构不一致。  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/880aab30-4973-43fb-a0cb-8e2914a63299/image.png 'image.png')

**解决方案**：在NPU中的input_layernorm也放到residual后面。

**结果**：对齐模型结构后Loss对齐。  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/08c4b411-aabd-486b-abe9-49a1ded42b87/image.png 'image.png')

### 3.2 分场景定位案例

#### 3.2.1 稳定复现场景

##### 3.2.1.1 溢出或NaN问题

**案例**：某视觉模型从GPU迁移到NPU ModelLink训练，从一开始就梯度溢出  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/3127cf1c-086e-48d8-8092-7762c8f5e3f9/image.png 'image.png')  
从用户共享的训练截图中可以看到第0步梯度反向时逐层变大直至溢出。

**定位方法**：

1. 使用精度采集工具采集第0步（溢出步）的mix级别数据，config.json配置如下：

    ```json
    {
        "task": "statistics",
        "dump_path": "/home/data_dump",
        "rank": [],
        "step": [0],
        "level": "mix",
        "enable_dataloader": false,
        "statistics": {
            "scope": [], 
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "statistics"
        }
    }
    ```

    从训练截图中可看到每次self_attn反向之后梯度逐层变大，查看self_attn代码发现使用了FA算子，该算子历史上因使用规范引起的精度问题较多，优先查看dump中对应的反向数据，发现每次经过FA层反向后，norm值量级明显增大。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/6693c940-3bc0-452c-b25d-6c203fcf1aa8/image.png 'image.png')

2. 快速验证，先在ModelLink的训练配置中规避FA融合算子，删除如下超参：

    ```bash
    --use_fused_attn
    ```

    溢出消失，明确该问题为FA分支引入，但性能下降明显，需进一步明确FA精度原因。
3. 通过查阅FA算子官网使用文档，分析FA算子在代码中的具体使用方式：
    该问题为变长场景，原始输入为batch_size=2，输入序列1的seq_len=3577，序列2的seq_len=1502，统一pad到3577长度，原始输入shape=[2, 3577, 32, 128]。  
    在进行FA计算前，会将batch_size和seq_len做flatten，此时shape=[7154, 32, 128]，下一步去除其中的pad，因此Q和KV的输入长度变成了[5079, 32, 128]。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/a66e4909-82f0-437e-8fc4-f5b8d046d488/image.png 'image.png')  
    按照官网说明，此时attention mask按照规则本该为[maxSq, maxSkv] ，即[3577, 3577]，但实际客户代码中使用[query.shape[0], key.shape[0]，即 [5079, 5079]，使用规范错误，导致算子底层执行计算时会按行读取，导致出现0、1的数值错位，最终导致梯度溢出。

**解决方案**：修正FA训练时传入的attention_mask。

**结果**：训练梯度溢出消失，Loss正常收敛。

##### 3.2.1.2 首Step Loss不一致

**案例**：某语音模型首Step就Loss对不齐  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/64e451db-52a8-41b8-820c-b55c378769de/image.png 'image.png')

**定位方法**：

1. 使用精度采集工具采集第0步的mix级别数据，config.json配置参考[3.2.1.1节](#3211-溢出或nan问题)中的定位方法
 精度采集工具在代码中插入方法可参考如下方式：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/0434a5a0-46fe-45f0-98e9-a05884141ad8/image.png 'image.png')
2. 通过分级可视化工具分析差异   
    可视化命令为：

    ```bash
    msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
    ```

    在输出目录中可以看到生成的vis后缀文件，用tensorboard打开可视化界面：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/99186c98-1b3f-47b8-bb6d-68113895e3f9/image.png 'image.png')  
    可以看到gelu算子标红，算子精度可疑。
3. 也可通过精度比对工具进行比对  
    运行如下比对命令得到比对的csv表格：

    ```bash
    msprobe compare -tp /target_dump/step0 -gp /golden_dump/step0 -o ./output
    ```

    分析表格发现gelu算子输入差异较小，输出差异较大。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/0c44ccee-3715-4dfc-b3c0-660e1d85db73/image.png 'image.png')

**解决方案**：将gelu算子计算转CPU，精度问题解决，明确该问题为gelu算子导致，但转CPU会影响性能，后续联系算子支撑人员提供了torch-npu的gelu修复包。

**结果**：用修复包后不转CPU也精度达标。

##### 3.2.1.3 长稳训练Loss不一致

**案例**：某搜索模型从fp32转bf16之后，前期Loss差异不大后期Loss跑飞  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/52b9c64d-a386-4a3a-8e5e-dc001fae004b/image.png 'image.png')

**定位方法**：

由于前期对齐后期跑飞，且跑飞时步数已较大，全程dump数据量多且步数不确定，因此优先采用monitor状态监测工具进行采集。

1. 查看Grad Norm与Loss趋势：  
    两者趋势一致，倾向于由梯度导致的Loss突变，因此用如下配置采集梯度数据，monitor_config.json内容如下：

    ```json
    {
      "targets": {},
      "wg_distribution": true, 
      "format": "csv",
      "ops": [
        "norm",
        "mean",
        "min",
        "max"
      ]
    }
    ```

    代码中插入方式如下：  
     ![image.png](https://raw.gitcode.com/user-images/assets/7898473/f42a70bb-0abb-4ad4-b59e-c547c831d444/image.png 'image.png')
    
2. 采集后可得到每张卡上的grad_unreduced-xx-xx.csv和grad_reduced-xx-xx.csv  
其中xx为步数，查看在360步之后的开始上扬位置的reduce前各层梯度数据，结果如下：  
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/8b62626a-e35e-42c1-a2a2-178be0c0815c/image.png 'image.png')  
其中横坐标为反向的层顺序，左边为output，右边为embedding，可看到权重梯度norm值较大的位置在embedding附近，而对比fp32的梯度数据在embedding层上也相对稳定。  
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/581924af-bb80-4eaa-88d3-fb962423c878/image.png 'image.png')  
因此重点怀疑embedding层梯度在bf16上相较于fp32存在数值不稳定现象。

**解决方案**：对embedding的梯度做梯度裁剪。

**结果**：Loss收敛正常无跑飞。

#### 3.2.2 不稳定复现场景

##### 3.2.2.1 内存踩踏案例

**案例**：某多模态模型从GPU迁移到NPU后做微调，使用框架为fsdp，训练第2步Loss出现NaN  
NPU上运行结果：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/751daf11-9dba-4536-a5cc-1d1a5eb6c10e/image.png 'image.png')  
GPU上运行结果：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/b60f3448-308a-4421-95aa-068f4cae60fa/image.png 'image.png')

**定位方法**：  

1. 缩小规模：
 该模型现网为128卡训练，做实验成本大，需首先缩小规模，减少层数后可在单机2卡稳定复现
2. 使用精度采集工具采集Step1（最开始出现NaN的步数）的mix级别数据并分析：
    加入工具后，发现NaN问题消失。  
    去除工具并打开流同步进一步验证：

    ```bash
    export ASCEND_LAUNCH_BLOCKING=1
    ```

    开启流同步后问题也消失。  
    基于以上2个现象怀疑该fsdp模型训练存在内存踩踏问题。
3. 缩小排查范围：
    该模型由四个部分组成：vae，dit，denoiser和conditioner。  
    从训练完整模型改为只训练dit.transformer.layers，Loss仍然有NaN，确认是transformer.layers问题。
4. 通过手动挂局部hook的方式打印梯度：
    发现第1步Loss NaN不是第一现场，先出现NaN的是第0步post_attention_layernorm反向梯度。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/9e98aad8-0158-4171-894e-6e3bb0778138/image.png 'image.png')  
    与打开流同步的无NaN的梯度数据进行对比，除了input_layernorm和post_attention_layernorm层的weight和bias，其余的参数都能对上。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/2b958529-a687-42da-84c8-6f26a49979d4/image.png 'image.png')  
    对应dump中的接口为Functional.layer_norm.10和Functional.layer_norm.11。
5. 结合具体代码进行分析：
    post_attention_layernorm对于图像和文本连续下发了两次。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/e273c809-1669-4ce6-9ae1-2081b474b387/image.png 'image.png')  
    将其次数改为1次时NaN消失，明确该问题出现在该算子重复调用时。  
    分析内存踩踏特征的方式是按异常数据是否存在规律性和连续性，所以需先采集对应数据。  
6. 改用异步dump：
    之前加入精度采集工具后NaN消失原因为对tensor取统计量（min、max等）和落盘的操作会影响流上算子的执行，导致NaN不复现。  
    通过改异步dump方式，训练过程中工具不触发同步操作，在当前Step训练结束后统一落盘，降低对算子执行顺序和流同步影响。  
    具体操作为：在config.json文件中加入async_dump: True的配置项。  
    重新采集Functional.layer_norm.10和Functional.layer_norm.11及其中间的torch.split.192反向数据，可在dump单个算子时复现NaN。
7. 分析异步dump数据：
    参考无Loss NaN的dump.json文件，torch.split.192.backward的输入应为Functional.layer_norm.11的输出，而不开流同步时异步dump的torch.split.192.backward的输入与Functional.layer_norm.11的输出对不上，对比本该相等的2组数据：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/f4c2607e-d9ff-42fd-9801-2f7f5d89bfe7/image.png 'image.png')  
    发现刚好踩了size=2048（0-2047基本不等，2048-3071相等），满足内存踩踏特征。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/d70832b1-c199-4ca8-9233-48ab0da20540/image.png 'image.png')
8. 算子内存地址打印：
    尝试通过修改torch_npu源码对算子的输入和输出tensor对应的ptr地址和shape进行打印。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/62c1f075-9756-4022-a450-c275882e551a/image.png 'image.png')  
    从日志发现两个连续layernorm中，存在cast算子输出对concat算子输入的踩踏（两者地址一致）  
    踩踏现场确认如下：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/1c25de8b-129f-4762-951d-0248f8402591/image.png 'image.png')

总结根因为缺失record的backend +多流并行的FSDP +连续下发的layernorm导致了内存踩踏。

**解决方案**：在torch_npu2.3的FSDP unshard流上添加record，确保流上的当前算子执行完成之前，tensor内存不会被下一个算子申请。

**结果**：Loss NaN消失，正常收敛。

##### 3.2.2.2 算子确定性问题案例

**案例**：某视觉模型存在确定性计算问题，重复训练Loss不一致  

**定位方法**：

1. 打开确定性（计算确定性+通信确定性）、设随机种子、关闭Dropout、固定数据集读取顺序  
    通过msprobe工具中的seed_all来自动实现以上目的（除数据集读取之外）：

    ```bash
    from msprobe.pytorch import seed_all
    seed_all(mode=True)
    ```

    打开后精度比对结果有改善，部分卡结果一致，但仍有部分卡的反向存在差异，且重复训练每次对应的卡和位置随机不固定。
2. 使用精度采集工具采集第0步mix级别数据，设置"summary_mode"为"md5"以凸显微小差异, config.json配置如下:：

    ```json
    {
        "task": "statistics",
        "dump_path": "/home/data_dump",
        "rank": [],
        "step": [0],
        "level": "mix",
        "enable_dataloader": false,
        "statistics": {
            "scope": [], 
            "list": [],
            "data_mode": ["all"],
            "summary_mode": "md5"
        }
    }
    ```

    比对2次采集的数据，最先出现异常的是masked_fill.23的输入。  
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/23f89bbd-8f60-44ca-a777-575602508164/image.png 'image.png')  
    根据dump结果中的stack.json调用栈和代码查找输入来源，代码往上翻找为mmcv的MSDA。  
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/46e01b68-e868-4574-9f06-91e29abba68c/image.png 'image.png')  
    与MSDA算子支撑人员确认该算子暂不支持确定性计算，建议可用小算子组合进行代替。  
    小算子代替后masked_fill.23输入一致，但发现grid_sample输出仍有差异。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/7f726fa8-2cce-4a44-8789-20a17071775c/image.png 'image.png')  
 与grid_sample算子支撑人员确认该算子也暂不支持确定性计算。

**解决方案**：MSDA算子转小算子拼接，grid_sample算子转CPU。

**结果**：随机性固定，重复训练结果完全一致。

##### 3.2.2.3 硬件压测案例

**案例**：某近5k卡大集群模型Loss不对齐，Grad Norm存在大量尖刺  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/8562b38b-a85e-4b44-99de-89e70595c56b/image.png 'image.png')  
由于集群较大，优先进行硬件压测，排查坏节点。  
4800卡分成100组*（3*16卡）任务，跑同一个训练任务，固定随机性+开启确定性计算，看最终Loss曲线，有没有哪组异常，缩小到异常的机组再做dmi压测。  
使用ascend-dmi -dg -i aicore -s -sc 60 -q命令进行机器压测，查看故障检测结果。  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/c86efd04-6b4d-414c-99b3-f8c4bffdbc31/image.png 'image.png')  
检测结果显示存在坏节点，将其排除后精度正常，表现为Loss后期不再有尖刺、Grad Norm尖刺频率明显改善。
![image.png](https://raw.gitcode.com/user-images/assets/7898473/91e3f75b-3c9b-4b25-86e4-789a76790ae0/image.png 'image.png')

## 4 msprobe工具定位

---
本章介绍msprobe工具包的安装、使用思路及步骤说明，也可同步参考[msprobe官网](https://gitcode.com/Ascend/msprobe)查看最新版本说明文档。

### 4.1 工具安装

msprobe工具包安装请参考[msProbe工具安装指南](../install_guide/msprobe_install_guide.md)

### 4.2 工具概述

下表是msprobe工具包所含的主要功能模块列表和简单介绍。

|工具集| 基本原理|
| --- | --- |
|精度采集工具<br>简称dump工具| 采集API、Module或混合级别的前反向输入输出，同步保存相对应的调用栈|
|训练状态监测工具<br>简称monitor工具| 可自定义选择采集激活值、权重梯度、优化器状态和通信算子的中间值，轻量化保存训练状态|
|分级可视化工具| 可将dump工具采集的精度数据进行解析，还原模型图结构，实现模型各层级精度比对，方便用户理解模型结构和分析精度问题|
|精度比对工具| 可将dump工具在NPU和标杆上采集的精度数据进行各种维度评测指标的精度比对|
|趋势可视化工具|  针对dump/monitor采集的数据，进行全局分卡、分层、分步数的趋势可视化分析|
|精度预检工具| 通过采样模型训练中使用的算子API和输入shape, dtype及数值分布或者真实输入数据，伪造输入张量或者使用真实输入数据在GPU和NPU分别运行并比对结果|

msprobe工具包内的工具分为数据采集和数据比对两大类，整体使用逻辑如下：

- 数据采集：
  - 精度采集工具，适用于问题第一现场步数确定的场景。
      1) 先采集统计量：
              - 模型规模小，直接采集mix（API级+模块级）统计量。
              - 模型规模大，先采集模块级统计量，定位到模块后再内部采集API级统计量。
      2) 在统计量上分析到可疑算子后，采集具体tensor值进行进一步分析。
  - 训练状态监测工具，适合大规模、不确定采集dump步数的场景。
- 数据分析：
  - 分级可视化工具，对dump保存的文件进行解析，还原模型图结构，实现模型各个层级的精度数据比对或溢出分析。
  - 精度比对工具，比对dump保存的文件并计算各种误差指标。
  - 趋势可视化工具， 针对dump或monitor保存的数据进行全局分卡、分层、分步数的趋势分析。
  - 精度预检工具，通过dump保存的文件中算子API的 shape、值域等信息构造一致的输入数据并在NPU、标杆上与CPU进行三方比对。

### 4.3 精度采集工具

对于一些问题第一现场明确的问题，可以优先采用精度采集工具进行训练数据采集。msprobe工具可在训练脚本内添加 dump 接口挂hook，并在启动训练时采集API或模块级的前反向输入输出数据的统计值或具体tensor值。

**使用说明**

1. 使用精度采集工具需要先配置config.json文件
    对于采集统计值来说，最常用的两种配置如下：
    - 采集指定步的统计量

      ```json
      {
          "task": "statistics",
          "dump_path": "/home/data_dump",
          "rank": [],
          "step": [0,1],
          "level": "mix",
          "statistics": {
              "scope": [], 
              "list": [],
              "data_mode": ["all"],
              "summary_mode": "statistics"
          }
      }
      ```

    - 采集指定步的tensor值

      ```json
      {
          "task": "tensor",
          "dump_path": "/home/data_dump",
          "rank": [],
          "step": [0,1],
          "level": "mix",
          "tensor": {
              "scope": [],
              "list":[],
             "data_mode": ["all"]
          }
      }
      ```

    此外，在这两种配置基础上常见的几种修改为：
    - 采集级别：修改level采集"mix"（API+模块级）、"L0"（模块级）、"L1"（API）。
    - 统计量+md5：修改统计量中的"summary_mode"为 "md5"。
    - 指定采集步数（或卡号）：修改"step"（或"rank"），[]代表采集所有，内有数值代表采集该步，多步用英文逗号分割。
    - 筛选目标API，具体有如下两种方式：
        - 修改"list"属性，添加时会采集名称包含该字符串的所有API或模块，多个采集对象用英文逗号分隔，若字符串为模块类型则展开内部API进行同步采集。
        - 修改"scope"属性，添加开始和结束的API或模块，可采集两者之间的API或模块。
2. 代码中插入方式

    ```python
    from msprobe.pytorch import PrecisionDebugger
    ...
    debugger = PrecisionDebugger(config_path="./config.json")
    # 训练步数循环体
    for step in …:
        debugger.start(model=model)
        output = model(data) # 模块forward的地方，也可能名为train_step(…)
        # loss计算等操作…
        loss.backward()
        debugger.stop()
        debugger.step()
    ```

    debugger插入说明如下：
    - 采集前统一建议使用seed_all()固定随机性，必要时加入mode=True参数打开确定性，放置位置越靠前越好。
    - PrecisionDebugger初始化位置放在训练开始前的位置，勿放入循环代码中重复定义。
    - debugger.start()放在要抓取的操作forward前的位置，代码里一般放是在调用model.forward()前或者train_step()前， 若想要采集"L0"或"mix"级别数据，须传入model。
    - debugger.stop()需放在loss.backward之后的位置，代码里一般是放在调用loss.backward()后或者train_step()后。
    - debugger.step()需放在一个迭代结束的位置，且必须在stop函数之后的位置调用。

3. 保存格式  
    结果保存在config.json中配置的dump_path下，整体格式如下：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/239c76c4-241e-442f-ab09-0f00c61285a5/image.png 'image.png')  
    对于保存的结果，需注意如下几点：
    - dump.json为统计量，一般包含Max、Min、Mean、Norm等。
    - stack.json可查看dump中API对应的调用栈。
    - construct.json为模型结构文件。
    - dump_tensor_data下存具体tensor值。
    - 只有采集"task"为"tensor"时， dump_tensor_data下才会有内容。
    - 只有采集"level"为"L0"和"mix"时，construct.json内才会有内容。

4. dump.json统计量结果详解  
    如下为dump.json采集结果样例：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/201776cc-0ca5-4f30-88e3-464e35df6ddf/image.png 'image.png')  
    可以看到，此时采集了linear层统计量，主要包含内容：
    - 输入input_args：含3个值，从上到下对应input、weight和bias，其中bias为空所以值为null。
    - 输出output：含1个值，对应linear层计算结果。

**使用思路**

下图是一个典型的“前几个Step的Loss与标杆差异大”的问题场景，第0个Step Loss完全一致，第一个Step Loss差异增大数倍。因而我们可以推断问题大概发生于Step0的backward或Step1的forward。  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/a6f1d7c5-8861-4178-8b72-c3f545158ed4/image.png 'image.png')  

1. 优先统计量采集：

    - 若该模型规模较小，如单机16卡训练，我们可以直接使用精度采集工具采集第0步和第1步的mix级别的统计量，再结合比对工具和分级可视化工具进行分析定位可疑算子。
    - 若该模型规模较大，可优先考虑是否能缩小规模，若不能可按照如下顺序来进行定位：
        1) 优先进行精度采集工具采集第0步和第1步的L0级别统计量，先定位到可疑模块。
        2) 针对可疑模块，在config.json中使用list参数限定采集模块。
        3) 采集限定模块内的L1级别统计量，定位到可疑算子。

    除此之外，若对于一些随机性固定的问题，在采集统计量时可进一步设置"summary_mode"为"md5"，能够更精确地展示出微小的差异。

2. 进一步采集tensor分析：
    在已经缩小怀疑范围或者已找到最终的可疑算子之后，可通过直接指定list参数采集对应算子的tensor数据来进行进一步的单算子分析。

### 4.4 分级可视化工具

对于精度采集工具在NPU和标杆上采集保存的dump数据，可通过分级可视化工具进行画图比对，能够更清晰的还原图结构，实现模型各个层级的精度数据比对，方便用户理解模型结构，分析精度问题。

**使用说明**

1. 构图：
    - 图构建（单步单卡精确到rank、单步多卡精确到step、多步精确到dump_path）

      ```bash
      msprobe graph_visualize -tp ./target_path -o ./output_path
      ```

    - 图比对

      ```bash
      msprobe graph_visualize -tp ./target_path -gp ./golden_path -o ./output_path
      ```

      构图完成会生成.vis.db后缀的文件，文件名称基于时间戳自动生成，格式为：build_{timestamp}.vis.db。
  
2. 可视化：  
    用如下所示进行可视化展示
    - 可直连服务器

      ```bash
      tensorboard --logdir out_path --bind_all
      ```

    - 不可直连服务器使用vscode远程连接

      ```bash
      tensorboard --logdir out_path
      ```

    启动后会打印地址和端口号。  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/6d5e914f-4841-4486-83a2-40a3f20e6bff/image.png 'image.png')

3. 浏览器查看并分析：
浏览器输入地址+端口号，打开可视化界面。

**使用思路**

注意：分级可视化工具只有在精度采集工具采用了L0或mix级别时才可以使用，即必须在dump结果中生成construct.json模型结构文件。  
分级可视化工具构图后比对时可按照如下方法进行比对：

- 比对模型结构：  
  在左边侧栏勾选灰色的无匹配节点， 选取后会出现所有节点匹配不上的列表，点击节点查看网络中具体信息。  
  ![image.png](../figures/visualization/vis_unmatch_info.png 'image.png')  
  点击缺失节点可展开堆栈和输入输出信息，可根据堆栈找到对应代码：
  - 若迁移后确实存在部分模型步骤缺失，则进行补齐。
  - 若仅为模块名称命名差异等导致，可点击左侧点点匹配按钮进行手动匹配 
  ![image.png](https://raw.gitcode.com/user-images/assets/7898473/49b8538e-3047-424d-a7d8-395cc75dc66d/image.png 'image.png')  
- 比对节点精度  
 左边侧栏除了选择无匹配节点外，还可以根据精度风险级别勾选高风险提示节点，可视化后颜色越深，精度比对差异越大，越可疑。
  除按照颜色深浅判断分析节点的优先级外，也可以优先查看首个精度出现差异的节点。  
  ![image.png](../figures/visualization/vis_precision_info.png 'image.png')
- 对于通信算子问题，可右键点击节点，在弹出菜单中选择数据发送或接收查看其他卡数据：  
 ![image.png](https://raw.gitcode.com/user-images/assets/7898473/958e5083-6920-4d89-83cd-bc09c73c3c39/image.png 'image.png')  
- 溢出分析  
 可在左边侧边栏进行溢出等级筛选，将特定等级的溢出检测节点按照顺序筛选出来，从颜色深的开始排查，用户可点击对应筛选项跳转到对应节点。  
    ![image.png](../figures/visualization/vis_overflow_check.png 'image.png')  
    溢出等级说明如下：
    - medium： 输入异常，输出正常，这类问题优先级低，用户需最后关注。
    - high：输入异常，输出异常，或者是输入输出的指标突增，数值scale超过阈值倍数，这类问题需要用户手动确认是否存在问题。
    - critical：输入正常，输出异常，这类问题需要用户首先关注。

除此之外，更详细的页面示意图使用可参考下图：
 ![image.png](../figures/visualization/vis_show_info.png 'image.png')  

### 4.5 精度比对工具

对于已经通过精度采集工具在NPU和标杆上采集保存的统计量或tensor的数据，除可视化工具之外，也可以使用精度采集比对工具进行各评测指标的精度比对。
**使用说明**

1. 比对命令：

    ```bash
    msprobe compare -tp /target_dump/step0 -gp /golden_dump/step0 -o ./output
    ```

2. 保存格式：
    比对后保存两类文件：
    - advisor_{timestamp}.txt 文件，给出可能存在精度问题的 API 的专家建议。
    - compare_result_{timestamp}.xlsx 文件，列出所有执行精度比对的 API 详细信息和比对结果，具体结果分析参照下文使用思路。

**使用思路**

对于使用精度采集工具分别在NPU和标杆上采集的dump数据，使用比对工具进行比对，将生成csv比对结果文件。  
对于非md5的统计量比对结果如下形式：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/a893f5a1-879b-46ea-a3e2-64d44aa6ace8/image.png 'image.png')  
比对结果通过颜色标记、比对结果以及具体的各比对指标下的精度数值，在分析结果时可以着重看**平均相对误差（MeanRelativeErr）列**，对于输入差异不大，输出差异大的算子需进行重点分析。  
对于md5的统计量比对结果如下形式：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/90f090b5-6eed-4f5e-b8ba-57764fe467bd/image.png 'image.png')  
比对结果可以直接筛选**Result列**，表示md5比对Pass或Different，对于结果为Different的算子需进行重点分析。  
对于tensor的比对结果如下形式：  
![image.png](https://raw.gitcode.com/user-images/assets/7898473/daf75315-e30b-43d6-adda-6910b89f8967/image.png 'image.png')  
比对结果通过颜色标记、比对结果以及具体的各比对指标下的精度数值，在分析结果时可以着重看**双千指标（One Thousandth Err Ratio和Five Thousandth Err Ratio）列**，对于输入差异不大，输出差异大的算子需进行重点分析。

### 4.6 训练状态监测工具

对于训练规模较大、问题现场步数不明确的精度问题，若采集dump数据落盘量过大，可以先通过轻量化的训练状态监测工具进行定位排查。  
训练状态轻量化监测工具，能够在较低性能损耗下收集和记录模型训练过程中的激活值、权重梯度、优化器状态和通信算子的中间值，实时呈现训练状态，同时该工具支持动态启停功能，能够在训练过程中随时重启监测同时修改监测目标及配置。

**使用说明**

1. 使用monitor监测工具需要先配置config.json文件：  
    对于训练监测来说，最常用的配置（监测权重梯度）如下：

    ```json
    {
      "targets": {},
      "wg_distribution": true, 
      "format": "csv",
      "ops": ["norm", "mean", "max", "min"],
      "ndigits": 16
    }
    ```

    除以上最常用的配置之外，还可支持如下几种常见修改：
    - 采集类别：还支持激活值梯度"xy_distribution"、优化器状态"mv_distribution"和通信信息"cc_distribution"等信息采集。
    - 开始采集步数：加入"start_step"指定开始采集步数
    - 采集次数：加入"collect_times"指定采集步数，默认为100000000，目的是一直采集。
    - 打印模型结构：改"print_struct"为True，将在第1步打印结构后自动退出。
    - 筛选采集模块：在"targets"中配置指定模块，具体格式可结合"print_struct"结果进行填写。
    - 落盘格式：常用可选"csv"和"tensorboard"。
    - 动态启停：除通过 "start_step"采集开始步数之外，对于一些大规模或问题现象不确定的精度问题还可以直接打开动态启停进行动态监测，设置环境变量`export DYNAMIC_MONITOR=True`，进入动态启停模式，进入后可随时通过修改config.json文件中的dynamic_on开关为True启动最新配置（支持中途修改）的监测。

2. 代码中插入方式（以Megatron core_r0.6.0为例，在training.py的pretrain函数中）：

    ```python
    model, optimizer, _ = setup_model_and_optimizer(model_provider, type)
    # 使能工具
    from msprobe.pytorch import TrainerMon, seed_all
    seed_all(mode=True) 
    # 监测工具初始化
    monitor = TrainerMon(
            config_file_path="./monitor_config.json",
            process_group=mpu.get_pipeline_model_parallel_group(),
            params_have_main_grad=True  # megatron=True，deepspeed=False
       )
       # 挂载监测对象
       monitor.set_monitor(
            model[0],
            grad_acc_steps=args.global_batch_size // args.data_parallel_size // args.micro_batch_size,
            optimizer=optimizer
        )
    ```

3. 保存格式：  
    保存路径通过环境变量MONITOR_OUTPUT_DIR设置，默认为"monitor_output"，各类数据对应的文件名如下，其中"xx"为对应步数：
    - 激活值对应actv_xx-xx.csv。
    - 激活值梯度对应actv_grad_xx-xx.csv。
    - 权重梯度reduce前对应grad_unreduced_xx-xx.csv。
    - 权重梯度reduce后对应grad_reduced_xx-xx.csv。
    - 优化器状态对应exp_avg_xx-xx.csv。
    - 权重对应param_xx-xx.csv。  
    同时采集以上几种状态的结果样例如下：  
    ![image.png](https://raw.gitcode.com/user-images/assets/7898473/9ae12b93-113f-4ff9-a6a8-6ec05bbaae84/image.png 'image.png')

**使用思路**

使用训练状态监测工具，在配置选择上根据Loss和Grad Norm表现来进行相对应数据的采集：

- 若该问题表现为先Grad Norm后Loss，则优先采集训练过程中梯度的数据，对应在config.json中的配置为wg_distribution。
- 若该问题同时主要表现在Loss上，则优先采集训练过程中激活值和权重的数据，对应在config.json中的配置为xy_distribution和param_distribution。

值得注意的是，在统计选项中必须包含mean值，以供后续分析。  
对于梯度数据可以从如下几方面进行分析：

1. 查看梯度unreduced和reduced前后差异：
    - 明确哪张卡存在问题，如查看NaN数据存在于哪张卡。
    - 复现reduce计算过程，查看reduce的均值结果是否符合预期，如对于通信顺序导致的一些差异。
2. 查看模型中各层的梯度分布：
    - 关注梯度与标杆出现明显差异的步数或层。
    - 关注NaN问题中Grad Norm值异常大的层。
    - 关注梯度随步数变化趋势与整体梯度趋势（特别是尖刺部分）一致的层。

对于一些千卡以上集群且问题现场尚不明确的任务，哪怕使用训练状态监测工具，落盘量也较大，可以使用动态启停来进行控制，即一开始先不监测，等观测到训练出现异常后再及时重启监测，可以抓取第一现场附近的信息，同时建议此步骤最好可以结合一些异常回滚机制，方便直接抓取第一现场。

### 4.7 趋势可视化工具

趋势可视化工具适用于采集数据规模较大，如卡数/步数/层数较多时，传统数据分析手段容易陷入局部陷阱，而趋势可视化提供了一个全局、分级的视角辅助用户进行判断。

### 4.8 脚本比对工具

首先准备两个用来训练的环境，使用工具采集两个环境下影响精度的配置并比对。

1. 数据采集。在其中两个环境分别执行如下操作：  
      在训练脚本开始处插入如下代码：

      ```python
      from msprobe.core.config_check import ConfigChecker
      ConfigChecker.apply_patches(fmk)
      ```

      在模型初始化后插入如下代码：

      ```python
      from msprobe.core.config_check import ConfigChecker
      ConfigChecker(model, shell_path, output_zip_path, fmk)
      ```

      说明： 
      - model：初始化好的模型。不传或缺省就不会采集权重和数据集
      - shell_path：训练脚本路径，类型为列表，传入一个或多个训练配置/启动脚本。不传或缺省就不会采集超参
      - output_zip_path：输出zip包的路径，不传默认为"./config_check_pack.zip"
      - fmk: 可选参数，训练框架，可选"pytorch”或“mindspore"，默认未配置，表示传入"pytorch"

      对比三方库版本（通过git安装的库除外），可以分别在NPU和标杆上设"pip data"为true进行信息采集。采集完成后会得到一个zip包，里面包括各项影响精度的配置。
2. 将两个zip包传到同一个环境下，使用如下命令进行比对：

      ```bash
      msprobe config_check -c bench_zip_path cmp_zip_path [-o output_path]
      ```

      其中bench_zip_path为标杆侧采集到的数据，cmp_zip_path为待对比侧采集到的数据，参数-o为可选的比对结果输出路径，默认为"./config_check_result"。  
      结果内含2个目录和1个文件：  
      - bench：bench_zip_path里打包的数据
      - cmp：cmp_zip_path里打包的数据
      - result.xlsx：比对结果。里面会有多个sheet页，其中summary总览通过情况，其余页是具体检查项的详情。  

      精度比对具体执行的检查包含：环境变量、三方库版本、训练超参、权重、数据集。

### 4.9 非工具手段补充

在某些情况下（如使用工具报错、使用工具后问题不复现、使用工具采集较慢），可以尝试进行手动挂hook。  
此方法基于PyTorch原生hook，可以获取到整网block粒度的数据。其在功能上相较于精度工具更加轻量，且可以自由的加入各种判断，非常灵活。但是无法获取到通信前后的数据，需要找到相关位置才能进行输出。  
手动挂hook的样例代码如下：

```python
def print_func(inputs, prefix):
      if isinstance(inputs, tuple) or isinstance(inputs, list):
          for i in inputs:
              print_func(i, prefix)
      elif isinstance(inputs, torch.Tensor):
          print(prefix, inputs.max())
      else:
          print(prefix, inputs)
def hook_func(name, module):
      def hook_function(module, inputs, outputs):
          print(module)
          print_func(inputs, name + ' inputs')
          print_func(outputs, name + ' outputs')

      return hook_function
for name, module in model.named_modules():
      if module is not None:
         module.register_backward_hook(hook_func('[forward]:' + name, module))
```

## 5 附录

---

### 5.1 根因介绍

#### 5.1.1 算子

算子精度异常，绝大部分情况是因为算子设计本身存在bug。一般发生在计算的前反向。

- 算子数值计算错误：因为计算逻辑等方面存在bug，从而导致算子数值计算错误。
- 算子内存踩踏：这种情况下指因为算子计算的内存问题导致的精度错误。例如分核计算和分ub计算，出现非整块，计算复杂，数据类型变化时，容易出现此类问题。框架分配内存也有概率出现问题，例如分配地址不对齐等。算子越位读取也是较为常见的情况。
- 算子内部计算未同步：这会导致计算数据读取到脏数据，多见于融合算子。
- 算子实现和GPU存在差异：因为有些算子在不同的PyTorch版本下，适配层实现有差异。
- 算子缓存读取逻辑异常：某些算子为了性能加速会使用缓存逻辑，而在特定场景下缓存读取可能存在bug。
- 算子确定性：确定性计算是NPU的一套机制，用于保证算子的计算确定性。之所以要有这个机制，是为了在debug过程中，让所有的算子计算结果前后完全一致可复现。

#### 5.1.2 通信

通信操作一般来讲，不会出现因为计算导致的精度问题(all reduce/reduce scatter存在加法计算)。最常见的问题就是在通信途中因为内存踩踏，数据传输格式不一致，通信链路存在问题或者通信缺乏收发的保护机制引起的bug。通常的表现为，通信数据前后的数据不一致。

- 内存踩踏：一般常见于计算流和通信流并行时，两者同时踩到同一块内存，从而导致数据异常。
- 数据传输格式不一致：在聚合类算子如allgather，如果在聚合过程中出现了数据格式不一样的情况，也会导致精度问题。
- 通信链路：例如pcie switch出现问题，导致的精度异常。此类问题很少见，一般和硬件相关，很难排查。
- 通信缺乏收发保护机制：此类场景比较常见，一般是因为通信时没有做保护，导致通信没有结束，就有后续操作读取或者修改了特定内存。

#### 5.1.3 业务代码

在实际定位过程中，有大量的精度问题最后定位原因为迁移后环境变量、超参数、数据处理、权重转换、模型实现或评测方案出现问题，这类问题通常建议在定位前优先排查。  
这个场景下引入一个难点：无法脱离GPU基线，在脱离GPU的情况下，目前没有一个完整的流程或者解决方案来自证清白。

#### 5.1.4 训练框架

对于迁移场景，很多都会涉及到训练框架的切换，如从纯GPU的Megatron切换为Mindspeed+Megatron或Modellink+Megatron，在此过程中引入了训练框架，因此若该框架本身存在bug，也会导致训练出现精度问题。

#### 5.1.5 硬件

硬件引起的精度很少见，但是是最难定位的。如果某个精度问题，可以排查到跟某个机器强绑定，并且能稳定复现，目前是可以较快排查到的。但是如果不能稳定复现，并且不跟特定机器绑定就难以定位。

1. 多bit翻转：此情况根因主要是因为硬件存在电压不稳定的问题，从而导致数值不稳定。目前没有相关的dfx能检测到这个情况。唯一的排查方法只能对aicore进行重复多次的压测，才能排查出来。
2. 电源故障：电源故障所引发的故障也是多个局点遇到的问题，本质上是电源不稳定导致的数值异常。此问题在IBMC的日志存在明显的报错，较好排查。

### 5.2 模型超参数

![image.png](https://raw.gitcode.com/user-images/assets/7898473/e4b647f4-0f5d-4864-9ed7-c15645579147/image.png 'image.png')  
如上图所示，模型的超参通常可调整的主要有学习率，batch_size、并行切分策略、模型参数、融合算子配置等，用户在进行NPU精度和GPU精度比对前，需要保证两边的配置一致。

1. 学习率和warm-up：不同的学习率调度器（决定什么阶段用多大的学习率）有不同的学习率调度相关超参，例如线性调度可以选择从一个初始学习率lr-warmup-init开始预热。可以选择多少比例的训练迭代步使用预热阶段的学习率。不同的训练框架有不同的参数命名，请结合代码实现设置对应的参数，如Modellink对应的学习率参数为：
    - lr
    - min-lr
    - lr-warmup-init 1e-8
    - lr-warmup-fraction 0.01
2. batch_size：batch_size会影响训练速度，有时候也会影响模型精度：
    - micro-batch-size：每个设备上处理的批次大小。
    - global-batch-size：完整一次梯度更新需要的batch_size大小
3. 切分策略：DP、TP、PP、EP、CP：
    - DP（data parallel）：数据并行（data parallelism）是大规模深度学习训练中常用的并行模式，它会在每个进程（设备）或模型并行组中维护完整的模型和参数，并在每个进程上或模型并行组中处理不同的数据。因此，数据并行非常适合大数据量的训练任务。
    - TP（tensor parallel）：张量并行也叫层内并行，通过将网络中的权重切分到不同的设备，从而降低单个设备的显存消耗，使得超大规模模型训练成为可能。张量并行不会增加设备等待时间，除了通信代价外，没有额外代价。
    - PP（pipeline parallel）：流水线并行将模型的不同层放置到不同的计算设备，降低单个计算设备的显存消耗，从而实现超大规模模型训练。流水线并行也叫层间并行，层输入输出的依赖性使得设备需要等待前一步的输出，通过batch进一步切分成微batch，网络层在多个设备上的特殊安排和巧妙的前向后向计算调度，可以最大程度减小设备等待（计算空泡），从而提高训练效率。 
    - EP（expert parallel）：专家并行在混合专家模型(MOE)中对不同的专家放置到不同的计算设备，使每个专家网络可以独立地学习和处理输入数据的不同方面。增加整个混合专家模型的扩展性，提高计算效率和泛化能力，在大规模MOE模型中备受关注。
    - CP（context parallel）：上下文并行将序列维度切分数据，实现支持序列并行的attention层，对计算设备实现负载均衡。在长序列数据训练任务中，上下文并行切分策略可以有效降低等待时间，提升吞吐率，是处理大规模数据集和复杂模型场景下的有效手段。
4. 模型结构，配置模型结构的超参主要有：
    - num-layer
    - hidden-size
    - seq-length
    - ffn-hidden-size
    - num-attention-heads
5. 融合算子配置：
    - use-flash-attn：FA融合算子开关，需重点关注
    - use-fused-swiglu
    - use-fused-rmsnorm
    - use-fused-rotary-pos-emb

### 5.3 非饱和模式

Inf/NaN模式，又称为非饱和模式，开启时计算结果如果溢出了数值就一定会变成Inf或NaN，否则不一定。英伟达默认开启，强烈推荐在NPU环境中开启，与标杆保持一致并使一些溢出问题更容易暴露出来。开启方式为启动训练的shell脚本里配置如下环境变量：

```bash
export INF_NAN_MODE_ENABLE=1
```

若不确定是否开启了这个模式，可以检查自己的shell脚本里是否存在该环境变量，或者在训练python脚本里打印该环境变量检查，如：

```python
import os 
inf_nan_mode = os.environ.get("INF_NAN_MODE_ENABLE", False) 
print(f"******INF_NAN_MODE is {inf_nan_mode}******")
```

注：部分产品没有实现这一模式，具体请以实际情况为准。
