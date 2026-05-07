# PyTorch 场景的精度数据采集基线

## "statistics"模式采集时间膨胀参考基线

该基线为PyTorch框架下，使用"statistics"模式且risk_level为ALL采集数据性能膨胀的参考基线。本基线测试了16层Llama 2大模型在不同采集模式4卡下的时间膨胀。
使用Mindspeed-llm框架运行，核心配置为seq-length为8192，mbs为2，gbs为32。

| 采集模式 | 无工具 (耗时) |   加工具并使能 Dump (耗时)    | 加工具并使能 Md5 Dump (耗时)  |
|:----:|:--------:|:---------------------:|:---------------------:|
|  L0  | ≈6406 ms |  ≈10356 ms  （膨胀1.6倍）  |  ≈36492 ms  （膨胀5.7倍）  |
|  L1  | ≈6406 ms  |  ≈30890 ms  （膨胀4.8倍）  |  ≈122285 ms  （膨胀19倍）  |
| mix  | ≈6406 ms  |  ≈35986 ms  （膨胀5.6倍）  |  ≈155236 ms  （膨胀24倍）  |

## "tensor"模式采集数据量参考基线

该基线为PyTorch框架下，使用"tensor"模式采集数据量参考基线。本基线测试了两个模型，分别为LLAMA2-7B和LLAMA2-13B，测试了不同采集模式下，不同global_batch_size下，单卡和8卡下，数据量的变化。

### LLAMA2-7B

<table> 
   <tbody>
    <tr>
     <th>采集模式</th>
     <th>global_batch_size</th>
     <th>单卡</th>
     <th>8卡</th>
    </tr> 
    <tr>
     <td rowspan="3">L0</td>
     <td>1</td>
     <td>7.8GB</td>
     <td>63GB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>16GB</td>
     <td>125GB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>24GB</td>
     <td>187GB</td>
    </tr> 
    <tr>
     <td rowspan="3">L1</td>
     <td>1</td>
     <td>300.8GB</td>
     <td>2.3TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>480GB</td>
     <td>3.6TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>640GB</td>
     <td>4.9TB</td>
    </tr> 
    <tr>
     <td rowspan="3">mix</td>
     <td>1</td>
     <td>313.6GB</td>
     <td>2.4TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>512GB</td>
     <td>3.8TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>672GB</td>
     <td>5.1TB</td>
    </tr> 
   </tbody>
</table>

### LLAMA2-13B

<table> 
   <tbody>
    <tr>
     <th>采集模式</th>
     <th>global_batch_size</th>
     <th>单卡</th>
     <th>8卡</th>
    </tr> 
    <tr>
     <td rowspan="3">L0</td>
     <td>1</td>
     <td>13GB</td>
     <td>97GB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>25GB</td>
     <td>194GB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>37GB</td>
     <td>291GB</td>
    </tr> 
    <tr>
     <td rowspan="3">L1</td>
     <td>1</td>
     <td>440GB</td>
     <td>3.4TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>720GB</td>
     <td>5.4TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>960GB</td>
     <td>7.3TB</td>
    </tr> 
    <tr>
     <td rowspan="3">mix</td>
     <td>1</td>
     <td>480GB</td>
     <td>3.6TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>720GB</td>
     <td>5.6TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>1000GB</td>
     <td>7.7TB</td>
    </tr> 
   </tbody>
</table>
