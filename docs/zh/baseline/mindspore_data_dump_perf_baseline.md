# MindSpore 场景的精度数据采集基线

## "statistics"模式（未开启md5）采集**时间**膨胀参考基线

该基线为MindSpore框架下，使用"statistics"模式采集数据性能膨胀的参考基线。测试了38B语言大模型在不同采集模式下，使用8卡时的性能膨胀。

| 采集模式 | 无工具（耗时） | 加工具但未使能 Dump（耗时） | 加工具并使能 Dump（耗时） |
| :------: | :------------: | :-------------------------: | :-----------------------: |
|    L0    |    ≈340 ms     |     ≈340 ms （无膨胀）      |   ≈1.2 s  （膨胀3.5倍）   |
|    L1    |    ≈340 ms     |  ≈0.7–1.2 s  （膨胀2~4倍）  |   ≈3.8 s   （膨胀11倍）   |
|   mix    |    ≈340 ms     |  ≈0.7–1.2 s  （膨胀2~4倍）  |   ≈5.5 s  （膨胀16倍）    |

## "tensor"模式采集**数据量**参考基线

该基线为MindSpore框架下，使用"tensor"模式采集数据量参考基线。本基线测试了38B语言大模型在不同采集模式下，不同global_batch_size下，单卡和8卡下，数据量的变化。

### 38B语言大模型

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
     <td>262GB</td>
     <td>2.1TB</td>
    </tr>
    <tr>
     <td>2</td>
     <td>480GB</td>
     <td>3.8TB</td>
    </tr>
    <tr>
     <td>3</td>
     <td>928GB</td>
     <td>7.4TB</td>
    </tr>
    <tr>
     <td rowspan="3">L1</td>
     <td>1</td>
     <td>2.1TB</td>
     <td>17.1TB</td>
    </tr>
    <tr>
     <td>2</td>
     <td>2.8TB</td>
     <td>22.7TB</td>
    </tr>
    <tr>
     <td>3</td>
     <td>4.2TB</td>
     <td>34.3TB</td>
    </tr>
    <tr>
     <td rowspan="3">mix</td>
     <td>1</td>
     <td>2.4TB</td>
     <td>19.2TB</td>
    </tr>
    <tr>
     <td>2</td>
     <td>3.3TB</td>
     <td>26.6TB</td>
    </tr>
    <tr>
     <td>3</td>
     <td>5.1TB</td>
     <td>41.4TB</td>
    </tr>
   </tbody>
</table>
