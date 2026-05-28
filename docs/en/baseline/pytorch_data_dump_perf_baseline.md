# Precision Data Collection Baseline in PyTorch

## Time Expansion Baseline for Data Collection in "statistics" Mode

This baseline is a reference for performance expansion when data is collected in "statistics" mode in the PyTorch framework. This baseline shows time expansion of a single-layer DeepSeek model with eight cards in different collection modes.

| Collection Mode| Without Tool (Time Required)|  With Tool but Dump Disabled (Time Required) |   With Tool and Dump Enabled (Time Required)  | With Tool and MD5 Dump Enabled (Time Required)|
|:--------:|:--------:|:-------------------:|:--------------------:|:--------------------:|
| L0       | ≈ 95.1 ms |   ≈ 95.5 ms (no expansion)   | ≈ 420.0 ms (4.5x) |  ≈ 1011.3 ms (10x) |
| L1       | ≈ 95.1 ms  | ≈ 115.8 ms (1.2x)| ≈ 2469.0 ms (26x) |  ≈ 8636.0 ms (90x) |
| mix      | ≈ 95.1 ms  | ≈ 117.8 ms (1.2x)| ≈ 3635.4 ms (38x)| ≈ 10698.3 ms (112x)|

## Data Size Baseline in "tensor" Mode

This baseline is a reference for data size changes in "tensor" mode in the PyTorch framework. It shows the data size changes of LLAMA2-7B and LLAMA2-13B across different collection modes, global batch sizes, and configurations (single-rank vs. eight-rank).

### LLAMA2-7B

<table> 
   <tbody>
    <tr>
     <th>Collection Mode</th>
     <th>global_batch_size</th>
     <th>Single-Rank</th>
     <th>Eight-Rank</th>
    </tr> 
    <tr>
     <td rowspan="3">L0</td>
     <td>1</td>
     <td>7.8 GB</td>
     <td>63 GB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>16 GB</td>
     <td>125 GB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>24 GB</td>
     <td>187 GB</td>
    </tr> 
    <tr>
     <td rowspan="3">L1</td>
     <td>1</td>
     <td>300.8 GB</td>
     <td>2.3 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>480 GB</td>
     <td>3.6 TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>640 GB</td>
     <td>4.9 TB</td>
    </tr> 
    <tr>
     <td rowspan="3">mix</td>
     <td>1</td>
     <td>313.6 GB</td>
     <td>2.4 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>512 GB</td>
     <td>3.8 TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>672 GB</td>
     <td>5.1 TB</td>
    </tr> 
   </tbody>
</table>

### LLAMA2-13B

<table> 
   <tbody>
    <tr>
     <th>Collection Mode</th>
     <th>global_batch_size</th>
     <th>Single-Rank</th>
     <th>Eight-Rank</th>
    </tr> 
    <tr>
     <td rowspan="3">L0</td>
     <td>1</td>
     <td>13 GB</td>
     <td>97 GB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>25 GB</td>
     <td>194 GB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>37 GB</td>
     <td>291 GB</td>
    </tr> 
    <tr>
     <td rowspan="3">L1</td>
     <td>1</td>
     <td>440 GB</td>
     <td>3.4 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>720 GB</td>
     <td>5.4 TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>960 GB</td>
     <td>7.3 TB</td>
    </tr> 
    <tr>
     <td rowspan="3">mix</td>
     <td>1</td>
     <td>480 GB</td>
     <td>3.6 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>720 GB</td>
     <td>5.6 TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>1000 GB</td>
     <td>7.7 TB</td>
    </tr> 
   </tbody>
</table>
