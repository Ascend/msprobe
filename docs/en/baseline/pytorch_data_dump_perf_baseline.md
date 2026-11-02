# Precision Data Collection Baseline in PyTorch

## Time Expansion Baseline for Data Collection in "statistics" Mode

This baseline serves as a reference for performance inflation when collecting data in "statistics" mode with `risk_level` set to `ALL` under the PyTorch framework. This baseline tests the time inflation of a 16-layer Llama 2 large model across different collection modes on 4 ranks.
The tests were run using the MindSpeed-LLM framework, with core configurations: `seq-length` set to `8192`, `mbs` set to `2`, and `gbs` set to `32`.

| Collection Mode| Without Tool (Time Required)|   With Tool and Dump Enabled (Time Required)  | With Tool and MD5 Dump Enabled (Time Required)|
|:--------:|:--------:|:--------------------:|:--------------------:|
|  L0  | ≈6406 ms |  ≈10356 ms (Expanded 1.6 times)  |  ≈36492 ms (Expanded 5.7 times)  |
|  L1  | ≈6406 ms  |  ≈30890 ms (Expanded 4.8 times)  |  ≈122285 ms (Expanded 19 times)  |
| mix  | ≈6406 ms  |  ≈35986 ms (Expanded 5.6 times)  |  ≈155236 ms (Expanded 24 times)  |

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
