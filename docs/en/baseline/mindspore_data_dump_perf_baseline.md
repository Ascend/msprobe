# Precision Data Collection Baseline in MindSpore

## Time Expansion Baseline for Data Collection in "statistics" Mode (MD5 Disabled)

This baseline is a reference for performance expansion when data is collected in "statistics" mode in the MindSpore framework. It shows performance scaling of a 38B language model with eight ranks in different collection modes.

| Collection Mode| Without Tool (Time Required)| With Tool but Dump Disabled (Time Required)| With Tool and Dump Enabled (Time Required)|
| :------: | :------------: | :-------------------------: | :-----------------------: |
|    L0    |    ≈ 340 ms     |     ≈ 340 ms (no change)     |   ≈ 1.2s (3.5x)  |
|    L1    |    ≈ 340 ms     |  ≈ 0.7–1.2s (2–4x) |   ≈ 3.8s (11x)  |
|   mix    |    ≈ 340 ms     |  ≈ 0.7–1.2s (2–4x) |   ≈ 5.5s (16x)   |

## Data Size Baseline in "tensor" Mode

This baseline is a reference for data size changes in "tensor" mode in the MindSpore framework. It shows the data size changes of a 38B language model across different collection modes, global batch sizes, and configurations (single-rank vs. eight-rank).

### Data Size Changes

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
     <td>262 GB</td>
     <td>2.1 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>480 GB</td>
     <td>3.8 TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>928 GB</td>
     <td>7.4 TB</td>
    </tr> 
    <tr>
     <td rowspan="3">L1</td>
     <td>1</td>
     <td>2.1 TB</td>
     <td>17.1 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>2.8 TB</td>
     <td>22.7 TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>4.2 TB</td>
     <td>34.3 TB</td>
    </tr> 
    <tr>
     <td rowspan="3">mix</td>
     <td>1</td>
     <td>2.4 TB</td>
     <td>19.2 TB</td>
    </tr> 
    <tr>
     <td>2</td>
     <td>3.3TB</td>
     <td>26.6TB</td>
    </tr> 
    <tr>
     <td>3</td>
     <td>5.1 TB</td>
     <td>41.4 TB</td>
    </tr> 
   </tbody>
</table>
