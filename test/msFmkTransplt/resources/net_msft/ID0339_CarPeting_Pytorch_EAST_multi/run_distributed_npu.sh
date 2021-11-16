export MASTER_ADDR=localhost
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export NPU_WORLD_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export NPU_CALCULATE_DEVICE=${i}
    export RANK=${rank}
    echo run process ${rank}
    please input your shell script here > output_npu_${i}.log 2>&1 &
    let rank++
done