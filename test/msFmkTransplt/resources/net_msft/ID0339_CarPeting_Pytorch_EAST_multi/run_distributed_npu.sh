export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29688
export HCCL_WHITELIST_DISABLE=1

NPUS=($(seq 0 7))
export RANK_SIZE=${#NPUS[@]}
rank=0
for i in ${NPUS[@]}
do
    export DEVICE_ID=${i}
    export RANK_ID=${rank}
    echo run process ${rank}
    please input your shell script here > output_npu_${i}.log 2>&1 &
    let rank++
done