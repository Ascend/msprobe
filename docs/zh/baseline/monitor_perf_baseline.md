# 训练状态监测工具标准性能基线报告

## 环境信息

NPU：Atlas A2 训练系列产品

CPU：

![输入图片说明](../figures/monitor/cpu_info.png)

Torch：2.1.0

CANN：8.0.RC2

除上述环境信息外，被监测的模块数量和结构也会影响性能，因此本次选取典型网络进行测试，并且选取耗时稳定后的步数进行测试。工具输出占用空间小，对内存无要求。

## 模型信息和性能基线

以下场景的性能基线测试数据均为多次测试后取平均值，因此实际运行时性能数据可能会根据环境状态稍有浮动。

### LLAMA2-13B

主要数据类型：BFLOAT16

模型层数：40

配置文件（采样了10层）：

```json
{  
    "targets": {  
        "language_model.encoder.layers.0": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.1": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.2": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.3": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.4": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.5": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.6": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.7": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.8": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"},
        "language_model.encoder.layers.9": {"input": "tuple[2]:0", "output": "tensor", "input_grad":"tuple[2]:0", "output_grad":"tuple[1]:0"}
    },
    "module_ranks": "0"
}  
```

启动命令参数：python3 -u pretrain_gpt.py --local-rank=1 --tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 --sequence-parallel --num-layers 40 --hidden-size 5120 --ffn-hidden-size 13824 --num-attention-heads 40 --tokenizer-type Llama2Tokenizer --tokenizer-model /new_data/LLM/checkpoint_origin/llama2-13b-hf/tokenizer.model --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 2 --global-batch-size 16 --make-vocab-size-divisible-by 1 --lr 1e-6 --train-iters 5000 --lr-decay-style cosine --untie-embeddings-and-output-weights --disable-bias-linear --attention-dropout 0.0 --init-method-std 0.01 --hidden-dropout 0.0 --position-embedding-type rope --normalization RMSNorm --use-fused-rmsnorm --swiglu --use-flash-attn --no-masked-softmax-fusion --attention-softmax-in-fp32 --min-lr 1e-8 --weight-decay 1e-1 --lr-warmup-fraction 0.01 --clip-grad 1.0 --adam-beta1 0.9 --initial-loss-scale 4096 --adam-beta2 0.95 --no-gradient-accumulation-fusion --load /data/LLM/checkpoint_magatron/llama2_13b_tp1_pp8 --no-load-optim --no-load-rng --use-fused-swiglu --use-fused-rotary-pos-emb --use-mc2 --bf16 --data-path /data/LLM/data_modellink/llama2_13b/alpaca_text_document --split 949,50,1 --log-interval 1 --save-interval 10000 --eval-interval 1000 --eval-iters 10 --distributed-backend nccl --save ./ckpt

不加工具原始耗时：**4s**

加工具后单卡耗时：**4.25s**

加工具后多卡耗时：**4.35s**
