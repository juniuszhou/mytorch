# Flops
how to compute flops

## flops of GPU
H100 SXM 
TF32 Tensor Core989 TFLOPSAI 训练常用（推荐精度）
BF16 / FP16 Tensor Core1,979 TFLOPS混合精度训练 / 推理

×2 的原因：FMA（Fused Multiply-Add） 指令，一个周期同时完成 1 次乘法 + 1 次加法 = 2 FLOPs。

MFU = 0.5 Model FLOPS Utilization（模型 FLOPS 利用率）
大量算力被浪费在内存搬运、通信、等待上。 所以一般 估计 MFU 为一半
模型规模典型    MFU     说明
7B~13B  45%~60%     较容易优化7
70B     35%~50%     常见水平
405B+（MoE）    30%~45%     较难优化

### formula. 这个除以2是因为GPU在标注自己的flops的时候，把它乘以了2. 因为它可以在一个时钟周期里面计算一个加和乘。
flops_per_second = 1979e12 / 2 * mfu

## flops of Model
$\text{Training FLOPs} \approx 6 \times N \times D$
N = 模型参数量（固定不变）
D = 总共处理过的 Tokens 数量

一个模型需要的flops。 70e9 参数总数  FP16计算
### 6 是一个计算单位，为什么是6

直观记忆法
操作    每次参数参与的操作  FLOPs/参数  总 FLOPs
Forward     x       × W     2    2N
Backward (dW)   x × dy      2   2N
Backward (dx)   Wᵀ × dy     2   2N
合计    -   6   6N

### Epoch 为什么没有参与计算
当前主流 LLM 预训练通常只训 1 ~ 2 个 Epoch。
原因：数据量已经非常巨大（几万亿甚至十几万亿 tokens），继续增加 Epoch 性价比不高，还容易过拟合。
所以大家更常用 “训练了多少 Tokens” 来衡量计算量，而不是 Epoch 数。


其中 2 是计算矩阵相乘需要的二个step 相乘和相加
另外4是计算梯度 2是计算weights 梯度，另外2 是 input output

total_flops = 6 * 70e9 * 15e12  # @inspect total_flops
    h100_flop_per_sec = 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # @inspect flops_per_day

## Memory of GPU
h100 80e9 80G memory
bytes_per_parameter = 2 + 2 + 4 + 4 parameters + gradients, optimizer state



## AMP automatic mixed precision
bf16 data type from google for ML training. 
there is a lib in torch to convert it automatically.
