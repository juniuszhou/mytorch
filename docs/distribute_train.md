# Tech and Terms for distributed training.

## 5D 并行。

并行方式,维度,典型组合示例（GPU 数量）,适用场景
DP,数据批次,ZeRO-3 / FSDP,通用加速
TP,隐藏维度,TP=8,单层太大
PP,模型层,PP=4,模型很深
CP,序列长度,CP=4,超长上下文（>32k token）
EP,MoE 专家,EP=8（8 experts）,MoE 模型

Megatron-LM 和 DeepSpeed 是目前（2026 年）训练超大规模语言模型（LLM，如百亿到万亿参数级别）时最核心、最常用的两个分布式训练框架。

## tech

### DiLoCo

DiLoCo（Distributed Low-Communication，分布式低通信优化）是由 DeepMind 于 2023 年提出的一种分布式优化算法，旨在解决大规模语言模型（LLM）在异构、低带宽分布式环境下的训练挑战。

### PyTorch FSDP/DDP/DP

DP: Data parallel. 数据并行
Multi thread, single node. 受到python全局线程同步锁的限制，只是用于测试。

DPP: Distributed Data Parallel
a good mapping rule is: one process to one GPU
rank: your gpu ID in whole world according to world_size
world_size: total number of processes
local size: processes per node

FSDP:
分片主要针对的是模型参数（parameters）和梯度（gradients）
沿 dim=0 分片（通常是 out_features 或 hidden_dim 的第一个维度）

FSDP 的分片针对的是 模型参数（weights）、梯度（gradients）和可选的优化器状态，
输入和输出不分片，仍然是完整的（replicated）。

模型参数被均匀分片（shard）到所有 GPU 上
例如：Linear(4096, 4096) 的权重 [4096, 4096] （output features, input features）→ 每个 GPU 持 [4096, 4096/world_size]

FSDP2 是按照 output features 来把权重分片的。

FSDP2 在 forward 的时候，会自动执行 all-gather，每个 rank 吧自己的参数发送出去。
每个 rank 也接收其他的参数，让自己暂时拥有所有的参数，这样可以做完整的 forward 计算。
在 forward 结束后，将其他的参数释放，只保留自己 rank 需要保存的份额。
这个过程是在每一个 Module 单独进行的，因此如果一个模型有几十层，那么对显存的消耗，或者重复不会那么多。
和单个 GPU 做所有计算，存贮所有参数相比。

Fully Sharded Data Parallel (FSDP) in PyTorch
Fully Sharded Data Parallel (FSDP) is a distributed training wrapper in PyTorch designed to shard (split) a model's parameters, gradients, and optimizer states across multiple GPUs or processes. This allows training very large models on hardware with limited memory per device, inspired by the ZeRO-3 technique from DeepSpeed. It's particularly useful for scaling models like large language models (LLMs) beyond what standard data parallelism can handle.

FSDP reduces the memory footprint per GPU by avoiding full model replication. Instead of duplicating the entire model on each device (as in traditional parallelism), it distributes shards dynamically during training.

---

| Feature           | DP                           | DDP                               | FSDP                              |
| ----------------- | ---------------------------- | --------------------------------- | --------------------------------- |
| Model replication | Full model on each GPU       | Full model on each GPU            | Sharded across GPUs               |
| Memory efficiency | Low (full model copy)        | Low (full model copy)             | High (model sharding)             |
| Communication     | All-reduce gradients         | All-reduce gradients              | All-gather/Reduce-scatter         |
| Scalability       | Limited by single GPU memory | Limited by single GPU memory      | Scales to very large models       |
| Setup complexity  | Simple                       | Moderate (requires process group) | Moderate (requires process group) |
| Use case          | Small models, single node    | Medium models, multi-GPU          | Large models, limited GPU memory  |

---

#### Model parallel

模型的并行和数据并行相对应，把训练任务分割。

pipeline parallet:
流水线并行，把每个层放到不同的 GPU 中训练，让 GPU 像流水线一样计算

#### Tensor parallel

把不同的 Tensor 放到不同的 GPU 中训练，像 transformer 这样的模型，每个头是可以单独并行计算的。

## 核心概念

数据并行 (DP)：每个设备持有完整模型副本，仅将输入数据（batch）分片。每个设备独立计算前向/反向传播，然后同步梯度（e.g., AllReduce）。
DP 都是通过不同的程序都独立的采样数据来达到 DP 的效果。

张量并行 (TP)：将模型参数（张量，如权重矩阵）分片到多个设备上，每个设备计算部分操作（e.g., 矩阵乘法的部分行/列）。需要频繁通信交换中间结果。

快速记忆口诀

想让输出被切片（中间结果 shard） → 用 Colwise
想把切片的结果再合起来（最终输出 replicate） → 用 Rowwise

权重矩阵切分方式
按列切分 Colwise （沿着输出维度切） weight.shape = (out_features, in_features) → 切成多个 (out_features/TP_degree, in_features)
按行切分 Rowwise （沿着输入维度切） weight.shape = (out_features, in_features) → 切成多个 (out_features, in_features/TP_degree)

现代框架支持 3D 并行（DP + TP + Pipeline
否则 TP 跨节点会变慢（推荐 TP 在节点内）

### DeepSpeed

## projects

### DeAI

PrimeIntellect-ai/ZeroBand

### federated training

### embedding 的 TP 实现和机制

在 embedding 层，它只是做一个查找的操作，根据 token id 找到多维向量的表示。它的 TP 配置如下。

"tok_embeddings": RowwiseParallel( # each GPU has a copy of the token embeddings, so we use Replicate()
input_layouts=Replicate(), # output sharded according to the tp_size，Shard(0) 是 batch 的维度
output_layouts=Shard(1),
),

RowwiseParallel 意味着每个 GPU 都只有一部分 token 的维度信息，因为它是按照 vocab 来 split 的。
那么它在 embedding 之后如何找到缺失的信息呢，它并不是把所有的结果都发送给其他 GPU。
而是通过 AllGather 的方式，只向其他 GPU 查询丢失的信息，这样通信量就大大减少了。

input 使用 Replicate，因为所有的输入都需要分到所有 GPU 去做查询。
output 使用 Shard(1) 使用多维向量来划分 TP。可以和接下来的 encoder 层衔接。

##

SequenceParallel 的核心是序列级并行（sequence-level parallelism），它不分片模型参数，而是聚焦于输入数据的分片，以节省激活内存。
Transformer 模型中，激活张量（如 hidden_states）形状通常为 [batch_size, seq_len, hidden_dim]
那么它 shard 的维度就是第二个，对 token 序列进行分片，然后分别计算。 它没有对模型参数进行分片。
数据结果也不需要做 AllGather 或者 AllReduce，结果都是独立的。

不像 Rowwise 线性层的 AllReduce

#### ColwiseParallel 一般用在 Linear 模型

ColwiseParallel 把 weight（形状 [in_features, out_features]）沿 out_features（dim=1，列维度） 分片
它默认的输入是 replica，复制整个数据。输出是沿着 Shard（1）分片

forward 没有 reduce，它的输出是分片的。
backward 有梯度聚合。

### pipeline 并行，是把整个计算分成不同的阶段，像是 CPU 的流水线。在第一组数据做第一个 stage 的计算，

剩下的 GPU 空闲。
然后第二个 GPU 计算数据组 1 的第二个阶段计算，第一个 GPU 开始第二组数据的计算。

n_microbatches 是一个超参数来控制，分成多少个迷你 batch，也就是分成多少个阶段。

### compare pipeline, sequence and Col/Row

n_microbatches dim=0（batch 维度）切分
Sequence Parallel dim=1 （sequence 维度）切分

(Colwise/Rowwise) dim=-1 或 dim=1（hidden_dim） 按照特征切分。

### FSDP 和 ColwiseParallel 比较：

FSDP 是对整个模型的数据进行并行计算
ColwiseParallel 对一个线性层的权重矩阵进行分片。

### Rendezvous

Rendezvous in PyTorch is the mechanism that allows multiple processes (usually one per GPU) to find and connect to each other at the beginning of distributed training.
It is the very first step of distributed communication — before any all_reduce, broadcast, all_gather, etc. can happen.

### all gather vs all reduce

all-gather 只是把数据做汇总，all-gather后所有节点都会有完整的数据
all-reduce 会把数据做计算，比如平均值，总值等

### ColwiseParallel RowwiseParallel 过程，在只有二个Linear层的情况下，如何通信

输入 X 是 Replicate（所有 GPU 都有完整拷贝）

↓
第一个 Linear 用 ColwiseParallel
↓
每个 GPU 只计算自己负责的那部分输出列 → 结果是 Shard(-1)
↓
**PyTorch 自动插入 All-Gather** → 把所有分片拼接起来，得到完整的中间激活（通常还是 Replicate 或 Shard，看后续需求）
↓
做 activation (silu / gelu 等) → 还是保持相同的分布
↓
第二个 Linear 用 RowwiseParallel
↓
每个 GPU 拿到的输入已经是 Shard(-1)，本地做部分 matmul
↓
**PyTorch 自动插入 All-Reduce** → 把各 GPU 的部分和求和，得到正确的完整输出（通常变回 Replicate）
↓
输出给下一层

### CP 的实现

CP 最主流的两种实现方式（2025–2026年主流做法）
核心逻辑是：每个 GPU 只永久持有自己负责的那一小块 Q（对应本地序列段），而 KV 是轮流借用的。最终每个 GPU 计算出的 O（attention 输出） 也是只对应自己那一段序列的输出。

Ring-Attention / Megatron-CP（最常见）
把一个很长的序列（比如128k、256k、1M）平均切成 N 份
第 i 个 GPU 只持有第 i 段的 Q、K、V（显存只存 1/N）
但计算 attention 时，每个 token 的 Q 理论上要看到全部的 K、V
所以他们用“环形传递 + 分块计算”的方式来解决：
计算过程（简化版）：
每个 GPU 先用自己本地的 KV block 计算 local attention（对自己负责的那一段）
然后把自己的 KV block 顺时针传给下一个 GPU（Ring all-to-all 通信）
收到别人的 KV block 后再算 cross-block 的 attention
重复 N-1 次，把整个序列的 KV 都“轮询”一遍
最终每个 GPU 都能得到完整的 attention 输出，但显存峰值只用了 1/N 左右
这就是为什么叫 Ring-Attention，本质是把全局 attention 拆成多次局部计算 + 环形传递。
DeepSpeed-Ulysses 风格的 Sequence Parallel
也是切序列，但通信模式不同
更多依赖 All-Gather + Reduce-Scatter
前向时把所有 KV 临时 gather 起来算 attention
反向时再 scatter 梯度
显存节省不如 Ring 激进，但通信模式更简单，在某些硬件上可能更快

### SequenceParallel

SequenceParallel 本身不 reduce，但當下一個模塊（尤其是 ColwiseParallel）需要 Replicate 輸入時，DTensor 會自動在 forward 時插入 all-gather 把 Shard(1) 變 Replicate；
backward 時反過來會 reduce-scatter 或 all-reduce 來同步 gradient。
你只要正確設定 input/output layouts，同步就自動發生，不需要手動寫 dist.all_reduce。
