import torch
import torch.nn as nn

# 在神经网络（如 Transformer）中，Embedding（嵌入层）是将离散输入（如 token ID）映射为连续向量的可学习组件。
# 其训练过程本质上是通过反向传播（Backpropagation）更新嵌入矩阵，使向量捕捉输入的语义或特征表示。
# Embedding 不是孤立训练的，而是嵌入整个模型的端到端训练中，与其他层（如注意力、FFN）共同优化。

# 创建 Embedding 层
vocab_size = 100  # 词汇表大小
embedding_dim = 100  # 嵌入维度
embedding = nn.Embedding(vocab_size, embedding_dim)

# 输入：索引张量
input_indices = torch.tensor([0, 2, 4])  # 形状: (3,)

print(input_indices.shape)  # ([3,])
# 前向传播
output = embedding(input_indices)
print(output)
print(output.shape)
