# Transformer

- embedding
   - text, vocab to vector or tensor
- position encoding
   - add the position info to vector 
- dropout
   - avoid overfitting
- encode / decode layers
   - encode / decoder
      - attention softmax() 
      - mask for decoder layer
   - forward
      - linear + activation + linear + dropout
   - residual output = x(input) + output
   - layer norm
- norm usually layer norm
   - layer norm output
- output
   - output vector to vocab

## embedding
会把二个矩阵相加，一个矩阵是token的embedding，另一个矩阵是position的embedding。 一般就是简单地相加。

## core architecture


核心架构
Transformer 采用编码器-解码器（Encoder-Decoder）结构：

编码器（Encoder）：处理输入序列，输出上下文表示。堆叠 N 层（通常 N=6），每层包括：
多头自注意力（Multi-Head Self-Attention）：计算序列中每个元素与其他元素的关联权重。
前馈神经网络（Feed-Forward Network）：位置独立的非线性变换。
残差连接（Residual Connection）和层归一化（Layer Normalization）：稳定训练。

解码器（Decoder）：生成输出序列，类似编码器，但多了一个掩码多头注意力（Masked Multi-Head Attention），防止未来信息泄露；还包括编码器-解码器注意力（Encoder-Decoder Attention），融合输入上下文。
位置编码（Positional Encoding）：由于无顺序性，使用正弦/余弦函数或可学习嵌入添加位置信息。

BertTokenizer 的分词算法详解
BertTokenizer 是 Hugging Face Transformers 库中用于 BERT 模型的 tokenizer，它的核心分词算法是 WordPiece（词片）。
WordPiece 是一种子词（subword）分词方法，由 Google 开发，用于解决稀有词问题，同时保持词汇表大小可控（通常 30k 左右）。
它不同于传统的词级分词（如空格分割），能将生僻词分解为常见子词片段，提高模型对未知词的鲁棒性。
WordPiece 的设计灵感来源于 Byte-Pair Encoding (BPE)，但有关键区别：WordPiece 使用基于似然的评分机制选择合并对，
且 tokenization 时采用贪婪的最长匹配策略，而非严格遵循合并规则。以下详细说明其工作原理、训练过程、与 BPE 的差异及示例。

1. WordPiece 的整体流程
   WordPiece 分成两个阶段：训练阶段（构建词汇表）和推理阶段（实际分词）。

训练阶段：从字符级开始，迭代合并高频子词对，直到达到目标词汇表大小。
推理阶段：对新文本进行贪婪匹配，优先选择词汇表中最长的匹配子词。

2. 训练过程（Training Process）
   训练在语料库上进行，目标是构建一个子词词汇表。步骤如下表所示（基于 Hugging Face 的实现）：

### 预分词

步骤描述关键细节

- a. 预分词 (Pre-tokenization)将原始文本按空格分割成词（使用 BERT 的预分词器，如处理标点）。保留词的偏移量，用于后续对齐。示例："Hello world" → ["Hello", "world"]。
- b. 构建初始字母表第一个字符作为独立 token，后续字符前缀 "##"（表示非词首）。示例："word" → ["w", "##o", "##r", "##d"]。字母表包含纯字母 + "##字母"。
- c. 计算频率统计每个词在语料中的出现频率 (word_freqs)。用于后续评分。
- d. 分割每个词将每个词拆成原子片段（字符级）。示例："about" → ["a", "##b", "##o", "##u", "##t"]。
- e. 评分候选合并对相邻片段对 (x, y) 计算分数：score = freq(pair) / (freq(x) \* freq(y))。分母惩罚高频片段的合并，鼓励创建稀有但有用的子词。
- f. 选择最佳对挑选分数最高的片段对。如果多个分数相同，任选一个（不影响最终结果）。g. 合并片段将对替换为新 token（去除第二个的 "##" 前缀），加入词汇表。示例：("a", "##b") → "ab"。
- h. 更新分割在所有词中替换该对的出现。重复 e-h，直到词汇表达到目标大小（e.g., 30k）。
- i. 添加特殊 token 在词汇表开头插入 [PAD]、[UNK]、[CLS]、[SEP]、[MASK]。BERT 特定，用于填充、未知词、分类等。
  注意：Google 未开源确切训练代码，但 Hugging Face 等库已复现该算法。



### multi head

每个头只用 d_model / h 维（e.g., 512/8=64 维），学习特定“通道”。 那么就是每个 head 的计算只是抽取整个 model 维度的一份来计算。

在 Transformer 的自注意力（Self-Attention）机制中，核心是生成三个投影矩阵：Query (Q)、Key (K) 和 Value (V)。这些是通过输入序列 $X$（形状：序列长度 $n \times$ 模型维度 $d_{\text{model}}$）与三个可学习权重矩阵 $W^Q$、$W^K$、$W^V$ 进行线性变换得到的。这些矩阵允许模型从不同子空间学习表示，捕捉查询-键匹配和值加权。
以下是详细的数学推导和计算过程，基于论文《Attention is All You Need》。假设单头注意力（多头是其并行扩展）。我们用矩阵乘法表示，实际实现中是张量操作。

1. 输入准备

输入：$X \in \mathbb{R}^{n \times d_{\text{model}}}$，其中 $n$ 是序列长度（e.g., 512），$d_{\text{model}}$ 是嵌入维度（e.g., 512）。
权重矩阵初始化：
$W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$：查询投影矩阵，$d_k$ 是键/查询维度（通常 $d_k = d_{\text{model}} / h$，h 是头数）。
$W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$：键投影矩阵。
$W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$：值投影矩阵，通常 $d_v = d_k$。

初始化方式：随机（e.g., Xavier 均匀分布），训练中通过反向传播更新。

2. 生成 Q, K, V 的计算过程
   这些是线性变换：矩阵乘法 $X W$。

Q 矩阵的维度是 嵌入维度 d * （嵌入维度 d / 头的数量 h）
Query (Q) 计算： 在寻找什么
$$Q = X W^Q \in \mathbb{R}^{n \times d_k}$$
直观解释：Q 表示序列中每个位置“在寻找什么”。每个行向量 $q_i = x_i W^Q$（第 i 个位置的查询向量）。
维度变化：从 $n \times d_{\text{model}}$ 到 $n \times d_k$（通常 $d_k < d_{\text{model}}$，降维学习）。

Key (K) 计算： 能提供什么。Q 矩阵的维度是 嵌入维度 d * （嵌入维度 d / 头的数量 h）
$$K = X W^K \in \mathbb{R}^{n \times d_k}$$
直观解释：K 表示序列中每个位置“能提供什么”。$k_j = x_j W^K$。
维度：同 Q。

Value (V) 计算：实际内容
$$V = X W^V \in \mathbb{R}^{n \times d_v}$$
直观解释：V 是实际内容，用于加权求和。$v_j = x_j W^V$。
维度：$n \times d_v$，通常 $d_v = d_k$。

3. 后续注意力计算（基于 Q, K, V）
   生成 Q, K, V 后，进入 Scaled Dot-Product Attention。三个矩阵的计算是注意力计算的基础。

注意力分数 (Scores)：
$$\text{Scores} = Q K^T \in \mathbb{R}^{n \times n}$$
$\text{Scores}_{i,j} = q_i \cdot k_j$（第 i 位置与第 j 位置的相似度）。
计算细节：矩阵乘法，时间 O(n² d_k)。

缩放 (Scaling)：
$$\text{Scaled Scores} = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$
为什么缩放：点积方差 ≈ d_k，未缩放 softmax 梯度小。缩放后方差 ≈ 1。
推导：假设 $q_i, k_j$ 元素 ~ N(0,1)，则 Var(q_i · k_j) = d_k，除以 √d_k 标准化。

Softmax 归一化 (Attention Weights)：
$$\alpha = \text{softmax}(\text{Scaled Scores}) \in \mathbb{R}^{n \times n}$$
$\alpha_{i,j} = \frac{\exp(s_{i,j})}{\sum_{l=1}^n \exp(s_{i,l})}$，每行和为 1。
可选掩码：解码器中添加 -∞ 到未来位置（causal mask）。

加权输出 (Output)：
$$\text{Attention}(Q, K, V) = \alpha V \in \mathbb{R}^{n \times d_v}$$
$o_i = \sum_{j=1}^n \alpha_{i,j} v_j$（第 i 位置的上下文表示）。

完整公式：
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

4. 多头注意力（Multi-Head Attention）扩展单头限于单一表示，多头并行 h 个头（e.g., h=8），每个头用独立的 $W_i^Q, W_i^K, W_i^V$（共享输入 X）。

每个头计算：
$$\text{head}_i = \text{Attention}(X W_i^Q, X W_i^K, X W_i^V)$$
每个头维度：d_k = d_model / h (e.g., 512/8=64)。

拼接与投影：
$$\text{MultiHead} = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \in \mathbb{R}^{n \times d_{\text{model}}}$$
$W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$：输出投影矩阵，融合多头信息。

5. 训练中的更新

梯度传播：从输出 O 反向到 V → α → Scores → Q/K（链式法则），最终更新 W^Q, W^K, W^V。
残差连接：输出 + X（稳定梯度）。
参数量：每个矩阵 ≈ d_model × d_k，三个矩阵总 ≈ 3 d_model²（单头）。

示例数值计算（简化，n=2, d_model=4, d_k=2）
假设 $X = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$，
$W^Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$（简化），类似 W^K=W^V。

Q = $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$，K 同，V 同。

Scores = $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$。

Scaled = Scores / √2 ≈ $\begin{pmatrix} 0.707 & 0 \\ 0 & 0.707 \end{pmatrix}$。

α ≈ softmax ≈ $\begin{pmatrix} 0.622 & 0.378 \\ 0.378 & 0.622 \end{pmatrix}$。

Output = α V ≈ $\begin{pmatrix} 0.622 & 0.378 \\ 0.378 & 0.622 \end{pmatrix}$。


### 解释
n token 序列长度，d 嵌入维度，h 头数。
输入是X 序列长度 * 嵌入维度的矩阵。 （n * d）  
序列长度 是总的token的数量，而不是单词数量，二者可能会有区别。像空格，标点符号等都会被视为token。
嵌入维度是每个token的表示维度。

Wq 是参数矩阵，它将输入的token的表示维度转换为查询维度。（d * d_k）d_k = d_model / h 
Q=X * Wq (n * d_k) 

a = Q * K^T (n * n) 整个矩阵作为一层的输出也非常重要，它代表了token之间的注意力。 

Multihead = (n * d) 回到和X一样的维度，它又作为下一层的输入。encode decode在论文中是6层。



### 
为什么需要 FeedForward？（重要性）

引入非线性：Attention 本身是线性加权求和（softmax 后仍是线性组合），没有非线性。FFN 通过激活函数引入强非线性，提升模型容量。
扩大模型容量：中间维度通常扩大 4 倍，大幅增加参数量（Transformer 中 FFN 参数通常占总参数的 2/3 左右）。
特征重表达：Attention 得到上下文信息后，FFN 对每个位置的表示进行独立的“思考”和变换

### residual network
避免梯度爆炸或者消失

2. Softmax 的核心作用

作用具体说明将分数转为概率分布把原始的 Attention Scores 转换成权重，且所有权重之和为 1（$  \sum \alpha_i = 1  $）。
引入非线性Softmax 是非线性函数，让 Attention 不再是简单的线性加权，而是动态加权求和。
突出重要信息，抑制噪声分数高的 Key 会得到很大的权重（接近 1），分数低的 Key 权重被压到接近 0，实现“聚焦重点”。
可解释性输出的 Attention Weights 可以直接理解为“这个 token 对当前 token 的注意力比例”。数值稳定配合 Scaling 操作，防止梯度爆炸或消失。

Softmax 在 Attention 层中也是在最后一个维度进行归一化