# Stanford CME295 Transformers & LLMs | Autumn 2025

## lesson 1

### tokenization

- word
- sub word, BPE
- character level. split each character

### word representation

coinsin similarity

- one hot encoding
- embedding

word2vec. use the vector of embedding to represent a word.

### RNN LSTM

它们的训练慢，不能并行。目前都被transformer所取代

### Attention

Q K V
query key value
输入一般是 (batch*size * input*len * d_model)
如果只有一个头，那么三个矩阵的维度都是 d_model * d*model。 是embedding的维度，也就是每个token的多维表示。
如果是多个头，那么 就是 e_dim \* （e_dim / head）。 会把第二个维度切分，这个很好理解，因为矩阵计算，要和输入
的最后一个维度一致。

流程：

你把想找的内容（Query）跟每一本书的标题（Key）做相似度匹配 → 得到匹配分数
把分数 softmax 变成权重（加起来等于1）
用这些权重去加权求和每一本书的正文内容（Value） → 得到你最终读到的信息

一句话总结：

Q 是提问者（我在找什么？）
K 是被问者（我能回答什么？）
V 是真正的内容提供者（如果我被选中，我给你什么信息？）

所以模型学到：

Key 向量可能偏向语法/位置/浅层匹配特征（方便计算相似度）
Value 向量更偏向深层语义、情感、事实信息（适合真正融合）

把 Key 和 Value 分开，给了模型更大的表达自由度，这是 Transformer 性能强的重要原因之一。

比喻 ｜ Query (Q) ｜ Key (K) ｜ Value (V)
搜索引擎 ｜ 你输入的搜索词 ｜ 网页的标题 ｜ 标签网页的正文内容

不同的attention head 倾向于关注序列中不同类型的关系或不同种类的模式。

## lesson 2 transformer

### position embedding

position embedding 的目的是将position的信息已某种方式加入到模型当中。
让靠近些的token得到更大的相似性，较远的则相反。

在输入一个序列的基础上，加入位置信息。这样增加了这个信息，同时让并行计算成为可能。
避免了像RNN那样，必须顺序计算，位置信息自然的得到。

learned or static

类型 ｜ 是否固定（不参与梯度更新）｜ 是否在训练中更新

原始 Sinusoidal（正弦位置编码）｜ 是 ｜ 否
RoPE（Rotary Position Embedding）｜ 是 ｜ 否

position embedding 和 input vector 只是简单的相加。

RoPE是目前主流的方式，它不是把position信息使用单独的一层来加入到input中。
它应用在每一层的每个q k矩阵中，让它们做一个rotation。rotation的角度和position相关。
这样达到模型中增加了position信息的目标。

### layer normalization

batchNorm depends on batch，现在已经不在使用了。
normalization: mean = 0, std = 1. no lmit for sum
softmax: each item > 0, sum = 1

普通归一化：只是“等比例缩放”
Softmax：是“等比例缩放 + 指数级放大最大值 + 指数级压制其他值”
softmax the distance will be magnified a lot.

RMS Norm 快，训练稳定，节省显存。不减均值 （mean subtraction），少了一个参数去学习

### sliding window attention

## lesson 5

### reasoning

chain of thought.
