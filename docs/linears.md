# Linear model

## ReLU rectified linear unit.

它把所有负数的输入都变成 0，正数不变。这样可以让梯度容易传播，矩阵变得稀疏，容易计算。

## GELU Gussian Error linear unit.

它和 ReLU 相似，但是都是可导的，更平滑。

当 x 很大正值 → 输出 ≈ x（几乎完全通过）
当 x 接近 0 → 输出 ≈ x × 0.5 左右（一半通过）
当 x 是负值 → 输出 很小但不为 0（有轻微泄漏）
当 x 很大负值 → 输出接近 0（但比 ReLU 更平滑地接近）

因为它保留了负数输入的特征，在使用到模型中的时候，效果往往比 ReLU 好。

## LayerNorm

LayerNorm 的核心作用（一句话总结）
对每个样本（每个 token）的特征向量独立进行归一化，让隐藏状态的均值接近 0、方差接近 1，从而稳定训练、加速收敛、缓解梯度消失/爆炸问题。
与 BatchNorm（BN）的主要区别对比：

## BatchNorm

Norm 发生在整个 batch 数据中，没有每个 token 的向量做 Norm 更细粒度。

# 是的，在大多数情况下，梯度的“长度”（或者说梯度的维度）确实和输出特征（output features）的个数是一样的，

# linear 层在做 forward 计算的时候会记录下 input，这样就可以在backward的时候用它来计算梯度了

## why activation

Without activation functions, stacking multiple linear transformations is pointless:
如果没有activation，在多层的Linear叠加都只是Linear模型的组合。
activation使得模型有了非线性化的特征
