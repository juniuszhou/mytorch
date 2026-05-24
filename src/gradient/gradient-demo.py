"""demo how to calculate the gradient of a function in pytorch"""

import torch


# 定义一个函数
def f(x):
    return x**2


# 计算梯度
x = torch.tensor(2.0, requires_grad=True)
y = f(x)

# 保留 y 的梯度，不然 y 的梯度会被释放
y.retain_grad()
y.backward()

# 每个 Tensor 有它自己的梯度
print(x.grad)

# y 没有梯度，因为 y 不是一个叶子节点
print(y.grad)
