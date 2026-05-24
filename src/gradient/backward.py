import numpy as np

# 设置随机种子，保证结果可复现
np.random.seed(42)

# ------------------------------
#  超参数 & 数据维度
# 是的，在大多数情况下，梯度的“长度”（或者说梯度的维度）确实和输出特征（output features）的个数是一样的，
# linear 层在做 forward 计算的时候会记录下 input，这样就可以在backward的时候用它来计算梯度了
# 比如在多层感知机（MLP）中，最后一层的梯度维度与输出特征维度相同。
# ------------------------------
input_size = 4  # 输入维度
hidden_size = 5  # 隐藏层神经元个数
output_size = 1  # 输出维度（这里做回归）
learning_rate = 0.01

# 随机生成一个样本（单样本，为了演示清晰）
x = np.array([0.5, -1.2, 0.3, 0.8])  # shape: (4,)
y_true = np.array([2.7])  # shape: (1,)

# ------------------------------
#  随机初始化权重和偏置
# ------------------------------
# 第一层：W1 (hidden × input), b1 (hidden,)
W1 = np.random.randn(hidden_size, input_size) * 0.01
print("W1 shape: ", W1.shape)
b1 = np.zeros(hidden_size)

# 第二层：W2 (output × hidden), b2 (output,)
W2 = np.random.randn(output_size, hidden_size) * 0.01
print("W2 shape: ", W2.shape)
b2 = np.zeros(output_size)

print("初始 W2（第二层权重）：\n", W2)
print("初始 b2：", b2)


# ------------------------------
#  定义激活函数和导数（这里用 ReLU）
# ------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(float)  # 0 或 1


# ------------------------------
#  前向传播
# ------------------------------
z1 = W1 @ x + b1  # shape: (5,)
a1 = relu(z1)  # shape: (5,)

z2 = W2 @ a1 + b2  # shape: (1,)
y_pred = z2  # 直接输出（回归）

loss = 0.5 * (y_pred - y_true) ** 2
print("\n前向传播结果：")
print(f"y_pred = {y_pred[0]:.4f}, y_true = {y_true[0]:.4f}, loss = {loss[0]:.6f}")

# ------------------------------
#  反向传播（手动计算梯度）
# ------------------------------
# 1. 输出层误差项 δ2
delta2 = y_pred - y_true  # shape: (1,)

# 2. 对第二层参数的梯度
dW2 = np.outer(delta2, a1)  # δ2 * a1^T    shape: (1,5)
db2 = delta2  # shape: (1,)

# 3. 传回隐藏层激活值的梯度
da1 = W2.T @ delta2  # shape: (5,) 回到 input features 的维度， hidden_size
print("在层之间传播的梯度 da1 shape: ", da1.shape)

# 4. 隐藏层误差项 δ1 = da1 ⊙ σ'(z1)
delta1 = da1 * relu_deriv(z1)  # shape: (5,)

# 5. 对第一层参数的梯度
dW1 = np.outer(delta1, x)  # δ1 * x^T    shape: (5,4)
db1 = delta1  # shape: (5,)

# ------------------------------
#  打印所有梯度（方便观察）
# ------------------------------
print("\n=== 梯度计算结果 ===")
print("dL/dW2 (第二层权重梯度):")
print(dW2)
print("\ndL/db2:")
print(db2)

print("\ndL/da1 (传回隐藏层激活的梯度):")
print(da1)

print("\ndL/dz1 (隐藏层误差项 δ1):")
print(delta1)

print("\ndL/dW1 (第一层权重梯度):")
print(dW1)
print("\ndL/db1:")
print(db1)

# ------------------------------
#  简单更新一次参数（仅演示）
# ------------------------------
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W1 -= learning_rate * dW1
b1 -= learning_rate * db1

print("\n=== 更新后第二层权重 W2 ===")
print(W2)
