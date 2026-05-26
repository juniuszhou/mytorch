# knowledge

## training pipeline.

```python
for data, target in dataloader:
    optimizer.zero_grad() # 1. 清零上一次的梯度（最重要！）

    output = model(data)           # 前向

    loss = criterion(output, target)

    loss.backward()                # 2. 反向传播 → 计算梯度并累加到 .grad 里

    optimizer.step()               # 3. ← 这里！根据 .grad 更新参数
```

梯度 shape = 参数本身的 shape
（没有转置、没有广播、没有 reshape，形状一模一样）

## memory related ops in torch

memmap, pin_memory
memmap map data from disc to memory
pin_memory
