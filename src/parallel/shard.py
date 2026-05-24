from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.device_mesh import init_device_mesh
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Shard, Replicate


# 示例模型：Transformer MLP 块
class MLP(nn.Module):
    def __init__(self, hidden_dim=4, intermediate_dim=8):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)  # Colwise 应用
        self.w2 = nn.Linear(intermediate_dim, hidden_dim)  # Rowwise 候选

    def forward(self, x):
        x = torch.relu(self.w1(x))  # 简化 FFN
        x = self.w2(x)
        return x


def main():
    tp_mesh = init_device_mesh("cuda", (1,))

    model = MLP().to("cuda")
    # Plan: w1 用 ColwiseParallel
    plan = {
        "w1": ColwiseParallel(output_layouts=Shard(1)),
        "w2": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
    }  # 默认 Shard(1)
    sharded_model = parallelize_module(model, tp_mesh, plan)

    # dist.init_process_group(backend="nccl")

    # 输入: [batch, seq, hidden_dim]，Replicate
    x = torch.randn(1, 1, 4).to("cuda")
    y = sharded_model(x)  # y: Shard(1) 输出
    print(y)


# dist.destroy_process_group()

#  torchrun --nnodes=1 --nproc_per_node=1 shard.py
if __name__ == "__main__":
    main()
