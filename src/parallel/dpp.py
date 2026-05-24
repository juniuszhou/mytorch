import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
from torch.nn.parallel import DistributedDataParallel as DDP

# (DDP) 是最常见的实现，它在每个进程中包装模型，并在反向传播后自动同步梯度。
# DDP) 在标准（非弹性）模式下，整个训练过程确实需要等待所有节点（或进程）都启动并完成初始化后才会开始。
# 如果使用 torchrun 的弹性模式（--rdzv_backend=c10d 或 etcd），DDP 支持部分启动（partial launch）。
# 训练可以从可用进程开始，并在运行中动态加入新进程，而无需等待所有节点。


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank):

    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}"
    )

    model = ToyModel().to(rank)
    # important, model to ddp_model
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    labels = torch.randn(20, 5).to(rank)

    for i in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        loss = loss_fn(outputs, labels)
        print(f"loss = {loss.item()}")
        loss.backward()
        optimizer.step()

    print(f"training completed in rank {rank}!")


def main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in (
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
        )
    }

    print(" Initializing process group with:")
    pprint(env_dict)

    rank = int(env_dict["RANK"])
    print(f"[{os.getpid()}] Rank: {rank}")
    local_rank = int(env_dict["LOCAL_RANK"])
    local_world_size = int(env_dict["LOCAL_WORLD_SIZE"])

    acc = torch.accelerator.current_accelerator()
    print(f"[{os.getpid()}] Using accelerator: {acc}")
    backend = torch.distributed.get_default_backend_for_device(acc)
    print(f"[{os.getpid()}] Using backend: {backend}")  # default is nccl

    torch.accelerator.set_device_index(rank)
    # it　creates a process group, which is a group of processes that can communicate with each other for distributed training.
    # major information are gradients, parameters, and model states.
    dist.init_process_group(backend=backend)

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n",
        end="",
    )

    demo_basic(rank)

    # Tear down the process group
    dist.destroy_process_group()


# rendezvous
# torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=101 --rdzv_endpoint="localhost:5972" dpp.py
if __name__ == "__main__":
    main()
