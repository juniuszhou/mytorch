"""
PyTorch DTensor / Tensor Parallel examples.

Demonstrates:
  1. ColwiseParallel + RowwiseParallel: Megatron-style MLP (weights sharded)
  2. SequenceParallel: LayerNorm with sequence-sharded activations (saves memory)

Run: torchrun --nproc-per-node 2 dtensor.py
(Falls back to CPU if fewer GPUs than processes)
"""

import os

import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
    SequenceParallel,
)


# =============================================================================
# Example 1: Standard TP (ColwiseParallel + RowwiseParallel) - Megatron-style MLP
# =============================================================================


class ToyMLP(nn.Module):
    """Two linear layers with ReLU. Classic Megatron-LM TP pattern."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def run_tp_example(device_mesh, device_type):
    """ColwiseParallel on fc1, RowwiseParallel on fc2."""
    model = ToyMLP().to(device_type)
    model = parallelize_module(
        model,
        device_mesh,
        {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
    )

    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device_type)
    out = model(x)
    return out


# =============================================================================
# Example 2: SequenceParallel - LayerNorm with sequence-sharded input
# =============================================================================


class NormOnlyBlock(nn.Module):
    """Single LayerNorm to demonstrate SequenceParallel.
    Input must be sharded on sequence dimension (dim=1).
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def run_sequence_parallel_example(device_mesh, device_type, rank, world_size):
    """SequenceParallel on LayerNorm. Each rank holds 1/world_size of the sequence."""
    model = NormOnlyBlock(hidden_size=64).to(device_type)
    model = parallelize_module(
        model,
        device_mesh,
        {"norm": SequenceParallel(sequence_dim=1, use_local_output=True)},
    )

    # Create sequence-sharded input: each rank has (B, S/world_size, H)
    batch, seq_total, hidden = 2, 16, 64
    seq_per_rank = seq_total // world_size
    torch.manual_seed(42 + rank)
    x_local = torch.randn(batch, seq_per_rank, hidden, device=device_type)

    # Wrap as DTensor sharded on sequence dim so parallelize_module recognizes it
    x = DTensor.from_local(x_local, device_mesh, [Shard(1)])

    out = model(x)
    return out


# =============================================================================
# Main
# =============================================================================


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    n_gpus = torch.cuda.device_count()
    use_cuda = n_gpus >= world_size
    init_process_group(backend="nccl" if use_cuda else "gloo")
    device_type = "cuda" if use_cuda else "cpu"
    device_mesh = init_device_mesh(device_type, (world_size,))

    if rank == 0:
        print("=" * 60)
        print("PyTorch DTensor / Tensor Parallel Examples")
        print(f"World size: {world_size}, Device: {device_type}")
        print("=" * 60)

    # Example 1: Colwise + Rowwise (standard TP)
    if rank == 0:
        print("\n[1] ColwiseParallel + RowwiseParallel (Megatron-style MLP)")
    out1 = run_tp_example(device_mesh, device_type)
    if rank == 0:
        print(f"    Output shape: {out1.shape}")

    # Example 2: SequenceParallel
    if rank == 0:
        print("\n[2] SequenceParallel (LayerNorm with sequence-sharded input)")
    out2 = run_sequence_parallel_example(device_mesh, device_type, rank, world_size)
    if rank == 0:
        print(f"    Output shape: {out2.shape}")

    if rank == 0:
        print("\nDone.")

    destroy_process_group()


if __name__ == "__main__":
    main()
