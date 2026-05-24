import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from dataset import HuggingFaceDatasetServer
from transformers import AutoTokenizer
import os
from torch.distributed.fsdp import fully_shard, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class SimpleModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, vocab_size)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    # x (batch_size, sequence_length, embedding_dim)
    def forward(self, x: Tensor):
        return self.linear2(self.norm2(self.linear1(self.norm1(self.embedding(x)))))


def verify_weight_synchronization(model: nn.Module, rank: int, world_size: int):
    """
    Verify that model weights are synchronized across all processes.
    This function gathers weights from all ranks and checks if they match.
    """
    # Get first parameter as a sample (e.g., embedding weight)
    sample_param = next(model.parameters())
    param_shape = sample_param.shape

    # Create tensors to gather weights from all processes
    gathered_weights = [torch.zeros_like(sample_param) for _ in range(world_size)]

    # Gather weights from all processes
    torch.distributed.all_gather(gathered_weights, sample_param.data)

    # Check if all weights are the same (only on rank 0 for cleaner output)
    if rank == 0:
        weights_match = all(
            torch.allclose(gathered_weights[0], gathered_weights[i], atol=1e-6)
            for i in range(1, world_size)
        )
        if weights_match:
            print(f"✓ Weights are synchronized across all {world_size} processes")
        else:
            print(f"✗ WARNING: Weights differ across processes!")
            for i in range(world_size):
                print(f"  Rank {i} weight sample: {gathered_weights[i][0, :3]}")


if __name__ == "__main__":
    print("Starting DDP training...")
    torch.distributed.init_process_group(backend="gloo")
    device = torch.device("cpu")

    # Create model
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    dataset = HuggingFaceDatasetServer()
    model = SimpleModel(vocab_size=tokenizer.vocab_size, embedding_dim=10)

    model.to(device)

    # layers = model.ModuleList()
    # for layer in model.layers:
    #     fully_shard(layer, strategy=ShardingStrategy.FULL_SHARD)

    vocab_size = tokenizer.vocab_size
    sequence_length = 10
    total_loops = 1
    index = 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    fsdp_model = fully_shard(
        model,
        # must set the mesh as cpu, otherwise will use cuda. because the model is on cpu.
        mesh=torch.distributed.DeviceMesh("cpu", list(range(world_size))),
    )
    optimizer = optim.Adam(fsdp_model.parameters(), lr=0.002)

    for rows in dataset:
        for row in rows:
            result = tokenizer(row, truncation=True)["input_ids"]
            # print(len(result))

            length = ((len(result) - 1) // 10) * 10
            result = result[: length + 1]
            # print(type(result))
            data_tensor = torch.tensor(result[:length])
            data_tensor = data_tensor.view(-1, 10)

            output_tensor = torch.tensor(result[1:])
            # shape: (batch_size, sequence_length)
            output_tensor = output_tensor.view(-1, 10)
            # print("output_tensor shape: ", output_tensor.shape)

            # shape: (batch_size, sequence_length, vocab_size)
            output = fsdp_model(data_tensor)
            # print("output shape: ", output.shape)

            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                output_tensor.view(-1),
                ignore_index=-1,
            )

            # Backward pass - compute gradients
            loss.backward()
            print(f"rank: {rank}, loss: {loss.item()}")

            optimizer.step()
            optimizer.zero_grad()

        if index > total_loops:
            break
        index += 1
