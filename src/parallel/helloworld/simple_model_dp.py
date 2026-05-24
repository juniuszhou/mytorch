import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from dataset import HuggingFaceDatasetServer
from transformers import AutoTokenizer
import os


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
    # Create model
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    dataset = HuggingFaceDatasetServer()
    model = SimpleModel(vocab_size=tokenizer.vocab_size, embedding_dim=10)

    optimizer = optim.Adam(model.parameters(), lr=0.002)

    vocab_size = tokenizer.vocab_size
    sequence_length = 10
    total_loops = 1
    index = 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.distributed.init_process_group(backend="gloo")

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
            output = model(data_tensor)
            # print("output shape: ", output.shape)

            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                output_tensor.view(-1),
                ignore_index=-1,
            )

            # Backward pass - compute gradients
            loss.backward()

            # ====================================================================
            # Synchronizing Gradients/Weights Across Processes
            # ====================================================================
            #
            # KEY CONCEPT: In distributed training, we synchronize GRADIENTS, not weights.
            #
            # Why synchronize gradients?
            # - Each process computes gradients on its own data shard
            # - These gradients are different across processes
            # - We need to average gradients so all processes update weights the same way
            # - After optimizer.step(), all models will have identical weights
            #
            # Process:
            #   1. Each process: loss.backward() → computes local gradients
            #   2. Synchronize: average gradients across all processes
            #   3. Each process: optimizer.step() → updates weights (now identical)
            #
            # Alternative: Synchronizing weights directly (less efficient)
            # - Update weights independently: optimizer.step()
            # - Then average weights across processes
            # - This is slower and less stable than gradient synchronization

            # Method 1: Manual gradient synchronization (without DDP)
            # Average gradients across all processes using all_reduce
            for param in model.parameters():
                if param.grad is not None:
                    # all_reduce sums gradients across all processes in-place
                    torch.distributed.all_reduce(
                        param.grad.data, op=torch.distributed.ReduceOp.SUM
                    )
                    # Average the gradients by dividing by world_size
                    param.grad.data /= world_size

            # Now optimizer.step() will update weights consistently across all processes
            # Since gradients are averaged, all processes will apply the same update
            optimizer.step()
            optimizer.zero_grad()

            # ====================================================================
            # Alternative: Using DistributedDataParallel (DDP) - Recommended!
            # ====================================================================
            # DDP automatically synchronizes gradients, so you don't need manual sync.
            # To use DDP, wrap your model:
            #
            # from torch.nn.parallel import DistributedDataParallel as DDP
            # model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
            # optimizer = optim.Adam(model.module.parameters(), lr=0.002)  # Note: model.module
            #
            # Then the training loop is simpler - DDP handles gradient sync automatically:
            # loss.backward()  # DDP hooks automatically sync gradients here
            # optimizer.step()
            # optimizer.zero_grad()
            #
            # DDP is more efficient and handles edge cases better than manual sync.

            # ====================================================================
            # Combining Losses from Different Processes
            # ====================================================================
            # In distributed training, each process computes its own loss.
            # To get the average loss across all processes, we need to:
            # 1. Detach the loss from computation graph (no gradients needed)
            # 2. Reduce the loss values across all processes
            # 3. Average by dividing by world_size

            # Method 1: all_reduce (recommended - all processes get the result)
            # all_reduce sums the loss across all processes in-place
            loss_value = loss.detach()
            torch.distributed.all_reduce(loss_value, op=torch.distributed.ReduceOp.SUM)
            avg_loss = loss_value / world_size

            # Print average loss (same value on all ranks)
            print(
                f"Rank {rank}: Local loss = {loss.item():.4f}, Average loss = {avg_loss.item():.4f}"
            )

            # Optional: Verify weight synchronization (can be expensive, use sparingly)
            # Uncomment to verify weights are synchronized after each step:
            # if index == 0:  # Only check first iteration
            #     verify_weight_synchronization(model, rank, world_size)

            # ====================================================================
            # Alternative Methods (commented out):
            # ====================================================================

            # Method 2: reduce to rank 0 only (for logging/monitoring)
            # loss_for_reduce = loss.detach().clone()
            # torch.distributed.reduce(loss_for_reduce, dst=0, op=torch.distributed.ReduceOp.SUM)
            # if rank == 0:
            #     avg_loss = loss_for_reduce / world_size
            #     print(f"Average loss: {avg_loss.item():.4f}")

            # Method 3: Using all_reduce with AVG operation (if available)
            # Note: AVG operation may not be available in all PyTorch versions
            # loss_value = loss.detach()
            # torch.distributed.all_reduce(loss_value, op=torch.distributed.ReduceOp.AVG)
            # avg_loss = loss_value  # Already averaged

            # Method 4: Manual gather and average (more control, but more complex)
            # loss_tensor = loss.detach().unsqueeze(0)  # Make it a 1D tensor
            # gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            # torch.distributed.all_gather(gathered_losses, loss_tensor)
            # avg_loss = torch.stack(gathered_losses).mean()

        if index > total_loops:
            break
        index += 1
