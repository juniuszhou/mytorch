import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from dataset import HuggingFaceDatasetServer
from transformers import AutoTokenizer


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


if __name__ == "__main__":
    # Create model
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    dataset = HuggingFaceDatasetServer()
    model = SimpleModel(vocab_size=tokenizer.vocab_size, embedding_dim=10)

    optimizer = optim.Adam(model.parameters(), lr=0.002)

    vocab_size = tokenizer.vocab_size
    sequence_length = 10
    total_loops = 10
    index = 0

    for rows in dataset:
        for row in rows:
            result = tokenizer(row, truncation=True)["input_ids"]
            print(len(result))

            length = ((len(result) - 1) // 10) * 10
            result = result[: length + 1]
            # print(type(result))
            data_tensor = torch.tensor(result[:length])
            data_tensor = data_tensor.view(-1, 10)

            output_tensor = torch.tensor(result[1:])
            # shape: (batch_size, sequence_length)
            output_tensor = output_tensor.view(-1, 10)
            print("output_tensor shape: ", output_tensor.shape)

            # shape: (batch_size, sequence_length, vocab_size)
            output = model(data_tensor)
            print("output shape: ", output.shape)

            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                output_tensor.view(-1),
                ignore_index=-1,
            )
            print("loss: ", loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if index > total_loops:
            break
        index += 1
