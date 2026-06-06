import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.login import login_huggingface
from utils.train_data import get_shakespeare_data_loader, get_shakespeare_dataset

login_huggingface()


class Tokenizer(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads

        # d_k 是 d_model 除以 n_heads，得到每个头的大小
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Output Projection（输出投影）
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    # query key value are the same input in self-attention
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections, x * W_q, x * W_k, x * W_v
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads
        # Q shape: (batch, n_heads, seq_len, d_k) after view
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        # scores 的shape 是 (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # 对于 (seq_len, seq_len) 来做 mask
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax 在 Attention 层中也是在最后一个维度进行归一化
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # context 的shape 是 (batch, n_heads, seq_len, d_k)
        context = torch.matmul(attn, V)

        # Concatenate heads
        # 对第2个维度进行拼接，得到 (batch, seq_len, d_model)
        # contiguous 提高 view 操作的效率，避免在 view 操作时，tensor 不连续，导致性能下降。
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        # 最后返回的值的是 (batch, seq_len, d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        # 把模型扩大4倍，中间维度，让模型有更多的表达能力
        self.linear1 = nn.Linear(d_model, d_ff)
        # 激活函数，让模型不在是线性的
        self.activation = nn.GELU()  # GELU is a smooth activation function that is similar to ReLU, but has a smoother gradient.
        # 把模型缩小到原来的维度，输出到原来的shape
        self.linear2 = nn.Linear(d_ff, d_model)
        # 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self Attention + Residual + Norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward + Residual + Norm
        ff_output = self.ff(x)
        # Residual connection, 训练的结果加上原来的输入，得到新的结果。这样训练更稳定，不容易出现梯度消失或爆炸。
        x = x + self.dropout(ff_output)
        # Normalization LayerNorm 是沿着「特征维度」（通常是 tensor 的 最后一个维度）进行归一化的
        x = self.norm2(x)

        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        output = self.fc_out(x)
        return output


# ====================== DATASET ======================
class ShakespeareDataset(Dataset):
    def __init__(self, dataset, block_size=128, device="cuda"):
        self.tokenizer = Tokenizer()
        self.block_size = block_size

        self.data = dataset

    def __len__(self):
        return (len(self.data) - self.block_size) // 20

    def __getitem__(self, idx):
        sample = self.data[idx : idx + self.block_size + 1]
        print("sample: ", sample)

        idx = self.tokenizer(sample).input_ids
        x = torch.tensor(idx[:-1], dtype=torch.long)
        y = torch.tensor(idx[1:], dtype=torch.long)
        return x, y


# ====================== TRAINING ======================
def train():
    # Hyperparameters
    # batch_size = 16
    block_size = 128
    d_model = 16
    n_heads = 2
    n_layers = 1
    epochs = 10
    lr = 3e-4
    device = "cuda"

    # Load dataset
    print("Loading Tiny Shakespeare dataset...")
    dataset = get_shakespeare_dataset()

    # Create dataset and dataloader
    shakespeare = ShakespeareDataset(dataset, block_size=block_size, device=device)
    # print(f"Dataset loaded on {shakespeare.data.device}")
    dataloader = get_shakespeare_data_loader()

    # Model
    model = SimpleTransformer(
        vocab_size=shakespeare.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=block_size,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(
        f"Training on {device} | Vocab size: {shakespeare.vocab_size} | Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)  # (batch, seq_len, vocab)
            loss = criterion(output.view(-1, shakespeare.vocab_size), y.view(-1))

            loss.backward()
            # 把所有参数的梯度拼在一起，算一个 总范数（L2 norm）
            # 如果这个范数 > 0.5，就把所有梯度 等比例缩小，让总范数刚好等于 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        # Generate sample text Cheap sanity check, to see if the model is learning and can generate meaningful text.
        # if (epoch + 1) % 2 == 0:
        print("\n--- Generated Sample ---")
        generate_text(model, shakespeare, device, max_new_tokens=300)
        print("------------------------\n")

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char2idx": shakespeare.char2idx,
            "idx2char": shakespeare.idx2char,
        },
        "simple_transformer_shakespeare.pth",
    )
    print("Training completed and model saved!")


# ====================== TEXT GENERATION ======================
@torch.no_grad()
def generate_text(model, dataset, device, max_new_tokens=300, temperature=0.8):
    model.eval()
    start_text = "To be or not to be"
    context = (
        torch
        .tensor([dataset.char2idx[c] for c in start_text], dtype=torch.long)
        .unsqueeze(0)
        .to(device)
    )

    for _ in range(max_new_tokens):
        output = model(context)
        next_char_logits = output[0, -1, :] / temperature
        probs = torch.softmax(next_char_logits, dim=-1)
        next_char = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_char.unsqueeze(0)], dim=1)

    generated = "".join([dataset.idx2char[idx.item()] for idx in context[0]])
    print(generated)


if __name__ == "__main__":
    train()
    # device = "cuda"
    # block_size = 128
    # batch_size = 16
    # epochs = 2

    # # Load dataset
    # dataloader = get_shakespeare_data_loader()

    # for epoch in range(epochs):
    #     total_loss = 0
    #     for i, batch in enumerate(dataloader):
    #         print(batch.shape)
    #         print(batch)
    #         break
