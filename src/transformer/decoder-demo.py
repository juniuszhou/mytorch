import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.pegasus.modeling_pegasus import (
    PegasusSinusoidalPositionalEmbedding,
)

from utils.train_data import load_shakespeare_text


class Tokenizer(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size

    def forward(self, texts):
        return self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )["input_ids"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.pe = PegasusSinusoidalPositionalEmbedding(max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        pos_emb = self.pe(x.shape[:2])
        return x + pos_emb.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        return self.W_o(context)


class FeedForward(nn.Module):
    # with SwiGLU implementation
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(
            self.dropout(self.activation(self.gate_proj(x)) * self.up_proj(x))
        )


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderDemo(nn.Module):
    def __init__(
        self, vocab_size, d_model, max_len, n_heads, n_layers, dropout, d_ff: int = 2048
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.d_ff = FeedForward(d_model, d_ff, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.fc_out(x)


class ShakespeareDataset(Dataset):
    def __init__(self, text, block_size=128, device="cuda"):
        self.tokenizer = Tokenizer()
        self.block_size = block_size
        self.vocab_size = self.tokenizer.vocab_size
        self.device = device
        self.data = self.tokenizer(text).squeeze(0)
        if device is not None:
            self.data = self.data.to(device)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y


def test_decoder_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_shakespeare_text()
    dataset = ShakespeareDataset(text, block_size=128)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0 if device.type == "cuda" else 2,
    )

    seq_len = 128

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf")).to(device)

    tokenizer = Tokenizer()
    decoder = DecoderDemo(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        max_len=1024,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    for i in range(100):
        progress_bar = tqdm(dataloader)
        for x, y in progress_bar:
            output = decoder(x, mask)
            loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {i}, Loss: {loss.item()}")


if __name__ == "__main__":
    test_decoder_demo()
