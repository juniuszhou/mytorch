import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer

from projects.loader import PretrainDataLoader, PretrainDataset
from projects.model import TransformerLM


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: PretrainDataLoader,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-4)
        self.dataloader = dataloader
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def train(self):
        index = 0
        for batch in self.dataloader:
            index += 1
            input_ids, labels = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if index % 100 == 0:
                print(
                    f"loss: {loss.item()}, perplexity: {np.exp(loss.item())}",
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()


def main():
    # Chinese corpus: smaller vocab (21128) than bert-base-uncased (30522).
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    context_length = 128
    batch_size = 4

    dataset = PretrainDataset(
        "data/train.jsonl",
        tokenizer,
        max_length=context_length,
    )
    dataloader = PretrainDataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        d_model=32,
        num_layers=4,
        num_heads=8,
        d_ff=64,
        rope_theta=10000.0,
    )
    print("total parameters: ", sum(p.numel() for p in model.parameters()))
    print(
        "approx activation memory per step (logits only): "
        f"{batch_size * context_length * tokenizer.vocab_size * 4 / 1e6:.1f} MB"
    )

    trainer = Trainer(model, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()
