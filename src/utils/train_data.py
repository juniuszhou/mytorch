import os

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# PyTorch Dataset wrapper around a Hugging Face dataset
class CustomDataset(Dataset):
    def __init__(self, dataset: HFDataset, tokenizer, max_length: int = 128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        encoded = self.tokenizer(
            sample["text"],
            # return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            device="cuda",
        )
        # print("encoded: ", encoded["input_ids"].shape)
        return torch.tensor(encoded["input_ids"])
        # return {
        #     "input_ids": encoded["input_ids"].squeeze(0),
        #     "attention_mask": encoded["attention_mask"].squeeze(0),
        # }


# customized collate function
def collate_batch(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


def load_shakespeare_text() -> str:
    """Load Tiny Shakespeare without deprecated dataset loading scripts."""
    if os.path.exists("data/tiny_shakespeare"):
        dataset = load_from_disk("data/tiny_shakespeare")
        return "\n".join(dataset["text"])
    else:
        dataset = load_dataset(
            "text",
            data_files={"train": SHAKESPEARE_URL},
            split="train",
        )
        dataset.save_to_disk("data/tiny_shakespeare")
    return "\n".join(dataset["text"])


def get_shakespeare_dataset() -> HFDataset:
    """Load Tiny Shakespeare without deprecated dataset loading scripts."""
    if os.path.exists("data/tiny_shakespeare"):
        dataset = load_from_disk("data/tiny_shakespeare")
        return dataset
    else:
        dataset = load_dataset(
            "text",
            data_files={"train": SHAKESPEARE_URL},
            split="train",
        )
        dataset.save_to_disk("data/tiny_shakespeare")
        return dataset


def get_shakespeare_data_loader() -> DataLoader:
    """Load Tiny Shakespeare without deprecated dataset loading scripts."""
    hf_dataset = get_shakespeare_dataset()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = CustomDataset(hf_dataset, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        # collate_fn=collate_batch,
    )

    return loader


def load_train_data(dataset_name: str):
    if dataset_name in {"karpathy/tiny_shakespeare", "tiny_shakespeare"}:
        return load_shakespeare_text()
    dataset = load_dataset(dataset_name)
    return dataset


def load_test_data(dataset_name: str):
    dataset = load_dataset(dataset_name)
    return dataset


def load_data(dataset_name: str):
    dataset = load_dataset(dataset_name)
    return dataset


def main():
    loader = get_shakespeare_data_loader()

    for i, batch in enumerate(loader):
        print("-" * 100)
        print("input_ids shape:", batch.shape)
        print("input_ids: ", batch)
        if i >= 2:
            break


# if __name__ == "__main__":
#     main()
