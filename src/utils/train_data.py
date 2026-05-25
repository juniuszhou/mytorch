import os

from datasets import load_dataset, load_from_disk

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


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
