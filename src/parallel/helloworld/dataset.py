"""
HuggingFace Datasets Server Dataset

This module provides a class to fetch data from HuggingFace datasets-server API.
The datasets-server API allows streaming data without downloading the full dataset.
"""

import time
from typing import Any, Dict, Iterator, Optional

import numpy as np
import requests
from torch.utils.data import IterableDataset


class HuggingFaceDatasetServer(IterableDataset):
    rows_base_url: str = "https://datasets-server.huggingface.co/rows"
    size_base_url: str = "https://datasets-server.huggingface.co/size"

    def __init__(
        self,
        dataset_name: str = "mlfoundations/dclm-baseline-1.0-parquet",
        start: int = 0,
        end: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.start = start
        if end is None:
            self.end = self._get_dataset_size()
        else:
            self.end = end
        self.limit = 10

    def _get_dataset_size(self) -> int:

        url = f"{self.size_base_url}?dataset={self.dataset_name}"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("size").get("dataset").get("num_rows")
        except Exception as e:
            print(f"Warning: Could not fetch dataset size: {e}")
            raise e

    def _fetch_rows(self, start: int, num_rows: int) -> Dict[str, Any]:
        url = f"{self.rows_base_url}?dataset={self.dataset_name}&offset={start}&length={num_rows}"
        params = {
            "config": "default",
            "split": "train",
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from {url}: {e}")

    def __iter__(self) -> Iterator[list[str]]:
        while True:
            result = list[str]()
            start = np.random.randint(self.start, self.end - self.limit)
            print("start position is: ", start)

            # Fetch batch of rows
            data = self._fetch_rows(start, self.limit)
            for item in data.get("rows", []):
                # print("+" * 70)
                result.append(item.get("row").get("text"))
            yield result

            # Small delay to avoid rate limiting
            time.sleep(0.1)


# ============================================================================
# Example Usage
# ============================================================================

# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
#     vocab_size = tokenizer.vocab_size
#     print("vocab size is: ", vocab_size)
#     dataset = HuggingFaceDatasetServer(
#         dataset_name="mlfoundations/dclm-baseline-1.0-parquet", start=0, end=None
#     )

#     print("dataset size is: ", dataset.end)

#     index = 0
#     for rows in dataset:
#         # print("=" * 70)
#         for row in rows:
#             result = tokenizer(row, truncation=True)["input_ids"]
#             print(len(result))
#             length = ((len(result) - 1) // 10) * 10
#             result = result[: length + 1]
#             print(type(result))
#             data_tensor = torch.tensor(result)
#             data_tensor = data_tensor.view(-1, 10)
#             print(data_tensor.shape)
#         if index > 1:
#             break
#         index += 1
