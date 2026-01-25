"""Dataset classes for text classification."""

from typing import List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification.

    Args:
        texts: List of text samples
        labels: List of integer labels
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Example:
        >>> dataset = TextClassificationDataset(
        ...     texts=["Hello world", "Goodbye"],
        ...     labels=[0, 1],
        ...     tokenizer=tokenizer,
        ...     max_length=512
        ... )
        >>> sample = dataset[0]
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class StreamingTextDataset(Dataset):
    """
    Memory-efficient dataset that loads data on-demand.

    Useful for large datasets that don't fit in memory.

    Args:
        file_path: Path to data file (CSV or JSON)
        tokenizer: HuggingFace tokenizer
        text_column: Name of text column
        label_column: Name of label column
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        label_column: str = "label",
        max_length: int = 512,
    ):
        import pandas as pd

        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

        # Load only metadata for indexing
        if file_path.endswith(".csv"):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            self.data = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        text = row[self.text_column]
        label = row[self.label_column]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
