"""Core data abstraction: each clean input paired with K corruption variants."""

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class MultiCorruptBatch:
    """A batch of examples, each with K corruption variants."""

    clean_strings: List[str]
    corrupted_strings: Dict[str, List[str]]  # corruption_name -> batch of strings
    labels: torch.Tensor  # (B, 2)
    corruption_names: List[str]

    @property
    def batch_size(self) -> int:
        return len(self.clean_strings)

    @property
    def n_corruptions(self) -> int:
        return len(self.corruption_names)

    def get_single_corruption(self, corruption_name: str):
        """Extract (clean, corrupted, labels) for one corruption -- EAP-compatible."""
        return (
            self.clean_strings,
            self.corrupted_strings[corruption_name],
            self.labels,
        )


@dataclass
class MultiCorruptExample:
    """A single clean input with K corruption variants."""

    clean_string: str
    corrupted_strings: Dict[str, str]
    label: torch.Tensor
    index: int = 0


class MultiCorruptDataset(Dataset):
    """
    Dataset pairing each clean input with K corruption variants.

    Args:
        clean_strings: List of N clean input strings.
        corrupted_strings: Dict mapping corruption_name -> list of N corrupted strings.
        labels: Tensor (N, 2) with [correct_token_id, incorrect_token_id].
    """

    def __init__(
        self,
        clean_strings: List[str],
        corrupted_strings: Dict[str, List[str]],
        labels: torch.Tensor,
    ):
        self.clean_strings = clean_strings
        self.corrupted_strings = corrupted_strings
        self.labels = labels
        self._corruption_names = sorted(corrupted_strings.keys())

        N = len(clean_strings)
        assert labels.shape[0] == N
        for name, corr in corrupted_strings.items():
            assert len(corr) == N, f"Corruption {name} has {len(corr)} != {N} examples"

    def __len__(self) -> int:
        return len(self.clean_strings)

    def __getitem__(self, idx: int) -> MultiCorruptExample:
        return MultiCorruptExample(
            clean_string=self.clean_strings[idx],
            corrupted_strings={
                name: self.corrupted_strings[name][idx] for name in self._corruption_names
            },
            label=self.labels[idx],
            index=idx,
        )

    @property
    def corruption_names(self) -> List[str]:
        return self._corruption_names

    @property
    def n_corruptions(self) -> int:
        return len(self._corruption_names)

    def to_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_multi_corrupt,
        )


def collate_multi_corrupt(examples: List[MultiCorruptExample]) -> MultiCorruptBatch:
    corruption_names = sorted(examples[0].corrupted_strings.keys())
    return MultiCorruptBatch(
        clean_strings=[ex.clean_string for ex in examples],
        corrupted_strings={
            name: [ex.corrupted_strings[name] for ex in examples] for name in corruption_names
        },
        labels=torch.stack([ex.label for ex in examples]),
        corruption_names=corruption_names,
    )
