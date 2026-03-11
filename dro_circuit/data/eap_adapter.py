"""Adapts MultiCorruptDataset to EAP-IG's expected DataLoader format."""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset


class SingleCorruptDataset(Dataset):
    """Wraps MultiCorruptDataset for one specific corruption, yielding EAP-compatible tuples."""

    def __init__(self, multi_dataset: MultiCorruptDataset, corruption_name: str):
        self._multi = multi_dataset
        self._corruption_name = corruption_name

    def __len__(self):
        return len(self._multi)

    def __getitem__(self, idx) -> Tuple[str, str, torch.Tensor]:
        return (
            self._multi.clean_strings[idx],
            self._multi.corrupted_strings[self._corruption_name][idx],
            self._multi.labels[idx],
        )


def collate_eap(batch: List[Tuple[str, str, torch.Tensor]]):
    """Collate into EAP-IG format: (clean_list, corrupt_list, label_tensor)."""
    clean, corrupt, labels = zip(*batch)
    return list(clean), list(corrupt), torch.stack(labels)


def make_eap_dataloader(
    multi_dataset: MultiCorruptDataset,
    corruption_name: str,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Create an EAP-IG compatible DataLoader for a single corruption variant."""
    ds = SingleCorruptDataset(multi_dataset, corruption_name)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_eap)


def make_all_eap_dataloaders(
    multi_dataset: MultiCorruptDataset,
    batch_size: int,
) -> Dict[str, DataLoader]:
    """Create one DataLoader per corruption variant."""
    return {
        name: make_eap_dataloader(multi_dataset, name, batch_size)
        for name in multi_dataset.corruption_names
    }
