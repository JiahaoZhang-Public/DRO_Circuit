"""Tests for multi-corrupt dataset and EAP adapter."""

import torch

from dro_circuit.data.eap_adapter import (
    SingleCorruptDataset,
    collate_eap,
    make_all_eap_dataloaders,
    make_eap_dataloader,
)
from dro_circuit.data.multi_corrupt_dataset import (
    MultiCorruptBatch,
    MultiCorruptDataset,
    collate_multi_corrupt,
)


def _make_toy_dataset(n=10, k=3):
    """Create a toy MultiCorruptDataset for testing."""
    clean = [f"clean sentence {i}" for i in range(n)]
    corruptions = {}
    for j in range(k):
        name = f"corrupt_{j}"
        corruptions[name] = [f"corrupt_{j} sentence {i}" for i in range(n)]
    labels = torch.randint(0, 100, (n, 2))
    return MultiCorruptDataset(clean, corruptions, labels)


class TestMultiCorruptDataset:
    def test_basic_properties(self):
        ds = _make_toy_dataset(n=10, k=3)
        assert len(ds) == 10
        assert ds.n_corruptions == 3
        assert sorted(ds.corruption_names) == ["corrupt_0", "corrupt_1", "corrupt_2"]

    def test_getitem(self):
        ds = _make_toy_dataset(n=10, k=3)
        item = ds[0]
        assert item.clean_string == "clean sentence 0"
        assert len(item.corrupted_strings) == 3
        assert item.corrupted_strings["corrupt_0"] == "corrupt_0 sentence 0"
        assert item.label.shape == (2,)
        assert item.index == 0

    def test_collate(self):
        ds = _make_toy_dataset(n=10, k=3)
        items = [ds[i] for i in range(4)]
        batch = collate_multi_corrupt(items)

        assert isinstance(batch, MultiCorruptBatch)
        assert batch.batch_size == 4
        assert batch.n_corruptions == 3
        assert len(batch.clean_strings) == 4
        assert len(batch.corrupted_strings["corrupt_0"]) == 4
        assert batch.labels.shape == (4, 2)

    def test_get_single_corruption(self):
        ds = _make_toy_dataset(n=10, k=3)
        items = [ds[i] for i in range(4)]
        batch = collate_multi_corrupt(items)

        clean, corrupt, labels = batch.get_single_corruption("corrupt_1")
        assert len(clean) == 4
        assert len(corrupt) == 4
        assert corrupt[0] == "corrupt_1 sentence 0"

    def test_dataloader(self):
        ds = _make_toy_dataset(n=10, k=3)
        dl = ds.to_dataloader(batch_size=4)
        batches = list(dl)
        assert len(batches) == 3  # 10 / 4 = 2.5, ceil = 3
        assert batches[0].batch_size == 4
        assert batches[-1].batch_size == 2  # last batch


class TestEAPAdapter:
    def test_single_corrupt_dataset(self):
        ds = _make_toy_dataset(n=10, k=3)
        single = SingleCorruptDataset(ds, "corrupt_0")
        assert len(single) == 10

        clean, corrupt, label = single[0]
        assert clean == "clean sentence 0"
        assert corrupt == "corrupt_0 sentence 0"
        assert label.shape == (2,)

    def test_collate_eap(self):
        batch = [
            ("clean 0", "corrupt 0", torch.tensor([1, 2])),
            ("clean 1", "corrupt 1", torch.tensor([3, 4])),
        ]
        clean, corrupt, labels = collate_eap(batch)
        assert clean == ["clean 0", "clean 1"]
        assert corrupt == ["corrupt 0", "corrupt 1"]
        assert labels.shape == (2, 2)

    def test_make_eap_dataloader(self):
        ds = _make_toy_dataset(n=10, k=3)
        dl = make_eap_dataloader(ds, "corrupt_0", batch_size=4)
        batch = next(iter(dl))
        clean, corrupt, labels = batch
        assert len(clean) == 4
        assert len(corrupt) == 4
        assert labels.shape == (4, 2)

    def test_make_all_dataloaders(self):
        ds = _make_toy_dataset(n=10, k=3)
        dls = make_all_eap_dataloaders(ds, batch_size=4)
        assert len(dls) == 3
        assert set(dls.keys()) == {"corrupt_0", "corrupt_1", "corrupt_2"}


class TestValidation:
    def test_mismatched_lengths_raises(self):
        clean = ["a", "b", "c"]
        corruptions = {"c1": ["x", "y"]}  # Wrong length
        labels = torch.zeros(3, 2)
        try:
            MultiCorruptDataset(clean, corruptions, labels)
            assert False, "Should have raised"
        except AssertionError:
            pass

    def test_mismatched_labels_raises(self):
        clean = ["a", "b", "c"]
        corruptions = {"c1": ["x", "y", "z"]}
        labels = torch.zeros(2, 2)  # Wrong length
        try:
            MultiCorruptDataset(clean, corruptions, labels)
            assert False, "Should have raised"
        except AssertionError:
            pass
