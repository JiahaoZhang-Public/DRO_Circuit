"""IOI task: model loading, dataset generation, corruption setup, and metrics."""

import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

_VENDOR_ACDC = str(Path(__file__).resolve().parents[2] / "vendor" / "Automatic-Circuit-Discovery")
_VENDOR_EAP = str(Path(__file__).resolve().parents[2] / "vendor" / "EAP-IG" / "src")
for p in [_VENDOR_ACDC, _VENDOR_EAP]:
    if p not in sys.path:
        sys.path.insert(0, p)

from acdc.ioi.ioi_dataset import IOIDataset
from eap.graph import Graph
from transformer_lens import HookedTransformer

from dro_circuit.corruption.ioi import IOI_CORRUPTIONS
from dro_circuit.data.multi_corrupt_dataset import MultiCorruptDataset
from dro_circuit.evaluation.metrics import logit_diff, logit_diff_loss


class IOITask:
    """
    Indirect Object Identification task configuration.

    Encapsulates model loading, dataset generation, corruption family setup,
    and metric definition for IOI.
    """

    def __init__(
        self,
        n_examples: int = 100,
        device: str = "cuda",
        seed: int = 42,
        corruption_families: Optional[List[str]] = None,
    ):
        self.n_examples = n_examples
        self.device = device
        self.seed = seed
        self._corruption_family_names = corruption_families or [
            "S2_IO",
            "IO_RAND",
            "S_RAND",
        ]

    def load_model(self) -> HookedTransformer:
        """Load GPT-2 with required hooks enabled."""
        model = HookedTransformer.from_pretrained("gpt2", device=self.device)
        model.set_use_attn_result(True)
        model.set_use_split_qkv_input(True)
        model.set_use_hook_mlp_in(True)
        return model

    def build_dataset(
        self, tokenizer=None
    ) -> Tuple[MultiCorruptDataset, IOIDataset]:
        """
        Build multi-corruption dataset for IOI.

        Returns:
            (multi_corrupt_dataset, raw_ioi_dataset)
        """
        clean_ds = IOIDataset(
            prompt_type="mixed",
            N=self.n_examples,
            tokenizer=tokenizer,
            seed=self.seed,
        )

        # Labels: (correct=IO token, incorrect=S token)
        labels = torch.stack(
            [
                torch.tensor(clean_ds.io_tokenIDs),
                torch.tensor(clean_ds.s_tokenIDs),
            ],
            dim=1,
        )  # (N, 2)

        # Generate corruptions
        corrupted_strings = {}
        for family_name in self._corruption_family_names:
            family = IOI_CORRUPTIONS[family_name]
            result = family.generate(clean_ds, seed=self.seed)
            corrupted_strings[family_name] = result.corrupted_strings

        multi_ds = MultiCorruptDataset(
            clean_strings=clean_ds.sentences,
            corrupted_strings=corrupted_strings,
            labels=labels,
        )

        return multi_ds, clean_ds

    def get_scoring_metric(self) -> Callable:
        """Metric for EAP-IG attribution (loss form, mean reduced)."""
        return logit_diff_loss

    def get_eval_metric(self) -> Callable:
        """Metric for circuit evaluation (loss form for evaluate_graph)."""
        return logit_diff_loss
