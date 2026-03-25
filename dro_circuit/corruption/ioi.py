"""IOI-specific corruption families wrapping ACDC's gen_flipped_prompts."""

import sys
from pathlib import Path
from typing import Tuple, Union

# Vendor import for ACDC
_VENDOR_ROOT = str(Path(__file__).resolve().parents[2] / "vendor" / "Automatic-Circuit-Discovery")
if _VENDOR_ROOT not in sys.path:
    sys.path.insert(0, _VENDOR_ROOT)

from acdc.ioi.ioi_dataset import IOIDataset

from dro_circuit.corruption.base import CorruptionFamily, CorruptionResult


class IOICorruptionFamily(CorruptionFamily):
    """Wraps IOIDataset.gen_flipped_prompts for a specific flip type."""

    def __init__(self, flip_type: Union[Tuple[str, str], str], seed: int = 0):
        self._flip_type = flip_type
        self._seed = seed

    def name(self) -> str:
        if isinstance(self._flip_type, tuple):
            return f"ioi_{self._flip_type[0]}_{self._flip_type[1]}"
        return f"ioi_{self._flip_type}"

    def generate(self, clean_dataset: IOIDataset, **kwargs) -> CorruptionResult:
        seed = kwargs.get("seed", self._seed)
        corrupted_dataset = clean_dataset.gen_flipped_prompts(self._flip_type, seed=seed)
        return CorruptionResult(
            corrupted_strings=corrupted_dataset.sentences,
            corruption_name=self.name(),
            metadata={
                "toks": corrupted_dataset.toks,
                "io_tokenIDs": corrupted_dataset.io_tokenIDs,
                "s_tokenIDs": corrupted_dataset.s_tokenIDs,
            },
        )


# Pre-defined IOI corruption families
IOI_CORRUPTIONS = {
    "S2_IO": IOICorruptionFamily(("S2", "IO"), seed=42),
    "IO_RAND": IOICorruptionFamily(("IO", "RAND"), seed=42),
    "S_RAND": IOICorruptionFamily(("S", "RAND"), seed=42),
    "S1_RAND": IOICorruptionFamily(("S1", "RAND"), seed=42),
    "IO_S1": IOICorruptionFamily(("IO", "S1"), seed=42),
    "S_IO": IOICorruptionFamily(("S", "IO"), seed=42),
}
