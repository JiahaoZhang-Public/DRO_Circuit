"""Configuration dataclasses for DRO circuit discovery experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    """Transformer model specification."""

    name: str = "gpt2"
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class CorruptionConfig:
    """Which corruption families to use for scoring and evaluation."""

    families: List[str] = field(default_factory=lambda: ["S2_IO", "IO_RAND", "S_RAND"])


@dataclass
class ScoringConfig:
    """EAP edge attribution parameters."""

    method: Literal["EAP", "EAP-IG-inputs", "EAP-IG-activations", "clean-corrupted"] = (
        "EAP"
    )
    ig_steps: int = 5
    intervention: Literal["patching", "zero", "mean"] = "patching"
    aggregation: str = "sum"
    batch_size: int = 32


@dataclass
class DROConfig:
    """DRO aggregation strategy over corruption scores.

    Attributes:
        aggregator: Which aggregation rule to use.
        cvar_alpha: CVaR parameter. 0→max (worst-case), 1→mean. Only used when aggregator='cvar'.
        softmax_temperature: Softmax temperature. 0→max, ∞→mean. Only used when aggregator='softmax'.
    """

    aggregator: Literal["max", "mean", "local_dro", "cvar", "softmax"] = "max"
    cvar_alpha: float = 0.5
    softmax_temperature: float = 1.0


@dataclass
class SelectionConfig:
    """Circuit edge selection parameters."""

    n_edges: int = 200
    selection_method: Literal["topn", "greedy"] = "topn"
    absolute: bool = True


@dataclass
class EvalConfig:
    """Evaluation parameters for activation patching."""

    intervention: Literal["patching", "zero", "mean"] = "patching"
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration, composing all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    task: str = "ioi"
    n_examples: int = 100
    seed: int = 42
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls()
        for section_name, section_cls in [
            ("model", ModelConfig),
            ("corruption", CorruptionConfig),
            ("scoring", ScoringConfig),
            ("dro", DROConfig),
            ("selection", SelectionConfig),
            ("eval", EvalConfig),
        ]:
            if section_name in raw:
                setattr(config, section_name, section_cls(**raw[section_name]))

        for key in ["task", "n_examples", "seed", "output_dir"]:
            if key in raw:
                setattr(config, key, raw[key])

        return config
