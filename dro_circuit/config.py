"""Configuration dataclasses for DRO circuit discovery experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "gpt2"
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class CorruptionConfig:
    families: List[str] = field(default_factory=lambda: ["S2_IO", "IO_RAND", "S_RAND"])


@dataclass
class ScoringConfig:
    method: Literal["EAP", "EAP-IG-inputs", "EAP-IG-activations", "clean-corrupted"] = (
        "EAP-IG-inputs"
    )
    ig_steps: int = 5
    intervention: Literal["patching", "zero", "mean"] = "patching"
    aggregation: str = "sum"
    batch_size: int = 32


@dataclass
class DROConfig:
    aggregator: Literal["max", "cvar", "softmax"] = "max"
    cvar_alpha: float = 0.5
    softmax_temperature: float = 1.0


@dataclass
class SelectionConfig:
    n_edges: int = 200
    selection_method: Literal["topn", "greedy"] = "topn"
    absolute: bool = True


@dataclass
class PlanBConfig:
    lr: float = 1e-2
    n_outer_steps: int = 200
    reg_type: Literal["L0", "L1"] = "L1"
    reg_lambda: float = 0.01
    temperature: float = 0.1
    adversary_temperature: float = 1.0


@dataclass
class EvalConfig:
    intervention: Literal["patching", "zero", "mean"] = "patching"
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    dro: DROConfig = field(default_factory=DROConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    plan_b: PlanBConfig = field(default_factory=PlanBConfig)
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
            ("plan_b", PlanBConfig),
            ("eval", EvalConfig),
        ]:
            if section_name in raw:
                setattr(config, section_name, section_cls(**raw[section_name]))

        for key in ["task", "n_examples", "seed", "output_dir"]:
            if key in raw:
                setattr(config, key, raw[key])

        return config
