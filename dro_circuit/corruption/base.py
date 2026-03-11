"""Abstract base class for corruption families."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CorruptionResult:
    """One corruption variant for a set of clean inputs."""

    corrupted_strings: List[str]
    corruption_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CorruptionFamily(ABC):
    """Generates corruption variants for clean inputs."""

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(self, clean_dataset: Any, **kwargs) -> CorruptionResult:
        ...
