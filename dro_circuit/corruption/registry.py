"""Task → corruption family registry."""

from typing import Dict, List

from dro_circuit.corruption.base import CorruptionFamily

_REGISTRY: Dict[str, Dict[str, CorruptionFamily]] = {}


def register_corruptions(task_name: str, families: Dict[str, CorruptionFamily]):
    _REGISTRY[task_name] = families


def get_corruptions(task_name: str, family_names: List[str]) -> List[CorruptionFamily]:
    task_families = _REGISTRY[task_name]
    return [task_families[name] for name in family_names]


def list_corruptions(task_name: str) -> List[str]:
    return list(_REGISTRY.get(task_name, {}).keys())
