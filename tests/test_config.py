"""Tests for configuration."""

import os
import tempfile

import yaml

from dro_circuit.config import ExperimentConfig


class TestExperimentConfig:
    def test_defaults(self):
        config = ExperimentConfig()
        assert config.task == "ioi"
        assert config.model.name == "gpt2"
        assert config.dro.aggregator == "max"
        assert config.selection.n_edges == 200

    def test_from_yaml(self):
        yaml_content = {
            "task": "ioi",
            "n_examples": 50,
            "seed": 123,
            "model": {"name": "gpt2", "device": "cpu"},
            "dro": {"aggregator": "cvar", "cvar_alpha": 0.3},
            "selection": {"n_edges": 100, "selection_method": "greedy"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            path = f.name

        try:
            config = ExperimentConfig.from_yaml(path)
            assert config.task == "ioi"
            assert config.n_examples == 50
            assert config.seed == 123
            assert config.model.device == "cpu"
            assert config.dro.aggregator == "cvar"
            assert config.dro.cvar_alpha == 0.3
            assert config.selection.n_edges == 100
            assert config.selection.selection_method == "greedy"
        finally:
            os.unlink(path)
