"""AirLLM v4 — Persistence layer.

Unified ModelPersister with platform auto-detection.
SafetensorModelPersister uses mmap for near-zero-copy loading.
"""

from pathlib import Path
import os

_persister = None


class ModelPersister:
    """Base class + factory for platform-specific persistence."""

    @classmethod
    def get_model_persister(cls):
        global _persister
        if _persister is not None:
            return _persister
        from sys import platform
        if platform == "darwin":
            from .mlx_model_persister import MlxModelPersister
            _persister = MlxModelPersister()
        else:
            _persister = SafetensorModelPersister()
        return _persister

    def model_persist_exist(self, layer_name, saving_path):
        raise NotImplementedError

    def persist_model(self, state_dict, layer_name, path):
        raise NotImplementedError

    def load_model(self, layer_name, path):
        raise NotImplementedError


class SafetensorModelPersister(ModelPersister):
    """Safetensor persistence with done-marker atomicity."""

    def model_persist_exist(self, layer_name, saving_path):
        sp = Path(saving_path)
        return ((sp / f"{layer_name}safetensors").exists() and
                (sp / f"{layer_name}safetensors.done").exists())

    def persist_model(self, state_dict, layer_name, saving_path):
        from safetensors.torch import save_file
        sp = Path(saving_path)
        save_file(state_dict, sp / f"{layer_name}safetensors")
        (sp / f"{layer_name}safetensors.done").touch()

    def load_model(self, layer_name, path):
        from safetensors.torch import load_file
        return load_file(str(Path(path) / f"{layer_name}.safetensors"), device="cpu")
