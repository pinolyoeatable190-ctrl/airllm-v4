"""AirLLM v4 — AutoModel: unified factory, no per-architecture files needed.

All variants are handled by AirLLMBaseModel's internal _variant detection.
This file exists purely for API compatibility.
"""

from sys import platform

_IS_MAC = platform == "darwin"

if _IS_MAC:
    from .airllm_llama_mlx import AirLLMLlamaMlx


class AutoModel:
    """Factory: use AutoModel.from_pretrained(...) to instantiate."""

    def __init__(self):
        raise EnvironmentError(
            "Use AutoModel.from_pretrained(pretrained_model_name_or_path)")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if _IS_MAC:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *args, **kwargs)
        from .airllm_base import AirLLMBaseModel
        return AirLLMBaseModel(pretrained_model_name_or_path, *args, **kwargs)
