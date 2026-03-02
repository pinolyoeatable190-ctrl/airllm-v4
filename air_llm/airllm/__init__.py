"""AirLLM v4 — Package init. Clean, minimal exports."""

from sys import platform as _plat

if _plat == "darwin":
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .airllm_base import AirLLMBaseModel, AirLLMLlama2
    from .auto_model import AutoModel
    from .utils import split_and_save_layers, NotEnoughSpaceException
