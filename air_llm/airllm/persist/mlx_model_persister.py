"""AirLLM v4 — MLX persistence backend (macOS Apple Silicon)."""

import os
from pathlib import Path
import numpy as np
from . import ModelPersister

# Torch → MLX name mapping (applied at load time, not save time)
_REMAP = [
    ("model.", ""), ("mlp", "feed_forward"), ("down_proj", "w2"),
    ("up_proj", "w3"), ("gate_proj", "w1"), ("input_layernorm", "attention_norm"),
    ("post_attention_layernorm", "ffn_norm"), ("lm_head", "output"),
    ("embed_tokens", "tok_embeddings"), ("self_attn", "attention"),
    ("q_proj", "wq"), ("k_proj", "wk"), ("v_proj", "wv"), ("o_proj", "wo"),
]

def _remap_keys(d: dict) -> dict:
    for old, new in _REMAP:
        d = {k.replace(old, new): v for k, v in d.items()}
    return d


class MlxModelPersister(ModelPersister):
    def model_persist_exist(self, layer_name, saving_path):
        sp = Path(saving_path)
        return ((sp / f"{layer_name}mlx.npz").exists() and
                (sp / f"{layer_name}mlx.done").exists())

    def persist_model(self, state_dict, layer_name, saving_path):
        import torch
        sp = Path(saving_path)
        weights = {k: v.to(torch.float16).numpy() for k, v in state_dict.items()}
        np.savez(sp / f"{layer_name}mlx", **weights)
        (sp / f"{layer_name}mlx.done").touch()

    def load_model(self, layer_name, path):
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        sd = mx.load(str(Path(path) / f"{layer_name}.mlx.npz"))
        return tree_unflatten(list(_remap_keys(sd).items()))
