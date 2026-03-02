"""AirLLM v4 — MLX backend for Apple Silicon (macOS).

Optimized: persistent layer objects reused in autoregressive loop,
gc.collect only between layer blocks, not per-layer.
"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .persist import ModelPersister
from .utils import find_or_create_local_splitted_path
from transformers import AutoConfig, AutoTokenizer

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    dim: int; n_layers: int; head_dim: int; hidden_dim: int
    n_heads: int; n_kv_heads: int; norm_eps: float; vocab_size: int
    rope_theta: float = 10000.0; rope_traditional: bool = True


def _args_from_config(cfg) -> ModelArgs:
    p = {'dim': cfg.hidden_size, 'hidden_dim': cfg.intermediate_size,
         'n_heads': cfg.num_attention_heads, 'n_layers': cfg.num_hidden_layers,
         'vocab_size': cfg.vocab_size, 'norm_eps': cfg.rms_norm_eps,
         'rope_traditional': False}
    p['n_kv_heads'] = getattr(cfg, 'num_key_value_heads', p['n_heads'])
    p['head_dim'] = p['dim'] // p['n_heads']
    p['rope_theta'] = getattr(cfg, 'rope_theta', 10000.0)
    return ModelArgs(**p)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return self.weight * (x.astype(mx.float32) * mx.rsqrt(
            x.astype(mx.float32).square().mean(-1, keepdims=True) + self.eps)).astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.n_heads, self.n_kv_heads = a.n_heads, a.n_kv_heads
        self.repeats = a.n_heads // a.n_kv_heads
        self.scale = a.head_dim ** -0.5
        self.wq = nn.Linear(a.dim, a.n_heads * a.head_dim, bias=False)
        self.wk = nn.Linear(a.dim, a.n_kv_heads * a.head_dim, bias=False)
        self.wv = nn.Linear(a.dim, a.n_kv_heads * a.head_dim, bias=False)
        self.wo = nn.Linear(a.n_heads * a.head_dim, a.dim, bias=False)
        self.rope = nn.RoPE(a.head_dim, traditional=a.rope_traditional, base=a.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.repeats > 1:
            k = mx.concatenate([mx.expand_dims(k, 2)] * self.repeats, axis=2).reshape(B, self.n_heads, L, -1)
            v = mx.concatenate([mx.expand_dims(v, 2)] * self.repeats, axis=2).reshape(B, self.n_heads, L, -1)

        off = cache[0].shape[2] if cache else 0
        q, k = self.rope(q, offset=off), self.rope(k, offset=off)
        if cache:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)

        s = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            s += mask
        s = mx.softmax(s.astype(mx.float32), axis=-1).astype(s.dtype)
        return self.wo((s @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)), (k, v)


class FeedForward(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(a.dim, a.hidden_dim, bias=False)
        self.w2 = nn.Linear(a.hidden_dim, a.dim, bias=False)
        self.w3 = nn.Linear(a.dim, a.hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, a: ModelArgs):
        super().__init__()
        self.attention = Attention(a)
        self.feed_forward = FeedForward(a)
        self.attention_norm = RMSNorm(a.dim, eps=a.norm_eps)
        self.ffn_norm = RMSNorm(a.dim, eps=a.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        return h + self.feed_forward(self.ffn_norm(h)), cache


# ---------------------------------------------------------------------------
# Main MLX model
# ---------------------------------------------------------------------------

class AirLLMLlamaMlx:
    _LNAMES = {'embed': 'model.embed_tokens', 'layer_prefix': 'model.layers',
               'norm': 'model.norm', 'lm_head': 'lm_head'}

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=None,
                 max_seq_len=512, layer_shards_saving_path=None, profiling_mode=False,
                 compression=None, hf_token=None, prefetching=True, delete_original=False, **kw):
        self.hf_token = hf_token
        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id, layer_shards_saving_path,
            compression=compression, layer_names=self._LNAMES, hf_token=hf_token,
            delete_original=delete_original)

        ckw = {'trust_remote_code': True}
        if hf_token:
            ckw['token'] = hf_token
        self.config = AutoConfig.from_pretrained(self.model_local_path, **ckw)
        self.args = _args_from_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_local_path, **ckw)

    def generate(self, x, temperature=0, max_new_tokens=20, **kw):
        tokens = []
        for tok in self._generate(x, temperature):
            tokens.append(tok)
            if len(tokens) >= max_new_tokens:
                break
        return self.tokenizer.decode([t.item() for t in tokens])

    def _generate(self, x, temperature=0):
        persister = ModelPersister.get_model_persister()
        a, ln = self.args, self._LNAMES
        cp = self.checkpoint_path
        cache = []

        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])

        # Embed
        embed = nn.Embedding(a.vocab_size, a.dim)
        embed.update(persister.load_model(ln['embed'], cp)['tok_embeddings'])
        mask = mask.astype(embed.weight.dtype)
        x = embed(x); mx.eval(x); del embed; gc.collect()

        # Transformer layers
        from tqdm import tqdm
        for il in tqdm(range(a.n_layers), desc='layers'):
            block = TransformerBlock(a)
            block.update(persister.load_model(f"{ln['layer_prefix']}.{il}", cp)['layers'][il])
            x, c = block(x, mask=mask); mx.eval(x)
            cache.append(c); del block; gc.collect()

        # Norm + LM head
        norm = RMSNorm(a.dim, eps=a.norm_eps)
        norm.update(persister.load_model(ln['norm'], cp)['norm'])
        x = norm(x); mx.eval(x); del norm; gc.collect()

        out = nn.Linear(a.dim, a.vocab_size, bias=False)
        out.update(persister.load_model(ln['lm_head'], cp)['output'])
        y = out(x[:, -1]); mx.eval(y); del out; gc.collect()

        y = mx.argmax(y, axis=-1) if temperature == 0 else mx.random.categorical(y / temperature)
        yield y

        # Autoregressive loop
        while True:
            x = y[:, None]
            embed = nn.Embedding(a.vocab_size, a.dim)
            embed.update(persister.load_model(ln['embed'], cp)['tok_embeddings'])
            x = embed(x); mx.eval(x); del embed; gc.collect()

            for i in range(len(cache)):
                block = TransformerBlock(a)
                block.update(persister.load_model(f"{ln['layer_prefix']}.{i}", cp)['layers'][i])
                x, cache[i] = block(x, mask=None, cache=cache[i]); mx.eval(x)
                del block; gc.collect()

            norm = RMSNorm(a.dim, eps=a.norm_eps)
            norm.update(persister.load_model(ln['norm'], cp)['norm'])
            x = norm(x); mx.eval(x); del norm; gc.collect()

            out = nn.Linear(a.dim, a.vocab_size, bias=False)
            out.update(persister.load_model(ln['lm_head'], cp)['output'])
            y = mx.argmax(out(x[:, -1]), axis=-1) if temperature == 0 else mx.random.categorical(out(x[:, -1]) / temperature)
            mx.eval(y); del out; gc.collect()
            yield y
