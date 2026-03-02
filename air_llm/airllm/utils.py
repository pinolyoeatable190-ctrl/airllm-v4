"""AirLLM v4 — Utils: I/O, compression, memory, layer splitting."""

import gc, json, os, ctypes, shutil, time
from pathlib import Path
from glob import glob
from typing import Optional
import torch
from safetensors.torch import load_file, save_file
from .persist import ModelPersister

try:
    import bitsandbytes as bnb
    _BNB = True
except ImportError:
    _BNB = False

import huggingface_hub

# --- Memory management (debounced) ---
_gc_n = 0

def clean_memory(force=False):
    global _gc_n
    _gc_n += 1
    if force or _gc_n >= 4:
        gc.collect()
        try: ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception: pass
        _gc_n = 0
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# --- Compression ---

def _bnb_qs_to_dict(qs):
    d = {
        'quant_type': qs.quant_type, 'absmax': qs.absmax,
        'blocksize': qs.blocksize, 'quant_map': qs.code,
        'dtype': str(qs.dtype).replace('torch.', ''), 'shape': tuple(qs.shape),
    }
    if qs.nested:
        d.update({
            'nested_absmax': qs.state2.absmax, 'nested_blocksize': qs.state2.blocksize,
            'nested_quant_map': qs.state2.code,
            'nested_dtype': str(qs.state2.dtype).replace('torch.', ''),
            'nested_offset': qs.offset.item(),
        })
    packed = {k: v for k, v in d.items() if isinstance(v, torch.Tensor)}
    non_t = {k: v for k, v in d.items() if not isinstance(v, torch.Tensor)}
    packed[f"quant_state.bitsandbytes__{qs.quant_type}"] = bnb.utils.pack_dict_to_tensor(non_t)
    return packed

def compress_layer_state_dict(sd, compression=None):
    if compression is None or not _BNB:
        return sd
    out = {}
    if compression == '4bit':
        for k, v in sd.items():
            vq, qs = bnb.functional.quantize_nf4(v.cuda(), blocksize=64)
            out[k] = vq
            for qk, qv in _bnb_qs_to_dict(qs).items():
                out[f"{k}.4bit.{qk}"] = qv
    elif compression == '8bit':
        for k, v in sd.items():
            vq, qs = bnb.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            out[k] = vq
            out[f"{k}.8bit.absmax"] = qs.absmax.clone().contiguous()
            out[f"{k}.8bit.code"] = qs.code.clone().contiguous()
    return out or sd

def uncompress_layer_state_dict(sd):
    keys = set(sd.keys())
    # Fast path: no compression markers at all
    if not any('4bit' in k or '8bit' in k for k in keys):
        return sd

    out = {}
    if any('4bit' in k for k in keys):
        base_keys = [k for k in keys if '4bit' not in k]
        for k in base_keys:
            qs_dict = {kk[len(k):]: kv for kk, kv in sd.items() if kk.startswith(k) and kk != k}
            qs = bnb.functional.QuantState.from_dict(qs_dict=qs_dict, device="cuda")
            out[k] = bnb.functional.dequantize_nf4(sd[k].cuda(), qs)
    elif any('8bit' in k for k in keys):
        base_keys = [k for k in keys if '8bit' not in k]
        for k in base_keys:
            out[k] = bnb.functional.dequantize_blockwise(
                sd[k].cuda(),
                bnb.functional.QuantState(
                    absmax=sd[f"{k}.8bit.absmax"].cuda(),
                    code=sd[f"{k}.8bit.code"].cuda(),
                    blocksize=2048, dtype=torch.float16))
    return out

# --- Layer I/O ---

def load_layer(local_path, layer_name, profiling=False):
    sd = ModelPersister.get_model_persister().load_model(layer_name, local_path)
    if profiling:
        t = time.time()
    result = uncompress_layer_state_dict(sd)
    if profiling:
        return result, time.time() - t
    return result

# --- Disk space ---
class NotEnoughSpaceException(Exception): pass


def _check_space(checkpoint_path, layer_shards_saving_path=None, compression=None, splitted_dir='splitted_model'):
    total_size = sum(os.path.getsize(f) for f in glob(str(checkpoint_path / '*')))
    saved_size = 0
    if layer_shards_saving_path:
        sp = Path(layer_shards_saving_path) / splitted_dir
        if sp.exists():
            saved_size = sum(os.path.getsize(f) for f in glob(str(sp / '*')))
    if compression == '4bit':
        total_size = int(total_size / 0.2813)
    elif compression == '8bit':
        total_size //= 2
    _, _, free = shutil.disk_usage(checkpoint_path if not layer_shards_saving_path else layer_shards_saving_path)
    if free + saved_size < total_size:
        raise NotEnoughSpaceException(
            f"Not enough space. Free: {free/2**30:.1f}GB, Need: {total_size/2**30:.1f}GB, Reusable: {saved_size/2**30:.1f}GB")

# --- Split & save ---
def _remove_file(path):
    real = os.path.realpath(path)
    os.remove(path)
    if real != path and os.path.exists(real):
        os.remove(real)

def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None, splitted_dir='splitted_model',
                          compression=None, layer_names=None, delete_original=False, repo_id=None, hf_token=None):
    """Split sharded checkpoint into per-layer files."""
    if compression and _BNB:
        splitted_dir = f"{splitted_dir}.{compression}"

    checkpoint_path = Path(checkpoint_path)
    saving_path = (Path(layer_shards_saving_path) / splitted_dir) if layer_shards_saving_path else (checkpoint_path / splitted_dir)

    # Detect format
    safetensors_fmt = not (checkpoint_path / 'pytorch_model.bin.index.json').exists()
    idx_file = 'model.safetensors.index.json' if safetensors_fmt else 'pytorch_model.bin.index.json'
    assert (checkpoint_path / idx_file).exists(), f'{idx_file} not found in {checkpoint_path}'

    with open(checkpoint_path / idx_file, 'rb') as f:
        index = json.load(f)['weight_map']

    prefix = (layer_names or {}).get('layer_prefix', 'model.layers')
    n_layers = len({int(k[len(prefix):].split('.')[1]) for k in index if k.startswith(prefix + '.')})

    if layer_names:
        layers = [layer_names['embed']] + [f"{prefix}.{i}" for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]
        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
    else:
        layers = ['model.embed_tokens'] + [f'model.layers.{i}' for i in range(n_layers)] + ['model.norm', 'lm_head']
    layers = [l + '.' for l in layers]

    # Check existing
    persister = ModelPersister.get_model_persister()
    if saving_path.exists():
        if all(persister.model_persist_exist(l, saving_path) for l in layers):
            return str(saving_path)

    if not delete_original:
        _check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_dir)

    saving_path.mkdir(parents=True, exist_ok=True)

    shard, n_shards = 0, len(set(index.values()))
    state_dict = {}
    single_modelfile = None

    from tqdm import tqdm
    for layer in tqdm(layers, desc='Splitting layers'):
        shards = [int(v.split('-')[1]) for k, v in index.items() if k.startswith(layer) and '-' in v and len(v.split('-')) > 1]
        if shards:
            if max(shards) > shard:
                if delete_original and shard:
                    ext = 'safetensors' if safetensors_fmt else 'bin'
                    _remove_file(checkpoint_path / f'{"model" if safetensors_fmt else "pytorch_model"}-{shard:05d}-of-{n_shards:05d}.{ext}')
                shard += 1
                fn = f'{"model" if safetensors_fmt else "pytorch_model"}-{shard:05d}-of-{n_shards:05d}.{"safetensors" if safetensors_fmt else "bin"}'
                to_load = checkpoint_path / fn
                if not to_load.exists() and repo_id:
                    huggingface_hub.snapshot_download(repo_id, allow_patterns=fn, token=hf_token)
                state_dict.update(load_file(to_load, device='cpu') if safetensors_fmt else torch.load(to_load, map_location='cpu'))
        else:
            single_modelfile = next(v for k, v in index.items() if k.startswith(layer))
            to_load = checkpoint_path / single_modelfile
            if not to_load.exists() and repo_id:
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(str(to_load)), token=hf_token)
            state_dict.update(load_file(to_load, device='cpu') if safetensors_fmt else torch.load(to_load, map_location='cpu'))

        layer_sd = {k: v for k, v in state_dict.items() if k.startswith(layer)}
        layer_sd = compress_layer_state_dict(layer_sd, compression)

        if not persister.model_persist_exist(layer, saving_path):
            persister.persist_model(layer_sd, layer, saving_path)

        for k in layer_sd:
            state_dict.pop(k, None)
        del layer_sd
        clean_memory()

    if delete_original and single_modelfile:
        _remove_file(checkpoint_path / single_modelfile)

    return str(saving_path)

def find_or_create_local_splitted_path(model_path_or_id, layer_shards_saving_path=None, compression=None,
                                       layer_names=None, hf_token=None, delete_original=False):
    """Locate or create split layer files."""
    mp = Path(model_path_or_id) if os.path.exists(model_path_or_id) else None

    if mp and ((mp / 'pytorch_model.bin.index.json').exists() or (mp / 'model.safetensors.index.json').exists()):
        return mp, split_and_save_layers(mp, layer_shards_saving_path, compression=compression,
                                         layer_names=layer_names, delete_original=delete_original)

    cache = huggingface_hub.snapshot_download(
        model_path_or_id, token=hf_token, ignore_patterns=['*.safetensors', '*.bin'])
    return Path(cache), split_and_save_layers(
        cache, layer_shards_saving_path, compression=compression, layer_names=layer_names,
        delete_original=delete_original, repo_id=model_path_or_id, hf_token=hf_token)
