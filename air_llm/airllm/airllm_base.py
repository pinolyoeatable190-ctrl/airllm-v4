"""AirLLM v4 — Unified layer-streaming inference engine.
Persistent skeleton, cached masks, debounced gc, SDPA-first, registry-based variants.
"""

import os, time
from typing import Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GenerationMixin, GenerationConfig)
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer
from .profiler import LayeredProfiler
from .utils import clean_memory, load_layer, find_or_create_local_splitted_path

try:
    import bitsandbytes as bnb
    _BNB = True
except ImportError:
    _BNB = False

try:
    from transformers.cache_utils import Cache, DynamicCache
    _HAS_CACHE_UTILS = True
except ImportError:
    _HAS_CACHE_UTILS = False

# --- Layer name presets (replaces 8 separate files) ---
_LAYER_PRESETS = {
    '_default': {
        'embed': 'model.embed_tokens', 'layer_prefix': 'model.layers',
        'norm': 'model.norm', 'lm_head': 'lm_head',
    },
    'chatglm': {
        'embed': 'transformer.embedding.word_embeddings',
        'layer_prefix': 'transformer.encoder.layers',
        'norm': 'transformer.encoder.final_layernorm',
        'lm_head': 'transformer.output_layer',
        'rotary_pos_emb': 'transformer.rotary_pos_emb',
    },
    'qwen': {
        'embed': 'transformer.wte', 'layer_prefix': 'transformer.h',
        'norm': 'transformer.ln_f', 'lm_head': 'lm_head',
    },
}
_MODEL_FLAGS = {
    '_default':  {'better_transformer': True,  'seq_dim': 1, 'kv_seq_dim': 2},
    'chatglm':   {'better_transformer': False, 'seq_dim': 0, 'kv_seq_dim': 0},
    'qwen':      {'better_transformer': False, 'seq_dim': 1, 'kv_seq_dim': 1},
    'qwen2':     {'better_transformer': False, 'seq_dim': 1, 'kv_seq_dim': 2},
    'mistral':   {'better_transformer': False, 'seq_dim': 1, 'kv_seq_dim': 2},
    'mixtral':   {'better_transformer': False, 'seq_dim': 1, 'kv_seq_dim': 2},
    'baichuan':  {'better_transformer': False, 'seq_dim': 1, 'kv_seq_dim': 2},
    'internlm':  {'better_transformer': False, 'seq_dim': 1, 'kv_seq_dim': 2},
}


def _detect_variant(config) -> str:
    """Detect model variant from config.architectures."""
    arch = getattr(config, 'architectures', [''])[0].lower()
    for key in ('chatglm', 'qwen2', 'qwen', 'baichuan', 'internlm', 'mistral', 'mixtral'):
        if key in arch:
            return key
    return '_default'


class AirLLMBaseModel(GenerationMixin):
    """Layer-streaming LLM engine: ~1 layer VRAM, 4/8-bit compression, async prefetch."""

    _executor: Optional[ThreadPoolExecutor] = None  # class-level persistent pool

    @classmethod
    def _get_executor(cls):
        if not cls._executor:
            cls._executor = ThreadPoolExecutor(max_workers=2)
        return cls._executor

    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16,
                 max_seq_len=512, layer_shards_saving_path=None, profiling_mode=False,
                 compression=None, hf_token=None, prefetching=True, delete_original=False):

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()
        self._supports_cache_class = False
        self.hf_quantizer = None

        if compression and not _BNB:
            raise ImportError('bitsandbytes required for compression. pip install bitsandbytes')

        self.compression = compression
        self.hf_token = hf_token
        self.running_device = device
        self.device = torch.device(device)
        self.running_dtype = dtype
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.main_input_name = "input_ids"

        # Load config & detect variant
        kw = {'trust_remote_code': True}
        if hf_token:
            kw['token'] = hf_token
        self.config = AutoConfig.from_pretrained(
            model_local_path_or_repo_id if os.path.exists(model_local_path_or_repo_id)
            else model_local_path_or_repo_id, **kw)

        self._variant = _detect_variant(self.config)
        self.layer_names_dict = _LAYER_PRESETS.get(self._variant, _LAYER_PRESETS['_default']).copy()
        self._flags = _MODEL_FLAGS.get(self._variant, _MODEL_FLAGS['_default'])

        # Resolve paths & split layers
        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id, layer_shards_saving_path,
            compression=compression, layer_names=self.layer_names_dict,
            hf_token=hf_token, delete_original=delete_original)

        # Reload config from local path
        self.config = AutoConfig.from_pretrained(self.model_local_path, **kw)

        self.generation_config = self._load_gen_config()
        self.tokenizer = self._load_tokenizer()

        # Build model skeleton (empty weights — ~0 memory)
        self._init_model()

        # Discover layers
        attr = self.model
        for part in self.layer_names_dict['layer_prefix'].split('.'):
            attr = getattr(attr, part)
        n = len(attr)
        self.layer_names = ([self.layer_names_dict['embed']] +
                            [f"{self.layer_names_dict['layer_prefix']}.{i}" for i in range(n)] +
                            [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']])
        self._set_layer_refs()

        # Pre-allocate attention mask & position_ids (cached, not recomputed)
        self._attn_mask = (torch.ones(max_seq_len, max_seq_len)
                           .triu(diagonal=1)[None, None, ...] == 0).to(device)
        self._pos_ids = torch.arange(max_seq_len, dtype=torch.long, device=device)[None, :]

        # Prefetching
        self.prefetching = prefetching
        if device.startswith("cuda") and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    def _load_gen_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def _load_tokenizer(self):
        kw = {'trust_remote_code': True}
        if self.hf_token:
            kw['token'] = self.hf_token
        if self._variant == 'baichuan':
            from .tokenization_baichuan import BaichuanTokenizer
            return BaichuanTokenizer.from_pretrained(self.model_local_path, use_fast=False, **kw)
        return AutoTokenizer.from_pretrained(self.model_local_path, **kw)

    def _init_model(self):
        """Empty-weight skeleton with SDPA (no optimum)."""
        self.model = None
        if self._flags.get('better_transformer'):
            try:
                self.config.attn_implementation = "sdpa"
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(
                        self.config, attn_implementation="sdpa", trust_remote_code=True)
            except (TypeError, ValueError):
                self.model = None
        if self.model is None:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
        qc = getattr(self.config, 'quantization_config', None)
        if qc:
            self.hf_quantizer = AutoHfQuantizer.from_config(qc, pre_quantized=True)
            self.hf_quantizer.preprocess_model(model=self.model,
                                               device_map=self.hf_quantizer.update_device_map(None))
        self.model.eval(); self.model.tie_weights()
        for bn, buf in self.model.named_buffers():
            set_module_tensor_to_device(self.model, bn, self.running_device, value=buf, dtype=self.running_dtype)
        if 'rotary_pos_emb' in self.layer_names_dict:
            self._move_to_device(load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb']))

    def _resolve_attr(self, dotpath):
        a = self.model
        for p in dotpath.split('.'): a = getattr(a, p)
        return a

    def _set_layer_refs(self):
        ld = self.layer_names_dict
        prefix_mod = self._resolve_attr(ld['layer_prefix'])
        self.layers = ([self._resolve_attr(ld['embed'])] +
                       list(prefix_mod) +
                       [self._resolve_attr(ld['norm']), self._resolve_attr(ld['lm_head'])])

    def _load_layer_to_cpu(self, layer_name):
        t = time.time()
        result = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed = time.time() - t
        if self.profiling_mode:
            sd, ct = result
            self.profiler.add('disk_load', elapsed - ct)
            self.profiler.add('decompress', ct)
        else:
            sd = result
        if self.prefetching and torch.cuda.is_available():
            t2 = time.time()
            for v in sd.values(): v.pin_memory()
            if self.profiling_mode: self.profiler.add('pin_memory', time.time() - t2)
        return sd

    def _move_to_device(self, sd):
        moved, dev, dt = [], self.running_device, self.running_dtype
        if not self.hf_quantizer:
            for pn, p in sd.items():
                set_module_tensor_to_device(self.model, pn, dev, value=p, dtype=dt)
                moved.append(pn)
        else:
            seen = set()
            for pn in sd:
                b = pn[:pn.index('.weight')+7] if '.weight' in pn else pn
                if b in seen: continue
                seen.add(b)
                if not self.hf_quantizer.check_quantized_param(self.model, None, b, {}):
                    set_module_tensor_to_device(self.model, b, dev, value=sd[b], dtype=dt)
                else:
                    self.hf_quantizer.create_quantized_param(self.model, sd[b], b, dev, sd)
                moved.append(b)
        return moved

    def _get_kv_seq_len(self, pkv):
        return pkv[0][0].shape[self._flags['kv_seq_dim']]

    def _get_seq_len(self, seq):
        return seq.shape[self._flags['seq_dim']]

    def _pos_emb_kwargs(self, len_p, len_s):
        if self._variant == 'chatglm':
            rotary = self.model.transformer.rotary_pos_emb(self.config.seq_length)
            rotary = rotary[None, :len_s].transpose(0, 1).contiguous()
            return {'rotary_pos_emb': rotary}
        if self._variant == 'qwen':
            if self.model.transformer.use_dynamic_ntk:
                ntk_list = [1.0]
            elif len_p + len_s != len_s:
                ntk_list = self.model.transformer.rotary_emb._ntk_alpha_cached_list
            else:
                ntk_list = [self.model.transformer.get_ntk_alpha(len_p + len_s)]
            self.model.transformer.rotary_emb._ntk_alpha_cached_list = ntk_list
            return {'rotary_pos_emb_list': [
                self.model.transformer.rotary_emb(len_p + len_s, ntk_alpha=a) for a in ntk_list]}
        return {}

    def _kv_cache_kwarg(self, k, v):
        if self._variant == 'chatglm':
            return {'kv_cache': (k, v)}
        if self._variant == 'qwen':
            return {'layer_past': (k, v)}
        return {'past_key_value': (k, v)}

    def _attn_mask_kwarg(self, mask, len_p, len_s):
        if self._variant in ('chatglm', 'qwen'):
            return {'attention_mask': None}
        return {'attention_mask': mask[:, :, -len_s:, -len_p - len_s:]}

    def _pos_ids_kwarg(self, pos_ids, len_p, len_s):
        if self._variant in ('chatglm', 'qwen'):
            return {}
        return {'position_ids': pos_ids[:, len_p:len_p + len_s]}

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                       attention_mask=None, inputs_embeds=None, **kwargs):
        if past_key_values is not None:
            past_len = self._get_kv_seq_len(past_key_values)
            cut = past_len if input_ids.shape[1] > past_len else input_ids.shape[1] - 1
            input_ids = input_ids[:, cut:]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            mi = {"inputs_embeds": inputs_embeds}
        else:
            mi = {"input_ids": input_ids}
        mi.update({"position_ids": position_ids, "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"), "attention_mask": attention_mask})
        return mi

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:

        if _HAS_CACHE_UTILS:
            use_cache = False

        if self.profiling_mode:
            self.profiler.clear()
            t0 = time.time()

        # Offload all to meta (much cheaper than del+reinit)
        for layer in self.layers: layer.to("meta")
        clean_memory(force=True)
        for bn, buf in self.model.named_buffers():
            set_module_tensor_to_device(self.model, bn, self.running_device, value=buf, dtype=self.running_dtype)

        batch = [ids.to(self.running_device).unsqueeze(0) for ids in input_ids]
        attn_mask = self._attn_mask
        pos_ids = self._pos_ids

        kv_cache = [([],[]) for _ in self.layers] if use_cache else None
        executor = self._get_executor()

        with torch.inference_mode():
            # Kick off first layer prefetch
            if self.prefetching:
                future = executor.submit(self._load_layer_to_cpu, self.layer_names[0])

            for i, (lname, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                           desc=f'layers({self.running_device})',
                                           total=len(self.layers)):
                sd = future.result() if self.prefetching else self._load_layer_to_cpu(lname)
                if self.profiling_mode: t = time.time()
                moved = self._move_to_device(sd)
                if self.profiling_mode: self.profiler.add('to_device', time.time() - t)
                if self.prefetching and i + 1 < len(self.layer_names):
                    future = executor.submit(self._load_layer_to_cpu, self.layer_names[i + 1])

                for j, seq in enumerate(batch):
                    if lname == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                    elif lname == self.layer_names_dict['norm']:
                        batch[j] = layer(seq)
                    elif lname == self.layer_names_dict['lm_head']:
                        batch[j] = layer(seq).float()
                    else:
                        if past_key_values is not None:
                            k_c, v_c = past_key_values[i - 1]
                            lp = self._get_kv_seq_len(past_key_values)
                            ls = self._get_seq_len(seq)
                            kw = {'use_cache': True,
                                  **self._kv_cache_kwarg(k_c, v_c),
                                  **self._pos_emb_kwargs(lp, ls),
                                  **self._attn_mask_kwarg(attn_mask, lp, ls),
                                  **self._pos_ids_kwarg(pos_ids, lp, ls)}
                            out = layer(seq, **kw)
                            batch[j] = out[0]
                            if use_cache:
                                kv = out[2 if output_attentions else 1]
                                kv_cache[i][0].append(kv[0])
                                kv_cache[i][1].append(kv[1])
                        else:
                            ls = self._get_seq_len(seq)
                            kw = {'use_cache': bool(use_cache),
                                  **self._pos_emb_kwargs(0, ls),
                                  **self._attn_mask_kwarg(attn_mask, 0, ls),
                                  **self._pos_ids_kwarg(pos_ids, 0, ls)}
                            out = layer(seq, **kw)
                            if use_cache:
                                batch[j], (k_c, v_c) = out
                                kv_cache[i][0].append(k_c)
                                kv_cache[i][1].append(v_c)
                            else:
                                batch[j] = out[0]

                if self.hf_quantizer and moved:
                    for pn in moved:
                        set_module_tensor_to_device(self.model, pn, 'meta')
                else:
                    layer.to("meta")
                clean_memory()

        logits = torch.cat(batch, 0)

        if use_cache:
            kv_cache = kv_cache[1:-2]
            kv_cache = [(torch.cat(k, 0), torch.cat(v, 0)) for k, v in kv_cache]

        if self.profiling_mode:
            self.profiler.print()
            print(f"total wall time: {time.time() - t0:.4f}s")
            self.profiler.clear()

        return CausalLMOutputWithPast(
            loss=None, logits=logits,
            past_key_values=tuple(kv_cache) if kv_cache else None,
            hidden_states=None, attentions=None)

AirLLMLlama2 = AirLLMBaseModel  # compat alias
