"""Baichuan tokenizer — minimal reimplementation for gated models.
Based on Baichuan Inc. / EleutherAI / HuggingFace, Apache 2.0.
"""

import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class BaichuanTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                 pad_token=None, sp_model_kwargs=None, add_bos_token=True, add_eos_token=False,
                 clean_up_tokenization_spaces=False, **kwargs):
        self.sp_model_kwargs = sp_model_kwargs or {}
        for t in ('bos_token', 'eos_token', 'unk_token', 'pad_token'):
            v = locals()[t]
            if isinstance(v, str):
                locals()[t] = AddedToken(v, lstrip=False, rstrip=False)
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token,
                         pad_token=pad_token, add_bos_token=add_bos_token, add_eos_token=add_eos_token,
                         sp_model_kwargs=self.sp_model_kwargs,
                         clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)

    def __getstate__(self):
        s = self.__dict__.copy(); s["sp_model"] = None; return s

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        v = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        v.update(self.added_tokens_encoder); return v

    def _tokenize(self, text):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        sub, out, prev_sp = [], "", False
        for i, t in enumerate(tokens):
            if t in self.all_special_tokens:
                if not prev_sp and i: out += " "
                out += self.sp_model.decode(sub) + t
                prev_sp, sub = True, []
            else:
                sub.append(t); prev_sp = False
        return out + self.sp_model.decode(sub)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory): return
        out = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model")
        if os.path.abspath(self.vocab_file) != os.path.abspath(out) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out)
        elif not os.path.isfile(self.vocab_file):
            with open(out, "wb") as f: f.write(self.sp_model.serialized_model_proto())
        return (out,)

    def build_inputs_with_special_tokens(self, ids_0, ids_1=None):
        b = [self.bos_token_id] if self.add_bos_token else []
        e = [self.eos_token_id] if self.add_eos_token else []
        out = b + ids_0 + e
        return out + b + ids_1 + e if ids_1 else out

    def get_special_tokens_mask(self, ids_0, ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(ids_0, ids_1, True)
        b = [1] if self.add_bos_token else []
        e = [1] if self.add_eos_token else []
        m = b + [0]*len(ids_0) + e
        return m + b + [0]*len(ids_1) + e if ids_1 else m

    def create_token_type_ids_from_sequences(self, ids_0, ids_1=None):
        b = [self.bos_token_id] if self.add_bos_token else []
        e = [self.eos_token_id] if self.add_eos_token else []
        out = [0] * len(b + ids_0 + e)
        return out + [1]*len(b + ids_1 + e) if ids_1 else out
