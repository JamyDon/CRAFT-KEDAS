from typing import Optional, Union, List, Dict

import torch
from .editor import BaseEditor
from ..models.recipe.recipe import RECIPE
from ..models.recipe.recipe_hparams import RECIPEHyperParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..util.alg_dict import ALG_DICT
from .craft_editor import CRAFTEditor

class RECIPEEditor(BaseEditor):
    """Editor for RECIPE"""

    @classmethod
    def from_hparams(cls, hparams: RECIPEHyperParams):
        return cls(hparams)

    def __init__(self, hparams: RECIPEHyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        # Model and tokenizer loading (simplified, adapt as needed)
        if hasattr(hparams, 'fp16') and hparams.fp16:
            self.base_model = AutoModelForCausalLM.from_pretrained(hparams.model_name, torch_dtype=torch.float16)
        else:   
            self.base_model = AutoModelForCausalLM.from_pretrained(hparams.model_name)
        self.tok = AutoTokenizer.from_pretrained(hparams.model_name)
        self.tok.pad_token_id = self.tok.eos_token_id

        self.model = RECIPE(
            model=self.base_model,
            tokenizer=self.tok,
            hparams=hparams
        )
        self.hparams = hparams

    # Implement edit and other required methods as needed 


class RECIPEEditorCRAFT(CRAFTEditor):
    """Editor for RECIPE"""

    @classmethod
    def from_hparams(cls, hparams: RECIPEHyperParams):
        return cls(hparams)

    def __init__(self, hparams: RECIPEHyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        # Model and tokenizer loading (simplified, adapt as needed)
        if hasattr(hparams, 'fp16') and hparams.fp16:
            self.base_model = AutoModelForCausalLM.from_pretrained(hparams.model_name, torch_dtype=torch.float16)
        else:   
            self.base_model = AutoModelForCausalLM.from_pretrained(hparams.model_name)
        self.tok = AutoTokenizer.from_pretrained(hparams.model_name)
        self.tok.pad_token_id = self.tok.eos_token_id

        self.model = RECIPE(
            model=self.base_model,
            tokenizer=self.tok,
            hparams=hparams
        )
        self.hparams = hparams