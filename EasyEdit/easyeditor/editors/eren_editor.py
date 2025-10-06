from typing import Optional, Union, List, Dict
from .editor import BaseEditor
from ..models.eren.eren_model import ERENModel
from ..models.eren.eren_hparams import ERENHyperParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..util.alg_dict import ALG_DICT
from .craft_editor import CRAFTEditor

class ERENEditor(BaseEditor):
    """Editor for EREN"""

    @classmethod
    def from_hparams(cls, hparams: ERENHyperParams):
        return cls(hparams)

    def __init__(self, hparams: ERENHyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.retriever_name = hparams.retriever_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        # Model and tokenizer loading (simplified, adapt as needed)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.pad_token_id = self.tok.eos_token_id

        self.model = ERENModel(
            model=self.base_model,
            tokenizer=self.tok,
            hparams=hparams
        )
        self.hparams = hparams
 

class ERENEditorCRAFT(CRAFTEditor):
    """Editor for EREN"""

    @classmethod
    def from_hparams(cls, hparams: ERENHyperParams):
        return cls(hparams)

    def __init__(self, hparams: ERENHyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.retriever_name = hparams.retriever_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        # Model and tokenizer loading (simplified, adapt as needed)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.pad_token_id = self.tok.eos_token_id

        self.model = ERENModel(
            model=self.base_model,
            tokenizer=self.tok,
            hparams=hparams
        )
        self.hparams = hparams
 