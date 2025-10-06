from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import torch
import numpy as np
import random
from .editor import BaseEditor
from ..models.ike.ike_model import IKEModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer,PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..evaluate.evaluate_utils import test_generation_quality
from .craft_editor import CRAFTEditor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)
def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
  
class IKEEditor(BaseEditor):
    """Editor for IKE"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.sentence_model_name = hparams.sentence_model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        device_map = None
        torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map
        }

        if 't5' in self.model_name.lower():
            base_model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
            self.tok = T5Tokenizer.from_pretrained(self.model_name)
        elif 'gpt' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'llama' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'baichuan' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
            self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'chatglm' in self.model_name.lower():
            base_model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            if 'chatglm2'in self.model_name.lower(): 
                self.tok.unk_token_id = 64787
            else: 
                self.tok.pad_token_id = self.tok.eos_token_id
        elif 'qwen2' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
            self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'qwen' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'mistral' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        else:
            raise NotImplementedError

        if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'KEDAS']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            self.tok.padding_side = 'left'
        if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
            self.tok.padding_side = 'right'

        self.sentence_model = SentenceTransformer(self.sentence_model_name)

        if self.alg_name == 'IKE':
            self.model = IKEModel(
                tokenizer=self.tok,
                model=base_model,
                sentence_model=self.sentence_model,
                hparams=hparams
            )
        else:
            raise NotImplementedError(f"Algorithm {self.alg_name} not implemented.")

        self.hparams = hparams


class IKEEditorCRAFT(CRAFTEditor):
    """Editor for IKE"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.sentence_model_name = hparams.sentence_model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        device_map = None
        torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map
        }

        if 't5' in self.model_name.lower():
            base_model = T5ForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
            self.tok = T5Tokenizer.from_pretrained(self.model_name)
        elif 'gpt' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'llama' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'baichuan' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs, trust_remote_code=True)
            self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'chatglm' in self.model_name.lower():
            base_model = AutoModel.from_pretrained(self.model_name,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
            if 'chatglm2'in self.model_name.lower(): 
                self.tok.unk_token_id = 64787
            else: 
                self.tok.pad_token_id = self.tok.eos_token_id
        elif 'qwen2' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
            self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'qwen' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name,fp32=False,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'mistral' in self.model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        else:
            raise NotImplementedError

        if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'KEDAS']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            self.tok.padding_side = 'left'
        if self.tok is not None and ('mistral' in self.model_name.lower() or 'llama' in self.model_name.lower() or 'qwen' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
            self.tok.padding_side = 'right'

        self.sentence_model = SentenceTransformer(self.sentence_model_name)

        if self.alg_name == 'IKE':
            self.model = IKEModel(
                tokenizer=self.tok,
                model=base_model,
                sentence_model=self.sentence_model,
                hparams=hparams
            )
        else:
            raise NotImplementedError(f"Algorithm {self.alg_name} not implemented.")

        self.hparams = hparams