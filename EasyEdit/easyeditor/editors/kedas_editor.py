from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import torch
import numpy as np
import random
from copy import deepcopy
from .editor import BaseEditor
from ..models.melo.melo import LORA
from ..models.kedas.kedas_model import KEDASModel
from ..models.lte.lte_model import LTEModel
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
  
class KEDASEditor(BaseEditor):
    """Editor for KEDAS"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.base_model_name = hparams.base_model_name
        self.lora_model_name = hparams.lora_model_name
        self.sentence_model_name = hparams.sentence_model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        device_map = "auto"
        torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map
        }

        if 't5' in self.base_model_name.lower():
            base_model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = T5Tokenizer.from_pretrained(self.base_model_name)
        elif 'gpt' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = GPT2Tokenizer.from_pretrained(self.base_model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'llama' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'baichuan' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs, trust_remote_code=True)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name,trust_remote_code=True)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'chatglm' in self.base_model_name.lower():
            base_model = AutoModel.from_pretrained(self.base_model_name,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name,trust_remote_code=True)
            if 'chatglm2'in self.base_model_name.lower(): 
                self.tok.unk_token_id = 64787
            else: 
                self.tok.pad_token_id = self.tok.eos_token_id
        elif 'qwen2' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'qwen' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name,fp32=False,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'mistral' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        else:
            raise NotImplementedError

        if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'KEDAS']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            self.tok.padding_side = 'left'
        if self.tok is not None and ('mistral' in self.base_model_name.lower() or 'llama' in self.base_model_name.lower() or 'qwen' in self.base_model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
            self.tok.padding_side = 'right'

        lora_model = PeftModel.from_pretrained(
            base_model,
            self.lora_model_name,
            torch_dtype=torch_dtype
        )

        if self.alg_name == 'KEDAS':
            self.model = KEDASModel(
                tokenizer=self.tok,
                origin_model=base_model,
                lora_model=lora_model,
                # sentence_model=sentence_model,
                hparams=hparams
            )
        elif self.alg_name == 'LTE':
            sentence_model = SentenceTransformer(self.sentence_model_name)
            self.model = LTEModel(
                tokenizer=self.tok,
                origin_model=base_model,
                lora_model=lora_model,
                sentence_model=sentence_model,
                hparams=hparams
            )
        else:
            raise NotImplementedError(f"Algorithm {self.alg_name} not implemented.")

        self.hparams = hparams

    # def edit(self,
    #          prompts: Union[str, List[str]],
    #          target_new: Union[str, List[str]],
    #          ground_truth: Optional[Union[str, List[str]]] = None,
    #          target_neg: Optional[Union[str, List[str]]] = None,
    #          rephrase_prompts: Optional[Union[str, List[str]]] = None,
    #          locality_inputs:  Optional[Dict] = None,
    #          portability_inputs: Optional[Dict] = None,
    #          sequential_edit=False,
    #          verbose=True,
    #          vanilla_generation=False,
    #          **kwargs
    #          ):
    #     """
    #     `prompts`: list or str
    #         the prompts to edit
    #     `ground_truth`: str
    #         the ground truth / expected output
    #     `locality_inputs`: dict
    #         for locality
    #     """
    #     test_generation = kwargs.pop('test_generation', False)

    #     if isinstance(prompts, List):
    #         assert len(prompts) == len(target_new)
    #     else:
    #         prompts, target_new = [prompts,], [target_new,]

    #     if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(self.alg_name):  # For Singleton Editing, bs=1
    #         assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

    #     if ground_truth is not None:
    #         ground_truth = [ground_truth,] if isinstance(ground_truth, str) else ground_truth
    #     else:# Default ground truth is <|endoftext|>
    #         ground_truth = ['<|endoftext|>'] * (len(prompts))

    #     if "requests" in kwargs.keys():
    #         requests = kwargs["requests"]
    #     else:
    #         requests = _prepare_requests(prompts, target_new, ground_truth, target_neg, rephrase_prompts, locality_inputs, portability_inputs, **kwargs)
    #     return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, vanilla_generation=vanilla_generation, **kwargs)

    # def edit_requests(self,
    #          requests,
    #          sequential_edit=False,
    #          verbose=True,
    #          test_generation=False,
    #          vanilla_generation=False,
    #          **kwargs
    #          ):
    #     """
    #     `prompts`: list or str
    #         the prompts to edit
    #     `ground_truth`: str
    #         the ground truth / expected output
    #     `locality_inputs`: dict
    #         for locality
    #     """
    #     eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
    #     if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
    #         assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
    #     all_metrics = []
    #     if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
    #         metrics = kwargs['pre_edit']
    #         assert len(metrics) == len(requests), f"Error: pre_edit metrics length {len(metrics)} does not match requests length {len(requests)}"
    #         all_metrics = metrics
    #     else:
    #         for i, request in enumerate(tqdm(requests)):
    #             metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,self.hparams.device,
    #                                                    eval_metric=eval_metric, test_generation=test_generation, vanilla_generation=vanilla_generation, pre_edit=True)}
    #             all_metrics.append(metrics)
    #         if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
    #             json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

    #     def edit_func(request):
    #         edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
    #             self.model,
    #             self.tok,
    #             [request],
    #             self.hparams,
    #             copy=False,
    #             return_orig_weights=True,
    #             keep_original_weight=False,
    #             train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
    #         )
    #         return edited_model, weights_copy, icl_examples

    #     def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
    #         eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
    #         all_metrics[idx].update({
    #             'case_id': idx,
    #             "requested_rewrite": request,
    #             "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device,
    #                                          eval_metric=eval_metric, test_generation=test_generation, vanilla_generation=vanilla_generation),
    #         })
    #         # print(all_metrics)
    #         if "metric_kwargs" in kwargs:
    #             all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
    #         if 'locality' in all_metrics[idx]['post'].keys():
    #             for locality_key in request['locality'].keys():
    #                 locality_result = []
    #                 if hasattr(self.hparams, 'evaluation_type') and self.hparams.evaluation_type == "LLM-judge":
    #                     locality_result.append(float(all_metrics[idx]['post']['locality'][f'{locality_key}_output']==all_metrics[idx]['pre']['locality'][f'{locality_key}_output']))
    #                 else:
    #                     for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
    #                         if len(ans) < len(label):
    #                             ans = ans + [self.tok.pad_token_id] * (len(label) - len(ans))
    #                         elif len(ans) > len(label):
    #                             label = label + [self.tok.pad_token_id] * (len(ans) - len(label))
    #                         locality_result.append(np.mean(np.equal(ans, label)))
    #                 all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
    #                 all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
    #             all_metrics[idx]['pre'].pop('locality')

    #         if verbose:
    #             LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")

    #     edit_time, evaluation_time = 0, 0
    #     if sequential_edit:
    #         for i, request in enumerate(tqdm(requests, total=len(requests))):
    #             edited_model, weights_copy, icl_examples = edit_func(request)
    #         for i, request in enumerate(requests):
    #             edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
    #     else:
    #         for i, request in enumerate(tqdm(requests, total=len(requests))):
    #             start_time = time()
    #             edited_model, weights_copy, icl_examples = edit_func(request)
    #             edit_time += time() - start_time
    #             start_time = time()
    #             edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
    #             evaluation_time += time() - start_time
    #             if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
    #                 with torch.no_grad():
    #                     weights_copy()
    #             elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
    #                 edited_model.unload()
    #                 del self.model.peft_config
    #             elif self.alg_name == 'MELO':
    #                 self.model = edited_model
    #             else:
    #                 with torch.no_grad():
    #                     for k, v in weights_copy.items():
    #                         nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")


    #     if isinstance(edited_model, LORA):
    #         edited_model = edited_model.model
    #     if len(all_metrics) != 0:
    #         summary_metrics(all_metrics)

    #     return all_metrics, edited_model, weights_copy, edit_time, evaluation_time


class KEDASEditorCRAFT(CRAFTEditor):
    """Editor for KEDAS"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.base_model_name = hparams.base_model_name
        self.lora_model_name = hparams.lora_model_name
        self.sentence_model_name = hparams.sentence_model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        make_logs()
        LOG.info("Instantiating model")

        device_map = "auto"
        torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
        
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map
        }

        if 't5' in self.base_model_name.lower():
            base_model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = T5Tokenizer.from_pretrained(self.base_model_name)
        elif 'gpt' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = GPT2Tokenizer.from_pretrained(self.base_model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'llama' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'baichuan' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs, trust_remote_code=True)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name,trust_remote_code=True)
            self.tok.pad_token_id = self.tok.eos_token_id
        elif 'chatglm' in self.base_model_name.lower():
            base_model = AutoModel.from_pretrained(self.base_model_name,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name,trust_remote_code=True)
            if 'chatglm2'in self.base_model_name.lower(): 
                self.tok.unk_token_id = 64787
            else: 
                self.tok.pad_token_id = self.tok.eos_token_id
        elif 'qwen2' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name,trust_remote_code=True, torch_dtype=torch_dtype if hparams.alg_name not in ['MEND'] else torch.bfloat16, device_map=device_map)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'qwen' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name,fp32=False,trust_remote_code=True, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name, eos_token='<|endoftext|>', pad_token='<|endoftext|>',unk_token='<|endoftext|>', trust_remote_code=True)
        elif 'mistral' in self.base_model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
            self.tok = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tok.pad_token_id = self.tok.eos_token_id
        else:
            raise NotImplementedError

        if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer) or isinstance(self.tok, LlamaTokenizerFast) or isinstance(self.tok, PreTrainedTokenizerFast)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit', 'KEDAS']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
            self.tok.padding_side = 'left'
        if self.tok is not None and ('mistral' in self.base_model_name.lower() or 'llama' in self.base_model_name.lower() or 'qwen' in self.base_model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME','AlphaEdit']):
            LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
            self.tok.padding_side = 'right'

        lora_model = PeftModel.from_pretrained(
            base_model,
            self.lora_model_name,
            torch_dtype=torch_dtype
        )

        if self.alg_name == 'KEDAS':
            self.model = KEDASModel(
                tokenizer=self.tok,
                origin_model=base_model,
                lora_model=lora_model,
                # sentence_model=sentence_model,
                hparams=hparams
            )
        elif self.alg_name == 'LTE':
            sentence_model = SentenceTransformer(self.sentence_model_name)
            self.model = LTEModel(
                tokenizer=self.tok,
                origin_model=base_model,
                lora_model=lora_model,
                sentence_model=sentence_model,
                hparams=hparams
            )
        else:
            raise NotImplementedError(f"Algorithm {self.alg_name} not implemented.")

        self.hparams = hparams
