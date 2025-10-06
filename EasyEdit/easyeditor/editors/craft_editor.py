from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import torch
import numpy as np
import random
from ..models.melo.melo import LORA
from ..util.globals import *
from .utils import summary_metrics
from ..evaluate import compute_edit_quality_craft, compute_sent_metric
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from .editor import BaseEditor

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
  
class CRAFTEditor(BaseEditor):
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        super().__init__(hparams)

    def _get_edit_requests(self, requests: List[Dict]):
        edit_requests = []
        for request in requests:
            prompt_list = request['prompt']
            target_new_list = request['target_new']
            subject_list = request['subject']
            for prompt, target_new, subject in zip(prompt_list, target_new_list, subject_list):
                edit_requests.append({
                    'prompt': prompt,
                    'target_new': target_new,
                    'subject': subject,
                })

        return edit_requests

    def edit(self,
             requests: List[Dict],
             sequential_edit=False,
             incremental_edit=False,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        test_generation = kwargs.pop('test_generation', False)

        return self.edit_requests(requests, sequential_edit, incremental_edit, verbose, test_generation=test_generation, keep_original_weight=keep_original_weight, **kwargs)

    def edit_requests(self,
             requests,
             sequential_edit=False,
             incremental_edit=False,
             verbose=True,
             test_generation=False,
             keep_original_weight=False,
             **kwargs
             ):
        eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                metrics = {"pre": compute_edit_quality_craft(self.model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation, pre_edit=True)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request: Union[Dict, List[Dict]], keep_original_weight=False):
            request = [request] if isinstance(request, Dict) else request
            edit_requests = self._get_edit_requests(request)
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                edit_requests,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=None
            )
            icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric= kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            all_metrics[idx].update({
                'case_id': idx,
                "requested_rewrite": request,
                "post": compute_edit_quality_craft(edited_model, self.model_name, self.hparams, self.tok, request, self.hparams.device, eval_metric=eval_metric, test_generation=test_generation),
            })
            if "metric_kwargs" in kwargs:
                all_metrics[idx].update(compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
            if 'locality' in all_metrics[idx]['post'].keys():
                for locality_key in request['locality'].keys():
                    locality_result = []
                    for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'], all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                        locality_result.append(np.mean(np.equal(ans, label)))
                    all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                    all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")
        
        edit_time, evaluation_time = 0, 0

        if sequential_edit:
            print("Sequential Editing...")
            start_time = time()
            if self.alg_name == 'WISE':
                for i, request in enumerate(tqdm(requests, total=len(requests))):
                    edited_model, weights_copy, icl_examples = edit_func(request, keep_original_weight=keep_original_weight)
            else:
                edited_model, weights_copy, icl_examples = edit_func(requests, keep_original_weight=False)
            edit_time += time() - start_time
            if self.alg_name == 'WISE' and hasattr(self.hparams, 'save_path') and self.hparams.save_path:
                print("Start saving the WISE model!")
                edited_model.save(self.hparams.save_path)
            start_time = time()
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
            evaluation_time += time() - start_time
        elif incremental_edit:
            print("Incremental Editing...")
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                start_time = time()
                edited_model, weights_copy, icl_examples = edit_func(request, keep_original_weight=False)
                edit_time += time() - start_time
                start_time = time()
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                evaluation_time += time() - start_time
        else:
            print("Single Editing...")
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                start_time = time()
                edited_model, weights_copy, icl_examples = edit_func(request, keep_original_weight=True)
                edit_time += time() - start_time
                start_time = time()
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                evaluation_time += time() - start_time
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA' or self.alg_name == 'QLoRA' or self.alg_name == 'DPO':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            
        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_metrics) != 0:
            summary_metrics(all_metrics)

        return all_metrics, edited_model, weights_copy, edit_time, evaluation_time
