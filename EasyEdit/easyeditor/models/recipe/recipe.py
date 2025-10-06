import torch
from copy import deepcopy
from typing import Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .recipe_hparams import RECIPEHyperParams
from .recipe_models import KnowledgeRepModel, PromptTransformer


class RECIPE(torch.nn.Module):
    """RECIPE Editor implementation adapted for EasyEdit"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                 hparams: RECIPEHyperParams):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams
        self.device = f'cuda:{hparams.device}'
        self.ckpt_path = hparams.ckpt_path
        self.model.to(self.device)
        self.name_or_path = self.model.name_or_path
        
        # Initialize RECIPE components
        self.knowl_rep_model = KnowledgeRepModel(
            hparams.knowledge_rep_dim,
            hparams.knowl_rep_prot_token_n, 
            self.device, 
            hparams.krm_base_path
        )
        self.prompt_transformer = PromptTransformer(
            hparams.knowledge_rep_dim,
            hparams.model_hidden_size, 
            hparams.prompt_token_n, 
            self.device
        )
        
        # Initialize hooks
        self.begin_layer = find_module(self.model, hparams.begin_layer_path)
        self.lm_head = find_module(self.model, hparams.lm_head_path)
        self._register_hooks()
        
        # Initialize editing prompts
        self.restore_to_original_model()
        self.auto_retrieve = hparams.auto_retrieve
        self.retr_top_k = hparams.retr_top_k
        self.retr_min_sim = hparams.retr_min_sim

        self.load_ckpt(self.ckpt_path, load_opt=False)

    def clear(self):
        self.knowledge_base = self.knowl_rep_model.get_knowl_rep_prot()
        self.prompts_base = torch.zeros([1, self.hparams.prompt_token_n, self.hparams.model_hidden_size], 
                                      device=self.device)
        self.adopted_prompts = []

    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def generate(self, **kwargs):
        return self.model.generate(
            **kwargs,
            generation_config=GenerationConfig(
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )

    def load_ckpt(self, ckpt_path, restrict = True, load_opt = True):
        ckpt = torch.load(ckpt_path, 'cpu')
        self.train_i = ckpt['i']
        self.train_epoch = ckpt['epoch']
        self.knowl_rep_model.load_state_dict(ckpt['knowl_rep_model'], restrict)
        self.prompt_transformer.load_state_dict(ckpt['prompt_transformer'], restrict)
        if load_opt:
            self.opt.load_state_dict(ckpt['opt'])
        print('Load RECIPE checkpoints from', ckpt_path)
        return ckpt['loss']
    
    def _register_hooks(self):
        """Register editing hooks"""
        def forward_pre_hook(module, args):
            # If do not has past_key_values, add editing prompts before reps.
            if not hasattr(module, 'has_past_kv') or not module.has_past_kv:
                if len(self.adopted_prompts) > 0:
                    args = args[0]
                    args = torch.stack([
                        torch.cat([p, inp[:-len(p) if len(p) != 0 else None]], 0)
                        for inp, p in zip(args, self.adopted_prompts)], 0)
                    return (args, )
        
        def forward_hook(module, args, output):
            if not hasattr(module, 'has_past_kv') or not module.has_past_kv:
                if len(self.adopted_prompts) > 0:
                    max_n = max([len(p) for p in self.adopted_prompts])
                    output = torch.stack([
                        ot[len(p):len(p)-max_n if len(p)-max_n != 0 else None]
                        for ot, p in zip(output, self.adopted_prompts)], 0)
            return output
        
        self.begin_layer_hook = self.begin_layer.register_forward_pre_hook(forward_pre_hook)
        self.lm_head_hook = self.lm_head.register_forward_hook(forward_hook)
        
        # Register model forward hook
        self._register_model_forward_hook()
    
    def _register_model_forward_hook(self):
        """Register model forward hook"""
        if hasattr(self.model, 'recipe_hooked'):
            return
        
        self.model.recipe_hooked = True
        original_forward = self.model.forward
        
        def forward_recipe(**kwargs):
            if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
                self.begin_layer.has_past_kv = True
                self.lm_head.has_past_kv = True
            else:
                self.begin_layer.has_past_kv = False
                self.lm_head.has_past_kv = False
                b, l = kwargs['input_ids'].shape
                inp_sents = [self.tokenizer.decode(i, skip_special_tokens=True) 
                           for i in kwargs['input_ids']]
                
                if self.auto_retrieve: 
                    retrieved_ids = self.retrieve_and_get_ids_sim(inp_sents)[0]
                    self.adopted_prompts = [
                        self.prompts_base[i].reshape(
                            len(i)*self.hparams.prompt_token_n, self.hparams.model_hidden_size
                        ) for i in retrieved_ids
                    ]
                
                if len(self.adopted_prompts) > 0:
                    if len(self.adopted_prompts) != b:
                        # Fill with empty prompts if needed
                        while len(self.adopted_prompts) < b:
                            self.adopted_prompts.append(torch.zeros([0, self.hparams.model_hidden_size], device=self.device))
                    
                    max_prompt_len = max([len(i) for i in self.adopted_prompts]) if self.adopted_prompts else 0
                    
                    if max_prompt_len > 0:
                        pad = torch.ones([b, max_prompt_len], dtype=torch.long).to(self.device)
                        
                        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                            kwargs['attention_mask'] = torch.cat([kwargs['attention_mask'], pad], 1)
                        
                        kwargs['input_ids'] = torch.cat(
                            [kwargs['input_ids'], pad * self.tokenizer.pad_token_id], 1
                        )
            
            return original_forward(**kwargs)
        
        self.model.forward = forward_recipe
    
    def retrieve_and_get_ids_sim(self, input_queries: List[str]):
        """Retrieve similar knowledge and get IDs"""
        query_reps = self.knowl_rep_model(input_queries, knowl_or_query='q')
        sim_matrix = (query_reps @ self.knowledge_base.T) / self.hparams.knowledge_rep_dim**0.5
        sim_with_prototype = sim_matrix[:, :1]
        sorted_sim, order = torch.sort(sim_matrix, 1, True)
        
        mask = sorted_sim[:, :self.retr_top_k] > self.retr_min_sim
        mask &= sorted_sim[:, :self.retr_top_k] > sim_with_prototype
        retrieved_ids = torch.masked_select(order[:, :self.retr_top_k], mask)
        retrieved_ids = torch.split(retrieved_ids, mask.sum(1).tolist())
        
        return retrieved_ids, (sorted_sim, order)
    
    def restore_to_original_model(self):
        """Restore to original model state"""
        self.knowledge_base_nl = ['<Knowledge_Representation_Prototype>']
        self.knowledge_base = self.knowl_rep_model.get_knowl_rep_prot()
        self.prompts_base = torch.zeros([1, self.hparams.prompt_token_n, self.hparams.model_hidden_size], 
                                      device=self.device)
        self.adopted_prompts = []

    def edit(self, requests: Union[List[Dict], Dict], keep_original_weight=False) -> None:
        """Edit the model"""
        if isinstance(requests, Dict):
            self.edit_one_piece(requests, keep_original_weight=keep_original_weight)
        else:
            for request in requests:
                self.edit_one_piece(request, keep_original_weight=keep_original_weight)
    
    def edit_one_piece(self, request: Dict, keep_original_weight=False) -> None:
        """Edit a single piece of knowledge"""
        # Format the knowledge string
        prompt = request.get('prompt', '')
        target_new = request.get('target_new', '')
        
        if prompt and target_new:
            if prompt[-1] != ' ' and target_new[0] != ' ':
                knowledge_str = prompt + ' ' + target_new
            else:
                knowledge_str = prompt + target_new
            
            # Update knowledge base
            self.knowledge_base_nl.append(knowledge_str)
            new_rep = self.knowl_rep_model([knowledge_str], knowl_or_query='k')
            new_prompt = self.prompt_transformer(new_rep)
            
            if keep_original_weight:
                self.clear()
            self.knowledge_base = torch.cat([self.knowledge_base, new_rep], 0)
            self.prompts_base = torch.cat([self.prompts_base, new_prompt], 0)
        
    def edit_batch(self, requests: List[Dict]):
        """Edit multiple pieces of knowledge"""
        knowledge_strs = []
        for request in requests:
            prompt = request.get('prompt', '')
            target_new = request.get('target_new', '')
            
            if prompt and target_new:
                if prompt[-1] != ' ' and target_new[0] != ' ':
                    knowledge_strs.append(prompt + ' ' + target_new)
                else:
                    knowledge_strs.append(prompt + target_new)
        
        if knowledge_strs:
            self.knowledge_base_nl.extend(knowledge_strs)
            new_reps = self.knowl_rep_model(knowledge_strs, knowl_or_query='k')
            new_prompts = self.prompt_transformer(new_reps)
            
            self.knowledge_base = torch.cat([self.knowledge_base, new_reps], 0)
            self.prompts_base = torch.cat([self.prompts_base, new_prompts], 0) 

def find_module(module, module_path:str):
    for comp in module_path.split('.'):
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module