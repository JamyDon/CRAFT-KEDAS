from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class RECIPEHyperParams(HyperParams):
    # Model path
    alg_name: str
    model_name: str
    ckpt_path: str
    krm_base_path: str
    device: int
    
    # Method
    prompt_token_n: int
    knowledge_rep_dim: int
    knowl_rep_prot_token_n: int
    model_hidden_size: int
    begin_layer_path: str
    lm_head_path: str
    
    # Training hyperparameters
    krm_lr: float
    pt_lr: float
    relia_lambda: float
    gen_lambda: float
    loc_lambda: float
    contra_lambda: float
    query_knowledge_t: float
    query_prototype_t: float
    constra_hinge_scale: float
    edit_hinge_scale: float
    
    # Retrieval parameters
    retr_top_k: int = 1
    retr_min_sim: float = -999
    auto_retrieve: bool = True
    
    # Training parameters
    # batch_size: int = 8
    # random_seed: int = 1
    # eps: float = 1e-8

    max_length: int = 40
    model_parallel: bool = False
    fp16: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'RECIPE') or print(f'RECIPEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config) 