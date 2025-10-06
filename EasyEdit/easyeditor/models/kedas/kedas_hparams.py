from dataclasses import dataclass
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class KEDASHyperParams(HyperParams):
    # Method
    alg_name: str
    results_dir: str

    # Database
    ike_top_k: int # top K nearest neighbors
    search_top_k: int # of dense search
    data_cache_path: str
    dist_fn: str # euc, mmd, arccos

    # Module templates
    device: int
    model_name: str
    filter_model_name: str
    base_model_name: str
    sentence_model_name: str
    lora_model_name: str
    max_length: int

    # Ablation
    editing_time_augmentation: bool = True
    self_adaptive_post_alignment_inference: bool = True
    with_filter: bool = True
    alignment: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'KEDAS') or print(f'KEDASHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        
        return cls(**config)
