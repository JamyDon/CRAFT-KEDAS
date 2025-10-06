from dataclasses import dataclass
from typing import List, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class IKEHyperParams(HyperParams):
    alg_name: str
    results_dir: str

    k: int
    max_length: int

    device: int
    model_name: str
    sentence_model_name: str
    training_data_path: str

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'IKE') or print(f'IKEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        
        return cls(**config)
