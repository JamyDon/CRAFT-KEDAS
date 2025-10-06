import copy
import os
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

LOG = logging.getLogger(__name__)

from .recipe import RECIPE
from .recipe_hparams import RECIPEHyperParams

def apply_recipe_to_model(
    model: RECIPE,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: RECIPEHyperParams,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[RECIPE, List[str]]:    
    model.edit(requests, keep_original_weight=keep_original_weight)
    
    return model, {}