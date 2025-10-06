from transformers import AutoModelForCausalLM, AutoTokenizer
from .ike_hparams import IKEHyperParams
from typing import Any, Dict, List, Tuple


def apply_ike_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: IKEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    for r in request:
        query = r['prompt']
        new_fact = r['prompt'] + ' ' + r['target_new']
        model.edit(query, new_fact, keep_original_weight=keep_original_weight)

    return model, {}
