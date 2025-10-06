from transformers import AutoModelForCausalLM, AutoTokenizer
from .kedas_hparams import KEDASHyperParams
from typing import Any, Dict, List, Tuple


def apply_kedas_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: KEDASHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    for r in request:
        # query = r['prompt']
        # new_fact = r['prompt'] + ' ' + r['target_new']
        request = r['prompt']
        target_new = r['target_new']
        model.edit(request, target_new, keep_original_weight=keep_original_weight)

    return model, {}
