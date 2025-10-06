from .eren_hparams import ERENHyperParams
from .eren_model import ERENModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Tuple

def apply_eren_to_model(
    model: ERENModel,
    tok,
    request: List[Dict],
    hparams: ERENHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[ERENModel, Dict[str, Any]]:
    # Gather edit statements from request
    statements = [r['prompt'] + ' ' + r['target_new'] for r in request]
    model.edit(statements, keep_original_weight=keep_original_weight)
    return model, {}