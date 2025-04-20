import torch
from contextlib import suppress
from functools import partial
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from contextlib import nullcontext

def get_autocast(precision, device_type='cuda'):
    if precision =='amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        amp_dtype = torch.bfloat16
    else:
        return suppress

    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)

def get_te_autocast(te_fp8):
    if te_fp8:
        #fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)
        # fp8_recipe = recipe.DelayedScaling(margin=2, interval=4, fp8_format=recipe.Format.HYBRID)
        fp8_recipe = recipe.DelayedScaling(margin=2, interval=4, fp8_format=recipe.Format.E4M3)
        return partial (te.fp8_autocast, enabled=True, fp8_recipe=fp8_recipe)
    else:
        return lambda: nullcontext()
