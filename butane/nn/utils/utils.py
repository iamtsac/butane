from typing import Union, Optional, List, Tuple, Callable, Dict
import os
import torch
from ..._typedefs import *
from ..._helpers import module_name
from ..._utils import apply_recursively


def zero_module(module: torch.nn.Module) -> torch.nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module

def freeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module) -> None:
    for p in module.parameters():
        p.requires_grad = True

def compute_grad_norm(model: torch.nn.Module) -> float:
    grads = [
        p.grad.detach()
        for p in model.parameters()
        if p.grad is not None
    ]

    if not len(grads):
        return 0.0

    device = grads[0].device
    return torch.norm(torch.stack([torch.norm(g, 2.0).to(device) for g in grads]), 2.0).double().item()

def get_lr(optimizer: torch.optim.Optimizer):
    return optimizer.param_groups[0]['lr']

def calculate_output_size(
    *modules, 
    input_dims: Union[IntParams, Tuple[IntParams, ...], List[IntParams], Dict[str, IntParams]]
) -> torch.Tensor:
    _input = None
    if isinstance(input_dims, dict):
        _input = {k: torch.randn(1, *v) for k, v in input_dims.items()}
    elif isinstance(input_dims, (list, tuple)) and len(input_dims) > 0 and isinstance(input_dims[0], (list, tuple)):
        _input = [torch.randn(1, *v) for v in input_dims]
    elif isinstance(input_dims, (list, tuple)) or isinstance(input_dims, (int, torch.Tensor)):
        shape = (input_dims,) if isinstance(input_dims, int) else input_dims
        _input = torch.randn(1, *shape)

    out_sz = None
    for module in modules:
        # 2. Handle 'output_size' bypass attribute
        if hasattr(module, 'output_size'):
            _input = []
            if isinstance(module.output_size, (list, tuple)) and \
               len(module.output_size) > 0 and \
               isinstance(module.output_size[0], (list, tuple, torch.Tensor)):
                for mos in module.output_size:
                    _input.append(torch.randn(1, *mos))
            else:
                _input = torch.randn(1, *module.output_size)
            continue

        try:
            param = next(module.parameters())
            device = param.device
            dtype = param.dtype
            _input = apply_recursively(_input, lambda x: x.to(device, dtype=dtype))
        except StopIteration:
            pass

        if isinstance(_input, dict):
            try:
                 _input = module(_input)
            except TypeError:
                 # Fallback for modules expecting kwargs (forward(x1=..., x2=...))
                 _input = module(**_input)
        elif isinstance(_input, (list, tuple)):
            try:
                _input = module(*_input)
            except TypeError:
                _input = module(_input)
        else:
            _input = module(_input)

    if isinstance(_input, (list, tuple)):
        output_size = []
        for _i in _input:
            output_size.append(torch.tensor(_i.size())[1:])
        return output_size
    return torch.tensor(_input.size())[1:]

@torch.no_grad()
def init_weights(model: torch.nn.Module, weight_init_method: Optional[Callable] =  None, bias_init_method: Optional[Callable] = None):
    for module in model.modules():
        if hasattr(module, 'weight'):
            if 'norm' in module._get_name().lower():
                continue
            if isinstance(module, torch.nn.Embedding):
                continue
            if weight_init_method is not None:
                weight_init_method(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None and bias_init_method is not None:
                bias_init_method(module.bias.data)

def load_state(
    fpath: str,
    *,
    model: ModuleParams,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ema: ModuleParams = None,
    **modules,
):

    def __load(obj, key):
        if obj is None:
            return None
        if not isinstance(obj, (list, tuple)):
            _state = torch.load(f"{fpath}/{key}.pt", map_location=_device) if cp is None else cp[key]
            obj.load_state_dict(_state)
        elif isinstance(obj, (list, tuple)):
                for i, m in enumerate(obj, start=1):
                    _state = torch.load(f"{fpath}/{key}_{i}.pt", map_location=_device[i]) if cp is None else cp[f'{key}_{i}']
                    m.load_state_dict(_state)
        return obj

    returns = {}
    if not os.path.exists(fpath):
        raise("Checkpoint does not exits")

    _device = None
    if isinstance(model, torch.nn.Module):
        _device = next(model.parameters()).device
    elif isinstance(model, (list, tuple)):
        for i, m in enumerate(model):
            _device = next(m.parameters()).device
            break
    else:
        raise ("Model should be either torch.nn.Module or list of torch.nn.Modules")

    cp = None
    for suffix in ["pt", "pth"]:
        if os.path.exists(fpath + "/checkpoint." + suffix):
            cp = torch.load(fpath + "/checkpoint." + suffix, map_location=_device, weights_only=suffix == ".pt")

    if cp is not None:
        for k, v in cp.items():
            if not isinstance(v, dict) or "state_dict" not in str(v):
                returns[k] = v

    returns["step"] = cp.get("step")
    returns["model"] = __load(model, "model")
    returns["optimizer"] = __load(optimizer, "optimizer")
    returns["ema"] = __load(ema, "ema")
    returns["lr_scheduler"] = __load(lr_scheduler, "lr_scheduler")
    for k,v in modules.items():
        returns.update({k: __load(v, k)})

    return returns
