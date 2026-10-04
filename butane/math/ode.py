from typing import Callable, Optional, Tuple, Union, List
import functools
import torch

def _to_tuple(x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    if isinstance(x, torch.Tensor):
        x = (x,)
    return x

def _step(
    func: Callable,
    h: float,
    x: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:

    if len(x) == 1:
        out = func(h, x[0])
        return (out,) if isinstance(out, torch.Tensor) else out
    out = func(h, x)
    return out

def _unwrap_output(
    xs: List[torch.Tensor],
    dxs: Optional[List[torch.Tensor]],
    is_tensor: bool
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

    if is_tensor:
        out_xs = xs[0]
        out_dxs = dxs[0] if dxs is not None else None
    else:
        out_xs = tuple(xs)
        out_dxs = tuple(dxs) if dxs is not None else None

    return out_xs, out_dxs

@torch.no_grad()
def _euler_explicit(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    return_trajectory: bool = False,
    return_func_outputs: bool = False,
) -> torch.Tensor:

    _is_tensor = isinstance(x0, torch.Tensor)
    x = _to_tuple(x0)
    num_vars = len(x)
    num_steps = len(steps)

    x = [instance.clone() for instance in x]

    xs, dxs = None, None
    if return_trajectory:
        xs = [torch.empty(num_steps, *x[i].shape, device=x[i].device, dtype=x[i].dtype) for i in range(num_vars)]
        for i in range(num_vars):
            xs[i][0] = x[i].clone()

        if return_func_outputs:
            dxs = [torch.zeros(num_steps, *x[i].shape, device=x[i].device, dtype=x[i].dtype) for i in range(num_vars)]

    for k, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        dh = h1 - h0
        _dx = _step(func, h0, tuple(x))

        for i in range(num_vars):
            x[i].add_(_dx[i], alpha=dh)
            if return_trajectory:
                xs[i][k].copy_(x[i])
                if return_func_outputs:
                    dxs[i][k-1].copy_(_dx[i])

    # func at the last grid point is only reported, the state is already integrated
    _dx = _step(func, steps[-1], tuple(x)) if return_func_outputs else None
    if return_trajectory:
        for i in range(num_vars):
            if return_func_outputs:
                dxs[i][-1].copy_(_dx[i])
        return _unwrap_output(xs, dxs, _is_tensor)
    else:
        return _unwrap_output(x, _dx, _is_tensor)

@torch.no_grad()
def _rk4(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    return_trajectory: bool = False,
    return_func_outputs: bool = False,
) -> torch.Tensor:

    _is_tensor = isinstance(x0, torch.Tensor)
    x = _to_tuple(x0)
    num_vars = len(x)
    num_steps = len(steps)

    x = [instance.clone() for instance in x]
    _tmp_tensor = [instance.clone() for instance in x]

    xs, dxs = None, None
    if return_trajectory:
        xs = [torch.empty(num_steps, *x[i].shape, device=x[i].device, dtype=x[i].dtype) for i in range(num_vars)]
        for i in range(num_vars):
            xs[i][0] = x[i].clone()

        if return_func_outputs:
            dxs = [torch.zeros(num_steps, *x[i].shape, device=x[i].device, dtype=x[i].dtype) for i in range(num_vars)]

    for k, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        dh = h1 - h0
        dh_2 = dh / 2.0

        k1 = _step(func, h0, tuple(x))
        for i in range(num_vars):
            torch.add(x[i], k1[i], alpha=dh_2, out=_tmp_tensor[i])

        k2 = _step(func, h0 + dh_2, tuple(_tmp_tensor))
        for i in range(num_vars):
            torch.add(x[i], k2[i], alpha=dh_2, out=_tmp_tensor[i])

        k3 = _step(func, h0 + dh_2, tuple(_tmp_tensor))
        for i in range(num_vars):
            torch.add(x[i], k3[i], alpha=dh, out=_tmp_tensor[i])

        k4 = _step(func, h1, tuple(_tmp_tensor))
        for i in range(num_vars):
            _tmp_tensor[i].copy_(k1[i])
            _tmp_tensor[i].add_(k2[i], alpha=2.0)
            _tmp_tensor[i].add_(k3[i], alpha=2.0)
            _tmp_tensor[i].add_(k4[i])
            _tmp_tensor[i].div_(6.0)
            x[i].add_(_tmp_tensor[i], alpha=dh)

            if return_trajectory:
                xs[i][k].copy_(x[i])
                if return_func_outputs:
                    dxs[i][k-1].copy_(_tmp_tensor[i])

    # func at the last grid point is only reported, the state is already integrated
    _dx = _step(func, steps[-1], tuple(x)) if return_func_outputs else None
    if return_trajectory:
        for i in range(num_vars):
            if return_func_outputs:
                dxs[i][-1].copy_(_dx[i])
        return _unwrap_output(xs, dxs, _is_tensor)
    else:
        return _unwrap_output(x, _dx, _is_tensor)

@torch.no_grad()
def _heun2(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    return_trajectory: bool = False,
    return_func_outputs: bool = False,
) -> torch.Tensor:

    _is_tensor = isinstance(x0, torch.Tensor)
    x = _to_tuple(x0)
    num_vars = len(x)
    num_steps = len(steps)

    x = [instance.clone() for instance in x]
    _tmp_tensor = [instance.clone() for instance in x]

    xs, dxs = None, None
    if return_trajectory:
        xs = [torch.empty(num_steps, *x[i].shape, device=x[i].device, dtype=x[i].dtype) for i in range(num_vars)]
        for i in range(num_vars):
            xs[i][0] = x[i].clone()

        if return_func_outputs:
            dxs = [torch.zeros(num_steps, *x[i].shape, device=x[i].device, dtype=x[i].dtype) for i in range(num_vars)]

    for k, (h0, h1) in enumerate(zip(steps[:-1], steps[1:]), start=1):
        dh = h1 - h0

        _dx_1 = _step(func, h0, tuple(x))
        for i in range(num_vars):
            torch.add(x[i], _dx_1[i], alpha=dh, out=_tmp_tensor[i])

        _dx_2 = _step(func, h0 + dh, tuple(_tmp_tensor))

        for i in range(num_vars):
            _tmp_tensor[i].copy_(_dx_1[i])
            _tmp_tensor[i].add_(_dx_2[i])
            _tmp_tensor[i].mul_(0.5)
            x[i].add_(_tmp_tensor[i], alpha=dh)

            if return_trajectory:
                xs[i][k].copy_(x[i])
                if return_func_outputs:
                    dxs[i][k-1].copy_(_tmp_tensor[i])

    # func at the last grid point is only reported, the state is already integrated
    _dx = _step(func, steps[-1], tuple(x)) if return_func_outputs else None
    if return_trajectory:
        for i in range(num_vars):
            if return_func_outputs:
                dxs[i][-1].copy_(_dx[i])
        return _unwrap_output(xs, dxs, _is_tensor)
    else:
        return _unwrap_output(x, _dx, _is_tensor)

def odeint(
    func: Callable,
    x0: torch.Tensor,
    steps: torch.Tensor,
    method: str = 'euler',
    return_trajectory: bool = False,
    return_func_outputs: bool = False,
) -> torch.Tensor:

    if method == "euler":
        return _euler_explicit(func, x0, steps, return_trajectory, return_func_outputs)
    if method == "heun2":
        return _heun2(func, x0, steps, return_trajectory, return_func_outputs)
    elif method == "rk4":
        return _rk4(func, x0, steps, return_trajectory, return_func_outputs)
    else:
        raise ValueError("Method does not exist")
