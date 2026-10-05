import torch


def apply_around_dim(
    operation: callable,
    x: torch.Tensor,
    dims: int | tuple[int],
    *args,
    **kwargs,
) -> torch.Tensor:

    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(d if d >= 0 else x.ndim + d for d in dims)

    all_dims = list(range(x.ndim))
    reduce_dims = tuple(d for d in all_dims if d not in dims)

    try:
        # Modern PyTorch reduction (torch.amin, torch.amax, torch.mean, torch.std)
        out = operation(x, dim=reduce_dims, *args, **kwargs)
        if isinstance(out, tuple):
            out = out.values
        return out
    except TypeError:
        keepdim = kwargs.get("keepdim", False)
        reduced_x = x
        for i, d in enumerate(reduce_dims):
            target_dim = d if keepdim else d - i
            reduced_x = operation(reduced_x, dim=target_dim, *args, **kwargs)
            if isinstance(reduced_x, tuple):
                reduced_x = reduced_x.values
        return reduced_x


def sum_around(x: torch.Tensor, dims: int | tuple[int], *, keepdim: bool = False) -> torch.Tensor:

    return apply_around_dim(torch.sum, x, dims, keepdim=keepdim)


def mean_around(x: torch.Tensor, dims: int | tuple[int], *, keepdim: bool = False) -> torch.Tensor:

    return apply_around_dim(torch.mean, x, dims, keepdim=keepdim)


def approx_cumulative_normal_function(x: torch.Tensor) -> torch.Tensor:
    # Formula according to E. Page (1976)
    y = torch.sqrt((torch.tensor(2.0) / torch.pi)) * (x + 0.044715 * x.pow(3))
    return 0.5 * (1.0 + torch.tanh(y))
