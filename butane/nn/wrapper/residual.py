import torch


class Residual(torch.nn.Module):
    """Wraps a module so its output is added to its input: ``x + module(x)``."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.module(x, *args, **kwargs)
