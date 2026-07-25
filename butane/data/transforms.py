from typing import Callable, Union

import torch


class Transforms:
    def __init__(self, *transformations, sequence_dim: int | None = None):
        self._transforms = list(transformations)
        self.sequence_dim = sequence_dim

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        transformed = x
        for transform in self._transforms:
            transformed = transform(transformed)
        return transformed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self._transforms:
            return x

        # If it's a sequence, we vmap the ENTIRE pipeline at once!
        if self.sequence_dim:
            return torch.vmap(self._forward_single, in_dims=self.sequence_dim, out_dims=self.sequence_dim)(x)
        else:
            return self._forward_single(x)

    def __add__(self, t: Callable):
        self._transforms.append(t)
        return self

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, slice):
            return Transforms(*self._transforms[idx], sequence_dim=self.sequence_dim)
        return self._transforms[idx]
