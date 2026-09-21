import re
from abc import ABC, abstractmethod
from typing import Callable

import torch

from ..math.ops import *
from .datasets import Dataset, TrajectoryDataset
from .transforms import Transforms


class Scaler(ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_fitted = False
        self._fitted_data_shape = None
        self.dims = (1,)

    @abstractmethod
    def _scale(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _unscale(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def fit(
        self,
        X: torch.Tensor,
        dims: int | tuple[int] = (1,),
        transforms: Callable | None = None,
    ) -> None: ...

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not self.is_fitted:
            return x

        self.to(device=x.device)
        if inverse:
            out = self._unscale(x)
        else:
            out = self._scale(x)

        if out.ndim > x.ndim:
            out = out.squeeze(0)
        return out

    def get_extra_state(self) -> dict:
        return {"is_fitted": self.is_fitted, "dims": self.dims}

    def set_extra_state(self, state: dict) -> None:
        self.is_fitted = state["is_fitted"]
        self.dims = state["dims"]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Fitting reshapes the buffers to the data, so an unfitted scaler takes the saved shapes first
        for name, buffer in self._buffers.items():
            if buffer is not None and prefix + name in state_dict:
                self._buffers[name] = torch.empty_like(state_dict[prefix + name], device=buffer.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class DummyScaler(Scaler):
    def __init__(self) -> None:
        super().__init__()

    def fit(
        self,
        X: torch.Tensor,
        dims: int | tuple[int] = (1,),
        transforms: Callable | None = None,
    ) -> None:
        self.is_fitted = True

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StandardScaler(Scaler):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mu", torch.tensor(0, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(1, dtype=torch.float32))

    def fit(
        self,
        X: torch.Tensor,
        dims: int | tuple[int] = (1,),
        transforms: Callable | None = None,
    ) -> None:

        self.dims = dims
        if transforms is not None:
            X = transforms(X)
        self._fitted_data_shape = X.shape
        self.register_buffer("mu", apply_around_dim(torch.mean, X, self.dims, keepdim=True))
        self.register_buffer("std", apply_around_dim(torch.std, X, self.dims, keepdim=True) + 1e-12)
        self.is_fitted = True

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) / self.std

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mu

    def __repr__(self) -> str:
        return f"StandardScaler(mean={self.mu.flatten()}, std={self.std.flatten()})"


class MinMaxScaler(Scaler):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("min_val", torch.tensor(min_val, dtype=torch.float32))
        self.register_buffer("max_val", torch.tensor(max_val, dtype=torch.float32))
        self.register_buffer("xmax", torch.empty(0, dtype=torch.float32))
        self.register_buffer("xmin", torch.empty(0, dtype=torch.float32))

    def fit(
        self,
        X: torch.Tensor,
        dims: int | tuple[int] = 1,
        transforms: Callable | None = None,
    ) -> None:

        self.dims = dims
        if transforms:
            X = transforms(X)

        self._fitted_data_shape = X.shape
        self.register_buffer("xmin", apply_around_dim(torch.amin, X, dims=self.dims, keepdim=True))
        self.register_buffer("xmax", apply_around_dim(torch.amax, X, dims=self.dims, keepdim=True))
        self.is_fitted = True

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        x_std = (x - self.xmin) / (self.xmax - self.xmin + eps)
        return x_std * (self.max_val - self.min_val) + self.min_val

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        x_std = (x - self.min_val) / (self.max_val - self.min_val + eps)
        return x_std * (self.xmax - self.xmin) + self.xmin

    def __repr__(self) -> str:
        return f"MinMaxScaler(min={self.xmin.flatten()}, max={self.xmax.flatten()})"


class ManualScaler(Scaler):
    def __init__(self, scale: float | list[float] | tuple[float] = 1.0) -> None:
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def fit(
        self,
        X: torch.Tensor,
        dims: int | tuple[int] = 1,
        transforms: Callable | None = None,
    ) -> None:
        self.dims = dims
        self.is_fitted = True

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        return x / (self.scale + eps)

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        return x * (self.scale + eps)

    def __repr__(self) -> str:
        return f"ManualScaler(scale={self.scale.flatten()})"
