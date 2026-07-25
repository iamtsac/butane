import re
from abc import ABC, abstractmethod
from typing import Callable

import torch

from ..math.ops import *
from .datasets import Dataset, TrajectoryDataset
from .transforms import Transforms


def _get_dict_depth(d: any) -> int:
    """Recursively calculates the maximum nesting depth of a dictionary."""
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(_get_dict_depth(v) for v in d.values())


class Scaler(ABC, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_fitted = False
        self._fitted_data_shape = None
        self._dims = (1,)

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
        self.dims = (1,)

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


class DecoupledScalerSchema(torch.nn.Module):
    """
    A unified PyTorch module schema that decouples and tracks independent
    scaling transformations for input contexts ('data') and target contexts ('targets').
    Supports both nested dictionary structures and pure tensor dataset payloads.
    """

    def __init__(self, scalers_cfg: dict[str, dict[str, type[Scaler]]], sep: str = "/"):
        super().__init__()
        self.scalers_cfg = scalers_cfg
        self.sep = sep

        # Stateful PyTorch containers tracked automatically for .to(device) and saving
        self.data_scalers = torch.nn.ModuleDict()
        self.target_scalers = torch.nn.ModuleDict()

    def fit(self, X: dict[str, dict[str, torch.Tensor]], feature_dim: int) -> None:
        """Fits scalers directly using solid collected PyTorch tensors."""
        scalers_map = {"data": self.data_scalers, "targets": self.target_scalers}

        # Inspect tensor keys and instantiate scaler match rules

        for context in ["data", "targets"]:
            if context not in self.scalers_cfg or context not in X:
                continue

            rules = self.scalers_cfg[context]
            default_scaler_cls = rules.get("default", DummyScaler)

            for key in X[context].keys():
                chosen_scaler_cls = default_scaler_cls
                for pattern_str, scaler_cls in rules.items():
                    if pattern_str != "default" and re.search(pattern_str, key):
                        chosen_scaler_cls = scaler_cls
                        break
                scalers_map[context][key] = chosen_scaler_cls()

        # Fit stateful parameters on aggregated tensors
        with torch.no_grad():
            for context in ["data", "targets"]:
                if context not in X:
                    continue

                for key, data_tensor in X[context].items():
                    if key not in scalers_map[context]:
                        continue

                    scaler_instance = scalers_map[context][key]
                    if isinstance(scaler_instance, DummyScaler):
                        scalers_map[context].pop(key)
                        continue

                    mask = X.get(f"{context}_masks", {}).get(key, None)
                    if mask is not None:
                        valid_data = data_tensor[mask == True]
                    else:
                        valid_data = data_tensor
                    lost_a_dim = (data_tensor.ndim - 1) == valid_data.ndim
                    if lost_a_dim:
                        valid_data = valid_data.unsqueeze(0)
                    if hasattr(scaler_instance, "fit"):
                        scaler_instance.fit(valid_data, dims=feature_dim)

    def forward(
        self,
        batch: dict[str, torch.Tensor] | torch.Tensor,
        context: str | None = "data",
        inverse: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if context is None:
            assert isinstance(batch, dict), "Batch must be a dict when context is None."
            out = {}
            if "data" in batch:
                out["data"] = self.forward(batch["data"], context="data", inverse=inverse)
            if "targets" in batch:
                out["targets"] = self.forward(batch["targets"], context="targets", inverse=inverse)
            for k, v in batch.items():
                if k not in out:
                    out[k] = v
            return out

        assert context in ["data", "targets"], "Context must be 'data' or 'targets'."
        scaler_group = self.data_scalers if context == "data" else self.target_scalers

        if torch.is_tensor(batch):
            return (
                scaler_group["__root__"](batch, inverse=inverse)
                if "__root__" in scaler_group
                else batch
            )

        return {
            k: scaler_group[k](v, inverse=inverse) if k in scaler_group else v
            for k, v in batch.items()
        }
