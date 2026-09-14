from typing import Optional
import math
import torch

from ..._typedefs import *


class LearnableEmbeddings(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_len: Optional[int] = 10_000):
        super().__init__()
        self._embeddings = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2, (
            f"Positional embeddings need a sequence axis (B, L, ...), got {x.dim()}D"
        )
        pos = torch.arange(x.size(1), dtype=torch.int32, device=self._embeddings.weight.device)
        return self._embeddings(pos)

    @property
    def embeddings(self) -> torch.Tensor:
        return self._embeddings.weight

class FourierEmbeddings(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        max_seq_len: Optional[int] = 10_000,
        learnable: bool = False
    ) -> None:
        super().__init__()
        d_model_half = d_model // 2
        _omega = torch.exp(
            -math.log(max_seq_len) * (torch.arange(0, d_model_half, dtype=torch.float32) / d_model_half)
        )
        self._omega = torch.nn.Parameter(_omega, requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x[..., None]
        seq_len = x.size(1)
        phase = self._omega[None] * x
        embeddings = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return embeddings

    @staticmethod
    def get_embeddings(x: torch.Tensor, d_model: int, max_seq_len: Optional[int] = 10_000) -> torch.Tensor:
        return FourierEmbeddings(d_model, max_seq_len=max_seq_len)(x)

class SinusoidalEmbeddings(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        max_seq_len: Optional[int] = 10_000,
        learnable: bool = False
    ) -> None :
        super().__init__()
        _pe = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.pow(10_000, torch.arange(0, d_model, 2) / d_model)
        # An odd d_model leaves one more sin channel than cos, so each half takes the slice of
        # div_term it actually has room for.
        n_sin, n_cos = _pe[:, 0::2].shape[1], _pe[:, 1::2].shape[1]
        _pe[:, 0::2] = torch.sin(pos / div_term[:n_sin])
        _pe[:, 1::2] = torch.cos(pos / div_term[:n_cos])
        self._PE = torch.nn.Parameter(_pe, requires_grad=learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2, (
            f"Positional embeddings need a sequence axis (B, L, ...), got {x.dim()}D"
        )
        seq_len = x.size(1)
        assert seq_len <= self._PE.size(0), (
            f"Sequence of {seq_len} exceeds the table's {self._PE.size(0)}; raise max_seq_len."
        )
        return self._PE[:seq_len, :]

    @staticmethod
    def get_embeddings(x: torch.Tensor, d_model: int, max_seq_len: Optional[int] = 10_000) -> torch.Tensor:
        return SinusoidalEmbeddings(d_model, max_seq_len=max_seq_len)(x)

class PatchEmbeddingsNd(torch.nn.Module):
    conv: torch.nn.Module

    def __init__(
        self,
        input_dims: IntParams,
        d_model: int,
        patch_size: int | IntParams = 16,
        bias: Optional[bool] = False,
        normalization: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self._input_dims = input_dims
        n_dims = len(self._input_dims) - 1 # remove channel dim
        self._d_model = d_model

        self._patch_size = patch_size
        if not isinstance(patch_size, (tuple, list)):
            self._patch_size = tuple([patch_size for _ in range(n_dims)])

        if normalization is not None:
            self.norm = normalization

        self.patchify = self.conv(
            in_channels=self._input_dims[0],
            out_channels=self._d_model,
            kernel_size=self._patch_size,
            stride=self._patch_size,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        patches = patches.flatten(2).transpose(-1,-2)
        if hasattr(self, 'norm'):
            patches = self.norm(patches)
        return patches

class PatchEmbeddings1d(PatchEmbeddingsNd):
    conv = torch.nn.Conv1d

class PatchEmbeddings2d(PatchEmbeddingsNd):
    conv = torch.nn.Conv2d

class PatchEmbeddings3d(PatchEmbeddingsNd):
    conv = torch.nn.Conv3d

class RelativePositionEmbeddings(torch.nn.Module):

    def __init__(self, d_model: int, max_seq_len: Optional[int] = 10_000):
        super().__init__()
        self._k = torch.nn.Embedding(max_seq_len, d_model)
        self._v = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self._PE[:seq_len, :].to(self.__dummy_param.device)
        return x

    @property
    def embeddings(self) -> torch.Tensor:
        return self._PE.to(self.__dumy_param.device)
