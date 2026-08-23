import math
from typing import Callable, Optional, Union

import torch

from ..._helpers import _fill_defaults, _prod, module_name
from ..._typedefs import *
from ..utils import utils


class _AttentionTemplate(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        kv_input_size: Optional[int] = None,
        apply_residual: bool = True,
        n_heads: int = 1,
        dropout_p: float = 0.0,
        causal: bool = False,
        prenorm: bool = True,
        zero_out: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = self._d_model // self._n_heads
        self._causal = causal
        self._apply_residual = apply_residual
        self._is_cross = kv_input_size is not None
        self._prenorm_enabled = prenorm
        self._kv_input_size = kv_input_size if kv_input_size is not None else self._d_model

        self.scale_factor = math.sqrt(self.d_k)

        # fmt: off
        self.query = torch.nn.Linear(self._d_model, self._d_model)
        self.key = torch.nn.Linear(self._kv_input_size, self._d_model)
        self.value = torch.nn.Linear(self._kv_input_size, self._d_model)
        # fmt: on

        self.norm_1 = (
            torch.nn.LayerNorm(self._d_model) if self._prenorm_enabled else torch.nn.Identity()
        )
        self.norm_2 = (
            torch.nn.LayerNorm(self._kv_input_size)
            if self._prenorm_enabled and self._is_cross
            else torch.nn.Identity()
        )

        self.dropout = torch.nn.Dropout(dropout_p)

        self.linear_projection = (
            torch.nn.Linear(self._d_model, self._d_model)
            if not zero_out
            else utils.zero_module(torch.nn.Linear(self._d_model, self._d_model))
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert x1.dim() == 3, f"Sequence attention expects 3D inputs (B, L, D), got {x1.dim()}D"
        if x2 is None:
            x2 = x1
        assert x2.dim() == 3, f"Sequence attention expects 3D inputs (B, L, D), got {x2.dim()}D"

        B, L_Q, *_spatial = x1.shape
        _, L_KV, _ = x2.shape

        q_input = self.norm_1(x1) if self._prenorm_enabled else x1

        if self._is_cross:
            kv_input = self.norm_2(x2) if self._prenorm_enabled else x2
        else:
            kv_input = q_input

        _q = self.query(q_input).reshape(B, L_Q, self._n_heads, self.d_k).transpose(1, 2)
        _k = self.key(kv_input).reshape(B, L_KV, self._n_heads, self.d_k).transpose(1, 2)
        _v = self.value(kv_input).reshape(B, L_KV, self._n_heads, self.d_k).transpose(1, 2)

        attn_mask = (
            utils.normalize_attention_mask(mask, self._n_heads, L_Q, L_KV) if mask is not None else None
        )
        _attention = torch.nn.functional.scaled_dot_product_attention(
            query=_q,
            key=_k,
            value=_v,
            attn_mask=attn_mask,
            is_causal=self._causal and mask is None,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=1.0 / self.scale_factor,
        )

        _attention = _attention.transpose(1, 2).reshape(B, L_Q, self._d_model)

        out = self.linear_projection(_attention.contiguous())
        if self._apply_residual:
            out = x1 + out
        return out.reshape(*out.shape[:-1], *_spatial)


class SelfAttention(_AttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        kwargs.pop("kv_input_size", None)
        super().__init__(d_model, kv_input_size=None, **kwargs)

    def forward(self, x1: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x1=x1, x2=None, mask=mask)


class CrossAttention(_AttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        if kwargs.get("kv_input_size") is None:
            kwargs["kv_input_size"] = d_model

        super().__init__(d_model, **kwargs)
        self._is_cross = True


class _SpatialAttentionTemplate(torch.nn.Module):
    conv: torch.nn.Module
    N: int

    def __init__(
        self,
        d_model: int,
        *,
        kv_input_size: Optional[int] = None,
        kv_n_dims: Optional[int] = None,
        kernel_size: int = 1,
        n_heads: int = 1,
        dropout_p: float = 0.0,
        bias: bool = False,
        prenorm: Optional[ModuleParams] = None,
        zero_out: bool = False,
        apply_residual: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = int(self._d_model) // self._n_heads
        self._is_cross = kv_input_size is not None
        self._kv_input_size = kv_input_size if kv_input_size is not None else self._d_model
        self._apply_residual = apply_residual
        self._dropout_p = dropout_p
        _padding = kernel_size // 2

        self.scale_factor = math.sqrt(self.d_k)

        self.kv_conv = self.conv
        if kv_n_dims is not None and kv_n_dims != self.N:
            if kv_n_dims == 1:
                self.kv_conv = torch.nn.Conv1d
            elif kv_n_dims == 2:
                self.kv_conv = torch.nn.Conv2d
            elif kv_n_dims == 3:
                self.kv_conv = torch.nn.Conv3d

        # fmt: off
        self.query = self.conv(self._d_model, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.key = self.kv_conv(self._kv_input_size, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.value = self.kv_conv(self._kv_input_size, self._d_model, kernel_size, padding=_padding, bias=bias)
        self.projection = (
            utils.zero_module(self.conv(self._d_model, self._d_model, 1, bias=bias))
            if zero_out
            else self.conv(self._d_model, self._d_model, 1, bias=bias)
        )
        # fmt: on

        self.norm_1, self.norm_2 = None, None
        if prenorm is not None:
            if module_name(prenorm) == "GroupNorm":
                self.norm_1 = prenorm(num_channels=self._d_model)
                self.norm_2 = prenorm(num_channels=self._kv_input_size) if self._is_cross else None
            elif module_name(prenorm) == "LayerNorm":
                raise ValueError("Cannot use LayerNorm in spatial attention")
            else:
                self.norm_1 = prenorm(self._d_model)
                self.norm_2 = prenorm(self._kv_input_size) if self._is_cross else None

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual_ = x1
        if x2 is None:
            x2 = x1

        B1, C1, *spatial1 = x1.shape
        B2, C2, *spatial2 = x2.shape

        if self.N == 1 and len(spatial1) > 1:
            x1 = x1.flatten(2)
        if self.N == 1 and len(spatial2) > 1:
            x2 = x2.flatten(2)

        L1 = x1.shape[2:].numel()
        L2 = x2.shape[2:].numel()

        q_input = self.norm_1(x1) if self.norm_1 is not None else x1
        if self._is_cross:
            kv_input = self.norm_2(x2) if self.norm_2 is not None else x2
        else:
            kv_input = q_input

        # conv projections
        q = self.query(q_input)
        k = self.key(kv_input)
        v = self.value(kv_input)

        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)

        q = q.transpose(1, 2).reshape(B1, L1, self._n_heads, self.d_k).transpose(1, 2)
        k = k.transpose(1, 2).reshape(B2, L2, self._n_heads, self.d_k).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B2, L2, self._n_heads, self.d_k).transpose(1, 2)

        attn_mask = (
            utils.normalize_attention_mask(mask, self._n_heads, L1, L2) if mask is not None else None
        )
        _attention = torch.nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            scale=1.0 / self.scale_factor,
        )
        _attention = _attention.transpose(1, 2).reshape(B1, L1, C1).transpose(1, 2)

        if self.N == 1:
            out = self.projection(_attention)
            out = out.reshape(B1, C1, *spatial1)
        else:
            _attention = _attention.reshape(B1, C1, *spatial1)
            out = self.projection(_attention)

        if self._apply_residual:
            out = residual_ + out
        return out


class SpatialSelfAttention(_SpatialAttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        kwargs.pop("kv_input_size", None)
        kwargs.pop("kv_n_dims", None)
        super().__init__(d_model, kv_input_size=None, kv_n_dims=None, **kwargs)

    def forward(self, x1: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return super().forward(x1, x2=None, mask=mask)


class SpatialCrossAttention(_SpatialAttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        if kwargs.get("kv_input_size") is None:
            kwargs["kv_input_size"] = d_model

        super().__init__(d_model, **kwargs)
        self._is_cross = True


class SpatialSelfAttention1d(SpatialSelfAttention):
    conv = torch.nn.Conv1d
    N = 1


class SpatialSelfAttention2d(SpatialSelfAttention):
    conv = torch.nn.Conv2d
    N = 2


class SpatialCrossAttention1d(SpatialCrossAttention):
    conv = torch.nn.Conv1d
    N = 1


class SpatialCrossAttention2d(SpatialCrossAttention):
    conv = torch.nn.Conv2d
    N = 2


class SpatialSelfAttention3d(SpatialSelfAttention):
    conv = torch.nn.Conv3d
    N = 3


class SpatialCrossAttention3d(SpatialCrossAttention):
    conv = torch.nn.Conv3d
    N = 3
