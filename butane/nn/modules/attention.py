import math
from typing import Callable, Optional, Union

import torch

from ..._helpers import _fill_defaults, _prod, module_name
from ..._typedefs import *
from ..utils import utils


def normalize_attention_mask(
    mask: torch.Tensor,
    n_heads: int,
    L_Q: int,
    L_KV: int,
) -> torch.Tensor:
    """
    Broadcasts an attention mask (1/True = attend, 0/False = masked out) to
    (B, H, L_Q, L_KV) regardless of whether it was given as a padding mask
    over queries (B, L_Q), a padding mask over keys/values (B, L_KV), a full
    pairwise mask (B, L_Q, L_KV), or an already per-head mask (B, H, L_Q, L_KV).
    """
    if mask.dim() == 2:
        if mask.size(-1) == L_KV:
            mask = mask[:, None, None, :]
        elif mask.size(-1) == L_Q:
            mask = mask[:, None, :, None]
        else:
            raise ValueError(
                f"2D attention mask has last dim {mask.size(-1)}, expected L_Q={L_Q} or L_KV={L_KV}"
            )
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() != 4:
        raise ValueError(f"Attention mask must be 2D, 3D or 4D, got {mask.dim()}D")
    return mask.bool()


class _AttentionTemplate(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        q_input_dim: int | None = None,
        *,
        kv_input_dim: int | None = None,
        n_heads: int = 1,
        dropout_p: float = 0.0,
        causal: bool = False,
        bias: bool = True,
        prenorm: bool = True,
        zero_out: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "Features cannot be devided equally to N heads"
        # Causal masking orders keys against queries by index. Across two unrelated sequences
        # that comparison is meaningless, and SDPA would silently apply it anyway.
        assert not (causal and kv_input_dim is not None), (
            "causal=True is only meaningful for self-attention; a cross-attention query index "
            "says nothing about which context positions may be attended."
        )

        self._d_model = d_model
        self._n_heads = n_heads
        self.d_k = self._d_model // self._n_heads
        self._causal = causal
        self._is_cross = kv_input_dim is not None
        self._prenorm_enabled = prenorm
        self._dropout_p = dropout_p
        self._q_input_dim = q_input_dim if q_input_dim is not None else self._d_model
        self._kv_input_dim = kv_input_dim if kv_input_dim is not None else self._q_input_dim

        self.scale_factor = math.sqrt(self.d_k)

        self.q_prenorm = (
            torch.nn.LayerNorm(self._q_input_dim) if self._prenorm_enabled else torch.nn.Identity()
        )
        self.kv_prenorm = (
            torch.nn.LayerNorm(self._kv_input_dim)
            if self._prenorm_enabled and self._is_cross
            else torch.nn.Identity()
        )

        # fmt: off
        self.query = torch.nn.Linear(self._q_input_dim, self._d_model, bias=bias)
        self.key = torch.nn.Linear(self._kv_input_dim, self._d_model, bias=bias)
        self.value = torch.nn.Linear(self._kv_input_dim, self._d_model, bias=bias)
        # fmt: on

        self.linear_projection = (
            torch.nn.Linear(self._d_model, self._d_model, bias=bias)
            if not zero_out
            else utils.zero_module(torch.nn.Linear(self._d_model, self._d_model, bias=bias))
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        q_pos: torch.Tensor | None = None,
        k_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:

        assert q.dim() == 3, f"Sequence attention expects 3D inputs (B, L, D), got {q.dim()}D"
        if kv is None:
            kv = q
        assert kv.dim() == 3, f"Sequence attention expects 3D inputs (B, L, D), got {kv.dim()}D"
        assert self._is_cross or kv is q, (
            "Built without kv_input_dim, so this module attends over its own query stream and "
            "the kv argument would be ignored. Pass kv_input_dim (or use CrossAttention)."
        )

        B, L_Q, _ = q.shape
        _, L_KV, _ = kv.shape

        q_input = self.q_prenorm(q)

        if self._is_cross:
            kv_input = self.kv_prenorm(kv)
        else:
            kv_input = q_input

        # Positions ride on the query/key only; the value stream stays positionally clean, so
        # what a token *is* never gets mixed with where it sits. Added after the prenorms -
        # adding before would let LayerNorm rescale the positional signal per token.
        assert q_pos is None or q_pos.shape[-1] == q_input.shape[-1], (
            f"q_pos must match the query width {q_input.shape[-1]}, got {q_pos.shape[-1]}"
        )
        assert k_pos is None or k_pos.shape[-1] == kv_input.shape[-1], (
            f"k_pos must match the key/value width {kv_input.shape[-1]}, got {k_pos.shape[-1]}"
        )

        q_src = q_input if q_pos is None else q_input + q_pos
        k_src = kv_input if k_pos is None else kv_input + k_pos

        _q = self.query(q_src).reshape(B, L_Q, self._n_heads, self.d_k).transpose(1, 2)
        _k = self.key(k_src).reshape(B, L_KV, self._n_heads, self.d_k).transpose(1, 2)
        _v = self.value(kv_input).reshape(B, L_KV, self._n_heads, self.d_k).transpose(1, 2)

        attn_mask = (
            normalize_attention_mask(mask, self._n_heads, L_Q, L_KV) if mask is not None else None
        )
        _attention = torch.nn.functional.scaled_dot_product_attention(
            query=_q,
            key=_k,
            value=_v,
            attn_mask=attn_mask,
            is_causal=self._causal and mask is None,
            dropout_p=self._dropout_p if self.training else 0.0,
            scale=1.0 / self.scale_factor,
        )

        _attention = _attention.transpose(1, 2).reshape(B, L_Q, self._d_model)

        return self.linear_projection(_attention.contiguous())


class SelfAttention(_AttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        kwargs.pop("kv_input_dim", None)
        kwargs.pop("q_input_dim", None)
        super().__init__(d_model, d_model, kv_input_dim=None, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(q=x, kv=None, mask=mask, q_pos=pos, k_pos=pos)


class CrossAttention(_AttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        if kwargs.get("kv_input_dim") is None:
            kwargs["kv_input_dim"] = d_model

        super().__init__(d_model, **kwargs)


class _SpatialAttentionTemplate(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        q_input_dim: int | None = None,
        *,
        kv_input_dim: Optional[int] = None,
        n_heads: int = 1,
        dropout_p: float = 0.0,
        bias: bool = True,
        prenorm: Optional[ModuleParams] = None,
        zero_out: bool = False,
    ):
        super().__init__()

        self._d_model = d_model
        self._is_cross = kv_input_dim is not None
        self._q_input_dim = q_input_dim if q_input_dim is not None else d_model
        self._kv_input_dim = kv_input_dim if kv_input_dim is not None else d_model

        # prenorm=False on the inner module: this wrapper has already normalized, in the layout
        # GroupNorm needs. The inner one would apply LayerNorm across the channel axis of the
        # flattened sequence, which is a different - and for conv features, wrong - norm.
        self.attention = _AttentionTemplate(
            d_model,
            self._q_input_dim,
            kv_input_dim=kv_input_dim,
            n_heads=n_heads,
            dropout_p=dropout_p,
            bias=bias,
            prenorm=False,
            zero_out=zero_out,
        )

        # Identity rather than None, so forward() never tests for their presence.
        self.q_prenorm, self.kv_prenorm = torch.nn.Identity(), torch.nn.Identity()
        if prenorm is not None:
            if module_name(prenorm) == "LayerNorm":
                raise ValueError("Cannot use LayerNorm in spatial attention")
            if module_name(prenorm) == "GroupNorm":
                self.q_prenorm = prenorm(num_channels=self._q_input_dim)
                if self._is_cross:
                    self.kv_prenorm = prenorm(num_channels=self._kv_input_dim)
            else:
                self.q_prenorm = prenorm(self._q_input_dim)
                if self._is_cross:
                    self.kv_prenorm = prenorm(self._kv_input_dim)

    @staticmethod
    def _to_sequence(x: torch.Tensor) -> torch.Tensor:
        """(B, C, *spatial) -> (B, L, C). The one place layout is converted."""
        return x.flatten(2).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        q_pos: Optional[torch.Tensor] = None,
        k_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert q.dim() >= 3, f"Spatial attention expects (B, C, *spatial), got {q.dim()}D"
        B, _, *spatial = q.shape

        q_seq = self._to_sequence(self.q_prenorm(q))
        kv_seq = None if kv is None else self._to_sequence(self.kv_prenorm(kv))

        out = self.attention(q_seq, kv_seq, mask, q_pos=q_pos, k_pos=k_pos)
        return out.transpose(1, 2).reshape(B, self._d_model, *spatial)


class SpatialSelfAttention(_SpatialAttentionTemplate):
    """A feature map attending over itself, at any spatial rank."""

    def __init__(self, d_model: int, **kwargs):
        kwargs.pop("kv_input_dim", None)
        kwargs.pop("q_input_dim", None)
        super().__init__(d_model, d_model, kv_input_dim=None, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().forward(x, kv=None, mask=mask, q_pos=pos, k_pos=pos)


class SpatialCrossAttention(_SpatialAttentionTemplate):
    def __init__(self, d_model: int, **kwargs):
        if kwargs.get("kv_input_dim") is None:
            kwargs["kv_input_dim"] = d_model
        super().__init__(d_model, **kwargs)
