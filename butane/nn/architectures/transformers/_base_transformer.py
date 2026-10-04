import copy
import math
import warnings
from functools import partial, reduce
from typing import Any, Type

import torch

from butane.nn.architectures.transformers.transformer_utils import unpatchify

from ...._typedefs import *
from ...modules.attention import (
    CrossAttention,
    SelfAttention,
)
from ...modules.embeddings import (
    FourierEmbeddings,
    PatchEmbeddingsNd,
)
from ...modules.mlp_block import MLPBlock
from ...modules.residual_blocks import *
from ...utils import utils
from .transformer_utils import *


def _per_token(modulation: torch.Tensor) -> torch.Tensor:
    """(B, F) -> (B, 1, F), one modulation broadcast over the tokens; (B, N, F) is already one per token."""
    return modulation.unsqueeze(1) if modulation.dim() == 2 else modulation


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_ratio: float,
        embedding_size: int | None = None,
        attention_heads: int = 8,
        cross_attention_heads: int | None = None,
        attention_dropout: float = 0.0,
        adaLN_zero: bool = True,
        ctx_cross_attention: bool = False,
        ctx_dim: int | None = None,
        activation: torch.nn.Module | None = None,
        causal: bool = False,
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = int(input_dims * output_ratio)
        self._embedding_size = embedding_size
        self._ctx_cross_attention = ctx_cross_attention
        self._cross_attention_heads = (
            cross_attention_heads if cross_attention_heads is not None else attention_heads
        )
        self._adaLN_zero_path = adaLN_zero

        self._has_condition = self._embedding_size is not None

        # If unconditioned, use standard LayerNorm with learnable parameters (elementwise_affine=True)
        self.norm_pre_sa = torch.nn.LayerNorm(
            self._input_dims, elementwise_affine=not self._has_condition, eps=1e-06
        )
        self.norm_pre_mlp = torch.nn.LayerNorm(
            self._input_dims, elementwise_affine=not self._has_condition, eps=1e-06
        )

        self.attn = SelfAttention(
            self._input_dims,
            n_heads=attention_heads,
            dropout_p=attention_dropout,
            prenorm=False,
            causal=causal,
        )

        self.mlp = MLPBlock(
            input_dims=self._input_dims,
            output_dims=self._input_dims,
            hidden_dims=[self._hidden_dims],
            activation_function=[
                activation if activation is not None else torch.nn.GELU(approximate="tanh")
            ],
            output_activation=False,
        )

        # Conditionally instantiate conditioning layers to save memory/parameters
        if self._has_condition:
            self.modulation_module = torch.nn.Sequential(
                torch.nn.SiLU(),
                utils.zero_module(
                    torch.nn.Linear(
                        self._embedding_size, (6 if self._adaLN_zero_path else 4) * self._input_dims
                    )
                ),
            )

        if self._ctx_cross_attention:
            self.cattn_context_norm = torch.nn.LayerNorm(self._input_dims, elementwise_affine=False)
            self.cattn_context_module = CrossAttention(
                self._input_dims,
                kv_input_dim=ctx_dim,
                n_heads=self._cross_attention_heads,
                dropout_p=attention_dropout,
                prenorm=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
        *,
        pos: torch.Tensor | None = None,
        ctx_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Vanilla Transformer/ViT/Trajectory Pathway (Zero modulation overhead)
        if not self._has_condition or emb is None:
            x = x + self.attn(self.norm_pre_sa(x), mask=mask, pos=pos)
            if self._ctx_cross_attention and ctx is not None:
                x = x + self.cattn_context_module(
                    self.cattn_context_norm(x), ctx, mask=ctx_mask, q_pos=pos, k_pos=ctx_pos
                )
            x = x + self.mlp(self.norm_pre_mlp(x))
            return x

        # Positions are applied inside attention, i.e. AFTER the adaLN modulation below.
        # Modulation is a content operation driven by the conditioning vector; scaling the
        # positional signal by gamma would make "where a token sits" conditioning-dependent.
        modulation = _per_token(self.modulation_module(emb))
        if self._adaLN_zero_path:
            gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = modulation.chunk(chunks=6, dim=-1)
            x = x + alpha_1 * self.attn(
                self.norm_pre_sa(x) * (1 + gamma_1) + beta_1,
                mask=mask,
                pos=pos,
            )

            if self._ctx_cross_attention and ctx is not None:
                x = x + self.cattn_context_module(
                    self.cattn_context_norm(x), ctx, mask=ctx_mask, q_pos=pos, k_pos=ctx_pos
                )

            x = x + alpha_2 * self.mlp(self.norm_pre_mlp(x) * (1 + gamma_2) + beta_2)
        else:
            gamma_1, beta_1, gamma_2, beta_2 = modulation.chunk(chunks=4, dim=-1)
            x = x + self.attn(
                self.norm_pre_sa(x) * (1 + gamma_1) + beta_1,
                mask=mask,
                pos=pos,
            )

            if self._ctx_cross_attention and ctx is not None:
                x = x + self.cattn_context_module(
                    self.cattn_context_norm(x), ctx, mask=ctx_mask, q_pos=pos, k_pos=ctx_pos
                )

            x = x + self.mlp(self.norm_pre_mlp(x) * (1 + gamma_2) + beta_2)
        return x


class FinalBlock(torch.nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_channels: int,
        patch_size: int = 1,
        embedding_size: int | None = None,
    ) -> None:

        super().__init__()
        self._input_dims = input_dims
        self._output_channels = output_channels
        self._embedding_size = embedding_size
        self._has_condition = embedding_size is not None

        p_size_tuple = (patch_size,) if isinstance(patch_size, int) else patch_size

        self.norm = torch.nn.LayerNorm(self._input_dims, elementwise_affine=not self._has_condition)
        self.fc1 = utils.zero_module(
            torch.nn.Linear(
                self._input_dims, reduce(lambda x, y: x * y, p_size_tuple) * output_channels
            )
        )

        if self._has_condition:
            self.film = torch.nn.Sequential(
                torch.nn.SiLU(),
                utils.zero_module(torch.nn.Linear(self._embedding_size, 2 * self._input_dims)),
            )

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        if self._has_condition and emb is not None:
            gamma, beta = _per_token(self.film(emb)).chunk(chunks=2, dim=-1)
            x = self.norm(x) * (1 + gamma) + beta
        else:
            x = self.norm(x)
        x = self.fc1(x)
        return x


class _BaseTransformer(torch.nn.Module):
    patch_embedder: type[PatchEmbeddingsNd]
    N: int = -1

    def __init__(
        self,
        input_dims: IntParams,
        hidden_dims: int = 1152,
        mlp_ratio: float = 4.0,
        patch_size: int = 8,
        depth: int = 12,
        attention_heads: int = 16,
        attention_dropout: float = 0.0,
        output_dims: int | None = None,
        time_dependent: bool = True,
        time_embedding_size: int | None = 64,
        time_scaling_coeff: float = 1.0,
        embedding_size: int | None = None,
        embedder: torch.nn.Module = FourierEmbeddings,
        learn_input_embeddings: bool = False,
        learn_time_embeddings: bool = False,
        learn_ctx_embeddings: bool = False,
        adaLN_zero: bool = True,
        n_classes: int | None = None,
        class_drop_prob: float = 0.0,
        ctx_dim: int | None = None,
        ctx_patch_size: int | None = None,
        ctx_concat: bool = False,
        ctx_cross_attention: bool = False,
        cross_attention_heads: int | None = None,
        ctx_in_context: bool = False,
        activation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._patch_size = patch_size
        self._output_dims = self._input_dims[0] if output_dims is None else output_dims
        self._attention_heads = attention_heads
        self._cross_heads = cross_attention_heads
        self._attention_dropout = attention_dropout
        self._mlp_ratio = mlp_ratio
        self._adaLN_zero_path = adaLN_zero
        self._learn_embeddings = learn_time_embeddings
        self._learn_ctx_embeddings = learn_ctx_embeddings
        self._learn_input_embeddings = learn_input_embeddings

        self._ctx_dim = ctx_dim
        self._ctx_in_context = ctx_in_context
        self._ctx_cross_attention = ctx_cross_attention
        self._ctx_concat = ctx_concat

        # Enforce exclusive conditioning choices
        assert self._ctx_in_context ^ self._ctx_cross_attention or not (
            self._ctx_in_context or self._ctx_cross_attention
        ), "Multiple types of conditioning were selected, but only one type should be"

        if not isinstance(self._patch_size, (tuple, list)):
            self._patch_size = (self._patch_size,) * (len(self._input_dims) - 1)

        self._n_classes = n_classes
        self._class_drop_prob = class_drop_prob

        self._has_ctx = self._ctx_in_context or self._ctx_cross_attention or self._ctx_concat

        self._time_dependent = time_dependent
        if self._time_dependent:
            self._time_scaling_coeff = time_scaling_coeff
            self._time_embedding_size = (
                time_embedding_size if time_embedding_size is not None else self._input_dims[0]
            )
            self._embedding_size = (
                embedding_size if embedding_size is not None else self._time_embedding_size * 4
            )

            self.time_embedder = embedder(
                d_model=self._embedding_size, learnable=self._learn_embeddings
            )
            self.embedding_projection = MLPBlock(
                input_dims=self._embedding_size,
                output_dims=self._hidden_dims,
                hidden_dims=[self._hidden_dims],
                activation_function=[torch.nn.SiLU()],
                output_activation=False,
                bias=[True],
            )

        assert not reduce(
            lambda x, y: x + y,
            [self._input_dims[1:][i] % self._patch_size[i] for i in range(len(self._patch_size))],
        ), "Input dimensions are not divisable by the patch size!"

        _emb = get_positional_embeddings(
            embedding_callback_fn=FourierEmbeddings.get_embeddings,
            d_model=self._hidden_dims,
            input_dims=self._input_dims,
            patch_size=self._patch_size,
        )
        self._input_embeddings = torch.nn.Parameter(
            _emb.float().unsqueeze(0),
            requires_grad=self._learn_input_embeddings,
        )
        self.patchify = self.patch_embedder(
            self._input_dims, patch_size=self._patch_size, d_model=self._hidden_dims, bias=True
        )

        if self._n_classes is not None:
            self.class_embedder = torch.nn.Embedding(
                self._n_classes if self._class_drop_prob == 0.0 else self._n_classes + 1,
                self._hidden_dims,
            )
            self._null_class_idx = self._n_classes

        self._has_global_cond = (
            self._time_dependent or (self._n_classes is not None) or self._ctx_concat
        )

        block_embedding_dims = None
        if self._has_global_cond:
            if self._time_dependent or (self._n_classes is not None):
                block_embedding_dims = self._hidden_dims
                if self._ctx_concat:
                    if self._ctx_dim is None:
                        raise ValueError("ctx_dim must be provided when ctx_concat=True")
                    block_embedding_dims += self._ctx_dim
            else:
                # time_dependent=False and n_classes=None, but ctx_concat=True
                if self._ctx_dim is None:
                    raise ValueError("ctx_dim must be provided when ctx_concat=True")
                block_embedding_dims = self._ctx_dim  # Only the vector context is present

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    input_dims=self._hidden_dims,
                    output_ratio=self._mlp_ratio,
                    embedding_size=block_embedding_dims,
                    attention_heads=self._attention_heads,
                    cross_attention_heads=self._cross_heads,
                    attention_dropout=self._attention_dropout,
                    adaLN_zero=self._adaLN_zero_path,
                    ctx_cross_attention=self._ctx_cross_attention,
                    ctx_dim=self._ctx_dim,
                    activation=activation,
                )
                for _ in range(depth)
            ]
        )

        self.output_layer = FinalBlock(
            self._hidden_dims,
            self._output_dims,
            patch_size=self._patch_size,
            embedding_size=block_embedding_dims,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self._time_dependent:
            utils.init_weights(
                self.embedding_projection,
                weight_init_method=torch.nn.init.xavier_uniform_,
                bias_init_method=partial(torch.nn.init.constant_, val=0),
            )

        for block in self.transformer_blocks:
            if getattr(block, "modulation_module", None) is not None:
                utils.zero_module(block.modulation_module)

        utils.zero_module(self.output_layer.fc1)
        if getattr(self.output_layer, "film", None) is not None:
            utils.zero_module(self.output_layer.film)

    def _prepare_time(self, t: torch.Tensor | None = None) -> torch.Tensor | None:
        emb = None
        if self._time_dependent:
            if t is None:
                raise ValueError(
                    "Model is configured as `time_dependent=True`, but time step `t` is None."
                )
            t = t.view(-1)
            emb = self.time_embedder(t)
            emb = self.embedding_projection(emb)
        elif t is not None:
            raise ValueError(
                "Model is configured as `time_dependent=False`, but time step `t` was provided."
            )
        return emb

    def _prepare_labels(
        self,
        emb: torch.Tensor | None,
        labels: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self._n_classes is not None:
            if labels is None:
                # Classifier-Free Guidance (CFG): Unconditional generation fallback
                if self._class_drop_prob > 0:
                    batch_size = emb.shape[0] if emb is not None else 1
                    labels = torch.full(
                        (batch_size,), self._null_class_idx, device=device, dtype=torch.long
                    )
                else:
                    raise ValueError(
                        f"Model requires {self._n_classes} class labels, but `labels` is None."
                    )
            else:
                assert len(labels.shape) == 1
                if self.training and self._class_drop_prob > 0:
                    drop_mask = torch.bernoulli(
                        torch.full(labels.shape, self._class_drop_prob, device=labels.device)
                    ).bool()
                    labels = torch.where(drop_mask, self._null_class_idx, labels)

            y_emb = self.class_embedder(labels)
            emb = emb + y_emb if emb is not None else y_emb
        return emb

    def _prepare_conditioning(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ctx is not None and self._has_ctx:
            if self._ctx_in_context or self._ctx_cross_attention:
                if self._ctx_in_context:
                    x = torch.cat([x, ctx], dim=1)

            elif ctx.ndim == 2:
                if self._ctx_concat:
                    if emb is not None:
                        emb = torch.cat([emb, ctx], dim=-1)
                    else:
                        emb = ctx  # Flat vector acts directly as the global style modulator
                else:
                    if emb is not None and emb.shape == ctx.shape:
                        emb = emb + ctx
        return x, emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._time_dependent and t is not None:
            t = t * self._time_scaling_coeff

        x = self.patchify(x) + self._input_embeddings
        emb = self._prepare_time(t=t)
        emb = self._prepare_labels(emb=emb, labels=labels, device=x.device)
        x, emb = self._prepare_conditioning(x, emb, ctx=ctx)
        return self._forward(x, emb, ctx=ctx, mask=mask, ctx_mask=ctx_mask)

    def _forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_n_patches = x.size(1)
        if self._ctx_in_context and ctx is not None:
            x_n_patches = x_n_patches - ctx.size(1)

        for block in self.transformer_blocks:
            x = block(x, emb, ctx, mask=mask, ctx_mask=ctx_mask)

        if self._ctx_in_context and ctx is not None:
            x = x[:, :x_n_patches]

        x = self.output_layer(x, emb)
        x = unpatchify(
            x,
            input_dims=self._input_dims,
            patch_size=self._patch_size,
            output_dims=self._output_dims,
        )
        return x
