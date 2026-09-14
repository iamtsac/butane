import torch
from ...._typedefs import *
from ...modules.embeddings import (
    PatchEmbeddings1d,
    PatchEmbeddings2d,
    PatchEmbeddings3d,
)
from ._base_transformer import _BaseTransformer


class ViT1d(_BaseTransformer):
    patch_embedder = PatchEmbeddings1d
    N = 1

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
        learn_input_embeddings: bool = False,
        ctx_dim: int | None = None,
        ctx_patch_size: int | None = None,
        ctx_cross_attention: bool = False,
        cross_attention_heads: int | None = None,
        ctx_in_context: bool = False,
        activation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(
            input_dims=input_dims, hidden_dims=hidden_dims, mlp_ratio=mlp_ratio,
            patch_size=patch_size, depth=depth, attention_heads=attention_heads,
            attention_dropout=attention_dropout, output_dims=output_dims,
            time_dependent=False, time_embedding_size=None, time_scaling_coeff=1.0,
            embedding_size=None, learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=False, learn_ctx_embeddings=False, adaLN_zero=False,
            n_classes=None, class_drop_prob=0.0, ctx_dim=ctx_dim, ctx_patch_size=ctx_patch_size,
            ctx_concat=False, ctx_cross_attention=ctx_cross_attention,
            cross_attention_heads=cross_attention_heads, ctx_in_context=ctx_in_context, activation=activation
        )


class ViT2d(_BaseTransformer):
    patch_embedder = PatchEmbeddings2d
    N = 2

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
        learn_input_embeddings: bool = False,
        ctx_dim: int | None = None,
        ctx_patch_size: int | None = None,
        ctx_cross_attention: bool = False,
        cross_attention_heads: int | None = None,
        ctx_in_context: bool = False,
        activation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(
            input_dims=input_dims, hidden_dims=hidden_dims, mlp_ratio=mlp_ratio,
            patch_size=patch_size, depth=depth, attention_heads=attention_heads,
            attention_dropout=attention_dropout, output_dims=output_dims,
            time_dependent=False, time_embedding_size=None, time_scaling_coeff=1.0,
            embedding_size=None, learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=False, learn_ctx_embeddings=False, adaLN_zero=False,
            n_classes=None, class_drop_prob=0.0, ctx_dim=ctx_dim, ctx_patch_size=ctx_patch_size,
            ctx_concat=False, ctx_cross_attention=ctx_cross_attention,
            cross_attention_heads=cross_attention_heads, ctx_in_context=ctx_in_context, activation=activation
        )


class ViT3d(_BaseTransformer):
    patch_embedder = PatchEmbeddings3d
    N = 3

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
        learn_input_embeddings: bool = False,
        ctx_dim: int | None = None,
        ctx_patch_size: int | None = None,
        ctx_cross_attention: bool = False,
        cross_attention_heads: int | None = None,
        ctx_in_context: bool = False,
        activation: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(
            input_dims=input_dims, hidden_dims=hidden_dims, mlp_ratio=mlp_ratio,
            patch_size=patch_size, depth=depth, attention_heads=attention_heads,
            attention_dropout=attention_dropout, output_dims=output_dims,
            time_dependent=False, time_embedding_size=None, time_scaling_coeff=1.0,
            embedding_size=None, learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=False, learn_ctx_embeddings=False, adaLN_zero=False,
            n_classes=None, class_drop_prob=0.0, ctx_dim=ctx_dim, ctx_patch_size=ctx_patch_size,
            ctx_concat=False, ctx_cross_attention=ctx_cross_attention,
            cross_attention_heads=cross_attention_heads, ctx_in_context=ctx_in_context, activation=activation
        )
