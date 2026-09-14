import torch

from ...._typedefs import *
from ...modules.embeddings import (
    FourierEmbeddings,
    PatchEmbeddings1d,
    PatchEmbeddings2d,
    PatchEmbeddings3d,
)
from ._base_transformer import _BaseTransformer


class DiT1d(_BaseTransformer):
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
        super().__init__(
            input_dims=input_dims, hidden_dims=hidden_dims, mlp_ratio=mlp_ratio,
            patch_size=patch_size, depth=depth, attention_heads=attention_heads,
            attention_dropout=attention_dropout, output_dims=output_dims,
            time_dependent=True, time_embedding_size=time_embedding_size,
            time_scaling_coeff=time_scaling_coeff, embedding_size=embedding_size,
            embedder=embedder, learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=learn_time_embeddings, learn_ctx_embeddings=learn_ctx_embeddings,
            adaLN_zero=adaLN_zero, n_classes=n_classes, class_drop_prob=class_drop_prob,
            ctx_dim=ctx_dim, ctx_patch_size=ctx_patch_size, ctx_concat=ctx_concat,
            ctx_cross_attention=ctx_cross_attention, cross_attention_heads=cross_attention_heads,
            ctx_in_context=ctx_in_context, activation=activation
        )


class DiT2d(_BaseTransformer):
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
        super().__init__(
            input_dims=input_dims, hidden_dims=hidden_dims, mlp_ratio=mlp_ratio,
            patch_size=patch_size, depth=depth, attention_heads=attention_heads,
            attention_dropout=attention_dropout, output_dims=output_dims,
            time_dependent=True, time_embedding_size=time_embedding_size,
            time_scaling_coeff=time_scaling_coeff, embedding_size=embedding_size,
            embedder=embedder, learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=learn_time_embeddings, learn_ctx_embeddings=learn_ctx_embeddings,
            adaLN_zero=adaLN_zero, n_classes=n_classes, class_drop_prob=class_drop_prob,
            ctx_dim=ctx_dim, ctx_patch_size=ctx_patch_size, ctx_concat=ctx_concat,
            ctx_cross_attention=ctx_cross_attention, cross_attention_heads=cross_attention_heads,
            ctx_in_context=ctx_in_context, activation=activation
        )


class DiT3d(_BaseTransformer):
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
        super().__init__(
            input_dims=input_dims, hidden_dims=hidden_dims, mlp_ratio=mlp_ratio,
            patch_size=patch_size, depth=depth, attention_heads=attention_heads,
            attention_dropout=attention_dropout, output_dims=output_dims,
            time_dependent=True, time_embedding_size=time_embedding_size,
            time_scaling_coeff=time_scaling_coeff, embedding_size=embedding_size,
            embedder=embedder, learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=learn_time_embeddings, learn_ctx_embeddings=learn_ctx_embeddings,
            adaLN_zero=adaLN_zero, n_classes=n_classes, class_drop_prob=class_drop_prob,
            ctx_dim=ctx_dim, ctx_patch_size=ctx_patch_size, ctx_concat=ctx_concat,
            ctx_cross_attention=ctx_cross_attention, cross_attention_heads=cross_attention_heads,
            ctx_in_context=ctx_in_context, activation=activation
        )
