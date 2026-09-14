import torch

from ...._typedefs import *
from ...modules.embeddings import PatchEmbeddings1d
from ._base_transformer import TransformerBlock, _BaseTransformer


def _xavier_init(module: torch.nn.Module) -> None:
    """Xavier-uniform every Linear, zero every bias."""

    def _init(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    module.apply(_init)


class TransformerEncoder(torch.nn.Module):
    """A plain stack of self-attention blocks over one sequence.

    No conditioning, no timestep, no patching - tokens in, tokens out. Wrap it or precede it
    with whatever produces the tokens.

    ``pos`` is re-applied at every layer and reaches queries and keys only, never values. Pass
    None if positions were instead added to the tokens before the stack; the two styles are
    alternatives, not additive.

    The trailing LayerNorm is not decoration: the blocks are pre-norm
    (``x = x + attn(norm(x))``), so without it the stack returns an un-normalized residual
    stream.
    """

    def __init__(
        self,
        d_model: int,
        depth: int,
        *,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        activation: torch.nn.Module | None = None,
        causal: bool = False,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        assert depth > 0, f"depth must be positive, got {depth}"

        self._d_model = d_model
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    output_ratio=mlp_ratio,
                    embedding_size=None,
                    attention_heads=n_heads,
                    attention_dropout=attention_dropout,
                    ctx_cross_attention=False,
                    activation=activation,
                    causal=causal,
                )
                for _ in range(depth)
            ]
        )
        self.norm = torch.nn.LayerNorm(d_model) if final_norm else torch.nn.Identity()
        _xavier_init(self)

    def forward(
        self,
        x: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dim() == 3, f"expected (B, L, {self._d_model}), got {x.dim()}D"
        for block in self.blocks:
            x = block(x, mask=mask, pos=pos)
        return self.norm(x)


class DetrDecoder(torch.nn.Module):
    """A DETR-style decoder: learned queries cross-attending to an encoder memory.

    The target stream starts at **zeros** and ``query_embed`` is re-injected as positions at
    every layer - into the self-attention queries and keys, and the cross-attention queries. So
    the queries are a steering signal that survives the whole stack rather than initial content
    layer one overwrites, which is what makes them mean anything by the last layer.

    The memory carries its own ``memory_pos`` into the cross-attention keys only, so the two
    streams never share a positional table and values stay positionally clean throughout.
    """

    def __init__(
        self,
        d_model: int,
        depth: int,
        *,
        memory_dim: int | None = None,
        n_heads: int = 8,
        cross_attention_heads: int | None = None,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        activation: torch.nn.Module | None = None,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        assert depth > 0, f"depth must be positive, got {depth}"

        self._d_model = d_model
        self._memory_dim = memory_dim if memory_dim is not None else d_model
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    output_ratio=mlp_ratio,
                    embedding_size=None,
                    attention_heads=n_heads,
                    cross_attention_heads=cross_attention_heads,
                    attention_dropout=attention_dropout,
                    ctx_cross_attention=True,
                    ctx_dim=self._memory_dim,
                    activation=activation,
                )
                for _ in range(depth)
            ]
        )
        self.norm = torch.nn.LayerNorm(d_model) if final_norm else torch.nn.Identity()
        _xavier_init(self)

    def forward(
        self,
        memory: torch.Tensor,
        query_embed: torch.Tensor,
        *,
        memory_pos: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        assert memory.dim() == 3, f"memory must be (B, L, D), got {memory.dim()}D"
        assert query_embed.dim() in (2, 3), (
            f"query_embed must be (N, D) or (B, N, D), got {query_embed.dim()}D"
        )
        assert memory.shape[-1] == self._memory_dim, (
            f"memory width {memory.shape[-1]} != memory_dim {self._memory_dim}"
        )
        assert memory_pos is None or memory_pos.shape[-1] == self._memory_dim, (
            f"memory_pos width {memory_pos.shape[-1]} != memory_dim {self._memory_dim}"
        )

        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(0).expand(memory.shape[0], -1, -1)
        assert query_embed.shape[-1] == self._d_model, (
            f"query_embed width {query_embed.shape[-1]} != d_model {self._d_model}"
        )

        target = torch.zeros_like(query_embed)
        for block in self.blocks:
            target = block(
                target,
                ctx=memory,
                ctx_mask=memory_mask,
                pos=query_embed,
                ctx_pos=memory_pos,
            )
        return self.norm(target)


class TransformerDecoder(torch.nn.Module):
    """The classic encoder-decoder decoder: a target sequence attending to itself and to memory."""

    def __init__(
        self,
        d_model: int,
        depth: int,
        *,
        memory_dim: int | None = None,
        n_heads: int = 8,
        cross_attention_heads: int | None = None,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        activation: torch.nn.Module | None = None,
        causal: bool = True,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        assert depth > 0, f"depth must be positive, got {depth}"

        self._d_model = d_model
        self._memory_dim = memory_dim if memory_dim is not None else d_model
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    output_ratio=mlp_ratio,
                    embedding_size=None,
                    attention_heads=n_heads,
                    cross_attention_heads=cross_attention_heads,
                    attention_dropout=attention_dropout,
                    ctx_cross_attention=True,
                    ctx_dim=self._memory_dim,
                    activation=activation,
                    causal=causal,
                )
                for _ in range(depth)
            ]
        )
        self.norm = torch.nn.LayerNorm(d_model) if final_norm else torch.nn.Identity()
        _xavier_init(self)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        assert tgt.dim() == 3, f"tgt must be (B, L, D), got {tgt.dim()}D"
        assert memory.dim() == 3, f"memory must be (B, L, D), got {memory.dim()}D"
        assert tgt.shape[-1] == self._d_model, (
            f"tgt width {tgt.shape[-1]} != d_model {self._d_model}"
        )
        assert memory.shape[-1] == self._memory_dim, (
            f"memory width {memory.shape[-1]} != memory_dim {self._memory_dim}"
        )
        assert memory_pos is None or memory_pos.shape[-1] == self._memory_dim, (
            f"memory_pos width {memory_pos.shape[-1]} != memory_dim {self._memory_dim}"
        )

        for block in self.blocks:
            tgt = block(
                tgt,
                ctx=memory,
                mask=tgt_mask,
                ctx_mask=memory_mask,
                pos=tgt_pos,
                ctx_pos=memory_pos,
            )
        return self.norm(tgt)


class Transformer1d(_BaseTransformer):
    patch_embedder = PatchEmbeddings1d
    N = 1

    def __init__(
        self,
        input_dims: IntParams,
        hidden_dims: int = 1152,
        mlp_ratio: float = 4.0,
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
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            mlp_ratio=mlp_ratio,
            patch_size=1,
            depth=depth,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            output_dims=output_dims,
            time_dependent=False,
            time_embedding_size=None,
            time_scaling_coeff=1.0,
            embedding_size=None,
            learn_input_embeddings=learn_input_embeddings,
            learn_time_embeddings=False,
            learn_ctx_embeddings=False,
            adaLN_zero=False,
            n_classes=None,
            class_drop_prob=0.0,
            ctx_dim=ctx_dim,
            ctx_patch_size=ctx_patch_size,
            ctx_concat=False,
            ctx_cross_attention=ctx_cross_attention,
            cross_attention_heads=cross_attention_heads,
            ctx_in_context=ctx_in_context,
            activation=activation,
        )
