import copy
import math
import warnings
from functools import partial
from typing import Literal

import torch

from ...._typedefs import *
from ...modules.attention import (
    CrossAttention,
    SpatialSelfAttention,
)
from ...modules.conv_blocks import Conv1dBlock, Conv2dBlock, Conv3dBlock
from ...modules.embeddings import (
    FourierEmbeddings,
)
from ...modules.mlp_block import MLPBlock
from ...modules.residual_blocks import *
from ...utils import utils
from ...wrapper import Residual, XDependent, XDependentSequential
from .blocks import *


class CrossAttentionCondition(XDependent):
    """A feature map attending over a (B, L, ctx_dim) condition sequence.

    The feature map is the only spatial side, so it is normalized channels-first and flattened
    here; the condition keeps the token layout its encoder produced.
    """

    def __init__(self, input_dims: int, attention: torch.nn.Module, n_groups: int, ctx_dim: int):
        super().__init__()
        self._input_dims = input_dims
        self.q_norm = torch.nn.GroupNorm(num_groups=n_groups, num_channels=input_dims)
        self.kv_norm = torch.nn.LayerNorm(ctx_dim)
        self.cross_attention = attention(input_dims)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if ctx is None:
            raise ValueError("CrossAttentionCondition expects a valid context tensor `ctx`.")
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(1)  # a flat condition is one token
        batch_size, _, *spatial = x.shape
        q = self.q_norm(x).flatten(2).transpose(1, 2)
        attended = self.cross_attention(q=q, kv=self.kv_norm(ctx))
        return x + attended.transpose(1, 2).reshape(batch_size, self._input_dims, *spatial)


# TOOD: Fix input size handling
# The model does not handle odd input size; only 2^n
class UNetNd(torch.nn.Module):
    conv: torch.nn.Module
    conv_block: torch.nn.Module
    pool: torch.nn.Module
    norm_type: torch.nn.Module
    residual_block_creator: torch.nn.Module
    downsample: torch.nn.Module
    upsample: torch.nn.Module
    attention_block = SpatialSelfAttention
    cross_attention_block = CrossAttention
    dims: int

    def __init__(
        self,
        input_dims: IntParams,
        channels: IntParams = [32, 64],
        n_residual_blocks: int = 2,
        output_channels: int | None = None,
        n_middle_blocks: int = 2,
        dropout: float = 0.0,
        n_groups: int = 32,
        resample_with_resblock: bool = False,
        conv_resample: bool = True,
        deconv_upsample: bool = False,
        zero_conv: bool = True,
        attention: bool = False,
        attention_channel_idx: IntParams = [],
        attention_heads: int = 1,
        attention_dropout: float = 0.0,
        fusion_type: Literal["film", "adagn", "additive", "multiplicative"] = "film",
        time_dependent: bool = True,
        time_embedding_size: int | None = None,
        time_scaling_coeff: float = 1.0,
        embedding_size: int | None = None,
        embedder: torch.nn.Module = FourierEmbeddings,
        learn_embeddings: bool = False,
        n_classes: int | None = None,
        class_drop_prob: float = 0.0,
        ctx_dim: int | None = None,
        ctx_spatial_concat: bool = False,
        ctx_cross_attention: bool = False,
        cross_attention_channel_idx: IntParams = [],
        ctx_concat: bool = False,
    ):
        super().__init__()
        self._input_dims = input_dims
        self._channels = channels
        self._dropout = dropout
        self._output_channels = (
            output_channels if output_channels is not None else self._input_dims[0]
        )

        self._fusion_type = fusion_type
        self._time_dependent = time_dependent
        self._time_scaling_coeff = time_scaling_coeff
        self._resample_with_resblock = resample_with_resblock
        self._n_residual_blocks = n_residual_blocks
        self._n_middle_blocks = n_middle_blocks
        self._n_classes = n_classes
        self._class_drop_prob = class_drop_prob

        self._ctx_dim = ctx_dim
        self._ctx_spatial_concat = ctx_spatial_concat
        self._ctx_cross_attention = ctx_cross_attention
        self._ctx_concat = ctx_concat

        spatial_match = lambda s1, s2: tuple(s1[1:]) == tuple(s2[1:])  # Check spatial dims
        full_match = lambda s1, s2: tuple(s1) == tuple(s2)  # Check size

        if self._ctx_spatial_concat:
            if ctx_dim is None:
                raise ValueError("ctx_dim must be provided when ctx_spatial_concat=True")
            self._input_dims[0] += ctx_dim

        self._setup_time_dependency(
            time_embedding_size=(
                time_embedding_size if time_embedding_size is not None else channels[0]
            ),
            embedding_size=embedding_size,
            learn_embeddings=learn_embeddings,
            embedder=embedder,
        )
        self._setup_class_condition()

        if attention:
            _attention_module, attention_channel_idx = self._setup_attention_module(
                is_cross=False,
                attention_heads=attention_heads,
                attention_channel_idx=attention_channel_idx,
                attention_dropout=attention_dropout,
                n_groups=n_groups,
                zero_conv=zero_conv,
            )

        if self._ctx_cross_attention:
            _cross_attention_module, cross_attention_channel_idx = self._setup_attention_module(
                is_cross=True,
                attention_heads=attention_heads,
                attention_channel_idx=attention_channel_idx,
                attention_dropout=attention_dropout,
                n_groups=n_groups,
                zero_conv=zero_conv,
            )

        if self._ctx_concat and self._time_dependent:
            if ctx_dim is None:
                raise ValueError("ctx_dim must be provided when ctx_concat=True")
            if self._ctx_cross_attention:
                raise ValueError(
                    "ctx_concat appends a flat (B, ctx_dim) condition to the time embedding, while "
                    "ctx_cross_attention reads a (B, L, ctx_dim) sequence; pick one."
                )
            self._embedding_size += ctx_dim

        self.input_layer = self.conv(self._input_dims[0], self._channels[0], 3, padding=1)
        _downsample_input_dims = utils.calculate_output_size(
            self.input_layer, input_dims=self._input_dims
        )
        _downsampling_channels = [self._channels[0]]
        self._resampled = []

        self.downsample_blocks = torch.nn.ModuleList([])
        for i, ch in enumerate(self._channels):
            for _ in range(self._n_residual_blocks):
                _subblock = XDependentSequential()
                _subblock.append(
                    self.residual_block_creator(
                        input_dims=_downsample_input_dims,
                        embedding_size=self._embedding_size,
                        dropout=self._dropout,
                        n_groups=n_groups,
                        output_channels=ch,
                        zero_out=zero_conv,
                        fusion_type=self._fusion_type,
                    )
                )
                _downsample_input_dims = utils.calculate_output_size(
                    _subblock[-1], input_dims=_downsample_input_dims
                )

                if i in attention_channel_idx:
                    _subblock.append(Residual(_attention_module(_downsample_input_dims[0])))

                if self._ctx_cross_attention and i in cross_attention_channel_idx:
                    _subblock.append(
                        CrossAttentionCondition(
                            _downsample_input_dims[0], _cross_attention_module, n_groups, self._ctx_dim
                        )
                    )

                self.downsample_blocks.append(_subblock)
                _downsampling_channels.append(int(_downsample_input_dims[0]))
                self._resampled.append(False)
            if (i + 1) != len(self._channels):
                self.downsample_blocks.append(
                    XDependentSequential(
                        self.residual_block_creator(
                            input_dims=_downsample_input_dims,
                            embedding_size=self._embedding_size,
                            dropout=self._dropout,
                            n_groups=n_groups,
                            output_channels=ch,
                            fusion_type=self._fusion_type,
                            zero_out=zero_conv,
                            downsample=True,
                        )
                        if self._resample_with_resblock
                        else self.downsample(
                            _downsample_input_dims, use_conv=conv_resample, output_channels=ch
                        )
                    )
                )
                self._resampled.append(True)
                _downsample_input_dims = utils.calculate_output_size(
                    self.downsample_blocks[-1][-1], input_dims=_downsample_input_dims
                )
                _downsampling_channels.append(int(_downsample_input_dims[0]))

        self.middle_blocks = torch.nn.ModuleList()
        _middle_input_dims = copy.deepcopy(_downsample_input_dims)
        for i in range(self._n_middle_blocks):
            _subblock = XDependentSequential()
            _subblock.append(
                self.residual_block_creator(
                    input_dims=_middle_input_dims,
                    embedding_size=self._embedding_size,
                    dropout=self._dropout,
                    n_groups=n_groups,
                    output_channels=_middle_input_dims[0],
                    zero_out=zero_conv,
                    fusion_type=self._fusion_type,
                )
            )
            if (i + 1) != self._n_middle_blocks:
                if attention:
                    _subblock.append(Residual(_attention_module(_middle_input_dims[0])))

                if self._ctx_cross_attention and i in cross_attention_channel_idx:
                    _subblock.append(
                        CrossAttentionCondition(
                            _middle_input_dims[0], _cross_attention_module, n_groups, self._ctx_dim
                        )
                    )
            self.middle_blocks.append(_subblock)

        # _mid_dims = copy.deepcopy(_middle_input_dims)
        # self._bottleneck_res = (
        #     _mid_dims.cpu().numpy().tolist() if hasattr(_mid_dims, "numpy") else list(_mid_dims)
        # )
        self.upsample_blocks = torch.nn.ModuleList([])
        _upsample_input_dims = copy.deepcopy(_middle_input_dims)
        for i, ch in reversed(list(enumerate(self._channels))):
            for j in range(self._n_residual_blocks + 1):
                _subblock = []
                _skip_connection_input_dims = copy.deepcopy(_upsample_input_dims)
                _skip_connection_input_dims[0] = (
                    _skip_connection_input_dims[0] + _downsampling_channels.pop()
                )
                _subblock.append(
                    self.residual_block_creator(
                        input_dims=_skip_connection_input_dims,
                        embedding_size=self._embedding_size,
                        dropout=self._dropout,
                        n_groups=n_groups,
                        output_channels=ch,
                        zero_out=zero_conv,
                        fusion_type=self._fusion_type,
                    )
                )
                _upsample_input_dims = utils.calculate_output_size(
                    _subblock[-1], input_dims=_skip_connection_input_dims
                )

                if i in attention_channel_idx:
                    _subblock.append(Residual(_attention_module(_upsample_input_dims[0])))

                if self._ctx_cross_attention and i in cross_attention_channel_idx:
                    _subblock.append(
                        CrossAttentionCondition(
                            _upsample_input_dims[0], _cross_attention_module, n_groups, self._ctx_dim
                        )
                    )

                if i and j == self._n_residual_blocks:
                    _subblock.append(
                        self.residual_block_creator(
                            input_dims=_upsample_input_dims,
                            embedding_size=self._embedding_size,
                            dropout=self._dropout,
                            n_groups=n_groups,
                            output_channels=ch,
                            fusion_type=self._fusion_type,
                            upsample=True,
                            zero_out=zero_conv,
                        )
                        if self._resample_with_resblock
                        else self.upsample(
                            _upsample_input_dims,
                            use_deconv=deconv_upsample,
                            refine=conv_resample,
                            output_channels=ch,
                        )
                    )
                    _upsample_input_dims = utils.calculate_output_size(
                        _subblock[-1], input_dims=_upsample_input_dims
                    )
                self.upsample_blocks.append(XDependentSequential(*_subblock))

        if self._ctx_spatial_concat:
            self.condition_residual_block = self.residual_block_creator(
                input_dims=[_upsample_input_dims[0] * 2, *_upsample_input_dims[1:]],
                embedding_size=self._embedding_size,
                dropout=self._dropout,
                n_groups=n_groups,
                output_channels=_upsample_input_dims[0],
                fusion_type=self._fusion_type,
                zero_out=zero_conv,
            )

        self.output_block = torch.nn.Sequential(
            torch.nn.GroupNorm(n_groups, _upsample_input_dims[0]),
            torch.nn.SiLU(),
            utils.zero_module(
                self.conv(_upsample_input_dims[0], self._output_channels, 3, padding=1)
            ),
        )

    def _setup_time_dependency(
        self,
        *,
        time_embedding_size: int,
        embedding_size: int | None,
        learn_embeddings: bool,
        embedder: object,
    ):
        # Create time embedding modules.
        if self._time_dependent:
            self._time_embedding_size = time_embedding_size
            self._embedding_size = (
                embedding_size if embedding_size is not None else self._time_embedding_size * 4
            )

            self.time_embedder = embedder(
                d_model=self._time_embedding_size, learnable=learn_embeddings
            )

            self.embedding_projection = MLPBlock(
                input_dims=self._time_embedding_size,
                output_dims=self._embedding_size,
                hidden_dims=[self._embedding_size],
                activation_function=[torch.nn.SiLU()],
                output_activation=True,
            )
        else:
            self._time_embedding_size = None
            self._embedding_size = None

    def _setup_class_condition(
        self,
    ):
        if self._n_classes is not None:
            self.class_embedder = torch.nn.Embedding(
                self._n_classes if self._class_drop_prob == 0.0 else self._n_classes + 1,
                self._embedding_size,
            )
            self._null_class_idx = self._n_classes

    def _setup_attention_module(
        self,
        *,
        is_cross: bool,
        attention_heads: int,
        attention_channel_idx: list[int],
        attention_dropout: float,
        n_groups: int,
        zero_conv: bool,
    ) -> tuple[object, list[int]]:
        if len(attention_channel_idx) == 0:
            attention_channel_idx = list(range(len(self._channels)))
        _attention_channel_idx = attention_channel_idx
        kwargs = dict(
            n_heads=attention_heads,
            dropout_p=attention_dropout,
            prenorm=partial(torch.nn.GroupNorm, num_groups=n_groups),
            bias=True,
            zero_out=zero_conv,
        )
        if is_cross:
            # CrossAttentionCondition normalizes both streams itself, in each one's own layout
            kwargs["kv_input_dim"] = self._ctx_dim
            kwargs["prenorm"] = None

        _attention_module = partial(
            self.attention_block if not is_cross else self.cross_attention_block, **kwargs
        )
        return _attention_module, _attention_channel_idx

    def _prepare_time(self, t: torch.Tensor | None = None) -> torch.Tensor:
        emb = None
        if self._time_dependent:
            if t is None:
                raise ValueError(
                    "Model is configured as `time_dependent=True`, but time step `t` is None."
                )
            emb = self.time_embedder(t)
            emb = self.embedding_projection(emb)
        elif t is not None:
            raise ValueError(
                "Model is configured as `time_dependent=False`, but time step `t` was provided."
            )
        return emb

    def _prepare_labels(
        self,
        emb: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self._n_classes is not None:
            if labels is None:
                # Classifier-Free Guidance (CFG): Unconditional generation
                # If training with dropout > 0, or inference with no label, we use the NULL token.
                if self._class_drop_prob > 0:
                    labels = torch.full(
                        (emb.shape[0],), self._null_class_idx, device=device, dtype=torch.long
                    )
                else:
                    raise ValueError(
                        f"Model requires {self._n_classes} class labels, but `ctx` is None."
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
        if ctx is None:
            return x, emb

        if ctx.ndim == x.ndim and ctx.shape[2:] == x.shape[2:]:
            x = torch.cat([x, ctx], dim=1)

        elif ctx.ndim == 2 and emb is not None:
            if self._ctx_concat:  # Vector concatenation path
                emb = torch.cat([emb, ctx], dim=-1)
            else:  # Standard vector addition fallback
                emb = emb + ctx
        return x, emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is not None:
            t = t * self._time_scaling_coeff
            t = t.flatten(1)

        emb = self._prepare_time(t=t)
        emb = self._prepare_labels(emb=emb, labels=labels, device=x.device)
        x, emb = self._prepare_conditioning(x, emb, ctx=ctx)
        return self._forward(x, emb, ctx)

    def _prepare_conditioning(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if ctx is None:
            return x, emb

        if self._ctx_spatial_concat and ctx.ndim == x.ndim and ctx.shape[2:] == x.shape[2:]:
            x = torch.cat([x, ctx], dim=1)

        # Handle vector concatenation to time embeddings safely
        elif ctx.ndim == 2 and emb is not None:
            if self._ctx_concat:
                emb = torch.cat([emb, ctx], dim=-1)
            else:
                # Avoid implicit shape broad-casting bugs if dims don't match
                if emb.shape == ctx.shape:
                    emb = emb + ctx
        return x, emb

    def _forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        h = self.input_layer(x)
        if self._ctx_spatial_concat:
            _residual = h.clone()

        _skip_connection = [h]
        for i, down in enumerate(self.downsample_blocks):
            h = down(h, emb, ctx)
            _skip_connection.append(h)

        for mb in self.middle_blocks:
            h = mb(h, emb, ctx)

        for up in self.upsample_blocks:
            _skip = _skip_connection.pop()
            h = torch.cat([h, _skip], dim=1)
            h = up(h, emb, ctx)

        if self._ctx_spatial_concat and hasattr(self, "condition_residual_block"):
            h = torch.cat((h, _residual), dim=1)
            h = self.condition_residual_block(h, emb)

        return self.output_block(h)

    def train(self, mode=True):
        super().train(mode)
        return self


class UNet1d(UNetNd):
    conv = torch.nn.Conv1d
    conv_block = Conv1dBlock
    pool = torch.nn.MaxPool1d
    norm_type = torch.nn.BatchNorm1d
    residual_block_creator = ResBlock1d
    downsample = Downsample1d
    upsample = Upsample1d
    dims = 1


class UNet2d(UNetNd):
    conv = torch.nn.Conv2d
    conv_block = Conv2dBlock
    pool = torch.nn.MaxPool2d
    norm_type = torch.nn.BatchNorm2d
    residual_block_creator = ResBlock2d
    downsample = Downsample2d
    upsample = Upsample2d
    dims = 2


class UNet3d(UNetNd):
    conv = torch.nn.Conv3d
    conv_block = Conv3dBlock
    pool = torch.nn.MaxPool3d
    norm_type = torch.nn.BatchNorm3d
    residual_block_creator = ResBlock3d
    downsample = Downsample3d
    upsample = Upsample3d
    dims = 3
