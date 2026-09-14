import copy
import math
import warnings
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from ...._typedefs import *
from ...modules import fusions
from ...modules.attention import (
    SpatialCrossAttention,
    SpatialSelfAttention,
)
from ...modules.conv_blocks import Conv1dBlock, Conv2dBlock, Conv3dBlock
from ...modules.embeddings import (
    FourierEmbeddings,
)
from ...modules.mlp_block import MLPBlock
from ...modules.residual_blocks import *
from ...utils import utils
from ...wrapper import XDependent, XDependentSequential


class DownsampleNd(torch.nn.Module):
    conv: torch.nn.Module
    pool: torch.nn.Module
    dims: int

    def __init__(
        self, input_dims: IntParams, output_channels: int | None = None, use_conv: bool = False
    ) -> None:
        super().__init__()
        if output_channels is None:
            output_channels = input_dims[0]

        stride = 2 if self.dims != 3 else (1, 2, 2)
        if use_conv:
            self.downsample_layer = self.conv(
                input_dims[0], output_channels, 3, stride=stride, padding=1
            )
        else:
            self.downsample_layer = self.pool(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample_layer(x)


class Downsample1d(DownsampleNd):
    conv = torch.nn.Conv1d
    pool = torch.nn.AvgPool1d
    dims = 1


class Downsample2d(DownsampleNd):
    conv = torch.nn.Conv2d
    pool = torch.nn.AvgPool2d
    dims = 2


class Downsample3d(DownsampleNd):
    conv = torch.nn.Conv3d
    pool = torch.nn.AvgPool3d
    dims = 3


class UpsampleNd(torch.nn.Module):
    conv: torch.nn.Module
    dims: int

    def __init__(
        self,
        input_dims: IntParams,
        output_channels: int | None = None,
        use_deconv: bool = False,
        refine: bool = False,
    ) -> None:
        super().__init__()
        if output_channels is None:
            output_channels = input_dims[0]

        self._use_deconv = use_deconv
        if self._use_deconv:
            _kernel_size = 4 if self.dims != 3 else (3, 4, 4)
            _stride = 4 if self.dims != 3 else (1, 2, 2)

            # Using kernel_size = stride ensures exact doubling without overlapping artifacts
            self.upsample_layer = self.conv_transpose(
                in_channels=input_dims[0],
                out_channels=output_channels,
                kernel_size=_stride,
                stride=_stride,
                padding=1,
            )
        if self.dims < 3:
            self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode="nearest")

        if refine:
            _refine_in_channels = output_channels if self._use_deconv else input_dims[0]
            self.refinement_layer = self.conv(
                _refine_in_channels, output_channels, kernel_size=3, padding=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_deconv:
            h = self.upsample_layer(x)
        else:
            if self.dims == 3:
                h = torch.nn.functional.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                h = self.upsample_layer(x)

        if hasattr(self, "refinement_layer"):
            h = self.refinement_layer(h)

        return h


class Upsample1d(UpsampleNd):
    conv = torch.nn.Conv1d
    conv_transpose = torch.nn.ConvTranspose1d
    dims = 1


class Upsample2d(UpsampleNd):
    conv = torch.nn.Conv2d
    conv_transpose = torch.nn.ConvTranspose2d
    dims = 2


class Upsample3d(UpsampleNd):
    conv = torch.nn.Conv3d
    conv_transpose = torch.nn.ConvTranspose3d
    dims = 3


class ResBlockNd(XDependent):
    conv: torch.nn.Module
    upsample_block: torch.nn.Module
    downsample_block: torch.nn.Module

    def __init__(
        self,
        input_dims: IntParams,
        embedding_size: int | None,
        output_channels: int,
        upsample: bool = False,
        downsample: bool = False,
        n_groups: int = 32,
        dropout: float = 0.0,
        fusion_type: str = "film",
        use_shortcut_conv: bool = False,
        zero_out: bool = True,
    ):
        super().__init__()
        self._fusion_type = fusion_type

        self.input_layer = torch.nn.Sequential(
            torch.nn.GroupNorm(n_groups, input_dims[0]),
            torch.nn.SiLU(),
        )
        self.block_1 = torch.nn.Sequential(self.conv(input_dims[0], output_channels, 3, padding=1))

        # Setup Normalization & Embedding Fusion blocks
        if embedding_size is not None:
            # Fetch from our registry
            fuser_cls = fusions.fusion_registry[self._fusion_type]

            # If adagn, it controls its own internal norm layer structure
            if self._fusion_type == "adagn":
                self.embedding_fuser = fuser_cls(
                    embedding_dims=embedding_size, feature_dims=output_channels, n_groups=n_groups
                )
                self.pre_block_2_norm = torch.nn.Identity()  # Bypassed because AdaGN does it
            else:
                self.embedding_fuser = fuser_cls(
                    embedding_dims=embedding_size, feature_dims=output_channels
                )
                self.pre_block_2_norm = torch.nn.GroupNorm(n_groups, output_channels)
        else:
            self.embedding_fuser = None
            self.pre_block_2_norm = torch.nn.GroupNorm(n_groups, output_channels)

        if upsample:
            self.up_down_x = self.upsample_block(input_dims, refine=False)
            self.up_down_h = self.upsample_block(input_dims, refine=False)
        elif downsample:
            self.up_down_x = self.downsample_block(input_dims, use_conv=False)
            self.up_down_h = self.downsample_block(input_dims, use_conv=False)
        else:
            self.up_down_x = torch.nn.Identity()
            self.up_down_h = torch.nn.Identity()

        # Block 2 only handles activation & regularization transformations now
        self.block_2 = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Dropout(p=dropout))

        self.output_layer = (
            utils.zero_module(self.conv(output_channels, output_channels, kernel_size=3, padding=1))
            if zero_out
            else self.conv(output_channels, output_channels, kernel_size=3, padding=1)
        )

        if output_channels == input_dims[0]:
            self.shortcut = torch.nn.Identity()
        elif use_shortcut_conv:
            self.shortcut = self.conv(input_dims[0], output_channels, 3, padding=1)
        else:
            self.shortcut = self.conv(input_dims[0], output_channels, 1)


    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor | None = None,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_x = self.input_layer(x)
        h_x = self.up_down_h(h_x)
        x = self.up_down_x(x)
        h_x = self.block_1(h_x)

        if self._fusion_type == "adagn":
            # AdaGN: normalizes internally AND fuses conditioning info at once
            if e is not None:
                h_x = self.embedding_fuser(h_x, e)
            else:
                # Fallback path if embedding layer execution is bypassed or omitted
                # We extract the inner gn layer module manually to avoid crashes
                h_x = self.embedding_fuser.gn(h_x)
        else:
            # Classic Post-Norm Path (FiLM / Additive)
            h_x = self.pre_block_2_norm(h_x)
            if e is not None and self.embedding_fuser is not None:
                h_x = self.embedding_fuser(h_x, e)

        h_x = self.output_layer(self.block_2(h_x))
        return self.shortcut(x) + h_x


class ResBlock1d(ResBlockNd):
    conv = torch.nn.Conv1d
    upsample_block = Upsample1d
    downsample_block = Downsample1d


class ResBlock2d(ResBlockNd):
    conv = torch.nn.Conv2d
    upsample_block = Upsample2d
    downsample_block = Downsample2d


class ResBlock3d(ResBlockNd):
    conv = torch.nn.Conv3d
    upsample_block = Upsample3d
    downsample_block = Downsample3d


class ConditionProjectionBlockNd(torch.nn.Module):
    conv: torch.nn.Module
    residual_block_creator: XDependent

    def __init__(
        self,
        input_dims: IntParams,
        embedding_size: int,
        channels: int,
        n_residual_blocks: int = 1,
        n_downsamples: Optional[int] = None,
        dropout: float = 0.0,
        n_groups: int = 32,
        attention: Optional[torch.nn.Module] = None,
        use_film: bool = True,
        zero_out: bool = False,
        linear_projection: bool = True,
    ) -> None:
        super().__init__()
        self._input_dims = input_dims
        self.blocks = torch.nn.ModuleList()

        _updated_input_dims = copy.deepcopy(self._input_dims)
        self.input_layer = self.conv(self._input_dims[0], channels, 3, padding=1)
        _updated_input_dims = utils.calculate_output_size(
            self.input_layer, input_dims=_updated_input_dims
        )
        if n_downsamples is None:
            n_downsamples = n_residual_blocks
        else:
            n_downsamples = min(n_downsamples, n_residual_blocks)

        _downsampling_counter = 0
        for i in range(n_residual_blocks):
            self.blocks.append(
                self.residual_block_creator(
                    input_dims=_updated_input_dims,
                    embedding_size=embedding_size,
                    dropout=dropout,
                    output_channels=channels,
                    use_film=use_film,
                    downsample=_downsampling_counter < n_downsamples,
                    zero_out=zero_out,
                )
            )
            _downsampling_counter += 1
            _updated_input_dims = utils.calculate_output_size(
                self.blocks[-1], input_dims=_updated_input_dims
            )

            if attention is not None and (i + 1) != n_residual_blocks:
                self.blocks.append(attention(_updated_input_dims[0]))

        self.pre_projection_block = torch.nn.Sequential(
            torch.nn.GroupNorm(n_groups, _updated_input_dims[0]),
            torch.nn.SiLU(),
            torch.nn.Flatten(1) if linear_projection else torch.nn.Identity(),
        )

        if linear_projection:
            self.linear_projection = MLPBlock(
                input_dims=_updated_input_dims.prod(),
                output_dims=embedding_size * 2 if use_film else embedding_size,
                hidden_dims=[embedding_size],
                activation_function=[torch.nn.SiLU()],
                output_activation=False,
                zero_out=zero_out,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for b in self.blocks:
            h = b(h)
        h = self.pre_projection_block(h)
        if hasattr(self, "linear_projection"):
            h = self.linear_projection(h)
        else:
            h = h.reshape(h.size(0), h.size(1), -1).transpose(-1, 1)
        return h


class ConditionProjectionBlock1d(ConditionProjectionBlockNd):
    conv = torch.nn.Conv1d
    residual_block_creator = ResBlock1d


class ConditionProjectionBlock2d(ConditionProjectionBlockNd):
    conv = torch.nn.Conv2d
    residual_block_creator = ResBlock2d


class ConditionProjectionBlock3d(ConditionProjectionBlockNd):
    conv = torch.nn.Conv3d
    residual_block_creator = ResBlock3d
