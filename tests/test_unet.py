import pytest
import torch

import butane


# Helper function to generate inputs and dimensions systematically
def get_test_inputs(dims: int, batch_size: int = 2):
    if dims == 1:
        input_dims = [8, 16]
        x = torch.randn(batch_size, 8, 16)
        spatial_ctx = torch.randn(batch_size, 4, 16)  # 4 context channels, matching length 16
    elif dims == 2:
        input_dims = [3, 16, 16]
        x = torch.randn(batch_size, 3, 16, 16)
        spatial_ctx = torch.randn(batch_size, 2, 16, 16)  # 2 context channels, matching H, W
    elif dims == 3:
        input_dims = [1, 8, 8, 8]
        x = torch.randn(batch_size, 1, 8, 8, 8)
        spatial_ctx = torch.randn(batch_size, 2, 8, 8, 8)  # 2 context channels, matching D, H, W
    else:
        raise ValueError(f"Invalid dimension rank: {dims}")

    t = torch.randn(batch_size, 1)
    return input_dims, x, t, spatial_ctx


def test_unet1d_unconditional():
    model = butane.nn.unet.UNet1d(
        input_dims=[8, 16],
        time_dependent=False,
    )
    x = torch.randn(2, 8, 16)
    out = model(x)
    assert out.shape == x.shape


def test_unet1d_time_conditioned():
    model = butane.nn.unet.UNet1d(
        input_dims=[8, 16],
        time_dependent=True,
    )
    x = torch.randn(2, 8, 16)
    t = torch.randn(2, 1)
    out = model(x, t)
    assert out.shape == x.shape


def test_unet1d_class_conditioned():
    model = butane.nn.unet.UNet1d(
        input_dims=[8, 16],
        n_classes=10,
    )
    x = torch.randn(2, 8, 16)
    t = torch.randn(2, 1)
    y = torch.randint(0, 10, (2,))
    # pass discrete conditioning down labels parameter channel
    out = model(x, t, labels=y)
    assert out.shape == x.shape


def test_unet1d_class_cfg():
    model = butane.nn.unet.UNet1d(
        input_dims=[8, 16],
        n_classes=10,
        class_drop_prob=0.1,
    )
    x = torch.randn(2, 8, 16)
    t = torch.randn(2, 1)

    # Test pass with active labels
    y = torch.randint(0, 10, (2,))
    out = model(x, t, labels=y)
    assert out.shape == x.shape

    # Test pass without labels (forces substitution via structural null class token)
    out_uncond = model(x, t, labels=None)
    assert out_uncond.shape == x.shape


def test_unet2d_basic():
    model = butane.nn.unet.UNet2d(
        input_dims=[3, 32, 32],
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.randn(2, 1)
    out = model(x, t)
    assert out.shape == x.shape


def test_unet3d_basic():
    model = butane.nn.unet.UNet3d(
        input_dims=[1, 8, 16, 16],
        channels=[32, 64],
    )
    x = torch.randn(2, 1, 8, 16, 16)
    t = torch.randn(2, 1)
    out = model(x, t)
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "unet_cls, dims",
    [
        (butane.nn.unet.UNet1d, 1),
        (butane.nn.unet.UNet2d, 2),
        (butane.nn.unet.UNet3d, 3),
    ],
)
class TestUNetConditioningVariations:
    @pytest.mark.parametrize("fusion_type", ["film", "adagn", "additive", "multiplicative"])
    def test_flat_vector_context_addition_or_film(self, unet_cls, dims, fusion_type):
        input_dims, x, t, _ = get_test_inputs(dims)
        model = unet_cls(
            input_dims=input_dims,
            time_dependent=True,
            fusion_type=fusion_type,
        )
        flat_ctx = torch.randn(x.shape[0], model._embedding_size)

        out = model(x, t, ctx=flat_ctx)
        assert out.shape == x.shape

    def test_flat_vector_context_concatenation(self, unet_cls, dims):
        input_dims, x, t, _ = get_test_inputs(dims)
        ctx_dim = 16

        model = unet_cls(
            input_dims=input_dims,
            time_dependent=True,
            ctx_concat=True,
            ctx_dim=ctx_dim,
        )
        flat_ctx = torch.randn(x.shape[0], ctx_dim)

        out = model(x, t, ctx=flat_ctx)
        assert out.shape == x.shape

    def test_spatial_concatenation(self, unet_cls, dims):
        input_dims, x, t, spatial_ctx = get_test_inputs(dims)

        model = unet_cls(
            input_dims=input_dims,
            time_dependent=True,
            ctx_spatial_concat=True,
            ctx_dim=spatial_ctx.shape[1],
        )

        out = model(x, t, ctx=spatial_ctx)
        assert out.shape == x.shape

    @pytest.mark.parametrize("ctx_shape", ["flat", "sequence"])
    def test_cross_attention_conditioning(self, unet_cls, dims, ctx_shape):
        input_dims, x, t, _ = get_test_inputs(dims)
        ctx_dim = 32

        model = unet_cls(
            input_dims=input_dims,
            time_dependent=True,
            ctx_cross_attention=True,
            cross_attention_channel_idx=[0, 1],
            attention_heads=2,
            ctx_dim=ctx_dim,
            ctx_concat=False,
        )

        if ctx_shape == "flat":
            ctx = torch.randn(x.shape[0], ctx_dim)
        else:
            ctx = torch.randn(x.shape[0], 5, ctx_dim)  # (B, L, ctx_dim) tokens

        out = model(x, t, ctx=ctx)
        assert out.shape == x.shape


class TestClassifierFreeGuidance:
    @pytest.mark.parametrize(
        "unet_cls, dims",
        [
            (butane.nn.unet.UNet1d, 1),
            (butane.nn.unet.UNet2d, 2),
        ],
    )
    def test_class_conditioning_and_cfg_dropout(self, unet_cls, dims):
        input_dims, x, t, _ = get_test_inputs(dims)
        model = unet_cls(
            input_dims=input_dims,
            n_classes=10,
            class_drop_prob=0.2,
        )
        y = torch.randint(0, 10, (x.shape[0],))

        model.train()
        out_train = model(x, t, labels=y)
        assert out_train.shape == x.shape

        model.eval()
        out_eval = model(x, t, labels=y)
        assert out_eval.shape == x.shape
