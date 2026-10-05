# ⛽ Butane

**Ready-made PyTorch building blocks for people who would rather not rewrite them.** MLPs, conv nets,
U-Nets, transformers, autoencoders, flow matching and diffusion, datasets, k-means and a few training
helpers, all configured with plain lists instead of hand-written `nn.Sequential` boilerplate.

> Butane started as a C++ (LibTorch) library for quick prototyping, with a direct translation to Python.
> The two have since diverged and development is Python only. The original C++ headers are still in
> [`butane/cpp`](butane/cpp) with their examples in [`examples/cpp`](examples/cpp), but they are not kept in
> sync. When I find the time I will translate everything to C++ code 🙂
```
import butane
```

Experiment logging is a separate package, [model-sitter](https://github.com/iamtsac/model-sitter)
(`import mosi`), used by the examples.

## Install

```
pixi install                  # core
pixi install -e examples      # + matplotlib and model-sitter (logging) for the examples
```

or `pip install -e .`. Torch comes from the CUDA 12.6 index under pixi (see `pyproject.toml`).

## Examples

### 1. A CNN classifier from two blocks

Every block takes lists with one entry per layer; a single entry is repeated for all layers.

```python
import torch, butane

cnn = butane.nn.Conv2dBlock(
    input_dims=[1, 28, 28],
    channels=[16, 32],                           # two conv layers
    activation_function=[torch.nn.GELU()],       # one entry: used for every layer
    conv_pad=[1],
    normalization=[True, False],                 # normalize after the first layer only
    normalization_type=torch.nn.BatchNorm2d,
)
head = butane.nn.MLPBlock(
    input_dims=cnn.output_size.prod().item(),    # blocks know their output size
    output_dims=10,
    hidden_dims=[64],
    dropout=[0.1],
)
classifier = torch.nn.Sequential(cnn, torch.nn.Flatten(1), head)
classifier(torch.randn(8, 1, 28, 28)).shape      # torch.Size([8, 10])
```

### 2. An autoencoder with the built-in trainer

```python
enc = torch.nn.Sequential(torch.nn.Flatten(1), butane.nn.MLPBlock(input_dims=784, output_dims=8, hidden_dims=[128]))
dec = torch.nn.Sequential(butane.nn.MLPBlock(input_dims=8, output_dims=784, hidden_dims=[128]),
                          butane.nn.Unflatten(1, torch.tensor([1, 28, 28])))
model = butane.nn.AE(enc, dec)                   # also VQVAE, MLVQVAE

ds = butane.data.Dataset(torch.rand(512, 1, 28, 28))
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
trainer = butane.nn.utils.ModelTrainer(model, dl, torch.optim.AdamW(model.parameters(), 1e-3))
trainer(3, model.loss_fn)                        # 3 epochs
```

### 3. Toy data and clustering

```python
x, y = butane.data.toy.make_moons(2000)          # also spiral, pinwheel, ring, eight gaussians, ...
kmeans = butane.clustering.MiniBatchKMeans(n_centroids=8, max_iters=50, random_state=0)
kmeans.fit(x)
kmeans.centroids.shape                           # torch.Size([8, 2])
```

### 4. Flow matching with a U-Net, logged with model-sitter

```python
import mosi

model = butane.nn.unet.UNet2d(input_dims=[1, 28, 28], channels=[32, 64], n_residual_blocks=1, output_channels=1,
                              attention=False, time_dependent=True, time_embedding_size=32)
fm = butane.nn.ConditionalFlowMatching(0.0)
fm.set_source_distribution(torch.distributions.Independent(
    torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28)), 3))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
ema = butane.nn.EMA(model=model, decay=0.999)
data = torch.randn(256, 1, 28, 28)               # stand-in for a real dataset

with mosi.Sitter(".tmp/readme_fm", overwrite=True, confirm=lambda msg: True) as log:
    for step in range(20):
        x1 = data[torch.randint(len(data), (32,))]
        x0 = fm.source_distribution().sample((x1.size(0),))
        t = fm.sample_timesteps(x1.size(0))
        x_t, u_t = fm(x0, x1, t)
        loss = torch.mean((model(x_t, t) - u_t) ** 2)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); ema.update()
        log.add_stats(loss=loss.item())
        if (step + 1) % 10 == 0:
            log.checkpoint(model=model, optimizer=optimizer, ema=ema)
        log.step()
```

This writes `stats.jsonl`, checkpoints and `env.json` (command line and git commit) to `.tmp/readme_fm/`.
Reload with `mosi.Sitter(".tmp/readme_fm", eval_mode=True).load_checkpoint("latest", model=model, ema=ema)`.

More in [`examples/python`](examples/python) (MNIST DDPM and inpainting, flow matching with U-Net and DiT,
class conditional and classifier-free guidance, OT, clustering, trajectory datasets).

## What exists

### `butane.nn`: layers and blocks
| | |
|---|---|
| MLP | `MLPBlock`, `ProbabilisticMLPBlock` |
| Convolution (1d / 2d / 3d) | `Conv{1,2,3}dBlock`, `ConvTranspose{1,2,3}dBlock`, `ConvUpsample{1,2,3}dBlock`, `ConvTransposeWRefinement{1,2,3}dBlock` |
| Residual | `Residual{1,2,3}dBlock`, `Residual` wrapper |
| Attention | `SelfAttention`, `CrossAttention`, `SpatialSelfAttention`, `SpatialCrossAttention` |
| Embeddings | `SinusoidalEmbeddings`, `FourierEmbeddings`, `LearnableEmbeddings`, `RelativePositionEmbeddings`, `PatchEmbeddings{1,2,3}d` |
| Conditioning | additive, multiplicative, `FiLMFusion` and `AdaGNFusion` fusions behind one registry |
| Quantizers | `Quantizer`, `Quantizer2d`, straight-through `STEQuantizer`, `STEQuantizer2d` |
| Activations | `Gaussian`, `Squashing`, `ScaledTanh` |
| Misc | `EMA`, `Ensemble`, `Unflatten`, `XDependent` / `XDependentSequential` (modules that take extra inputs) |

### `butane.nn`: architectures
| | |
|---|---|
| Autoencoders | `AE`, `VQVAE`, `MLVQVAE` |
| U-Net | `UNet1d`, `UNet2d`, `UNet3d` (time, class and cross-attention conditioning, FiLM) |
| Transformers | `ViT{1,2,3}d`, `DiT{1,2,3}d` (per-token modulation), `TransformerEncoder`, `TransformerDecoder`, `DetrDecoder`, `Transformer1d` |
| Other | `TimeMLP` (time-dependent MLP) |

### `butane.nn`: generative
`Diffusion` (DDPM style, cosine and other schedules), `FlowMatching`, `ConditionalFlowMatching`,
`TargetConditionalFlowMatching`, `MiddleVarianceFlowMatching`, `CurvedFlowMatching`, normalizing flows.

### `butane.nn.functional`, `butane.nn.utils`
Losses and ops (`pinball_loss`, `kl_div_gaussians`, `pgm` / `fgm` adversarial steps), plus
`ModelTrainer`, `compute_grad_norm`, `freeze_module` / `unfreeze_module`, `zero_module`, `init_weights`,
`get_lr`, `calculate_output_size`, `load_state`.

### `butane.data`
| | |
|---|---|
| Datasets | `Dataset` (data + targets, device handling, transforms), `TorchDatasetWrapper`, `PairDataset`, goal-conditioned `TrajectoryDataset` |
| Scalers | `StandardScaler`, `MinMaxScaler`, `ManualScaler` |
| Ops | `drop`, `drop_to_max_size`, `bagging`, `trim`, `randperm`, `Transforms`, `jigsaw` |
| Toy data (`butane.data.toy`) | `make_moons`, `make_spiral`, `make_pinwheel`, `make_ring`, `make_eight_normal`, `make_disjoint_circle` |

### The rest
| | |
|---|---|
| `butane.math` | `odeint`, `OTPlanner` (optimal transport), reductions around a dimension |
| `butane.optim` | `pcgrad` gradient surgery, `CoefficientScheduler` |
| `butane.clustering` | `KMeans`, `MiniBatchKMeans` (GPU, k-means++ init) |

## Tests

```
pixi run -e test pytest
```
