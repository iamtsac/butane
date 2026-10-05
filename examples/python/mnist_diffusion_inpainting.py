import torch
import numpy as np
import matplotlib.pyplot as plt

import butane
import mosi

if __name__ == "__main__":
    dev = torch.device("cuda")
    ds = butane.data.Dataset(torch.load('data/mnist/mnist_train_data.pt'))
    ds.data = (ds.data * 2)  - 1.
    ds.to(dev)
    # butane.data.ops.drop_to_max_size(ds, 5000)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model = butane.nn.unet.UNet2d(
        [1, 28, 28],
        channels=[32, 64, 128],
        self_condition=False,
        attention=True,
        use_film=False,
        attention_resolution=[3]
    ).to(dev)

    diffusion = butane.nn.Diffusion(
        num_timesteps=1000,
        scheduler='cosine',
        variance_type='fixed_large',
        scale_timesteps=True,
    ).to(dev)

    # Load a pretrained model
    mosi.Sitter(".tmp/mnist_ddpm", eval_mode=True).load_checkpoint(1000, model=model)

    x_orig = ds.data[:30]
    M = torch.ones_like(x_orig)
    M[:, :, 7:21, 7:21] = 0.

    out = diffusion.inpainting(
        model=model,
        x_T=torch.randn_like(x_orig),
        x_original=x_orig,
        mask=M,
        keep_record=False
    ).cpu()

    for i in out:
        fig, ax = plt.subplots()
        ax.imshow(i.cpu().moveaxis(0, -1))
        plt.show()
