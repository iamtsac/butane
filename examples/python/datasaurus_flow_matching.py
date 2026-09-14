import torch
import numpy as np
import butane
import matplotlib.pyplot as plt
import pandas as pd
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.conditional_flow_matching import *


def dino_dataset(n=2000):
    df = pd.read_csv("data/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), n)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    X = np.stack((x, y), axis=1)
    return torch.tensor(X)

class TimeDependentModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.time_proj = torch.nn.Sequential(
            butane.nn.FourierEmbeddings(2),
            butane.nn.MLPBlock(
                input_dims = 2,
                output_dims = 2,
                hidden_dims=[256, 256],
                activation_function=[torch.nn.GELU()],
                output_activation=False),
        )
        self.mlp = butane.nn.MLPBlock(
            input_dims = 4,
            output_dims=2,
            hidden_dims=[256, 256],
            activation_function=[torch.nn.GELU()],
            output_activation=False)

    def forward(self, x, t, cond=None):
        # return self.mlp(x + self.time_proj(t))
        return self.mlp(torch.hstack([x, self.time_proj(t)]))


if __name__ == "__main__":
    dev = torch.device('cuda')
    ds = butane.data.Dataset(dino_dataset(8000))
    ds.to(dev)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    standard_scaler = butane.data.StandardScaler()
    transforms = butane.data.Transforms(
        standard_scaler,
        lambda x : x * 2,
    )
    standard_scaler.fit(ds)

    model = TimeDependentModel().to(dev)
    fm = butane.nn.ConditionalFlowMatching(0).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 2000
    for epoch in range(epochs):
        sum_loss = 0.
        n_batches = len(dl)
        for batch in dl:
            optimizer.zero_grad()
            x1 = transforms(batch['data'])
            x0 = torch.randn_like(x1)
            t = fm.sample_timesteps(x1.size(0))
            x_t, ut = fm.forward(x0, x1, t)
            vt = model(x_t.to(dev), t)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        avg_loss = sum_loss / n_batches
        print(f"Epoch {epoch} -> Loss: {avg_loss}")

    sampled_hist = torch.vmap(standard_scaler.reverse)(fm.sample(
        model = model,
        dims = [1000,2],
        sample_fn = torch.randn,
        timesteps = 100,
        condition = None,
        keep_record = True,
        solver='rk4'
    )).cpu()
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(0, len(sampled_hist), 1):
        ax.cla()
        ax.scatter(sampled_hist[i][:, 0], sampled_hist[i][:, 1], s=10)
        plt.pause(0.005)
    plt.ioff()
    ax.cla()
    ax.scatter(sampled_hist[-1][:, 0], sampled_hist[-1][:, 1], s=10)
    plt.show()
