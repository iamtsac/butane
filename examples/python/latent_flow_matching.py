import torch
import numpy as np
import butane
import matplotlib.pyplot as plt

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
    ds = butane.data.Dataset(torch.load('data/mnist/mnist_train_data.pt'))
    ds.to(dev)
    butane.data.ops.drop_to_max_size(ds, 8000)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    c_enc = butane.nn.Conv2dBlock(
        input_dims = [1, 28, 28],
        channels = [32, 32],
        activation_function = [torch.nn.GELU()],
        conv_stride = [1, 1],
        conv_bias = [True, True],
        pool_kernels = [0, 0],
        normalization = [True, False]
    )

    mlp_enc = butane.nn.MLPBlock(
        input_dims = c_enc.output_size.prod().item(),
        output_dims=2,
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
        output_activation=False)

    mlp_dec = butane.nn.MLPBlock(
        input_dims = 2,
        output_dims=c_enc.output_size.prod().item(),
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU()],
        output_activation=False)

    c_dec = butane.nn.ConvTranspose2dBlock(
        input_dims = [c_enc.output_size[0].item(), c_enc.output_size[1].item(), c_enc.output_size[2].item()],
        channels = [32, 1],
        activation_function = [torch.nn.GELU(), torch.nn.Sigmoid()],
        conv_stride = [1, 1],
        conv_bias = [True, True],
        output_activation = False,
        normalization = [False, False])

    encoder = torch.nn.Sequential(c_enc, torch.nn.Flatten(1), mlp_enc)
    decoder = torch.nn.Sequential(mlp_dec, butane.nn.Unflatten(1, c_enc.output_size), c_dec)
    model = butane.nn.AE(encoder, decoder).to(dev)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-03, weight_decay=0.01)
    trainer = butane.nn.utils.ModelTrainer(model, dl, optimizer)
    trainer(50, model.loss_fn)

    model.eval()

    standard_scaler = butane.data.StandardScaler()
    transforms = butane.data.Transforms(
        standard_scaler,
    )
    standard_scaler.fit(model.encode(ds[:]['data']))

    fm_model = TimeDependentModel().to(dev)
    fm = butane.nn.ConditionalFlowMatching(0).to(dev)
    optimizer = torch.optim.AdamW(fm_model.parameters(), lr=1e-3)
    epochs = 50
    for epoch in range(epochs):
        sum_loss = 0.
        n_batches = len(dl)
        for batch in dl:
            optimizer.zero_grad()
            x1 = transforms(model.encode(batch['data']))
            x0 = torch.randn_like(x1)
            t = fm.sample_timesteps(x1.size(0))
            x_t, ut = fm.forward(x0, x1, t)
            vt = fm_model(x_t.to(dev), t)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        avg_loss = sum_loss / n_batches
        print(f"Epoch {epoch} -> Loss: {avg_loss}")

    fm_model.eval()
    sampled_hist = torch.vmap(standard_scaler.reverse)(fm.sample(
        model = fm_model,
        dims = [1000,2],
        sample_fn = torch.randn,
        timesteps = 100,
        condition = None,
        keep_record = True,
        solver='rk4'
    ))

    fig, ax = plt.subplots()
    plt.ion()
    for i in range(sampled_hist.size(1)):
        for gen_d in sampled_hist[:,i]:
            ax.cla()
            gen_dr = model.decode(gen_d.unsqueeze(0))
            gen_drn = gen_dr.squeeze().detach().cpu().numpy()
            ax.imshow(gen_drn)
            plt.pause(0.01)

    # fig, ax = plt.subplots()
    # latents = model.encode(ds.data).detach().cpu().numpy()
    # ax.scatter(latents[:, 0], latents[:, 1])
    # plt.show()
    # for d in ds.data.clone():
    #     fig, ax = plt.subplots(1,2)
    #     dn = d.squeeze().cpu().numpy()
    #     dr = model(d.unsqueeze(0))
    #     drn = dr.squeeze().detach().cpu().numpy()
    #     ax[0].imshow(dn)
    #     ax[1].imshow(drn)
    #     plt.show()

