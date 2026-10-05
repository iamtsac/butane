import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch

import butane
import mosi


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
    fm: butane.nn.FlowMatching,
    ema: torch.nn.Module | None = None,
    logger: mosi.Sitter | None = None,
):
    model_mode = model.training
    model.eval()
    if ema:
        ema.enable()
    x0 = fm.source_distribution().sample((10,))
    generations = fm.flow(
        model=model,
        x0=x0,
        n_timesteps=100,
        keep_record=True,
    )
    generations = generations[-1].moveaxis(1, -1).cpu()

    if ema:
        ema.disable()
    model.train(model_mode)

    for i in range(10):
        fig, ax = plt.subplots()
        ax.imshow(generations[i])
        if logger is not None:
            logger.add_image(f"outputs_{i}.png", fig)
        else:
            plt.show()
        plt.close()
    return generations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--fpath", type=str)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = ((torch.load("data/mnist/mnist_train_data.pt") / 255.) * 2) - 1
    ds = butane.data.Dataset(
        data=data,
        targets=torch.load("data/mnist/mnist_train_targets.pt"),
        on_demand_device_load=False,
        device="cpu",
    )
    butane.data.ops.drop_to_max_size(ds, 40_000)
    dl = butane.data.utils.InfiniteIterator(
        torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, pin_memory=False)
    )
    class_conditioned = False

    model = butane.nn.DiT2d(
        input_dims=[1,28,28],
        hidden_dims=768,
        mlp_ratio=4.0,
        patch_size=4,
        depth=12,
        attention_heads=16,
        attention_dropout=0.0,
        output_dims=None,
        time_dependent=True,
        time_embedding_size=64,
        time_scaling_coeff=1.0,
        embedding_size=256,
        learn_input_embeddings=False,
        learn_time_embeddings=False,
        learn_ctx_embeddings=False,
        adaLN_zero=True,
        n_classes=None,
        class_drop_prob=0.0,
        ctx_dim=None,
        ctx_patch_size=None,
        ctx_concat=False,
        ctx_cross_attention=False,
        cross_attention_heads=None,
        ctx_in_context=False,
    ).to(device)
    print(
        f" Model Parameters: {torch.nn.utils.parameters_to_vector(model.parameters()).size()[0]}"
    )

    fm = butane.nn.ConditionalFlowMatching(0.00).to(device)
    fm.set_source_distribution(
        torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(1, 28, 28), torch.ones(1, 28, 28)), 3
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ema = butane.nn.EMA(model=model, decay=0.9999)
    logger = mosi.Sitter(".tmp/mnist_fm_dit", overwrite=True, eval_mode=args.eval_only)

    if not args.eval_only:
        training_steps = 10000
        for s in range(training_steps):
            batch = next(dl)
            x1 = batch["data"].to(device, non_blocking=True)
            x0 = fm.source_distribution().sample((x1.size(0),)).to(x1.device)
            t = fm.sample_timesteps(x1.size(0))
            x_t, u_t = fm(x0, x1, t)

            v_t = model(x_t.to(device), t)
            loss = torch.mean((v_t - u_t.to(device)) ** 2)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = butane.nn.utils.compute_grad_norm(model)
            optimizer.step()
            ema.update()

            if s % 100 == 0:
                logger.add_stats(loss=loss.item())
                print(f"Step {s} -> Loss: {loss.item()} Grad Norm: {grad_norm}")
            if ((s + 1) % 1000) == 0:
                logger.checkpoint(model=model, optimizer=optimizer, ema=ema)
                eval_model(model=model, fm=fm, ema=ema, logger=logger)
            logger.step()
    else:
        logger.load_checkpoint(
            args.fpath,
            model=model,
            optimizer=optimizer,
            ema=ema,
        )
        model.eval()
        eval_model(
            model=model,
            fm=fm,
            ema=ema,
            logger=None,
        )
