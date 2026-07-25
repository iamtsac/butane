import functools
import itertools
import time
from typing import Any, Callable, Literal

import torch

from ...._utils import *
from ....math import *


class FlowMatching(torch.nn.Module):
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        # trick to get module's device
        self.register_buffer("_device_buffer", torch.zeros(1))
        self._sigma = sigma
        self.__source_distribution = None

    def set_source_distribution(self, dist: object):
        self.__source_distribution = dist

    def source_distribution(self) -> object:
        return self.__source_distribution

    def sample_timesteps(self, n: int, skewed: bool = False) -> torch.Tensor:
        if skewed:
            mu = -1.2
            std = 1.2
            epsilon = torch.randn((n,), device=self.device)
            sigma = (epsilon * std + mu).exp()
            time = (1 / (1 + sigma)).clamp(0.0001, 1.0)
            return time.unsqueeze(-1)
        else:
            return torch.rand(n, 1, device=self.device)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement forward()")

    @property
    def device(self):
        return self._device_buffer.device

    @staticmethod
    def save_vector_field_hook(storage_list: list):
        """
        A ready-made hook factory that allows users to capture model outputs (v_out)
        and move them to CPU automatically during integration loops.
        """

        def hook(t: torch.Tensor, x: torch.Tensor, v_out: torch.Tensor):
            storage_list.append(v_out.detach().cpu())

        return hook

    @torch.no_grad()
    def flow(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        n_timesteps: int,
        func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any] | None = None,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        method: Literal["euler", "heun2", "rk4"] = "euler",
        reverse: bool = False,
        guidance_scale: float = 1.0,
        state_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
        edm_time_grid: bool = False,
        batch_size: int = 128,
        target_device: torch.device | None = None,
    ) -> torch.Tensor:

        if target_device is None:
            target_device = self.device
        model.to(self.device)
        x0 = x0.to(self.device)
        n_generations = 1
        n_conditions = x0.size(0)
        spatial_dims = x0.shape[1:]

        if condition is not None:
            condition = apply_recursively(condition, lambda x: x.to(self.device))

        if edm_time_grid:
            timesteps = self.edm_time_grid(n_timesteps=n_timesteps, reverse=reverse).to(self.device)
        else:
            timesteps = torch.linspace(
                0.0 if not reverse else 1.0 - 1e-05,
                1.0 - 1e-05 if not reverse else 0.0,
                n_timesteps,
                device=self.device,
            )

        if multiple_gen_per_condition:
            n_generations, n_conditions = x0.size(0), x0.size(1)
            spatial_dims = x0.shape[2:]
            x0 = x0.transpose(0, 1).flatten(0, 1)
            if condition is not None:
                condition = apply_recursively(
                    condition, lambda x: x.repeat_interleave(repeats=n_generations, dim=0)
                )

        if func is None:
            func = lambda t, x, c: model(x, t, c)

        def func_wrapper(t, x, c):
            t_expanded = t.expand(x.size(0)).unsqueeze(-1)  # (B, 1)
            if guidance_scale != 1.0 and c is not None:
                v_cond = func(t_expanded, x, c)
                v_uncond = func(t_expanded, x, None)
                v_out = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_out = func(t_expanded, x, c)

            if state_hook is not None:
                state_hook(t_expanded, x, v_out)

            return v_out

        x0_iter = batching(x0, batch_size, dim=0)
        if condition is not None:
            condition_iter = batching(condition, batch_size, dim=0)
        else:
            condition_iter = itertools.repeat(None)

        out_shape = (
            (n_timesteps, x0.size(0), *spatial_dims) if keep_record else (x0.size(0), *spatial_dims)
        )
        xs = torch.empty(out_shape, device=target_device, dtype=x0.dtype)

        current_idx = 0
        for x0_batch, cond_batch in zip(x0_iter, condition_iter):
            batch_n = x0_batch.size(0)
            x, _ = odeint(
                func=functools.partial(func_wrapper, c=cond_batch),
                x0=x0_batch,
                steps=timesteps,
                method=method,
                return_trajectory=keep_record,
                return_func_outputs=False,
            )
            if keep_record:
                xs[:, current_idx : current_idx + batch_n] = x[1:].to(target_device)
            else:
                xs[current_idx : current_idx + batch_n] = x
            current_idx += batch_n

        def _revert_shape(x: torch.Tensor):
            return (
                x.view(
                    n_timesteps if keep_record else 1,
                    n_conditions,
                    n_generations if multiple_gen_per_condition else 1,
                    *spatial_dims,
                )
                .movedim((0, 1, 2), (1, 2, 0))
                .squeeze(1)
            )

        xs = _revert_shape(xs)

        if not multiple_gen_per_condition:
            xs = xs.squeeze(0)

        return xs

    @torch.no_grad()
    def flow_likelihood(
        self,
        model: torch.nn.Module,
        x1: torch.Tensor,
        n_timesteps: int,
        func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any] | None = None,
        condition: torch.Tensor | None = None,
        keep_record: bool = False,
        multiple_gen_per_condition: bool = False,
        edm_time_grid: bool = False,
        method: Literal["euler", "heun2", "rk4"] = "euler",
        state_hook: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None] | None = None,
        batch_size: int = 128,
        target_device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if target_device is None:
            target_device = self.device
        model.to(self.device)
        x1 = x1.to(self.device)
        n_generations = 1
        n_conditions = x1.size(0)
        spatial_dims = x1.shape[1:]

        if condition is not None:
            condition = apply_recursively(condition, lambda x: x.to(self.device))

        if multiple_gen_per_condition:
            n_generations, n_conditions = x1.size(0), x1.size(1)
            spatial_dims = x1.shape[2:]
            x1 = x1.transpose(0, 1).flatten(0, 1)
            if condition is not None:
                condition = apply_recursively(
                    condition, lambda x: x.repeat_interleave(repeats=n_generations, dim=0)
                )

        z = (torch.randn_like(x1).to(self.device) < 0) * 2.0 - 1.0
        if edm_time_grid:
            timesteps = self.edm_time_grid(n_timesteps=n_timesteps, reverse=True).to(self.device)
        else:
            timesteps = torch.linspace(1-1e-05, 0, n_timesteps, device=self.device)

        x1_iter = batching(x1, batch_size, dim=0)
        z_iter = batching(z, batch_size, dim=0)

        if condition is not None:
            condition_iter = batching(condition, batch_size, dim=0)
        else:
            condition_iter = itertools.repeat(None)

        out_shape = (
            (n_timesteps, x1.size(0), *spatial_dims) if keep_record else (x1.size(0), *spatial_dims)
        )
        xs = torch.empty(out_shape, device=target_device, dtype=x1.dtype)
        lls = torch.empty(x1.size(0), device=target_device, dtype=x1.dtype)

        current_idx = 0

        for x1_batch, z_batch, cond_batch in zip(x1_iter, z_iter, condition_iter):
            batch_n = x1_batch.size(0)

            # Redefining func dynamically per batch block to handle tracking correctly
            target_func = func if func is not None else lambda t, x, c: model(x, t, c)

            def likelihood_func_wrapper(t, x, c):
                x_val, _ = x
                with torch.set_grad_enabled(True):
                    x_val = x_val.detach().requires_grad_(True)
                    t_in = t.expand(x_val.size(0)).unsqueeze(-1)  # (B, 1)
                    ut = target_func(t_in, x_val, c)

                    # Hutchinson's Trace Estimator
                    ut_dot_z = torch.einsum("ij,ij->i", ut.flatten(1), z_batch.flatten(1))
                    grad_ut_dot_z = torch.autograd.grad(
                        outputs=ut_dot_z,
                        inputs=x_val,
                        grad_outputs=torch.ones_like(ut_dot_z),
                    )[0]
                    div = torch.einsum("ij,ij->i", grad_ut_dot_z.flatten(1), z_batch.flatten(1))

                # --- GENERIC STATE HOOK ---
                if state_hook is not None:
                    state_hook(t_in, x_val, ut)

                return ut.detach(), div.detach()

            init_state = (x1_batch, torch.zeros(batch_n, device=self.device))

            traj, _ = odeint(
                functools.partial(likelihood_func_wrapper, c=cond_batch),
                init_state,
                timesteps,
                method=method,
                return_trajectory=True,
                return_func_outputs=False,
            )

            x_traj, logdet_traj = traj

            if keep_record:
                xs[:, current_idx : current_idx + batch_n] = x_traj[1:].to(target_device)
            else:
                xs[current_idx : current_idx + batch_n] = x_traj[-1].to(target_device)

            x0_final = x_traj[-1].to(target_device)
            delta_logp = logdet_traj[-1].to(target_device)

            log_p0 = self.__source_distribution.log_prob(x0_final.cpu()).to(target_device)
            log_p0 = log_p0.flatten().to(target_device)
            total_ll = log_p0 + delta_logp

            lls[current_idx : current_idx + batch_n] = total_ll
            current_idx += batch_n

        def _revert_shape(x: torch.Tensor, is_ll: bool = False):
            if is_ll:
                return x.view(
                    n_conditions, n_generations if multiple_gen_per_condition else 1
                ).movedim((0, 1), (1, 0))

            return (
                x.view(
                    n_timesteps if keep_record else 1,
                    n_conditions,
                    n_generations if multiple_gen_per_condition else 1,
                    *spatial_dims,
                )
                .movedim((0, 1, 2), (1, 2, 0))
                .squeeze(1)
            )

        xs = _revert_shape(xs, is_ll=False)
        lls = _revert_shape(lls, is_ll=True)
        if not multiple_gen_per_condition:
            xs = xs.squeeze(0)
            lls = lls.squeeze(0)

        return xs, lls

    @torch.no_grad()
    def log_likelihood(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        n_timesteps: int,
        monte_carlo_estiamtes: int = 5,
        condition: torch.Tensor | None = None,
        multiple_gen_per_condition: bool = False,
        edm_time_grid: bool = False,
        method: Literal["euler", "heun2", "rk4"] = "euler",
        target_device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x1 = self.flow(
            model=model,
            x0=x0,
            n_timesteps=n_timesteps,
            condition=condition,
            keep_record=False,
            multiple_gen_per_condition=multiple_gen_per_condition,
            method=method,
            edm_time_grid=edm_time_grid,
            target_device=target_device,
        )

        monte_carlo_lls = []
        for _ in range(monte_carlo_estiamtes):
            _, log_likelihood = self.flow_likelihood(
                model=model,
                x1=x1,
                n_timesteps=n_timesteps,
                condition=condition,
                keep_record=False,
                multiple_gen_per_condition=multiple_gen_per_condition,
                edm_time_grid=edm_time_grid,
                target_device=target_device,
            )
            monte_carlo_lls.append(log_likelihood)
        monte_carlo_lls = torch.stack(monte_carlo_lls)
        log_likelihood_estimate = monte_carlo_lls.mean(0)
        return x1, log_likelihood_estimate

    @staticmethod
    def format_hook_data(buffer: list[torch.Tensor], n_timesteps: int) -> torch.Tensor:
        batch_chunks = [buffer[i:i + n_timesteps+1] for i in range(0, len(buffer), n_timesteps+1)]
        stacked_batches = [torch.stack(batch, dim=0) for batch in batch_chunks]
        return torch.cat(stacked_batches, dim=1)

    @staticmethod
    def edm_time_grid(n_timesteps: int, r: int = 7, reverse: bool = False):
        sigma_max = 80.0
        sigma_min = 0.002
        t = torch.arange(0, n_timesteps, dtype=torch.float64) / (n_timesteps - 1)
        timesteps = (sigma_max ** (1 / r) + t * (sigma_min ** (1 / r) - sigma_max ** (1 / r))) ** r
        timesteps = (timesteps / (1 + timesteps)).squeeze()
        timesteps = torch.cat([timesteps, torch.full_like(timesteps[:1], 1.0)])
        if not reverse:
            timesteps = 1 - timesteps.clamp(0.0, 1.0)
        return timesteps.float()


class ConditionalFlowMatching(FlowMatching):
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self._sigma
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = x1 - x0
        return x_t, u_t


class TargetConditionalFlowMatching(FlowMatching):
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = t * x1
        sigma_t = 1 - (1 - self._sigma) * t
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = (x1 - (1 - self._sigma) * x_t) / (1 - (1 - self._sigma) * t).clamp(min=1e-8)
        return x_t, u_t


class MiddleVarianceFlowMatching(FlowMatching):
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]

        mu_t = t * x1 + (1 - t) * x0

        scale = -4 * (t - 0.5) ** 2 + 1
        sigma_t = self._sigma * scale
        d_sigma_t = self._sigma * (-4 * (2 * t - 1))

        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon

        u_t = (x1 - x0) + d_sigma_t * epsilon
        return x_t, u_t


class CurvedFlowMatching(FlowMatching):
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = (x1 - x0) * (2 * t - t**2) + x0
        sigma_t = self._sigma
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma_t * epsilon
        u_t = (x1 - x0) * (2 - 2 * t)
        return x_t, u_t

    @staticmethod
    def interpolate(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        while len(x0.size()) != len(t.size()):
            t = t[..., None]
        mu_t = (x1 - x0) * (2 * t - t**2) + x0
        epsilon = torch.randn_like(x0)
        x_t = mu_t + sigma * epsilon
        u_t = (x1 - x0) * (2 - 2 * t)
        return x_t, u_t
