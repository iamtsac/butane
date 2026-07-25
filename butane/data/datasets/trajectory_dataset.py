import copy
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch

from .dataset import Dataset


def _default_pad_left(input_tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Pads the tensor on the left by repeating the first element."""
    return torch.cat([input_tensor[0:1].repeat_interleave(pad_len, dim=0), input_tensor], dim=0)


def _default_pad_right(input_tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Pads the tensor on the right by repeating the last element."""
    return torch.cat([input_tensor, input_tensor[-1:].repeat_interleave(pad_len, dim=0)], dim=0)


class TrajectoryDataset(Dataset):
    """Dataset class for multi-modal trajectory data supporting unequal sequence lengths."""

    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor | np.ndarray] | None = None,
        *,
        horizon: int = 1,
        history: int = 1,
        align_start: bool = False,
        trim_end: int = 0,
        anchor_key: str | None = None,
        pad_left_fn: dict[str, Callable[[torch.Tensor, int], torch.Tensor]]
        | Callable[[torch.Tensor, int], torch.Tensor] = _default_pad_left,
        pad_right_fn: dict[str, Callable[[torch.Tensor, int], torch.Tensor]]
        | Callable[[torch.Tensor, int], torch.Tensor] = _default_pad_right,
        on_demand_device_load: bool = False,
        return_tuple: bool = False,
        device: torch.device = "cpu",
    ) -> None:

        super().__init__(
            on_demand_device_load=on_demand_device_load,
            return_tuple=return_tuple,
            device=device,
        )

        assert horizon > 0, "Horizon cannot be less than 1."
        assert history > 0, "History cannot be less than 1."

        self.horizon = horizon
        self.history = history - 1  # Number of lookback steps before pivot
        self.align_start = align_start
        self.trim_end = trim_end
        self.anchor_key = anchor_key

        # Storage for episodic data sequences: dict[str, list[torch.Tensor]]
        self.data: dict[str, list[torch.Tensor]] = {}

        # Ingest and separate into discrete episodes
        self._ingest_data(data)
        self._setup_padding(pad_left_fn=pad_left_fn, pad_right_fn=pad_right_fn)

    def _slice_and_pad(
        self,
        key: str,
        tensor: torch.Tensor,
        w_start: int,
        w_end: int,
    ) -> torch.Tensor:
        """Slices a target window relative to an individual modality's length."""
        ep_len = tensor.shape[0]
        valid_start = max(w_start, 0)
        valid_end = min(w_end, ep_len)
        total_len = w_end - w_start

        # Padding mask
        mask = torch.ones(total_len, dtype=torch.bool, device=self._device)

        if valid_start < valid_end:
            # Overlap exists with the actual data stream
            sliced = tensor[valid_start:valid_end]
            pad_left = valid_start - w_start
            pad_right = w_end - valid_end

            if pad_left > 0:
                sliced = self.pad_left_fn[key](sliced, pad_left)
                mask[:pad_left] = False
            if pad_right > 0:
                sliced = self.pad_right_fn[key](sliced, pad_right)
                mask[-pad_right:] = False
            return sliced, mask
        else:
            # Edge case: Window falls completely outside this modality's bounds
            mask[:] = False
            total_len = w_end - w_start
            if w_end <= 0:
                fallback_slice = tensor[0:1]
                return self.pad_left_fn[key](fallback_slice, total_len - 1), mask
            else:
                fallback_slice = tensor[ep_len - 1 : ep_len]
                return self.pad_right_fn[key](fallback_slice, total_len - 1), mask

    def __getitem__(
        self, idx: int, return_pad_masks: bool = False
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Unified item lookup using a clean episodic tracking system."""
        ep_idx = self.sample_to_episode[idx].item()
        local_t = self.sample_to_local_t[idx].item()

        # Calculate History Window Bounds
        hist_w_start = local_t - self.history
        hist_w_end = local_t + 1

        # Calculate Horizon Window Bounds
        horiz_w_start = (local_t - self.history) if self.align_start else local_t
        horiz_w_end = horiz_w_start + self.horizon

        data_dict: dict[str, torch.Tensor] = {}
        target_dict: dict[str, torch.Tensor] = {}

        # Explicitly tracking padding masks for inputs and targets
        # Needed for scaling, so one does not include the padded values
        # when fitting a scaler.
        data_masks: dict[str, torch.Tensor] = {}
        target_masks: dict[str, torch.Tensor] = {}

        for key, ep_list in self.data.items():
            tensor = ep_list[ep_idx]

            data_dict[key], data_masks[key] = self._slice_and_pad(
                key, tensor, hist_w_start, hist_w_end
            )
            target_dict[key], target_masks[key] = self._slice_and_pad(
                key, tensor, horiz_w_start, horiz_w_end
            )

        sample = {
            "data": data_dict,
            "targets": target_dict,
        }

        if return_pad_masks:
            sample.update({"data_masks": data_masks, "targets_masks": target_masks})
            return sample
        return sample

    def __len__(self) -> int:
        return len(self.sample_to_episode)

    def collect_samples(
        self,
        num_samples: int = 10000,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Subsamples trajectory windows into uniform (N_windows, T_window, ...) PyTorch tensors.
        Preserves exact sequence dimensions for Transforms while retaining masks for scaler fitting.
        """
        total_samples = len(self)
        indices = (
            torch.arange(total_samples)
            if total_samples <= num_samples
            else torch.randperm(total_samples)[:num_samples]
        )

        accumulators = {"data": defaultdict(list), "targets": defaultdict(list)}
        mask_accumulators = {"data": defaultdict(list), "targets": defaultdict(list)}

        for idx in indices:
            idx_item = idx.item()
            try:
                sample = self.__getitem__(idx_item, return_pad_masks=True)
                has_masks = True
            except TypeError:
                sample = self.__getitem__(idx_item)
                has_masks = False

            for ctx in ["data", "targets"]:
                if ctx not in sample:
                    continue

                ctx_dict = sample[ctx]
                mask_key = f"{ctx}_masks"
                masks = sample.get(mask_key, {}) if has_masks else {}

                if torch.is_tensor(ctx_dict):
                    accumulators[ctx]["__root__"].append(ctx_dict.detach().cpu())
                    m = (
                        masks
                        if torch.is_tensor(masks)
                        else torch.ones(ctx_dict.shape[0], dtype=torch.bool)
                    )
                    mask_accumulators[ctx]["__root__"].append(m.detach().cpu())

                elif isinstance(ctx_dict, dict):
                    for k, v in ctx_dict.items():
                        if not torch.is_tensor(v):
                            continue
                        accumulators[ctx][k].append(v.detach().cpu())

                        m = (
                            masks.get(k, torch.ones(v.shape[0], dtype=torch.bool))
                            if isinstance(masks, dict)
                            else torch.ones(v.shape[0], dtype=torch.bool)
                        )
                        mask_accumulators[ctx][k].append(m.detach().cpu())

        # Stack into solid (N_windows, T_window, ...) tensors
        collected_tensors = {"data": {}, "targets": {}, "data_masks": {}, "targets_masks": {}}
        for ctx in ["data", "targets"]:
            for key, tensor_list in accumulators[ctx].items():
                if tensor_list:
                    collected_tensors[ctx][key] = torch.stack(tensor_list, dim=0)
                    collected_tensors[f"{ctx}_masks"][key] = torch.stack(
                        mask_accumulators[ctx][key], dim=0
                    )

        return collected_tensors

    @staticmethod
    def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        if not isinstance(d, dict):
            return d
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, torch.as_tensor(v)))
        return dict(items)

    @staticmethod
    def _get_size_recursively(d: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
        size_d = dict()
        for k, v in d.items():
            if isinstance(v, dict):
                size_d[k] = TrajectoryDataset._get_size_recursively(v, device)
            else:
                shape = list(v.shape)

                # If it's a 1D trajectory [T], explicitly treat it as [T, 1]
                if len(shape) == 1:
                    shape.append(1)

                size_d[k] = torch.tensor(shape, device=device)
        return size_d

    def sizes(self) -> dict[str, Any]:
        dummy_sample = self.__getitem__(0)
        _sizes = dict(data=dict(), targets=dict())
        _sizes["data"] = self._get_size_recursively(dummy_sample["data"], device=self._device)
        _sizes["targets"] = self._get_size_recursively(dummy_sample["targets"], device=self._device)
        return _sizes

    def _ingest_data(self, data: Any):
        raw_dict = data if isinstance(data, dict) else {"_internal": data}

        anchor = (
            self.anchor_key
            if (self.anchor_key and self.anchor_key in raw_dict)
            else list(raw_dict.keys())[0]
        )
        first_val = raw_dict[anchor]

        # Parse data into discrete arrays per episode per modality
        if isinstance(first_val, (torch.Tensor, np.ndarray)):
            first_tensor = torch.as_tensor(first_val, device=self._device)
            n_episodes = first_tensor.shape[0]
            for key, val in raw_dict.items():
                val_tensor = torch.as_tensor(val, device=self._device)
                self.data[key] = [val_tensor[i] for i in range(n_episodes)]
        elif isinstance(first_val, (list, tuple)):
            n_episodes = len(first_val)
            for key, val in raw_dict.items():
                if isinstance(val, dict):
                    raise TypeError(
                        "Dictionary should have max depth of 1. Use `TrajectoryDataset.faltten_dict` to flatten it."
                    )
                self.data[key] = [torch.as_tensor(d, device=self._device) for d in val]

        sample_to_episode = []
        sample_to_local_t = []

        for i in range(n_episodes):
            ep_steps = self.data[anchor][i].shape[0]
            for t in range(ep_steps - self.trim_end):
                sample_to_episode.append(i)
                sample_to_local_t.append(t)

        self.sample_to_episode = torch.tensor(
            sample_to_episode, dtype=torch.long, device=self._device
        )
        self.sample_to_local_t = torch.tensor(
            sample_to_local_t, dtype=torch.long, device=self._device
        )

    def _setup_padding(self, pad_left_fn: Any, pad_right_fn: Any):
        self.pad_left_fn = {}
        self.pad_right_fn = {}
        for k in self.data.keys():
            if not isinstance(pad_left_fn, dict):
                self.pad_left_fn[k] = pad_left_fn
            else:
                self.pad_left_fn[k] = pad_left_fn.get(k, _default_pad_left)

            if not isinstance(pad_right_fn, dict):
                self.pad_right_fn[k] = pad_right_fn
            else:
                self.pad_right_fn[k] = pad_right_fn.get(k, _default_pad_right)
