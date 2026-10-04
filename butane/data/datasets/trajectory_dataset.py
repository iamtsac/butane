import copy
from collections import defaultdict
from typing import Any, Callable, Collection, Sequence

import numpy as np
import torch

from ._trajectory_dataset_utils import (
    GoalSampler,
    _default_pad_left,
    _default_pad_right,
)
from .dataset import Dataset


class TrajectoryDataset(Dataset):
    """Dataset class for multi-modal trajectory data supporting unequal sequence lengths.

    Two independent ways to shorten an episode, easily confused:
      - `trim_end` drops PIVOT INDICES. The frames stay in the tensors and still show up as
        history/horizon context of earlier pivots; only the windows anchored ON them go away.
      - `drop_last_frames` deletes the FRAMES themselves, per episode, before indexing.
    """

    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor | np.ndarray] | None = None,
        *,
        horizon: int = 1,
        history: int = 1,
        align_start: bool = False,
        trim_end: int = 0,
        drop_last_frames: int | Sequence[int] = 0,
        anchor_key: str | None = None,
        pad_left_fn: dict[str, Callable[[torch.Tensor, int], torch.Tensor]]
        | Callable[[torch.Tensor, int], torch.Tensor] = _default_pad_left,
        pad_right_fn: dict[str, Callable[[torch.Tensor, int], torch.Tensor]]
        | Callable[[torch.Tensor, int], torch.Tensor] = _default_pad_right,
        goals: dict[str, GoalSampler] | None = None,
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
        if trim_end < 0:
            raise ValueError(f"trim_end must be >= 0, got {trim_end}")

        self.horizon = horizon
        self.history = history - 1  # Number of lookback steps before pivot
        self.align_start = align_start
        self.trim_end = trim_end
        self.drop_last_frames = drop_last_frames
        self.anchor_key = anchor_key
        self.goals = dict(goals or {})  # Before ingest: `_build_index` reads it

        # Storage for episodic data sequences: dict[str, list[torch.Tensor]]
        self.data: dict[str, list[torch.Tensor]] = {}

        # Ingest and separate into discrete episodes
        self._ingest_data(data)
        self._check_goals()
        self._setup_padding(pad_left_fn=pad_left_fn, pad_right_fn=pad_right_fn)

    @property
    def horizon_anchor_index(self) -> int:
        """Index WITHIN the returned horizon window that holds the pivot step ("now").

        The horizon window starts at `local_t - history` when align_start is set and at
        `local_t` otherwise (see __getitem__), so where "now" lands inside the returned
        window is a function of align_start - not something a caller can hardcode safely.
        Anything that re-anchors a horizon against the current pose (a relative/delta action
        encoding, say) must read the index from here rather than recomputing `history - 1`
        on its own: the two agree only while align_start is True, and nothing about flipping
        that flag would otherwise announce that every such anchor had silently moved.
        """
        return self.history if self.align_start else 0

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

        for name, sampler in self.goals.items():
            # One frame for every source: `_check_goals` made sure their episodes line up
            lengths, starts = self._goal_index[sampler.sources[0]]
            t = min(local_t, int(lengths[ep_idx]) - 1)  # Modalities may differ in length
            goal_ep, goal_t = sampler.sample(ep_idx, t, lengths, starts)
            for key, source in sampler.output_keys(name).items():
                data_dict[key] = self.data[source][goal_ep][goal_t : goal_t + 1]
                data_masks[key] = torch.ones(1, dtype=torch.bool, device=self._device)

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
        keys: Collection[str] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Subsamples trajectory windows into uniform (N_windows, T_window, ...) PyTorch tensors.
        Preserves exact sequence dimensions for Transforms while retaining masks for scaler fitting.
        `keys` limits the collected entries to those keys, None collects all of them.
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
                        if not torch.is_tensor(v) or (keys is not None and k not in keys):
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

    def _resolve_anchor_key(self, raw_dict: dict[str, Any]) -> str:
        """The key whose per-episode length defines how many samples an episode yields.

        An anchor_key that was asked for but isn't in the data is an error, not something to
        quietly paper over: the fallback below picks whatever happens to be first in the
        dict, which is insertion-order-dependent and usually an unrelated modality. That
        substitution is invisible - every key having the same length makes it look correct
        right up until one of them doesn't, at which point the sample count silently comes
        from the wrong stream.
        """
        if self.anchor_key is not None and self.anchor_key not in raw_dict:
            raise KeyError(
                f"anchor_key {self.anchor_key!r} is not present in the data. Available keys: "
                f"{sorted(raw_dict.keys())}. Pass a key that exists, or pass anchor_key=None "
                f"to deliberately anchor on the first key."
            )
        return self.anchor_key if self.anchor_key is not None else list(raw_dict.keys())[0]

    def _ingest_data(self, data: Any):
        # If dat is not a dict structre, create a n internal dict representation.
        raw_dict = data if isinstance(data, dict) else {"_internal": data}
        if not raw_dict:
            raise ValueError("TrajectoryDataset received no data.")

        anchor = self._resolve_anchor_key(raw_dict)
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

        self._apply_drop_last_frames(n_episodes)
        self._build_index()

    def _build_index(self) -> None:
        """Maps every window index to its (episode, pivot step), from the episodes now held.

        Called once on ingest and again by `split`, which drops whole episodes and so invalidates
        every index past the first one it removed.
        """
        anchor = self._resolve_anchor_key(self.data)
        n_episodes = len(self.data[anchor])

        sample_to_episode = []
        sample_to_local_t = []
        starved = []

        for i in range(n_episodes):
            ep_steps = self.data[anchor][i].shape[0]
            n_samples = ep_steps - self.trim_end
            if n_samples <= 0:
                starved.append((i, ep_steps))
                continue
            for t in range(n_samples):
                sample_to_episode.append(i)
                sample_to_local_t.append(t)

        if starved:
            raise ValueError(
                f"trim_end={self.trim_end} leaves no samples in {len(starved)} episode(s) - "
                f"e.g. episode {starved[0][0]} has {starved[0][1]} step(s) on anchor "
                f"{anchor!r}. trim_end drops PIVOT INDICES from the end of each episode, so "
                f"it must stay below the shortest episode's length."
            )
        if not sample_to_episode:
            raise ValueError("No samples could be built - every episode was empty.")

        self.sample_to_episode = torch.tensor(
            sample_to_episode, dtype=torch.long, device=self._device
        )
        self.sample_to_local_t = torch.tensor(
            sample_to_local_t, dtype=torch.long, device=self._device
        )

        # Per goal: episode lengths and where each episode starts in the flattened frames, read
        # off its first source
        self._goal_index = {}
        for sampler in self.goals.values():
            source = sampler.sources[0]
            if source in self.data and source not in self._goal_index:
                lengths = torch.tensor([ep.shape[0] for ep in self.data[source]])
                self._goal_index[source] = (lengths, torch.cumsum(lengths, 0) - lengths)

    def _check_goals(self) -> None:
        seen: set[str] = set()
        for name, sampler in self.goals.items():
            for source in sampler.sources:
                if source not in self.data:
                    raise KeyError(
                        f"goal {name!r} reads {source!r}, which is not in the data "
                        f"({sorted(self.data)}) - was it dropped by a filter?"
                    )
            # One step index has to be valid in every source of the goal
            lengths = {
                source: [ep.shape[0] for ep in self.data[source]] for source in sampler.sources
            }
            if len({tuple(v) for v in lengths.values()}) > 1:
                raise ValueError(
                    f"goal {name!r} reads {sampler.sources}, whose episodes differ in length - "
                    f"one goal frame needs the same steps in every source"
                )
            for key in sampler.output_keys(name):
                if key in self.data:
                    raise ValueError(f"goal {key!r} would shadow the stored key of the same name")
                if key in seen:
                    raise ValueError(f"two goals both produce {key!r}")
                seen.add(key)

    def split(
        self, percentage: float, generator: torch.Generator | None = None
    ) -> "TrajectoryDataset":
        """Keeps `percentage` of the windows here and returns the rest, as `Dataset.split` does
        with rows - except the cut falls on whole EPISODES.

        Neighbouring windows share frames through their history and horizon, so a row-wise split
        would leave the same frames on both sides: anything fitted on the returned half (a
        conformal bound, a validation score) would already have seen its frames in training.
        Whole episodes means the kept share only lands as close to `percentage` as one episode
        allows.
        """
        anchor = self._resolve_anchor_key(self.data)
        windows = [max(ep.shape[0] - self.trim_end, 0) for ep in self.data[anchor]]
        order = torch.randperm(len(windows), generator=generator).tolist()

        # Stop at the episode boundary nearest the asked share, over or under it
        wanted = percentage * sum(windows)
        kept_windows, keep_n = 0, 0
        for position, episode in enumerate(order):
            if abs(kept_windows + windows[episode] - wanted) > abs(kept_windows - wanted):
                break
            kept_windows += windows[episode]
            keep_n = position + 1

        keep, moved = sorted(order[:keep_n]), sorted(order[keep_n:])
        if not keep or not moved:
            raise ValueError(
                f"a {percentage:.3g} split of {len(order)} episode(s) leaves one side empty - a "
                f"trajectory dataset splits whole episodes, so it needs at least one on each side."
            )

        split_ds = TrajectoryDataset(
            data={key: [episodes[i] for i in moved] for key, episodes in self.data.items()},
            horizon=self.horizon,
            history=self.history + 1,  # __init__ stores it as the number of lookback steps
            align_start=self.align_start,
            trim_end=self.trim_end,
            drop_last_frames=0,  # already applied to these frames on ingest
            anchor_key=self.anchor_key,
            pad_left_fn=self.pad_left_fn,
            pad_right_fn=self.pad_right_fn,
            goals=self.goals,
            on_demand_device_load=self._on_demand_device_load,
            return_tuple=self._return_tuple,
            device=self._device,
        )
        self.data = {key: [episodes[i] for i in keep] for key, episodes in self.data.items()}
        self._build_index()
        return split_ds

    def _apply_drop_last_frames(self, n_episodes: int) -> None:
        """Removes trailing FRAMES from every modality of each episode, in place.

        Distinct from trim_end, which only drops pivot indices from the sample index and
        leaves the frames themselves in the tensors - still reachable as horizon/history
        context of earlier pivots. Use this one to actually delete a trailing segment (a
        post-episode hold the robot spent frozen, a settling period), and trim_end to merely
        stop anchoring windows there.

        Accepts an int (same count for every episode) or one count per episode - a fixed
        wall-clock tail spans a different number of frames in each episode whenever the
        recording rate varied, so a per-episode list is the only correct form there.
        """
        drop = self.drop_last_frames
        if isinstance(drop, int):
            if drop == 0:
                return
            drops = [drop] * n_episodes
        else:
            drops = list(drop)
            if len(drops) != n_episodes:
                raise ValueError(
                    f"drop_last_frames has {len(drops)} entries but there are {n_episodes} "
                    f"episodes; pass one count per episode or a single int."
                )
        if any(d < 0 for d in drops):
            raise ValueError(f"drop_last_frames entries must be >= 0, got {drops}")

        for key, ep_list in self.data.items():
            for i, d in enumerate(drops):
                if d == 0:
                    continue
                ep = ep_list[i]
                if d >= ep.shape[0]:
                    raise ValueError(
                        f"drop_last_frames[{i}]={d} would empty episode {i} of key {key!r} "
                        f"(length {ep.shape[0]})."
                    )
                ep_list[i] = ep[: ep.shape[0] - d]

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
