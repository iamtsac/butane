"""Window geometry, anchor resolution, and frame trimming on TrajectoryDataset."""

import numpy as np
import pytest
import torch

from butane.data.datasets.trajectory_dataset import TrajectoryDataset


def _data(n_episodes=3, length=20, dim=2):
    return {
        "actions": [np.arange(length * dim, dtype=np.float32).reshape(length, dim) + i
                    for i in range(n_episodes)],
        "obs": [np.zeros((length, dim), dtype=np.float32) for _ in range(n_episodes)],
    }


# ---------------------------------------------------------------------------
# horizon_anchor_index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("history", [1, 2, 4])
def test_horizon_anchor_index_tracks_align_start(history):
    """Where "now" sits inside the returned horizon window is a function of align_start, so a
    caller re-anchoring a horizon must read it from here rather than assuming history-1."""
    aligned = TrajectoryDataset(data=_data(), horizon=6, history=history, align_start=True)
    assert aligned.horizon_anchor_index == history - 1

    unaligned = TrajectoryDataset(data=_data(), horizon=6, history=history, align_start=False)
    assert unaligned.horizon_anchor_index == 0


@pytest.mark.parametrize("align_start", [True, False])
def test_horizon_anchor_index_points_at_the_pivot_row(align_start):
    """Not just an arithmetic identity: the row it names must really be the pivot step."""
    history, horizon = 3, 5
    ds = TrajectoryDataset(data=_data(), horizon=horizon, history=history,
                           align_start=align_start, anchor_key="actions")
    sample = ds[7]
    pivot_from_history = sample["data"]["actions"][-1]  # history window always ends at "now"
    pivot_from_horizon = sample["targets"]["actions"][ds.horizon_anchor_index]
    torch.testing.assert_close(pivot_from_history, pivot_from_horizon)


# ---------------------------------------------------------------------------
# anchor_key resolution
# ---------------------------------------------------------------------------


def test_missing_anchor_key_raises_instead_of_falling_back():
    """Silently anchoring on whatever key happened to be first is how a dead anchor_key can
    sit in a task for a long time looking correct - every key having the same length hides
    it right up until one of them doesn't."""
    with pytest.raises(KeyError, match="anchor_key"):
        TrajectoryDataset(data=_data(), horizon=4, history=2, anchor_key="does_not_exist")


def test_anchor_key_none_still_falls_back_to_first_key():
    ds = TrajectoryDataset(data=_data(), horizon=4, history=2, anchor_key=None)
    assert len(ds) == 3 * 20


def test_anchor_key_decides_the_sample_count():
    """With ragged modalities the anchor is what sets how many windows an episode yields."""
    data = _data(n_episodes=2, length=20)
    data["actions"] = [a[:15] for a in data["actions"]]
    assert len(TrajectoryDataset(data=data, horizon=4, history=2, anchor_key="actions")) == 2 * 15
    assert len(TrajectoryDataset(data=data, horizon=4, history=2, anchor_key="obs")) == 2 * 20


# ---------------------------------------------------------------------------
# trim_end vs drop_last_frames
# ---------------------------------------------------------------------------


def test_trim_end_drops_pivots_but_keeps_the_frames():
    """trim_end removes windows anchored on the last steps; those steps are still reachable
    as horizon context of earlier pivots. That distinction is the whole reason
    drop_last_frames exists separately."""
    plain = TrajectoryDataset(data=_data(n_episodes=1, length=20), horizon=4, history=2,
                              anchor_key="actions")
    trimmed = TrajectoryDataset(data=_data(n_episodes=1, length=20), horizon=4, history=2,
                                trim_end=5, anchor_key="actions")
    assert len(plain) - len(trimmed) == 5
    assert trimmed.data["actions"][0].shape[0] == 20  # frames untouched


def test_drop_last_frames_removes_the_frames_themselves():
    ds = TrajectoryDataset(data=_data(n_episodes=2, length=20), horizon=4, history=2,
                           drop_last_frames=6, anchor_key="actions")
    for key in ("actions", "obs"):
        assert all(ep.shape[0] == 14 for ep in ds.data[key])
    assert len(ds) == 2 * 14


def test_drop_last_frames_accepts_one_count_per_episode():
    """A fixed wall-clock tail spans a different number of frames per episode whenever the
    recording rate varied, so a scalar is the wrong shape for that case."""
    ds = TrajectoryDataset(data=_data(n_episodes=3, length=20), horizon=4, history=2,
                           drop_last_frames=[0, 5, 10], anchor_key="actions")
    assert [ep.shape[0] for ep in ds.data["actions"]] == [20, 15, 10]
    assert len(ds) == 20 + 15 + 10


def test_drop_last_frames_applies_to_every_modality():
    ds = TrajectoryDataset(data=_data(n_episodes=1, length=20), horizon=4, history=2,
                           drop_last_frames=[7], anchor_key="actions")
    assert ds.data["actions"][0].shape[0] == ds.data["obs"][0].shape[0] == 13


@pytest.mark.parametrize("bad", [[1, 2], [-1, 0, 0], [25, 0, 0]])
def test_drop_last_frames_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        TrajectoryDataset(data=_data(n_episodes=3, length=20), horizon=4, history=2,
                          drop_last_frames=bad, anchor_key="actions")


# ---------------------------------------------------------------------------
# geometry validation
# ---------------------------------------------------------------------------


def test_trim_end_that_empties_an_episode_raises():
    """Previously this produced a dataset that silently contained fewer episodes than it
    was given, or none at all."""
    with pytest.raises(ValueError, match="trim_end"):
        TrajectoryDataset(data=_data(n_episodes=2, length=20), horizon=4, history=2,
                          trim_end=20, anchor_key="actions")


def test_negative_trim_end_raises():
    with pytest.raises(ValueError, match="trim_end"):
        TrajectoryDataset(data=_data(), horizon=4, history=2, trim_end=-1)


def test_empty_data_raises():
    with pytest.raises(ValueError):
        TrajectoryDataset(data={}, horizon=4, history=2)
