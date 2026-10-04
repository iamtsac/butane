"""Goal sampling for goal-conditioned models on TrajectoryDataset."""

import numpy as np
import pytest
import torch

from butane.data.datasets.trajectory_dataset import GoalSampler, TrajectoryDataset


def _data(n_episodes=4, length=30):
    """Every "obs" frame is (episode, step), so a goal says exactly where it was read from."""
    return {
        "actions": [np.zeros((length, 2), dtype=np.float32) for _ in range(n_episodes)],
        "obs": [np.stack([np.full(length, i), np.arange(length)], axis=1).astype(np.float32)
                for i in range(n_episodes)],
    }


def _goals(ds, n=None):
    """(pivot episode, pivot step, goal episode, goal step) for the first `n` windows."""
    rows = []
    for idx in range(len(ds) if n is None else n):
        sample = ds[idx]
        pivot = sample["data"]["obs"][-1]
        goal = sample["data"]["goal"][0]
        rows.append((*pivot.tolist(), *goal.tolist()))
    return np.array(rows, dtype=int)


def test_no_goals_leaves_samples_unchanged():
    ds = TrajectoryDataset(data=_data(), horizon=4, history=2)
    assert set(ds[0]["data"]) == {"actions", "obs"}


def test_goal_is_one_frame_with_a_mask():
    torch.manual_seed(0)
    ds = TrajectoryDataset(data=_data(), horizon=4, history=3, goals={"goal": GoalSampler(source="obs")})
    sample = ds.__getitem__(5, return_pad_masks=True)
    assert sample["data"]["goal"].shape == (1, 2)
    assert sample["data_masks"]["goal"].tolist() == [True]
    assert "goal" not in sample["targets"]
    assert ds.sizes()["data"]["goal"].tolist() == [1, 2]
    assert ds.collect_samples(50)["data"]["goal"].shape == (50, 1, 2)


def test_current_goal_is_the_pivot_frame():
    ds = TrajectoryDataset(data=_data(), horizon=4, history=2,
                           goals={"goal": GoalSampler(source="obs", p_current=1.0, p_trajectory=0.0)})
    rows = _goals(ds)
    assert (rows[:, :2] == rows[:, 2:]).all()


def test_uniform_trajectory_goal_is_a_later_frame_of_the_same_episode():
    torch.manual_seed(0)
    length = 30
    ds = TrajectoryDataset(data=_data(length=length), horizon=4, history=2,
                           goals={"goal": GoalSampler(source="obs")})
    rows = np.concatenate([_goals(ds) for _ in range(5)])
    assert (rows[:, 0] == rows[:, 2]).all()
    last = rows[:, 1] == length - 1
    assert (rows[~last, 3] > rows[~last, 1]).all() and (rows[last, 3] == length - 1).all()
    # Uniform up to the end: from the first pivot every later step shows up, about equally often
    steps = np.array([int(ds[0]["data"]["goal"][0, 1]) for _ in range(2900)])
    counts = np.bincount(steps, minlength=length)
    assert counts[0] == 0 and (counts[1:] > 50).all()


def test_geometric_trajectory_goal_offsets_follow_the_discount():
    torch.manual_seed(0)
    discount = 0.8
    ds = TrajectoryDataset(data=_data(n_episodes=2, length=400), horizon=4, history=1,
                           goals={"goal": GoalSampler(source="obs", geometric=True, discount=discount)})
    rows = np.concatenate([_goals(ds, n=200) for _ in range(20)])  # pivots far from the episode end
    offsets = rows[:, 3] - rows[:, 1]
    assert offsets.min() >= 1
    assert abs(offsets.mean() - 1 / (1 - discount)) < 0.25


def test_mixture_frequencies():
    torch.manual_seed(0)
    n_episodes = 20
    sampler = GoalSampler(source="obs", p_current=0.2, p_trajectory=0.5, p_random=0.3)
    ds = TrajectoryDataset(data=_data(n_episodes=n_episodes, length=50), horizon=4, history=1,
                           goals={"goal": sampler})
    rows = np.concatenate([_goals(ds) for _ in range(3)])
    other_episode = (rows[:, 0] != rows[:, 2]).mean()
    current = ((rows[:, 0] == rows[:, 2]) & (rows[:, 1] == rows[:, 3])).mean()
    # A random goal lands in another episode (n-1)/n of the time, and on the pivot itself almost never
    assert abs(other_episode - 0.3 * (n_episodes - 1) / n_episodes) < 0.02
    assert abs(current - 0.2) < 0.02


def test_split_keeps_goals_inside_each_side():
    torch.manual_seed(0)
    ds = TrajectoryDataset(data=_data(n_episodes=6), horizon=4, history=2,
                           goals={"goal": GoalSampler(source="obs", p_trajectory=0.0, p_random=1.0)})
    held_out = ds.split(0.5)
    for side in (ds, held_out):
        episodes = {int(ep[0, 0]) for ep in side.data["obs"]}
        rows = np.concatenate([_goals(side) for _ in range(3)])
        assert set(rows[:, 2]) == episodes
    assert held_out.goals == ds.goals


def test_goal_source_must_exist():
    with pytest.raises(KeyError, match="dropped by a filter"):
        TrajectoryDataset(data=_data(), horizon=4, history=2, goals={"goal": GoalSampler(source="missing")})


def test_goal_name_cannot_shadow_a_stored_key():
    with pytest.raises(ValueError, match="shadow"):
        TrajectoryDataset(data=_data(), horizon=4, history=2, goals={"obs": GoalSampler(source="obs")})


@pytest.mark.parametrize("kwargs", [
    dict(p_current=0.5, p_trajectory=0.6),
    dict(p_current=-0.1, p_trajectory=1.1),
    dict(geometric=True, discount=1.0),
])
def test_invalid_sampler(kwargs):
    with pytest.raises(ValueError):
        GoalSampler(**kwargs)


def _two_sources(n_episodes=4, length=30):
    """"state" is (episode, step) and "image" encodes the same (episode, step), so both say where they came from."""
    data = _data(n_episodes, length)
    data["state"] = data.pop("obs")
    data["image"] = [(s[:, 0] * 1000 + s[:, 1])[:, None, None, None].repeat(4, axis=2).repeat(4, axis=3)
                     for s in data["state"]]  # (T, 1, 4, 4)
    return data


def test_tuple_source_reads_every_key_at_the_same_frame():
    torch.manual_seed(0)
    sampler = GoalSampler(source=("image", "state"), p_current=0.2, p_trajectory=0.5, p_random=0.3)
    ds = TrajectoryDataset(data=_two_sources(), horizon=4, history=2, goals={"goal": sampler})
    for idx in range(len(ds)):
        sample = ds.__getitem__(idx, return_pad_masks=True)
        image, state = sample["data"]["goal/image"], sample["data"]["goal/state"]
        assert image.shape == (1, 1, 4, 4) and state.shape == (1, 2)
        assert image[0, 0, 0, 0] == state[0, 0] * 1000 + state[0, 1]
        assert sample["data_masks"]["goal/image"].tolist() == sample["data_masks"]["goal/state"].tolist() == [True]
    assert "goal" not in ds[0]["data"]


def test_separate_goals_are_drawn_independently():
    torch.manual_seed(0)
    ds = TrajectoryDataset(data=_two_sources(), horizon=4, history=2,
                           goals={"goal/image": GoalSampler(source="image"), "goal/state": GoalSampler(source="state")})
    matched = [
        float(s["data"]["goal/image"][0, 0, 0, 0]) == float(s["data"]["goal/state"][0, 0] * 1000 + s["data"]["goal/state"][0, 1])
        for s in (ds[i] for i in range(len(ds)))
    ]
    assert not all(matched)


def test_tuple_sources_must_line_up():
    data = _two_sources()
    data["image"][1] = data["image"][1][:-3]
    with pytest.raises(ValueError, match="differ in length"):
        TrajectoryDataset(data=data, horizon=4, history=2, goals={"goal": GoalSampler(source=("image", "state"))})


def test_goal_keys_cannot_collide():
    with pytest.raises(ValueError, match="both produce"):
        TrajectoryDataset(data=_two_sources(), horizon=4, history=2, goals={
            "goal": GoalSampler(source=("image", "state")),
            "goal/state": GoalSampler(source="state"),
        })


def test_tuple_goals_survive_split():
    torch.manual_seed(0)
    ds = TrajectoryDataset(data=_two_sources(n_episodes=6), horizon=4, history=2,
                           goals={"goal": GoalSampler(source=("image", "state"), p_trajectory=0.0, p_random=1.0)})
    held_out = ds.split(0.5)
    sample = held_out[0]["data"]
    assert sample["goal/image"][0, 0, 0, 0] == sample["goal/state"][0, 0] * 1000 + sample["goal/state"][0, 1]


def test_empty_tuple_source():
    with pytest.raises(ValueError, match="at least one"):
        GoalSampler(source=())
