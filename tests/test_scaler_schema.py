import pytest
import torch

import butane


def generate_raw_trajectory_data() -> dict[str, list[torch.Tensor]]:
    """
    Generates raw, full-length sequential episode arrays.
    This simulates the raw data dictionary structure that your TrajectoryDataset
    takes in and processes internally using its history/horizon slicing loops.
    """
    # Simulate an episode containing 5 raw continuous frames of data
    # Features increase linearly to make the expected math easily trackable
    episode_1_qpos = torch.tensor(
        [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0], [5.0, -5.0]], dtype=torch.float32
    )

    episode_1_rgb = torch.ones((5, 3), dtype=torch.float32) * 255.0
    episode_1_actions = torch.tensor(
        [[-1.0, 1.0], [-0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, -1.0]], dtype=torch.float32
    )

    # Wrap them into lists of episodes, which is standard for trajectory maps
    return {
        "obs/qpos": [episode_1_qpos],
        "obs/rgb": [episode_1_rgb],
        "actions": [episode_1_actions],
    }


def generate_raw_static_tensor_data() -> dict[str, torch.Tensor]:
    """
    Generates plain, non-sequential data pools.
    This simulates the dataset structures designed for feed-forward models
    where each sample is a standalone vector instead of an episodic window.
    """
    return {
        "data": torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=torch.float32),
        "targets": torch.tensor([[-0.5, 0.5], [0.0, 0.0], [0.5, -0.5]], dtype=torch.float32),
    }


def test_scaler_schema_fit_with_real_datasets():
    raw_episodes = {
        "obs/qpos": [
            torch.tensor(
                [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0], [5.0, -5.0]],
                dtype=torch.float32,
            )
        ],
        "obs/rgb": [torch.ones((5, 3), dtype=torch.float32) * 255.0],
        "actions": [
            torch.tensor(
                [[-1.0, 1.0], [-0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, -1.0]],
                dtype=torch.float32,
            )
        ],
    }

    trajectory_dataset = butane.data.TrajectoryDataset(data=raw_episodes, history=2, horizon=2)

    dict_cfg = {
        "data": {
            "default": butane.data.DummyScaler,
            r"^obs(?!.*rgb).*$": butane.data.StandardScaler,
        },
        "targets": {"default": butane.data.DummyScaler, "actions": butane.data.MinMaxScaler},
    }

    traj_schema = butane.data.DecoupledScalerSchema(dict_cfg, sep="/")
    traj_schema.fit(trajectory_dataset, num_samples=10)

    assert "obs/qpos" in traj_schema.data_scalers
    assert "obs/rgb" not in traj_schema.data_scalers  # Excluded by regex rule
    assert "actions" in traj_schema.target_scalers

    assert list(traj_schema.data_scalers["obs/qpos"].mu.shape) == [1, 1, 2]
    assert list(traj_schema.target_scalers["actions"].xmin.shape) == [1, 1, 2]

    static_data = torch.tensor([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=torch.float32)
    static_targets = torch.tensor([[-0.5, 0.5], [0.0, 0.0], [0.5, -0.5]], dtype=torch.float32)

    tensor_dataset = butane.data.Dataset(data=static_data, targets=static_targets)

    tensor_cfg = {
        "data": {"default": butane.data.StandardScaler},
        "targets": {"default": butane.data.MinMaxScaler},
    }

    tensor_schema = butane.data.DecoupledScalerSchema(tensor_cfg, sep="/")
    tensor_schema.fit(tensor_dataset, num_samples=5)

    # Assertions for Static Setup
    assert "__root__" in tensor_schema.data_scalers
    assert "__root__" in tensor_schema.target_scalers

    # Static data skips extra view dims -> returns standard 2D arrays [1, features]
    assert list(tensor_schema.data_scalers["__root__"].mu.shape) == [1, 2]
    assert list(tensor_schema.target_scalers["__root__"].xmin.shape) == [1, 2]

    # Mean calculation validation of [1, 2, 3] equals exactly 2.0
    assert torch.allclose(tensor_schema.data_scalers["__root__"].mu, torch.tensor([[2.0, -2.0]]))


def test_scaler_schema_forward_and_reverse_execution():
    # =========================================================================
    # SETUP: Fit schemas quickly using mock dataset routines from previous step
    # =========================================================================
    raw_episodes = {
        "obs/qpos": [torch.tensor([[2.0, -2.0], [2.0, -2.0], [2.0, -2.0]], dtype=torch.float32)],
        "obs/rgb": [torch.ones((3, 3), dtype=torch.float32) * 255.0],
        "actions": [torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)],
    }
    trajectory_dataset = butane.data.TrajectoryDataset(data=raw_episodes, history=1, horizon=1)

    dict_cfg = {
        "data": {"default": butane.data.StandardScaler},
        "targets": {"default": butane.data.MinMaxScaler},
    }
    traj_schema = butane.data.DecoupledScalerSchema(dict_cfg, sep="/")
    traj_schema.fit(trajectory_dataset, num_samples=5)

    # =========================================================================
    # TEST 1: 3D SEQUENTIAL MODEL BATCHES [N, T, Features]
    # =========================================================================
    # A evaluation batch with Shape: [Batch=1, Time=1, Features=2]
    # Since dataset pool values were all exactly 2.0, mu is [[2.0, -2.0]].
    # Subtraction (2.0 - 2.0) should center this sample value right down to 0.0!
    traj_batch = {
        "data": {"obs/qpos": torch.tensor([[[2.0, -2.0]]], dtype=torch.float32)},
        "targets": {"actions": torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)},
        "unrelated_key": "stays_untouched",
    }

    # 1. Scaling the full batch container concurrently by passing context=None
    scaled_traj_batch = traj_schema.forward(traj_batch, context=None, inverse=False)

    # Assertions
    assert torch.allclose(scaled_traj_batch["data"]["obs/qpos"], torch.tensor([[[0.0, 0.0]]]))
    assert scaled_traj_batch["unrelated_key"] == "stays_untouched"

    # 2. Reversing the transformation back into raw physical properties
    descaled_traj_batch = traj_schema.forward(scaled_traj_batch, context=None, inverse=True)
    assert torch.allclose(descaled_traj_batch["data"]["obs/qpos"], traj_batch["data"]["obs/qpos"])
    assert torch.allclose(
        descaled_traj_batch["targets"]["actions"], traj_batch["targets"]["actions"]
    )

    # =========================================================================
    # TEST 2: 2D FLAT PURE TENSOR BATCHES [N, Features]
    # =========================================================================
    static_data = torch.tensor(
        [[1.0, -1.0], [3.0, -3.0]], dtype=torch.float32
    )  # Mean will be [2.0, -2.0]

    static_targets = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)

    tensor_dataset = butane.data.Dataset(data=static_data, targets=static_targets)

    tensor_schema = butane.data.DecoupledScalerSchema(dict_cfg, sep="/")
    tensor_schema.fit(tensor_dataset, num_samples=5)

    # This evaluation vector matches the computed mean exactly
    pure_tensor_input = torch.tensor([[2.0, -2.0]], dtype=torch.float32)

    # Apply forward pass to a raw vector using explicit context routing
    scaled_tensor = tensor_schema.forward(pure_tensor_input, context="data", inverse=False)

    assert torch.allclose(scaled_tensor, torch.tensor([[0.0, 0.0]]))

    # Restore raw vector values via reverse wrapper
    descaled_tensor = tensor_schema.forward(scaled_tensor, context="data", inverse=True)
    assert torch.allclose(descaled_tensor, pure_tensor_input)
