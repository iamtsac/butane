import h5py
import numpy as np
import pytest
import torch

from butane.data import TrajectoryDataset


@pytest.fixture
def list_dataset():
    """Generates a dataset from a list of variable-length 1D tensors."""
    data = []
    for _ in range(10):
        ep_len = torch.randint(20, 100, size=(1,)).item()
        data.append(torch.arange(ep_len).unsqueeze(-1).float())

    return TrajectoryDataset(data=data, horizon=16, history=2, align_start=True)


@pytest.fixture
def dict_dataset():
    data_dict = {"obs": [], "actions": []}
    for _ in range(10):
        ep_len = torch.randint(20, 100, size=(1,)).item()
        data_dict["obs"].append(torch.randn(ep_len, 3, 64, 64))  # Fake images
        data_dict["actions"].append(torch.randn(ep_len, 7))  # Fake joint actions

    return TrajectoryDataset(data=data_dict, horizon=8, history=4, align_start=False)

def test_list_initialization_and_shapes(list_dataset):
    assert len(list_dataset) > 0, "Dataset length should be greater than 0"

    sample = list_dataset[10]

    assert "data" in sample, "Missing 'data' root key"
    assert "targets" in sample, "Missing 'targets' root key"

    data_seq, target_seq = sample["data"], sample["targets"]

    assert data_seq["_internal"].shape[0] == 2, "History length mismatch"
    assert target_seq["_internal"].shape[0] == 16, "Horizon length mismatch"


def test_dict_initialization_and_shapes(dict_dataset):
    sample = dict_dataset[10]

    assert "data" in sample, "Missing 'data' root key"
    assert "targets" in sample, "Missing 'targets' root key"

    assert "obs" in sample["data"]
    assert "actions" in sample["targets"]

    assert sample["data"]["obs"].shape[0] == 4
    assert sample["targets"]["actions"].shape[0] == 8


def test_padding_edge_cases(dict_dataset):
    sample = dict_dataset[0]

    obs_history = sample["data"]["obs"]

    assert obs_history.shape[0] == 4

    assert torch.allclose(obs_history[0], obs_history[1]), "Left padding failed to replicate"


def test_episode_boundary_isolation():
    """Verifies that items sampled at episode seams do not leak cross-episode data."""
    # Two distinct episodes: Ep 0 is filled with 1.0, Ep 1 is filled with 2.0
    data = [
        torch.ones(20, 1),
        torch.ones(20, 1) * 2.0
    ]
    # history=3 (means 2 lookback steps), horizon=5
    dataset = TrajectoryDataset(data=data, horizon=5, history=3, align_start=False)

    # The last element of episode 0 is index 19
    sample = dataset[19]

    # The history loop can look back into its own episode
    assert torch.all(sample["data"]["_internal"] == 1.0)

    # The horizon reaches forward past index 19. Steps 20+ do not exist in Ep 0,
    # so it MUST pad using index 19's value (1.0), NOT bleed into Ep 1 (2.0).
    assert torch.all(sample["targets"]["_internal"] == 1.0), "Horizon leaked into the next episode!"

@pytest.mark.parametrize(
    "align_start, expected_horizon_start_val",
    [
        (True, 3.0),   # Mandated: start_idx = idx - (history - 1) -> 5 - 2 = 3
        (False, 5.0)   # Mandated: tracks from current idx directly -> 5
    ]
)
def test_temporal_alignment_modes(align_start, expected_horizon_start_val):
    """Validates that history and horizon anchors are correctly offset based on align_start."""
    # Single episode tracking an ascending counter matrix
    data = [torch.arange(30).unsqueeze(-1).float()]

    # history=3 meanslookback window size is 3 (steps: idx-2, idx-1, idx)
    dataset = TrajectoryDataset(data=data, horizon=5, history=3, align_start=align_start)
    sample = dataset[5]

    # Invariant: History sequence always terminates at the requested sample index
    assert sample["data"]["_internal"][-1].item() == 5.0

    # Invariant: Verify anchor shifts match specification layout matches exactly
    actual_horizon_start = sample["targets"]["_internal"][0].item()
    assert actual_horizon_start == expected_horizon_start_val, (
        f"Alignment mismatch! Expected horizon to start at {expected_horizon_start_val}, "
        f"but got {actual_horizon_start} using align_start={align_start}"
    )

def test_right_edge_padding(list_dataset):
    """Ensures right-edge padding triggers and clones the terminal step correctly."""
    # Grab the absolute final index of the entire dataset
    last_idx = len(list_dataset) - 1
    sample = list_dataset[last_idx]

    horizon_seq = sample["targets"]["_internal"]

    # Ensure it didn't crash and filled out the whole requested horizon
    assert horizon_seq.shape[0] == list_dataset.horizon

    # The trailing elements should be exactly identical to the last legitimate step
    assert torch.allclose(horizon_seq[-1], horizon_seq[-2]), "Right padding failed to clone terminal frame"

def load_h5_data(group: h5py.Group) -> dict[str, np.ndarray]:
    """Helper utility to parse h5 groups into standard memory arrays."""
    return {key: group[key][()] for key in group.keys()}


def test_h5_ingestion_data_integrity(tmp_path):
    """Checks that numpy arrays parsed out of H5 files retain precision and map perfectly."""
    fpath = tmp_path / "test_data.h5"
    with h5py.File(fpath, "w") as f:
        traj = f.create_group("traj_0")
        # Creating a predictable sequence matrix
        traj.create_dataset("obs", data=np.expand_dims(np.arange(10), axis=-1))
        traj.create_dataset("actions", data=np.expand_dims(np.arange(10), axis=-1) * 10)

    with h5py.File(fpath, "r") as f:
        parsed_data = {"obs": [], "actions": []}
        for k in f.keys():
            traj_data = load_h5_data(f[k])
            parsed_data["obs"].append(traj_data["obs"])
            parsed_data["actions"].append(traj_data["actions"])

    dataset = TrajectoryDataset(data=parsed_data, horizon=2, history=2)
    sample = dataset[4]

    # Check manual math matching: index 4 should extract obs value 4
    assert sample["data"]["obs"][-1].item() == 4.0
    assert sample["targets"]["actions"][0].item() == 40.0

def test_variable_length_h5_ingestion_integrity(tmp_path):
    """Verifies indexing precision and boundaries with multi-trajectory variable-length data."""
    fpath = tmp_path / "test_variable_data.h5"

    # Define distinct episode lengths
    ep0_len = 8   # Steps 0 to 7
    ep1_len = 12  # Steps 8 to 19 (Total dataset steps = 20)

    with h5py.File(fpath, "w") as f:
        # Trajectory 0: Length 8
        traj0 = f.create_group("traj_0")
        traj0.create_dataset("obs", data=np.expand_dims(np.arange(ep0_len), axis=-1).astype(np.float32))
        traj0.create_dataset("actions", data=np.expand_dims(np.arange(ep0_len), axis=-1).astype(np.float32) * 10.0)

        # Trajectory 1: Length 12
        traj1 = f.create_group("traj_1")
        traj1.create_dataset("obs", data=np.expand_dims(np.arange(ep1_len), axis=-1).astype(np.float32))
        traj1.create_dataset("actions", data=np.expand_dims(np.arange(ep1_len), axis=-1).astype(np.float32) * 10.0)

    # Ingest datasets from storage
    with h5py.File(fpath, "r") as f:
        parsed_data = {"obs": [], "actions": []}
        # Explicit sorting ensures predictable trajectory sequencing (traj_0 then traj_1)
        for k in sorted(f.keys()):
            traj_data = load_h5_data(f[k])
            parsed_data["obs"].append(traj_data["obs"])
            parsed_data["actions"].append(traj_data["actions"])

    # Define unified structural configs for our multi-modal initialization schema
    print(parsed_data)
    dataset = TrajectoryDataset(
        data=parsed_data,
        align_start=False,
        horizon=4,
        history=2,
    )

    # Invariant: Total length must equal the exact sum of individual steps
    assert len(dataset) == ep0_len + ep1_len  # 8 + 12 = 20

    # -------------------------------------------------------------------------
    # Scenario A: Sample a standard step inside Trajectory 0
    # -------------------------------------------------------------------------
    sample_mid_ep0 = dataset[4]
    # Obs history length config = 2 (looks back at step 3 and step 4)
    assert sample_mid_ep0["data"]["obs"].shape[0] == 2
    assert sample_mid_ep0["data"]["obs"][-1].item() == 4.0

    # Actions target horizon config = 4 (looks forward from step 4: steps 4, 5, 6, 7)
    assert sample_mid_ep0["targets"]["actions"].shape[0] == 4
    assert sample_mid_ep0["targets"]["actions"][0].item() == 40.0

    # -------------------------------------------------------------------------
    # Scenario B: Sample the boundary terminal step of Trajectory 0 (Index 7)
    # -------------------------------------------------------------------------
    sample_edge_ep0 = dataset[7]

    # Horizon for actions is size 4. Looking forward from step 7 means steps [7, 8, 9, 10].
    # But step 7 is the END of Trajectory 0! It MUST NOT bleed into Trajectory 1's data.
    # It should right-pad by replicating step 7's action value (70.0) across the tail.
    actions_horizon = sample_edge_ep0["targets"]["actions"]
    assert actions_horizon.shape[0] == 4
    assert actions_horizon[0].item() == 70.0
    assert torch.all(actions_horizon == 70.0), "Boundary isolation broken! Data leaked into Trajectory 1."

    # -------------------------------------------------------------------------
    # Scenario C: Sample the initial step of Trajectory 1 (Global Index 8)
    # -------------------------------------------------------------------------
    sample_start_ep1 = dataset[8]

    # Obs history config is size 2. Looking back from Trajectory 1's first frame (0.0)
    # means it must left-pad using its own initial state value (0.0), NOT touch Trajectory 0.
    obs_history = sample_start_ep1["data"]["obs"]
    assert obs_history.shape[0] == 2
    assert torch.all(obs_history == 0.0), "Boundary isolation broken! Lookback leaked into Trajectory 0."

    # Verify the forward horizon maps to the local coordinates of Trajectory 1
    assert sample_start_ep1["targets"]["actions"][0].item() == 0.0
    assert sample_start_ep1["targets"]["actions"][1].item() == 10.0
