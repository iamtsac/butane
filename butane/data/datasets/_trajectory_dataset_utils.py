import dataclasses
import math

import torch


def _default_pad_left(input_tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Pads the tensor on the left by repeating the first element."""
    return torch.cat([input_tensor[0:1].repeat_interleave(pad_len, dim=0), input_tensor], dim=0)


def _default_pad_right(input_tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Pads the tensor on the right by repeating the last element."""
    return torch.cat([input_tensor, input_tensor[-1:].repeat_interleave(pad_len, dim=0)], dim=0)


@dataclasses.dataclass(frozen=True)
class GoalSampler:
    """Where the goal frame of a goal-conditioned model comes from: the pivot frame itself, a later
    frame of the same episode, or any frame of the dataset. The three probabilities sum to 1.

    A tuple `source` reads every key at the same frame, e.g. a goal image and the goal state of
    one moment, and returns them as "<name>/<source>".
    """

    source: str | tuple[str, ...] = "observations"  # Key(s) the goal frame is read from
    p_current: float = 0.0  # The pivot frame itself
    p_trajectory: float = 1.0  # A later frame of the same episode
    p_random: float = 0.0  # Any frame of any episode
    geometric: bool = False  # Later frame: geometric offset (True) or uniform up to the episode end (False)
    discount: float = 0.99  # Geometric offsets follow Geometric(1 - discount), in [1, inf)

    def __post_init__(self):
        if not self.sources:
            raise ValueError("a goal needs at least one source key")
        probs = (self.p_current, self.p_trajectory, self.p_random)
        if min(probs) < 0 or not math.isclose(sum(probs), 1.0):
            raise ValueError(f"goal probabilities must be >= 0 and sum to 1, got {probs}")
        if self.geometric and not 0 < self.discount < 1:
            raise ValueError(f"geometric sampling needs 0 < discount < 1, got {self.discount}")

    @property
    def sources(self) -> tuple[str, ...]:
        return (self.source,) if isinstance(self.source, str) else tuple(self.source)

    def output_keys(self, name: str) -> dict[str, str]:
        """Sample key -> the stored key it is read from."""
        if isinstance(self.source, str):
            return {name: self.source}
        return {f"{name}/{source}": source for source in self.source}

    def sample(self, episode: int, t: int, lengths: torch.Tensor, starts: torch.Tensor) -> tuple[int, int]:
        """(episode, step) of a goal for pivot `t` of `episode`. Uses the global torch RNG, which the
        DataLoader seeds per worker - numpy's would hand every worker the same goals."""
        u = torch.rand(()).item()
        if u < self.p_current:
            return episode, t
        if u < self.p_current + self.p_trajectory:
            last = int(lengths[episode]) - 1
            if self.geometric:
                # Inverse CDF of Geometric(1 - discount) on [1, inf)
                offset = int(math.log(1.0 - torch.rand(()).item()) / math.log(self.discount)) + 1
                return episode, min(t + offset, last)
            return episode, int(torch.randint(min(t + 1, last), last + 1, ()))
        frame = int(torch.randint(int(lengths.sum()), ()))
        goal_episode = int(torch.searchsorted(starts, frame, right=True)) - 1
        return goal_episode, frame - int(starts[goal_episode])
