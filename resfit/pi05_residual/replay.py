"""Replay helpers, including a no-C++ prioritized sampler fallback."""

from __future__ import annotations

from pathlib import Path

import torch
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class PythonPrioritizedSampler(PrioritizedSampler):
    """PrioritizedSampler compatible fallback for mismatched TorchRL wheels."""

    def __init__(self, max_capacity: int, alpha: float, beta: float, eps: float):
        self._max_capacity = max_capacity
        self._alpha = alpha
        self._beta = beta
        self._eps = eps
        self._rng = None
        self.reduction = "max"
        self.dtype = torch.float32
        self._priorities = torch.zeros(max_capacity, dtype=torch.float32)
        self._max_priority = (None, None)

    @property
    def default_priority(self) -> float:
        value = self._max_priority[0]
        return 1.0 if value is None else float((value + self._eps) ** self._alpha)

    def add(self, index) -> None:
        self._priorities[torch.as_tensor(index, dtype=torch.long)] = self.default_priority

    def extend(self, index) -> None:
        self.add(index)

    def sample(self, storage, batch_size: int):
        size = len(storage)
        if not size:
            raise RuntimeError("cannot sample an empty replay buffer")
        powered = self._priorities[:size]
        if not torch.any(powered > 0):
            powered.fill_(1.0)
        probabilities = powered / powered.sum()
        index = torch.multinomial(
            probabilities, batch_size, replacement=True, generator=self._rng
        )
        weights = (size * probabilities[index]).pow(-self._beta)
        weights /= weights.max()
        return index, {"_weight": weights}

    @torch.no_grad()
    def update_priority(self, index, priority, *, storage=None) -> None:
        index = torch.as_tensor(index, dtype=torch.long, device="cpu").reshape(-1)
        priority = torch.as_tensor(
            priority, dtype=torch.float32, device="cpu"
        ).reshape(-1)
        if priority.numel() == 1:
            priority = priority.expand_as(index)
        raw_max, offset = priority.max(dim=0)
        if self._max_priority[0] is None or raw_max > self._max_priority[0]:
            self._max_priority = (raw_max, index[offset])
        self._priorities[index] = (priority + self._eps).pow(self._alpha)

    def state_dict(self):
        return {
            "priorities": self._priorities.clone(),
            "max_priority": self._max_priority,
        }

    def load_state_dict(self, state_dict) -> None:
        self._priorities.copy_(state_dict["priorities"])
        self._max_priority = state_dict["max_priority"]

    def _empty(self) -> None:
        self._priorities.zero_()
        self._max_priority = (None, None)

    def dumps(self, path) -> None:
        torch.save(self.state_dict(), Path(path) / "python_prioritized_sampler.pt")

    def loads(self, path) -> None:
        self.load_state_dict(
            torch.load(Path(path) / "python_prioritized_sampler.pt", weights_only=False)
        )
