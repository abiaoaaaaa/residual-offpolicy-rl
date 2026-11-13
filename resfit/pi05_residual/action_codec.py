"""Convert OpenPI absolute joint chunks to normalized frame deltas."""

from __future__ import annotations

import torch


class ResidualActionFilter:
    """EMA plus a per-step slew limit for physical residual corrections."""

    def __init__(self, alpha: float, slew_limit_rad: float, dimensions: int = 14):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("residual EMA alpha must be in (0, 1]")
        if slew_limit_rad <= 0.0:
            raise ValueError("residual slew limit must be positive")
        self.alpha = float(alpha)
        self.slew_limit_rad = float(slew_limit_rad)
        self.dimensions = int(dimensions)
        self.previous = torch.zeros(self.dimensions, dtype=torch.float32)

    def reset(self, correction: torch.Tensor | None = None) -> None:
        self.previous.zero_()
        if correction is not None:
            self.sync(correction)

    def sync(self, correction: torch.Tensor) -> None:
        value = torch.as_tensor(correction, dtype=torch.float32).reshape(-1)
        if value.numel() < self.dimensions or not torch.isfinite(value[: self.dimensions]).all():
            raise ValueError("residual correction must contain finite arm dimensions")
        self.previous.copy_(value[: self.dimensions])

    def __call__(self, correction: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(correction, dtype=torch.float32).reshape(-1)
        if value.numel() < self.dimensions or not torch.isfinite(value[: self.dimensions]).all():
            raise ValueError("residual correction must contain finite arm dimensions")
        target = value[: self.dimensions]
        filtered = self.previous + self.alpha * (target - self.previous)
        filtered = self.previous + torch.clamp(
            filtered - self.previous, -self.slew_limit_rad, self.slew_limit_rad
        )
        self.previous.copy_(filtered)
        return filtered


class JointDeltaCodec:
    def __init__(self, limits: tuple[float, ...]):
        self.limits = torch.tensor(limits, dtype=torch.float32)
        if torch.any(self.limits <= 0):
            raise ValueError("joint delta limits must be positive")

    def encode_chunk(
        self, current_state: torch.Tensor, absolute_chunk: torch.Tensor
    ) -> torch.Tensor:
        """[B,H,A] absolute targets -> [B,H,A] normalized frame deltas."""
        current_state = current_state[..., : absolute_chunk.shape[-1]].to(
            absolute_chunk.device
        )
        previous = torch.cat((current_state[:, None], absolute_chunk[:, :-1]), dim=1)
        limits = self.limits.to(absolute_chunk.device)
        return torch.clamp((absolute_chunk - previous) / limits, -1.0, 1.0)

    def decode_step(
        self, current_state: torch.Tensor, normalized_delta: torch.Tensor
    ) -> torch.Tensor:
        limits = self.limits.to(normalized_delta.device)
        current_state = current_state[..., : normalized_delta.shape[-1]].to(
            normalized_delta.device
        )
        return current_state + normalized_delta * limits
