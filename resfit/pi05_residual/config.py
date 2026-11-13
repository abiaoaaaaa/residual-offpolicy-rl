from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Pi05ResidualConfig:
    pi05_checkpoint: str
    task: str
    pi05_config: str = ""
    dataset_stats: dict | None = None
    cameras: tuple[str, str, str] = (
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    state_dim: int = 64
    raw_state_dim: int = 16
    action_dim: int = 16
    chunk_size: int = 30
    rl_chunk_size: int = 10
    device: str = "cuda"
    dtype: str = "bfloat16"
    num_inference_steps: int = 10
    image_size: int = 224
    rl_image_size: int = 128
    min_state_std: float = 0.1
    residual_joint_limit_rad: float = math.radians(5.0)
    residual_ema_alpha: float = 0.35
    residual_slew_limit_rad: float = math.radians(1.0)
    action_lowpass_alpha: float = 1.0
    bridge_joint_step_limit_rad: float = 0.08
    enable_gripper_residual: bool = False
    action_low: float = -1.0
    action_high: float = 1.0
    replay_capacity: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    learning_starts: int = 1_000
    total_timesteps: int = 100_000
    updates_per_step: int = 1
    max_episode_steps: int = 300
    rl_cameras: list[str] = field(default_factory=list)
    # Measured 30Hz per-frame delta maxima. Three frames form one 10Hz step.
    joint_delta_limit: tuple[float, ...] = (
        0.195938, 0.065462, 0.134875, 0.421450, 0.232409, 0.306070,
        0.438561, 0.128781, 0.101106, 0.118625, 0.204251, 0.527576,
        0.261271, 0.791527, 0.959146, 1.041155,
    )

    def __post_init__(self):
        # Small synthetic tests use state_dim as their complete raw state.
        if self.state_dim not in (64, 16) and self.raw_state_dim == 16:
            object.__setattr__(self, "raw_state_dim", self.state_dim)
        if len(self.cameras) != 3:
            raise ValueError("pi0.5 residual requires exactly three camera keys")
        if self.action_dim <= 0 or self.state_dim <= 0 or self.raw_state_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if len(self.joint_delta_limit) != self.action_dim:
            raise ValueError("joint_delta_limit length must equal action_dim")
        if self.state_dim not in (self.raw_state_dim, self.raw_state_dim + 48):
            raise ValueError("state_dim must be raw_state_dim or raw_state_dim + 48 context dims")
        if self.chunk_size != 30 or self.rl_chunk_size != 10:
            raise ValueError("Pi0.5 residual uses a 30-action chunk and 10 RL actions")
        if self.residual_joint_limit_rad <= 0:
            raise ValueError("residual_joint_limit_rad must be positive")
        if not 0.0 < self.residual_ema_alpha <= 1.0:
            raise ValueError("residual_ema_alpha must be in (0, 1]")
        if self.residual_slew_limit_rad <= 0:
            raise ValueError("residual_slew_limit_rad must be positive")
        if not 0.0 < self.action_lowpass_alpha <= 1.0:
            raise ValueError("action_lowpass_alpha must be in (0, 1]")
        if self.bridge_joint_step_limit_rad <= 0:
            raise ValueError("bridge_joint_step_limit_rad must be positive")
        if self.residual_joint_limit_rad > min(self.rl_joint_delta_limit[: min(14, self.action_dim)]):
            raise ValueError("residual joint limit exceeds the normalized action range")

    @property
    def rl_joint_delta_limit(self) -> tuple[float, ...]:
        return tuple(x * (self.chunk_size // self.rl_chunk_size) for x in self.joint_delta_limit)
