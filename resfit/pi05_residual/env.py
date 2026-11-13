from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import time

from .config import Pi05ResidualConfig
from .action_codec import JointDeltaCodec, ResidualActionFilter
from .policy import Pi05BasePolicy
from .observation import CONTEXT_DIM, residual_context


class Pi05ResidualEnv:
    """Adapter for any Gym-like env exposing three RGB views and state.

    The wrapped env must return a dict with the configured camera keys and
    ``observation.state``. Images may be HWC uint8 or CHW float tensors.
    """

    def __init__(self, env: Any, policy: Pi05BasePolicy, cfg: Pi05ResidualConfig):
        self.env, self.policy, self.cfg = env, policy, cfg
        self.codec = JointDeltaCodec(cfg.rl_joint_delta_limit)
        self._residual_filter = ResidualActionFilter(
            cfg.residual_ema_alpha if cfg.action_dim >= 14 else 1.0,
            cfg.residual_slew_limit_rad if cfg.action_dim >= 14 else float("inf"),
            dimensions=min(14, cfg.action_dim),
        )
        self._context = np.zeros(CONTEXT_DIM, dtype=np.float32)
        self._previous_executed_action = None
        self._base_action = None
        self._base_chunk = None
        self._base_absolute_chunk = None
        self._base_reference_chunk = None
        self._chunk_pos = 0
        self._last_base_inference_seconds = 0.0
        self._last_prefetch_wait_seconds = 0.0
        self._last_filtered_action = None
        self._prefetch = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05-prefetch")

    def _infer_chunk(self, pi_obs: dict[str, torch.Tensor]):
        started = time.perf_counter()
        chunk = self.policy.action_chunk(pi_obs)
        return (
            chunk,
            time.perf_counter() - started,
            pi_obs["observation.state"].clone(),
        )

    def _install_chunk(
        self,
        absolute_chunk: torch.Tensor,
        inference_state: torch.Tensor,
    ) -> None:
        if absolute_chunk.shape[1] != self.cfg.chunk_size:
            raise ValueError(f"Pi0.5 must return {self.cfg.chunk_size} actions")
        stride = self.cfg.chunk_size // self.cfg.rl_chunk_size
        selected = absolute_chunk[:, stride - 1 :: stride]
        if selected.shape[-1] != self.cfg.action_dim or not torch.isfinite(selected).all():
            raise ValueError("Pi0.5 returned an invalid absolute joint chunk")
        self._base_absolute_chunk = selected.detach().cpu()
        self._base_reference_chunk = torch.cat(
            (inference_state[:, None, : self.cfg.action_dim], selected[:, :-1]), dim=1
        ).detach().cpu()
        self._base_chunk = self.codec.encode_chunk(
            inference_state, selected
        ).detach().cpu()
        self._chunk_pos = 0

    def _tensor_obs(self, raw: dict[str, Any], image_size: int) -> dict[str, torch.Tensor]:
        out = {"observation.state": torch.as_tensor(raw["observation.state"]).float().reshape(1, -1)}
        for key in self.cfg.cameras:
            image = torch.as_tensor(raw[key])
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            elif image.ndim == 4 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            was_uint8 = image.dtype == torch.uint8
            if was_uint8:
                image = image.float() / 255.0
            image = image.unsqueeze(0) if image.ndim == 3 else image
            h, w = image.shape[-2:]
            size = image_size
            scale = min(size / h, size / w)
            resized = F.interpolate(
                image,
                size=(max(1, round(h * scale)), max(1, round(w * scale))),
                mode="bilinear",
                align_corners=False,
            )
            pad_h, pad_w = size - resized.shape[-2], size - resized.shape[-1]
            image = F.pad(
                resized,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
            out[key] = (image * 255.0).round().to(torch.uint8) if was_uint8 else image
        return out

    def _obs(self, raw: dict[str, Any], *, refresh_base: bool = False) -> dict[str, torch.Tensor]:
        pi_obs = self._tensor_obs(raw, self.cfg.image_size)
        obs = self._tensor_obs(raw, self.cfg.rl_image_size)
        self._current_state = pi_obs["observation.state"].clone()
        if refresh_base or self._base_chunk is None or self._chunk_pos >= self.cfg.rl_chunk_size:
            wait_started = time.perf_counter()
            if self._prefetch is None:
                absolute_chunk, inference_s, inference_state = self._infer_chunk(pi_obs)
                prefetch_wait_s = 0.0
            else:
                absolute_chunk, inference_s, inference_state = self._prefetch.result()
                prefetch_wait_s = time.perf_counter() - wait_started
                self._prefetch = None
            self._last_base_inference_seconds = inference_s
            self._last_prefetch_wait_seconds = prefetch_wait_s
            self._install_chunk(absolute_chunk, inference_state)
        elif (
            self._chunk_pos == self.cfg.rl_chunk_size - 2
            and self._prefetch is None
        ):
            # Two 100ms actions remain, enough to hide the measured ~90ms
            # warm Pi0.5 inference while preserving a recent observation.
            self._prefetch = self._executor.submit(self._infer_chunk, pi_obs)
        self._base_action = self._base_chunk[:, self._chunk_pos].clone()
        self._base_absolute_action = self._base_absolute_chunk[:, self._chunk_pos].clone()
        self._base_reference_action = self._base_reference_chunk[:, self._chunk_pos].clone()
        state_mean = getattr(self.policy, "state_mean", torch.zeros(self.cfg.raw_state_dim))
        state_std = getattr(self.policy, "state_std", torch.ones(self.cfg.raw_state_dim)).clamp_min(self.cfg.min_state_std)
        state = (obs["observation.state"] - state_mean) / state_std
        if self.cfg.state_dim == self.cfg.raw_state_dim + CONTEXT_DIM:
            state = torch.cat([state, torch.as_tensor(self._context).reshape(1, -1)], dim=-1)
        obs["observation.state"] = state
        obs["observation.base_action"] = self._base_action.clone()
        return obs

    def reset(self, **kwargs):
        if self._prefetch is not None:
            self._prefetch.result()
            self._prefetch = None
        raw, info = self.env.reset(**kwargs)
        self._base_chunk = None
        self._base_absolute_chunk = None
        self._base_reference_chunk = None
        self._last_filtered_action = torch.as_tensor(
            raw["observation.state"], dtype=torch.float32
        ).reshape(-1).numpy().copy()
        self._residual_filter.reset()
        self._context.fill(0.0)
        self._previous_executed_action = self._last_filtered_action.copy()
        obs = self._obs(raw, refresh_base=True)
        # Reset performs a blocking base-policy inference, but there is no
        # control tick to attribute that latency to.
        self._last_base_inference_seconds = 0.0
        return obs, info

    def step(self, residual_action: torch.Tensor):
        if self._base_action is None:
            raise RuntimeError("reset() must be called before step()")
        # Actor.forward returns a per-joint normalized residual corresponding
        # to a maximum physical correction of five degrees.
        residual = torch.as_tensor(residual_action).reshape(1, -1).float()
        if residual.shape[-1] != self.cfg.action_dim or not torch.isfinite(residual).all():
            raise ValueError("residual_action must have finite values for every action dimension")
        joint_dim = min(14, self.cfg.action_dim)
        normalized_limit = self.cfg.residual_joint_limit_rad / self.codec.limits[:joint_dim]
        if torch.any(residual[:, :joint_dim].abs() > normalized_limit + 1e-6):
            raise ValueError(
                "residual_action exceeds the five-degree per-joint contract"
            )
        if not self.cfg.enable_gripper_residual and residual.shape[-1] == 16:
            residual = residual.clone()
            residual[:, [14, 15]] = 0.0
        combined_delta = torch.clamp(self._base_action + residual, -1.0, 1.0)
        correction = (combined_delta - self._base_action) * self.codec.limits
        filtered_correction = self._residual_filter(correction[0])
        raw_action = (self._base_absolute_action[0] + correction[0]).numpy()
        action = raw_action.copy()
        action[:14] = (self._base_absolute_action[0, :14] + filtered_correction).numpy()
        if self.cfg.action_lowpass_alpha < 1.0:
            alpha = self.cfg.action_lowpass_alpha
            action[:14] = (
                (1.0 - alpha) * self._last_filtered_action[:14]
                + alpha * action[:14]
            )
        base_reference_action = self._base_reference_action.clone()
        self._chunk_pos += 1
        raw_next, reward, terminated, truncated, info = self.env.step(action)
        if info.get("execution", "active") == "active":
            self._last_filtered_action = action.copy()
        info = dict(info)
        executed_delta = self._base_action + torch.as_tensor(
            (action - self._base_absolute_action[0].numpy()) / self.codec.limits.numpy(),
            dtype=torch.float32,
        ).reshape(1, -1)
        executed_scaled = torch.clamp(executed_delta, -1.0, 1.0)
        executed_residual = executed_scaled - self._base_action
        info["proposed_scaled_action"] = combined_delta.clone()
        # Replay action is the actor proposal; execution details are separate.
        info["scaled_action"] = combined_delta.clone()
        info["residual_action"] = residual.clone()
        info["executed_residual_action"] = executed_residual.clone()
        info["residual_physical_rad"] = correction.clone()
        info["filtered_residual_physical_rad"] = filtered_correction.clone()
        info["absolute_action"] = action.copy()
        info["raw_absolute_action"] = raw_action.copy()
        info["base_reference_action"] = base_reference_action.numpy()[0].copy()
        if self.cfg.state_dim == self.cfg.raw_state_dim + CONTEXT_DIM:
            limits = np.asarray(self.cfg.rl_joint_delta_limit, dtype=np.float32)
            self._context = residual_context(
                executed_residual.numpy()[0],
                action - self._previous_executed_action,
                residual.numpy()[0] - executed_residual.numpy()[0],
                limits,
            )
            self._previous_executed_action = action.copy()
        next_obs = self._obs(raw_next)
        # The base chunk is refreshed while constructing the next observation,
        # so report that refresh on the transition that consumed the old
        # chunk.  This keeps timing diagnostics aligned with replay ``next``.
        base_inference_seconds = self._last_base_inference_seconds
        info["base_inference_seconds"] = base_inference_seconds
        info["base_prefetch_wait_seconds"] = self._last_prefetch_wait_seconds
        info["chunk_position"] = self._chunk_pos
        info["chunk_refreshed"] = base_inference_seconds > 0.0
        self._last_base_inference_seconds = 0.0
        self._last_prefetch_wait_seconds = 0.0
        return next_obs, reward, terminated, truncated, info

    def normalize_executed_action(
        self, executed_action: np.ndarray, reference: np.ndarray | None = None
    ) -> np.ndarray:
        executed = np.asarray(executed_action, dtype=np.float32).reshape(-1)
        if executed.shape != (self.cfg.action_dim,) or not np.isfinite(executed).all():
            raise ValueError("intervention requires a finite executed absolute action")
        if reference is None:
            reference = self._base_reference_action.numpy()[0]
        normalized = (executed - reference) / self.codec.limits.numpy()
        if np.max(np.abs(normalized)) > 1.0 + 1e-5:
            raise ValueError("executed human action exceeds critic's normalized action range")
        return normalized.astype(np.float32)

    def observe_intervention(self, raw_next: dict[str, Any], executed_action: np.ndarray | None):
        """Advance the Pi reference clock while VR, not Pi, owns the robot."""
        reference = self._base_reference_action.numpy()[0].copy()
        normalized = None
        if executed_action is not None:
            normalized = self.normalize_executed_action(executed_action, reference)
            self._last_filtered_action = np.asarray(executed_action, dtype=np.float32).copy()
            if self.cfg.state_dim == self.cfg.raw_state_dim + CONTEXT_DIM:
                limits = np.asarray(self.cfg.rl_joint_delta_limit, dtype=np.float32)
                # There is no actor proposal while VR owns the robot.  Encode
                # that explicitly as proposal=0, so the next state signals a
                # takeover instead of reusing stale policy context.
                self._context = residual_context(
                    normalized,
                    np.asarray(executed_action, dtype=np.float32) - self._previous_executed_action,
                    -normalized,
                    limits,
                )
                self._previous_executed_action = np.asarray(
                    executed_action, dtype=np.float32
                ).copy()
        self._chunk_pos += 1
        next_obs = self._obs(raw_next)
        return next_obs, normalized, reference
