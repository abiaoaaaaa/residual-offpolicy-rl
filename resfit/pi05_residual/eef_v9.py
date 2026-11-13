"""Offline-safe EEF residual v9 geometry and stateful IK helpers.

The module is deliberately independent of a robot driver.  A robot-specific
FK callback can be backed by Pinocchio, while the same codec is exercised with
the offline tests and replay converters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


ARM_COUNT = 2
POSE_DIM = 7  # xyz + quaternion xyzw
EEF_ACTION_DIM = 12  # xyz + rotation-vector for each arm


class FK(Protocol):
    def __call__(self, joints: np.ndarray) -> np.ndarray:
        """Return bilateral poses with shape ``(2, 7)``."""


def _poses(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (ARM_COUNT, POSE_DIM):
        value = value[None]
    if value.ndim != 3 or value.shape[1:] != (ARM_COUNT, POSE_DIM):
        raise ValueError(f"expected (...,2,7) EEF poses, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("EEF poses contain NaN or infinity")
    quat_norm = np.linalg.norm(value[..., 3:7], axis=-1)
    if np.any(quat_norm < 1e-8):
        raise ValueError("EEF quaternion has near-zero norm")
    value[..., 3:7] /= quat_norm[..., None]
    return value


def _rot(quat_xyzw: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64))


def eef_pose_delta(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Base-frame delta from one bilateral pose to the next.

    Translation is expressed in the fixed robot base frame.  Rotation is the
    base-frame rotation vector ``log(R_current R_previous.T)``.
    """
    prev, curr = _poses(previous), _poses(current)
    if prev.shape != curr.shape:
        raise ValueError("previous and current pose batches must have equal shape")
    out = np.empty((*prev.shape[:-2], EEF_ACTION_DIM), dtype=np.float64)
    for arm in range(ARM_COUNT):
        out[..., arm * 6 : arm * 6 + 3] = curr[..., arm, :3] - prev[..., arm, :3]
        relative = _rot(curr[..., arm, 3:7]) * _rot(prev[..., arm, 3:7]).inv()
        out[..., arm * 6 + 3 : arm * 6 + 6] = relative.as_rotvec()
    return out.astype(np.float32)


def apply_base_delta(reference: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a base-frame bilateral delta to an absolute EEF pose."""
    pose = _poses(reference).copy()
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape == (EEF_ACTION_DIM,):
        delta = delta[None]
    if delta.shape[-1] != EEF_ACTION_DIM or delta.shape[0] != pose.shape[0]:
        raise ValueError(f"expected delta shape ({pose.shape[0]},12), got {delta.shape}")
    for arm in range(ARM_COUNT):
        sl = slice(arm * 6, arm * 6 + 6)
        pose[:, arm, :3] += delta[:, sl.start : sl.start + 3]
        increment = Rotation.from_rotvec(delta[:, sl.start + 3 : sl.stop])
        pose[:, arm, 3:7] = (increment * _rot(pose[:, arm, 3:7])).as_quat()
    return pose.astype(np.float32)


@dataclass(frozen=True)
class EEFV9Limits:
    """Physical per-step limits at the 10 Hz residual control rate."""

    translation_m: float = 0.005
    rotation_rad: float = float(np.deg2rad(2.0))

    @property
    def vector(self) -> np.ndarray:
        return np.tile(
            np.array(
                [self.translation_m] * 3 + [self.rotation_rad] * 3,
                dtype=np.float32,
            ),
            ARM_COUNT,
        )


class EEFV9Codec:
    """Chunk delta codec and bounded residual composition for v9."""

    def __init__(self, limits: EEFV9Limits | None = None):
        self.limits = (limits or EEFV9Limits()).vector

    def encode_chunk(self, current_pose: np.ndarray, absolute_chunk: np.ndarray) -> np.ndarray:
        current = _poses(current_pose)
        chunk = _poses(absolute_chunk)
        if current.shape[0] != 1:
            raise ValueError("encode_chunk expects one current pose")
        previous = np.concatenate([current, chunk[:-1]], axis=0)
        return np.clip(eef_pose_delta(previous, chunk) / self.limits, -1.0, 1.0)

    def bound_residual(self, residual: np.ndarray) -> np.ndarray:
        value = np.asarray(residual, dtype=np.float32)
        if value.shape[-1] != EEF_ACTION_DIM:
            raise ValueError("EEF residual must have 12 dimensions")
        return np.clip(value, -1.0, 1.0)

    def apply_residual(self, absolute_reference: np.ndarray, residual: np.ndarray) -> np.ndarray:
        bounded = self.bound_residual(residual)
        physical = bounded * self.limits
        return apply_base_delta(absolute_reference, physical)


class EEFResidualFilter:
    """EMA and per-step slew limiting in physical EEF units."""

    def __init__(self, limits: EEFV9Limits | None = None, alpha: float = 0.35):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0,1]")
        self.limits = (limits or EEFV9Limits()).vector
        self.alpha = float(alpha)
        self.previous = np.zeros(EEF_ACTION_DIM, dtype=np.float32)

    def reset(self, value: np.ndarray | None = None) -> None:
        self.previous[:] = 0.0
        if value is not None:
            self.previous[:] = np.asarray(value, dtype=np.float32)

    def __call__(self, residual_normalized: np.ndarray) -> np.ndarray:
        target = np.asarray(residual_normalized, dtype=np.float32) * self.limits
        if target.shape != (EEF_ACTION_DIM,) or not np.isfinite(target).all():
            raise ValueError("invalid EEF residual")
        filtered = self.previous + self.alpha * (target - self.previous)
        filtered = self.previous + np.clip(filtered - self.previous, -self.limits, self.limits)
        self.previous[:] = filtered
        return filtered / self.limits


class StatefulNumericalIK:
    """Driver-independent, stateful bilateral IK for offline validation.

    ``fk`` must return poses for a 14-D joint vector.  The previous solution is
    used as the seed, which prevents frame-to-frame branch hopping.  A robot
    deployment may replace this class with the native Pinocchio/QP solver while
    retaining the same target and seed contract.
    """

    def __init__(self, fk: FK, *, max_nfev: int = 80, position_weight: float = 1.0,
                 rotation_weight: float = 0.25):
        self.fk, self.max_nfev = fk, int(max_nfev)
        self.position_weight, self.rotation_weight = position_weight, rotation_weight
        self.previous: np.ndarray | None = None

    def reset(self, joints: np.ndarray) -> None:
        value = np.asarray(joints, dtype=np.float64).reshape(-1)
        if value.shape != (14,):
            raise ValueError("IK seed must contain 14 arm joints")
        self.previous = value.copy()

    def solve(self, target_pose: np.ndarray, seed: np.ndarray | None = None) -> tuple[np.ndarray, float, bool]:
        target = _poses(target_pose)[0]
        x0 = np.asarray(seed if seed is not None else self.previous, dtype=np.float64).reshape(-1)
        if x0.shape != (14,):
            raise ValueError("stateful IK requires a 14-D previous joint seed")

        def residual(q: np.ndarray) -> np.ndarray:
            actual = _poses(self.fk(q))[0]
            terms = []
            for arm in range(ARM_COUNT):
                terms.extend(self.position_weight * (actual[arm, :3] - target[arm, :3]))
                rot_err = _rot(target[arm, 3:7]) * _rot(actual[arm, 3:7]).inv()
                terms.extend(self.rotation_weight * rot_err.as_rotvec())
            return np.asarray(terms)

        result = least_squares(residual, x0, max_nfev=self.max_nfev, method="trf")
        solved = result.x.astype(np.float32)
        err = float(np.linalg.norm(residual(result.x)))
        self.previous = result.x.copy()
        return solved, err, bool(result.success)
