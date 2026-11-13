"""Offline EEF v9 planner: Pi joint chunk -> EEF residual -> stateful IK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eef_v9 import (
    EEF_ACTION_DIM,
    EEFResidualFilter,
    EEFV9Codec,
    FK,
    StatefulNumericalIK,
    _poses,
    apply_base_delta,
)


@dataclass(frozen=True)
class EEFPlanStep:
    pi_joint_target: np.ndarray
    pi_eef_reference: np.ndarray
    proposed_residual: np.ndarray
    executed_residual: np.ndarray
    command_eef_pose: np.ndarray
    solved_joint_target: np.ndarray
    ik_error: float
    ik_success: bool


class EEFV9Planner:
    """Plan one 1-second Pi chunk at the 10 Hz residual rate.

    The first EEF delta is measured-pose -> ``pi_chunk[2]``.  Subsequent base
    deltas are consecutive selected Pi references.  Residuals are applied to
    each absolute Pi EEF reference, then solved with the previous IK result as
    the next seed.
    """

    def __init__(self, fk: FK, *, codec: EEFV9Codec | None = None,
                 residual_filter: EEFResidualFilter | None = None):
        self.fk = fk
        self.codec = codec or EEFV9Codec()
        self.residual_filter = residual_filter or EEFResidualFilter(
            limits=None, alpha=0.35
        )
        self.ik = StatefulNumericalIK(fk)

    def plan(
        self,
        current_joint: np.ndarray,
        pi_joint_chunk: np.ndarray,
        proposed_residuals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[EEFPlanStep]]:
        current_joint = np.asarray(current_joint, dtype=np.float64).reshape(-1)
        pi_joint_chunk = np.asarray(pi_joint_chunk, dtype=np.float64)
        residuals = np.asarray(proposed_residuals, dtype=np.float32)
        if current_joint.shape != (14,):
            raise ValueError("current_joint must have shape (14,)")
        if pi_joint_chunk.ndim != 2 or pi_joint_chunk.shape[0] != 30 or pi_joint_chunk.shape[1] not in (14, 16):
            raise ValueError("Pi chunk must have shape (30,14) or (30,16)")
        if residuals.shape != (10, EEF_ACTION_DIM):
            raise ValueError("residuals must have shape (10,12)")

        selected_joint = pi_joint_chunk[2::3, :14]
        current_pose = _poses(self.fk(current_joint))[0]
        references = _poses(np.stack([self.fk(q) for q in selected_joint]))
        base_delta = self.codec.encode_chunk(current_pose, references)
        self.ik.reset(current_joint)
        self.residual_filter.reset()
        solved = np.empty_like(selected_joint, dtype=np.float32)
        steps: list[EEFPlanStep] = []
        for index in range(10):
            executed_residual = self.residual_filter(residuals[index])
            command_pose = apply_base_delta(references[index], executed_residual * self.codec.limits)
            joint, error, success = self.ik.solve(command_pose, seed=self.ik.previous)
            solved[index] = joint
            steps.append(
                EEFPlanStep(
                    pi_joint_target=selected_joint[index].astype(np.float32),
                    pi_eef_reference=references[index].astype(np.float32),
                    proposed_residual=residuals[index].copy(),
                    executed_residual=executed_residual.copy(),
                    command_eef_pose=command_pose[0].astype(np.float32),
                    solved_joint_target=joint.copy(),
                    ik_error=error,
                    ik_success=success,
                )
            )
        return base_delta.astype(np.float32), solved, steps
