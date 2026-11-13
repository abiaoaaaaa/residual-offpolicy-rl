"""Numeric residual context shared by collector and offline conversion."""

from __future__ import annotations

import numpy as np


CONTEXT_DIM = 48


def residual_context(previous_residual, absolute_delta, disagreement, limits) -> np.ndarray:
    previous_residual = np.asarray(previous_residual, dtype=np.float32).reshape(-1)
    absolute_delta = np.asarray(absolute_delta, dtype=np.float32).reshape(-1)
    disagreement = np.asarray(disagreement, dtype=np.float32).reshape(-1)
    limits = np.asarray(limits, dtype=np.float32).reshape(-1)
    if not (previous_residual.shape == absolute_delta.shape == disagreement.shape == limits.shape):
        raise ValueError("residual context vectors must have the same shape")
    delta = np.clip(absolute_delta / np.maximum(limits, 1e-6), -1.0, 1.0)
    return np.concatenate([previous_residual, delta, disagreement]).astype(np.float32)
