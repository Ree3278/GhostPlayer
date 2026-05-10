"""Evaluation metrics for GhostPlayer."""

from __future__ import annotations

import numpy as np


def displacement_errors(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    """Return Euclidean position errors for matching ``(..., 2)`` arrays."""

    if predicted.shape != actual.shape:
        raise ValueError(f"predicted and actual must have the same shape, got {predicted.shape} and {actual.shape}.")

    if predicted.shape[-1] != 2:
        raise ValueError(f"Expected final coordinate dimension of 2, got {predicted.shape[-1]}.")

    return np.linalg.norm(predicted - actual, axis=-1)


def average_displacement_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute ADE over all unmasked coordinate predictions."""

    errors = displacement_errors(predicted, actual)
    if mask is not None:
        if mask.shape != errors.shape:
            raise ValueError(f"mask shape must match error shape, got {mask.shape} and {errors.shape}.")
        errors = errors[mask]

    if errors.size == 0:
        raise ValueError("Cannot compute ADE over zero predictions.")

    return float(errors.mean())


def per_play_average_displacement_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    defender_mask: np.ndarray,
) -> np.ndarray:
    """Compute one ADE value per play/example over defender nodes."""

    if predicted.ndim != 3 or actual.ndim != 3:
        raise ValueError("predicted and actual must have shape (examples, nodes, 2).")

    errors = displacement_errors(predicted, actual)
    if defender_mask.shape != errors.shape[:2]:
        raise ValueError(
            f"defender_mask must have shape {errors.shape[:2]}, got {defender_mask.shape}."
        )

    masked_errors = np.where(defender_mask, errors, np.nan)
    return np.nanmean(masked_errors, axis=1)
