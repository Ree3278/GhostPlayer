"""Loss functions for GhostPlayer models."""

from __future__ import annotations

import torch
from torch import nn


def masked_mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE over ``(..., nodes, 2)`` predictions selected by ``mask``."""

    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target shapes must match, got {predicted.shape} and {target.shape}.")
    if predicted.shape[-1] != 2:
        raise ValueError(f"Expected final coordinate dimension of 2, got {predicted.shape[-1]}.")
    if mask.shape != predicted.shape[:-1]:
        raise ValueError(f"mask shape must match prediction node shape, got {mask.shape} and {predicted.shape[:-1]}.")

    mask = mask.bool()
    if not torch.any(mask):
        raise ValueError("Cannot compute masked MSE with an empty mask.")

    squared_error = (predicted - target).pow(2).sum(dim=-1)
    return squared_error[mask].mean()


def masked_ade(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute average Euclidean displacement error over masked nodes."""

    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target shapes must match, got {predicted.shape} and {target.shape}.")
    if mask.shape != predicted.shape[:-1]:
        raise ValueError(f"mask shape must match prediction node shape, got {mask.shape} and {predicted.shape[:-1]}.")

    mask = mask.bool()
    if not torch.any(mask):
        raise ValueError("Cannot compute masked ADE with an empty mask.")

    errors = torch.linalg.vector_norm(predicted - target, dim=-1)
    return errors[mask].mean()


class MaskedMSELoss(nn.Module):
    """Module wrapper for masked coordinate MSE."""

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return masked_mse_loss(predicted, target, mask)
