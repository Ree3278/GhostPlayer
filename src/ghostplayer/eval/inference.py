"""Inference helpers for GhostPlayer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ghostplayer.data.build_graphs import GraphDataset
from ghostplayer.models.baseline import DefenderMLP
from ghostplayer.models.st_gat import GraphSequenceTorchDataset, STGAT, batch_to_device


@dataclass(slots=True)
class PredictionResult:
    """Model predictions aligned to a serialized graph dataset."""

    model_name: str
    predictions: np.ndarray


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_baseline_model(checkpoint_path: Path, device: torch.device) -> DefenderMLP:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DefenderMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_stgat_model(checkpoint_path: Path, device: torch.device) -> STGAT:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = STGAT(
        num_positions=int(checkpoint["num_positions"]),
        num_team_types=int(checkpoint["num_team_types"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        gat_layers=int(checkpoint["gat_layers"]),
        gat_heads=int(checkpoint["gat_heads"]),
        temporal_hidden_dim=int(checkpoint["temporal_hidden_dim"]),
        position_embedding_dim=int(checkpoint["position_embedding_dim"]),
        team_embedding_dim=int(checkpoint["team_embedding_dim"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        output_horizon=int(checkpoint.get("output_horizon", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_baseline(
    dataset: GraphDataset,
    checkpoint_path: Path,
    *,
    batch_size: int = 4096,
    device: torch.device | None = None,
) -> PredictionResult:
    """Run the per-target-player baseline and return graph-shaped predictions."""

    device = device or torch.device("cpu")
    model = load_baseline_model(checkpoint_path, device)
    predictions = np.zeros_like(dataset.target_positions, dtype=np.float32)

    features: list[np.ndarray] = []
    indices: list[tuple[int, int]] = []
    for example_index in range(dataset.history_continuous.shape[0]):
        for node_index in np.flatnonzero(dataset.defender_mask[example_index]):
            features.append(dataset.history_continuous[example_index, :, node_index, :].reshape(-1))
            indices.append((example_index, int(node_index)))

    if not features:
        raise ValueError("Dataset has no masked target nodes for baseline inference.")

    feature_tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    model_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, feature_tensor.shape[0], batch_size):
            batch = feature_tensor[start : start + batch_size].to(device)
            model_predictions.append(model(batch).cpu().numpy())

    flat_predictions = np.concatenate(model_predictions, axis=0)
    for (example_index, node_index), prediction in zip(indices, flat_predictions):
        predictions[example_index, node_index, :] = prediction

    return PredictionResult(model_name="baseline", predictions=predictions)


def predict_stgat(
    dataset: GraphDataset,
    checkpoint_path: Path,
    *,
    batch_size: int = 128,
    device: torch.device | None = None,
    return_trajectory: bool = False,
) -> PredictionResult:
    """Run ST-GAT inference and return graph-shaped predictions."""

    device = device or torch.device("cpu")
    model = load_stgat_model(checkpoint_path, device)
    loader = DataLoader(GraphSequenceTorchDataset(dataset), batch_size=batch_size, shuffle=False)
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            batch_predictions = model(
                batch.history_continuous,
                batch.position_ids,
                batch.team_type_ids,
                batch.history_ball_active,
            )
            if batch_predictions.ndim == 4 and not return_trajectory:
                batch_predictions = batch_predictions[:, 0]
            predictions.append(batch_predictions.cpu().numpy())

    return PredictionResult(model_name="st_gat", predictions=np.concatenate(predictions, axis=0).astype(np.float32))
