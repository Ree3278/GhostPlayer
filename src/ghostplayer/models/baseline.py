"""Baseline model definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ghostplayer.data.build_graphs import GraphDataset


@dataclass(slots=True)
class BaselineArrays:
    """Flattened per-defender arrays for the non-relational baseline."""

    features: np.ndarray
    targets: np.ndarray
    metadata: np.ndarray


class DefenderBaselineDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset for one-defender-at-a-time baseline samples."""

    def __init__(self, arrays: BaselineArrays) -> None:
        self.features = torch.as_tensor(arrays.features, dtype=torch.float32)
        self.targets = torch.as_tensor(arrays.targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


class DefenderMLP(nn.Module):
    """MLP baseline using only a single defender's own history."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def graph_dataset_to_baseline_arrays(dataset: GraphDataset) -> BaselineArrays:
    """Extract one sample per defender from graph examples."""

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata_rows: list[list[int]] = []

    for example_index in range(dataset.history_continuous.shape[0]):
        defender_node_indices = np.flatnonzero(dataset.defender_mask[example_index])

        for node_index in defender_node_indices:
            defender_history = dataset.history_continuous[example_index, :, node_index, :]
            features.append(defender_history.reshape(-1))
            targets.append(dataset.target_positions[example_index, node_index, :])
            metadata_rows.append(
                [
                    int(dataset.metadata[example_index, 0]),
                    int(dataset.metadata[example_index, 1]),
                    int(dataset.metadata[example_index, 2]),
                    int(dataset.metadata[example_index, 3]),
                    int(node_index),
                ]
            )

    if not features:
        raise ValueError("Graph dataset did not contain any defender baseline samples.")

    return BaselineArrays(
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        metadata=np.asarray(metadata_rows, dtype=np.int64),
    )
