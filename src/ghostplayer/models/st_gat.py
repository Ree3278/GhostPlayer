"""Spatio-temporal GAT model definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ghostplayer.data.build_graphs import GraphDataset
from ghostplayer.utils.schema import CONTINUOUS_FEATURE_COLUMNS, TEAM_TYPE_TO_ID, TOTAL_NODE_COUNT


@dataclass(slots=True)
class STGATBatch:
    """Typed batch fields consumed by the ST-GAT trainer."""

    history_continuous: torch.Tensor
    position_ids: torch.Tensor
    team_type_ids: torch.Tensor
    history_ball_active: torch.Tensor
    target_positions: torch.Tensor
    prediction_mask: torch.Tensor
    target_trajectories: torch.Tensor | None = None
    target_trajectory_mask: torch.Tensor | None = None


class GraphSequenceTorchDataset(Dataset[dict[str, torch.Tensor]]):
    """Torch dataset backed by serialized graph arrays."""

    def __init__(self, dataset: GraphDataset) -> None:
        self.history_continuous = torch.as_tensor(dataset.history_continuous, dtype=torch.float32)
        self.position_ids = torch.as_tensor(dataset.position_ids, dtype=torch.long)
        self.team_type_ids = torch.as_tensor(dataset.team_type_ids, dtype=torch.long)
        self.history_ball_active = torch.as_tensor(dataset.history_ball_active, dtype=torch.float32)
        self.target_positions = torch.as_tensor(dataset.target_positions, dtype=torch.float32)
        self.prediction_mask = torch.as_tensor(dataset.defender_mask, dtype=torch.bool)
        self.target_trajectories = (
            torch.as_tensor(dataset.target_trajectories, dtype=torch.float32)
            if dataset.target_trajectories is not None
            else None
        )
        self.target_trajectory_mask = (
            torch.as_tensor(dataset.target_trajectory_mask, dtype=torch.bool)
            if dataset.target_trajectory_mask is not None
            else None
        )

    def __len__(self) -> int:
        return int(self.history_continuous.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "history_continuous": self.history_continuous[index],
            "position_ids": self.position_ids[index],
            "team_type_ids": self.team_type_ids[index],
            "history_ball_active": self.history_ball_active[index],
            "target_positions": self.target_positions[index],
            "prediction_mask": self.prediction_mask[index],
        }
        if self.target_trajectories is not None:
            item["target_trajectories"] = self.target_trajectories[index]
        if self.target_trajectory_mask is not None:
            item["target_trajectory_mask"] = self.target_trajectory_mask[index]
        return item


class DenseGraphAttentionLayer(nn.Module):
    """Multi-head graph attention for dense per-frame node tensors."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.node_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """Apply attention to ``(batch, frames, nodes, features)`` tensors."""

        batch_size, num_frames, num_nodes, _ = node_features.shape
        projected = self.node_projection(node_features)
        projected = projected.view(batch_size, num_frames, num_nodes, self.num_heads, self.head_dim)

        source_scores = (projected * self.attention_source).sum(dim=-1)
        target_scores = (projected * self.attention_target).sum(dim=-1)
        attention_logits = source_scores.unsqueeze(3) + target_scores.unsqueeze(2)
        attention_logits = self.activation(attention_logits)

        eye = torch.eye(num_nodes, dtype=torch.bool, device=node_features.device)
        attention_logits = attention_logits.masked_fill(eye.view(1, 1, num_nodes, num_nodes, 1), float("-inf"))
        attention_weights = torch.softmax(attention_logits, dim=3)
        attention_weights = self.dropout(attention_weights)

        attended = torch.einsum("bfnmh,bfmhd->bfnhd", attention_weights, projected)
        attended = attended.reshape(batch_size, num_frames, num_nodes, self.num_heads * self.head_dim)
        return self.output_projection(attended)


class STGAT(nn.Module):
    """Spatio-temporal graph attention model for GhostPlayer graph sequences."""

    def __init__(
        self,
        *,
        continuous_dim: int = len(CONTINUOUS_FEATURE_COLUMNS),
        num_positions: int = 64,
        num_team_types: int = len(TEAM_TYPE_TO_ID),
        position_embedding_dim: int = 8,
        team_embedding_dim: int = 4,
        hidden_dim: int = 128,
        gat_layers: int = 2,
        gat_heads: int = 4,
        temporal_hidden_dim: int = 128,
        dropout: float = 0.1,
        output_horizon: int = 1,
    ) -> None:
        super().__init__()
        if output_horizon < 1:
            raise ValueError(f"output_horizon must be >= 1, got {output_horizon}.")
        self.output_horizon = int(output_horizon)
        input_dim = continuous_dim + 1 + position_embedding_dim + team_embedding_dim
        self.position_embedding = nn.Embedding(num_positions, position_embedding_dim)
        self.team_embedding = nn.Embedding(num_team_types, team_embedding_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.gat_layers = nn.ModuleList(
            [
                DenseGraphAttentionLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads=gat_heads,
                    dropout=dropout,
                )
                for _ in range(gat_layers)
            ]
        )
        self.gat_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(gat_layers)])
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(temporal_hidden_dim),
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden_dim, self.output_horizon * 2),
        )

    @classmethod
    def from_graph_dataset(
        cls,
        dataset: GraphDataset,
        **kwargs: object,
    ) -> "STGAT":
        num_positions = int(np.max(dataset.position_ids)) + 1
        num_team_types = int(np.max(dataset.team_type_ids)) + 1
        kwargs.setdefault(
            "output_horizon",
            int(dataset.target_trajectories.shape[1]) if dataset.target_trajectories is not None else 1,
        )
        return cls(num_positions=num_positions, num_team_types=num_team_types, **kwargs)

    def forward(
        self,
        history_continuous: torch.Tensor,
        position_ids: torch.Tensor,
        team_type_ids: torch.Tensor,
        history_ball_active: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target coordinates for every graph node.

        Single-step models return ``(batch, nodes, 2)``. Trajectory models
        return ``(batch, horizon, nodes, 2)``.
        """

        batch_size, num_frames, num_nodes, _ = history_continuous.shape
        position_features = self.position_embedding(position_ids)
        team_features = self.team_embedding(team_type_ids)
        position_features = position_features.unsqueeze(1).expand(-1, num_frames, -1, -1)
        team_features = team_features.unsqueeze(1).expand(-1, num_frames, -1, -1)

        node_features = torch.cat(
            [
                history_continuous,
                history_ball_active.unsqueeze(-1),
                position_features,
                team_features,
            ],
            dim=-1,
        )
        node_features = self.dropout(torch.relu(self.input_projection(node_features)))

        for gat_layer, norm in zip(self.gat_layers, self.gat_norms):
            residual = node_features
            node_features = gat_layer(node_features)
            node_features = self.dropout(torch.relu(node_features))
            node_features = norm(node_features + residual)

        temporal_input = node_features.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, num_frames, -1)
        _, hidden = self.temporal(temporal_input)
        final_hidden = hidden[-1].reshape(batch_size, num_nodes, -1)
        raw_output = self.output_head(final_hidden).view(batch_size, num_nodes, self.output_horizon, 2)
        if self.output_horizon == 1:
            return raw_output[:, :, 0, :]
        return raw_output.permute(0, 2, 1, 3).contiguous()


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> STGATBatch:
    """Move a DataLoader batch to a device and expose typed fields."""

    return STGATBatch(
        history_continuous=batch["history_continuous"].to(device),
        position_ids=batch["position_ids"].to(device),
        team_type_ids=batch["team_type_ids"].to(device),
        history_ball_active=batch["history_ball_active"].to(device),
        target_positions=batch["target_positions"].to(device),
        prediction_mask=batch["prediction_mask"].to(device),
        target_trajectories=(
            batch["target_trajectories"].to(device) if "target_trajectories" in batch else None
        ),
        target_trajectory_mask=(
            batch["target_trajectory_mask"].to(device) if "target_trajectory_mask" in batch else None
        ),
    )
