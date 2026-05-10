from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ghostplayer.data.build_sequences import SequenceExample
from ghostplayer.utils.schema import CONTINUOUS_FEATURE_COLUMNS, TOTAL_NODE_COUNT, SequenceMetadata


@dataclass(slots=True)
class GraphSequenceExample:
    """Graph-ready version of a dense tracking sequence."""

    metadata: SequenceMetadata
    frame_ids: np.ndarray
    edge_index: np.ndarray
    history_continuous: np.ndarray
    position_ids: np.ndarray
    team_type_ids: np.ndarray
    history_ball_active: np.ndarray
    target_positions: np.ndarray
    defender_mask: np.ndarray
    target_trajectories: np.ndarray | None = None
    target_trajectory_mask: np.ndarray | None = None
    target_frame_ids: np.ndarray | None = None


@dataclass(slots=True)
class GraphDataset:
    """Stacked graph examples ready for training serialization."""

    edge_index: np.ndarray
    history_continuous: np.ndarray
    position_ids: np.ndarray
    team_type_ids: np.ndarray
    history_ball_active: np.ndarray
    target_positions: np.ndarray
    defender_mask: np.ndarray
    metadata: np.ndarray
    frame_ids: np.ndarray
    target_trajectories: np.ndarray | None = None
    target_trajectory_mask: np.ndarray | None = None
    target_frame_ids: np.ndarray | None = None


def fully_connected_edge_index(
    num_nodes: int = TOTAL_NODE_COUNT,
    *,
    include_self_edges: bool = False,
) -> np.ndarray:
    """Return a reusable fully connected directed edge index."""

    source_nodes: list[int] = []
    target_nodes: list[int] = []

    for source in range(num_nodes):
        for target in range(num_nodes):
            if not include_self_edges and source == target:
                continue
            source_nodes.append(source)
            target_nodes.append(target)

    return np.asarray([source_nodes, target_nodes], dtype=np.int64)


def sequence_to_graph(
    sequence: SequenceExample,
    *,
    edge_index: np.ndarray | None = None,
) -> GraphSequenceExample:
    """Convert one dense sequence into a graph-ready example."""

    if sequence.history_continuous.shape[1] != TOTAL_NODE_COUNT:
        raise ValueError(
            "Expected history_continuous node dimension "
            f"{TOTAL_NODE_COUNT}, got {sequence.history_continuous.shape[1]}."
        )

    if sequence.history_continuous.shape[2] != len(CONTINUOUS_FEATURE_COLUMNS):
        raise ValueError(
            "Expected continuous feature dimension "
            f"{len(CONTINUOUS_FEATURE_COLUMNS)}, got {sequence.history_continuous.shape[2]}."
        )

    graph_edge_index = fully_connected_edge_index() if edge_index is None else edge_index
    return GraphSequenceExample(
        metadata=sequence.metadata,
        frame_ids=sequence.frame_ids,
        edge_index=graph_edge_index,
        history_continuous=sequence.history_continuous,
        position_ids=sequence.position_ids,
        team_type_ids=sequence.team_type_ids,
        history_ball_active=sequence.history_ball_active,
        target_positions=sequence.target_positions,
        defender_mask=sequence.defender_mask,
        target_trajectories=getattr(sequence, "target_trajectories", None),
        target_trajectory_mask=getattr(sequence, "target_trajectory_mask", None),
        target_frame_ids=getattr(sequence, "target_frame_ids", None),
    )


def build_graph_examples(sequences: list[SequenceExample]) -> list[GraphSequenceExample]:
    """Convert dense sequences into graph examples sharing one edge index."""

    edge_index = fully_connected_edge_index()
    return [sequence_to_graph(sequence, edge_index=edge_index) for sequence in sequences]


def stack_graph_examples(examples: list[GraphSequenceExample]) -> GraphDataset:
    """Stack graph examples into contiguous arrays for fast model loading."""

    if not examples:
        raise ValueError("Cannot stack an empty graph example list.")

    edge_index = examples[0].edge_index
    metadata = np.asarray(
        [
            [
                example.metadata.game_id,
                example.metadata.play_id,
                example.metadata.start_frame_id,
                example.metadata.target_frame_id,
            ]
            for example in examples
        ],
        dtype=np.int64,
    )

    target_trajectories = [example.target_trajectories for example in examples]
    target_trajectory_mask = [example.target_trajectory_mask for example in examples]
    target_frame_ids = [example.target_frame_ids for example in examples]

    return GraphDataset(
        edge_index=edge_index,
        history_continuous=np.stack([example.history_continuous for example in examples]).astype(np.float32),
        position_ids=np.stack([example.position_ids for example in examples]).astype(np.int64),
        team_type_ids=np.stack([example.team_type_ids for example in examples]).astype(np.int64),
        history_ball_active=np.stack([example.history_ball_active for example in examples]).astype(np.float32),
        target_positions=np.stack([example.target_positions for example in examples]).astype(np.float32),
        defender_mask=np.stack([example.defender_mask for example in examples]).astype(bool),
        metadata=metadata,
        frame_ids=np.stack([example.frame_ids for example in examples]).astype(np.int64),
        target_trajectories=(
            np.stack(target_trajectories).astype(np.float32)
            if all(value is not None for value in target_trajectories)
            else None
        ),
        target_trajectory_mask=(
            np.stack(target_trajectory_mask).astype(bool)
            if all(value is not None for value in target_trajectory_mask)
            else None
        ),
        target_frame_ids=(
            np.stack(target_frame_ids).astype(np.int64)
            if all(value is not None for value in target_frame_ids)
            else None
        ),
    )


def save_graph_dataset(dataset: GraphDataset, output_path: Path) -> None:
    """Serialize a graph dataset to compressed NumPy format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "edge_index": dataset.edge_index,
        "history_continuous": dataset.history_continuous,
        "position_ids": dataset.position_ids,
        "team_type_ids": dataset.team_type_ids,
        "history_ball_active": dataset.history_ball_active,
        "target_positions": dataset.target_positions,
        "defender_mask": dataset.defender_mask,
        "metadata": dataset.metadata,
        "frame_ids": dataset.frame_ids,
    }
    if dataset.target_trajectories is not None:
        arrays["target_trajectories"] = dataset.target_trajectories
    if dataset.target_trajectory_mask is not None:
        arrays["target_trajectory_mask"] = dataset.target_trajectory_mask
    if dataset.target_frame_ids is not None:
        arrays["target_frame_ids"] = dataset.target_frame_ids

    np.savez_compressed(
        output_path,
        **arrays,
    )


def load_graph_dataset(input_path: Path) -> GraphDataset:
    """Load a graph dataset saved by ``save_graph_dataset``."""

    with np.load(input_path) as data:
        return GraphDataset(
            edge_index=data["edge_index"],
            history_continuous=data["history_continuous"],
            position_ids=data["position_ids"],
            team_type_ids=data["team_type_ids"],
            history_ball_active=data["history_ball_active"],
            target_positions=data["target_positions"],
            defender_mask=data["defender_mask"],
            metadata=data["metadata"],
            frame_ids=data["frame_ids"],
            target_trajectories=data["target_trajectories"] if "target_trajectories" in data.files else None,
            target_trajectory_mask=data["target_trajectory_mask"] if "target_trajectory_mask" in data.files else None,
            target_frame_ids=data["target_frame_ids"] if "target_frame_ids" in data.files else None,
        )
