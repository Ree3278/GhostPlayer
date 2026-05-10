"""Data preparation for the NFL Big Data Bowl 2026 movement-prediction schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ghostplayer.data.build_graphs import GraphDataset, fully_connected_edge_index
from ghostplayer.data.preprocess import DatasetSplits
from ghostplayer.utils.config import SplitConfig
from ghostplayer.utils.schema import (
    CONTINUOUS_FEATURE_COLUMNS,
    DEFAULT_FIELD_LENGTH,
    DEFAULT_FIELD_WIDTH,
    TEAM_TYPE_TO_ID,
    TOTAL_NODE_COUNT,
)

PLAYER_NODE_COUNT = TOTAL_NODE_COUNT - 1
BALL_LANDING_NODE_INDEX = TOTAL_NODE_COUNT - 1

ROLE_ORDER = {
    "Passer": 0,
    "Targeted Receiver": 1,
    "Other Route Runner": 2,
    "Defensive Coverage": 3,
}


@dataclass(slots=True)
class BigDataBowl2026Tables:
    """Loaded 2026 competition files."""

    supplementary: pd.DataFrame
    input_tracking: pd.DataFrame
    output_tracking: pd.DataFrame


@dataclass(slots=True)
class BigDataBowl2026BuildSummary:
    """Build statistics for the 2026 graph serialization path."""

    total_plays: int = 0
    examples_built: int = 0
    dropped_short_input: int = 0
    dropped_missing_output: int = 0
    dropped_no_prediction_nodes: int = 0
    truncated_player_contexts: int = 0
    trajectory_horizon: int = 0


def find_2026_data_root(data_dir: Path) -> Path:
    """Find the extracted 2026 competition directory under ``data_dir``."""

    candidates = [
        path
        for path in [data_dir, *data_dir.iterdir()]
        if path.is_dir() and (path / "train").exists() and list((path / "train").glob("input_*.csv"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a 2026 data directory under {data_dir}. "
            "Expected a folder containing train/input_*.csv and train/output_*.csv."
        )
    return candidates[0]


def load_2026_tables(data_root: Path) -> BigDataBowl2026Tables:
    """Load all 2026 input/output tracking files plus supplementary metadata."""

    supplementary_path = data_root / "supplementary_data.csv"
    train_dir = data_root / "train"
    input_paths = sorted(train_dir.glob("input_*.csv"))
    output_paths = sorted(train_dir.glob("output_*.csv"))

    if not supplementary_path.exists():
        raise FileNotFoundError(f"Missing supplementary_data.csv at {supplementary_path}")
    if not input_paths or not output_paths:
        raise FileNotFoundError(f"Missing input/output CSV files under {train_dir}")

    inputs = []
    for path in input_paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frame["source_week"] = _week_from_file_name(path.name)
        inputs.append(frame)

    outputs = []
    for path in output_paths:
        frame = pd.read_csv(path, low_memory=False)
        frame["source_file"] = path.name
        frame["source_week"] = _week_from_file_name(path.name)
        outputs.append(frame)

    return BigDataBowl2026Tables(
        supplementary=pd.read_csv(supplementary_path, low_memory=False),
        input_tracking=pd.concat(inputs, ignore_index=True),
        output_tracking=pd.concat(outputs, ignore_index=True),
    )


def split_game_ids_2026(
    supplementary: pd.DataFrame,
    config: SplitConfig,
) -> DatasetSplits:
    """Split 2026 data by ``game_id``."""

    if abs(config.train_fraction + config.validation_fraction + config.test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")

    unique_game_ids = sorted(supplementary["game_id"].dropna().unique().tolist())
    shuffled = pd.Series(unique_game_ids).sample(frac=1.0, random_state=config.random_seed).tolist()
    train_end = int(len(shuffled) * config.train_fraction)
    validation_end = train_end + int(len(shuffled) * config.validation_fraction)

    return DatasetSplits(
        train_game_ids=[int(game_id) for game_id in shuffled[:train_end]],
        validation_game_ids=[int(game_id) for game_id in shuffled[train_end:validation_end]],
        test_game_ids=[int(game_id) for game_id in shuffled[validation_end:]],
    )


def build_position_vocab_2026(input_tracking: pd.DataFrame) -> dict[str, int]:
    """Build stable ids for player positions plus the ball-landing context node."""

    positions = sorted(input_tracking["player_position"].dropna().astype(str).str.upper().unique().tolist())
    vocab = {"UNK": 0, "BALL_LAND": 1}
    for position in positions:
        if position not in vocab:
            vocab[position] = len(vocab)
    return vocab


def build_2026_graph_dataset(
    input_tracking: pd.DataFrame,
    output_tracking: pd.DataFrame,
    *,
    game_ids: list[int] | None = None,
    lookback_frames: int = 10,
    target_output_frame_id: int = 1,
    trajectory_horizon: int | None = None,
    position_vocab: dict[str, int] | None = None,
) -> tuple[GraphDataset, BigDataBowl2026BuildSummary, dict[str, int]]:
    """Convert 2026 competition files into the project's graph dataset format."""

    required_input = {
        "game_id",
        "play_id",
        "player_to_predict",
        "nfl_id",
        "frame_id",
        "play_direction",
        "player_position",
        "player_side",
        "player_role",
        "x",
        "y",
        "s",
        "a",
        "dir",
        "o",
        "ball_land_x",
        "ball_land_y",
    }
    required_output = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y"}
    _require_columns(input_tracking, required_input, "input_tracking")
    _require_columns(output_tracking, required_output, "output_tracking")

    position_vocab = position_vocab or build_position_vocab_2026(input_tracking)
    play_directions = (
        input_tracking[["game_id", "play_id", "play_direction"]]
        .drop_duplicates(subset=["game_id", "play_id"])
        .copy()
    )
    input_tracking = _normalize_orientation(input_tracking)
    output_tracking = output_tracking.merge(play_directions, on=["game_id", "play_id"], how="left")
    output_tracking = _normalize_output_orientation(output_tracking)

    if game_ids is not None:
        game_id_set = set(int(game_id) for game_id in game_ids)
        input_tracking = input_tracking.loc[input_tracking["game_id"].isin(game_id_set)].copy()
        output_tracking = output_tracking.loc[output_tracking["game_id"].isin(game_id_set)].copy()

    if output_tracking.empty:
        raise ValueError("No output_tracking rows remain after filtering.")

    output_horizon = int(output_tracking["frame_id"].max()) if trajectory_horizon is None else int(trajectory_horizon)
    if output_horizon < target_output_frame_id:
        raise ValueError(
            "trajectory_horizon must be >= target_output_frame_id, "
            f"got {output_horizon} and {target_output_frame_id}."
        )
    target_frame_ids = np.arange(1, output_horizon + 1, dtype=np.int64)

    output_first = output_tracking.loc[output_tracking["frame_id"].eq(target_output_frame_id)].copy()
    output_lookup = {
        (int(row.game_id), int(row.play_id), int(row.nfl_id)): (float(row.x), float(row.y))
        for row in output_first.itertuples(index=False)
    }
    trajectory_rows = output_tracking.loc[
        output_tracking["frame_id"].between(1, output_horizon, inclusive="both")
    ].copy()
    trajectory_lookup = {
        (int(row.game_id), int(row.play_id), int(row.nfl_id), int(row.frame_id)): (float(row.x), float(row.y))
        for row in trajectory_rows.itertuples(index=False)
    }

    examples = []
    metadata_rows: list[list[int]] = []
    frame_id_rows: list[np.ndarray] = []
    summary = BigDataBowl2026BuildSummary(
        total_plays=int(input_tracking[["game_id", "play_id"]].drop_duplicates().shape[0]),
        trajectory_horizon=output_horizon,
    )

    for (game_id, play_id), play_tracking in input_tracking.groupby(["game_id", "play_id"], sort=True):
        built = _build_one_2026_example(
            play_tracking=play_tracking,
            output_lookup=output_lookup,
            trajectory_lookup=trajectory_lookup,
            target_frame_ids=target_frame_ids,
            position_vocab=position_vocab,
            lookback_frames=lookback_frames,
        )

        if built is None:
            reason = _drop_reason(play_tracking, output_lookup, lookback_frames)
            if reason == "short_input":
                summary.dropped_short_input += 1
            elif reason == "missing_output":
                summary.dropped_missing_output += 1
            else:
                summary.dropped_no_prediction_nodes += 1
            continue

        arrays, metadata, frame_ids, truncated = built
        examples.append(arrays)
        metadata_rows.append(metadata)
        frame_id_rows.append(frame_ids)
        summary.truncated_player_contexts += int(truncated)

    if not examples:
        raise ValueError("No 2026 graph examples were built.")

    summary.examples_built = len(examples)
    stacked = {
        key: np.stack([example[key] for example in examples])
        for key in (
            "history_continuous",
            "position_ids",
            "team_type_ids",
            "history_ball_active",
            "target_positions",
            "prediction_mask",
            "target_trajectories",
            "target_trajectory_mask",
            "target_frame_ids",
        )
    }

    dataset = GraphDataset(
        edge_index=fully_connected_edge_index(),
        history_continuous=stacked["history_continuous"].astype(np.float32),
        position_ids=stacked["position_ids"].astype(np.int64),
        team_type_ids=stacked["team_type_ids"].astype(np.int64),
        history_ball_active=stacked["history_ball_active"].astype(np.float32),
        target_positions=stacked["target_positions"].astype(np.float32),
        defender_mask=stacked["prediction_mask"].astype(bool),
        metadata=np.asarray(metadata_rows, dtype=np.int64),
        frame_ids=np.stack(frame_id_rows).astype(np.int64),
        target_trajectories=stacked["target_trajectories"].astype(np.float32),
        target_trajectory_mask=stacked["target_trajectory_mask"].astype(bool),
        target_frame_ids=stacked["target_frame_ids"].astype(np.int64),
    )
    return dataset, summary, position_vocab


def _build_one_2026_example(
    *,
    play_tracking: pd.DataFrame,
    output_lookup: dict[tuple[int, int, int], tuple[float, float]],
    trajectory_lookup: dict[tuple[int, int, int, int], tuple[float, float]],
    target_frame_ids: np.ndarray,
    position_vocab: dict[str, int],
    lookback_frames: int,
) -> tuple[dict[str, np.ndarray], list[int], np.ndarray, bool] | None:
    game_id = int(play_tracking["game_id"].iloc[0])
    play_id = int(play_tracking["play_id"].iloc[0])
    frame_ids = sorted(play_tracking["frame_id"].dropna().astype(int).unique().tolist())
    if len(frame_ids) < lookback_frames:
        return None

    history_frame_ids = frame_ids[-lookback_frames:]
    candidate_players = _ordered_players(play_tracking)
    prediction_player_ids = [
        int(player_id)
        for player_id in candidate_players
        if bool(play_tracking.loc[play_tracking["nfl_id"].eq(player_id), "player_to_predict"].iloc[0])
    ]
    prediction_player_ids = [
        player_id for player_id in prediction_player_ids if (game_id, play_id, player_id) in output_lookup
    ]
    if not prediction_player_ids:
        return None

    ordered_player_ids = prediction_player_ids + [
        int(player_id) for player_id in candidate_players if int(player_id) not in set(prediction_player_ids)
    ]
    truncated = len(ordered_player_ids) > PLAYER_NODE_COUNT
    ordered_player_ids = ordered_player_ids[:PLAYER_NODE_COUNT]

    history = np.zeros((lookback_frames, TOTAL_NODE_COUNT, len(CONTINUOUS_FEATURE_COLUMNS)), dtype=np.float32)
    ball_active = np.zeros((lookback_frames, TOTAL_NODE_COUNT), dtype=np.float32)
    position_ids = np.zeros(TOTAL_NODE_COUNT, dtype=np.int64)
    team_type_ids = np.zeros(TOTAL_NODE_COUNT, dtype=np.int64)
    prediction_mask = np.zeros(TOTAL_NODE_COUNT, dtype=bool)
    targets = np.zeros((TOTAL_NODE_COUNT, 2), dtype=np.float32)
    target_trajectories = np.zeros((target_frame_ids.shape[0], TOTAL_NODE_COUNT, 2), dtype=np.float32)
    target_trajectory_mask = np.zeros((target_frame_ids.shape[0], TOTAL_NODE_COUNT), dtype=bool)

    frame_tables = {
        int(frame_id): rows.drop_duplicates(subset=["nfl_id"], keep="last").set_index("nfl_id")
        for frame_id, rows in play_tracking.loc[play_tracking["frame_id"].isin(history_frame_ids)].groupby("frame_id")
    }
    player_meta = play_tracking.drop_duplicates(subset=["nfl_id"], keep="last").set_index("nfl_id")

    for node_index, player_id in enumerate(ordered_player_ids):
        meta = player_meta.loc[player_id]
        position = str(meta["player_position"]).upper()
        side = str(meta["player_side"])
        position_ids[node_index] = position_vocab.get(position, position_vocab["UNK"])
        team_type_ids[node_index] = TEAM_TYPE_TO_ID["defense" if side == "Defense" else "offense"]

        if player_id in prediction_player_ids:
            prediction_mask[node_index] = True
            targets[node_index] = np.asarray(output_lookup[(game_id, play_id, player_id)], dtype=np.float32)
            for output_index, output_frame_id in enumerate(target_frame_ids):
                trajectory_key = (game_id, play_id, player_id, int(output_frame_id))
                if trajectory_key in trajectory_lookup:
                    target_trajectories[output_index, node_index] = np.asarray(
                        trajectory_lookup[trajectory_key],
                        dtype=np.float32,
                    )
                    target_trajectory_mask[output_index, node_index] = True

        for time_index, frame_id in enumerate(history_frame_ids):
            if frame_id not in frame_tables or player_id not in frame_tables[frame_id].index:
                return None
            row = frame_tables[frame_id].loc[player_id]
            history[time_index, node_index] = row.loc[list(CONTINUOUS_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)

    first_row = play_tracking.iloc[0]
    ball_x = float(first_row["ball_land_x"])
    ball_y = float(first_row["ball_land_y"])
    if str(first_row["play_direction"]).lower() == "left":
        ball_x = DEFAULT_FIELD_LENGTH - ball_x
        ball_y = DEFAULT_FIELD_WIDTH - ball_y

    position_ids[BALL_LANDING_NODE_INDEX] = position_vocab["BALL_LAND"]
    team_type_ids[BALL_LANDING_NODE_INDEX] = TEAM_TYPE_TO_ID["ball"]
    history[:, BALL_LANDING_NODE_INDEX, 0] = ball_x
    history[:, BALL_LANDING_NODE_INDEX, 1] = ball_y
    ball_active[:, BALL_LANDING_NODE_INDEX] = 1.0

    arrays = {
        "history_continuous": history,
        "position_ids": position_ids,
        "team_type_ids": team_type_ids,
        "history_ball_active": ball_active,
        "target_positions": targets,
        "prediction_mask": prediction_mask,
        "target_trajectories": target_trajectories,
        "target_trajectory_mask": target_trajectory_mask,
        "target_frame_ids": target_frame_ids,
    }
    metadata = [game_id, play_id, int(history_frame_ids[0]), int(history_frame_ids[-1] + 1)]
    return arrays, metadata, np.asarray([*history_frame_ids, history_frame_ids[-1] + 1], dtype=np.int64), truncated


def _ordered_players(play_tracking: pd.DataFrame) -> list[int]:
    player_rows = (
        play_tracking.drop_duplicates(subset=["nfl_id"], keep="last")
        .assign(
            predict_order=lambda frame: ~frame["player_to_predict"].astype(bool),
            side_order=lambda frame: frame["player_side"].map({"Offense": 0, "Defense": 1}).fillna(9),
            role_order=lambda frame: frame["player_role"].map(ROLE_ORDER).fillna(99),
        )
        .sort_values(["predict_order", "side_order", "role_order", "player_position", "nfl_id"])
    )
    return [int(player_id) for player_id in player_rows["nfl_id"].tolist()]


def _drop_reason(
    play_tracking: pd.DataFrame,
    output_lookup: dict[tuple[int, int, int], tuple[float, float]],
    lookback_frames: int,
) -> str:
    if play_tracking["frame_id"].nunique() < lookback_frames:
        return "short_input"
    game_id = int(play_tracking["game_id"].iloc[0])
    play_id = int(play_tracking["play_id"].iloc[0])
    predicted = play_tracking.loc[play_tracking["player_to_predict"], "nfl_id"].drop_duplicates().astype(int)
    if predicted.empty:
        return "no_prediction_nodes"
    if not any((game_id, play_id, int(player_id)) in output_lookup for player_id in predicted):
        return "missing_output"
    return "unknown"


def _normalize_orientation(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    moving_left = normalized["play_direction"].astype(str).str.lower().eq("left")
    normalized.loc[moving_left, "x"] = DEFAULT_FIELD_LENGTH - normalized.loc[moving_left, "x"]
    normalized.loc[moving_left, "y"] = DEFAULT_FIELD_WIDTH - normalized.loc[moving_left, "y"]
    for column in ("o", "dir"):
        normalized.loc[moving_left, column] = (normalized.loc[moving_left, column].astype(float) + 180.0) % 360.0
    normalized["play_direction"] = "right"
    return normalized


def _normalize_output_orientation(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    moving_left = normalized["play_direction"].astype(str).str.lower().eq("left")
    normalized.loc[moving_left, "x"] = DEFAULT_FIELD_LENGTH - normalized.loc[moving_left, "x"]
    normalized.loc[moving_left, "y"] = DEFAULT_FIELD_WIDTH - normalized.loc[moving_left, "y"]
    return normalized.drop(columns=["play_direction"])


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _week_from_file_name(file_name: str) -> int | None:
    match = re.search(r"_w(\d+)", file_name)
    return int(match.group(1)) if match else None
