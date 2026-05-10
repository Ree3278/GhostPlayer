"""Sequence-building helpers for turning tracking windows into model examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ghostplayer.data.preprocess import validate_required_columns
from ghostplayer.utils.config import PreprocessingConfig
from ghostplayer.utils.schema import (
    CONTINUOUS_FEATURE_COLUMNS,
    DEFENDER_COUNT,
    REQUIRED_SEQUENCE_PLAY_COLUMNS,
    TEAM_TYPE_TO_ID,
    TOTAL_NODE_COUNT,
    NodeAssignment,
    SequenceMetadata,
)

OFFENSE_POSITION_ORDER = {
    "QB": 0,
    "RB": 1,
    "HB": 2,
    "FB": 3,
    "TE": 4,
    "WR": 5,
    "LT": 6,
    "LG": 7,
    "C": 8,
    "RG": 9,
    "RT": 10,
    "T": 11,
    "G": 12,
    "OL": 13,
    "OT": 14,
}

DEFENSE_POSITION_ORDER = {
    "DE": 0,
    "DT": 1,
    "NT": 2,
    "EDGE": 3,
    "LB": 4,
    "MLB": 5,
    "ILB": 6,
    "OLB": 7,
    "CB": 8,
    "DB": 9,
    "S": 10,
    "FS": 11,
    "SS": 12,
}


@dataclass(slots=True)
class SequenceExample:
    """Dense sequence example with fixed node ordering."""

    metadata: SequenceMetadata
    frame_ids: np.ndarray
    node_assignments: tuple[NodeAssignment, ...]
    history_continuous: np.ndarray
    position_ids: np.ndarray
    team_type_ids: np.ndarray
    history_ball_active: np.ndarray
    target_positions: np.ndarray
    defender_mask: np.ndarray


@dataclass(slots=True)
class SequenceBuildSummary:
    """Build statistics for debugging data contract issues."""

    total_plays: int = 0
    valid_plays: int = 0
    dropped_plays_missing_roles: int = 0
    dropped_plays_invalid_team_counts: int = 0
    dropped_windows_missing_player_rows: int = 0
    dropped_windows_non_consecutive_frames: int = 0
    total_examples: int = 0


def build_position_vocab(players: pd.DataFrame) -> dict[str, int]:
    """Create a stable integer mapping for player positions."""

    if "position" not in players.columns:
        return {"UNK": 0, "BALL": 1}

    positions = sorted(players["position"].dropna().astype(str).str.upper().unique().tolist())
    vocab = {"UNK": 0, "BALL": 1}
    for position in positions:
        if position not in vocab:
            vocab[position] = len(vocab)
    return vocab


def enrich_tracking_with_roles(
    tracking: pd.DataFrame,
    plays: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Attach position and offense/defense/ball labels to tracking rows."""

    validate_required_columns(plays, REQUIRED_SEQUENCE_PLAY_COLUMNS, "plays")

    play_columns = ["gameId", "playId", "possessionTeam", "defensiveTeam"]
    enriched = tracking.merge(plays[play_columns].drop_duplicates(), on=["gameId", "playId"], how="inner")

    if "position" not in enriched.columns and "position" in players.columns:
        player_columns = ["nflId", "position"]
        enriched = enriched.merge(players[player_columns].drop_duplicates(), on="nflId", how="left")

    if "club" not in enriched.columns and "teamAbbr" in enriched.columns:
        enriched = enriched.rename(columns={"teamAbbr": "club"})

    if "club" not in enriched.columns:
        raise ValueError("tracking must include 'club' (or 'teamAbbr') to infer offense and defense.")

    club = enriched["club"].fillna("").astype(str)
    nfl_id_missing = enriched["nflId"].isna()
    is_ball = club.str.lower().eq("football") | nfl_id_missing

    if "displayName" in enriched.columns:
        display_name = enriched["displayName"].fillna("").astype(str)
        is_ball |= display_name.str.lower().eq("football")

    team_type = np.select(
        [
            is_ball,
            club.eq(enriched["possessionTeam"]),
            club.eq(enriched["defensiveTeam"]),
        ],
        [
            "ball",
            "offense",
            "defense",
        ],
        default="unknown",
    )

    enriched["teamType"] = team_type
    enriched["position"] = enriched.get("position", "UNK")
    enriched["position"] = enriched["position"].fillna("UNK").astype(str).str.upper()
    return enriched


def _sort_player_rows(frame_rows: pd.DataFrame, team_type: str) -> pd.DataFrame:
    order_map = OFFENSE_POSITION_ORDER if team_type == "offense" else DEFENSE_POSITION_ORDER
    sorted_rows = frame_rows.copy()
    sorted_rows["positionOrder"] = sorted_rows["position"].map(order_map).fillna(999)
    return sorted_rows.sort_values(["positionOrder", "position", "nflId"]).drop(columns=["positionOrder"])


def build_node_assignments(play_tracking: pd.DataFrame) -> tuple[NodeAssignment, ...] | None:
    """Lock a play-level node order: offense, defense, then ball."""

    reference_frame_id = int(play_tracking["frameId"].min())
    reference_rows = play_tracking.loc[play_tracking["frameId"] == reference_frame_id].copy()
    reference_rows = reference_rows.loc[reference_rows["teamType"].isin({"offense", "defense", "ball"})]

    offense_rows = (
        reference_rows.loc[reference_rows["teamType"] == "offense", ["nflId", "position"]]
        .dropna(subset=["nflId"])
        .drop_duplicates(subset=["nflId"])
    )
    defense_rows = (
        reference_rows.loc[reference_rows["teamType"] == "defense", ["nflId", "position"]]
        .dropna(subset=["nflId"])
        .drop_duplicates(subset=["nflId"])
    )
    ball_exists = not reference_rows.loc[reference_rows["teamType"] == "ball"].empty

    if len(offense_rows) != DEFENDER_COUNT or len(defense_rows) != DEFENDER_COUNT or not ball_exists:
        return None

    offense_rows = _sort_player_rows(offense_rows, "offense")
    defense_rows = _sort_player_rows(defense_rows, "defense")

    assignments: list[NodeAssignment] = []

    for row in offense_rows.itertuples(index=False):
        assignments.append(
            NodeAssignment(
                node_index=len(assignments),
                node_type="offense",
                nfl_id=int(row.nflId),
                role=row.position,
            )
        )

    for row in defense_rows.itertuples(index=False):
        assignments.append(
            NodeAssignment(
                node_index=len(assignments),
                node_type="defense",
                nfl_id=int(row.nflId),
                role=row.position,
            )
        )

    assignments.append(
        NodeAssignment(
            node_index=len(assignments),
            node_type="ball",
            nfl_id=None,
            role="BALL",
        )
    )

    if len(assignments) != TOTAL_NODE_COUNT:
        return None

    return tuple(assignments)


def _frame_row_lookup(frame_rows: pd.DataFrame) -> tuple[dict[int, dict[str, object]], pd.DataFrame]:
    player_rows = (
        frame_rows.loc[frame_rows["teamType"].isin({"offense", "defense"})]
        .dropna(subset=["nflId"])
        .drop_duplicates(subset=["nflId"], keep="first")
        .set_index("nflId")
    )
    ball_rows = frame_rows.loc[frame_rows["teamType"] == "ball"]
    return player_rows.to_dict("index"), ball_rows


def _node_position_id(role: str, position_vocab: dict[str, int]) -> int:
    return position_vocab.get(role, position_vocab["UNK"])


def _materialize_window(
    play_tracking: pd.DataFrame,
    assignments: tuple[NodeAssignment, ...],
    frame_ids: list[int],
    target_frame_id: int,
    position_vocab: dict[str, int],
) -> SequenceExample | None:
    history = np.zeros((len(frame_ids), TOTAL_NODE_COUNT, len(CONTINUOUS_FEATURE_COLUMNS)), dtype=np.float32)
    ball_active = np.zeros((len(frame_ids), TOTAL_NODE_COUNT), dtype=np.float32)
    team_type_ids = np.zeros(TOTAL_NODE_COUNT, dtype=np.int64)
    position_ids = np.zeros(TOTAL_NODE_COUNT, dtype=np.int64)
    defender_mask = np.zeros(TOTAL_NODE_COUNT, dtype=bool)
    target_positions = np.zeros((TOTAL_NODE_COUNT, 2), dtype=np.float32)

    frame_tables: dict[int, tuple[dict[int, dict[str, object]], pd.DataFrame]] = {}
    relevant_rows = play_tracking.loc[play_tracking["frameId"].isin(frame_ids + [target_frame_id])]
    for frame_id, frame_rows in relevant_rows.groupby("frameId", sort=False):
        frame_tables[int(frame_id)] = _frame_row_lookup(frame_rows)

    for node in assignments:
        position_ids[node.node_index] = _node_position_id(node.role, position_vocab)
        team_type_ids[node.node_index] = TEAM_TYPE_TO_ID[node.node_type]
        defender_mask[node.node_index] = node.node_type == "defense"

    for time_index, frame_id in enumerate(frame_ids):
        if frame_id not in frame_tables:
            return None
        player_lookup, ball_rows = frame_tables[frame_id]
        ball_row = ball_rows.iloc[0] if not ball_rows.empty else None

        for node in assignments:
            if node.node_type == "ball":
                if ball_row is None:
                    continue
                history[time_index, node.node_index, :] = ball_row.loc[list(CONTINUOUS_FEATURE_COLUMNS)].to_numpy(
                    dtype=np.float32
                )
                ball_active[time_index, node.node_index] = 1.0
                continue

            if node.nfl_id not in player_lookup:
                return None
            row = player_lookup[node.nfl_id]
            history[time_index, node.node_index, :] = np.asarray(
                [row[column] for column in CONTINUOUS_FEATURE_COLUMNS],
                dtype=np.float32,
            )

    if target_frame_id not in frame_tables:
        return None

    target_player_lookup, target_ball_rows = frame_tables[target_frame_id]
    target_ball_row = target_ball_rows.iloc[0] if not target_ball_rows.empty else None

    for node in assignments:
        if node.node_type == "ball":
            if target_ball_row is not None:
                target_positions[node.node_index, :] = target_ball_row.loc[["x", "y"]].to_numpy(dtype=np.float32)
            continue

        if node.nfl_id not in target_player_lookup:
            return None
        row = target_player_lookup[node.nfl_id]
        target_positions[node.node_index, :] = np.asarray([row["x"], row["y"]], dtype=np.float32)

    metadata = SequenceMetadata(
        game_id=int(play_tracking["gameId"].iloc[0]),
        play_id=int(play_tracking["playId"].iloc[0]),
        start_frame_id=int(frame_ids[0]),
        target_frame_id=int(target_frame_id),
    )
    return SequenceExample(
        metadata=metadata,
        frame_ids=np.asarray(frame_ids + [target_frame_id], dtype=np.int64),
        node_assignments=assignments,
        history_continuous=history,
        position_ids=position_ids,
        team_type_ids=team_type_ids,
        history_ball_active=ball_active,
        target_positions=target_positions,
        defender_mask=defender_mask,
    )


def _consecutive_frame_ids(frame_ids: Iterable[int]) -> bool:
    ordered = list(frame_ids)
    return all(current - previous == 1 for previous, current in zip(ordered, ordered[1:]))


def build_sequence_examples(
    plays: pd.DataFrame,
    players: pd.DataFrame,
    tracking: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[list[SequenceExample], SequenceBuildSummary, dict[str, int]]:
    """Build dense sliding-window examples for baseline and graph training."""

    enriched_tracking = enrich_tracking_with_roles(tracking, plays, players)
    position_vocab = build_position_vocab(players)
    total_plays = int(enriched_tracking[["gameId", "playId"]].drop_duplicates().shape[0])
    summary = SequenceBuildSummary(total_plays=total_plays)
    examples: list[SequenceExample] = []

    for (_, _), play_tracking in enriched_tracking.groupby(["gameId", "playId"], sort=True):
        assignments = build_node_assignments(play_tracking)
        if assignments is None:
            if play_tracking["teamType"].eq("unknown").any():
                summary.dropped_plays_missing_roles += 1
            else:
                summary.dropped_plays_invalid_team_counts += 1
            continue

        ordered_frames = sorted(play_tracking["frameId"].dropna().astype(int).unique().tolist())
        if not _consecutive_frame_ids(ordered_frames):
            summary.dropped_windows_non_consecutive_frames += 1
            continue

        summary.valid_plays += 1
        max_start = len(ordered_frames) - config.lookback_frames - config.prediction_horizon + 1
        if max_start <= 0:
            continue

        for start_index in range(max_start):
            history_frame_ids = ordered_frames[start_index : start_index + config.lookback_frames]
            target_frame_id = ordered_frames[
                start_index + config.lookback_frames + config.prediction_horizon - 1
            ]
            example = _materialize_window(
                play_tracking=play_tracking,
                assignments=assignments,
                frame_ids=history_frame_ids,
                target_frame_id=target_frame_id,
                position_vocab=position_vocab,
            )
            if example is None:
                summary.dropped_windows_missing_player_rows += 1
                continue
            examples.append(example)

    summary.total_examples = len(examples)
    return examples, summary, position_vocab
