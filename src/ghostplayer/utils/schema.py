"""Shared schema constants and typed metadata for the GhostPlayer pipeline."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FIELD_LENGTH = 120.0
DEFAULT_FIELD_WIDTH = 53.3
DEFENDER_COUNT = 11
TOTAL_PLAYER_COUNT = 22
TOTAL_NODE_COUNT = 23
TARGET_COLUMNS = ("x", "y")
NODE_FEATURE_COLUMNS = (
    "x",
    "y",
    "s",
    "a",
    "o",
    "dir",
    "position",
    "teamType",
    "ballActive",
)

RAW_FILE_NAMES = {
    "games": "games.csv",
    "plays": "plays.csv",
    "players": "players.csv",
    "player_play": "player_play.csv",
    "tracking": "tracking.csv",
}

REQUIRED_PLAYS_COLUMNS = (
    "gameId",
    "playId",
)

REQUIRED_TRACKING_COLUMNS = (
    "gameId",
    "playId",
    "frameId",
    "event",
    "playDirection",
    "x",
    "y",
    "nflId",
)


@dataclass(frozen=True, slots=True)
class SequenceMetadata:
    """Minimal metadata needed to trace a serialized training example."""

    game_id: int
    play_id: int
    start_frame_id: int
    target_frame_id: int


@dataclass(frozen=True, slots=True)
class NodeAssignment:
    """Stable node ordering entry for a player or ball node."""

    node_index: int
    node_type: str
    nfl_id: int | None
    role: str
