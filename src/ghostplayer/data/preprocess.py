"""Core preprocessing transforms for pass-play sequence generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ghostplayer.utils.config import EventConfig, PreprocessingConfig, SplitConfig
from ghostplayer.utils.schema import (
    DEFAULT_FIELD_LENGTH,
    DEFAULT_FIELD_WIDTH,
    REQUIRED_PLAYS_COLUMNS,
    REQUIRED_TRACKING_COLUMNS,
)


@dataclass(slots=True)
class DropSummary:
    """Track how many plays are removed by each preprocessing filter."""

    missing_snap_event: int = 0
    missing_terminal_event: int = 0
    short_window: int = 0


@dataclass(slots=True)
class PreprocessingArtifacts:
    """Result of filtering and normalizing raw play data."""

    plays: pd.DataFrame
    tracking: pd.DataFrame
    play_windows: pd.DataFrame
    drop_summary: DropSummary


@dataclass(slots=True)
class DatasetSplits:
    """Disjoint splits keyed by ``gameId``."""

    train_game_ids: list[int]
    validation_game_ids: list[int]
    test_game_ids: list[int]


def validate_required_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def filter_qualifying_pass_plays(
    plays: pd.DataFrame,
    *,
    require_dropback: bool = True,
) -> pd.DataFrame:
    """Keep only pass plays with evidence of a forward pass attempt."""

    validate_required_columns(plays, REQUIRED_PLAYS_COLUMNS, "plays")

    mask = pd.Series(True, index=plays.index)

    if require_dropback and "isDropback" in plays.columns:
        mask &= plays["isDropback"].fillna(0).astype(int).eq(1)

    if "qbSpike" in plays.columns:
        mask &= plays["qbSpike"].fillna(0).astype(int).eq(0)

    if "qbKneel" in plays.columns:
        mask &= plays["qbKneel"].fillna(0).astype(int).eq(0)

    if "passResult" in plays.columns:
        mask &= plays["passResult"].notna()

    return plays.loc[mask].copy()


def normalize_field_orientation(
    tracking: pd.DataFrame,
    *,
    field_length: float = DEFAULT_FIELD_LENGTH,
    field_width: float = DEFAULT_FIELD_WIDTH,
) -> pd.DataFrame:
    """Flip left-moving plays so the offense always moves left to right."""

    validate_required_columns(tracking, REQUIRED_TRACKING_COLUMNS, "tracking")

    normalized = tracking.copy()
    moving_left = normalized["playDirection"].str.lower().eq("left")

    normalized.loc[moving_left, "x"] = field_length - normalized.loc[moving_left, "x"]
    normalized.loc[moving_left, "y"] = field_width - normalized.loc[moving_left, "y"]

    for column in ("o", "dir"):
        if column in normalized.columns:
            normalized.loc[moving_left, column] = (
                normalized.loc[moving_left, column].astype(float) + 180.0
            ) % 360.0

    normalized["playDirection"] = "right"
    return normalized


def _event_frame_lookup(
    tracking: pd.DataFrame,
    events: tuple[str, ...],
) -> pd.DataFrame:
    event_rows = tracking.loc[tracking["event"].isin(events), ["gameId", "playId", "frameId", "event"]]
    if event_rows.empty:
        return event_rows

    return (
        event_rows.sort_values(["gameId", "playId", "frameId"])
        .drop_duplicates(subset=["gameId", "playId"], keep="first")
        .reset_index(drop=True)
    )


def build_play_windows(
    tracking: pd.DataFrame,
    *,
    event_config: EventConfig,
    min_usable_frames: int,
) -> tuple[pd.DataFrame, DropSummary]:
    """Compute per-play frame windows from snap through terminal pass event."""

    validate_required_columns(tracking, REQUIRED_TRACKING_COLUMNS, "tracking")

    snap_frames = _event_frame_lookup(tracking, event_config.snap_events)
    terminal_frames = _event_frame_lookup(tracking, event_config.terminal_events)
    drop_summary = DropSummary()

    play_keys = tracking[["gameId", "playId"]].drop_duplicates()
    windows = play_keys.merge(
        snap_frames.rename(columns={"frameId": "startFrameId", "event": "startEvent"}),
        on=["gameId", "playId"],
        how="left",
    ).merge(
        terminal_frames.rename(columns={"frameId": "endFrameId", "event": "endEvent"}),
        on=["gameId", "playId"],
        how="left",
    )

    drop_summary.missing_snap_event = int(windows["startFrameId"].isna().sum())
    drop_summary.missing_terminal_event = int(windows["endFrameId"].isna().sum())

    windows = windows.loc[windows["startFrameId"].notna() & windows["endFrameId"].notna()].copy()
    windows["startFrameId"] = windows["startFrameId"].astype(int)
    windows["endFrameId"] = windows["endFrameId"].astype(int)
    windows = windows.loc[windows["endFrameId"] >= windows["startFrameId"]].copy()
    windows["usableFrames"] = windows["endFrameId"] - windows["startFrameId"] + 1

    short_mask = windows["usableFrames"] < min_usable_frames
    drop_summary.short_window = int(short_mask.sum())
    windows = windows.loc[~short_mask].reset_index(drop=True)
    return windows, drop_summary


def slice_tracking_to_windows(
    tracking: pd.DataFrame,
    play_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict tracking rows to the valid per-play frame interval."""

    merged = tracking.merge(
        play_windows[["gameId", "playId", "startFrameId", "endFrameId"]],
        on=["gameId", "playId"],
        how="inner",
    )
    frame_mask = merged["frameId"].between(merged["startFrameId"], merged["endFrameId"])
    return merged.loc[frame_mask].drop(columns=["startFrameId", "endFrameId"]).reset_index(drop=True)


def fill_tracking_numerics(
    tracking: pd.DataFrame,
    columns: tuple[str, ...] = ("s", "a", "o", "dir"),
) -> pd.DataFrame:
    """Apply explicit missing-value handling for key tracking numerics."""

    filled = tracking.copy()
    available_columns = [column for column in columns if column in filled.columns]
    if not available_columns:
        return filled

    filled[available_columns] = (
        filled.groupby(["gameId", "playId", "nflId"], dropna=False)[available_columns]
        .transform(lambda group: group.ffill().bfill())
        .fillna(0.0)
    )
    return filled


def preprocess_pass_plays(
    plays: pd.DataFrame,
    tracking: pd.DataFrame,
    config: PreprocessingConfig,
) -> PreprocessingArtifacts:
    """Run the Milestone 1 filtering pipeline over plays and tracking data."""

    qualifying_plays = filter_qualifying_pass_plays(
        plays,
        require_dropback=config.require_dropback,
    )
    play_keys = qualifying_plays[["gameId", "playId"]].drop_duplicates()
    filtered_tracking = tracking.merge(play_keys, on=["gameId", "playId"], how="inner")
    normalized_tracking = normalize_field_orientation(filtered_tracking)
    filled_tracking = fill_tracking_numerics(normalized_tracking)
    play_windows, drop_summary = build_play_windows(
        filled_tracking,
        event_config=config.events,
        min_usable_frames=config.minimum_usable_frames,
    )
    valid_play_keys = play_windows[["gameId", "playId"]].drop_duplicates()
    sliced_tracking = slice_tracking_to_windows(filled_tracking, play_windows)
    sliced_plays = qualifying_plays.merge(valid_play_keys, on=["gameId", "playId"], how="inner")

    return PreprocessingArtifacts(
        plays=sliced_plays,
        tracking=sliced_tracking,
        play_windows=play_windows,
        drop_summary=drop_summary,
    )


def split_game_ids(
    plays: pd.DataFrame,
    config: SplitConfig,
) -> DatasetSplits:
    """Split plays by unique ``gameId`` so play fragments never cross partitions."""

    if config.train_fraction <= 0 or config.validation_fraction < 0 or config.test_fraction < 0:
        raise ValueError("Split fractions must be non-negative, with train fraction greater than zero.")

    total_fraction = config.train_fraction + config.validation_fraction + config.test_fraction
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")

    validate_required_columns(plays, (config.group_key,), "plays")

    unique_game_ids = sorted(plays[config.group_key].dropna().unique().tolist())
    if not unique_game_ids:
        return DatasetSplits([], [], [])

    shuffled = pd.Series(unique_game_ids).sample(
        frac=1.0,
        random_state=config.random_seed,
    )
    game_ids = shuffled.tolist()

    train_end = int(len(game_ids) * config.train_fraction)
    validation_end = train_end + int(len(game_ids) * config.validation_fraction)

    train_ids = game_ids[:train_end]
    validation_ids = game_ids[train_end:validation_end]
    test_ids = game_ids[validation_end:]

    if not train_ids:
        raise ValueError("Split config produced an empty training set.")

    return DatasetSplits(
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        test_game_ids=test_ids,
    )
