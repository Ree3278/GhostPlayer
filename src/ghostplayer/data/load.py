"""Raw dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ghostplayer.utils.config import DataPaths
from ghostplayer.utils.schema import RAW_FILE_NAMES


@dataclass(slots=True)
class RawTables:
    """Container for the Big Data Bowl source tables."""

    games: pd.DataFrame
    plays: pd.DataFrame
    players: pd.DataFrame
    tracking: pd.DataFrame
    player_play: pd.DataFrame | None = None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def load_tracking_files(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all tracking CSVs found under ``data_dir``."""

    tracking_paths = sorted(data_dir.glob("tracking_week_*.csv"))
    if not tracking_paths:
        fallback = data_dir / RAW_FILE_NAMES["tracking"]
        if fallback.exists():
            tracking_paths = [fallback]

    if not tracking_paths:
        raise FileNotFoundError(
            "No tracking CSVs found. Expected files named 'tracking_week_*.csv' or 'tracking.csv'."
        )

    tracking_frames = [pd.read_csv(path) for path in tracking_paths]
    return pd.concat(tracking_frames, ignore_index=True)


def load_raw_tables(paths: DataPaths) -> RawTables:
    """Load the raw Kaggle tables required by the preprocessing pipeline."""

    data_dir = paths.data_dir
    player_play_path = data_dir / RAW_FILE_NAMES["player_play"]

    return RawTables(
        games=_read_csv(data_dir / RAW_FILE_NAMES["games"]),
        plays=_read_csv(data_dir / RAW_FILE_NAMES["plays"]),
        players=_read_csv(data_dir / RAW_FILE_NAMES["players"]),
        tracking=load_tracking_files(data_dir),
        player_play=_read_csv(player_play_path) if player_play_path.exists() else None,
    )
