"""Configuration objects for data and training pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataPaths:
    """Filesystem locations used throughout the project."""

    project_root: Path = Path(".")
    raw_dir_name: str = "data"
    processed_dir_name: str = "processed"

    @property
    def data_dir(self) -> Path:
        return self.project_root / self.raw_dir_name

    @property
    def processed_dir(self) -> Path:
        return self.project_root / self.processed_dir_name


@dataclass(slots=True)
class EventConfig:
    """Canonical event mapping used to define valid play windows."""

    snap_events: tuple[str, ...] = ("ball_snap",)
    terminal_events: tuple[str, ...] = (
        "pass_arrived",
        "pass_outcome_caught",
        "pass_outcome_incomplete",
        "interception",
    )


@dataclass(slots=True)
class SplitConfig:
    """Train/validation/test split settings."""

    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 7
    group_key: str = "gameId"


@dataclass(slots=True)
class PreprocessingConfig:
    """Milestone 1 preprocessing contract."""

    lookback_frames: int = 10
    prediction_horizon: int = 1
    require_dropback: bool = True
    minimum_usable_frames: int = 11
    offense_direction: str = "right"
    events: EventConfig = field(default_factory=EventConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
