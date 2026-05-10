"""Compatibility helpers for older or derived Big Data Bowl tracking schemas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ghostplayer.data.load import RawTables
from ghostplayer.utils.schema import CONTINUOUS_FEATURE_COLUMNS


@dataclass(slots=True)
class LegacySchemaResult:
    """Normalized tables plus a report describing compatibility repairs."""

    tables: RawTables
    report: dict[str, object]


def normalize_legacy_schema(raw_tables: RawTables) -> LegacySchemaResult:
    """Normalize older/derived Kaggle releases into the project schema.

    Some public mirrors of older Big Data Bowl data omit ``gameId``, ``playId``,
    and ``nflId`` from tracking. Player ids can be repaired from display names.
    Missing play keys cannot be recovered exactly, so this creates synthetic
    play ids from timestamp segments when no original keys exist.
    """

    games = raw_tables.games.copy()
    plays = raw_tables.plays.copy()
    players = raw_tables.players.copy()
    tracking = raw_tables.tracking.copy()
    report: dict[str, object] = {}

    tracking, id_report = _repair_tracking_player_ids(tracking, players)
    report.update(id_report)

    tracking, club_report = _normalize_tracking_club(tracking, games)
    report.update(club_report)

    plays, defensive_report = _ensure_defensive_team(plays, games)
    report.update(defensive_report)

    tracking, play_key_report, synthetic_plays = _ensure_tracking_play_keys(tracking, games)
    report.update(play_key_report)
    if synthetic_plays is not None:
        plays = synthetic_plays

    for column in CONTINUOUS_FEATURE_COLUMNS:
        if column not in tracking.columns:
            tracking[column] = 0.0
            report.setdefault("continuous_columns_created", []).append(column)

    if "event" not in tracking.columns:
        tracking["event"] = ""
        report["event_created_empty"] = True

    return LegacySchemaResult(
        tables=RawTables(
            games=games,
            plays=plays,
            players=players,
            tracking=tracking,
            player_play=raw_tables.player_play,
        ),
        report=report,
    )


def _repair_tracking_player_ids(
    tracking: pd.DataFrame,
    players: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {}

    if "nflId" not in tracking.columns:
        if "displayName" not in tracking.columns:
            raise ValueError("tracking is missing both nflId and displayName; cannot identify players.")
        if not {"nflId", "displayName"}.issubset(players.columns):
            raise ValueError("players.csv must include nflId and displayName to repair tracking IDs.")

        name_lookup = (
            players[["nflId", "displayName"]]
            .dropna(subset=["nflId", "displayName"])
            .drop_duplicates(subset=["displayName"], keep="first")
        )
        tracking = tracking.merge(name_lookup, on="displayName", how="left")
        report["nflId_repaired_from_displayName"] = True
    elif "displayName" in tracking.columns and {"nflId", "displayName"}.issubset(players.columns):
        missing_id = tracking["nflId"].isna() & ~tracking["displayName"].fillna("").str.lower().eq("football")
        if missing_id.any():
            name_lookup = (
                players[["nflId", "displayName"]]
                .dropna(subset=["nflId", "displayName"])
                .drop_duplicates(subset=["displayName"], keep="first")
                .rename(columns={"nflId": "repairedNflId"})
            )
            tracking = tracking.merge(name_lookup, on="displayName", how="left")
            tracking.loc[missing_id, "nflId"] = tracking.loc[missing_id, "repairedNflId"]
            tracking = tracking.drop(columns=["repairedNflId"])
            report["nflId_missing_values_repaired"] = int(missing_id.sum())

    if "displayName" in tracking.columns:
        football = tracking["displayName"].fillna("").str.lower().eq("football")
        tracking.loc[football, "nflId"] = np.nan

    return tracking, report


def _normalize_tracking_club(
    tracking: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {}

    if "club" in tracking.columns:
        return tracking, report

    if "teamAbbr" in tracking.columns:
        return tracking.rename(columns={"teamAbbr": "club"}), {"club_source": "teamAbbr"}

    if "team" not in tracking.columns:
        tracking["club"] = np.nan
        return tracking, {"club_created_empty": True}

    team_values = set(tracking["team"].dropna().astype(str).str.lower().unique())
    if team_values.issubset({"home", "away", "football"}):
        game_cols = ["gameId", "homeTeamAbbr", "visitorTeamAbbr"]
        if not set(game_cols).issubset(games.columns):
            raise ValueError("games.csv needs homeTeamAbbr and visitorTeamAbbr to map home/away teams.")
        tracking = tracking.merge(games[game_cols].drop_duplicates(), on="gameId", how="left")
        team_lower = tracking["team"].fillna("").astype(str).str.lower()
        tracking["club"] = np.select(
            [team_lower.eq("football"), team_lower.eq("home"), team_lower.eq("away")],
            ["football", tracking["homeTeamAbbr"], tracking["visitorTeamAbbr"]],
            default=tracking["team"],
        )
        return tracking, {"club_source": "team plus games home/away mapping"}

    tracking["club"] = tracking["team"]
    return tracking, {"club_source": "team"}


def _ensure_defensive_team(
    plays: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "defensiveTeam" in plays.columns:
        return plays, {}

    if "possessionTeam" not in plays.columns:
        raise ValueError("plays.csv is missing possessionTeam; cannot derive defensiveTeam.")

    game_cols = ["gameId", "homeTeamAbbr", "visitorTeamAbbr"]
    if not set(game_cols).issubset(games.columns):
        raise ValueError("games.csv needs homeTeamAbbr and visitorTeamAbbr to derive defensiveTeam.")

    plays = plays.merge(games[game_cols].drop_duplicates(), on="gameId", how="left")
    plays["defensiveTeam"] = np.where(
        plays["possessionTeam"].eq(plays["homeTeamAbbr"]),
        plays["visitorTeamAbbr"],
        plays["homeTeamAbbr"],
    )
    plays = plays.drop(columns=["homeTeamAbbr", "visitorTeamAbbr"])
    return plays, {"defensiveTeam_derived_from_games": True}


def _ensure_tracking_play_keys(
    tracking: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame | None]:
    if {"gameId", "playId"}.issubset(tracking.columns):
        return tracking, {}, None

    if "time" not in tracking.columns:
        raise ValueError("tracking is missing gameId/playId and time; cannot create synthetic play keys.")
    if "sourceWeek" not in tracking.columns:
        raise ValueError("tracking is missing gameId/playId and sourceWeek; reload data with the updated loader.")

    keyed, synthetic_plays, report = _build_synthetic_play_keys(tracking, games)
    return keyed, report, synthetic_plays


def _build_synthetic_play_keys(
    tracking: pd.DataFrame,
    games: pd.DataFrame,
    *,
    gap_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    working = tracking.copy()
    working["time"] = pd.to_datetime(working["time"])

    if "club" not in working.columns:
        working["club"] = np.nan

    unique_times = (
        working[["sourceFile", "sourceWeek", "time"]]
        .drop_duplicates()
        .sort_values(["sourceFile", "time"])
        .reset_index(drop=True)
    )
    unique_times["gap"] = unique_times.groupby("sourceFile")["time"].diff().dt.total_seconds()
    unique_times["newSegment"] = unique_times["gap"].isna() | unique_times["gap"].gt(gap_seconds)
    unique_times["sourceSegmentId"] = unique_times.groupby("sourceFile")["newSegment"].cumsum().astype(int) - 1

    working = working.merge(
        unique_times[["sourceFile", "time", "sourceSegmentId"]],
        on=["sourceFile", "time"],
        how="left",
    )

    segment_rows: list[dict[str, object]] = []
    for (source_file, segment_id), group in working.groupby(["sourceFile", "sourceSegmentId"], sort=True):
        clubs = sorted(
            club
            for club in group["club"].dropna().astype(str).unique().tolist()
            if club and club.lower() != "football"
        )
        if len(clubs) < 2:
            continue

        source_week = int(group["sourceWeek"].dropna().iloc[0])
        candidate_games = games.loc[games["week"].eq(source_week)].copy()
        if "gameDate" in candidate_games.columns:
            segment_date = group["time"].min().date()
            game_dates = pd.to_datetime(candidate_games["gameDate"]).dt.date
            candidate_games = candidate_games.loc[game_dates.eq(segment_date)]

        candidate_games = candidate_games.loc[
            candidate_games.apply(
                lambda row: set(clubs[:2]).issubset({row["homeTeamAbbr"], row["visitorTeamAbbr"]}),
                axis=1,
            )
        ]
        if len(candidate_games) != 1:
            continue

        game = candidate_games.iloc[0]
        possession_team = clubs[0]
        defensive_team = clubs[1]
        segment_rows.append(
            {
                "sourceFile": source_file,
                "sourceSegmentId": int(segment_id),
                "sourceWeek": source_week,
                "segmentStartTime": group["time"].min(),
                "gameId": int(game["gameId"]),
                "possessionTeam": possession_team,
                "defensiveTeam": defensive_team,
            }
        )

    if not segment_rows:
        raise ValueError(
            "Could not reconstruct any gameId/playId values from tracking time and club. "
            "Use tracking CSVs that include gameId, playId, and team/club columns."
        )

    segment_map = pd.DataFrame(segment_rows).sort_values(["gameId", "segmentStartTime"]).reset_index(drop=True)
    segment_map["playOrder"] = segment_map.groupby("gameId").cumcount() + 1
    segment_map["playId"] = 900000 + segment_map["playOrder"]

    keyed = working.merge(
        segment_map[["sourceFile", "sourceSegmentId", "gameId", "playId"]],
        on=["sourceFile", "sourceSegmentId"],
        how="inner",
    )
    synthetic_plays = segment_map[["gameId", "playId", "possessionTeam", "defensiveTeam"]].copy()

    report = {
        "synthetic_play_keys_created": True,
        "synthetic_play_key_reason": "tracking missing original gameId/playId",
        "tracking_segments_detected": int(unique_times[["sourceFile", "sourceSegmentId"]].drop_duplicates().shape[0]),
        "tracking_segments_mapped_to_games": int(segment_map.shape[0]),
        "tracking_rows_kept_after_key_reconstruction": int(keyed.shape[0]),
        "tracking_rows_dropped_after_key_reconstruction": int(working.shape[0] - keyed.shape[0]),
    }
    return keyed, synthetic_plays, report
