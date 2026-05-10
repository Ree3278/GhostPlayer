"""Streamlit dashboard for GhostPlayer held-out play review."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ghostplayer.data.build_graphs import GraphDataset, load_graph_dataset
from ghostplayer.eval.inference import predict_stgat

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
TEAM_TYPE_NAMES = {
    0: "Offense",
    1: "Defense",
    2: "Ball landing",
}
TEAM_COLORS = {
    "Offense": "#1f77b4",
    "Defense": "#d62728",
    "Ball landing": "#2ca02c",
    "Target to predict": "#000000",
    "Target actual": "#111111",
    "Baseline ghost": "#ff7f0e",
    "ST-GAT ghost": "#17becf",
    "Baseline error": "#ffb36c",
    "ST-GAT error": "#76d7e6",
}


@st.cache_resource(show_spinner=False)
def load_graphs(path: str) -> GraphDataset:
    return load_graph_dataset(Path(path))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict[str, object]:
    with Path(path).open() as file:
        return json.load(file)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "graphs": root / "processed" / "test_graphs.npz",
        "prediction_errors": root / "processed" / "eval" / "test" / "prediction_errors.csv",
        "per_play_errors": root / "processed" / "eval" / "test" / "per_play_errors.csv",
        "metrics": root / "processed" / "eval" / "test" / "metrics.json",
        "metadata": root / "processed" / "dataset_metadata_2026.json",
        "trajectory_checkpoint": root / "processed" / "models_trajectory" / "st_gat.pt",
        "trajectory_metrics": root / "processed" / "eval" / "test_trajectory" / "trajectory_metrics.json",
        "trajectory_horizon_errors": root / "processed" / "eval" / "test_trajectory" / "trajectory_horizon_errors.csv",
    }


def validate_paths(paths: dict[str, Path], required_names: list[str] | None = None) -> None:
    required_names = required_names or list(paths)
    missing = [name for name in required_names if not paths[name].exists()]
    if missing:
        missing_text = ", ".join(f"{name}: {paths[name]}" for name in missing)
        st.error(f"Missing dashboard artifact(s): {missing_text}")
        st.stop()


@st.cache_resource(show_spinner="Running trajectory ST-GAT inference...")
def load_trajectory_predictions(graphs_path: str, checkpoint_path: str) -> np.ndarray:
    dataset = load_graph_dataset(Path(graphs_path))
    result = predict_stgat(
        dataset,
        Path(checkpoint_path),
        batch_size=128,
        return_trajectory=True,
    )
    return result.predictions


def inverse_vocab(metadata: dict[str, object]) -> dict[int, str]:
    vocab = metadata.get("position_vocab", {})
    return {int(value): str(key) for key, value in vocab.items()}


def play_options(per_play_errors: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        per_play_errors.pivot_table(
            index=["example_index", "game_id", "play_id", "target_node_count"],
            columns="model",
            values="ade",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if {"baseline", "st_gat"}.issubset(pivot.columns):
        pivot["improvement"] = pivot["baseline"] - pivot["st_gat"]
    else:
        pivot["improvement"] = np.nan
    return pivot.sort_values(["improvement", "st_gat"], ascending=[False, True], na_position="last")


def context_frame(dataset: GraphDataset, example_index: int, frame_offset: int, position_names: dict[int, str]) -> pd.DataFrame:
    rows = []
    history = dataset.history_continuous[example_index]
    team_type_ids = dataset.team_type_ids[example_index]
    position_ids = dataset.position_ids[example_index]
    mask = dataset.defender_mask[example_index]

    for node_index in range(history.shape[1]):
        x = float(history[frame_offset, node_index, 0])
        y = float(history[frame_offset, node_index, 1])
        if x == 0.0 and y == 0.0 and not bool(mask[node_index]):
            continue
        team_name = TEAM_TYPE_NAMES.get(int(team_type_ids[node_index]), "Unknown")
        rows.append(
            {
                "node_index": node_index,
                "x": x,
                "y": y,
                "team": team_name,
                "position": position_names.get(int(position_ids[node_index]), "UNK"),
                "target": bool(mask[node_index]),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(prediction_errors: pd.DataFrame, example_index: int) -> pd.DataFrame:
    play_rows = prediction_errors.loc[prediction_errors["example_index"].eq(example_index)].copy()
    actual_rows = (
        play_rows.drop_duplicates(subset=["node_index"])
        .assign(model="actual", x=lambda frame: frame["actual_x"], y=lambda frame: frame["actual_y"])
        [["model", "node_index", "x", "y", "error"]]
    )
    predicted_rows = play_rows.assign(x=lambda frame: frame["predicted_x"], y=lambda frame: frame["predicted_y"])[
        ["model", "node_index", "x", "y", "error"]
    ]
    return pd.concat([actual_rows, predicted_rows], ignore_index=True)


def empty_series() -> list[float]:
    return []


def get_target_trajectories(dataset: GraphDataset) -> np.ndarray | None:
    return getattr(dataset, "target_trajectories", None)


def get_target_trajectory_mask(dataset: GraphDataset) -> np.ndarray | None:
    return getattr(dataset, "target_trajectory_mask", None)


def get_target_frame_ids(dataset: GraphDataset) -> np.ndarray | None:
    return getattr(dataset, "target_frame_ids", None)


def trajectory_errors(dataset: GraphDataset, predictions: np.ndarray) -> np.ndarray:
    target_trajectories = get_target_trajectories(dataset)
    if target_trajectories is None:
        raise ValueError("Trajectory dashboard requires target_trajectories in the graph dataset.")
    return np.linalg.norm(predictions - target_trajectories, axis=-1)


def trajectory_play_options(dataset: GraphDataset, predictions: np.ndarray) -> pd.DataFrame:
    target_trajectory_mask = get_target_trajectory_mask(dataset)
    if target_trajectory_mask is None:
        raise ValueError("Trajectory dashboard requires target_trajectory_mask in the graph dataset.")

    errors = trajectory_errors(dataset, predictions)
    masked_errors = np.where(target_trajectory_mask, errors, np.nan)
    trajectory_ade = np.nanmean(masked_errors, axis=(1, 2))
    frame1_ade = np.nanmean(masked_errors[:, 0, :], axis=1)
    final_ade = []
    for example_index in range(masked_errors.shape[0]):
        active_frames = np.flatnonzero(target_trajectory_mask[example_index].any(axis=1))
        if len(active_frames) == 0:
            final_ade.append(np.nan)
            continue
        final_ade.append(float(np.nanmean(masked_errors[example_index, active_frames[-1], :])))

    return pd.DataFrame(
        {
            "example_index": np.arange(dataset.metadata.shape[0], dtype=int),
            "game_id": dataset.metadata[:, 0].astype(int),
            "play_id": dataset.metadata[:, 1].astype(int),
            "target_node_count": dataset.defender_mask.sum(axis=1).astype(int),
            "trajectory_ade": trajectory_ade,
            "frame1_ade": frame1_ade,
            "final_ade": final_ade,
        }
    )


def target_nodes(dataset: GraphDataset, example_index: int) -> list[int]:
    target_trajectory_mask = get_target_trajectory_mask(dataset)
    if target_trajectory_mask is not None:
        return [int(node) for node in np.flatnonzero(target_trajectory_mask[example_index].any(axis=0))]
    return [int(node) for node in np.flatnonzero(dataset.defender_mask[example_index])]


def target_frame_id(dataset: GraphDataset, example_index: int, horizon_index: int) -> int:
    target_frame_ids = get_target_frame_ids(dataset)
    if target_frame_ids is not None:
        return int(target_frame_ids[example_index, horizon_index])
    return horizon_index + 1


def path_segments(
    points: np.ndarray,
    mask: np.ndarray,
    nodes: list[int],
    horizon_index: int,
    *,
    frame_mode: str = "valid",
) -> pd.DataFrame:
    rows = []
    for node_index in nodes:
        if frame_mode == "valid":
            frame_indices = np.flatnonzero(mask[: horizon_index + 1, node_index])
        elif frame_mode == "missing":
            frame_indices = np.flatnonzero(~mask[: horizon_index + 1, node_index])
        elif frame_mode == "all":
            frame_indices = np.arange(horizon_index + 1)
        else:
            raise ValueError(f"Unknown frame_mode: {frame_mode}")

        for frame_index in frame_indices:
            rows.append(
                {
                    "x": float(points[frame_index, node_index, 0]),
                    "y": float(points[frame_index, node_index, 1]),
                    "node_index": node_index,
                    "frame_index": int(frame_index),
                }
            )
        rows.append({"x": np.nan, "y": np.nan, "node_index": node_index, "frame_index": np.nan})
    return pd.DataFrame(rows)


def current_points(
    points: np.ndarray,
    mask: np.ndarray,
    nodes: list[int],
    horizon_index: int,
    *,
    require_label: bool = True,
) -> pd.DataFrame:
    rows = []
    for node_index in nodes:
        if require_label and not bool(mask[horizon_index, node_index]):
            continue
        rows.append(
            {
                "node_index": node_index,
                "x": float(points[horizon_index, node_index, 0]),
                "y": float(points[horizon_index, node_index, 1]),
            }
        )
    return pd.DataFrame(rows)


def trajectory_error_lines(
    actual_points: np.ndarray,
    predicted_points: np.ndarray,
    mask: np.ndarray,
    nodes: list[int],
    horizon_index: int,
) -> pd.DataFrame:
    rows = []
    for node_index in nodes:
        if not bool(mask[horizon_index, node_index]):
            continue
        rows.extend(
            [
                {
                    "x": float(actual_points[horizon_index, node_index, 0]),
                    "y": float(actual_points[horizon_index, node_index, 1]),
                },
                {
                    "x": float(predicted_points[horizon_index, node_index, 0]),
                    "y": float(predicted_points[horizon_index, node_index, 1]),
                },
                {"x": np.nan, "y": np.nan},
            ]
        )
    return pd.DataFrame(rows)


def trajectory_trace_data(
    dataset: GraphDataset,
    predictions: np.ndarray,
    example_index: int,
    horizon_index: int,
) -> dict[str, pd.DataFrame]:
    target_trajectories = get_target_trajectories(dataset)
    target_trajectory_mask = get_target_trajectory_mask(dataset)
    if target_trajectories is None or target_trajectory_mask is None:
        raise ValueError("Trajectory dashboard requires trajectory labels.")

    nodes = target_nodes(dataset, example_index)
    actual_points = target_trajectories[example_index]
    predicted_points = predictions[example_index]
    mask = target_trajectory_mask[example_index]
    return {
        "actual_path": path_segments(actual_points, mask, nodes, horizon_index),
        "ghost_path": path_segments(predicted_points, mask, nodes, horizon_index),
        "ghost_forecast_path": path_segments(predicted_points, mask, nodes, horizon_index, frame_mode="missing"),
        "actual_current": current_points(actual_points, mask, nodes, horizon_index),
        "ghost_current": current_points(predicted_points, mask, nodes, horizon_index, require_label=False),
        "error_lines": trajectory_error_lines(actual_points, predicted_points, mask, nodes, horizon_index),
    }


def trajectory_plot(
    dataset: GraphDataset,
    predictions: np.ndarray,
    example_index: int,
    horizon_index: int,
    position_names: dict[int, str],
) -> go.Figure:
    last_context = context_frame(dataset, example_index, dataset.history_continuous.shape[1] - 1, position_names)
    trace_data = trajectory_trace_data(dataset, predictions, example_index, horizon_index)
    output_frame_id = target_frame_id(dataset, example_index, horizon_index)
    fig = go.Figure(layout=field_layout(f"Ghost Trajectory: output frame {output_frame_id}"))
    add_field_markings(fig)

    for team_name, marker_symbol in [("Offense", "circle"), ("Defense", "diamond"), ("Ball landing", "x")]:
        subset = last_context.loc[last_context["team"].eq(team_name)]
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers+text",
                text=subset["node_index"],
                textposition="top center",
                name=f"Last input {team_name}",
                marker=dict(size=10, color=TEAM_COLORS[team_name], symbol=marker_symbol, opacity=0.45),
                hovertemplate="last input node=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=trace_data["actual_path"].get("x", []),
            y=trace_data["actual_path"].get("y", []),
            mode="lines",
            name="Actual target path",
            line=dict(color="#222222", width=4),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trace_data["ghost_path"].get("x", []),
            y=trace_data["ghost_path"].get("y", []),
            mode="lines",
            name="ST-GAT ghost path (scored)",
            line=dict(color=TEAM_COLORS["ST-GAT ghost"], width=4, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trace_data["ghost_forecast_path"].get("x", []),
            y=trace_data["ghost_forecast_path"].get("y", []),
            mode="lines",
            name="ST-GAT forecast beyond labels",
            line=dict(color="rgba(23,190,207,0.35)", width=3, dash="dot"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trace_data["error_lines"].get("x", []),
            y=trace_data["error_lines"].get("y", []),
            mode="lines",
            name="Current error",
            line=dict(color=TEAM_COLORS["ST-GAT error"], width=2, dash="dot"),
            hoverinfo="skip",
        )
    )

    for key, label, color, symbol in [
        ("actual_current", "Actual now", TEAM_COLORS["Target actual"], "circle"),
        ("ghost_current", "Ghost now", TEAM_COLORS["ST-GAT ghost"], "star"),
    ]:
        subset = trace_data[key]
        fig.add_trace(
            go.Scatter(
                x=subset.get("x", []),
                y=subset.get("y", []),
                mode="markers+text",
                text=subset.get("node_index", []),
                textposition="top center",
                name=label,
                marker=dict(size=16, color=color, symbol=symbol, line=dict(width=1, color="#111")),
                hovertemplate="node=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    fit_field_to_points(
        fig,
        [
            last_context,
            trace_data["actual_path"],
            trace_data["ghost_path"],
            trace_data["ghost_forecast_path"],
            trace_data["actual_current"],
            trace_data["ghost_current"],
        ],
    )
    return fig


def trajectory_animation(
    dataset: GraphDataset,
    predictions: np.ndarray,
    example_index: int,
    position_names: dict[int, str],
    max_frames: int,
) -> go.Figure:
    horizon_count = min(max_frames, predictions.shape[1])
    fig = trajectory_plot(dataset, predictions, example_index, 0, position_names)
    frames = []
    for horizon_index in range(horizon_count):
        trace_data = trajectory_trace_data(dataset, predictions, example_index, horizon_index)
        frames.append(
            go.Frame(
                name=str(horizon_index + 1),
                data=[
                    fig.data[0],
                    fig.data[1],
                    fig.data[2],
                    go.Scatter(x=trace_data["actual_path"].get("x", []), y=trace_data["actual_path"].get("y", [])),
                    go.Scatter(x=trace_data["ghost_path"].get("x", []), y=trace_data["ghost_path"].get("y", [])),
                    go.Scatter(
                        x=trace_data["ghost_forecast_path"].get("x", []),
                        y=trace_data["ghost_forecast_path"].get("y", []),
                    ),
                    go.Scatter(x=trace_data["error_lines"].get("x", []), y=trace_data["error_lines"].get("y", [])),
                    go.Scatter(
                        x=trace_data["actual_current"].get("x", []),
                        y=trace_data["actual_current"].get("y", []),
                        text=trace_data["actual_current"].get("node_index", []),
                    ),
                    go.Scatter(
                        x=trace_data["ghost_current"].get("x", []),
                        y=trace_data["ghost_current"].get("y", []),
                        text=trace_data["ghost_current"].get("node_index", []),
                    ),
                ],
                layout=go.Layout(title_text=f"Ghost Trajectory: output frame {target_frame_id(dataset, example_index, horizon_index)}"),
            )
        )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 120, "redraw": False}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.15,
                "y": -0.08,
                "len": 0.8,
                "steps": [
                    {
                        "label": str(index + 1),
                        "method": "animate",
                        "args": [[str(index + 1)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    }
                    for index in range(horizon_count)
                ],
            }
        ],
    )
    return fig


def selected_trajectory_error_frame(dataset: GraphDataset, predictions: np.ndarray, example_index: int) -> pd.DataFrame:
    target_trajectory_mask = get_target_trajectory_mask(dataset)
    if target_trajectory_mask is None:
        raise ValueError("Trajectory dashboard requires target_trajectory_mask.")

    errors = trajectory_errors(dataset, predictions)[example_index]
    rows = []
    for horizon_index in range(errors.shape[0]):
        frame_errors = errors[horizon_index, target_trajectory_mask[example_index, horizon_index]]
        if frame_errors.size == 0:
            continue
        rows.append(
            {
                "horizon_index": horizon_index,
                "output_frame_id": target_frame_id(dataset, example_index, horizon_index),
                "ade": float(frame_errors.mean()),
                "max_error": float(frame_errors.max()),
            }
        )
    return pd.DataFrame(rows)


def trajectory_error_plot(
    dataset: GraphDataset,
    predictions: np.ndarray,
    example_index: int,
    horizon_index: int,
) -> go.Figure:
    frame = selected_trajectory_error_frame(dataset, predictions, example_index)
    selected_frame_id = target_frame_id(dataset, example_index, horizon_index)
    fig = go.Figure(layout=go.Layout(height=280, margin=dict(l=30, r=30, t=35, b=30)))
    fig.add_trace(
        go.Scatter(
            x=frame["output_frame_id"],
            y=frame["ade"],
            mode="lines+markers",
            name="ADE",
            line=dict(color=TEAM_COLORS["ST-GAT ghost"], width=3),
            hovertemplate="output frame=%{x}<br>ADE=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=selected_frame_id, line_width=2, line_dash="dash", line_color="#333333")
    fig.update_layout(
        title="Selected Play Error Over Future Frames",
        xaxis_title="Output frame",
        yaxis_title="ADE",
        plot_bgcolor="#fbfbfb",
        paper_bgcolor="#ffffff",
    )
    return fig


def target_overlay_frame(prediction_errors: pd.DataFrame, example_index: int) -> dict[str, pd.DataFrame]:
    frame = prediction_frame(prediction_errors, example_index)
    actual = frame.loc[frame["model"].eq("actual")]
    baseline = frame.loc[frame["model"].eq("baseline")]
    stgat = frame.loc[frame["model"].eq("st_gat")]
    return {
        "actual": actual,
        "baseline": baseline,
        "st_gat": stgat,
        "baseline_lines": error_line_frame(actual, baseline),
        "st_gat_lines": error_line_frame(actual, stgat),
    }


def error_line_frame(actual: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    actual_by_node = actual.set_index("node_index")
    rows = []
    for row in predicted.itertuples(index=False):
        if int(row.node_index) not in actual_by_node.index:
            continue
        actual_row = actual_by_node.loc[int(row.node_index)]
        rows.extend(
            [
                {"x": float(actual_row.x), "y": float(actual_row.y)},
                {"x": float(row.x), "y": float(row.y)},
                {"x": np.nan, "y": np.nan},
            ]
        )
    return pd.DataFrame(rows)


def field_layout(title: str) -> go.Layout:
    return go.Layout(
        title=title,
        xaxis=dict(range=[0, FIELD_LENGTH], title="Field x", constrain="domain", gridcolor="#e5e5e5"),
        yaxis=dict(range=[0, FIELD_WIDTH], title="Field y", scaleanchor="x", scaleratio=1, gridcolor="#e5e5e5"),
        plot_bgcolor="#f6fbf4",
        paper_bgcolor="#ffffff",
        height=620,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )


def add_field_markings(fig: go.Figure) -> None:
    for yard in range(10, 111, 10):
        fig.add_vline(x=yard, line_width=1, line_color="#d7ddd2")
    fig.add_hline(y=FIELD_WIDTH / 2, line_width=1, line_dash="dot", line_color="#c6d2bd")


def fit_field_to_points(fig: go.Figure, frames: list[pd.DataFrame]) -> None:
    x_values = []
    y_values = []
    for frame in frames:
        if frame.empty:
            continue
        x_values.extend(pd.to_numeric(frame.get("x", pd.Series(dtype=float)), errors="coerce").dropna().tolist())
        y_values.extend(pd.to_numeric(frame.get("y", pd.Series(dtype=float)), errors="coerce").dropna().tolist())

    if not x_values or not y_values:
        return

    fig.update_xaxes(range=[min(0.0, min(x_values) - 5.0), max(FIELD_LENGTH, max(x_values) + 5.0)])
    fig.update_yaxes(range=[min(0.0, min(y_values) - 5.0), max(FIELD_WIDTH, max(y_values) + 5.0)])


def context_animation(
    dataset: GraphDataset,
    prediction_errors: pd.DataFrame,
    example_index: int,
    position_names: dict[int, str],
) -> go.Figure:
    first = context_frame(dataset, example_index, 0, position_names)
    target_overlay = target_overlay_frame(prediction_errors, example_index)
    fig = go.Figure(layout=field_layout("Input Context + Ghost Target Frame"))
    add_field_markings(fig)

    for team_name, marker_symbol in [("Offense", "circle"), ("Defense", "diamond"), ("Ball landing", "x")]:
        subset = first.loc[first["team"].eq(team_name)]
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers+text",
                text=subset["node_index"],
                textposition="top center",
                name=team_name,
                marker=dict(size=12, color=TEAM_COLORS[team_name], symbol=marker_symbol, line=dict(width=1, color="#222")),
                customdata=np.stack([subset["position"], subset["target"]], axis=-1) if len(subset) else None,
                hovertemplate="node=%{text}<br>pos=%{customdata[0]}<br>target=%{customdata[1]}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )

    target_subset = first.loc[first["target"]]
    fig.add_trace(
        go.Scatter(
            x=target_subset["x"],
            y=target_subset["y"],
            mode="markers+text",
            text=target_subset["node_index"],
            textposition="bottom center",
            name="Target to predict",
            marker=dict(size=20, color="rgba(0,0,0,0)", symbol="circle-open", line=dict(width=3, color=TEAM_COLORS["Target to predict"])),
            hovertemplate="target node=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    for trace_name, color, symbol in [
        ("Target actual", TEAM_COLORS["Target actual"], "circle"),
        ("Baseline ghost", TEAM_COLORS["Baseline ghost"], "triangle-up"),
        ("ST-GAT ghost", TEAM_COLORS["ST-GAT ghost"], "star"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers+text",
                text=[],
                textposition="top center",
                name=trace_name,
                marker=dict(size=16, color=color, symbol=symbol, line=dict(width=1, color="#222")),
                hovertemplate="node=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    for trace_name, color in [("Baseline error", TEAM_COLORS["Baseline error"]), ("ST-GAT error", TEAM_COLORS["ST-GAT error"])]:
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name=trace_name,
                line=dict(color=color, width=2, dash="dot"),
                hoverinfo="skip",
            )
        )

    frames = []
    for frame_offset in range(dataset.history_continuous.shape[1]):
        frame = context_frame(dataset, example_index, frame_offset, position_names)
        target_frame = frame.loc[frame["target"]]
        frames.append(
            go.Frame(
                name=str(frame_offset + 1),
                data=[
                    go.Scatter(
                        x=frame.loc[frame["team"].eq(team_name), "x"],
                        y=frame.loc[frame["team"].eq(team_name), "y"],
                        text=frame.loc[frame["team"].eq(team_name), "node_index"],
                    )
                    for team_name in ["Offense", "Defense", "Ball landing"]
                ]
                + [
                    go.Scatter(x=target_frame["x"], y=target_frame["y"], text=target_frame["node_index"]),
                    go.Scatter(x=[], y=[], text=[]),
                    go.Scatter(x=[], y=[], text=[]),
                    go.Scatter(x=[], y=[], text=[]),
                    go.Scatter(x=[], y=[]),
                    go.Scatter(x=[], y=[]),
                ],
            )
        )

    last_context = context_frame(dataset, example_index, dataset.history_continuous.shape[1] - 1, position_names)
    last_target = last_context.loc[last_context["target"]]
    frames.append(
        go.Frame(
            name="Ghost target",
            data=[
                go.Scatter(
                    x=last_context.loc[last_context["team"].eq(team_name), "x"],
                    y=last_context.loc[last_context["team"].eq(team_name), "y"],
                    text=last_context.loc[last_context["team"].eq(team_name), "node_index"],
                    opacity=0.35,
                )
                for team_name in ["Offense", "Defense", "Ball landing"]
            ]
            + [
                go.Scatter(x=last_target["x"], y=last_target["y"], text=last_target["node_index"]),
                go.Scatter(
                    x=target_overlay["actual"]["x"],
                    y=target_overlay["actual"]["y"],
                    text=target_overlay["actual"]["node_index"],
                ),
                go.Scatter(
                    x=target_overlay["baseline"]["x"],
                    y=target_overlay["baseline"]["y"],
                    text=target_overlay["baseline"]["node_index"],
                ),
                go.Scatter(
                    x=target_overlay["st_gat"]["x"],
                    y=target_overlay["st_gat"]["y"],
                    text=target_overlay["st_gat"]["node_index"],
                ),
                go.Scatter(x=target_overlay["baseline_lines"].get("x", []), y=target_overlay["baseline_lines"].get("y", [])),
                go.Scatter(x=target_overlay["st_gat_lines"].get("x", []), y=target_overlay["st_gat_lines"].get("y", [])),
            ],
        )
    )
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": -0.08,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 450, "redraw": False}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.15,
                "y": -0.08,
                "len": 0.8,
                "steps": [
                    {
                        "label": str(index + 1),
                        "method": "animate",
                        "args": [[str(index + 1)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    }
                    for index in range(dataset.history_continuous.shape[1])
                ]
                + [
                    {
                        "label": "Ghost target",
                        "method": "animate",
                        "args": [["Ghost target"], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    }
                ],
            }
        ],
    )
    return fig


def prediction_plot(prediction_errors: pd.DataFrame, example_index: int) -> go.Figure:
    frame = prediction_frame(prediction_errors, example_index)
    fig = go.Figure(layout=field_layout("Target Frame: Actual vs Ghost"))
    add_field_markings(fig)
    overlay = target_overlay_frame(prediction_errors, example_index)

    for trace_name, color, line_frame in [
        ("Baseline error", TEAM_COLORS["Baseline error"], overlay["baseline_lines"]),
        ("ST-GAT error", TEAM_COLORS["ST-GAT error"], overlay["st_gat_lines"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=line_frame.get("x", []),
                y=line_frame.get("y", []),
                mode="lines",
                name=trace_name,
                line=dict(color=color, width=2, dash="dot"),
                hoverinfo="skip",
            )
        )

    styles = {
        "actual": ("Target actual", "circle"),
        "baseline": ("Baseline ghost", "triangle-up"),
        "st_gat": ("ST-GAT ghost", "star"),
    }
    for model_name, (label, symbol) in styles.items():
        subset = frame.loc[frame["model"].eq(model_name)]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers+text",
                text=subset["node_index"],
                textposition="top center",
                name=label,
                marker=dict(size=14, color=TEAM_COLORS[label], symbol=symbol, line=dict(width=1, color="#222")),
                customdata=subset[["error"]].to_numpy(),
                hovertemplate="node=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>error=%{customdata[0]:.2f}<extra></extra>",
            )
        )
    return fig


def metric_cards(metrics: dict[str, object], selected: pd.DataFrame) -> None:
    baseline = metrics.get("baseline", {})
    stgat = metrics.get("st_gat", {})
    comparison = metrics.get("comparison", {})
    selected_baseline = selected.loc[selected["model"].eq("baseline"), "ade"]
    selected_stgat = selected.loc[selected["model"].eq("st_gat"), "ade"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline ADE", f"{float(baseline.get('ade', 0.0)):.3f}")
    col2.metric("ST-GAT ADE", f"{float(stgat.get('ade', 0.0)):.3f}")
    col3.metric("Mean Improvement", f"{float(comparison.get('mean_ade_improvement', 0.0)):.3f}")
    col4.metric("ST-GAT Win Rate", f"{100 * float(comparison.get('st_gat_win_rate', 0.0)):.1f}%")

    col5, col6, col7 = st.columns(3)
    col5.metric("Selected Baseline ADE", f"{float(selected_baseline.iloc[0]):.3f}" if len(selected_baseline) else "n/a")
    col6.metric("Selected ST-GAT ADE", f"{float(selected_stgat.iloc[0]):.3f}" if len(selected_stgat) else "n/a")
    if len(selected_baseline) and len(selected_stgat):
        col7.metric("Selected Improvement", f"{float(selected_baseline.iloc[0] - selected_stgat.iloc[0]):.3f}")
    else:
        col7.metric("Selected Improvement", "n/a")


def trajectory_metric_cards(
    trajectory_metrics: dict[str, object],
    selected_row: pd.Series,
    selected_error_frame: pd.DataFrame,
) -> None:
    stgat_metrics = trajectory_metrics.get("st_gat", {})
    first_ade = float(selected_row["frame1_ade"])
    full_ade = float(selected_row["trajectory_ade"])
    final_ade = float(selected_row["final_ade"])
    worst_frame = selected_error_frame.sort_values("ade", ascending=False).head(1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trajectory Test ADE", f"{float(stgat_metrics.get('ade', 0.0)):.3f}")
    col2.metric("Selected Full-Path ADE", f"{full_ade:.3f}")
    col3.metric("Selected Frame-1 ADE", f"{first_ade:.3f}")
    col4.metric("Selected Final ADE", f"{final_ade:.3f}")

    if not worst_frame.empty:
        col5, col6 = st.columns(2)
        col5.metric("Worst Future Frame", f"{int(worst_frame.iloc[0]['output_frame_id'])}")
        col6.metric("Worst-Frame ADE", f"{float(worst_frame.iloc[0]['ade']):.3f}")


def main() -> None:
    st.set_page_config(page_title="GhostPlayer", layout="wide")
    root = project_root()
    paths = default_paths(root)
    validate_paths(
        paths,
        [
            "graphs",
            "prediction_errors",
            "per_play_errors",
            "metrics",
            "metadata",
            "trajectory_checkpoint",
            "trajectory_metrics",
        ],
    )

    dataset = load_graphs(str(paths["graphs"]))
    prediction_errors = load_csv(str(paths["prediction_errors"]))
    per_play_errors = load_csv(str(paths["per_play_errors"]))
    metrics = load_metadata(str(paths["metrics"]))
    trajectory_metrics = load_metadata(str(paths["trajectory_metrics"]))
    metadata = load_metadata(str(paths["metadata"]))
    position_names = inverse_vocab(metadata)
    if get_target_trajectories(dataset) is None or get_target_trajectory_mask(dataset) is None:
        st.error("The graph artifact does not include trajectory labels. Rerun the 2026 processing notebook.")
        st.stop()

    trajectory_predictions = load_trajectory_predictions(str(paths["graphs"]), str(paths["trajectory_checkpoint"]))
    options = trajectory_play_options(dataset, trajectory_predictions)

    st.title("GhostPlayer")
    st.caption("Held-out pass-play movement: actual target-player paths vs ST-GAT Ghost trajectories.")

    with st.sidebar:
        st.header("Play Selection")
        sort_mode = st.selectbox(
            "Sort plays by",
            ["Worst full-path ADE", "Best full-path ADE", "Worst final-frame ADE", "Worst frame-1 ADE", "Game/play order"],
        )
        if sort_mode == "Worst full-path ADE":
            options = options.sort_values("trajectory_ade", ascending=False)
        elif sort_mode == "Best full-path ADE":
            options = options.sort_values("trajectory_ade", ascending=True)
        elif sort_mode == "Worst final-frame ADE":
            options = options.sort_values("final_ade", ascending=False)
        elif sort_mode == "Worst frame-1 ADE":
            options = options.sort_values("frame1_ade", ascending=False)
        elif sort_mode == "Game/play order":
            options = options.sort_values(["game_id", "play_id"])

        labels = [
            f"{row.game_id} / {row.play_id} | targets={int(row.target_node_count)} | "
            f"path={row.trajectory_ade:.2f} | final={row.final_ade:.2f}"
            for row in options.itertuples(index=False)
        ]
        selected_label = st.selectbox("Held-out play", labels)
        selected_row = options.iloc[labels.index(selected_label)]
        example_index = int(selected_row["example_index"])
        max_horizon = trajectory_predictions.shape[1]
        horizon_index = st.slider(
            "Future output frame",
            min_value=1,
            max_value=max_horizon,
            value=min(10, max_horizon),
            help="Scrub the future path. Frame 1 is the old single-target prediction point.",
        ) - 1
        animation_frames = st.slider(
            "Animation frames",
            min_value=10,
            max_value=max_horizon,
            value=min(60, max_horizon),
            help="Limit animation length if the browser feels slow.",
        )

        st.divider()
        st.write("Artifacts")
        st.code(str(paths["graphs"].relative_to(root)))
        st.code(str(paths["trajectory_checkpoint"].relative_to(root)))

    selected_error_frame = selected_trajectory_error_frame(dataset, trajectory_predictions, example_index)
    trajectory_metric_cards(trajectory_metrics, selected_row, selected_error_frame)

    st.subheader(f"Game {int(selected_row['game_id'])}, Play {int(selected_row['play_id'])}")
    st.info(
        "Solid black path is the actual future path. Dashed cyan path is the ST-GAT Ghost path. "
        "Light dotted cyan is the model forecast after actual labels end for that play, so it is not scored. "
        "The faded dots are the last input-context frame the model saw before predicting the future."
    )
    trajectory_tab, frame1_tab, table_tab = st.tabs(["Trajectory Ghost", "Frame-1 Comparison", "Error Tables"])

    with trajectory_tab:
        st.plotly_chart(
            trajectory_plot(dataset, trajectory_predictions, example_index, horizon_index, position_names),
            width="stretch",
        )
        st.plotly_chart(
            trajectory_error_plot(dataset, trajectory_predictions, example_index, horizon_index),
            width="stretch",
        )
        with st.expander("Play Full Ghost Animation", expanded=False):
            st.plotly_chart(
                trajectory_animation(
                    dataset,
                    trajectory_predictions,
                    example_index,
                    position_names,
                    animation_frames,
                ),
                width="stretch",
            )

    with frame1_tab:
        selected_errors = per_play_errors.loc[per_play_errors["example_index"].eq(example_index)]
        metric_cards(metrics, selected_errors)
        st.plotly_chart(context_animation(dataset, prediction_errors, example_index, position_names), width="stretch")
        st.plotly_chart(prediction_plot(prediction_errors, example_index), width="stretch")

    with table_tab:
        st.write("Trajectory error by future frame")
        st.dataframe(selected_error_frame, width="stretch", hide_index=True)
        st.write("Frame-1 prediction errors")
        selected_prediction_errors = prediction_errors.loc[prediction_errors["example_index"].eq(example_index)]
        st.dataframe(
            selected_prediction_errors.sort_values(["model", "node_index"])[
                [
                    "model",
                    "node_index",
                    "predicted_x",
                    "predicted_y",
                    "actual_x",
                    "actual_y",
                    "error",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
