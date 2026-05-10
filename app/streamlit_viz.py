"""Streamlit dashboard for GhostPlayer held-out play review."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ghostplayer.data.build_graphs import GraphDataset, load_graph_dataset

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
    }


def validate_paths(paths: dict[str, Path]) -> None:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(f"{name}: {paths[name]}" for name in missing)
        st.error(f"Missing dashboard artifact(s): {missing_text}")
        st.stop()


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


def main() -> None:
    st.set_page_config(page_title="GhostPlayer", layout="wide")
    root = project_root()
    paths = default_paths(root)
    validate_paths(paths)

    dataset = load_graphs(str(paths["graphs"]))
    prediction_errors = load_csv(str(paths["prediction_errors"]))
    per_play_errors = load_csv(str(paths["per_play_errors"]))
    metrics = load_metadata(str(paths["metrics"]))
    metadata = load_metadata(str(paths["metadata"]))
    position_names = inverse_vocab(metadata)

    options = play_options(per_play_errors)

    st.title("GhostPlayer")
    st.caption("Held-out pass-play movement: actual target players vs baseline and ST-GAT Ghost predictions.")

    with st.sidebar:
        st.header("Play Selection")
        sort_mode = st.selectbox("Sort plays by", ["ST-GAT improvement", "Worst ST-GAT ADE", "Worst baseline ADE", "Game/play order"])
        if sort_mode == "Worst ST-GAT ADE" and "st_gat" in options.columns:
            options = options.sort_values("st_gat", ascending=False)
        elif sort_mode == "Worst baseline ADE" and "baseline" in options.columns:
            options = options.sort_values("baseline", ascending=False)
        elif sort_mode == "Game/play order":
            options = options.sort_values(["game_id", "play_id"])
        else:
            options = options.sort_values("improvement", ascending=False, na_position="last")

        labels = [
            f"{row.game_id} / {row.play_id} | targets={int(row.target_node_count)} | "
            f"base={getattr(row, 'baseline', np.nan):.2f} | gat={getattr(row, 'st_gat', np.nan):.2f}"
            for row in options.itertuples(index=False)
        ]
        selected_label = st.selectbox("Held-out play", labels)
        selected_row = options.iloc[labels.index(selected_label)]
        example_index = int(selected_row["example_index"])

        st.divider()
        st.write("Artifacts")
        st.code(str(paths["graphs"].relative_to(root)))
        st.code(str(paths["prediction_errors"].relative_to(root)))

    selected_errors = per_play_errors.loc[per_play_errors["example_index"].eq(example_index)]
    metric_cards(metrics, selected_errors)

    st.subheader(f"Game {int(selected_row['game_id'])}, Play {int(selected_row['play_id'])}")
    st.info(
        "Read this as: the black ring marks the player(s) being predicted during the input frames. "
        "On the final Ghost target frame, black dots are actual locations, orange triangles are baseline ghosts, "
        "and cyan stars are ST-GAT ghosts."
    )
    st.plotly_chart(context_animation(dataset, prediction_errors, example_index, position_names), width="stretch")
    st.plotly_chart(prediction_plot(prediction_errors, example_index), width="stretch")

    with st.expander("Prediction Errors For Selected Play", expanded=False):
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
