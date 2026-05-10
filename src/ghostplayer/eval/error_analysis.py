"""Error analysis utilities for GhostPlayer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ghostplayer.data.build_graphs import GraphDataset, load_graph_dataset
from ghostplayer.eval.inference import PredictionResult, predict_baseline, predict_stgat, resolve_device
from ghostplayer.eval.metrics import (
    displacement_errors,
    masked_error_summary,
    per_play_average_displacement_error,
)


def prediction_error_rows(
    dataset: GraphDataset,
    result: PredictionResult,
) -> pd.DataFrame:
    """Return one row per masked target-node prediction."""

    errors = displacement_errors(result.predictions, dataset.target_positions)
    rows: list[dict[str, float | int | str]] = []
    for example_index, node_index in zip(*np.where(dataset.defender_mask)):
        metadata = dataset.metadata[example_index]
        rows.append(
            {
                "model": result.model_name,
                "example_index": int(example_index),
                "game_id": int(metadata[0]),
                "play_id": int(metadata[1]),
                "start_frame_id": int(metadata[2]),
                "target_frame_id": int(metadata[3]),
                "node_index": int(node_index),
                "predicted_x": float(result.predictions[example_index, node_index, 0]),
                "predicted_y": float(result.predictions[example_index, node_index, 1]),
                "actual_x": float(dataset.target_positions[example_index, node_index, 0]),
                "actual_y": float(dataset.target_positions[example_index, node_index, 1]),
                "error": float(errors[example_index, node_index]),
            }
        )
    return pd.DataFrame(rows)


def per_play_error_rows(
    dataset: GraphDataset,
    result: PredictionResult,
) -> pd.DataFrame:
    """Return one row per play/example with masked ADE."""

    per_play_ade = per_play_average_displacement_error(
        result.predictions,
        dataset.target_positions,
        dataset.defender_mask,
    )
    target_counts = dataset.defender_mask.sum(axis=1)
    rows = []
    for example_index, metadata in enumerate(dataset.metadata):
        rows.append(
            {
                "model": result.model_name,
                "example_index": int(example_index),
                "game_id": int(metadata[0]),
                "play_id": int(metadata[1]),
                "start_frame_id": int(metadata[2]),
                "target_frame_id": int(metadata[3]),
                "target_node_count": int(target_counts[example_index]),
                "ade": float(per_play_ade[example_index]),
            }
        )
    return pd.DataFrame(rows)


def trajectory_horizon_error_rows(
    dataset: GraphDataset,
    result: PredictionResult,
) -> pd.DataFrame:
    """Return ADE by future output frame for a trajectory prediction result."""

    if dataset.target_trajectories is None or dataset.target_trajectory_mask is None:
        raise ValueError("Trajectory error rows require target_trajectories in the dataset.")

    errors = displacement_errors(result.predictions, dataset.target_trajectories)
    if dataset.target_trajectory_mask.shape != errors.shape:
        raise ValueError(
            "target_trajectory_mask shape must match trajectory errors, "
            f"got {dataset.target_trajectory_mask.shape} and {errors.shape}."
        )

    frame_ids = (
        dataset.target_frame_ids[0]
        if dataset.target_frame_ids is not None
        else np.arange(1, errors.shape[1] + 1, dtype=np.int64)
    )
    rows = []
    for horizon_index, output_frame_id in enumerate(frame_ids):
        frame_mask = dataset.target_trajectory_mask[:, horizon_index, :]
        frame_errors = errors[:, horizon_index, :][frame_mask]
        if frame_errors.size == 0:
            continue
        rows.append(
            {
                "model": result.model_name,
                "horizon_index": int(horizon_index),
                "output_frame_id": int(output_frame_id),
                "count": int(frame_errors.size),
                "ade": float(frame_errors.mean()),
                "median_error": float(np.median(frame_errors)),
                "p90_error": float(np.percentile(frame_errors, 90)),
            }
        )
    return pd.DataFrame(rows)


def compare_results(
    dataset: GraphDataset,
    results: list[PredictionResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Build comparison tables and metrics for multiple prediction results."""

    prediction_rows = pd.concat([prediction_error_rows(dataset, result) for result in results], ignore_index=True)
    per_play_rows = pd.concat([per_play_error_rows(dataset, result) for result in results], ignore_index=True)
    comparison_rows = []
    metrics: dict[str, dict[str, float | int]] = {}

    for result in results:
        summary = masked_error_summary(result.predictions, dataset.target_positions, dataset.defender_mask)
        metrics[result.model_name] = summary
        comparison_rows.append({"model": result.model_name, **summary})

    model_names = {result.model_name for result in results}
    if {"baseline", "st_gat"}.issubset(model_names):
        baseline_play = per_play_rows.loc[per_play_rows["model"].eq("baseline"), ["example_index", "ade"]]
        stgat_play = per_play_rows.loc[per_play_rows["model"].eq("st_gat"), ["example_index", "ade"]]
        merged = baseline_play.merge(stgat_play, on="example_index", suffixes=("_baseline", "_st_gat"))
        improvement = merged["ade_baseline"] - merged["ade_st_gat"]
        metrics["comparison"] = {
            "mean_ade_improvement": float(improvement.mean()),
            "median_ade_improvement": float(improvement.median()),
            "st_gat_win_rate": float((improvement > 0).mean()),
        }
        comparison_rows.append({"model": "st_gat_vs_baseline", **metrics["comparison"]})

    comparison = pd.DataFrame(comparison_rows)
    return prediction_rows, per_play_rows, comparison, metrics


def write_evaluation_outputs(
    *,
    output_dir: Path,
    prediction_rows: pd.DataFrame,
    per_play_rows: pd.DataFrame,
    comparison: pd.DataFrame,
    metrics: dict[str, dict[str, float | int]],
    top_k: int,
) -> None:
    """Write evaluation tables to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows.to_csv(output_dir / "prediction_errors.csv", index=False)
    per_play_rows.to_csv(output_dir / "per_play_errors.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    worst_plays = (
        per_play_rows.sort_values(["model", "ade"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    worst_predictions = (
        prediction_rows.sort_values(["model", "error"], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    worst_plays.to_csv(output_dir / "worst_plays.csv", index=False)
    worst_predictions.to_csv(output_dir / "worst_predictions.csv", index=False)

    with (output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)


def write_trajectory_evaluation_outputs(
    *,
    output_dir: Path,
    horizon_rows: pd.DataFrame,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    """Write trajectory evaluation tables to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    horizon_rows.to_csv(output_dir / "trajectory_horizon_errors.csv", index=False)
    with (output_dir / "trajectory_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)


def evaluate(args: argparse.Namespace) -> None:
    dataset = load_graph_dataset(args.data)
    device = resolve_device(args.device)

    if args.trajectory:
        if args.baseline_checkpoint is not None:
            raise ValueError("Trajectory evaluation currently supports ST-GAT only, not the one-frame baseline.")
        if args.stgat_checkpoint is None:
            raise ValueError("Trajectory evaluation requires --stgat-checkpoint.")
        if dataset.target_trajectories is None or dataset.target_trajectory_mask is None:
            raise ValueError("Trajectory evaluation requires a graph dataset with target_trajectories.")

        result = predict_stgat(
            dataset,
            args.stgat_checkpoint,
            batch_size=args.gnn_batch_size,
            device=device,
            return_trajectory=True,
        )
        metrics = {
            result.model_name: masked_error_summary(
                result.predictions,
                dataset.target_trajectories,
                dataset.target_trajectory_mask,
            )
        }
        horizon_rows = trajectory_horizon_error_rows(dataset, result)
        write_trajectory_evaluation_outputs(
            output_dir=args.output_dir,
            horizon_rows=horizon_rows,
            metrics=metrics,
        )
        print(pd.DataFrame([{"model": result.model_name, **metrics[result.model_name]}]).to_string(index=False))
        print(f"outputs={args.output_dir}")
        return

    results: list[PredictionResult] = []

    if args.baseline_checkpoint is not None:
        results.append(
            predict_baseline(
                dataset,
                args.baseline_checkpoint,
                batch_size=args.baseline_batch_size,
                device=device,
            )
        )

    if args.stgat_checkpoint is not None:
        results.append(
            predict_stgat(
                dataset,
                args.stgat_checkpoint,
                batch_size=args.gnn_batch_size,
                device=device,
            )
        )

    if not results:
        raise ValueError("At least one checkpoint must be provided.")

    prediction_rows, per_play_rows, comparison, metrics = compare_results(dataset, results)
    write_evaluation_outputs(
        output_dir=args.output_dir,
        prediction_rows=prediction_rows,
        per_play_rows=per_play_rows,
        comparison=comparison,
        metrics=metrics,
        top_k=args.top_k,
    )

    print(comparison.to_string(index=False))
    print(f"outputs={args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GhostPlayer baseline and ST-GAT checkpoints.")
    parser.add_argument("--data", type=Path, required=True, help="Graph dataset .npz to evaluate.")
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--stgat-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("processed/eval"))
    parser.add_argument("--baseline-batch-size", type=int, default=4096)
    parser.add_argument("--gnn-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trajectory", action="store_true", help="Evaluate a full-horizon ST-GAT trajectory checkpoint.")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
