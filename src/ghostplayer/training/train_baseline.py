"""Training entry point for the baseline model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ghostplayer.data.build_graphs import load_graph_dataset
from ghostplayer.eval.metrics import average_displacement_error
from ghostplayer.models.baseline import (
    DefenderBaselineDataset,
    DefenderMLP,
    graph_dataset_to_baseline_arrays,
)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(training):
        for features, batch_targets in loader:
            features = features.to(device)
            batch_targets = batch_targets.to(device)
            batch_predictions = model(features)
            loss = loss_fn(batch_predictions, batch_targets)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            predictions.append(batch_predictions.detach().cpu().numpy())
            targets.append(batch_targets.detach().cpu().numpy())

    if total_count == 0:
        raise ValueError("Cannot run an epoch over an empty loader.")

    predicted = np.concatenate(predictions, axis=0)
    actual = np.concatenate(targets, axis=0)
    return total_loss / total_count, average_displacement_error(predicted, actual)


def train_baseline(args: argparse.Namespace) -> None:
    train_graphs = load_graph_dataset(args.train_data)
    validation_graphs = load_graph_dataset(args.validation_data)

    train_arrays = graph_dataset_to_baseline_arrays(train_graphs)
    validation_arrays = graph_dataset_to_baseline_arrays(validation_graphs)

    train_loader = DataLoader(
        DefenderBaselineDataset(train_arrays),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        DefenderBaselineDataset(validation_arrays),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = resolve_device(args.device)
    model = DefenderMLP(
        input_dim=train_arrays.features.shape[1],
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_validation_ade = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "baseline.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ade = run_epoch(model, train_loader, optimizer, device)
        validation_loss, validation_ade = run_epoch(model, validation_loader, None, device)

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} train_ade={train_ade:.4f} "
            f"validation_loss={validation_loss:.6f} validation_ade={validation_ade:.4f}"
        )

        if validation_ade < best_validation_ade:
            best_validation_ade = validation_ade
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_arrays.features.shape[1],
                    "hidden_dims": tuple(args.hidden_dims),
                    "dropout": args.dropout,
                    "best_validation_ade": best_validation_ade,
                },
                checkpoint_path,
            )

    print(f"best_validation_ade={best_validation_ade:.4f}")
    print(f"checkpoint={checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the per-defender GhostPlayer baseline.")
    parser.add_argument("--train-data", type=Path, required=True, help="Path to train graph dataset .npz.")
    parser.add_argument("--validation-data", type=Path, required=True, help="Path to validation graph dataset .npz.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed/models"), help="Checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cpu, cuda.")
    return parser.parse_args()


def main() -> None:
    train_baseline(parse_args())


if __name__ == "__main__":
    main()
