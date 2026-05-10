"""Training entry point for the graph model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ghostplayer.data.build_graphs import load_graph_dataset
from ghostplayer.models.st_gat import GraphSequenceTorchDataset, STGAT, batch_to_device
from ghostplayer.training.losses import masked_ade, masked_mse_loss


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    target_mode: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_ade = 0.0
    total_masked_nodes = 0

    with torch.set_grad_enabled(training):
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device)
            predictions = model(
                batch.history_continuous,
                batch.position_ids,
                batch.team_type_ids,
                batch.history_ball_active,
            )
            if target_mode == "trajectory":
                if batch.target_trajectories is None or batch.target_trajectory_mask is None:
                    raise ValueError("Trajectory training requires target_trajectories in the graph dataset.")
                targets = batch.target_trajectories
                target_mask = batch.target_trajectory_mask
            else:
                targets = batch.target_positions
                target_mask = batch.prediction_mask

            loss = masked_mse_loss(predictions, targets, target_mask)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            masked_items = int(target_mask.sum().item())
            total_loss += float(loss.item()) * masked_items
            total_ade += float(masked_ade(predictions, targets, target_mask).item()) * masked_items
            total_masked_nodes += masked_items

    if total_masked_nodes == 0:
        raise ValueError("Cannot run an epoch with zero masked prediction nodes.")

    return total_loss / total_masked_nodes, total_ade / total_masked_nodes


def train_gnn(args: argparse.Namespace) -> None:
    train_graphs = load_graph_dataset(args.train_data)
    validation_graphs = load_graph_dataset(args.validation_data)
    target_mode = args.target_mode
    if target_mode == "auto":
        target_mode = "trajectory" if train_graphs.target_trajectories is not None else "single"

    output_horizon = 1
    if target_mode == "trajectory":
        if train_graphs.target_trajectories is None:
            raise ValueError("Requested trajectory training, but train_data has no target_trajectories.")
        if validation_graphs.target_trajectories is None:
            raise ValueError("Requested trajectory training, but validation_data has no target_trajectories.")
        output_horizon = int(train_graphs.target_trajectories.shape[1])
        validation_horizon = int(validation_graphs.target_trajectories.shape[1])
        if validation_horizon != output_horizon:
            raise ValueError(
                "Train and validation trajectory horizons must match, "
                f"got {output_horizon} and {validation_horizon}."
            )

    train_loader = DataLoader(
        GraphSequenceTorchDataset(train_graphs),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        GraphSequenceTorchDataset(validation_graphs),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = resolve_device(args.device)
    model = STGAT.from_graph_dataset(
        train_graphs,
        hidden_dim=args.hidden_dim,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        temporal_hidden_dim=args.temporal_hidden_dim,
        position_embedding_dim=args.position_embedding_dim,
        team_embedding_dim=args.team_embedding_dim,
        dropout=args.dropout,
        output_horizon=output_horizon,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "st_gat.pt"
    best_validation_ade = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ade = run_epoch(model, train_loader, optimizer, device, target_mode)
        validation_loss, validation_ade = run_epoch(model, validation_loader, None, device, target_mode)

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
                    "hidden_dim": args.hidden_dim,
                    "gat_layers": args.gat_layers,
                    "gat_heads": args.gat_heads,
                    "temporal_hidden_dim": args.temporal_hidden_dim,
                    "position_embedding_dim": args.position_embedding_dim,
                    "team_embedding_dim": args.team_embedding_dim,
                    "dropout": args.dropout,
                    "num_positions": int(train_graphs.position_ids.max()) + 1,
                    "num_team_types": int(train_graphs.team_type_ids.max()) + 1,
                    "best_validation_ade": best_validation_ade,
                    "target_mode": target_mode,
                    "output_horizon": output_horizon,
                },
                checkpoint_path,
            )

    print(f"best_validation_ade={best_validation_ade:.4f}")
    print(f"checkpoint={checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GhostPlayer ST-GAT graph model.")
    parser.add_argument("--train-data", type=Path, required=True, help="Path to train graph dataset .npz.")
    parser.add_argument("--validation-data", type=Path, required=True, help="Path to validation graph dataset .npz.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed/models"), help="Checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--temporal-hidden-dim", type=int, default=128)
    parser.add_argument("--position-embedding-dim", type=int, default=8)
    parser.add_argument("--team-embedding-dim", type=int, default=4)
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cpu, cuda.")
    parser.add_argument(
        "--target-mode",
        choices=("auto", "single", "trajectory"),
        default="auto",
        help="Train on one-frame targets or full future trajectories when available.",
    )
    return parser.parse_args()


def main() -> None:
    train_gnn(parse_args())


if __name__ == "__main__":
    main()
