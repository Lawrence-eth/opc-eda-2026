"""Training script for Graph Transformer SP prediction model.

Usage:
    python scripts/train_gt_model.py --dataset data/training_sp_10k.npz --epochs 50
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from gt_model import GraphTransformer, sp_loss


class SPDataset(Dataset):
    """Dataset of (features, sp_plus, sp_minus) pairs."""

    def __init__(self, data_path: str):
        print(f"Loading dataset from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        self.samples = data["dataset"].tolist()
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        n = sample["n_blocks"]

        # Convert to tensors
        areas = torch.tensor(sample["areas"], dtype=torch.float32)
        constraints = torch.tensor(sample["constraints"], dtype=torch.int64)
        b2b_conn = torch.tensor(sample["b2b_conn"], dtype=torch.float32)
        p2b_conn = torch.tensor(sample["p2b_conn"], dtype=torch.float32)
        pins_pos = torch.tensor(sample["pins_pos"], dtype=torch.float32)
        sp_plus = torch.tensor(sample["sp_plus"], dtype=torch.int64)
        sp_minus = torch.tensor(sample["sp_minus"], dtype=torch.int64)

        # Create normalized target positions
        # For a permutation of n elements, normalize rank to [0, 1]
        target_plus = torch.zeros(n, dtype=torch.float32)
        target_minus = torch.zeros(n, dtype=torch.float32)
        for rank, block_idx in enumerate(sp_plus):
            target_plus[block_idx] = rank / max(n - 1, 1)
        for rank, block_idx in enumerate(sp_minus):
            target_minus[block_idx] = rank / max(n - 1, 1)

        return {
            "areas": areas,
            "constraints": constraints,
            "b2b_conn": b2b_conn,
            "p2b_conn": p2b_conn,
            "pins_pos": pins_pos,
            "target_plus": target_plus,
            "target_minus": target_minus,
            "n": n,
        }


def collate_fn(batch):
    """Collate variable-length sequences with padding."""
    max_n = max(item["n"] for item in batch)
    bsz = len(batch)

    # Pad all tensors to max_n
    areas = torch.full((bsz, max_n), -1.0, dtype=torch.float32)
    constraints = torch.zeros(bsz, max_n, 5, dtype=torch.int64)
    target_plus = torch.zeros(bsz, max_n, dtype=torch.float32)
    target_minus = torch.zeros(bsz, max_n, dtype=torch.float32)
    mask = torch.zeros(bsz, max_n, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["n"]
        areas[i, :n] = item["areas"]
        constraints[i, :n, :] = item["constraints"]
        target_plus[i, :n] = item["target_plus"]
        target_minus[i, :n] = item["target_minus"]
        mask[i, :n] = True

    # B2B and P2B connectivity - keep as lists of tensors (variable length)
    b2b_conns = [item["b2b_conn"] for item in batch]
    p2b_conns = [item["p2b_conn"] for item in batch]
    pins_poss = [item["pins_pos"] for item in batch]

    return {
        "areas": areas,
        "constraints": constraints,
        "b2b_conn": b2b_conns,
        "p2b_conn": p2b_conns,
        "pins_pos": pins_poss,
        "target_plus": target_plus,
        "target_minus": target_minus,
        "mask": mask,
    }


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        areas = batch["areas"].to(device)
        constraints = batch["constraints"].to(device)
        target_plus = batch["target_plus"].to(device)
        target_minus = batch["target_minus"].to(device)
        mask = batch["mask"].to(device)

        # B2B and P2B connectivity - keep on CPU, model will process per sample
        b2b_conn = batch["b2b_conn"]
        p2b_conn = batch["p2b_conn"]

        optimizer.zero_grad()

        # Forward pass
        # Note: model expects b2b_conn and p2b_conn as tensors, but they have variable shapes
        # We need to process per sample in the batch
        loss = torch.tensor(0.0, device=device)
        bsz = areas.shape[0]
        for i in range(bsz):
            valid_n = int(mask[i].sum().item())
            if valid_n < 2:
                continue

            pred_plus, pred_minus = model(
                areas[i:i+1, :valid_n],
                b2b_conn[i],
                p2b_conn[i],
                None,  # pins_pos not used in current model
                constraints[i:i+1, :valid_n],
            )

            l = sp_loss(
                pred_plus,
                pred_minus,
                target_plus[i:i+1, :valid_n],
                target_minus[i:i+1, :valid_n],
                mask[i:i+1, :valid_n],
            )
            loss = loss + l

        if bsz > 0:
            loss = loss / bsz

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        count += 1

        if count % 100 == 0:
            print(f"  Batch {count}: loss={loss.item():.4f}")

    return total_loss / max(count, 1)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            areas = batch["areas"].to(device)
            constraints = batch["constraints"].to(device)
            target_plus = batch["target_plus"].to(device)
            target_minus = batch["target_minus"].to(device)
            mask = batch["mask"].to(device)
            b2b_conn = batch["b2b_conn"]
            p2b_conn = batch["p2b_conn"]

            pins_pos = batch["pins_pos"]

            loss = torch.tensor(0.0, device=device)
            bsz = areas.shape[0]
            for i in range(bsz):
                valid_n = int(mask[i].sum().item())
                if valid_n < 2:
                    continue

                pred_plus, pred_minus = model(
                    areas[i:i+1, :valid_n],
                    b2b_conn[i],
                    p2b_conn[i],
                    None,  # pins_pos not used in current model
                    constraints[i:i+1, :valid_n],
                )

                l = sp_loss(
                    pred_plus,
                    pred_minus,
                    target_plus[i:i+1, :valid_n],
                    target_minus[i:i+1, :valid_n],
                    mask[i:i+1, :valid_n],
                )
                loss = loss + l

            if bsz > 0:
                loss = loss / bsz

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to NPZ dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output", default="models/gt_model.pt")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    full_dataset = SPDataset(args.dataset)
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = GraphTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, args.output)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
