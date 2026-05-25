"""Minimal training script for Graph Transformer - simplified for debug."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from gt_model import GraphTransformer


class SPDataset(Dataset):
    def __init__(self, data_path: str):
        data = np.load(data_path, allow_pickle=True)
        self.samples = data["dataset"].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "n": s["n_blocks"],
            "areas": torch.tensor(s["areas"], dtype=torch.float32),
            "constraints": torch.tensor(s["constraints"], dtype=torch.int64),
            "b2b_conn": torch.tensor(s["b2b_conn"], dtype=torch.float32),
            "p2b_conn": torch.tensor(s["p2b_conn"], dtype=torch.float32),
            "sp_plus": torch.tensor(s["sp_plus"], dtype=torch.int64),
            "sp_minus": torch.tensor(s["sp_minus"], dtype=torch.int64),
        }


def collate_fn(batch):
    max_n = max(item["n"] for item in batch)
    bsz = len(batch)
    areas = torch.full((bsz, max_n), -1.0, dtype=torch.float32)
    constraints = torch.zeros(bsz, max_n, 5, dtype=torch.int64)
    target_plus = torch.zeros(bsz, max_n, dtype=torch.float32)
    target_minus = torch.zeros(bsz, max_n, dtype=torch.float32)
    mask = torch.zeros(bsz, max_n, dtype=torch.bool)
    b2b_conns = []
    p2b_conns = []

    for i, item in enumerate(batch):
        n = item["n"]
        areas[i, :n] = item["areas"]
        constraints[i, :n] = item["constraints"]
        mask[i, :n] = True

        # Create normalized targets
        sp_p = item["sp_plus"]
        sp_m = item["sp_minus"]
        for rank, block_idx in enumerate(sp_p):
            target_plus[i, block_idx] = rank / max(n - 1, 1)
        for rank, block_idx in enumerate(sp_m):
            target_minus[i, block_idx] = rank / max(n - 1, 1)

        b2b_conns.append(item["b2b_conn"])
        p2b_conns.append(item["p2b_conn"])

    return {
        "areas": areas,
        "constraints": constraints,
        "b2b_conn": b2b_conns,
        "p2b_conn": p2b_conns,
        "target_plus": target_plus,
        "target_minus": target_minus,
        "mask": mask,
    }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        areas = batch["areas"].to(device)
        constraints = batch["constraints"].to(device)
        target_plus = batch["target_plus"].to(device)
        target_minus = batch["target_minus"].to(device)
        mask = batch["mask"].to(device)
        b2b_conn = batch["b2b_conn"]
        p2b_conn = batch["p2b_conn"]

        optimizer.zero_grad()
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
                None,
                constraints[i:i+1, :valid_n],
            )
            l = torch.nn.functional.mse_loss(pred_plus, target_plus[i:i+1, :valid_n])
            l += torch.nn.functional.mse_loss(pred_minus, target_minus[i:i+1, :valid_n])
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


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            areas = batch["areas"].to(device)
            constraints = batch["constraints"].to(device)
            target_plus = batch["target_plus"].to(device)
            target_minus = batch["target_minus"].to(device)
            mask = batch["mask"].to(device)
            b2b_conn = batch["b2b_conn"]
            p2b_conn = batch["p2b_conn"]

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
                    None,
                    constraints[i:i+1, :valid_n],
                )
                l = torch.nn.functional.mse_loss(pred_plus, target_plus[i:i+1, :valid_n])
                l += torch.nn.functional.mse_loss(pred_minus, target_minus[i:i+1, :valid_n])
                loss = loss + l
            if bsz > 0:
                loss = loss / bsz
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--output", default="models/gt_model.pt")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print(f"Device: {device}")

    ds = SPDataset(args.dataset)
    n_val = int(len(ds) * args.val_split)
    n_train = len(ds) - n_val
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Train: {n_train}, Val: {n_val}")

    model = GraphTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.1,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()
        print(f"  Train: {train_loss:.4f}, Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(), "val_loss": val_loss,
                       "args": vars(args)}, args.output)
            print(f"  Saved (val={val_loss:.4f})")

    print(f"\nDone. Best val: {best_val:.4f}")


if __name__ == "__main__":
    main()
