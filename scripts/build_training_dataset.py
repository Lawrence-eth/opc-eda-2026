"""Build training dataset for Graph Transformer SP prediction.

Extracts Sequence Pair labels from 1M ground truth placements.
Output: HDF5 or NPZ file with (features, sp_plus, sp_minus) per sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "FloorSet"))
from iccad2026contest.iccad2026_evaluate import get_training_dataloader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sp_labels import extract_pairwise_relations, relations_to_sequence_pair


def extract_sp_from_ground_truth(fp_sol: torch.Tensor, area_targets: torch.Tensor) -> tuple:
    """Extract SP permutations from ground truth floorplan solution.

    Args:
        fp_sol: [N, 4] tensor of (w, h, x, y) per block.
        area_targets: [N] tensor of area targets (to determine valid blocks).

    Returns:
        (sp_plus, sp_minus) as Python lists, or (None, None) if extraction fails.
    """
    # Determine valid blocks (non-padded)
    valid_mask = area_targets > 0
    n = int(valid_mask.sum().item())
    if n < 2:
        return None, None

    # Extract rectangles for valid blocks
    rectangles = []
    for i in range(n):
        w, h, x, y = fp_sol[i].tolist()
        rectangles.append((w, h, x, y))

    try:
        relations = extract_pairwise_relations(rectangles)
        sp_plus, sp_minus = relations_to_sequence_pair(relations)
        return sp_plus, sp_minus
    except Exception as e:
        print(f"SP extraction failed: {e}")
        return None, None


def build_dataset(output_path: str, num_samples: int | None = None, batch_size: int = 64):
    """Build training dataset from FloorSet ground truth."""
    print(f"Building dataset: output={output_path}, samples={num_samples or 'all'}")

    dataloader = get_training_dataloader(
        data_path="../",
        batch_size=batch_size,
        num_samples=num_samples,
        shuffle=False,
    )

    dataset = []
    skipped = 0
    extracted = 0

    for batch_idx, batch in enumerate(dataloader):
        (area_target, b2b_conn, p2b_conn, pins_pos,
         constraints, tree_sol, fp_sol, metrics) = batch

        # Process each sample in the batch
        batch_size_actual = area_target.shape[0]
        for b in range(batch_size_actual):
            areas = area_target[b]
            fp = fp_sol[b]
            valid_mask = areas > 0
            n = int(valid_mask.sum().item())

            if n < 2:
                skipped += 1
                continue

            sp_plus, sp_minus = extract_sp_from_ground_truth(fp, areas)
            if sp_plus is None:
                skipped += 1
                continue

            # Store sample
            dataset.append({
                "n_blocks": n,
                "areas": areas[:n].numpy().astype(np.float32),
                "constraints": constraints[b, :n].numpy().astype(np.int32),
                "b2b_conn": b2b_conn[b].numpy(),
                "p2b_conn": p2b_conn[b].numpy(),
                "pins_pos": pins_pos[b].numpy().astype(np.float32),
                "sp_plus": np.array(sp_plus, dtype=np.int32),
                "sp_minus": np.array(sp_minus, dtype=np.int32),
                "fp_sol": fp[:n].numpy().astype(np.float32),
            })
            extracted += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}: extracted={extracted}, skipped={skipped}")

    print(f"Done. Extracted={extracted}, skipped={skipped}")

    # Save as NPZ
    np.savez_compressed(output_path, dataset=dataset)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="training_sp_dataset.npz")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_dataset(args.output, args.num_samples, args.batch_size)
