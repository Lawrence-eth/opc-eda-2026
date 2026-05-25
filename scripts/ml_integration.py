"""Integration: Graph Transformer SP prediction + heuristic legalizer.

This module provides an alternative solve path for MyOptimizer:
1. Load trained GT model (if available).
2. Predict SP permutations from test case features.
3. Pack SP using deterministic legalizer (scripts/sp_labels.py).
4. Run existing heuristic refiners on the packed layout.

If the model is not available or prediction fails, falls back to
original heuristic solve.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add scripts directory to path
scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts_dir))

from gt_model import GraphTransformer
from sp_labels import pack_sequence_pair


class MLGuidedOptimizer:
    """Wrapper that adds ML-guided SP prediction to the heuristic optimizer."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint.get("args", {})

        self.model = GraphTransformer(
            d_model=args.get("d_model", 256),
            n_heads=args.get("n_heads", 8),
            n_layers=args.get("n_layers", 4),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_layout(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """Predict layout using ML-guided SP + deterministic packer."""
        try:
            preplaced = set()
            movable = []
            if constraints is not None and constraints.dim() > 1 and constraints.shape[1] > 1:
                for i in range(block_count):
                    if constraints[i, 1] != 0 and target_positions is not None:
                        if i < target_positions.shape[0] and target_positions[i, 2] > 0 and target_positions[i, 3] > 0:
                            preplaced.add(i)
                    else:
                        movable.append(i)
            else:
                movable = list(range(block_count))

            areas = area_targets.unsqueeze(0).to(self.device)
            cons = constraints.unsqueeze(0).to(self.device) if constraints is not None else None
            b2b = b2b_connectivity.to(self.device)
            p2b = p2b_connectivity.to(self.device)
            pins = pins_pos.to(self.device) if pins_pos is not None else None

            sp_plus, sp_minus = self.model.predict_permutations(
                areas, b2b, p2b, pins, cons
            )

            sp_plus_movable = [b for b in sp_plus if b in movable]
            sp_minus_movable = [b for b in sp_minus if b in movable]

            dims = self._estimate_dims(block_count, area_targets, constraints)

            movable_positions = pack_sequence_pair(
                sp_plus_movable, sp_minus_movable, len(movable), dims
            )

            if not movable_positions or len(movable_positions) != len(movable):
                return None

            final_positions = {}
            for i in preplaced:
                if i < target_positions.shape[0]:
                    final_positions[i] = (
                        float(target_positions[i, 0]),
                        float(target_positions[i, 1]),
                        float(target_positions[i, 2]),
                        float(target_positions[i, 3]),
                    )

            if preplaced:
                max_x = max((final_positions[i][0] + final_positions[i][2]) for i in preplaced)
                offset_x = max_x + 1.0
            else:
                offset_x = 0.0

            for idx, block_id in enumerate(movable):
                x, y, x2, y2 = movable_positions[idx]
                w = x2 - x
                h = y2 - y
                final_positions[block_id] = (x + offset_x, y, w, h)

            rects = [final_positions.get(i, (0.0, 0.0, 1.0, 1.0)) for i in range(block_count)]

            if self._has_overlap(rects):
                return None

            return rects

        except Exception:
            return None

    def _estimate_dims(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Dict[int, Tuple[float, float]]:
        """Estimate width/height for each block from area targets."""
        dims: Dict[int, Tuple[float, float]] = {}
        for i in range(block_count):
            area = float(area_targets[i].item())
            if area <= 0:
                continue
            # Use aspect ratio 1.0 for soft blocks, or from constraints for fixed blocks
            ar = 1.0
            if constraints is not None and constraints.dim() > 1:
                # Check if fixed block
                if constraints[i, 0] != 0 or constraints[i, 1] != 0:
                    # Fixed block: use exact dimensions from constraints if available
                    # For now, approximate as sqrt(area)
                    ar = 1.0
            w = math.sqrt(area * ar)
            h = area / w if w > 0 else 1.0
            dims[i] = (w, h)
        return dims

    @staticmethod
    def _has_overlap(rects: List[Tuple[float, float, float, float]]) -> bool:
        """Check if any two rectangles overlap."""
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if rects[i] is None or rects[j] is None:
                    continue
                x1, y1, w1, h1 = rects[i]
                x2, y2, w2, h2 = rects[j]
                if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                    return True
        return False


def integrate_ml_optimizer(
    optimizer_instance,
    model_path: str | Path = "/workspace/eda/opc-eda-2026/models/gt_model_50k.pt",
) -> bool:
    """Monkey-patch ML-guided solve into an existing MyOptimizer instance.

    Returns True if patching succeeded, False otherwise.
    """
    path = Path(model_path)
    if not path.exists():
        return False

    try:
        ml_opt = MLGuidedOptimizer(path)
    except Exception:
        return False

    # Store original solve
    original_solve = optimizer_instance.solve

    def ml_solve(
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:
        # Try ML prediction first
        ml_positions = ml_opt.predict_layout(
            block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints,
            target_positions
        )
        if ml_positions is not None:
            return ml_positions
        # Fall back to heuristic
        return original_solve(
            block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints, target_positions
        )

    optimizer_instance.solve = ml_solve
    return True
