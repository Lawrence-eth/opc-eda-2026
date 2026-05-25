"""Inference wrapper for Graph Transformer SP prediction model.

Loads a trained GraphTransformer and predicts Sequence Pair permutations
from block features and connectivity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Import model architecture
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from gt_model import GraphTransformer


class SPModelInference:
    """Wrapper for loading and using the trained Graph Transformer model."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        args = checkpoint.get("args", {})

        self.model = GraphTransformer(
            d_model=args.get("d_model", 256),
            n_heads=args.get("n_heads", 8),
            n_layers=args.get("n_layers", 4),
            dropout=0.0,  # No dropout at inference
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()

    @torch.no_grad()
    def predict_sp(
        self,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Tuple[List[int], List[int]]:
        """Predict SP permutations for a single test case.

        Args:
            area_targets: [N] tensor of area targets.
            b2b_connectivity: [n_edges, 3] tensor of (block_a, block_b, weight).
            p2b_connectivity: [n_edges, 3] tensor of (pin, block, weight).
            pins_pos: [n_pins, 2] tensor of pin positions.
            constraints: [N, 5] tensor of constraint features.

        Returns:
            (sp_plus, sp_minus): Permutations as lists of block indices.
        """
        # Add batch dimension
        areas = area_targets.unsqueeze(0).to(self.device)  # [1, N]
        b2b = b2b_connectivity.to(self.device)
        p2b = p2b_connectivity.to(self.device)
        pins = pins_pos.to(self.device) if pins_pos is not None else None
        cons = constraints.unsqueeze(0).to(self.device)  # [1, N, 5]

        sp_plus, sp_minus = self.model.predict_permutations(
            areas, b2b, p2b, pins, cons
        )
        return sp_plus, sp_minus


def load_sp_model(model_path: str | Path = "/workspace/eda/opc-eda-2026/models/gt_model_50k.pt") -> Optional[SPModelInference]:
    """Load the trained SP model if available."""
    if not Path(model_path).exists():
        return None
    try:
        return SPModelInference(model_path)
    except Exception:
        return None
