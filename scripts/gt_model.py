"""Graph Transformer model for Sequence Pair topology prediction.

Architecture:
- Input: block features (areas, constraints) + graph edges (B2B/P2B connectivity)
- Encoder: 4-layer Graph Transformer with edge-weighted attention
- Output: Normalized positions in Γ+ and Γ- permutations
- Inference: argsort predicted positions to get SP permutations
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTransformer(nn.Module):
    """Predict Sequence Pair permutations from block features and connectivity.

    Args:
        d_model: Hidden dimension (default 256)
        n_heads: Number of attention heads (default 8)
        n_layers: Number of transformer layers (default 4)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Input embeddings
        self.area_embed = nn.Linear(1, d_model // 4)
        self.constraint_embed = nn.Embedding(10, d_model // 4)  # Small vocab for constraint codes
        self.type_embed = nn.Embedding(5, d_model // 4)  # Boundary type, cluster id, etc.
        self.edge_proj = nn.Linear(1, d_model // 4)

        # Combine all input features
        self.input_proj = nn.Linear(d_model, d_model)

        # Graph Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output heads: predict normalized position in [0, 1] for each permutation
        self.head_plus = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        self.head_minus = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        area_target: torch.Tensor,  # [bsz, n]
        b2b_conn: torch.Tensor,     # [bsz, n_edges, 3] (block_a, block_b, weight)
        p2b_conn: torch.Tensor,     # [bsz, n_edges, 3] (pin, block, weight)
        pins_pos: torch.Tensor,     # [bsz, n_pins, 2]
        constraints: torch.Tensor,    # [bsz, n, 5]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning normalized positions for Γ+ and Γ-.

        Returns:
            pos_plus: [bsz, n] predicted normalized positions in Γ+
            pos_minus: [bsz, n] predicted normalized positions in Γ-
        """
        bsz, n = area_target.shape

        # Build node features
        area_feat = self.area_embed(area_target.unsqueeze(-1))  # [bsz, n, d/4]

        # Simple constraint encoding (just use first few dims)
        # constraints: [fixed, preplaced, mib_group, cluster_group, boundary]
        constraint_ids = constraints.clamp(0, 9).long()  # Clamp to valid range
        constraint_feat = self.constraint_embed(constraint_ids).mean(dim=2)  # [bsz, n, d/4]

        # Combine features
        node_feat = torch.cat([area_feat, constraint_feat], dim=-1)  # [bsz, n, d/2]

        # Pad to d_model if needed
        if node_feat.shape[-1] < self.d_model:
            pad = torch.zeros(bsz, n, self.d_model - node_feat.shape[-1], device=node_feat.device)
            node_feat = torch.cat([node_feat, pad], dim=-1)
        else:
            node_feat = node_feat[:, :, :self.d_model]

        x = self.input_proj(node_feat)  # [bsz, n, d]

        # Create adjacency matrix from B2B connectivity
        # Simple approach: create dense adjacency matrix
        adj = torch.zeros(bsz, n, n, device=x.device)
        for b in range(bsz):
            edges = b2b_conn[b]
            if isinstance(edges, np.ndarray):
                edges = torch.from_numpy(edges)
            if edges.numel() == 0:
                continue
            if edges.dim() == 1 and edges.shape[0] == 3:
                edges = edges.unsqueeze(0)
            for edge in edges:
                if edge[0] == -1:
                    continue
                i, j = int(edge[0].item()), int(edge[1].item())
                w = abs(float(edge[2].item()))
                if 0 <= i < n and 0 <= j < n:
                    adj[b, i, j] = w
                    adj[b, j, i] = w

        # Apply Graph Transformer layers
        for layer in self.layers:
            x = layer(x, adj)

        x = self.norm(x)

        # Predict positions
        pos_plus = self.head_plus(x).squeeze(-1)  # [bsz, n]
        pos_minus = self.head_minus(x).squeeze(-1)  # [bsz, n]

        return pos_plus, pos_minus

    def predict_permutations(
        self,
        area_target: torch.Tensor,
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> Tuple[List[int], List[int]]:
        """Inference: predict SP permutations from block features.

        Returns:
            (sp_plus, sp_minus) as lists of block indices.
        """
        self.eval()
        with torch.no_grad():
            pos_plus, pos_minus = self.forward(area_target, b2b_conn, p2b_conn, pins_pos, constraints)

        # Get permutations by sorting predicted positions
        sp_plus = torch.argsort(pos_plus, dim=-1).squeeze(0).tolist()
        sp_minus = torch.argsort(pos_minus, dim=-1).squeeze(0).tolist()

        return sp_plus, sp_minus


class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer with edge-weighted attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Edge weight projection for attention bias
        self.edge_proj = nn.Linear(1, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bsz, n, d_model] node features
            adj: [bsz, n, n] edge weights (connectivity)
        """
        bsz, n, d = x.shape

        # Self-attention with edge bias
        residual = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(bsz, n, self.n_heads, self.head_dim).transpose(1, 2)  # [bsz, h, n, d/h]
        k = self.k_proj(x_norm).view(bsz, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, n, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [bsz, h, n, n]

        # Add edge bias
        edge_bias = self.edge_proj(adj.unsqueeze(-1)).permute(0, 3, 1, 2)  # [bsz, h, n, n]
        scores = scores + edge_bias

        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [bsz, h, n, d/h]
        out = out.transpose(1, 2).contiguous().view(bsz, n, d)
        out = self.out_proj(out)
        out = self.dropout(out)

        x = residual + out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


def sp_loss(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    target_plus: torch.Tensor,
    target_minus: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss between predicted and target normalized positions."""
    loss_plus = F.mse_loss(pred_plus, target_plus, reduction='none')
    loss_minus = F.mse_loss(pred_minus, target_minus, reduction='none')

    if mask is not None:
        loss_plus = loss_plus * mask.float()
        loss_minus = loss_minus * mask.float()
        n_valid = mask.sum().clamp(min=1)
        return (loss_plus.sum() + loss_minus.sum()) / n_valid
    else:
        return loss_plus.mean() + loss_minus.mean()
