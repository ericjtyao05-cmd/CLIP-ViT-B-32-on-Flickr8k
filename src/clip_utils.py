from __future__ import annotations
import torch


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [N, D], b: [M, D] -> sim: [N, M]
    assumes a and b are already l2-normalized
    """
    return a @ b.T
