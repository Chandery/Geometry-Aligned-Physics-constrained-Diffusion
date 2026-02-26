import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DifferentiableDRRProjector(nn.Module):
    """
    Lightweight differentiable "DRR-like" projector via axis-wise integration.

    This is a pragmatic approximation (orthographic ray integration) intended to be
    differentiable and easy to plug into training as L_PF.
    """

    def __init__(
        self,
        axes: Tuple[int, ...] = (2, 4),
        normalize: str = "zscore",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.axes = tuple(int(a) for a in axes)
        self.normalize = str(normalize)
        self.eps = float(eps)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return x
        if self.normalize == "zscore":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            var = (x - mean).pow(2).mean(dim=(-2, -1), keepdim=True)
            return (x - mean) / torch.sqrt(var + self.eps)
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def forward(self, vol: torch.Tensor, target_hw: Tuple[int, int]) -> List[torch.Tensor]:
        """
        Args:
          vol: (B, 1, D, H, W)
          target_hw: (Ht, Wt) to resize projections (to match x-ray resolution)

        Returns:
          list of projections, each (B, 1, Ht, Wt)
        """
        if vol.dim() != 5:
            raise ValueError(f"Expected vol shape (B,1,D,H,W), got {tuple(vol.shape)}")

        projs: List[torch.Tensor] = []
        for axis in self.axes:
            # mean integration (stable scale vs sum)
            p = vol.mean(dim=axis)  # (B,1,*,*)
            # ensure it's (B,1,H,W) for interpolate; if needed, swap last 2 dims is unnecessary for MSE
            p = self._norm(p)
            p = F.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
            projs.append(p)
        return projs

