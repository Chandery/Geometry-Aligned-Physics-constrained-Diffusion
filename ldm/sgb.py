import torch
import torch.nn as nn
from typing import Optional, Tuple


def _make_3d_coord_grid(d: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """
    Returns a normalized 3D coordinate grid in [-1, 1] with shape (d*h*w, 3),
    ordered as (y, x, z) to match (H, W, D) convention used in this repo.
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    zs = torch.linspace(-1.0, 1.0, d, device=device, dtype=dtype)
    yy, xx, zz = torch.meshgrid(ys, xs, zs, indexing="ij")  # (H, W, D)
    coords = torch.stack([yy, xx, zz], dim=-1).reshape(h * w * d, 3)
    return coords


class SGBModule(nn.Module):
    """
    Spatial Geometric Bridging (SGB):
    Use coordinate-aware cross-attention to map 2D features (K,V) to 3D conditional features queried by 3D coords (Q).

    Input:
      - feat2d: (B, C, H, W)
    Output:
      - feat3d: (B, C, H, W, D)  where D == target_d
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        target_d: int,
        target_hw: Optional[int] = None,
        coord_mlp_hidden: int = 64,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.target_d = int(target_d)
        self.target_hw = None if target_hw is None else int(target_hw)

        self.coord_mlp = nn.Sequential(
            nn.Linear(3, coord_mlp_hidden),
            nn.SiLU(),
            nn.Linear(coord_mlp_hidden, self.embed_dim),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(self.embed_dim)

        self._cached_grid = None  # (H*W*D, 3) on correct device/dtype
        self._cached_shape = None  # (D, H, W)

    def _get_grid(self, d: int, h: int, w: int, device, dtype) -> torch.Tensor:
        shape = (d, h, w)
        if (
            self._cached_grid is None
            or self._cached_shape != shape
            or self._cached_grid.device != device
            or self._cached_grid.dtype != dtype
        ):
            self._cached_grid = _make_3d_coord_grid(d=d, h=h, w=w, device=device, dtype=dtype)
            self._cached_shape = shape
        return self._cached_grid

    def forward(self, feat2d: torch.Tensor) -> torch.Tensor:
        if feat2d.dim() != 4:
            raise ValueError(f"SGBModule expects 4D (B,C,H,W) feat2d, got shape {tuple(feat2d.shape)}")

        b, c, h_in, w_in = feat2d.shape
        if c != self.embed_dim:
            raise ValueError(f"SGBModule embed_dim={self.embed_dim} but got C={c} in feat2d")

        h = self.target_hw if self.target_hw is not None else h_in
        w = self.target_hw if self.target_hw is not None else w_in
        d = self.target_d

        # If needed, caller should resize feat2d externally to (H,W) that matches the latent grid.
        if h_in != h or w_in != w:
            raise ValueError(f"SGBModule got feat2d HW=({h_in},{w_in}) but expected ({h},{w})")

        # K,V: flatten 2D tokens
        kv = feat2d.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, HW, C)

        # Q: coordinate embedding on 3D grid
        coords = self._get_grid(d=d, h=h, w=w, device=feat2d.device, dtype=feat2d.dtype)  # (HWD, 3)
        q = self.coord_mlp(coords).unsqueeze(0).expand(b, -1, -1)  # (B, HWD, C)

        out, _ = self.attn(q, kv, kv, need_weights=False)  # (B, HWD, C)
        out = self.out_norm(out)

        out = out.reshape(b, h, w, d, c).permute(0, 4, 1, 2, 3).contiguous()  # (B, C, H, W, D)
        return out

