"""Resize displacement fields between image resolutions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_flow(flow: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize (B, 2, H, W) flow and scale displacement components."""
    _, _, h, w = flow.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return flow

    scaled = flow.clone()
    scaled[:, 0] *= th / h
    scaled[:, 1] *= tw / w
    return F.interpolate(scaled, size=(th, tw), mode="bilinear", align_corners=True)


def warp_with_flow(
    tensor: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp ``tensor`` with backward displacement field ``flow`` (B, 2, H, W)."""
    b, _, h, w = tensor.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=tensor.device, dtype=tensor.dtype),
        torch.arange(w, device=tensor.device, dtype=tensor.dtype),
        indexing="ij",
    )
    grid = torch.stack((yy, xx), dim=0).float().unsqueeze(0).expand(b, -1, -1, -1)
    new_grid = grid + flow.to(device=tensor.device, dtype=tensor.dtype)
    new_grid[:, 0, :, :] = 2.0 * (new_grid[:, 0, :, :] / max(h - 1, 1) - 0.5)
    new_grid[:, 1, :, :] = 2.0 * (new_grid[:, 1, :, :] / max(w - 1, 1) - 0.5)
    new_grid = new_grid.permute(0, 2, 3, 1)[..., [1, 0]]
    return F.grid_sample(
        tensor, new_grid, mode=mode, padding_mode=padding_mode, align_corners=True
    )


def predict_flow_at_image_size(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    model_hw: tuple[int, int] | None,
    predict_fn,
) -> torch.Tensor:
    """
    Run a fixed-resolution registration model and return flow at ``fixed`` size.

    ``predict_fn(fixed, moving)`` must return (B, 2, H', W') flow that warps
    ``moving`` into ``fixed`` space.
    """
    _, _, h, w = fixed.shape
    if model_hw is None or (h, w) == model_hw:
        return predict_fn(fixed, moving)

    mh, mw = model_hw
    fixed_s = F.interpolate(fixed, size=(mh, mw), mode="bilinear", align_corners=True)
    moving_s = F.interpolate(moving, size=(mh, mw), mode="bilinear", align_corners=True)
    flow_s = predict_fn(fixed_s, moving_s)
    return resize_flow(flow_s, (h, w))
