"""Convert cached ANTs affine .mat transforms to 2D displacement fields."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def apply_affine_mat_to_image(image: np.ndarray, mat_path: str) -> np.ndarray:
    """
    Warp a single 2D image (H, W) with a cached ANTs affine .mat.

    Aligns ``prior`` (moving) into ``current`` (fixed) space using ANTs' own
    resampler - the same convention as cache_affine_transforms.py
    (fixed=current, moving=prior, transformlist=fwdtransforms).

    This warps the image directly (no coordinate-map -> displacement-field
    conversion), so it is exact for a 2D affine and sidesteps any 2D/3D
    dimensionality issues. Intended to run inside CPU dataloader workers.
    """
    import ants

    mat_path = str(Path(mat_path).resolve())
    if not Path(mat_path).is_file():
        raise FileNotFoundError(f"Affine transform not found: {mat_path}")

    img = np.ascontiguousarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"apply_affine_mat_to_image expects 2D (H, W), got {img.shape}")

    moving = ants.from_numpy(img)
    # fixed defines the output grid geometry; same shape/spacing/origin as moving.
    fixed = ants.from_numpy(np.zeros_like(img))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[mat_path],
        interpolator="linear",
    )
    return warped.numpy().astype(np.float32)


def apply_or_compute_affine(
    prior: np.ndarray,
    current: np.ndarray,
    mat_path: str,
    compute_missing: bool = False,
    save_computed: bool = True,
    random_seed: int = 7,
) -> np.ndarray:
    """
    Warp ``prior`` (moving) into ``current`` (fixed) space with a cached ANTs affine.

    If the ``.mat`` file is missing and ``compute_missing`` is True, the affine is
    estimated on the fly with ``ants.registration(type_of_transform="Affine")``
    (uint8 inputs, matching cache_affine_transforms.py) and, when ``save_computed``
    is set, written to ``mat_path`` so subsequent epochs reuse it. If registration
    fails, the prior is returned unchanged (identity) so training never crashes on
    a single bad pair.

    Both arrays are 2D (H, W). Returns the warped prior as float32 (H, W),
    preserving the input intensity scale (the transform is applied to the float
    image, not the uint8 one used only to estimate it).
    """
    import ants

    mat_path = str(Path(mat_path).resolve())
    prior = np.ascontiguousarray(prior, dtype=np.float32)
    current = np.ascontiguousarray(current, dtype=np.float32)
    if prior.ndim != 2 or current.ndim != 2:
        raise ValueError(
            f"apply_or_compute_affine expects 2D (H, W); got prior={prior.shape}, current={current.shape}"
        )

    moving = ants.from_numpy(prior)
    fixed = ants.from_numpy(current)

    if Path(mat_path).is_file():
        transformlist = [mat_path]
    else:
        if not compute_missing:
            raise FileNotFoundError(f"Affine transform not found: {mat_path}")
        fixed_u8 = ants.from_numpy(np.clip(current * 255.0, 0, 255).astype(np.uint8))
        moving_u8 = ants.from_numpy(np.clip(prior * 255.0, 0, 255).astype(np.uint8))
        reg = ants.registration(
            fixed=fixed_u8,
            moving=moving_u8,
            type_of_transform="Affine",
            random_seed=random_seed,
        )
        fwd = reg.get("fwdtransforms") or []
        if not fwd:
            # Registration produced no transform: fall back to identity.
            return prior
        transformlist = list(fwd)
        if save_computed:
            try:
                Path(mat_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fwd[0], mat_path)
            except Exception:
                pass

    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transformlist,
        interpolator="linear",
    )
    return warped.numpy().astype(np.float32)


@lru_cache(maxsize=256)
def _affine_mat_to_flow_numpy(mat_path: str, height: int, width: int) -> np.ndarray:
    """
    Build a displacement field (2, H, W): channel 0 = dy, channel 1 = dx.

    Uses ANTs to warp a coordinate map (prior -> fixed space), matching the registration benchmark
    moving=fixed convention from cache_affine_transforms.py.
    """
    import ants

    mat_path = str(Path(mat_path).resolve())
    if not Path(mat_path).is_file():
        raise FileNotFoundError(f"Affine transform not found: {mat_path}")

    h, w = int(height), int(width)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    coord = np.stack([yy, xx], axis=0)

    moving = ants.from_numpy(coord)
    fixed = ants.from_numpy(coord)
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[mat_path],
        interpolator="linear",
    )
    flow = warped.numpy() - coord
    return flow.astype(np.float32)


def affine_mat_to_flow(
    mat_path: str | Path,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return flow tensor [1, 2, H, W] on the requested device."""
    flow_np = _affine_mat_to_flow_numpy(str(mat_path), int(height), int(width))
    return torch.from_numpy(flow_np).unsqueeze(0).to(device=device, dtype=dtype)


def batch_affine_flows_from_paths(
    mat_paths: list[str],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Stack per-sample flows into [B, 2, H, W]."""
    flows = [
        affine_mat_to_flow(path, height, width, device, dtype)
        for path in mat_paths
    ]
    return torch.cat(flows, dim=0)


def compose_displacement_fields(flow1: torch.Tensor, flow2: torch.Tensor) -> torch.Tensor:
    """
    Compose displacement fields so that
    warp(warp(I, flow1), flow2) == warp(I, composed).

    ``flow1`` is applied first, ``flow2`` second (backward warp convention).
    """
    b, _, h, w = flow1.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow1.device, dtype=flow1.dtype),
        torch.arange(w, device=flow1.device, dtype=flow1.dtype),
        indexing="ij",
    )
    base = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

    pos2 = base + flow2
    pos2_norm = pos2.clone()
    pos2_norm[:, 0] = 2.0 * (pos2[:, 0] / max(h - 1, 1) - 0.5)
    pos2_norm[:, 1] = 2.0 * (pos2[:, 1] / max(w - 1, 1) - 0.5)
    grid = pos2_norm.permute(0, 2, 3, 1)[..., [1, 0]]

    flow1_at_pos2 = F.grid_sample(
        flow1, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return flow2 + flow1_at_pos2
