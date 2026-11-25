"""Registration backends for LMV-Net longitudinal feature alignment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn

from registration.affine_flow import batch_affine_flows_from_paths
from registration.flow_ops import predict_flow_at_image_size, warp_with_flow


def _resolve_mat_paths(
    affine_mat_paths: Sequence[str] | None,
    batch_size: int,
    view: str,
) -> list[str] | None:
    if affine_mat_paths is None:
        return None
    paths = [str(p) for p in affine_mat_paths]
    if len(paths) != batch_size:
        raise ValueError(
            f"Expected {batch_size} affine mat paths for {view}, got {len(paths)}"
        )
    if not any(paths):
        return None
    missing = [p for p in paths if not p]
    if missing:
        raise ValueError(f"Missing affine mat path for {view} in batch")
    return paths


def _batch_affine_flow(
    fixed: torch.Tensor,
    mat_paths: list[str],
) -> torch.Tensor:
    _, _, h, w = fixed.shape
    return batch_affine_flows_from_paths(
        mat_paths, h, w, fixed.device, fixed.dtype
    )


def _zero_flow_like(fixed: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        fixed.size(0), 2, fixed.size(2), fixed.size(3),
        device=fixed.device, dtype=fixed.dtype,
    )


def _ants_prewarp_moving(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    mat_paths: list[str],
) -> torch.Tensor:
    """Apply cached ANTs affine to prior/moving at image resolution (benchmark-style)."""
    flow_ants = _batch_affine_flow(fixed, mat_paths)
    return warp_with_flow(moving, flow_ants)


class RegistrationBackend(ABC, nn.Module):
    """
    Predict per-view 2D displacement fields at image resolution (B, 2, H, W).

    When ``compose_ants_affine`` is enabled, backends also return ``ants_cc`` /
    ``ants_mlo`` and predict DL flows on ANTs-pre-warped priors (residual only).
    """

    compose_ants_affine: bool = False

    @abstractmethod
    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Return per-view residual flows in ``cc`` / ``mlo``.

        With ANTs pre-warp enabled, also include ``ants_cc`` / ``ants_mlo``.
        """


class UnregisteredBackend(RegistrationBackend):
    """Identity registration - zero displacement (prior features unwarped)."""

    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        return {
            "cc": _zero_flow_like(fixed_cc),
            "mlo": _zero_flow_like(fixed_mlo),
        }


class AntsAffineBackend(RegistrationBackend):
    """Cached ANTs affine only."""

    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        b = fixed_cc.size(0)
        paths_cc = _resolve_mat_paths(affine_mat_cc, b, "CC")
        paths_mlo = _resolve_mat_paths(affine_mat_mlo, b, "MLO")
        if not paths_cc or not paths_mlo:
            raise ValueError(
                "ants_affine requires affine_mat_cc and affine_mat_mlo in each batch"
            )
        return {
            "cc": _zero_flow_like(fixed_cc),
            "mlo": _zero_flow_like(fixed_mlo),
            "ants_cc": _batch_affine_flow(fixed_cc, paths_cc),
            "ants_mlo": _batch_affine_flow(fixed_mlo, paths_mlo),
        }


class MammoRegNetBackend(RegistrationBackend):
    """Single-view MammoRegNet; optional ANTs image pre-warp before DL (residual flow)."""

    def __init__(self, model: nn.Module, compose_ants_affine: bool = False):
        super().__init__()
        self.model = model
        self.compose_ants_affine = compose_ants_affine
        self.model.requires_grad_(False)
        self.model.eval()

    def _flow_for_pair(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mat_paths: list[str] | None,
        view: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        moving_in = moving
        flow_ants = None
        if self.compose_ants_affine:
            if not mat_paths:
                raise ValueError(
                    f"compose_ants_affine=True but no affine mat paths for {view}"
                )
            moving_in = _ants_prewarp_moving(fixed, moving, mat_paths)
            flow_ants = _batch_affine_flow(fixed, mat_paths)
        flow_dl = self.model(fixed, moving_in)[1]
        return flow_dl, flow_ants

    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        b = fixed_cc.size(0)
        paths_cc = _resolve_mat_paths(affine_mat_cc, b, "CC") if self.compose_ants_affine else None
        paths_mlo = _resolve_mat_paths(affine_mat_mlo, b, "MLO") if self.compose_ants_affine else None
        flow_cc, ants_cc = self._flow_for_pair(fixed_cc, moving_cc, paths_cc, "CC")
        flow_mlo, ants_mlo = self._flow_for_pair(fixed_mlo, moving_mlo, paths_mlo, "MLO")
        out = {"cc": flow_cc, "mlo": flow_mlo}
        if self.compose_ants_affine:
            out["ants_cc"] = ants_cc
            out["ants_mlo"] = ants_mlo
        return out


class VoxelMorphBackend(RegistrationBackend):
    """VoxelMorph dense flow; optional ANTs image pre-warp before DL (residual flow)."""

    def __init__(
        self,
        model: nn.Module,
        compose_ants_affine: bool = False,
        model_hw: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.model = model
        self.compose_ants_affine = compose_ants_affine
        self.model_hw = model_hw
        self.model.requires_grad_(False)
        self.model.eval()

    def _predict_flow(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        def _forward(f_fix, f_mov):
            _, flow = self.model(f_mov, f_fix, registration=True)
            return flow

        return predict_flow_at_image_size(fixed, moving, self.model_hw, _forward)

    def _flow_for_pair(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mat_paths: list[str] | None,
        view: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        moving_in = moving
        flow_ants = None
        if self.compose_ants_affine:
            if not mat_paths:
                raise ValueError(
                    f"compose_ants_affine=True but no affine mat paths for {view}"
                )
            moving_in = _ants_prewarp_moving(fixed, moving, mat_paths)
            flow_ants = _batch_affine_flow(fixed, mat_paths)
        flow_dl = self._predict_flow(fixed, moving_in)
        return flow_dl, flow_ants

    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        b = fixed_cc.size(0)
        paths_cc = _resolve_mat_paths(affine_mat_cc, b, "CC") if self.compose_ants_affine else None
        paths_mlo = _resolve_mat_paths(affine_mat_mlo, b, "MLO") if self.compose_ants_affine else None
        flow_cc, ants_cc = self._flow_for_pair(fixed_cc, moving_cc, paths_cc, "CC")
        flow_mlo, ants_mlo = self._flow_for_pair(fixed_mlo, moving_mlo, paths_mlo, "MLO")
        out = {"cc": flow_cc, "mlo": flow_mlo}
        if self.compose_ants_affine:
            out["ants_cc"] = ants_cc
            out["ants_mlo"] = ants_mlo
        return out


class TransMorphBackend(RegistrationBackend):
    """TransMorph flow; optional ANTs image pre-warp before DL (residual flow)."""

    def __init__(
        self,
        model: nn.Module,
        compose_ants_affine: bool = False,
        model_hw: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.model = model
        self.compose_ants_affine = compose_ants_affine
        self.model_hw = model_hw
        self.model.requires_grad_(False)
        self.model.eval()

    def _predict_flow(self, fixed: torch.Tensor, moving: torch.Tensor) -> torch.Tensor:
        def _forward(f_fix, f_mov):
            x_in = torch.cat([f_mov, f_fix], dim=1)
            _, flow = self.model(x_in)
            return flow

        return predict_flow_at_image_size(fixed, moving, self.model_hw, _forward)

    def _flow_for_pair(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mat_paths: list[str] | None,
        view: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        moving_in = moving
        flow_ants = None
        if self.compose_ants_affine:
            if not mat_paths:
                raise ValueError(
                    f"compose_ants_affine=True but no affine mat paths for {view}"
                )
            moving_in = _ants_prewarp_moving(fixed, moving, mat_paths)
            flow_ants = _batch_affine_flow(fixed, mat_paths)
        flow_dl = self._predict_flow(fixed, moving_in)
        return flow_dl, flow_ants

    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        b = fixed_cc.size(0)
        paths_cc = _resolve_mat_paths(affine_mat_cc, b, "CC") if self.compose_ants_affine else None
        paths_mlo = _resolve_mat_paths(affine_mat_mlo, b, "MLO") if self.compose_ants_affine else None
        flow_cc, ants_cc = self._flow_for_pair(fixed_cc, moving_cc, paths_cc, "CC")
        flow_mlo, ants_mlo = self._flow_for_pair(fixed_mlo, moving_mlo, paths_mlo, "MLO")
        out = {"cc": flow_cc, "mlo": flow_mlo}
        if self.compose_ants_affine:
            out["ants_cc"] = ants_cc
            out["ants_mlo"] = ants_mlo
        return out


class DVFFBackend(RegistrationBackend):
    """DVFF; optional ANTs image pre-warp before DL (residual flow)."""

    def __init__(self, model: nn.Module, compose_ants_affine: bool = False):
        super().__init__()
        self.model = model
        self.compose_ants_affine = compose_ants_affine
        self.use_fusion = bool(getattr(model, "use_fusion", True))
        self.model.requires_grad_(False)
        self.model.eval()

    def _flow_for_pair(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        mat_paths: list[str] | None,
        view: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        moving_in = moving
        flow_ants = None
        if self.compose_ants_affine:
            if not mat_paths:
                raise ValueError(
                    f"compose_ants_affine=True but no affine mat paths for {view}"
                )
            moving_in = _ants_prewarp_moving(fixed, moving, mat_paths)
            flow_ants = _batch_affine_flow(fixed, mat_paths)

        out = self.model.register_pair(fixed, moving_in)
        return out["flow"], flow_ants

    def get_flows(
        self,
        fixed_cc: torch.Tensor,
        moving_cc: torch.Tensor,
        fixed_mlo: torch.Tensor,
        moving_mlo: torch.Tensor,
        affine_mat_cc: Sequence[str] | None = None,
        affine_mat_mlo: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.use_fusion:
            b = fixed_cc.size(0)
            paths_cc = _resolve_mat_paths(affine_mat_cc, b, "CC") if self.compose_ants_affine else None
            paths_mlo = _resolve_mat_paths(affine_mat_mlo, b, "MLO") if self.compose_ants_affine else None
            flow_cc, ants_cc = self._flow_for_pair(fixed_cc, moving_cc, paths_cc, "CC")
            flow_mlo, ants_mlo = self._flow_for_pair(fixed_mlo, moving_mlo, paths_mlo, "MLO")
            out = {"cc": flow_cc, "mlo": flow_mlo}
            if self.compose_ants_affine:
                out["ants_cc"] = ants_cc
                out["ants_mlo"] = ants_mlo
            return out

        moving_cc_in = moving_cc
        moving_mlo_in = moving_mlo
        flow_ants_cc = None
        flow_ants_mlo = None

        if self.compose_ants_affine:
            b = fixed_cc.size(0)
            paths_cc = _resolve_mat_paths(affine_mat_cc, b, "CC")
            paths_mlo = _resolve_mat_paths(affine_mat_mlo, b, "MLO")
            if not paths_cc or not paths_mlo:
                raise ValueError(
                    "compose_ants_affine=True but affine mat paths missing in batch"
                )
            flow_ants_cc = _batch_affine_flow(fixed_cc, paths_cc)
            flow_ants_mlo = _batch_affine_flow(fixed_mlo, paths_mlo)
            moving_cc_in = warp_with_flow(moving_cc, flow_ants_cc)
            moving_mlo_in = warp_with_flow(moving_mlo, flow_ants_mlo)

        outputs = self.model(
            fixed_mlo, fixed_cc, moving_mlo_in, moving_cc_in,
        )
        out = {"cc": outputs["flow_cc"], "mlo": outputs["flow_mlo"]}
        if self.compose_ants_affine:
            out["ants_cc"] = flow_ants_cc
            out["ants_mlo"] = flow_ants_mlo
        return out
