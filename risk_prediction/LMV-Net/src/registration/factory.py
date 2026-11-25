"""Load registration backend for risk-prediction experiments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

from models import MammoRegNet
from registration.backends import (
    AntsAffineBackend,
    DVFFBackend,
    MammoRegNetBackend,
    RegistrationBackend,
    TransMorphBackend,
    UnregisteredBackend,
    VoxelMorphBackend,
)
from registration.checkpoint_io import ensure_local_checkpoint, torch_load_checkpoint

REGISTRATION_METHOD_ALIASES = {
    "affine": "ants_affine",
    "ants": "ants_affine",
    "none": "unreg",
    "unregistered": "unreg",
    "vm": "voxelmorph",
    "vxm": "voxelmorph",
    "tm": "transmorph",
    "mrn": "mammoregnet",
    "dvff": "dvff",
    "dvff": "dvff",
}

REGISTRATION_METHODS = (
    "unreg",
    "ants_affine",
    "voxelmorph",
    "transmorph",
    "mammoregnet",
    "dvff",
)

METHODS_WITHOUT_CHECKPOINT = frozenset({"unreg", "ants_affine"})

DEFAULT_MODEL_HW = (1152, 896)  # H, W - MammoRegNet / VM / TM training size


def normalize_registration_method(method: str) -> str:
    key = method.lower().strip()
    return REGISTRATION_METHOD_ALIASES.get(key, key)


def repo_root() -> Path:
    """
    Root of this repository (directory that contains ``Methods/``).

    When ``train_auto_resume.sh`` stages ``src/`` to ``/tmp``, ``__file__`` no
    longer lives under the real repo - use ``REPO_ROOT`` env var in that case.
    """
    for key in ("REPO_ROOT", "DVFF_MAMMOREG_REPO"):
        env_root = os.environ.get(key)
        if env_root:
            root = Path(env_root).expanduser().resolve()
            if (root / "Methods").is_dir():
                return root

    candidate = Path(__file__).resolve().parents[4]
    if (candidate / "Methods").is_dir():
        return candidate

    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry).expanduser().resolve()
        if (root / "Methods" / "MammoRegNet").is_dir():
            return root

    raise RuntimeError(
        "Could not locate repository root (expected a directory containing "
        "Methods/). Set REPO_ROOT=/path/to/this repository before launching."
    )


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_repo_on_path() -> Path:
    root = repo_root()
    _prepend_sys_path(root)
    return root


def _ensure_voxelmorph_on_path() -> Path:
    root = repo_root()
    vxm_dir = root / "Methods" / "voxelmorph"
    if not (vxm_dir / "voxelmorph").is_dir():
        raise FileNotFoundError(
            f"VoxelMorph sources not found at {vxm_dir}. "
            f"Check REPO_ROOT={root} points to the full repository checkout."
        )
    _prepend_sys_path(vxm_dir)
    os.environ.setdefault("VXM_BACKEND", "pytorch")
    return vxm_dir


def _ensure_transmorph_on_path() -> Path:
    root = repo_root()
    tm_dir = root / "Methods" / "transmorph"
    if not (tm_dir / "models" / "TransMorph.py").is_file():
        raise FileNotFoundError(
            f"TransMorph sources not found at {tm_dir}. "
            f"Check REPO_ROOT={root} points to the full repository checkout."
        )
    _prepend_sys_path(tm_dir)
    return tm_dir


def _load_state_dict(checkpoint_path: str, map_location="cpu") -> dict:
    local_path = ensure_local_checkpoint(checkpoint_path)
    checkpoint = torch_load_checkpoint(local_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_mammoregnet_checkpoint(
    checkpoint_path: str,
    map_location="cpu",
    coarse_affine: bool = True,
) -> nn.Module:
    state_dict = _load_state_dict(checkpoint_path, map_location)

    if coarse_affine:
        model = MammoRegNet()
        model.load_state_dict(state_dict, strict=True)
    else:
        _ensure_repo_on_path()
        from Methods.MammoRegNet.models.MammoRegNet import MammoRegNet as MethodsMammoRegNet

        model = MethodsMammoRegNet(coarse_affine=False)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


def load_dvff_checkpoint(
    checkpoint_path: str,
    map_location="cpu",
    coarse_affine: bool = False,
    use_fusion: bool = True,
) -> nn.Module:
    _ensure_repo_on_path()
    from Methods.DVFF.models.dvff_net import DVFFNet

    state_dict = _load_state_dict(checkpoint_path, map_location)
    fusion_levels = DVFFNet.fusion_levels_from_state_dict(state_dict)
    model = DVFFNet(
        use_fusion=use_fusion,
        coarse_affine=coarse_affine,
        fusion_levels=fusion_levels,
    )
    if not use_fusion:
        # Inference without fusion blocks: keep encoder/decoder/transformer only.
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith(("fusion_blocks.", "view_embed."))
        }
    model.load_state_dict(state_dict, strict=use_fusion)
    if use_fusion:
        model.set_fusion_enabled(True)
    model.eval()
    return model


def load_voxelmorph_checkpoint(
    checkpoint_path: str,
    map_location="cpu",
) -> tuple[nn.Module, tuple[int, int] | None]:
    _ensure_voxelmorph_on_path()
    import voxelmorph as vxm

    local_path = ensure_local_checkpoint(checkpoint_path)
    # Load on CPU first — VxmDense.load(..., cuda) mmap's from NFS and can SIGBUS.
    checkpoint = torch_load_checkpoint(local_path, map_location="cpu")
    model = vxm.torch.networks.VxmDense(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if map_location not in (None, "cpu"):
        model = model.to(map_location)
    inshape = tuple(model.config.get("inshape", DEFAULT_MODEL_HW))
    model.eval()
    return model, inshape


def _snapshot_models_namespace() -> dict[str, object]:
    return {
        key: sys.modules[key]
        for key in list(sys.modules)
        if key == "models" or key.startswith("models.")
    }


def _clear_models_namespace() -> None:
    for key in list(sys.modules):
        if key == "models" or key.startswith("models."):
            del sys.modules[key]


def _restore_models_namespace(snapshot: dict[str, object]) -> None:
    for key in list(sys.modules):
        if (key == "models" or key.startswith("models.")) and key not in snapshot:
            del sys.modules[key]
    sys.modules.update(snapshot)


def _import_transmorph_module():
    """
    Import Methods/transmorph ``models.TransMorph`` without colliding with
    LMV-Net's top-level ``models`` package (already loaded for MammoRegNet).
    """
    tm_dir = _ensure_transmorph_on_path()
    snapshot = _snapshot_models_namespace()
    _clear_models_namespace()
    try:
        import importlib

        return importlib.import_module("models.TransMorph")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to import TransMorph from {tm_dir / 'models'}. "
            "Ensure REPO_ROOT points to the full repository checkout."
        ) from exc
    finally:
        _restore_models_namespace(snapshot)


def load_transmorph_checkpoint(
    checkpoint_path: str,
    map_location="cpu",
) -> tuple[nn.Module, tuple[int, int] | None]:
    transmorph_mod = _import_transmorph_module()

    config = transmorph_mod.CONFIGS["TransMorph"]
    model = transmorph_mod.TransMorph(config)
    state_dict = _load_state_dict(checkpoint_path, map_location)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    img_size = tuple(config.img_size)
    return model, img_size


def load_registration_backend(
    method: str,
    checkpoint_path: str | None = None,
    map_location="cpu",
    coarse_affine: bool = True,
    use_fusion: bool = True,
    compose_ants_affine: bool = False,
    prewarp_in_loader: bool = False,
) -> RegistrationBackend:
    method = normalize_registration_method(method)
    if method not in REGISTRATION_METHODS:
        raise ValueError(
            f"Unknown registration_method={method!r}. "
            f"Choose from: {', '.join(REGISTRATION_METHODS)}"
        )

    # When the dataloader already applied the ANTs affine to the prior images,
    # the model must NOT warp again. It only predicts the residual (non-rigid)
    # DL flow on the already-aligned prior.
    if prewarp_in_loader:
        compose_ants_affine = False
        if method == "ants_affine":
            # Prior is fully aligned in the loader; no residual to predict.
            return UnregisteredBackend()

    if method == "unreg":
        return UnregisteredBackend()

    if method == "ants_affine":
        return AntsAffineBackend()

    if not checkpoint_path:
        raise ValueError(f"--registration_checkpoint is required for {method}")

    if method == "mammoregnet":
        model = load_mammoregnet_checkpoint(
            checkpoint_path, map_location=map_location, coarse_affine=coarse_affine
        )
        return MammoRegNetBackend(model, compose_ants_affine=compose_ants_affine)

    if method == "dvff":
        model = load_dvff_checkpoint(
            checkpoint_path,
            map_location=map_location,
            coarse_affine=coarse_affine,
            use_fusion=use_fusion,
        )
        return DVFFBackend(model, compose_ants_affine=compose_ants_affine)

    if method == "voxelmorph":
        model, model_hw = load_voxelmorph_checkpoint(
            checkpoint_path, map_location=map_location
        )
        return VoxelMorphBackend(
            model,
            compose_ants_affine=compose_ants_affine,
            model_hw=model_hw,
        )

    if method == "transmorph":
        model, model_hw = load_transmorph_checkpoint(
            checkpoint_path, map_location=map_location
        )
        return TransMorphBackend(
            model,
            compose_ants_affine=compose_ants_affine,
            model_hw=model_hw,
        )

    raise RuntimeError(f"Unhandled registration method: {method}")


def default_ants_prewarp(method: str) -> bool:
    """
    Apply cached ANTs affine on prior images before DL residual flow.

    True for methods that assume an affine-prealigned prior (voxelmorph,
    transmorph, ants_affine, dvff); False for mammoregnet
    (its own coarse affine head handles global alignment) and unreg.
    """
    method = normalize_registration_method(method)
    if method in (
        "ants_affine",
        "voxelmorph",
        "transmorph",
        "dvff",
    ):
        return True
    return False


def default_coarse_affine(method: str) -> bool:
    method = normalize_registration_method(method)
    if method == "dvff":
        return False
    if method == "mammoregnet":
        return True
    return True


def requires_registration_checkpoint(method: str) -> bool:
    return normalize_registration_method(method) not in METHODS_WITHOUT_CHECKPOINT
