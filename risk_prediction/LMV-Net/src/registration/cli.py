"""Shared CLI arguments for registration method selection."""

from __future__ import annotations

import argparse

from registration.factory import (
    REGISTRATION_METHODS,
    default_ants_prewarp,
    default_coarse_affine,
    normalize_registration_method,
    requires_registration_checkpoint,
)

ALIGNMENT_LEVELS = ("features", "images")


def add_registration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registration_method",
        type=str,
        default="mammoregnet",
        help=(
            "Longitudinal alignment backend. Canonical: "
            f"{', '.join(REGISTRATION_METHODS)}. "
            "Aliases: affine→ants_affine, vm→voxelmorph, tm→transmorph, "
            "mrn→mammoregnet, dvff→dvff, none→unreg."
        ),
    )
    parser.add_argument(
        "--registration_checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint for voxelmorph, transmorph, mammoregnet, or "
            "dvff. Not required for unreg or ants_affine."
        ),
    )
    parser.add_argument(
        "--mammoregnet_path",
        type=str,
        default=None,
        help="Deprecated alias for --registration_checkpoint (mammoregnet).",
    )
    parser.add_argument(
        "--affine_cache_root",
        type=str,
        default=None,
        help="Root of cached ANTs .mat files from cache_affine_transforms.py.",
    )
    parser.add_argument(
        "--alignment_level",
        type=str,
        default="images",
        choices=ALIGNMENT_LEVELS,
        help=(
            "Where to apply registration flows: "
            "'images' warps prior images before encoding (default); "
            "'features' warps prior MIRAI features."
        ),
    )
    parser.add_argument(
        "--ants_prewarp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "benchmark-style ANTs affine pre-warp on prior images, then DL residual "
            "flow on (current, prior_ants). Applied at --alignment_level. "
            "Default: True for ants_affine and dvff, False otherwise."
        ),
    )
    parser.add_argument(
        "--ants_prewarp_in_loader",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the cached ANTs affine pre-warp to prior images inside the "
            "(CPU, multi-worker) dataloader instead of the model forward pass. "
            "Runs before flip (keeps the pair aligned) and warps the image "
            "directly via ANTs. Only active when ANTs pre-warp is enabled."
        ),
    )
    parser.add_argument(
        "--affine_compute_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a cached ANTs .mat is missing, estimate the affine on the fly in the "
            "dataloader (and cache it) instead of skipping the sample. Only active "
            "together with --ants_prewarp_in_loader."
        ),
    )
    parser.add_argument(
        "--coarse_affine",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use affine head at coarsest MRN decoder level (mammoregnet / dvff only). "
            "Default: True for mammoregnet, False for dvff."
        ),
    )
    parser.add_argument(
        "--dvff_use_fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "DVFF only. If True (default), register CC and MLO jointly "
            "with cross-view fusion (full study). If False, run single-view "
            "register_pair on each projection independently (no fusion); encoder/decoder "
            "weights are still loaded from the dual-view checkpoint."
        ),
    )


def resolve_registration_settings(args: argparse.Namespace) -> argparse.Namespace:
    args.registration_method = normalize_registration_method(args.registration_method)
    if args.registration_method not in REGISTRATION_METHODS:
        raise ValueError(
            f"Unknown registration_method={args.registration_method!r}. "
            f"Choose from: {', '.join(REGISTRATION_METHODS)}"
        )

    checkpoint = args.registration_checkpoint or args.mammoregnet_path
    if requires_registration_checkpoint(args.registration_method) and not checkpoint:
        raise ValueError(
            f"--registration_checkpoint is required for registration_method="
            f"{args.registration_method!r}"
        )

    if args.ants_prewarp is None:
        args.ants_prewarp = default_ants_prewarp(args.registration_method)
    if args.coarse_affine is None:
        args.coarse_affine = default_coarse_affine(args.registration_method)

    if args.ants_prewarp and not args.affine_cache_root:
        raise ValueError(
            "--affine_cache_root is required when ANTs pre-warp is enabled "
            f"(registration_method={args.registration_method}, ants_prewarp=True)"
        )

    args.alignment_level = str(args.alignment_level).lower()
    if args.alignment_level not in ALIGNMENT_LEVELS:
        raise ValueError(
            f"Unknown alignment_level={args.alignment_level!r}. "
            f"Choose from: {', '.join(ALIGNMENT_LEVELS)}"
        )

    args.registration_checkpoint = checkpoint
    return args


def dataset_registration_kwargs(args: argparse.Namespace) -> dict:
    return {
        "registration_method": args.registration_method,
        "affine_cache_root": args.affine_cache_root,
        "ants_prewarp": args.ants_prewarp,
        "prewarp_in_loader": getattr(args, "ants_prewarp_in_loader", False),
        "compute_missing_affine": getattr(args, "affine_compute_missing", False),
    }
