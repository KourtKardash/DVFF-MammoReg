"""Resolve cached ANTs affine .mat paths for CSAW longitudinal pairs."""

from __future__ import annotations

from pathlib import Path


def exam_key_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def affine_mat_path(
    cache_root: str | Path,
    split: str,
    patient_id: str,
    laterality: str,
    view: str,
    prior_exam_key,
    current_exam_key,
) -> Path:
    """Match layout from risk_prediction/scripts/cache_affine_transforms.py."""
    prior_key = exam_key_str(prior_exam_key)
    current_key = exam_key_str(current_exam_key)
    return (
        Path(cache_root)
        / split
        / str(patient_id)
        / f"{laterality}_{view}_{prior_key}-{current_key}.mat"
    )


def affine_mats_exist(
    cache_root: str | Path,
    split: str,
    patient_id: str,
    laterality: str,
    prior_exam_key,
    current_exam_key,
) -> bool:
    """True if both CC and MLO cached .mat transforms exist."""
    return (
        affine_mat_path(
            cache_root, split, patient_id, laterality, "CC", prior_exam_key, current_exam_key
        ).is_file()
        and affine_mat_path(
            cache_root, split, patient_id, laterality, "MLO", prior_exam_key, current_exam_key
        ).is_file()
    )
