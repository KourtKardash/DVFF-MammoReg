"""
Build a synthetic dataset tree for the sanity checks.

Nothing here is read by the experiment code in ``Methods/`` or
``risk_prediction/``: the tree is written into a temporary directory and the
checks run with that directory as the working directory, so the production
scripts see it through the same relative paths (``Dataset/...``, ``outputs/...``)
they normally use, without anything being written into the repository.

The images are random noise inside an elliptical "breast" region. They are not
meant to be realistic; they only have to satisfy the shape, dtype and metadata
expectations of the loaders.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Native size of the pipeline: PIL (width, height).
IMAGE_SIZE = (896, 1152)

# Filename stems are what rTRE.py uses to attribute a landmark pair to a source
# collection, so the mock names have to follow the same conventions.
TRAIN_SPEC = [
    # (source dataset, patient id, laterality, stem template, ordering key)
    ("CSAW-CC", "1001", "L", "{pid}-{idx}.png", "Study Date"),
    ("NLBS", "2002", "R", "p_{pid}_{idx}.png", "Patient age"),
]
EVAL_SPEC = [
    ("CSAW-CC", "3001", "{pid}-{idx}.png", "Study Date"),
    ("NLBS", "4002", "p_{pid}_{idx}.png", "Patient age"),
]


def _breast_image(rng: np.random.Generator) -> np.ndarray:
    """Random texture inside an ellipse, zero outside."""
    width, height = IMAGE_SIZE
    yy, xx = np.mgrid[0:height, 0:width]
    inside = ((xx / (0.75 * width)) ** 2 + ((yy - height / 2) / (0.45 * height)) ** 2) < 1.0
    texture = rng.integers(60, 240, size=(height, width), dtype=np.uint8)
    return (texture * inside).astype(np.uint8)


def _mask_from_image(image: np.ndarray) -> np.ndarray:
    """Binary breast mask, stored the way the loaders expect it (0 / 255)."""
    return ((image > 0) * 255).astype(np.uint8)


def _write_pair(rng, image_dir: Path, mask_dir: Path, stems: list[str]):
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        image = _breast_image(rng)
        Image.fromarray(image).save(image_dir / stem)
        Image.fromarray(_mask_from_image(image)).save(mask_dir / f"{Path(stem).stem}.png")
        paths.append(image_dir / stem)
    return paths


def _landmark_xml(path: Path, image_names: list[str], rng, jitter: float = 0.0):
    """Write a CVAT-style landmark file with five points per image."""
    root = ET.Element("annotations")
    width, height = IMAGE_SIZE
    for name in image_names:
        image = ET.SubElement(
            root, "image", name=name, width=str(width), height=str(height)
        )
        for point_id in range(1, 6):
            x = 200.0 + 100.0 * point_id + jitter * rng.normal()
            y = 300.0 + 120.0 * point_id + jitter * rng.normal()
            points = ET.SubElement(image, "points", points=f"{x:.2f},{y:.2f}")
            attribute = ET.SubElement(points, "attribute", name="id")
            attribute.text = str(point_id)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build(root: Path, seed: int = 0) -> Path:
    """
    Create ``<root>/Dataset/`` with a training split and an evaluation split.

    ``File Path`` entries in the metadata are stored relative to ``root``, which
    is how the real metadata is written and how the scripts look them up: they
    are run with the data root as the working directory.

    Returns the ``Dataset`` directory.
    """
    rng = np.random.default_rng(seed)
    dataset = root / "Dataset"

    # ---------------- training split (affinely pre-warped in the real data) ---
    train_root = dataset / "affine_train"
    train_masks = dataset / "affine_train_masks"
    rows = []
    for source, patient, laterality, template, order_key in TRAIN_SPEC:
        for view in ("MLO", "CC"):
            folder = f"{patient}_{laterality}_{view}"
            stems = [
                template.format(pid=patient, idx=f"{view.lower()}{idx}")
                for idx in (1, 2)
            ]
            paths = _write_pair(
                rng, train_root / source / folder, train_masks / source / folder, stems
            )
            for order, path in enumerate(paths):
                rows.append(
                    {
                        "File Path": str(path.relative_to(root)),
                        "Patient ID": patient,
                        "Laterality": laterality,
                        "View": view,
                        "Source_dataset": source,
                        "Study Date": 20200101 + order,
                        "Patient age": 50 + order,
                    }
                )
    pd.DataFrame(rows).to_csv(train_root / "metadata_train_correct.csv", index=False)

    # ---------------- evaluation split ---------------------------------------
    eval_root = dataset / "evaluation"
    eval_masks = dataset / "evaluation_masks"
    eval_rows = []
    landmark_names = []
    for source, patient, template, order_key in EVAL_SPEC:
        stems = [template.format(pid=patient, idx=idx) for idx in (1, 2)]
        paths = _write_pair(
            rng, eval_root / source / patient, eval_masks / source / patient, stems
        )
        for order, path in enumerate(paths):
            eval_rows.append(
                {
                    "File Path": str(path.relative_to(root)),
                    "Patient ID": patient,
                    "Laterality": "L",
                    "View": "MLO",
                    "Source_dataset": source,
                    "Study Date": 20200101 + order,
                    "Patient age": 50 + order,
                }
            )
        landmark_names.extend(Path(p).name for p in paths)
    pd.DataFrame(eval_rows).to_csv(
        eval_root / "metadata_evaluation_correct.csv", index=False
    )

    # Moving landmarks and two independent readers on the fixed image.
    _landmark_xml(dataset / "moving_landmarks.xml", landmark_names, rng, jitter=0.0)
    _landmark_xml(dataset / "fixed_landmarks_1.xml", landmark_names, rng, jitter=3.0)
    _landmark_xml(dataset / "fixed_landmarks_2.xml", landmark_names, rng, jitter=3.0)

    return dataset


def build_segmentation(root: Path, count: int = 6, seed: int = 1) -> tuple[Path, Path]:
    """
    Create a small image/mask corpus for the breast-segmentation U-Net.

    The masks mirror the directory layout of the images, which is what
    ``segmentation/train.py`` and ``segmentation/predict_for_train.py`` expect.
    Images are written small: the U-Net has no size constraint, and the loader
    resizes to its working resolution anyway, so there is no reason to make the
    check slow.

    Returns ``(images_dir, masks_dir)``.
    """
    rng = np.random.default_rng(seed)
    images_dir = root / "seg_images"
    masks_dir = root / "seg_masks"
    for index in range(count):
        collection = "CSAW-CC" if index % 2 == 0 else "NLBS"
        image_dir = images_dir / collection
        mask_dir = masks_dir / collection
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        height, width = 96, 80
        yy, xx = np.mgrid[0:height, 0:width]
        inside = ((xx / (0.8 * width)) ** 2 + ((yy - height / 2) / (0.45 * height)) ** 2) < 1.0
        image = (rng.integers(60, 240, size=(height, width)) * inside).astype(np.uint8)
        Image.fromarray(image).save(image_dir / f"case_{index}.png")
        # The loader thresholds the mask at >180, so it must be a bright binary.
        Image.fromarray((inside * 255).astype(np.uint8)).save(mask_dir / f"case_{index}.png")

    return images_dir, masks_dir


def build_analysis(root: Path, folders: list[str], n_patients: int = 8, seed: int = 2) -> Path:
    """
    Create an ``outputs/`` tree of per-method landmark files for the analysis code.

    Each method folder gets the three landmark files its evaluation script writes:
    the warped moving landmarks and the two readers' fixed landmarks. Every method
    is given a different residual error so that the paired tests have something to
    separate; the numbers mean nothing beyond that.

    Returns the ``outputs`` directory.
    """
    rng = np.random.default_rng(seed)
    outputs = root / "outputs"

    # Names must match the patient-id convention of the analysis script: the
    # trailing "-<index>.png" is stripped to give the patient.
    image_names = [
        f"{1000 + patient}-{index}.png"
        for patient in range(n_patients)
        for index in (1, 2)
    ]

    width, height = IMAGE_SIZE
    truth = {
        name: np.stack(
            [rng.uniform(100, width - 100, size=5), rng.uniform(100, height - 100, size=5)],
            axis=1,
        )
        for name in image_names
    }

    def write(path: Path, points_by_name):
        root_el = ET.Element("annotations")
        for name, points in points_by_name.items():
            image = ET.SubElement(
                root_el, "image", name=name, width=str(width), height=str(height)
            )
            for point_id, (x, y) in enumerate(points, start=1):
                tag = ET.SubElement(image, "points", points=f"{x:.2f},{y:.2f}")
                ET.SubElement(tag, "attribute", name="id").text = str(point_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root_el).write(path, encoding="utf-8", xml_declaration=True)

    for rank, folder in enumerate(folders):
        # A method-specific error scale, decreasing down the list.
        scale = 40.0 - 2.5 * rank
        moving = {n: truth[n] + rng.normal(0, scale, truth[n].shape) for n in image_names}
        reader1 = {n: truth[n] + rng.normal(0, 2.0, truth[n].shape) for n in image_names}
        reader2 = {n: truth[n] + rng.normal(0, 2.0, truth[n].shape) for n in image_names}
        write(outputs / folder / "moving_landmarks.xml", moving)
        write(outputs / folder / "fixed_landmarks_1.xml", reader1)
        write(outputs / folder / "fixed_landmarks_2.xml", reader2)

    return outputs
