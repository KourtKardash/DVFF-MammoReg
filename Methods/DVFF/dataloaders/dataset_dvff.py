import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from Methods.MammoRegNet.dataloaders.dataset_pairs import (
    get_file_order,
)


def build_dvff_groups(metadata: pd.DataFrame) -> list[dict]:
    """
    Build training/evaluation groups from metadata CSV.

    Each group corresponds to one breast (Patient ID + Laterality) that has
    at least two MLO and two CC images available on disk.
    """
    grouped = defaultdict(lambda: {"MLO": [], "CC": []})

    for _, row in metadata.iterrows():
        view = str(row["View"]).strip().upper()
        if view not in {"MLO", "CC"}:
            continue

        file_path = os.path.normpath(str(row["File Path"]))
        if not os.path.exists(file_path):
            continue

        key = (
            str(row["Patient ID"]),
            str(row["Laterality"]),
            str(row["Source_dataset"]),
        )
        grouped[key][view].append(file_path)

    groups = []
    for (patient_id, laterality, subdataset), views in grouped.items():
        if len(views["MLO"]) >= 2 and len(views["CC"]) >= 2:
            groups.append(
                {
                    "patient_id": patient_id,
                    "laterality": laterality,
                    "subdataset": subdataset,
                    "mlo_paths": sorted(set(views["MLO"])),
                    "cc_paths": sorted(set(views["CC"])),
                }
            )

    return groups


def select_image_pair(image_paths, subdataset, metadata):
    """Pick a moving/fixed pair using the same temporal logic as MammoDataset."""
    if len(image_paths) < 2:
        raise ValueError(f"Expected at least 2 images, got {len(image_paths)}")

    if len(image_paths) == 2:
        return get_file_order(image_paths, subdataset, metadata)

    pair = random.sample(image_paths, 2)
    return get_file_order(pair, subdataset, metadata)


def _metadata_sort_key(path: str, subdataset: str, path_to_meta: dict):
    row = path_to_meta[os.path.normpath(path)]
    if subdataset in ["NLBS", "KAU-BCMD"]:
        return row["Patient age"]
    date_value = str(row["Study Date"]).replace("-", "")
    try:
        return int(date_value)
    except ValueError:
        return 0


def select_canonical_pair(image_paths, subdataset, metadata):
    """Deterministic moving/fixed pair for evaluation (earliest vs latest)."""
    if len(image_paths) < 2:
        raise ValueError(f"Expected at least 2 images, got {len(image_paths)}")

    path_to_meta = {
        os.path.normpath(str(row["File Path"])): row
        for _, row in metadata.iterrows()
    }
    sorted_paths = sorted(
        image_paths,
        key=lambda path: _metadata_sort_key(path, subdataset, path_to_meta),
    )
    return get_file_order([sorted_paths[0], sorted_paths[-1]], subdataset, metadata)


class DVFFDataset(Dataset):
    """
    Dual-view dataset driven entirely by metadata CSV.

    For each sample, selects one breast (Patient ID + Laterality) and returns:
    fixed_mlo, fixed_cc, moving_mlo, moving_cc.
    """

    def __init__(
        self,
        root_dir: str,
        transforms=None,
        image_size=(896, 1152),
        metadata_path: str | None = None,
        length: int = 1000,
    ):
        self.root_dir = root_dir
        root = Path(root_dir)
        self.masks_root = str(root.parent / f"{root.name}_masks")
        self.image_size = image_size
        self.transforms = transforms
        self.length = length
        if metadata_path is None:
            metadata_path = str(root / "metadata_train_correct.csv")
        self.metadata = pd.read_csv(metadata_path)
        self.groups = build_dvff_groups(self.metadata)

        if not self.groups:
            raise RuntimeError(
                f"No dual-view groups found in metadata: {metadata_path}. "
                "Expected groups with >=2 MLO and >=2 CC images per "
                "(Patient ID, Laterality, Source_dataset)."
            )

    def _mask_path_for_image(self, image_path: str) -> str:
        rel_path = os.path.relpath(image_path, self.root_dir)
        mask_path = os.path.join(self.masks_root, rel_path)
        if os.path.exists(mask_path):
            return mask_path

        mask_png = os.path.splitext(mask_path)[0] + ".png"
        if os.path.exists(mask_png):
            return mask_png

        raise FileNotFoundError(
            f"Mask not found for image {image_path}. "
            f"Expected at {mask_path} or {mask_png}"
        )

    def load_mask(self, image_path: str) -> np.ndarray:
        mask_path = self._mask_path_for_image(image_path)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)
        return mask

    def load_image(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = self.load_mask(path)
        img = img * mask
        return img

    def __getitem__(self, index):
        group = random.choice(self.groups)
        subdataset = group["subdataset"]

        moving_mlo_path, fixed_mlo_path = select_image_pair(
            group["mlo_paths"], subdataset, self.metadata
        )
        moving_cc_path, fixed_cc_path = select_image_pair(
            group["cc_paths"], subdataset, self.metadata
        )

        fixed_mlo = self.load_image(fixed_mlo_path)
        fixed_cc = self.load_image(fixed_cc_path)
        moving_mlo = self.load_image(moving_mlo_path)
        moving_cc = self.load_image(moving_cc_path)

        sample = {
            "fixed_mlo": torch.from_numpy(fixed_mlo).unsqueeze(0).float(),
            "fixed_cc": torch.from_numpy(fixed_cc).unsqueeze(0).float(),
            "moving_mlo": torch.from_numpy(moving_mlo).unsqueeze(0).float(),
            "moving_cc": torch.from_numpy(moving_cc).unsqueeze(0).float(),
            "patient_id": group["patient_id"],
            "laterality": group["laterality"],
            "subdataset": subdataset,
            "fixed_mlo_path": fixed_mlo_path,
            "fixed_cc_path": fixed_cc_path,
            "moving_mlo_path": moving_mlo_path,
            "moving_cc_path": moving_cc_path,
        }
        return sample

    def __len__(self):
        return self.length
