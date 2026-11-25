import glob
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_file_order(file_names, subdataset, metadata):
    path_to_meta = {}
    for _, row in metadata.iterrows():
        path_to_meta[os.path.normpath(str(row["File Path"]))] = row

    path1 = os.path.normpath(str(file_names[0]))
    path2 = os.path.normpath(str(file_names[1]))

    meta1 = path_to_meta[path1]
    meta2 = path_to_meta[path2]
    if subdataset in ["NLBS", "KAU-BCMD"]:
        age1 = meta1["Patient age"]
        age2 = meta2["Patient age"]

        if age1 <= age2:
            return file_names[0], file_names[1]
        else:
            return file_names[1], file_names[0]

    elif subdataset in ["INBreast", "CSAW-CC"]:
        date1 = str(meta1["Study Date"])
        date2 = str(meta2["Study Date"])

        try:
            date1_val = int(date1)
            date2_val = int(date2)

            if date1_val <= date2_val:
                return file_names[0], file_names[1]
            else:
                return file_names[1], file_names[0]
        except ValueError:
            print("Warning: Could not parse dates, using default order")
            return file_names[0], file_names[1]


class MammoDataset(Dataset):
    def __init__(self, root_dir, transforms=None, image_size=(896, 1152)):
        self.root_dir = root_dir
        root = Path(root_dir)
        self.masks_root = str(root.parent / f"{root.name}_masks")
        self.image_size = image_size
        self.transforms = transforms
        self.subfolders = []
        for subdir in ["INBreast", "KAU-BCMD", "NLBS", "CSAW-CC"]:
            subdir_path = os.path.join(root_dir, subdir)
            pair_dirs = glob.glob(os.path.join(subdir_path, "*"))
            self.subfolders.extend(pair_dirs)
        metadata_path = str(root / "metadata_train_correct.csv")
        self.metadata = pd.read_csv(metadata_path)

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
        subfolder = random.choice(self.subfolders)
        images = glob.glob(os.path.join(subfolder, "*.jpg")) + glob.glob(
            os.path.join(subfolder, "*.png")
        )
        moving_img_path, fixed_img_path = get_file_order(
            images, Path(subfolder).parent.name, self.metadata
        )

        fixed_img = self.load_image(fixed_img_path)
        moving_img = self.load_image(moving_img_path)

        img_fixed = torch.from_numpy(fixed_img).unsqueeze(0).to(torch.float32)
        img_moving = torch.from_numpy(moving_img).unsqueeze(0).to(torch.float32)

        data = {
            "img_fix": img_fixed,
            "img_mov": img_moving,
        }
        return data

    def __len__(self):
        return 1000
