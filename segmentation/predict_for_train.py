"""
Predict breast masks for the training split with the trained U-Net.

Writes one mask per input image, mirroring the directory layout of the images.
The predicted mask is post-processed by a morphological close and open, hole
filling, and keeping the largest connected component.

Example:
    python segmentation/predict_for_train.py \
        --images_dir Dataset/train \
        --masks_dir Dataset/train_masks \
        --checkpoint segmentation/unet_mammography_segmentation.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2
from scipy import ndimage

from nn import UNet


def collect_image_mask_paths(images_dir, masks_dir):
    mask_paths = []
    image_paths = []

    masks_root = Path(masks_dir)
    images_root = Path(images_dir)
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}

    for image_path in images_root.rglob("*"):
        if image_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        rel_path = image_path.relative_to(images_root)
        
        mask_path = masks_root / rel_path
        mask_path = mask_path.with_suffix(".png")
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        
        mask_paths.append(str(mask_path))
        image_paths.append(str(image_path))
    
    print(f"Found {len(image_paths)} pairs (image, mask).")
    return image_paths, mask_paths


def predict(model, image_path, device='cuda', threshold=0.5):
    model.eval()

    image = Image.open(image_path).convert('L')
    original_shape = (image.width, image.height)
    image = image.resize((928, 1168), Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, 1))
    
    with torch.no_grad():
        image_tensor = torch.from_numpy(image).to(device)
        output = model(image_tensor)
        prediction = output.cpu().numpy()[0, 0]
    
    mask_binary = (prediction > threshold).astype(np.uint8) * 255

    mask_binary = Image.fromarray(mask_binary)
    mask_binary = mask_binary.resize(original_shape, Image.Resampling.NEAREST)

    mask_binary = np.array(mask_binary)
    mask_binary = (mask_binary // 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_closed = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_closed = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    mask_filled = ndimage.binary_fill_holes(mask_closed).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled, connectivity=8)
    
    if num_labels <= 1:
        # Nothing was found: return an empty mask at the original image size, so
        # that every return path of this function agrees on the output shape.
        return Image.fromarray(np.zeros(mask_filled.shape, dtype=np.uint8))

    largest_label_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_label = largest_label_idx + 1
    
    mask_largest = (labels == largest_label).astype(np.uint8) * 255

    return Image.fromarray(mask_largest)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict breast masks for the training split."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory of mammograms to segment (searched recursively)",
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        help="Where to write the predicted masks, mirroring --images_dir",
    )
    parser.add_argument(
        "--checkpoint",
        default="segmentation/unet_mammography_segmentation.pth",
        help="Trained U-Net weights, written by segmentation/train.py",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_arguments()

    image_paths, mask_paths = collect_image_mask_paths(args.images_dir, args.masks_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    
    print(f"Using device: {device}")
    for img_path, mask_path in zip(image_paths, mask_paths):
        print(img_path)
        mask_pred = predict(model, img_path, device=device, threshold=args.threshold)
        mask_pred.save(mask_path)

if __name__ == "__main__":
    main()