"""
Train the U-Net that completes the breast masks of the training split.

Settings follow the supplementary material: a composite loss
0.5*BCE + 0.5*(1 - Dice), a learning rate of 1e-4, a batch size of 4, and 50
epochs, on the manually annotated training masks.

Example:
    python segmentation/train.py \
        --images_dir Dataset/train \
        --masks_dir Dataset/masks-train \
        --checkpoint segmentation/unet_mammography_segmentation.pth
"""

import argparse
import os
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataloader import MammographyDataset, get_train_transform, get_val_transform
from nn import UNet, BCEDiceLoss


def collect_image_mask_paths(masks_dir, images_dir):
    mask_paths = []
    image_paths = []

    masks_root = Path(masks_dir)
    images_root = Path(images_dir)

    for mask_path in masks_root.rglob("*.png"):
        rel_path = mask_path.relative_to(masks_root)
        
        stem = mask_path.stem
        
        image_candidates = [
            images_root / rel_path.with_suffix(".jpg"),
            images_root / rel_path.with_suffix(".png")
        ]
        
        image_found = None

        for candidate in image_candidates:
            if candidate.exists():
                image_found = candidate
                break
        
        if image_found is None:
            print(f"Warning: there are no images for {mask_path}")
            continue
        
        mask_paths.append(str(mask_path))
        image_paths.append(str(image_found))
    
    print(f"Found {len(image_paths)} pairs (image, mask).")
    return image_paths, mask_paths


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=500, device='cuda'):
    model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
        
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the breast-mask segmentation U-Net."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory of training mammograms (searched recursively)",
    )
    parser.add_argument(
        "--masks_dir",
        required=True,
        help="Directory of the manual masks, mirroring the layout of --images_dir",
    )
    parser.add_argument(
        "--checkpoint",
        default="unet_mammography_segmentation.pth",
        help="Where to write the trained weights",
    )
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main():
    args = parse_arguments()

    image_paths, mask_paths = collect_image_mask_paths(args.masks_dir, args.images_dir)

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=args.val_split, random_state=args.seed
    )
    
    train_dataset = MammographyDataset(train_images, train_masks, transform=get_train_transform())
    val_dataset = MammographyDataset(val_images, val_masks, transform=get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=1)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print(f"Using device: {device}")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=args.num_epochs, device=device
    )

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.parent != Path(""):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved weights to {checkpoint_path}")

    return model


if __name__ == "__main__":
    model = main()